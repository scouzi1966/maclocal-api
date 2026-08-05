// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id, let revision):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json", "*.jinja", "tiktoken.model"]
            return try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: progressHandler
            )
        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!

    // Check if the model has vision parameters — if not, skip vision_tower weights
    // to avoid loading ~10 GB of unused vision weights for VLM safetensors used as LLM.
    let modelKeys = Set(model.parameters().flattened().map { $0.0 })
    let hasVisionParams = modelKeys.contains(where: { $0.hasPrefix("vision_tower") })

    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                if !hasVisionParams && key.hasPrefix("vision_tower") {
                    continue
                }
                weights[key] = value
            }
        }
    }

    // Official DeepSeek V4 checkpoints use singular `.scale` sidecars and
    // store the packed bytes as F8_E4M3/I8 plus F8_E8M0 scales. The model's
    // sanitize pass converts those keys and byte views to MLX's ordinary
    // `.weight`/`.scales` representation. Remember the source layout so the
    // loader can instantiate the matching quantized modules without requiring
    // a generated, thousands-entry `quantization` config dictionary.
    let hasOfficialBlockScaledWeights = weights.keys.contains { key in
        guard key.hasSuffix(".scale") else { return false }
        let base = String(key.dropLast(".scale".count))
        return weights["\(base).weight"] != nil
    }

    // per-model cleanup
    weights = model.sanitize(weights: weights)
    let usesSymmetricQ8 =
        (model as? DeepseekV4SymmetricQ8Model)?.usesDeepseekV4SymmetricQ8 == true

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil || hasOfficialBlockScaledWeights {
        quantize(model: model, filter: { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else if let quantization {
                    return quantization.asTuple
                } else if hasOfficialBlockScaledWeights,
                    let weight = weights["\(path).weight"],
                    let scales = weights["\(path).scales"],
                    let inferred = inferOfficialBlockQuantization(
                        weightShape: weight.shape,
                        scaleShape: scales.shape)
                {
                    return inferred.asTuple
                } else {
                    return nil
                }
            } else {
                return nil
            }
        }, apply: { module, groupSize, bits, mode in
            // Workaround for mlx-swift bug: QuantizedLinear.init calls
            // MLX.quantized() without passing mode, producing non-nil biases
            // for MXFP modes (which require biases=nil). Use the direct init
            // for both MXFP4 and MXFP8 checkpoint layers.
            if (mode == .mxfp4 || mode == .mxfp8 ||
                (usesSymmetricQ8 && mode == .affine && bits == 8 && groupSize == 32)),
                let linear = module as? Linear
            {
                let (qw, scales, biases) = MLX.quantized(
                    linear.weight, groupSize: groupSize, bits: bits, mode: mode)
                return DeepseekV4QuantizedLinear(
                    weight: qw, bias: linear.bias, scales: scales,
                    biases: usesSymmetricQ8 && mode == .affine ? nil : biases,
                    groupSize: groupSize, bits: bits, mode: mode)
            }
            return quantizeSingle(layer: module, groupSize: groupSize, bits: bits, mode: mode)
        })
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    // Use .noUnusedKeys only (skip .shapeMismatch) to match Python's strict=False.
    // Custom modules like GLM5's MultiLinear have manually quantized weights with
    // packed shapes that differ from the model's logical init shapes.
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    eval(model)
}

/// Infer MLX's floating-point quantization mode from an official packed weight
/// and E8M0 scale shape. MXFP4 stores eight logical values per UInt32 and one
/// scale per 32 values; MXFP8 stores four values per UInt32 with the same scale
/// granularity.
public func inferOfficialBlockQuantization(
    weightShape: [Int],
    scaleShape: [Int]
) -> BaseConfiguration.Quantization? {
    guard let packedColumns = weightShape.last,
        let scaleColumns = scaleShape.last,
        scaleColumns > 0,
        packedColumns % scaleColumns == 0
    else { return nil }

    switch packedColumns / scaleColumns {
    case 4:
        return .init(groupSize: 32, bits: 4, mode: .mxfp4)
    case 8:
        return .init(groupSize: 32, bits: 8, mode: .mxfp8)
    default:
        return nil
    }
}
