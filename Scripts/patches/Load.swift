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

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model, filter: { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }, apply: { module, groupSize, bits, mode in
            // Workaround for mlx-swift bug: QuantizedLinear.init calls
            // MLX.quantized() without passing mode, so the default `.affine`
            // quantizer rejects group sizes MXFP4/NVFP4 actually need (16 for
            // NVFP4, non-standard for MXFP4) and also produces non-nil biases
            // for fp4 modes (which require biases=nil). Use the direct init
            // with explicit biases:nil for these fp4 modes.
            if (mode == .mxfp4 || mode == .nvfp4), let linear = module as? Linear {
                let (qw, scales, _) = MLX.quantized(
                    linear.weight, groupSize: groupSize, bits: bits, mode: mode)
                return QuantizedLinear(
                    weight: qw, bias: linear.bias, scales: scales, biases: nil,
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

    // Materialize weights in chunks instead of one `MLX.eval(model)` call.
    // MoE models (Qwen3.5-35B-A3B with ~1700 arrays, Gemma 4 E4B similar)
    // hit a Metal command-buffer resource limit when all weight arrays are
    // submitted in one encoding. The failure surfaces asynchronously as
    //   [METAL] Command buffer execution failed: Insufficient Memory
    // (kIOGPUCommandBufferCallbackErrorOutOfMemory) and leaves Metal in
    // a degraded state so the first real inference SIGABRTs a few seconds
    // later. Chunking to 512 arrays per eval keeps each command buffer
    // within Metal's per-encoding resource ceiling.
    let allArrays = model.parameters().flattened().map { $0.1 }
    let chunkSize = 512
    var idx = 0
    while idx < allArrays.count {
        let end = min(idx + chunkSize, allArrays.count)
        MLX.eval(Array(allArrays[idx..<end]))
        Stream.gpu.synchronize()
        MLX.GPU.clearCache()
        idx = end
    }
    // Final drain: force any lingering async Metal work from the quantize
    // apply-closures (which create lazy ops on initial-random weights that
    // get orphaned by model.update) to surface here instead of firing
    // asynchronously from the caller after loadWeights returns.
    Stream.gpu.synchronize()
    MLX.GPU.clearCache()
}
