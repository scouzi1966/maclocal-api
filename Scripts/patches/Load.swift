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
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
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
            // MLX.quantized() without passing mode, producing non-nil biases
            // even for MXFP4 (which requires biases=nil). Use the direct init
            // with explicit biases:nil for MXFP4 quantization.
            if mode == .mxfp4, let linear = module as? Linear {
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

    eval(model)
}
