//
//  KimiK25.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/kimi_k25.py
//  Kimi K2.5 wraps DeepseekV3 with a language_model prefix (VLM-style container).
//  Vision components are stripped during sanitize.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Top-level config: wraps DeepseekV3Configuration in text_config
public struct KimiK25Configuration: Codable, Sendable {
    var modelType: String
    var textConfig: DeepseekV3Configuration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }
}

// MARK: - Language Model

/// Wraps DeepseekV3ModelInner + lm_head under the language_model prefix
class KimiK25LanguageModel: Module {
    let model: DeepseekV3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(config: DeepseekV3Configuration) {
        self.model = DeepseekV3ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }
}

// MARK: - Model

public class KimiK25Model: Module, LLMModel, KVCacheDimensionProvider {
    public var kvHeads: [Int]

    let args: KimiK25Configuration
    @ModuleInfo(key: "language_model") var languageModel: KimiK25LanguageModel

    public init(_ args: KimiK25Configuration) {
        self.args = args
        // MLA uses 1 KV head per layer (compressed latent space)
        self.kvHeads = (0 ..< args.textConfig.numHiddenLayers).map { _ in 1 }
        self._languageModel.wrappedValue = KimiK25LanguageModel(config: args.textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // 1. Strip vision components (this is a text-only path)
        var filtered = weights.filter { key, _ in
            !key.hasPrefix("vision_tower.") &&
            !key.hasPrefix("vision_model.") &&
            !key.hasPrefix("multi_modal_projector.") &&
            !key.hasPrefix("mm_projector.")
        }

        // 2. Delegate to DeepseekV3's sanitize for expert stacking and dequant.
        //    DeepseekV3's sanitize expects keys prefixed with "model.layers.N..."
        //    but our weights are "language_model.model.layers.N...".
        //    Extract language_model.* weights, strip the prefix, run sanitize, re-prefix.
        let lmPrefix = "language_model."
        var lmWeights: [String: MLXArray] = [:]
        var otherWeights: [String: MLXArray] = [:]

        for (key, value) in filtered {
            if key.hasPrefix(lmPrefix) {
                let strippedKey = String(key.dropFirst(lmPrefix.count))
                lmWeights[strippedKey] = value
            } else {
                otherWeights[key] = value
            }
        }

        // Create a temporary DeepseekV3Model to use its sanitize
        let dsModel = DeepseekV3Model(args.textConfig)
        let sanitizedLM = dsModel.sanitize(weights: lmWeights)

        // Re-prefix and merge
        filtered = [:]
        for (key, value) in sanitizedLM {
            filtered[lmPrefix + key] = value
        }
        for (key, value) in otherWeights {
            filtered[key] = value
        }

        return filtered
    }
}

// MARK: - LoRA

extension KimiK25Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
