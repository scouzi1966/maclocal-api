//
//  Phi4SiglipVL.swift
//  mlx-swift-lm
//
//  Phi-4-reasoning-vision-15B (`model_type: phi4-siglip`) compatibility model.
//  This wraps Gemma3 VLM internals but provides explicit Phi model/processor types
//  plus weight-key normalization for common conversion layouts.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public typealias Phi4SiglipConfiguration = Gemma3Configuration

public final class Phi4Siglip: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "phi4") private var model: Gemma3

    public let config: Phi4SiglipConfiguration

    public var kvHeads: [Int] { model.kvHeads }
    public var loraLayers: [Module] { model.loraLayers }

    public init(_ config: Phi4SiglipConfiguration) {
        self.config = config
        self._model.wrappedValue = Gemma3(config)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        try model.prepare(input, cache: cache, windowSize: windowSize)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        model.callAsFunction(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Normalizes common converted Phi-4 key prefixes to Gemma3's expected names.
        // Examples:
        // - model.language_model.*         -> language_model.*
        // - visual.* / vision_model.*      -> vision_tower.*
        // - mm_projector.*                 -> multi_modal_projector.*
        var remapped: [String: MLXArray] = [:]
        remapped.reserveCapacity(weights.count)

        for (key, value) in weights {
            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }
            if newKey.hasPrefix("visual.") {
                newKey = "vision_tower." + String(newKey.dropFirst("visual.".count))
            } else if newKey.hasPrefix("vision_model.") {
                newKey = "vision_tower." + String(newKey.dropFirst("vision_model.".count))
            } else if newKey.hasPrefix("mm_projector.") {
                newKey = "multi_modal_projector."
                    + String(newKey.dropFirst("mm_projector.".count))
            }
            remapped[newKey] = value
        }

        return model.sanitize(weights: remapped)
    }
}

public typealias Phi4ProcessorConfiguration = Gemma3ProcessorConfiguration
public typealias Phi4Processor = Gemma3Processor
