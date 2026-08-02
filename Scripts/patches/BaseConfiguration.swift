// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// Base ``LanguageModel`` configuration -- provides `modelType`
/// and `quantization` (used in loading the model).
///
/// This is used by ``ModelFactory/load(from:configuration:progressHandler:)``
/// to determine the type of model to load.
public struct BaseConfiguration: Codable, Sendable {
    public let modelType: String

    public struct Quantization: Codable, Sendable, Equatable {
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        public init(groupSize: Int, bits: Int, mode: QuantizationMode) {
            self.groupSize = groupSize
            self.bits = bits
            self._mode = mode
        }

        public let groupSize: Int
        public let bits: Int
        private var _mode: QuantizationMode? = nil
        public var mode: QuantizationMode { _mode ?? .affine }

        public var asTuple: (Int, Int, QuantizationMode) { (groupSize, bits, mode) }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.groupSize = try container.decode(Int.self, forKey: .groupSize)
            self.bits = try container.decode(Int.self, forKey: .bits)
            if let rawMode = try container.decodeIfPresent(String.self, forKey: ._mode) {
                let normalized = rawMode.lowercased()
                if let mode = QuantizationMode(rawValue: normalized) {
                    self._mode = mode
                } else if normalized == "affine+mxtq" || normalized == "affine_mxtq" {
                    // JANG VL bundles use this composite marker to say
                    // non-routed affine weights plus MXTQ routed-expert
                    // sidecars. BaseConfiguration only needs the ordinary
                    // affine fallback; JangLoader consumes the MXTQ sidecar
                    // metadata separately.
                    self._mode = .affine
                } else {
                    throw DecodingError.dataCorruptedError(
                        forKey: ._mode,
                        in: container,
                        debugDescription:
                            "Cannot initialize QuantizationMode from invalid String value \(rawMode)")
                }
            } else {
                self._mode = nil
            }
        }
    }

    /// handling instructions for ``PerLayerQuantization``
    public enum QuantizationOption: Sendable {
        case skip
        case quantize(Quantization)
    }

    /// Per-layer ``Quantization`` values with optional default.
    public struct PerLayerQuantization: Sendable {
        public var quantization: Quantization? = nil
        public var perLayerQuantization: [String: QuantizationOption]

        public init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        /// Return an explicit layer override without falling back to the
        /// top-level quantization default.
        ///
        /// Checkpoint metadata conventionally prefixes text-model paths with
        /// `model.`, while Swift module trees expose the same paths without
        /// that wrapper. Keep the stored metadata unchanged for round trips,
        /// but normalize both spellings at the lookup boundary.
        func explicitQuantizationOption(layer: String) -> QuantizationOption? {
            for candidate in Self.layerPathCandidates(layer) {
                if let option = perLayerQuantization[candidate] {
                    return option
                }
            }
            return nil
        }

        /// The quantization to apply for the given layer name or nil for no quantization.
        public func quantization(layer: String) -> Quantization? {
            if let perLayer = explicitQuantizationOption(layer: layer) {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }

        private static func layerPathCandidates(_ layer: String) -> [String] {
            var seen = Set<String>()
            var candidates = [String]()

            func add(_ value: String) {
                guard seen.insert(value).inserted else { return }
                candidates.append(value)
            }

            func addWithAttentionAlias(_ value: String) {
                add(value)

                // DeepseekV4Model.sanitize structurally renames the dense
                // shared expert's w1/w2/w3 checkpoint leaves to the runtime's
                // gate/down/up projections. Preserve exact quantization
                // metadata across that rename. The exact path is still tried
                // first, so a native gate/down/up declaration always wins.
                let dsv4SharedExpertAliases = [
                    (".mlp.shared_experts.gate_proj", ".ffn.shared_experts.w1"),
                    (".mlp.shared_experts.down_proj", ".ffn.shared_experts.w2"),
                    (".mlp.shared_experts.up_proj", ".ffn.shared_experts.w3"),
                ]
                for (runtimeName, checkpointName) in dsv4SharedExpertAliases {
                    if value.contains(runtimeName) {
                        add(value.replacingOccurrences(
                            of: runtimeName, with: checkpointName))
                    } else if value.contains(checkpointName) {
                        add(value.replacingOccurrences(
                            of: checkpointName, with: runtimeName))
                    }
                }

                if value.contains(".attn.") {
                    add(value.replacingOccurrences(of: ".attn.", with: ".self_attn."))
                } else if value.hasSuffix(".attn") {
                    add(String(value.dropLast(".attn".count)) + ".self_attn")
                } else if value.contains(".self_attn.") {
                    // DSV4 checkpoints name attention leaves `layers.N.attn.*`,
                    // while `DeepseekV4Model.sanitize` exposes the same modules as
                    // `model.layers.N.self_attn.*`. Quantization is resolved after
                    // sanitization, so the alias must work in both directions or
                    // exact 4-bit/group-64 declarations fall through to ambiguous
                    // shape inference (512 packed columns can also look like
                    // 8-bit/group-32) and crash the first 4096-wide projection.
                    add(value.replacingOccurrences(
                        of: ".self_attn.", with: ".attn."))
                } else if value.hasSuffix(".self_attn") {
                    add(String(value.dropLast(".self_attn".count)) + ".attn")
                }
            }

            addWithAttentionAlias(layer)
            if layer.hasPrefix("language_model.model.") {
                let modelPath = String(layer.dropFirst("language_model.".count))
                addWithAttentionAlias(modelPath)
                addWithAttentionAlias(String(modelPath.dropFirst("model.".count)))
            } else if layer.hasPrefix("language_model.") {
                let innerPath = String(layer.dropFirst("language_model.".count))
                addWithAttentionAlias(innerPath)
                if innerPath.hasPrefix("model.") {
                    addWithAttentionAlias(String(innerPath.dropFirst("model.".count)))
                } else {
                    addWithAttentionAlias("model.\(innerPath)")
                }
            } else if layer.hasPrefix("model.") {
                let barePath = String(layer.dropFirst("model.".count))
                addWithAttentionAlias(barePath)
                addWithAttentionAlias("language_model.\(layer)")
                addWithAttentionAlias("language_model.\(barePath)")
            } else {
                addWithAttentionAlias("model.\(layer)")
                addWithAttentionAlias("language_model.\(layer)")
                addWithAttentionAlias("language_model.model.\(layer)")
            }
            return candidates
        }
    }

    /// Special codable to support a mixed key: Int / key: Quantization
    /// structure for hereogenous quantization, e.g.
    ///
    /// ```
    /// "quantization": {
    ///     "group_size": 64,
    ///     "bits": 4,
    ///     "model.embed_tokens": {
    ///         "group_size": 32,
    ///         "bits": 4
    ///     },
    ///     "model.layers.0.self_attn.q_norm": false,
    /// ```
    ///
    /// This mixed type structure requires manual decoding.
    public struct QuantizationContainer: Codable, Sendable {
        public var quantization: Quantization
        public var perLayerQuantization: PerLayerQuantization

        // based on Dictionary's coding key
        internal struct _DictionaryCodingKey: CodingKey {
            internal let stringValue: String
            internal let intValue: Int?

            internal init(stringValue: String) {
                self.stringValue = stringValue
                self.intValue = Int(stringValue)
            }

            internal init(intValue: Int) {
                self.stringValue = "\(intValue)"
                self.intValue = intValue
            }
        }

        public init(from decoder: any Decoder) throws {
            // handle the embedded Quantization
            self.quantization = try Quantization(from: decoder)

            // and the interleaved per-layer values
            var perLayerQuantization = [String: QuantizationOption]()
            let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
            for key in container.allKeys {
                switch key.stringValue {
                case Quantization.CodingKeys.groupSize.rawValue: continue
                case Quantization.CodingKeys.bits.rawValue: continue
                case Quantization.CodingKeys._mode.rawValue: continue

                // additional keys that are not layer instructions, see
                // mlx-community/bitnet-b1.58-2B-4T-4bit
                case "quant_method", "linear_class", "quantization_mode": continue

                // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
                // DSV4 JANGTQ bundles stamp the routed-expert MXTQ bit
                // width as scalar siblings of `bits` / `group_size` inside
                // the `quantization` dict (`mxtq_bits: 2` /
                // `routed_expert_bits: 2`). Without these in the skip
                // list the per-layer-override decode below tries to
                // parse `2` as a `Quantization` sub-dict and throws
                // `configurationDecodingError`, blocking model load.
                case "mxtq_bits", "routed_expert_bits", "routed_expert_bit_plan", "mxtq_seed":
                    continue

                // 2026-05-06 (ZAYA prep):
                // New hybrid bundles may stamp role-level quantization
                // policy beside the MLX affine defaults. These values
                // describe converter/runtime policy, not per-layer
                // quantization overrides, so keep them out of the
                // override decoder. Example: ZAYA uses
                // `expert_layout: "split_switch_mlp"` plus role bit
                // floors for embed/router paths.
                case "expert_layout", "embed_bits", "router_bits",
                    "attention_bits", "lm_head_bits", "cca_conv_bits",
                    "norms_residual_bits":
                    continue

                default:
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else if Self.isScalarMetadata(container, key: key) {
                        // MXFP/JANG metadata can live beside MLX's affine
                        // quantization defaults. Per-layer overrides are
                        // dictionaries; scalar/list siblings describe
                        // converter/runtime policy and must not be decoded as
                        // layer quantization.
                        continue
                    } else {
                        perLayerQuantization[key.stringValue] = .quantize(
                            try Self.decodeLayerQuantization(
                                from: container,
                                key: key,
                                defaultGroupSize: quantization.groupSize))
                    }
                }
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: perLayerQuantization)
        }

        private static func decodeLayerQuantization(
            from container: KeyedDecodingContainer<_DictionaryCodingKey>,
            key: _DictionaryCodingKey,
            defaultGroupSize: Int
        ) throws -> Quantization {
            do {
                return try container.decode(Quantization.self, forKey: key)
            } catch DecodingError.keyNotFound(let missing, _)
                where missing.stringValue == Quantization.CodingKeys.groupSize.rawValue
            {
                let nested = try container.nestedContainer(
                    keyedBy: Quantization.CodingKeys.self,
                    forKey: key)
                let bits = try nested.decode(Int.self, forKey: .bits)
                let mode = try nested.decodeIfPresent(String.self, forKey: ._mode)
                    .flatMap { QuantizationMode(rawValue: $0.lowercased()) }
                    ?? quantizationModeDefaultForLayerOverride()
                return Quantization(groupSize: defaultGroupSize, bits: bits, mode: mode)
            }
        }

        private static func quantizationModeDefaultForLayerOverride() -> QuantizationMode {
            .affine
        }

        private static func isScalarMetadata(
            _ container: KeyedDecodingContainer<_DictionaryCodingKey>,
            key: _DictionaryCodingKey
        ) -> Bool {
            (try? container.decode(String.self, forKey: key)) != nil
                || (try? container.decode(Int.self, forKey: key)) != nil
                || (try? container.decode(Double.self, forKey: key)) != nil
                || (try? container.decode([String].self, forKey: key)) != nil
                || (try? container.decode([Int].self, forKey: key)) != nil
        }

        public func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                switch value {
                case .skip:
                    try container.encode(false, forKey: .init(stringValue: key))
                case .quantize(let q):
                    try container.encode(q, forKey: .init(stringValue: key))
                }
            }
        }
    }

    public var quantizationContainer: QuantizationContainer?

    /// EOS token IDs from config.json. Can be a single Int or an array of Ints.
    public var eosTokenIds: IntOrIntArray?

    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    public var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    public var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantizationContainer = "quantization"
        case eosTokenIds = "eos_token_id"
    }
}
