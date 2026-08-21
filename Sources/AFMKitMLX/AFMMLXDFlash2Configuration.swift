import Foundation

public enum AFMMLXDFlash2Requirement: String, Codable, CaseIterable, Sendable {
    case preferred
    case required
}

public struct AFMMLXDFlash2Configuration: Equatable, Sendable {
    public static let architecture = "DFlash2DraftModel"

    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let vocabularySize: Int
    public let targetLayers: Int
    public let targetLayerIDs: [Int]
    public let checkpointBlockSize: Int
    public let maskTokenID: Int
    public let convolutionKernelSize: Int
    public let convolutionGroupSize: Int
    public let selectorRank: Int
    public let selectorTopK: Int
    public let maxPositionEmbeddings: Int
    public let slidingWindow: Int?
    public let layerTypes: [String]
    public let ropeTheta: Double
    public let bosTokenIDs: Set<Int>
    public let eosTokenIDs: Set<Int>
    public let padTokenIDs: Set<Int>

    public init(metadata: [String: Any]) throws {
        let architectures = metadata["architectures"] as? [String] ?? []
        guard architectures.contains(Self.architecture) else {
            throw AFMMLXDFlash2ConfigurationError.unsupportedArchitecture(architectures)
        }
        guard (metadata["is_causal"] as? Bool) == false else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue("is_causal must be false")
        }
        guard let dflash = metadata["dflash_config"] as? [String: Any] else {
            throw AFMMLXDFlash2ConfigurationError.missingValue("dflash_config")
        }

        hiddenSize = try Self.positiveInt(metadata, "hidden_size")
        intermediateSize = try Self.positiveInt(metadata, "intermediate_size")
        hiddenLayers = try Self.positiveInt(metadata, "num_hidden_layers")
        attentionHeads = try Self.positiveInt(metadata, "num_attention_heads")
        keyValueHeads = try Self.positiveInt(metadata, "num_key_value_heads")
        headDimension = try Self.positiveInt(metadata, "head_dim")
        vocabularySize = try Self.positiveInt(metadata, "vocab_size")
        targetLayers = try Self.positiveInt(metadata, "num_target_layers")
        targetLayerIDs = try Self.intArray(dflash, "target_layer_ids")
        checkpointBlockSize = try Self.positiveInt(dflash, "block_size")
        maskTokenID = try Self.nonnegativeInt(dflash, "mask_token_id")
        convolutionKernelSize = try Self.positiveInt(dflash, "conv_kernel_size")
        convolutionGroupSize = try Self.positiveInt(dflash, "conv_group_size")
        selectorRank = try Self.positiveInt(dflash, "selector_rank")
        selectorTopK = try Self.positiveInt(dflash, "selector_top_k")
        maxPositionEmbeddings = try Self.positiveInt(metadata, "max_position_embeddings")
        slidingWindow = try Self.optionalPositiveInt(metadata, "sliding_window")
        layerTypes = try Self.stringArray(metadata, "layer_types")
        let rope = metadata["rope_parameters"] as? [String: Any]
            ?? metadata["rope_scaling"] as? [String: Any]
            ?? [:]
        ropeTheta = try Self.positiveDouble(
            rope["rope_theta"] == nil ? metadata : rope,
            "rope_theta")
        bosTokenIDs = Self.tokenIDs(in: metadata, key: "bos_token_id")
        eosTokenIDs = Self.tokenIDs(in: metadata, key: "eos_token_id")
        padTokenIDs = Self.tokenIDs(in: metadata, key: "pad_token_id")

        guard convolutionKernelSize == 2 else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "conv_kernel_size must be 2 for the supported DFlash2 runtime")
        }
        guard hiddenSize.isMultiple(of: convolutionGroupSize) else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "hidden_size must be divisible by conv_group_size")
        }
        guard targetLayerIDs.count == hiddenLayers,
              targetLayerIDs == targetLayerIDs.sorted(),
              Set(targetLayerIDs).count == targetLayerIDs.count,
              targetLayerIDs.allSatisfy({ $0 >= 0 && $0 < targetLayers }) else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "target_layer_ids must contain one unique, ordered target layer per draft layer")
        }
        guard maskTokenID < vocabularySize else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "mask_token_id must be inside the draft vocabulary")
        }
        guard selectorTopK <= vocabularySize else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "selector_top_k must not exceed vocab_size")
        }
        guard attentionHeads.isMultiple(of: keyValueHeads) else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "num_attention_heads must be divisible by num_key_value_heads")
        }
        guard layerTypes.count == hiddenLayers,
              layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" }) else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "layer_types must contain one supported attention type per draft layer")
        }
        if layerTypes.contains("sliding_attention"), slidingWindow == nil {
            throw AFMMLXDFlash2ConfigurationError.missingValue("sliding_window")
        }
    }

    public init(directory: URL) throws {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        guard let metadata = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue("config.json must contain an object")
        }
        try self.init(metadata: metadata)
    }

    public func validateTarget(metadata: [String: Any]) throws {
        let text = metadata["text_config"] as? [String: Any] ?? metadata
        let targetHidden = try Self.positiveInt(text, "hidden_size")
        let targetLayerCount = try Self.positiveInt(text, "num_hidden_layers")
        let targetVocabulary = try Self.positiveInt(text, "vocab_size")
        let targetMaxPositions = try Self.positiveInt(text, "max_position_embeddings")

        guard targetHidden == hiddenSize else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "hidden_size (targetHidden) does not match drafter (hiddenSize)")
        }
        guard targetLayerCount == targetLayers else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "num_hidden_layers (targetLayerCount) does not match drafter (targetLayers)")
        }
        guard targetVocabulary == vocabularySize else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "vocab_size (targetVocabulary) does not match drafter (vocabularySize)")
        }
        guard targetMaxPositions == maxPositionEmbeddings else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "max_position_embeddings \(targetMaxPositions) does not match drafter \(maxPositionEmbeddings)")
        }

        let targetRope = text["rope_parameters"] as? [String: Any]
            ?? text["rope_scaling"] as? [String: Any]
            ?? [:]
        let targetRopeTheta = try Self.positiveDouble(
            targetRope["rope_theta"] == nil ? text : targetRope,
            "rope_theta")
        guard abs(targetRopeTheta - ropeTheta) <= max(1, ropeTheta) * 1e-9 else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "rope_theta \(targetRopeTheta) does not match drafter \(ropeTheta)")
        }

        try Self.validateTokenIDs(
            name: "bos_token_id", draft: bosTokenIDs,
            target: Self.targetTokenIDs(metadata, key: "bos_token_id"))
        try Self.validateTokenIDs(
            name: "eos_token_id", draft: eosTokenIDs,
            target: Self.targetTokenIDs(metadata, key: "eos_token_id"))
        try Self.validateTokenIDs(
            name: "pad_token_id", draft: padTokenIDs,
            target: Self.targetTokenIDs(metadata, key: "pad_token_id"))

        if let slidingWindow,
           let targetSlidingWindow = try Self.optionalPositiveInt(text, "sliding_window"),
           targetSlidingWindow != slidingWindow {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "sliding_window \(targetSlidingWindow) does not match drafter \(slidingWindow)")
        }

        let modelType = Self.canonical(text["model_type"] as? String)
        let topLevelType = Self.canonical(metadata["model_type"] as? String)
        let supported = modelType == "qwen3_5_text"
            || modelType == "muse_glimmer_text"
            || topLevelType == "muse_glimmer"
        guard supported else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "target model_type is not a supported Qwen 3.8 or Muse Glimmer verifier")
        }
    }

    public func effectiveBlockSize(requested: Int?) throws -> Int {
        guard let requested else { return checkpointBlockSize }
        guard requested >= 2 else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "runtime block size must be at least 2")
        }
        return min(requested, checkpointBlockSize)
    }

    /// Validate the released DFlash 2 tensor contract from safetensor headers
    /// before MLX maps or allocates the multi-gigabyte payloads.
    public func validateWeights(in directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw AFMMLXDFlash2ConfigurationError.missingValue("*.safetensors")
        }

        var actual: [String: [Int]] = [:]
        for file in files {
            for (name, shape) in try Self.safetensorShapes(at: file) {
                guard actual.updateValue(shape, forKey: name) == nil else {
                    throw AFMMLXDFlash2ConfigurationError.invalidValue(
                        "duplicate tensor \(name)")
                }
            }
        }

        let expected = expectedTensorShapes()
        for (name, shape) in expected {
            let aliases = name.hasSuffix("_codebook.weight")
                ? [name, String(name.dropLast(".weight".count))]
                : [name]
            guard let key = aliases.first(where: { actual[$0] != nil }),
                  let found = actual[key] else {
                throw AFMMLXDFlash2ConfigurationError.missingValue("tensor \(name)")
            }
            guard found == shape else {
                throw AFMMLXDFlash2ConfigurationError.invalidValue(
                    "tensor \(key) has shape \(found); expected \(shape)")
            }
        }

        let normalizedActual = Set(actual.keys.map { key in
            key.hasSuffix("_codebook") ? key + ".weight" : key
        })
        let unexpected = normalizedActual.subtracting(expected.keys).sorted()
        guard unexpected.isEmpty else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "unexpected tensors: \(unexpected.joined(separator: ", "))")
        }
    }

    func expectedTensorShapes() -> [String: [Int]] {
        var result: [String: [Int]] = [
            "candidate_selector.hidden_projection.weight": [selectorRank, hiddenSize],
            "candidate_selector.predecessor_codebook.weight": [vocabularySize, selectorRank],
            "candidate_selector.successor_codebook.weight": [vocabularySize, selectorRank],
            "fc.weight": [hiddenSize, hiddenSize * targetLayerIDs.count],
            "hidden_norm.weight": [hiddenSize],
            "norm.weight": [hiddenSize],
        ]
        let dynamicKernelOutputs = 2 * convolutionKernelSize * (hiddenSize / convolutionGroupSize)
        for layer in 0 ..< hiddenLayers {
            let prefix = "layers.\(layer)"
            result["\(prefix).input_layernorm.weight"] = [hiddenSize]
            result["\(prefix).post_attention_layernorm.weight"] = [hiddenSize]
            result["\(prefix).self_attn.q_proj.weight"] = [attentionHeads * headDimension, hiddenSize]
            result["\(prefix).self_attn.k_proj.weight"] = [keyValueHeads * headDimension, hiddenSize]
            result["\(prefix).self_attn.v_proj.weight"] = [keyValueHeads * headDimension, hiddenSize]
            result["\(prefix).self_attn.o_proj.weight"] = [hiddenSize, attentionHeads * headDimension]
            result["\(prefix).self_attn.q_norm.weight"] = [headDimension]
            result["\(prefix).self_attn.k_norm.weight"] = [headDimension]
            result["\(prefix).mlp.gate_proj.weight"] = [intermediateSize, hiddenSize]
            result["\(prefix).mlp.up_proj.weight"] = [intermediateSize, hiddenSize]
            result["\(prefix).mlp.down_proj.weight"] = [hiddenSize, intermediateSize]
            for name in ["attention_conv", "mlp_conv"] {
                result["\(prefix).\(name).base_kernel"] = [2, convolutionKernelSize, hiddenSize]
                result["\(prefix).\(name).kernel_projection.weight"] = [dynamicKernelOutputs, hiddenSize]
            }
        }
        return result
    }

    private static func canonical(_ value: String?) -> String {
        value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
    }

    private static func positiveInt(_ object: [String: Any], _ key: String) throws -> Int {
        let value = try nonnegativeInt(object, key)
        guard value > 0 else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue("\(key) must be positive")
        }
        return value
    }

    private static func optionalPositiveInt(
        _ object: [String: Any], _ key: String
    ) throws -> Int? {
        guard object[key] != nil, !(object[key] is NSNull) else { return nil }
        return try positiveInt(object, key)
    }

    private static func positiveDouble(_ object: [String: Any], _ key: String) throws -> Double {
        guard let value = (object[key] as? NSNumber)?.doubleValue else {
            throw AFMMLXDFlash2ConfigurationError.missingValue(key)
        }
        guard value > 0, value.isFinite else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue("\(key) must be positive and finite")
        }
        return value
    }

    private static func nonnegativeInt(_ object: [String: Any], _ key: String) throws -> Int {
        guard let value = (object[key] as? NSNumber)?.intValue else {
            throw AFMMLXDFlash2ConfigurationError.missingValue(key)
        }
        guard value >= 0 else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue("\(key) must be nonnegative")
        }
        return value
    }

    private static func intArray(_ object: [String: Any], _ key: String) throws -> [Int] {
        guard let values = object[key] as? [Any] else {
            throw AFMMLXDFlash2ConfigurationError.missingValue(key)
        }
        let result = values.compactMap { ($0 as? NSNumber)?.intValue }
        guard result.count == values.count else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue("\(key) must contain integers")
        }
        return result
    }

    private static func stringArray(_ object: [String: Any], _ key: String) throws -> [String] {
        guard let values = object[key] as? [String] else {
            throw AFMMLXDFlash2ConfigurationError.missingValue(key)
        }
        return values
    }

    private static func tokenIDs(in object: [String: Any], key: String) -> Set<Int> {
        if let value = object[key] as? NSNumber { return [value.intValue] }
        if let values = object[key] as? [NSNumber] { return Set(values.map(\.intValue)) }
        return []
    }

    private static func targetTokenIDs(_ metadata: [String: Any], key: String) -> Set<Int> {
        var result = tokenIDs(in: metadata, key: key)
        if let text = metadata["text_config"] as? [String: Any] {
            result.formUnion(tokenIDs(in: text, key: key))
        }
        if let generation = metadata["generation_config"] as? [String: Any] {
            result.formUnion(tokenIDs(in: generation, key: key))
        }
        return result
    }

    private static func validateTokenIDs(
        name: String, draft: Set<Int>, target: Set<Int>
    ) throws {
        guard !draft.isEmpty, !target.isEmpty else { return }
        guard draft.isSubset(of: target) else {
            throw AFMMLXDFlash2ConfigurationError.incompatibleTarget(
                "\(name) \(draft.sorted()) does not match target \(target.sorted())")
        }
    }

    private static func safetensorShapes(at url: URL) throws -> [String: [Int]] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "truncated safetensor header in \(url.lastPathComponent)")
        }
        let headerSize = prefix.enumerated().reduce(UInt64(0)) { value, item in
            value | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard headerSize > 0, headerSize <= 64 * 1024 * 1024,
              let header = try handle.read(upToCount: Int(headerSize)),
              header.count == Int(headerSize),
              let object = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw AFMMLXDFlash2ConfigurationError.invalidValue(
                "invalid safetensor header in \(url.lastPathComponent)")
        }
        var result: [String: [Int]] = [:]
        for (name, entry) in object where name != "__metadata__" {
            guard let metadata = entry as? [String: Any],
                  let rawShape = metadata["shape"] as? [Any] else {
                throw AFMMLXDFlash2ConfigurationError.invalidValue(
                    "tensor \(name) has no shape metadata")
            }
            let shape = rawShape.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == rawShape.count, shape.allSatisfy({ $0 >= 0 }) else {
                throw AFMMLXDFlash2ConfigurationError.invalidValue(
                    "tensor \(name) has invalid shape metadata")
            }
            result[name] = shape
        }
        return result
    }
}

public enum AFMMLXDFlash2ConfigurationError: LocalizedError, Equatable, Sendable {
    case missingValue(String)
    case invalidValue(String)
    case unsupportedArchitecture([String])
    case incompatibleTarget(String)

    public var errorDescription: String? {
        switch self {
        case .missingValue(let key):
            return "DFlash2 config is missing \(key)"
        case .invalidValue(let message):
            return "Invalid DFlash2 config: \(message)"
        case .unsupportedArchitecture(let architectures):
            return "Expected architectures to contain \(AFMMLXDFlash2Configuration.architecture); found \(architectures)"
        case .incompatibleTarget(let message):
            return "DFlash2 drafter is incompatible with the target: \(message)"
        }
    }
}

public struct AFMMLXSpeculativeTelemetry: Equatable, Sendable {
    public let strategy: String
    public let draftedTokens: Int
    public let acceptedDraftTokens: Int
    public let emittedTokens: Int
    public let verificationCycles: Int
    public let draftTime: TimeInterval
    public let verificationTime: TimeInterval
    public let rollbackTime: TimeInterval

    public init(
        strategy: String,
        draftedTokens: Int,
        acceptedDraftTokens: Int,
        emittedTokens: Int,
        verificationCycles: Int,
        draftTime: TimeInterval,
        verificationTime: TimeInterval,
        rollbackTime: TimeInterval
    ) {
        self.strategy = strategy
        self.draftedTokens = draftedTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.emittedTokens = emittedTokens
        self.verificationCycles = verificationCycles
        self.draftTime = draftTime
        self.verificationTime = verificationTime
        self.rollbackTime = rollbackTime
    }

    public var meanAcceptanceLength: Double {
        verificationCycles > 0 ? Double(acceptedDraftTokens) / Double(verificationCycles) : 0
    }
}
