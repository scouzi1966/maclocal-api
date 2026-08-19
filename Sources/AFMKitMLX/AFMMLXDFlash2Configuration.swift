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
