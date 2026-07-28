import Foundation

public struct AFMMLXModelArchitecturePreflight: Hashable, Sendable {
    public var modelID: String
    public var modelType: String
    public var canonicalModelType: String
    public var isVisionConfiguration: Bool
    public var requiresVisionModelFactory: Bool

    public init(
        modelID: String,
        modelType: String,
        canonicalModelType: String,
        isVisionConfiguration: Bool,
        requiresVisionModelFactory: Bool
    ) {
        self.modelID = modelID
        self.modelType = modelType
        self.canonicalModelType = canonicalModelType
        self.isVisionConfiguration = isVisionConfiguration
        self.requiresVisionModelFactory = requiresVisionModelFactory
    }
}

public struct AFMMLXRemoteModelLoadPlan: Hashable, Sendable {
    public let repoID: String
    public let modelName: String
    public let isVision: Bool
    public let preflightModelType: String?
    public let correctedVisionFromRequest: Bool
    public let forceLLMOnlyApplied: Bool

    public init(
        repoID: String,
        modelName: String,
        isVision: Bool,
        preflightModelType: String?,
        correctedVisionFromRequest: Bool,
        forceLLMOnlyApplied: Bool
    ) {
        self.repoID = repoID
        self.modelName = modelName
        self.isVision = isVision
        self.preflightModelType = preflightModelType
        self.correctedVisionFromRequest = correctedVisionFromRequest
        self.forceLLMOnlyApplied = forceLLMOnlyApplied
    }
}

public enum AFMMLXModelArchitecturePreflightError: Error, LocalizedError, Sendable {
    case invalidConfiguration(String)
    case unsupportedArchitecture(modelType: String, modelID: String)
    case metalCrashArchitecture(modelType: String, modelID: String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let modelID):
            return "Could not read model_type from \(modelID)'s config.json."
        case .unsupportedArchitecture(let modelType, let modelID):
            return "Unsupported model architecture '\(modelType)' for \(modelID)."
        case .metalCrashArchitecture(let modelType, let modelID):
            return "Model '\(modelID)' (architecture: \(modelType)) is blocked because it crashes the Metal GPU driver."
        }
    }
}

public enum AFMMLXRemoteModelLoadPolicy {
    public static func modelName(from repoID: String) -> String {
        String(repoID.split(separator: "/").last ?? Substring(repoID))
    }

    public static func plan(
        repoID: String,
        requestedIsVision: Bool,
        forceLLMOnly: Bool,
        preflight: AFMMLXModelArchitecturePreflight?
    ) -> AFMMLXRemoteModelLoadPlan {
        var resolvedIsVision = preflight?.isVisionConfiguration ?? requestedIsVision
        let correctedVisionFromRequest = resolvedIsVision != requestedIsVision
        let forceLLMOnlyApplied = forceLLMOnly && resolvedIsVision
        if forceLLMOnlyApplied {
            resolvedIsVision = false
        }

        return AFMMLXRemoteModelLoadPlan(
            repoID: repoID,
            modelName: modelName(from: repoID),
            isVision: resolvedIsVision,
            preflightModelType: preflight?.modelType,
            correctedVisionFromRequest: correctedVisionFromRequest,
            forceLLMOnlyApplied: forceLLMOnlyApplied
        )
    }
}

public enum AFMMLXModelArchitecture {
    public static func canonicalModelType(_ modelType: String) -> String {
        switch modelType.lowercased() {
        case "qwen3.5":
            return "qwen3_5"
        case "qwen3.5_moe":
            return "qwen3_5_moe"
        case "qwen3.5_next":
            return "qwen3_next"
        case "qwen3.5_vl":
            return "qwen3_vl"
        case "qwen3.6":
            return "qwen3_6"
        case "qwen3.6_moe":
            return "qwen3_6_moe"
        case "qwen3_6_next", "qwen3.6_next":
            return "qwen3_next"
        case "qwen3_6_vl", "qwen3.6_vl":
            return "qwen3_vl"
        case "qwen3_5_vl":
            return "qwen3_vl"
        case "gemma_4", "gemma-4":
            return "gemma4"
        case "gemma_4_text", "gemma-4-text":
            return "gemma4_text"
        default:
            return modelType.lowercased()
        }
    }

    public static let languageModelTypes: Set<String> = [
        "llama",
        "mistral",
        "mistral3",
        "phi",
        "phi3",
        "phimoe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma3n",
        "gemma4",
        "gemma4_text",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_6",
        "qwen3_6_moe",
        "acereason",
        "starcoder2",
        "cohere",
        "openelm",
        "internlm2",
        "deepseek_v3",
        "kimi_k2",
        "kimi_k25",
        "joyai_llm_flash",
        "nemotron_h",
        "granite",
        "granitemoehybrid",
        "mimo",
        "glm4",
        "glm4_moe",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "falcon_h1",
        "bitnet",
        "smollm3",
        "ernie4_5",
        "lfm2",
        "lfm2_moe",
        "baichuan_m1",
        "exaone4",
        "gpt_oss",
        "lille-130m",
        "olmoe",
        "olmo2",
        "olmo3",
        "bailing_moe",
        "nanochat",
        "afmoe",
        "jamba_3b",
        "apertus",
        "minimax_m2"
    ]

    public static let visionModelTypes: Set<String> = [
        "paligemma",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_6",
        "qwen3_6_moe",
        "idefics3",
        "gemma3",
        "smolvlm",
        "fastvlm",
        "llava_qwen2",
        "pixtral",
        "mistral3",
        "lfm2_vl",
        "lfm2-vl"
    ]

    public static let metalCrashModelTypes: Set<String> = [
        "afmoe",
    ]

    public static var supportedModelTypes: Set<String> {
        languageModelTypes.union(visionModelTypes)
    }

    public static func isSupported(_ modelType: String) -> Bool {
        supportedModelTypes.contains(canonicalModelType(modelType))
    }

    public static func crashesMetal(_ modelType: String) -> Bool {
        metalCrashModelTypes.contains(canonicalModelType(modelType))
    }

    public static func isVisionModelType(_ modelType: String) -> Bool {
        visionModelTypes.contains(canonicalModelType(modelType))
    }

    public static func isLanguageModelType(_ modelType: String) -> Bool {
        let canonical = canonicalModelType(modelType)
        return languageModelTypes.contains(canonical) && !visionModelTypes.contains(canonical)
    }

    public static func isDualModeModelType(_ modelType: String) -> Bool {
        let canonical = canonicalModelType(modelType)
        return languageModelTypes.contains(canonical) && visionModelTypes.contains(canonical)
    }

    public static func isDualModeConfiguration(_ config: [String: Any]) -> Bool {
        guard let modelType = config["model_type"] as? String,
              isDualModeModelType(modelType) else {
            return false
        }
        return AFMMLXModelDescriptor.isVisionModelConfiguration(config)
    }

    public static func isDualModeConfiguration(in modelDirectory: URL) -> Bool {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        return isDualModeConfiguration(config)
    }

    public static func preflightConfiguration(
        _ config: [String: Any],
        modelID: String
    ) throws -> AFMMLXModelArchitecturePreflight {
        guard let modelType = config["model_type"] as? String else {
            throw AFMMLXModelArchitecturePreflightError.invalidConfiguration(modelID)
        }

        guard isSupported(modelType) else {
            throw AFMMLXModelArchitecturePreflightError.unsupportedArchitecture(
                modelType: modelType,
                modelID: modelID
            )
        }

        if crashesMetal(modelType) {
            throw AFMMLXModelArchitecturePreflightError.metalCrashArchitecture(
                modelType: modelType,
                modelID: modelID
            )
        }

        return AFMMLXModelArchitecturePreflight(
            modelID: modelID,
            modelType: modelType,
            canonicalModelType: canonicalModelType(modelType),
            isVisionConfiguration: AFMMLXModelDescriptor.isVisionModelConfiguration(config),
            requiresVisionModelFactory: AFMMLXModelDescriptor.requiresVisionModelFactory(config)
        )
    }

    public static let supportedNamePatterns: [String] = [
        "apertus",
        "baichuan",
        "bitnet",
        "cohere",
        "deepseek",
        "ernie",
        "exaone",
        "falcon",
        "gemma",
        "glm-4-",
        "glm-4.7",
        "gpt-oss",
        "granite",
        "internlm",
        "jamba",
        "kimi",
        "lfm",
        "lille",
        "llama",
        "minimax",
        "mistral",
        "mixtral",
        "nanochat",
        "nemotron",
        "olmo",
        "openelm",
        "phi",
        "qwen",
        "smollm",
        "starcoder",
        "fastvlm",
        "idefics",
        "llava",
        "paligemma",
        "pixtral"
    ]

    public static let dualModeNamePatterns: [String] = [
        "qwen3.5-",
        "qwen3.5_",
        "qwen3.6-",
        "qwen3.6_"
    ]

    public static let visionNamePatterns: [String] = [
        "-vl",
        "_vl",
        "vision",
        "llava",
        "paligemma",
        "idefics",
        "pixtral",
        "fastvlm",
        "smolvlm"
    ]

    public static func matchesSupportedNamePattern(_ repoID: String) -> Bool {
        let lowerID = repoID.lowercased()
        return supportedNamePatterns.contains { lowerID.contains($0) }
    }

    public static func looksLikeDualMode(_ repoID: String) -> Bool {
        let lowerID = repoID.lowercased()
        return dualModeNamePatterns.contains { lowerID.contains($0) }
    }

    public static func looksLikeVisionModel(_ repoID: String) -> Bool {
        let lowerID = repoID.lowercased()
        return visionNamePatterns.contains { lowerID.contains($0) }
    }
}
