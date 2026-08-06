import Foundation
import AFMKitCore
import MLXLMCommon
import MLXLLM
import MLXVLM

public struct AFMMLXGenerationPreset: Hashable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var repetitionPenalty: Double?
    public var maxTokens: Int?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        repetitionPenalty: Double? = nil,
        maxTokens: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
    }

    public static func generationConfigPreset(_ json: [String: Any]) -> AFMMLXGenerationPreset? {
        let temperature = doubleValue(json["temperature"])
        let topP = doubleValue(json["top_p"])
        let repetitionPenalty = doubleValue(json["repetition_penalty"])
        let maxTokens = intValue(json["max_new_tokens"])

        guard temperature != nil || topP != nil || repetitionPenalty != nil || maxTokens != nil else {
            return nil
        }

        return AFMMLXGenerationPreset(
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            maxTokens: maxTokens
        )
    }

    public static func generationConfigPreset(in modelDirectory: URL) -> AFMMLXGenerationPreset? {
        let url = modelDirectory.appendingPathComponent("generation_config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        return generationConfigPreset(json)
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        default:
            return nil
        }
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as Double where value.rounded() == value:
            return Int(value)
        case let value as NSNumber:
            return value.intValue
        default:
            return nil
        }
    }
}

public struct AFMMLXLocalModelMetadata: Hashable, Sendable {
    public var modelType: String?
    public var contextWindow: Int?
    public var generationPreset: AFMMLXGenerationPreset?
    public var hasImplicitReasoning: Bool
    public var supportsThinkingToggle: Bool

    public init(
        modelType: String? = nil,
        contextWindow: Int? = nil,
        generationPreset: AFMMLXGenerationPreset? = nil,
        hasImplicitReasoning: Bool = false,
        supportsThinkingToggle: Bool = false
    ) {
        self.modelType = modelType
        self.contextWindow = contextWindow
        self.generationPreset = generationPreset
        self.hasImplicitReasoning = hasImplicitReasoning
        self.supportsThinkingToggle = supportsThinkingToggle
    }

    public static func inspect(
        modelDirectory: URL,
        modelName: String
    ) -> AFMMLXLocalModelMetadata {
        let config = jsonObject(at: modelDirectory.appendingPathComponent("config.json"))
        let tokenizer = jsonObject(at: modelDirectory.appendingPathComponent("tokenizer_config.json"))
        let generation = jsonObject(at: modelDirectory.appendingPathComponent("generation_config.json"))
        let jinja = try? String(
            contentsOf: modelDirectory.appendingPathComponent("chat_template.jinja"),
            encoding: .utf8
        )

        let templates = chatTemplates(in: tokenizer)
        let templateDefaultsToThinking = templates.contains(where: chatTemplateDefaultsToThinking)
            || jinja.map(chatTemplateDefaultsToThinking) == true
        let generationDefaultsToThinking = generation?["enable_thinking"] as? Bool == true
        let implicitReasoning = templateDefaultsToThinking
            || generationDefaultsToThinking
            || modelNameLooksReasoningCapable(modelName)
        let templateSupportsThinkingToggle = templates.contains { $0.contains("enable_thinking") }
            || jinja?.contains("enable_thinking") == true

        return AFMMLXLocalModelMetadata(
            modelType: config?["model_type"] as? String,
            contextWindow: config?["max_position_embeddings"] as? Int,
            generationPreset: generation.flatMap(AFMMLXGenerationPreset.generationConfigPreset),
            hasImplicitReasoning: implicitReasoning,
            supportsThinkingToggle: implicitReasoning && templateSupportsThinkingToggle
        )
    }

    public static func inspect(modelName: String) -> AFMMLXLocalModelMetadata {
        AFMMLXLocalModelMetadata(
            hasImplicitReasoning: modelNameLooksReasoningCapable(modelName)
        )
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func chatTemplates(in tokenizerConfig: [String: Any]?) -> [String] {
        guard let tokenizerConfig else { return [] }
        if let template = tokenizerConfig["chat_template"] as? String {
            return [template]
        }
        if let templates = tokenizerConfig["chat_template"] as? [[String: Any]] {
            return templates.compactMap { $0["template"] as? String }
        }
        return []
    }

    private static func chatTemplateDefaultsToThinking(_ template: String) -> Bool {
        guard template.contains("<think>") && template.contains("add_generation_prompt") else {
            return false
        }
        if !template.contains("enable_thinking") {
            return true
        }
        return template.contains("enable_thinking is false")
    }

    private static func modelNameLooksReasoningCapable(_ modelName: String) -> Bool {
        let nameLower = modelName.lowercased()
        return nameLower.contains("thinking") || nameLower.contains("nemotron") ||
               nameLower.contains("glm-5") || nameLower.contains("glm-4.7") ||
               nameLower.contains("glm 4.7") || nameLower.contains("minimax") ||
               nameLower.contains("kimi-k2") || nameLower.contains("kimi k2") ||
               nameLower.contains("kimi-dev")
    }
}

public struct AFMMLXCuratedModel: Hashable, Identifiable, Sendable {
    public var id: String { repoID }

    public var displayName: String
    public var repoID: String
    public var capabilities: AFMModelCapabilities
    public var contextWindow: Int?
    public var generationPreset: AFMMLXGenerationPreset

    public init(
        displayName: String,
        repoID: String,
        capabilities: AFMModelCapabilities,
        contextWindow: Int? = nil,
        generationPreset: AFMMLXGenerationPreset
    ) {
        self.displayName = displayName
        self.repoID = repoID
        self.capabilities = capabilities
        self.contextWindow = contextWindow
        self.generationPreset = generationPreset
    }

    public var isLanguageModel: Bool {
        capabilities.contains(.text)
    }

    public var isVisionModel: Bool {
        capabilities.contains(.vision)
    }

    public var descriptor: AFMModelDescriptor {
        var metadata: [String: AFMJSONValue] = [
            "repoID": .string(repoID),
            "catalog": .string("afmkit-mlx-curated"),
        ]
        metadata["temperature"] = generationPreset.temperature.map { .number($0) }
        metadata["topP"] = generationPreset.topP.map { .number($0) }
        metadata["repetitionPenalty"] = generationPreset.repetitionPenalty.map { .number($0) }
        metadata["maxTokens"] = generationPreset.maxTokens.map { .integer($0) }

        return AFMModelDescriptor(
            providerID: AFMMLXProviderFactory.providerID,
            modelID: AFMModelID(rawValue: repoID),
            displayName: displayName,
            capabilities: capabilities,
            contextWindow: contextWindow,
            privacyBoundary: .device,
            requiresNetwork: false,
            metadata: metadata
        )
    }

    public var modelConfiguration: ModelConfiguration? {
        AFMMLXModelCatalog.modelConfiguration(for: repoID)
    }
}

public enum AFMMLXModelCatalog {
    public static let defaultModelID = "mlx-community/Qwen3-VL-4B-Instruct-5bit"

    public static let textModels: [AFMMLXCuratedModel] = [
        textModel(
            displayName: "Qwen3-0.6B-4bit",
            repoID: "mlx-community/Qwen3-0.6B-4bit",
            temperature: 0.7,
            topP: 0.8,
            maxTokens: 8192
        ),
        textModel(
            displayName: "Qwen3-Coder-Next-4bit",
            repoID: "mlx-community/Qwen3-Coder-Next-4bit",
            temperature: 0.2,
            topP: 0.95,
            maxTokens: 16384
        ),
        textModel(
            displayName: "Qwen3.6-35B-A3B-4bit",
            repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
            temperature: 0.6,
            topP: 0.95,
            maxTokens: 32768
        ),
        textModel(
            displayName: "Gemma-3-4B-it-8bit",
            repoID: "mlx-community/gemma-3-4b-it-8bit",
            temperature: 0.7,
            topP: 0.95,
            maxTokens: 16384
        ),
        textModel(
            displayName: "Llama-3.2-1B-Instruct-4bit",
            repoID: "mlx-community/Llama-3.2-1B-Instruct-4bit",
            temperature: 0.7,
            topP: 0.9,
            maxTokens: 8192
        ),
        textModel(
            displayName: "Qwen2.5-0.5B-Instruct-4bit",
            repoID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            temperature: 0.7,
            topP: 0.8,
            maxTokens: 8192
        ),
        textModel(
            displayName: "GPT-OSS-20B-MXFP4-Q8",
            repoID: "mlx-community/gpt-oss-20b-MXFP4-Q8",
            temperature: 0.6,
            topP: 0.95,
            maxTokens: 16384
        ),
        textModel(
            displayName: "North-Mini-Code-1.0-4bit",
            repoID: "mlx-community/North-Mini-Code-1.0-4bit",
            temperature: 0.3,
            topP: 0.95,
            maxTokens: 16384
        ),
    ]

    public static let visionModels: [AFMMLXCuratedModel] = [
        visionModel(
            displayName: "Gemma-4-E2B-it-4bit",
            repoID: "mlx-community/gemma-4-e2b-it-4bit",
            temperature: 1.0,
            topP: 0.95,
            maxTokens: 8192
        ),
        visionModel(
            displayName: "Gemma-4-E4B-it-4bit",
            repoID: "mlx-community/gemma-4-e4b-it-4bit",
            temperature: 1.0,
            topP: 0.95,
            maxTokens: 16384
        ),
        visionModel(
            displayName: "Qwen3-VL-4B-Instruct-4bit",
            repoID: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            maxTokens: 16384
        ),
        visionModel(
            displayName: "Qwen3-VL-4B-Instruct-5bit",
            repoID: "mlx-community/Qwen3-VL-4B-Instruct-5bit",
            maxTokens: 16384
        ),
        visionModel(
            displayName: "Qwen3-VL-4B-Instruct-8bit",
            repoID: "mlx-community/Qwen3-VL-4B-Instruct-8bit",
            maxTokens: 16384
        ),
        visionModel(
            displayName: "Qwen3-VL-8B-Instruct-4bit",
            repoID: "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Qwen3-VL-8B-Instruct-5bit",
            repoID: "mlx-community/Qwen3-VL-8B-Instruct-5bit",
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Qwen3-VL-8B-Instruct-8bit",
            repoID: "mlx-community/Qwen3-VL-8B-Instruct-8bit",
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Qwen3-VL-8B-Instruct-bf16",
            repoID: "mlx-community/Qwen3-VL-8B-Instruct-bf16",
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Qwen3.6-27B-Instruct-4bit",
            repoID: "mlx-community/Qwen3.6-27B-4bit",
            temperature: 0.6,
            topP: 0.95,
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Gemma-4-26B-A4B-it-4bit",
            repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
            temperature: 1.0,
            topP: 0.95,
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Gemma-4-31B-it-4bit",
            repoID: "mlx-community/gemma-4-31b-it-4bit",
            temperature: 1.0,
            topP: 0.95,
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Gemma-4-31B-it-8bit",
            repoID: "mlx-community/gemma-4-31b-it-8bit",
            temperature: 1.0,
            topP: 0.95,
            maxTokens: 32768
        ),
        visionModel(
            displayName: "Qwen3.6-35B-A3B-8bit",
            repoID: "mlx-community/Qwen3.6-35B-A3B-8bit",
            temperature: 0.6,
            topP: 0.95,
            maxTokens: 32768
        ),
    ]

    public static let availableModels: [AFMMLXCuratedModel] = textModels + visionModels

    public static func model(for repoID: String) -> AFMMLXCuratedModel? {
        availableModels.first { $0.repoID == repoID }
    }

    public static var genericTextModelConfiguration: ModelConfiguration {
        LLMRegistry.llama3_2_1B_4bit
    }

    public static var genericVisionModelConfiguration: ModelConfiguration {
        VLMRegistry.qwen3VL4BInstruct4Bit
    }

    public static func genericModelConfiguration(isVision: Bool) -> ModelConfiguration {
        isVision ? genericVisionModelConfiguration : genericTextModelConfiguration
    }

    public static func modelConfiguration(for repoID: String) -> ModelConfiguration? {
        switch repoID {
        case "mlx-community/Qwen3-0.6B-4bit",
             "mlx-community/Qwen2.5-0.5B-Instruct-4bit":
            return LLMRegistry.qwen3_0_6b_4bit
        case "mlx-community/Qwen3-Coder-Next-4bit",
             "mlx-community/gemma-3-4b-it-8bit":
            return LLMRegistry.qwen3_4b_4bit
        case "mlx-community/Qwen3.6-35B-A3B-4bit",
             "mlx-community/North-Mini-Code-1.0-4bit",
             "mlx-community/gpt-oss-20b-MXFP4-Q8":
            return LLMRegistry.qwen3_8b_4bit
        case "mlx-community/Llama-3.2-1B-Instruct-4bit":
            return LLMRegistry.llama3_2_1B_4bit
        case "mlx-community/Qwen3-VL-4B-Instruct-4bit",
             "mlx-community/Qwen3-VL-4B-Instruct-5bit",
             "mlx-community/Qwen3-VL-8B-Instruct-4bit",
             "mlx-community/Qwen3-VL-8B-Instruct-5bit",
             "mlx-community/gemma-4-e2b-it-4bit",
             "mlx-community/gemma-4-e4b-it-4bit",
             "mlx-community/Qwen3.6-27B-4bit",
             "mlx-community/gemma-4-26b-a4b-it-4bit",
             "mlx-community/gemma-4-31b-it-4bit":
            return VLMRegistry.qwen3VL4BInstruct4Bit
        case "mlx-community/Qwen3-VL-4B-Instruct-8bit",
             "mlx-community/Qwen3-VL-8B-Instruct-8bit",
             "mlx-community/Qwen3-VL-8B-Instruct-bf16",
             "mlx-community/gemma-4-31b-it-8bit",
             "mlx-community/Qwen3.6-35B-A3B-8bit":
            return VLMRegistry.qwen3VL4BInstruct8Bit
        default:
            return nil
        }
    }

    private static func textModel(
        displayName: String,
        repoID: String,
        temperature: Double,
        topP: Double,
        maxTokens: Int
    ) -> AFMMLXCuratedModel {
        AFMMLXCuratedModel(
            displayName: displayName,
            repoID: repoID,
            capabilities: [.text, .streaming],
            contextWindow: maxTokens,
            generationPreset: AFMMLXGenerationPreset(
                temperature: temperature,
                topP: topP,
                repetitionPenalty: nil,
                maxTokens: maxTokens
            )
        )
    }

    private static func visionModel(
        displayName: String,
        repoID: String,
        temperature: Double = 0.7,
        topP: Double = 0.8,
        maxTokens: Int
    ) -> AFMMLXCuratedModel {
        AFMMLXCuratedModel(
            displayName: displayName,
            repoID: repoID,
            capabilities: [.text, .vision, .streaming],
            contextWindow: maxTokens,
            generationPreset: AFMMLXGenerationPreset(
                temperature: temperature,
                topP: topP,
                repetitionPenalty: nil,
                maxTokens: maxTokens
            )
        )
    }
}
