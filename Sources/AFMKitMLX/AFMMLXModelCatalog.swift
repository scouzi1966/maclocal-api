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
            displayName: "Qwen3.5-35B-A3B-4bit",
            repoID: "mlx-community/Qwen3.5-35B-A3B-4bit",
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
    ]

    public static let visionModels: [AFMMLXCuratedModel] = [
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
    ]

    public static let availableModels: [AFMMLXCuratedModel] = textModels + visionModels

    public static func model(for repoID: String) -> AFMMLXCuratedModel? {
        availableModels.first { $0.repoID == repoID }
    }

    public static func modelConfiguration(for repoID: String) -> ModelConfiguration? {
        switch repoID {
        case "mlx-community/Qwen3-0.6B-4bit",
             "mlx-community/Qwen2.5-0.5B-Instruct-4bit":
            return LLMRegistry.qwen3_0_6b_4bit
        case "mlx-community/Qwen3-Coder-Next-4bit",
             "mlx-community/gemma-3-4b-it-8bit":
            return LLMRegistry.qwen3_4b_4bit
        case "mlx-community/Qwen3.5-35B-A3B-4bit",
             "mlx-community/gpt-oss-20b-MXFP4-Q8":
            return LLMRegistry.qwen3_8b_4bit
        case "mlx-community/Llama-3.2-1B-Instruct-4bit":
            return LLMRegistry.llama3_2_1B_4bit
        case "mlx-community/Qwen3-VL-4B-Instruct-4bit",
             "mlx-community/Qwen3-VL-4B-Instruct-5bit",
             "mlx-community/Qwen3-VL-8B-Instruct-4bit",
             "mlx-community/Qwen3-VL-8B-Instruct-5bit":
            return VLMRegistry.qwen3VL4BInstruct4Bit
        case "mlx-community/Qwen3-VL-4B-Instruct-8bit",
             "mlx-community/Qwen3-VL-8B-Instruct-8bit",
             "mlx-community/Qwen3-VL-8B-Instruct-bf16":
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
        maxTokens: Int
    ) -> AFMMLXCuratedModel {
        AFMMLXCuratedModel(
            displayName: displayName,
            repoID: repoID,
            capabilities: [.text, .vision, .streaming],
            contextWindow: maxTokens,
            generationPreset: AFMMLXGenerationPreset(
                temperature: 0.7,
                topP: 0.8,
                repetitionPenalty: nil,
                maxTokens: maxTokens
            )
        )
    }
}
