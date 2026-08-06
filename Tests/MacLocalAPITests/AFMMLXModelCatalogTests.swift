import XCTest
@testable import AFMKitMLX

final class AFMMLXModelCatalogTests: XCTestCase {
    func testCatalogPreservesVestaModelOrderAndDefault() {
        XCTAssertEqual(
            AFMMLXModelCatalog.availableModels.map(\.repoID),
            [
                "mlx-community/Qwen3-0.6B-4bit",
                "mlx-community/Qwen3-Coder-Next-4bit",
                "mlx-community/Qwen3.5-35B-A3B-4bit",
                "mlx-community/gemma-3-4b-it-8bit",
                "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "mlx-community/gpt-oss-20b-MXFP4-Q8",
                "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                "mlx-community/Qwen3-VL-4B-Instruct-5bit",
                "mlx-community/Qwen3-VL-4B-Instruct-8bit",
                "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                "mlx-community/Qwen3-VL-8B-Instruct-5bit",
                "mlx-community/Qwen3-VL-8B-Instruct-8bit",
                "mlx-community/Qwen3-VL-8B-Instruct-bf16",
            ]
        )
        XCTAssertEqual(
            AFMMLXModelCatalog.defaultModelID,
            "mlx-community/Qwen3-VL-4B-Instruct-5bit"
        )
    }

    func testGenerationPresetsPreserveKnownValues() throws {
        let smallQwen = try XCTUnwrap(
            AFMMLXModelCatalog.model(for: "mlx-community/Qwen3-0.6B-4bit")
        )
        XCTAssertEqual(smallQwen.generationPreset.temperature, 0.7)
        XCTAssertEqual(smallQwen.generationPreset.topP, 0.8)
        XCTAssertEqual(smallQwen.generationPreset.maxTokens, 8192)

        let coder = try XCTUnwrap(
            AFMMLXModelCatalog.model(for: "mlx-community/Qwen3-Coder-Next-4bit")
        )
        XCTAssertEqual(coder.generationPreset.temperature, 0.2)
        XCTAssertEqual(coder.generationPreset.topP, 0.95)
        XCTAssertEqual(coder.generationPreset.maxTokens, 16384)

        let vision = try XCTUnwrap(
            AFMMLXModelCatalog.model(for: "mlx-community/Qwen3-VL-8B-Instruct-5bit")
        )
        XCTAssertTrue(vision.isVisionModel)
        XCTAssertEqual(vision.generationPreset.temperature, 0.7)
        XCTAssertEqual(vision.generationPreset.topP, 0.8)
        XCTAssertEqual(vision.generationPreset.maxTokens, 32768)
    }

    func testGenerationConfigPresetReadsKnownSamplingKeys() throws {
        let preset = try XCTUnwrap(AFMMLXGenerationPreset.generationConfigPreset([
            "temperature": 0.2,
            "top_p": 1,
            "repetition_penalty": 1.05,
            "max_new_tokens": 4096,
            "ignored": "value",
        ]))

        XCTAssertEqual(preset.temperature, 0.2)
        XCTAssertEqual(preset.topP, 1.0)
        XCTAssertEqual(preset.repetitionPenalty, 1.05)
        XCTAssertEqual(preset.maxTokens, 4096)
    }

    func testGenerationConfigPresetReturnsNilWithoutSamplingKeys() {
        XCTAssertNil(AFMMLXGenerationPreset.generationConfigPreset([
            "enable_thinking": true,
            "chat_template": "{{ messages }}",
        ]))
    }

    func testGenerationConfigPresetReadsModelDirectoryFile() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMMLXGenerationPreset-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = try JSONSerialization.data(withJSONObject: [
            "temperature": 0.4,
            "top_p": 0.9,
            "max_new_tokens": 128,
        ])
        try data.write(to: directory.appendingPathComponent("generation_config.json"))

        let preset = try XCTUnwrap(AFMMLXGenerationPreset.generationConfigPreset(in: directory))
        XCTAssertEqual(preset.temperature, 0.4)
        XCTAssertEqual(preset.topP, 0.9)
        XCTAssertNil(preset.repetitionPenalty)
        XCTAssertEqual(preset.maxTokens, 128)
    }

    func testLocalModelMetadataReadsDirectoryDefaults() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMMLXLocalModelMetadata-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen3.6",
            "max_position_embeddings": 65_536,
        ]).write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: [
            "temperature": 0.25,
            "top_p": 0.9,
            "enable_thinking": true,
        ]).write(to: directory.appendingPathComponent("generation_config.json"))
        try JSONSerialization.data(withJSONObject: [
            "chat_template": "{% if add_generation_prompt %}{% if enable_thinking is false %}<think></think>{% else %}<think>{% endif %}{% endif %}",
        ]).write(to: directory.appendingPathComponent("tokenizer_config.json"))

        let metadata = AFMMLXLocalModelMetadata.inspect(
            modelDirectory: directory,
            modelName: "Qwen3.6-27B"
        )

        XCTAssertEqual(metadata.modelType, "qwen3.6")
        XCTAssertEqual(metadata.contextWindow, 65_536)
        XCTAssertEqual(metadata.generationPreset?.temperature, 0.25)
        XCTAssertEqual(metadata.generationPreset?.topP, 0.9)
        XCTAssertTrue(metadata.hasImplicitReasoning)
        XCTAssertTrue(metadata.supportsThinkingToggle)
    }

    func testLocalModelMetadataUsesReasoningNameFallback() throws {
        let metadata = AFMMLXLocalModelMetadata.inspect(modelName: "mlx-community/Kimi-K2.5-4bit")

        XCTAssertTrue(metadata.hasImplicitReasoning)
        XCTAssertFalse(metadata.supportsThinkingToggle)
        XCTAssertNil(metadata.generationPreset)
    }

    func testDescriptorsExposeCapabilitiesAndMetadata() throws {
        let model = try XCTUnwrap(
            AFMMLXModelCatalog.model(for: "mlx-community/Qwen3-VL-4B-Instruct-5bit")
        )
        let descriptor = model.descriptor

        XCTAssertEqual(descriptor.providerID.rawValue, "mlx")
        XCTAssertEqual(descriptor.modelID.rawValue, model.repoID)
        XCTAssertEqual(descriptor.displayName, model.displayName)
        XCTAssertEqual(descriptor.privacyBoundary, .device)
        XCTAssertEqual(descriptor.requiresNetwork, false)
        XCTAssertTrue(descriptor.capabilities.contains(.text))
        XCTAssertTrue(descriptor.capabilities.contains(.vision))
        XCTAssertTrue(descriptor.capabilities.contains(.streaming))
        XCTAssertEqual(descriptor.contextWindow, 16384)
        XCTAssertEqual(descriptor.metadata["repoID"], .string(model.repoID))
        XCTAssertEqual(descriptor.metadata["catalog"], .string("afmkit-mlx-curated"))
        XCTAssertEqual(descriptor.metadata["maxTokens"], .integer(16384))
    }

    func testCuratedModelsExposeLoadableMLXConfigurations() {
        for model in AFMMLXModelCatalog.availableModels {
            XCTAssertNotNil(
                model.modelConfiguration,
                "\(model.repoID) is missing its MLX runtime configuration"
            )
            XCTAssertNotNil(
                AFMMLXModelCatalog.modelConfiguration(for: model.repoID),
                "\(model.repoID) is missing its catalog MLX runtime configuration"
            )
        }

        XCTAssertNil(AFMMLXModelCatalog.modelConfiguration(for: "example/missing"))
    }

    func testGenericRuntimeConfigurationsProvideStableFallbacks() {
        XCTAssertEqual(
            AFMMLXModelCatalog.genericModelConfiguration(isVision: false).name,
            "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
        XCTAssertEqual(
            AFMMLXModelCatalog.genericModelConfiguration(isVision: true).name,
            "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"
        )
    }
}
