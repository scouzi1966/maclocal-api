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
}
