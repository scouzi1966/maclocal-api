import XCTest
@testable import AFMKitMLX

final class AFMMLXModelArchitectureTests: XCTestCase {
    func testCanonicalModelTypeNormalizesKnownAliases() {
        XCTAssertEqual(AFMMLXModelArchitecture.canonicalModelType("qwen3.5"), "qwen3_5")
        XCTAssertEqual(AFMMLXModelArchitecture.canonicalModelType("qwen3.6_next"), "qwen3_next")
        XCTAssertEqual(AFMMLXModelArchitecture.canonicalModelType("qwen3.5_vl"), "qwen3_vl")
        XCTAssertEqual(AFMMLXModelArchitecture.canonicalModelType("gemma-4-text"), "gemma4_text")
        XCTAssertEqual(AFMMLXModelArchitecture.canonicalModelType("LLaMA"), "llama")
    }

    func testSupportedAndBlockedModelTypes() {
        XCTAssertTrue(AFMMLXModelArchitecture.isSupported("qwen3.5"))
        XCTAssertTrue(AFMMLXModelArchitecture.isSupported("qwen3_vl"))
        XCTAssertTrue(AFMMLXModelArchitecture.isSupported("afmoe"))
        XCTAssertFalse(AFMMLXModelArchitecture.isSupported("unknown_arch"))

        XCTAssertTrue(AFMMLXModelArchitecture.crashesMetal("afmoe"))
        XCTAssertFalse(AFMMLXModelArchitecture.crashesMetal("qwen3"))
    }

    func testLanguageVisionAndDualModeClassification() {
        XCTAssertTrue(AFMMLXModelArchitecture.isLanguageModelType("llama"))
        XCTAssertFalse(AFMMLXModelArchitecture.isLanguageModelType("qwen3.5"))

        XCTAssertTrue(AFMMLXModelArchitecture.isVisionModelType("qwen3.5"))
        XCTAssertTrue(AFMMLXModelArchitecture.isDualModeModelType("qwen3.5"))
        XCTAssertFalse(AFMMLXModelArchitecture.isDualModeModelType("qwen3_vl"))
    }

    func testDualModeConfigurationRequiresVisionConfig() {
        XCTAssertTrue(
            AFMMLXModelArchitecture.isDualModeConfiguration([
                "model_type": "qwen3.5",
                "text_config": ["model_type": "qwen3_5"],
                "vision_config": ["model_type": "qwen3_vl"],
            ])
        )

        XCTAssertFalse(
            AFMMLXModelArchitecture.isDualModeConfiguration([
                "model_type": "qwen3.5",
                "text_config": ["model_type": "qwen3_5"],
            ])
        )
    }

    func testRepositoryNameHeuristics() {
        XCTAssertTrue(AFMMLXModelArchitecture.matchesSupportedNamePattern("mlx-community/Qwen3-4B-4bit"))
        XCTAssertTrue(AFMMLXModelArchitecture.matchesSupportedNamePattern("mlx-community/FastVLM-1.5B"))
        XCTAssertFalse(AFMMLXModelArchitecture.matchesSupportedNamePattern("example/random-model"))

        XCTAssertTrue(AFMMLXModelArchitecture.looksLikeDualMode("mlx-community/Qwen3.5-35B-A3B-4bit"))
        XCTAssertFalse(AFMMLXModelArchitecture.looksLikeDualMode("mlx-community/Qwen3-VL-4B-Instruct"))

        XCTAssertTrue(AFMMLXModelArchitecture.looksLikeVisionModel("mlx-community/paligemma-3b"))
        XCTAssertFalse(AFMMLXModelArchitecture.looksLikeVisionModel("mlx-community/Llama-3.2-1B-Instruct"))
    }
}
