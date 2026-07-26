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

    func testDualModeConfigurationReadsModelDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let config: [String: Any] = [
            "model_type": "qwen3.5",
            "text_config": ["model_type": "qwen3_5"],
            "vision_config": ["model_type": "qwen3_vl"],
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))

        XCTAssertTrue(AFMMLXModelArchitecture.isDualModeConfiguration(in: directory))
    }

    func testPreflightConfigurationReturnsSharedLoadPolicy() throws {
        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            [
                "model_type": "qwen3.5",
                "text_config": ["model_type": "qwen3_5"],
                "vision_config": ["model_type": "qwen3_vl"],
            ],
            modelID: "mlx-community/Qwen3.5-35B-A3B-4bit"
        )

        XCTAssertEqual(preflight.modelType, "qwen3.5")
        XCTAssertEqual(preflight.canonicalModelType, "qwen3_5")
        XCTAssertTrue(preflight.isVisionConfiguration)
        XCTAssertTrue(preflight.requiresVisionModelFactory)
    }

    func testPreflightConfigurationRejectsUnsupportedAndBlockedArchitectures() {
        XCTAssertThrowsError(try AFMMLXModelArchitecture.preflightConfiguration(
            ["model_type": "unknown_arch"],
            modelID: "example/Unknown"
        )) { error in
            guard case AFMMLXModelArchitecturePreflightError.unsupportedArchitecture(
                let modelType,
                let modelID
            ) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(modelType, "unknown_arch")
            XCTAssertEqual(modelID, "example/Unknown")
        }

        XCTAssertThrowsError(try AFMMLXModelArchitecture.preflightConfiguration(
            ["model_type": "afmoe"],
            modelID: "apple/AFMoE"
        )) { error in
            guard case AFMMLXModelArchitecturePreflightError.metalCrashArchitecture(
                let modelType,
                let modelID
            ) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(modelType, "afmoe")
            XCTAssertEqual(modelID, "apple/AFMoE")
        }
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
