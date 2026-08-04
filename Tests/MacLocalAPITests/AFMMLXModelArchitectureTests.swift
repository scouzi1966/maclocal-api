import XCTest
@testable import AFMKitMLX

final class AFMMLXModelArchitectureTests: XCTestCase {
    func testDeepSeekV4UltraUsesMeasuredMetalSchedulingLimits() {
        XCTAssertEqual(
            AFMMLXMetalSchedulingPolicy.recommendedLimits(
                canonicalModelType: "deepseek_v4",
                processorBrand: "Apple M3 Ultra",
                environment: [:]),
            AFMMLXMetalSchedulingLimits(
                maxOperationsPerBuffer: 200,
                maxMegabytesPerBuffer: 100_000)
        )
    }

    func testMetalSchedulingPolicyDoesNotAffectOtherArchitecturesOrHardware() {
        XCTAssertNil(AFMMLXMetalSchedulingPolicy.recommendedLimits(
            canonicalModelType: "qwen3_5_moe",
            processorBrand: "Apple M3 Ultra",
            environment: [:]))
        XCTAssertNil(AFMMLXMetalSchedulingPolicy.recommendedLimits(
            canonicalModelType: "deepseek_v4",
            processorBrand: "Apple M3 Max",
            environment: [:]))
    }

    func testMetalSchedulingPolicyPreservesExplicitMLXOverrides() {
        XCTAssertNil(AFMMLXMetalSchedulingPolicy.recommendedLimits(
            canonicalModelType: "deepseek_v4",
            processorBrand: "Apple M3 Ultra",
            environment: [AFMMLXMetalSchedulingPolicy.operationsEnvironmentKey: "75"]))
        XCTAssertNil(AFMMLXMetalSchedulingPolicy.recommendedLimits(
            canonicalModelType: "deepseek_v4",
            processorBrand: "Apple M3 Ultra",
            environment: [AFMMLXMetalSchedulingPolicy.megabytesEnvironmentKey: "125"]))
    }

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
        XCTAssertTrue(AFMMLXModelArchitecture.isSupported("deepseek_v4"))
        XCTAssertTrue(AFMMLXModelArchitecture.isSupported("afmoe"))
        XCTAssertFalse(AFMMLXModelArchitecture.isSupported("unknown_arch"))

        XCTAssertTrue(AFMMLXModelArchitecture.crashesMetal("afmoe"))
        XCTAssertFalse(AFMMLXModelArchitecture.crashesMetal("qwen3"))
    }

    func testLanguageVisionAndDualModeClassification() {
        XCTAssertTrue(AFMMLXModelArchitecture.isLanguageModelType("llama"))
        XCTAssertTrue(AFMMLXModelArchitecture.isLanguageModelType("deepseek_v4"))
        XCTAssertFalse(AFMMLXModelArchitecture.isVisionModelType("deepseek_v4"))
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

    func testDeepseekV40731PreflightUsesLanguageModelFactory() throws {
        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            [
                "model_type": "deepseek_v4",
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 43,
                "n_routed_experts": 256,
                "num_nextn_predict_layers": 1,
            ],
            modelID: "Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX"
        )

        XCTAssertEqual(preflight.canonicalModelType, "deepseek_v4")
        XCTAssertFalse(preflight.isVisionConfiguration)
        XCTAssertFalse(preflight.requiresVisionModelFactory)
    }

    func testDirectoryPreflightDetectsArchitectureIndependentOfModelID() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = try JSONSerialization.data(withJSONObject: [
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
        ])
        try data.write(to: directory.appendingPathComponent("config.json"))

        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            in: directory,
            modelID: "local/arbitrary-folder-name"
        )

        XCTAssertEqual(preflight.canonicalModelType, "deepseek_v4")
        XCTAssertEqual(preflight.modelID, "local/arbitrary-folder-name")
    }

    func testRemoteModelLoadPlanUsesRequestedVisionWithoutPreflight() {
        let plan = AFMMLXRemoteModelLoadPolicy.plan(
            repoID: "mlx-community/Qwen3-4B-4bit",
            requestedIsVision: false,
            forceLLMOnly: false,
            preflight: nil
        )

        XCTAssertEqual(plan.repoID, "mlx-community/Qwen3-4B-4bit")
        XCTAssertEqual(plan.modelName, "Qwen3-4B-4bit")
        XCTAssertFalse(plan.isVision)
        XCTAssertNil(plan.preflightModelType)
        XCTAssertFalse(plan.correctedVisionFromRequest)
        XCTAssertFalse(plan.forceLLMOnlyApplied)
    }

    func testRemoteModelLoadPlanCorrectsVisionFromPreflight() {
        let preflight = AFMMLXModelArchitecturePreflight(
            modelID: "mlx-community/Qwen3.5-35B-A3B-4bit",
            modelType: "qwen3.5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: true,
            requiresVisionModelFactory: true
        )

        let plan = AFMMLXRemoteModelLoadPolicy.plan(
            repoID: preflight.modelID,
            requestedIsVision: false,
            forceLLMOnly: false,
            preflight: preflight
        )

        XCTAssertEqual(plan.modelName, "Qwen3.5-35B-A3B-4bit")
        XCTAssertTrue(plan.isVision)
        XCTAssertEqual(plan.preflightModelType, "qwen3.5")
        XCTAssertTrue(plan.correctedVisionFromRequest)
        XCTAssertFalse(plan.forceLLMOnlyApplied)
    }

    func testRemoteModelLoadPlanForceLLMOnlyOverridesVisionPreflight() {
        let preflight = AFMMLXModelArchitecturePreflight(
            modelID: "mlx-community/Qwen3.5-35B-A3B-4bit",
            modelType: "qwen3.5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: true,
            requiresVisionModelFactory: true
        )

        let plan = AFMMLXRemoteModelLoadPolicy.plan(
            repoID: preflight.modelID,
            requestedIsVision: true,
            forceLLMOnly: true,
            preflight: preflight
        )

        XCTAssertFalse(plan.isVision)
        XCTAssertFalse(plan.correctedVisionFromRequest)
        XCTAssertTrue(plan.forceLLMOnlyApplied)
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
