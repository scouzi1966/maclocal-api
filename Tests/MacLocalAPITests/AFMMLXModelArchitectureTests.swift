import Foundation
import XCTest
import MLXVLM
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
        XCTAssertTrue(AFMMLXModelArchitecture.isSupported("muse_glimmer"))
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
        XCTAssertTrue(AFMMLXModelArchitecture.isVisionModelType("muse_glimmer"))
        XCTAssertFalse(AFMMLXModelArchitecture.isLanguageModelType("muse_glimmer"))
        XCTAssertFalse(AFMMLXModelArchitecture.isDualModeModelType("muse_glimmer"))
        XCTAssertFalse(AFMMLXModelArchitecture.isDualModeModelType("qwen3_vl"))
    }

    func testMuseGlimmerConfigurationUsesVisionFactory() throws {
        let config: [String: Any] = [
            "model_type": "muse_glimmer",
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "image_token_id": 200092,
            "text_config": [
                "model_type": "muse_glimmer_text",
                "num_attention_heads": 32,
            ],
            "vision_config": ["model_type": "muse_glimmer_vision"],
        ]
        XCTAssertFalse(AFMMLXModelArchitecture.isDualModeConfiguration(config))
        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            config, modelID: "mlx-community/Muse-Glimmer-30B-4bit")
        XCTAssertTrue(preflight.isVisionConfiguration)
        XCTAssertTrue(preflight.requiresVisionModelFactory)
    }

    func testMuseGlimmerPublishedConfigDecodesArchitectureFieldsWithoutModelIDHeuristics() throws {
        let configData = try JSONSerialization.data(withJSONObject: [
            "model_type": "muse_glimmer",
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "image_token_id": 200092,
            "video_token_id": 200091,
            "text_config": [
                "model_type": "muse_glimmer_text",
                "hidden_size": 6656,
                "num_hidden_layers": 52,
                "num_attention_heads": 32,
                "layer_types": ["sliding_attention", "full_attention"],
                "layer_rope_theta": [500000.0, 0],
                "qk_scale_factor": 3.87,
            ],
            "vision_config": [
                "model_type": "muse_glimmer_vision",
                "hidden_size": 1536,
                "num_hidden_layers": 50,
            ],
        ])
        let decoded = try JSONSerialization.jsonObject(with: configData) as! [String: Any]

        XCTAssertEqual(decoded["model_type"] as? String, "muse_glimmer")
        XCTAssertEqual((decoded["text_config"] as? [String: Any])?["hidden_size"] as? Int, 6656)
        XCTAssertEqual((decoded["vision_config"] as? [String: Any])?["num_hidden_layers"] as? Int, 50)

        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            decoded, modelID: "arbitrary/local-folder")
        XCTAssertEqual(preflight.canonicalModelType, "muse_glimmer")
        XCTAssertTrue(preflight.isVisionConfiguration)
        XCTAssertTrue(preflight.requiresVisionModelFactory)
        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(forceVLM: false, architecture: preflight),
            .vlm)
    }

    func testMuseGlimmerFactorySelectionIsArchitectureDriven() throws {
        let config: [String: Any] = [
            "model_type": "muse_glimmer",
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "text_config": ["hidden_size": 6656],
            "vision_config": ["hidden_size": 1536],
        ]
        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            config, modelID: "some-owner/not-muse-named-directory")
        let plan = AFMMLXRemoteModelLoadPolicy.plan(
            repoID: "some-owner/not-muse-named-directory",
            requestedIsVision: false,
            forceLLMOnly: false,
            preflight: preflight)

        XCTAssertTrue(plan.isVision)
        XCTAssertTrue(plan.correctedVisionFromRequest)
        XCTAssertTrue(plan.forceLLMOnlyApplied == false)
    }

    func testUnsafeHybridArchitecturesForceSerialGeneration() {
        XCTAssertTrue(MLXModelService.requiresSerialGeneration(canonicalModelType: "cohere2_moe"))
        XCTAssertTrue(MLXModelService.requiresSerialGeneration(canonicalModelType: "muse_glimmer"))
        XCTAssertFalse(MLXModelService.requiresSerialGeneration(canonicalModelType: "qwen3"))
        XCTAssertFalse(MLXModelService.requiresSerialGeneration(canonicalModelType: "nemotron_h"))
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

    func testQwen38PublishedConfigurationUsesExistingQwen35MultimodalArchitecture() throws {
        let config: [String: Any] = [
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": 248056,
            "text_config": [
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "full_attention_interval": 4,
                "mtp_num_hidden_layers": 1,
            ],
            "vision_config": [
                "model_type": "qwen3_5",
                "depth": 27,
                "hidden_size": 1152,
            ],
            "quantization": [
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
            ],
        ]

        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            config,
            modelID: "mlx-community/Qwen3.8-27B-mxfp8"
        )

        XCTAssertEqual(preflight.modelType, "qwen3_5")
        XCTAssertEqual(preflight.canonicalModelType, "qwen3_5")
        XCTAssertTrue(preflight.isVisionConfiguration)
        XCTAssertTrue(preflight.requiresVisionModelFactory)
        XCTAssertTrue(AFMMLXModelArchitecture.isDualModeConfiguration(config))
    }

    func testQwen38PublishedConfigurationDecodesWithQwen35VLMFactoryConfiguration() throws {
        let config: [String: Any] = [
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "intermediate_size": 17408,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_num_value_heads": 48,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 248320,
                "full_attention_interval": 4,
                "max_position_embeddings": 262144,
                "rope_parameters": [
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10_000_000,
                ],
            ],
            "vision_config": [
                "model_type": "qwen3_5",
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 5120,
                "num_heads": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
            ],
        ]

        let data = try JSONSerialization.data(withJSONObject: config)
        XCTAssertNoThrow(try JSONDecoder().decode(Qwen3_5MoEVLConfiguration.self, from: data))
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

    func testPreflightRejectsDFlashDraftCheckpointBeforeQwenFactorySelection() {
        let config: [String: Any] = [
            "architectures": ["DFlashDraftModel"],
            "auto_map": ["AutoModel": "dflash.DFlashDraftModel"],
            "dflash_config": ["block_size": 16],
            "model_type": "qwen3",
            "num_hidden_layers": 6,
        ]

        XCTAssertEqual(
            AFMMLXModelArchitecture.draftOnlyArchitecture(in: config),
            "DFlashDraftModel"
        )
        XCTAssertThrowsError(try AFMMLXModelArchitecture.preflightConfiguration(
            config,
            modelID: "z-lab/Qwen3.6-35B-A3B-DFlash"
        )) { error in
            guard case AFMMLXModelArchitecturePreflightError.draftOnlyArchitecture(
                let architecture,
                let modelID
            ) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(architecture, "DFlashDraftModel")
            XCTAssertEqual(modelID, "z-lab/Qwen3.6-35B-A3B-DFlash")
            XCTAssertTrue(error.localizedDescription.contains("not a standalone language model"))
        }
    }

    func testPreflightRejectsDFlashDraftCheckpointFromAutoMap() {
        let config: [String: Any] = [
            "architectures": [],
            "auto_map": ["AutoModel": "dflash.DFlashDraftModel"],
            "model_type": "qwen3",
        ]

        XCTAssertEqual(
            AFMMLXModelArchitecture.draftOnlyArchitecture(in: config),
            "dflash.DFlashDraftModel"
        )
        XCTAssertThrowsError(try AFMMLXModelArchitecture.preflightConfiguration(
            config,
            modelID: "local/dflash"
        ))
    }

    func testRepositoryNameHeuristics() {
        XCTAssertTrue(AFMMLXModelArchitecture.matchesSupportedNamePattern("mlx-community/Qwen3-4B-4bit"))
        XCTAssertTrue(AFMMLXModelArchitecture.matchesSupportedNamePattern("mlx-community/FastVLM-1.5B"))
        XCTAssertFalse(AFMMLXModelArchitecture.matchesSupportedNamePattern("example/random-model"))

        XCTAssertTrue(AFMMLXModelArchitecture.looksLikeDualMode("mlx-community/Qwen3.5-35B-A3B-4bit"))
        XCTAssertTrue(AFMMLXModelArchitecture.looksLikeDualMode("Qwen/Qwen3.8-27B-FP8"))
        XCTAssertFalse(AFMMLXModelArchitecture.looksLikeDualMode("mlx-community/Qwen3-VL-4B-Instruct"))

        XCTAssertTrue(AFMMLXModelArchitecture.looksLikeVisionModel("mlx-community/paligemma-3b"))
        XCTAssertFalse(AFMMLXModelArchitecture.looksLikeVisionModel("mlx-community/Llama-3.2-1B-Instruct"))
    }
}
