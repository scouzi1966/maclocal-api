import XCTest
import AFMKitCore
import MLX
import MLXLMCommon
@testable import AFMKitMLX
@testable import AFMOpenAICompat

final class AFMMLXDFlash2ConfigurationTests: XCTestCase {
    func testSelectorParametersUseCheckpointCompatibleNestedKeys() throws {
        let model = try tinyDraftModel()
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("candidate_selector.hidden_projection.weight"))
        XCTAssertTrue(keys.contains("candidate_selector.predecessor_codebook.weight"))
        XCTAssertTrue(keys.contains("candidate_selector.successor_codebook.weight"))
    }

    func testGreedyGenerationRemainsTargetEquivalentAndCancellable() throws {
        let draft = try tinyDraftModel()
        let target = DeterministicDFlash2Target()
        let generator = try DFlash2Generator(target: target, draft: draft, blockSize: 4)

        let result = generator.generate(promptIDs: [1], maxTokens: 4)
        XCTAssertEqual(result.tokenIDs, [5, 6, 7, 8])
        XCTAssertEqual(result.statistics.emittedTokens, 4)
        XCTAssertGreaterThan(result.statistics.verificationCycles, 0)

        let cancelled = generator.generate(
            promptIDs: [1], maxTokens: 4, shouldStop: { true })
        XCTAssertEqual(cancelled.tokenIDs, [])
    }

    private func tinyDraftModel() throws -> DFlash2DraftModel {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata: [String: Any] = [
            "architectures": ["DFlash2DraftModel"],
            "is_causal": false,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 32,
            "num_target_layers": 2,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1_000_000,
            "dflash_config": [
                "target_layer_ids": [1],
                "block_size": 4,
                "mask_token_id": 31,
                "conv_kernel_size": 2,
                "conv_group_size": 8,
                "selector_rank": 4,
                "selector_top_k": 4,
            ],
        ]
        try JSONSerialization.data(withJSONObject: metadata)
            .write(to: directory.appendingPathComponent("config.json"))

        let config = try DFlash2DraftConfiguration.load(directory: directory.path)
        return DFlash2DraftModel(config)
    }

    func testProviderNeutralStartupConfigurationMapsToMLXRuntime() {
        let configuration = AFMProviderConfiguration(values: [
            "speculativeDecoding": .object([
                "mode": .string("dflash2"),
                "drafter": .string("incoai/example"),
                "maxDraftTokens": .integer(4),
                "requirement": .string("required"),
            ]),
        ])

        let runtime = AFMMLXRuntimeConfiguration(providerConfiguration: configuration)
        XCTAssertEqual(runtime.dflash2Drafter, "incoai/example")
        XCTAssertEqual(runtime.dflash2BlockSize, 5)
        XCTAssertEqual(runtime.dflash2Requirement, .required)
    }

    func testRequestPolicyFallsBackOrRequiresBeforeEmission() throws {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.dflash2Drafter = "incoai/loaded"

        let disabled = try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "off"))
        XCTAssertFalse(disabled.permitsRuntime)
        XCTAssertFalse(disabled.requiresRuntime)

        let mismatch = try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(
                mode: "dflash2", requirement: "required", drafter: "incoai/other"))
        XCTAssertFalse(mismatch.permitsRuntime)
        XCTAssertTrue(mismatch.requiresRuntime)
        XCTAssertNotNil(mismatch.denialReason)

        let invalidCount = try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "dflash2", maxDraftTokens: 0))
        XCTAssertFalse(invalidCount.permitsRuntime)
        XCTAssertEqual(invalidCount.denialReason, "max_draft_tokens must be at least 1")

        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(requirement: "best-effort")))
    }

    func testQwenReleasedContractValidatesByMetadata() throws {
        let config = try AFMMLXDFlash2Configuration(metadata: draftMetadata(
            hidden: 5_120,
            targetLayers: 64,
            vocabulary: 248_320,
            block: 8,
            mask: 248_070,
            targetLayerIDs: [5, 19, 33, 47, 61]))

        try config.validateTarget(metadata: [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 5_120,
                "num_hidden_layers": 64,
                "vocab_size": 248_320,
            ],
        ])
        XCTAssertEqual(try config.effectiveBlockSize(requested: 5), 5)
        XCTAssertEqual(try config.effectiveBlockSize(requested: 20), 8)
    }

    func testMuseReleasedContractValidatesByMetadata() throws {
        let config = try AFMMLXDFlash2Configuration(metadata: draftMetadata(
            hidden: 6_656,
            targetLayers: 52,
            vocabulary: 202_048,
            block: 16,
            mask: 201_818,
            targetLayerIDs: [1, 13, 25, 37, 49]))

        try config.validateTarget(metadata: [
            "model_type": "muse_glimmer",
            "text_config": [
                "model_type": "muse_glimmer_text",
                "hidden_size": 6_656,
                "num_hidden_layers": 52,
                "vocab_size": 202_048,
            ],
        ])
    }

    func testRepositoryNameCannotMakeLegacyDFlashLookLikeDFlash2() {
        var metadata = draftMetadata(
            hidden: 5_120,
            targetLayers: 64,
            vocabulary: 248_320,
            block: 8,
            mask: 248_070,
            targetLayerIDs: [5, 19, 33, 47, 61])
        metadata["architectures"] = ["DFlashDraftModel"]

        XCTAssertThrowsError(try AFMMLXDFlash2Configuration(metadata: metadata)) {
            XCTAssertEqual(
                $0 as? AFMMLXDFlash2ConfigurationError,
                .unsupportedArchitecture(["DFlashDraftModel"]))
        }
    }

    func testRejectsNonTwoTapConvolutionMetadata() {
        var metadata = draftMetadata(
            hidden: 5_120,
            targetLayers: 64,
            vocabulary: 248_320,
            block: 8,
            mask: 248_070,
            targetLayerIDs: [5, 19, 33, 47, 61])
        var dflash = metadata["dflash_config"] as? [String: Any] ?? [:]
        dflash["conv_kernel_size"] = 3
        metadata["dflash_config"] = dflash

        XCTAssertThrowsError(try AFMMLXDFlash2Configuration(metadata: metadata))
    }

    func testTargetShapeMismatchFailsBeforeWeightsLoad() throws {
        let config = try AFMMLXDFlash2Configuration(metadata: draftMetadata(
            hidden: 5_120,
            targetLayers: 64,
            vocabulary: 248_320,
            block: 8,
            mask: 248_070,
            targetLayerIDs: [5, 19, 33, 47, 61]))

        XCTAssertThrowsError(try config.validateTarget(metadata: [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 5_120,
                "num_hidden_layers": 64,
                "vocab_size": 202_048,
            ],
        ]))
    }

    func testOpenAIRequestDecodesNeutralSpeculativeControls() throws {
        let data = Data(#"""
        {
          "model":"target",
          "messages":[{"role":"user","content":"hello"}],
          "speculative_decoding":{
            "mode":"dflash2",
            "requirement":"required",
            "drafter":"incoai/example",
            "max_draft_tokens":4
          }
        }
        """#.utf8)
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
        XCTAssertEqual(request.speculativeDecoding?.mode, "dflash2")
        XCTAssertEqual(request.speculativeDecoding?.requirement, "required")
        XCTAssertEqual(request.speculativeDecoding?.drafter, "incoai/example")
        XCTAssertEqual(request.speculativeDecoding?.maxDraftTokens, 4)
    }

    func testDFlash2FallsBackForStringStopsAndSampling() {
        let withStops = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .dflash2,
            installedRuntime: .dflash2,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: true)
        XCTAssertEqual(withStops.path, .fallback)
        XCTAssertEqual(withStops.reason, .stopSequences)

        let sampled = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .dflash2,
            installedRuntime: .dflash2,
            temperature: 0.7,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false)
        XCTAssertEqual(sampled.path, .fallback)
        XCTAssertEqual(sampled.reason, .samplingEnabled)
    }

    func testNeutralTelemetryDerivesMeanAcceptanceLength() {
        let telemetry = AFMMLXSpeculativeTelemetry(
            strategy: "dflash2",
            draftedTokens: 8,
            acceptedDraftTokens: 6,
            emittedTokens: 8,
            verificationCycles: 4,
            draftTime: 0.1,
            verificationTime: 0.2,
            rollbackTime: 0.03)
        XCTAssertEqual(telemetry.meanAcceptanceLength, 1.5)
    }

    private func draftMetadata(
        hidden: Int,
        targetLayers: Int,
        vocabulary: Int,
        block: Int,
        mask: Int,
        targetLayerIDs: [Int]
    ) -> [String: Any] {
        [
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "is_causal": false,
            "hidden_size": hidden,
            "intermediate_size": hidden * 3,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": vocabulary,
            "num_target_layers": targetLayers,
            "dflash_config": [
                "target_layer_ids": targetLayerIDs,
                "block_size": block,
                "mask_token_id": mask,
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 256,
                "selector_top_k": 16,
            ],
        ]
    }
}

private final class DeterministicDFlash2Target: DFlash2Target {
    var position = 0
    let dflash2HiddenSize = 16
    let dflash2LayerCount = 2
    let dflash2VocabularySize = 32

    func dflash2NewCache() -> [any KVCache] {
        position = 0
        return []
    }

    func dflash2Embed(_ tokenIDs: MLXArray) -> MLXArray {
        MLXArray.zeros([1, tokenIDs.dim(1), dflash2HiddenSize])
    }

    func dflash2Project(_ hidden: MLXArray) -> MLXArray {
        MLXArray.zeros([1, hidden.dim(1), dflash2VocabularySize])
    }

    func dflash2Forward(
        _ tokenIDs: MLXArray,
        captureLayerIDs: [Int],
        cache: [any KVCache]
    ) -> DFlash2TargetOutput {
        let length = tokenIDs.dim(1)
        let logits = MLXArray.zeros([1, length, dflash2VocabularySize])
        for index in 0 ..< length {
            logits[0, index, 5 + position + index] = MLXArray(Float(100))
        }
        position += length
        return DFlash2TargetOutput(
            hidden: MLXArray.zeros([1, length, dflash2HiddenSize * captureLayerIDs.count]),
            logits: logits)
    }

    func dflash2CaptureCache(_ cache: [any KVCache]) -> Any { position }

    func dflash2RestoreCache(_ snapshot: Any, into cache: [any KVCache]) {
        position = snapshot as? Int ?? position
    }
}
