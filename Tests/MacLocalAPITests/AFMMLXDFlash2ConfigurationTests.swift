import XCTest
import AFMKit
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

        let result = try generator.generate(promptIDs: [1], maxTokens: 4)
        XCTAssertEqual(result.tokenIDs, [5, 6, 7, 8])
        XCTAssertEqual(result.statistics.emittedTokens, 4)
        XCTAssertGreaterThan(result.statistics.verificationCycles, 0)

        XCTAssertThrowsError(try generator.generate(
            promptIDs: [1], maxTokens: 4, shouldStop: { true })) {
            XCTAssertTrue($0 is CancellationError)
        }

        var partiallyEmitted = 0
        XCTAssertThrowsError(try generator.generate(
            promptIDs: [1],
            maxTokens: 4,
            onToken: { _ in
                partiallyEmitted += 1
                return partiallyEmitted < 2
            })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(partiallyEmitted, 2)

        let secondaryEOS = try generator.generate(
            promptIDs: [1], maxTokens: 4, stopTokenIDs: [7])
        XCTAssertEqual(secondaryEOS.tokenIDs, [5, 6])
        XCTAssertEqual(secondaryEOS.statistics.emittedTokens, 2)
    }

    private func tinyDraftModel() throws -> DFlash2DraftModel {
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-artifacts", isDirectory: true)
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
            "max_position_embeddings": 4_096,
            "layer_types": ["sliding_attention"],
            "sliding_window": 2_048,
            "rope_parameters": ["rope_theta": 1_000_000],
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
        XCTAssertFalse(disabled.permitsOtherRuntimes)
        XCTAssertFalse(disabled.requiresRuntime)

        service.dflash2Requirement = .required
        let explicitlyDisabled = try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "off"))
        XCTAssertFalse(explicitlyDisabled.permitsRuntime)
        XCTAssertFalse(explicitlyDisabled.permitsOtherRuntimes)
        XCTAssertFalse(explicitlyDisabled.requiresRuntime)

        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(
                mode: "dflash2", requirement: "required", drafter: "incoai/other")))

        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "dflash2", maxDraftTokens: 0)))
        XCTAssertNoThrow(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "dflash2", maxDraftTokens: 4)))
        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "dflash2", maxDraftTokens: 5)))

        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(requirement: "best-effort")))
        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "eagle3")))
        XCTAssertThrowsError(try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "off", requirement: "required")))
    }

    func testServiceSamplingCompatibilityPreservesNormalDefaultsAndPenalties() {
        let service = MLXModelService(resolver: MLXCacheResolver())

        XCTAssertEqual(
            service.speculativeRequestCompatibility(
                temperature: nil, topP: nil, repetitionPenalty: nil,
                topK: nil, minP: nil, presencePenalty: nil,
                hasTools: false, hasResponseFormat: false,
                wantsLogprobs: false, hasStopSequences: false).denialReason,
            "sampling")
        XCTAssertTrue(service.speculativeRequestCompatibility(
            temperature: 0, topP: 1, repetitionPenalty: 1,
            topK: 0, minP: 0, presencePenalty: 0,
            hasTools: false, hasResponseFormat: false,
            wantsLogprobs: false, hasStopSequences: false).isEligible)
        XCTAssertEqual(service.speculativeRequestCompatibility(
            temperature: 0, topP: 1, repetitionPenalty: 1.1,
            topK: 0, minP: 0, presencePenalty: 0,
            hasTools: false, hasResponseFormat: false,
            wantsLogprobs: false, hasStopSequences: false).denialReason, "penalties")
        XCTAssertEqual(service.speculativeRequestCompatibility(
            temperature: 0, topP: 1, repetitionPenalty: 1,
            topK: 0, minP: 0, presencePenalty: 0.4,
            hasTools: false, hasResponseFormat: false,
            wantsLogprobs: false, hasStopSequences: false).denialReason, "penalties")
        XCTAssertEqual(service.speculativeRequestCompatibility(
            temperature: 0, topP: nil, repetitionPenalty: nil,
            topK: 20, minP: nil, presencePenalty: nil,
            hasTools: false, hasResponseFormat: false,
            wantsLogprobs: false, hasStopSequences: false).denialReason, "sampling")
        XCTAssertEqual(service.speculativeRequestCompatibility(
            temperature: 0, topP: nil, repetitionPenalty: nil,
            topK: nil, minP: 0.05, presencePenalty: nil,
            hasTools: false, hasResponseFormat: false,
            wantsLogprobs: false, hasStopSequences: false).denialReason, "sampling")
        XCTAssertEqual(service.speculativeRequestCompatibility(
            temperature: 0, topP: nil, repetitionPenalty: nil,
            topK: nil, minP: nil, presencePenalty: 0.4,
            hasTools: false, hasResponseFormat: false,
            wantsLogprobs: false, hasStopSequences: false).denialReason, "penalties")
    }

    func testExplicitPreferredAndOffDoNotSelectOtherSpeculativeRuntimes() throws {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.dflash2Drafter = "incoai/loaded"

        let preferred = try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(requirement: "preferred"))
        XCTAssertTrue(preferred.permitsRuntime)
        XCTAssertFalse(preferred.permitsOtherRuntimes)

        let off = try service.dflash2RequestPolicy(
            SpeculativeDecodingOptions(mode: "off"))
        XCTAssertFalse(off.permitsRuntime)
        XCTAssertFalse(off.permitsOtherRuntimes)
        XCTAssertEqual(off.denialReason, "disabled")
    }

    func testServiceBuildsCompleteEOSSet() {
        let eos = MLXModelService.completeEOSTokenIDs(
            configuredTokenIDs: [10, 11],
            tokenizerTokenID: 12,
            extraTokens: ["<end>", "<missing>"],
            tokenID: { $0 == "<end>" ? 13 : nil })
        XCTAssertEqual(eos, [10, 11, 12, 13])
    }

    func testServiceInstancesHaveIndependentDFlashRuntimeScopes() {
        let first = MLXModelService(resolver: MLXCacheResolver())
        let second = MLXModelService(resolver: MLXCacheResolver())
        XCTAssertNotEqual(first.dflash2RuntimeScopeID, second.dflash2RuntimeScopeID)

        first.dflash2Drafter = "incoai/first"
        second.dflash2Drafter = "incoai/second"
        XCTAssertNoThrow(try first.dflash2RequestPolicy(
            SpeculativeDecodingOptions(drafter: "incoai/first")))
        XCTAssertNoThrow(try second.dflash2RequestPolicy(
            SpeculativeDecodingOptions(drafter: "incoai/second")))
        XCTAssertThrowsError(try first.dflash2RequestPolicy(
            SpeculativeDecodingOptions(drafter: "incoai/second")))
    }

    func testPrefixConcurrencyAndBatchFallbackReasonsAreStable() throws {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.dflash2Drafter = "incoai/loaded"
        service.enablePrefixCaching = true
        XCTAssertEqual(service.dflash2StartupFallbackReason(), "prefix_cache")
        service.maxConcurrent = 2
        XCTAssertEqual(service.dflash2StartupFallbackReason(), "concurrency")

        let forced = AFMMLXBatchSpeculativePolicy.forceAutoregressive(
            SpeculativeDecodingOptions(mode: "dflash2", requirement: "preferred"))
        let preferred = try service.dflash2RequestPolicy(forced)
        XCTAssertFalse(preferred.permitsRuntime)
        XCTAssertFalse(preferred.permitsOtherRuntimes)
        XCTAssertFalse(preferred.requiresRuntime)
        XCTAssertEqual(preferred.denialReason, "batch")

        let required = try service.dflash2RequestPolicy(
            AFMMLXBatchSpeculativePolicy.forceAutoregressive(
                SpeculativeDecodingOptions(mode: "dflash2", requirement: "required")))
        XCTAssertTrue(required.requiresRuntime)
        XCTAssertEqual(required.denialReason, "batch")
    }

    func testPromptOpenedReasoningBoundaryIsRestoredForNonStreamingOutput() {
        let restored = MLXModelService.restorePromptOpenedReasoningBoundary(
            generatedText: "private reasoning</think>visible answer",
            promptSuffix: "assistant\n<think>\n",
            startTag: "<think>",
            endTag: "</think>")
        var translator = MLXStreamEventTranslator(
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            maximumResponseTokens: nil)
        let events = translator.consume(StreamChunk(text: restored)) + translator.finish()
        let visible = events.compactMap { event -> String? in
            guard case .responseText(_, let delta, _) = event else { return nil }
            return delta
        }.joined()
        let reasoning = events.compactMap { event -> String? in
            guard case .reasoningText(_, let delta, _) = event else { return nil }
            return delta
        }.joined()
        XCTAssertEqual(reasoning, "private reasoning")
        XCTAssertEqual(visible, "visible answer")
    }

    func testSpeculativeCompletionPopulatesBaseRequestTelemetry() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        StatsAggregator.shared.reset()
        defer { StatsAggregator.shared.reset() }
        StatsAggregator.shared.requestStarted()
        service.recordSpeculativeRequestCompletion(
            queuedAt: Date(timeIntervalSince1970: 100),
            completedAt: Date(timeIntervalSince1970: 100.3),
            promptTokens: 12,
            completionTokens: 4,
            promptTime: 0.1,
            maxTokens: 10)

        let snapshot = StatsAggregator.shared.snapshot()
        XCTAssertEqual(snapshot.promptTokensTotal, 12)
        XCTAssertEqual(snapshot.genTokensTotal, 4)
        XCTAssertEqual(snapshot.requestsStartedTotal, 1)
        XCTAssertEqual(snapshot.requestsCompletedTotal, 1)
        XCTAssertEqual(snapshot.requestSuccessByReason["stop"], 1)
        XCTAssertEqual(snapshot.promptTokens.count, 1)
        XCTAssertEqual(snapshot.generationTokens.count, 1)
        XCTAssertEqual(snapshot.prefillTime.count, 1)
        XCTAssertEqual(snapshot.decodeTime.count, 1)
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
                "max_position_embeddings": 262_144,
                "eos_token_id": 248_044,
                "rope_parameters": ["rope_theta": 10_000_000],
            ],
        ])
        XCTAssertEqual(try config.effectiveBlockSize(requested: 5), 5)
        XCTAssertThrowsError(try config.effectiveBlockSize(requested: 20))
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
                "max_position_embeddings": 131_072,
                "bos_token_id": 200_000,
                "eos_token_id": 200_001,
                "sliding_window": 2_048,
                "rope_parameters": ["rope_theta": 500_000],
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
                "max_position_embeddings": 262_144,
                "eos_token_id": 248_044,
                "rope_parameters": ["rope_theta": 10_000_000],
            ],
        ]))
    }

    func testRejectsMismatchedTargetContextAndTokenizerContract() throws {
        let config = try AFMMLXDFlash2Configuration(metadata: draftMetadata(
            hidden: 5_120,
            targetLayers: 64,
            vocabulary: 248_320,
            block: 8,
            mask: 248_070,
            targetLayerIDs: [5, 19, 33, 47, 61]))
        let target: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 5_120,
                "num_hidden_layers": 64,
                "vocab_size": 248_320,
                "max_position_embeddings": 262_144,
                "eos_token_id": 7,
                "rope_parameters": ["rope_theta": 10_000_000],
            ],
        ]

        XCTAssertThrowsError(try config.validateTarget(metadata: target)) {
            XCTAssertTrue($0.localizedDescription.contains("eos_token_id"))
        }
    }

    func testSafetensorShapesValidateBeforeWeightLoad() throws {
        let config = try AFMMLXDFlash2Configuration(metadata: draftMetadata(
            hidden: 16,
            targetLayers: 2,
            vocabulary: 32,
            block: 4,
            mask: 31,
            targetLayerIDs: [1]))
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-artifacts", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var shapes = config.expectedTensorShapes()
        shapes["candidate_selector.predecessor_codebook"] =
            shapes.removeValue(forKey: "candidate_selector.predecessor_codebook.weight")
        shapes["candidate_selector.successor_codebook"] =
            shapes.removeValue(forKey: "candidate_selector.successor_codebook.weight")
        try writeSafetensorHeader(shapes: shapes, to: directory.appendingPathComponent("model.safetensors"))
        XCTAssertNoThrow(try config.validateWeights(in: directory))

        shapes["fc.weight"] = [15, 16]
        try writeSafetensorHeader(shapes: shapes, to: directory.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try config.validateWeights(in: directory)) {
            XCTAssertTrue($0.localizedDescription.contains("fc.weight"))
        }
    }

    func testSharedRuntimePreflightPreservesValidatedTargetContract() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-artifacts", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let target = root.appendingPathComponent("target", isDirectory: true)
        let draft = root.appendingPathComponent("draft", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: draft, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var draftObject = draftMetadata(
            hidden: 16,
            targetLayers: 2,
            vocabulary: 32,
            block: 4,
            mask: 31,
            targetLayerIDs: [1])
        draftObject["eos_token_id"] = 30
        draftObject["pad_token_id"] = 30
        try JSONSerialization.data(withJSONObject: draftObject, options: [.sortedKeys])
            .write(to: draft.appendingPathComponent("config.json"))
        let configuration = try AFMMLXDFlash2Configuration(metadata: draftObject)
        try writeSafetensorHeader(
            shapes: configuration.expectedTensorShapes(),
            to: draft.appendingPathComponent("model.safetensors"))

        let targetObject: [String: Any] = [
            "model_type": "qwen3_5",
            "generation_config": ["eos_token_id": [30]],
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "vocab_size": 32,
                "max_position_embeddings": 262_144,
                "eos_token_id": 30,
                "rope_parameters": ["rope_theta": 10_000_000],
            ],
        ]
        let targetData = try JSONSerialization.data(
            withJSONObject: targetObject, options: [.sortedKeys])
        try targetData.write(to: target.appendingPathComponent("config.json"))

        let preflight = try AFMMLXDFlash2PreflightValidator.validate(
            targetDirectory: target,
            drafterDirectory: draft,
            requestedBlockSize: 3)
        XCTAssertEqual(preflight.blockSize, 3)
        XCTAssertEqual(preflight.targetConfigurationData, targetData)
        XCTAssertEqual(preflight.configuration.hiddenSize, 16)
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

    func testAFMRequestDecodesNeutralSpeculativeControls() throws {
        let request = try AFMRequest(
            openAIMessages: [Message(role: "user", content: "hello")],
            generationConfig: GenerationConfig(metadata: [
                "speculativeDecoding": .object([
                    "mode": .string("dflash2"),
                    "requirement": .string("required"),
                    "drafter": .string("incoai/example"),
                    "maxDraftTokens": .integer(4),
                ]),
            ])
        )

        XCTAssertEqual(
            request.speculativeDecodingOptions(),
            SpeculativeDecodingOptions(
                mode: "dflash2",
                requirement: "required",
                drafter: "incoai/example",
                maxDraftTokens: 4
            )
        )
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
        XCTAssertEqual(
            AFMMLXSpeculativeTelemetry(metadataValue: telemetry.metadataValue),
            telemetry
        )

        StatsAggregator.shared.reset()
        StatsAggregator.shared.addSpeculative(
            strategy: telemetry.strategy,
            draftedTokens: telemetry.draftedTokens,
            acceptedDraftTokens: telemetry.acceptedDraftTokens,
            emittedTokens: telemetry.emittedTokens,
            verificationCycles: telemetry.verificationCycles,
            draftSeconds: telemetry.draftTime,
            verificationSeconds: telemetry.verificationTime,
            rollbackSeconds: telemetry.rollbackTime
        )
        let snapshot = StatsAggregator.shared.snapshot()
        XCTAssertEqual(snapshot.speculativeDraftedTokensTotal, 8)
        XCTAssertEqual(snapshot.speculativeAcceptedTokensTotal, 6)
        XCTAssertEqual(snapshot.speculativeEmittedTokensTotal, 8)
        XCTAssertEqual(snapshot.speculativeVerificationCyclesTotal, 4)
        StatsAggregator.shared.reset()

        let fallback = AFMMLXSpeculativeTelemetry.fallback(
            strategy: "dflash2", reason: "incompatible_request")
        XCTAssertEqual(fallback.fallbackReason, "incompatible_request")
        XCTAssertEqual(
            AFMMLXSpeculativeTelemetry(metadataValue: fallback.metadataValue),
            fallback
        )
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
            "num_hidden_layers": targetLayerIDs.count,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": vocabulary,
            "num_target_layers": targetLayers,
            "max_position_embeddings": targetLayers == 52 ? 131_072 : 262_144,
            "layer_types": Array(repeating: "sliding_attention", count: targetLayerIDs.count),
            "sliding_window": 2_048,
            "rope_parameters": [
                "rope_theta": targetLayers == 52 ? 500_000 : 10_000_000,
            ],
            "bos_token_id": targetLayers == 52 ? 200_000 : NSNull(),
            "eos_token_id": targetLayers == 52 ? 200_001 : 248_044,
            "pad_token_id": targetLayers == 52 ? 200_018 : 248_044,
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

    private func writeSafetensorHeader(shapes: [String: [Int]], to url: URL) throws {
        let object = shapes.mapValues { shape -> [String: Any] in
            ["dtype": "BF16", "shape": shape, "data_offsets": [0, 0]]
        }
        var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        while !header.count.isMultiple(of: 8) { header.append(0x20) }
        var size = UInt64(header.count).littleEndian
        var data = Data(bytes: &size, count: MemoryLayout<UInt64>.size)
        data.append(header)
        try data.write(to: url)
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
