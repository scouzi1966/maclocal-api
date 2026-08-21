import Foundation
import MLX
import MLXLMCommon
@testable import AFMKitMLX
@testable import MLXLLM
@testable import MLXVLM
import XCTest

final class DFlash2ProductionPathTests: XCTestCase {
    func testReleasedMuseAssistantFactoryLoadsCheckpointNames() throws {
        let directory = try makeArtifactDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let metadata = museAssistantMetadata(hidden: 8, targetLayerIDs: [1])
        try writeJSON(metadata, to: directory.appendingPathComponent("config.json"))
        let config = try DFlash2DraftConfiguration.load(
            directory: directory.path, targetLayers: 2, vocabularySize: 16)
        let source = DFlashDraftModel(config)
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        weights["encoder.fc.weight"] = weights.removeValue(forKey: "fc.weight")
        weights["encoder.output_norm_enc.weight"] = weights.removeValue(
            forKey: "hidden_norm.weight")
        try save(
            arrays: weights,
            url: directory.appendingPathComponent("model.safetensors"))

        let loaded = try DFlashDraftModelFactory.load(
            directory: directory.path, targetLayers: 2, vocabularySize: 16)
        XCTAssertTrue(loaded is DFlashDraftModel)
        let output = loaded.callAsFunction(
            noiseEmbedding: MLXArray.zeros([1, 2, 8]),
            targetHidden: MLXArray.zeros([1, 1, 8]))
        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 2, 8])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testProductionQwenFactoryTargetRunsDFlashGeneration() async throws {
        let target = try await makeQwenTarget()
        let draft = try makeDFlash2Draft(
            hidden: 16, targetLayers: 1, vocabulary: 32, targetLayerIDs: [0])
        let generator = try DFlash2Generator(target: target, draft: draft, blockSize: 2)

        let result = try generator.generate(promptIDs: [1, 2], maxTokens: 2)

        XCTAssertEqual(result.tokenIDs.count, 2)
        XCTAssertTrue(result.tokenIDs.allSatisfy { 0 ..< 32 ~= $0 })
        XCTAssertEqual(result.statistics.verificationCycles, 1)
    }

    func testProductionMuseFactoryTargetRunsDFlashGeneration() async throws {
        let target = try await makeMuseTarget()
        let draft = try makeDFlash2Draft(
            hidden: 8, targetLayers: 2, vocabulary: 16, targetLayerIDs: [1])
        let generator = try DFlash2Generator(target: target, draft: draft, blockSize: 2)

        let result = try generator.generate(promptIDs: [1, 2], maxTokens: 2)

        XCTAssertEqual(result.tokenIDs.count, 2)
        XCTAssertTrue(result.tokenIDs.allSatisfy { 0 ..< 16 ~= $0 })
        XCTAssertEqual(result.statistics.verificationCycles, 1)
    }

    func testRotatingCacheSnapshotRestoresWrappedStorageAndCircularIndex() {
        let cache = RotatingKVCache(maxSize: 2_048, keep: 0)
        for token in 0 ..< 2_055 {
            let value = MLXArray(Float(token)).reshaped(1, 1, 1, 1)
            _ = cache.update(keys: value, values: value + 10_000)
            if token.isMultiple(of: 256) {
                MLX.eval(cache.state)
            }
        }
        MLX.eval(cache.state)

        let expectedMetaState = cache.metaState
        let expectedState = cache.state.map { $0.asArray(Float.self) }
        let snapshot = DFlash2CacheSnapshot.capture([cache])

        for token in 0 ..< 17 {
            let value = MLXArray(Float(50_000 + token)).reshaped(1, 1, 1, 1)
            _ = cache.update(keys: value, values: value)
        }
        MLX.eval(cache.state)
        XCTAssertNotEqual(cache.metaState, expectedMetaState)

        DFlash2CacheSnapshot.restore(snapshot, into: [cache])
        MLX.eval(cache.state)
        XCTAssertEqual(cache.metaState, expectedMetaState)
        XCTAssertEqual(cache.state.map { $0.asArray(Float.self) }, expectedState)

        let next = MLXArray(Float(99_999)).reshaped(1, 1, 1, 1)
        _ = cache.update(keys: next, values: next)
        XCTAssertEqual(Int(cache.metaState[3]), 2_056)
        XCTAssertEqual(Int(cache.metaState[4]), 8)
    }

    func testCancellationAfterDraftDoesNotRestoreOrReplay() throws {
        let target = TrackingDFlashTarget()
        let draft = try ScriptedDFlashDraft(proposals: [6, 7, 8])
        let generator = try DFlash2Generator(target: target, draft: draft, blockSize: 4)
        var cancellationChecks = 0

        XCTAssertThrowsError(try generator.generate(
            promptIDs: [1],
            maxTokens: 4,
            shouldStop: {
                cancellationChecks += 1
                return cancellationChecks == 3
            })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(target.forwardLengths, [1])
        XCTAssertEqual(target.restoreCount, 0)
    }

    func testCancellationAfterVerificationDoesNotRestoreOrReplay() throws {
        let target = TrackingDFlashTarget()
        let draft = try ScriptedDFlashDraft(proposals: [6, 7, 8])
        let generator = try DFlash2Generator(target: target, draft: draft, blockSize: 4)
        var cancellationChecks = 0

        XCTAssertThrowsError(try generator.generate(
            promptIDs: [1],
            maxTokens: 4,
            shouldStop: {
                cancellationChecks += 1
                return cancellationChecks == 4
            })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(target.forwardLengths, [1, 4])
        XCTAssertEqual(target.restoreCount, 0)
    }

    func testUnknownTerminalStopsAcceptanceReplayAndTelemetryAtBoundary() throws {
        let target = TrackingDFlashTarget()
        let draft = try ScriptedDFlashDraft(proposals: [6, 7, 8])
        let generator = try DFlash2Generator(target: target, draft: draft, blockSize: 4)

        let result = try generator.generate(
            promptIDs: [1], maxTokens: 4, stopTokenIDs: [7])

        XCTAssertEqual(result.tokenIDs, [5, 6])
        XCTAssertEqual(result.statistics.acceptedDraftTokens, 2)
        XCTAssertEqual(result.statistics.emittedTokens, 2)
        XCTAssertEqual(target.forwardLengths, [1, 4, 3])
        XCTAssertEqual(target.restoreCount, 1)
    }

    func testBatchPromotionDrainsSerialOwnerAndBlocksNewSerialWork() async throws {
        let coordinator = MLXModelExecutionCoordinator()
        guard case .serial = await coordinator.acquireGeneration() else {
            return XCTFail("initial generation should own the serial model path")
        }

        let promotion = Task { await coordinator.beginPromotion() }
        try await waitUntil {
            await coordinator.snapshot().promotionInProgress
        }
        let blockedGeneration = Task { await coordinator.acquireGeneration() }
        try await waitUntil {
            await coordinator.snapshot().waitingGenerationAcquisitions == 1
        }
        var snapshot = await coordinator.snapshot()
        XCTAssertEqual(snapshot.activeSerialGenerations, 1)
        XCTAssertTrue(snapshot.promotionInProgress)

        await coordinator.releaseSerialGeneration()
        let ownsPromotion = await promotion.value
        XCTAssertTrue(ownsPromotion)
        await coordinator.finishPromotion(schedulerInstalled: true)
        guard case .scheduler = await blockedGeneration.value else {
            return XCTFail("generation should route through the installed scheduler")
        }
        snapshot = await coordinator.snapshot()
        XCTAssertEqual(snapshot.activeSerialGenerations, 0)
        XCTAssertTrue(snapshot.schedulerInstalled)
        await coordinator.releaseSchedulerGeneration()
    }

    func testSchedulerRemovalWaitsForUsersAndBlocksSerialOverlap() async throws {
        let coordinator = MLXModelExecutionCoordinator()
        let ownsPromotion = await coordinator.beginPromotion()
        XCTAssertTrue(ownsPromotion)
        await coordinator.finishPromotion(schedulerInstalled: true)
        guard case .scheduler = await coordinator.acquireGeneration() else {
            return XCTFail("generation should hold a scheduler-user lease")
        }

        let removal = Task { await coordinator.beginSchedulerRemoval() }
        try await waitUntil {
            await coordinator.snapshot().promotionInProgress
        }
        let blockedGeneration = Task { await coordinator.acquireGeneration() }
        try await waitUntil {
            await coordinator.snapshot().waitingGenerationAcquisitions == 1
        }

        var snapshot = await coordinator.snapshot()
        XCTAssertEqual(snapshot.activeSchedulerUsers, 1)
        XCTAssertTrue(snapshot.schedulerInstalled)
        XCTAssertTrue(snapshot.promotionInProgress)

        await coordinator.releaseSchedulerGeneration()
        let ownsRemoval = await removal.value
        XCTAssertTrue(ownsRemoval)
        snapshot = await coordinator.snapshot()
        XCTAssertEqual(snapshot.activeSchedulerUsers, 0)
        XCTAssertTrue(snapshot.schedulerInstalled)

        await coordinator.finishSchedulerRemoval()
        guard case .serial = await blockedGeneration.value else {
            return XCTFail("serial execution must begin only after scheduler removal finishes")
        }
        await coordinator.releaseSerialGeneration()
    }

    func testDFlash2FastPathDiagnosticsInitializeAndCleanupOnce() {
        var events: [String] = []
        var finishedMetrics: MLXGenerationDiagnosticsMetrics?
        let scope = MLXGenerationDiagnosticsScope {
            events.append("start")
            return { metrics in
                events.append("finish")
                finishedMetrics = metrics
            }
        }

        XCTAssertEqual(events, ["start"])
        XCTAssertFalse(scope.isFinished)

        let metrics = MLXGenerationDiagnosticsMetrics(
            promptTokens: 7,
            completionTokens: 3,
            promptTime: 0.25,
            generateTime: 0.5)
        scope.finish(metrics: metrics)
        scope.finish(metrics: .init())

        XCTAssertEqual(events, ["start", "finish"])
        XCTAssertEqual(finishedMetrics, metrics)
        XCTAssertTrue(scope.isFinished)
    }

    func testDFlash2FastPathDiagnosticsCleanupOnAbandonedScope() {
        var cleanupCount = 0
        var scope: MLXGenerationDiagnosticsScope? = MLXGenerationDiagnosticsScope {
            return { _ in cleanupCount += 1 }
        }

        XCTAssertFalse(scope!.isFinished)
        scope = nil

        XCTAssertEqual(cleanupCount, 1)
    }

    private func makeDFlash2Draft(
        hidden: Int,
        targetLayers: Int,
        vocabulary: Int,
        targetLayerIDs: [Int]
    ) throws -> DFlash2DraftModel {
        let directory = try makeArtifactDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata: [String: Any] = [
            "architectures": ["DFlash2DraftModel"],
            "is_causal": false,
            "hidden_size": hidden,
            "intermediate_size": hidden * 2,
            "num_hidden_layers": targetLayerIDs.count,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": hidden / 2,
            "vocab_size": vocabulary,
            "num_target_layers": targetLayers,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 4_096,
            "layer_types": Array(repeating: "sliding_attention", count: targetLayerIDs.count),
            "sliding_window": 2_048,
            "rope_parameters": ["rope_theta": 1_000_000],
            "dflash_config": [
                "target_layer_ids": targetLayerIDs,
                "block_size": 4,
                "mask_token_id": vocabulary - 1,
                "conv_kernel_size": 2,
                "conv_group_size": 4,
                "selector_rank": 4,
                "selector_top_k": 4,
            ],
        ]
        try writeJSON(metadata, to: directory.appendingPathComponent("config.json"))
        return DFlash2DraftModel(try DFlash2DraftConfiguration.load(directory: directory.path))
    }

    private func makeQwenTarget() async throws -> Qwen3_5MoEModel {
        let metadata: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "vocab_size": 32,
                "full_attention_interval": 1,
                "num_experts": 0,
                "rope_parameters": [
                    "rope_theta": 1_000_000,
                    "partial_rotary_factor": 1,
                ],
            ],
        ]
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: JSONSerialization.data(withJSONObject: metadata),
            modelType: "qwen3_5")
        guard let target = model as? Qwen3_5MoEModel else {
            throw NSError(
                domain: "DFlash2ProductionPathTests",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Qwen registry returned \(type(of: model))"])
        }
        return target
    }

    private func makeMuseTarget() async throws -> MuseGlimmer {
        let metadata: [String: Any] = [
            "model_type": "muse_glimmer",
            "text_config": [
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "layer_types": ["sliding_attention", "full_attention"],
                "layer_rope_theta": [500_000.0, 0.0],
                "sliding_window": 4,
            ],
            "vision_config": [
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "patch_size": 2,
                "patch_temporal": 1,
                "merge_size": 1,
                "pos_emb_height": 2,
                "pos_emb_width": 2,
                "layer_types": ["full_attention"],
            ],
            "image_token_id": 14,
            "video_token_id": 15,
            "out_hidden_size": 8,
            "projector_hidden_size": 8,
        ]
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: JSONSerialization.data(withJSONObject: metadata),
            modelType: "muse_glimmer")
        guard let target = model as? MuseGlimmer else {
            throw NSError(
                domain: "DFlash2ProductionPathTests",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Muse registry returned \(type(of: model))"])
        }
        return target
    }

    private func museAssistantMetadata(
        hidden: Int,
        targetLayerIDs: [Int]
    ) -> [String: Any] {
        [
            "architectures": ["MuseGlimmerAssistantModel"],
            "block_size": 4,
            "head_dim": hidden / 2,
            "hidden_size": hidden,
            "intermediate_size": hidden * 2,
            "layer_types": Array(repeating: "sliding_attention", count: targetLayerIDs.count),
            "mask_token_id": 15,
            "max_position_embeddings": 4_096,
            "model_type": "muse_glimmer_assistant",
            "num_attention_heads": 2,
            "num_hidden_layers": targetLayerIDs.count,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.00001,
            "rope_parameters": ["rope_theta": 500_000.0],
            "sliding_window": 2_048,
            "target_layer_ids": targetLayerIDs,
        ]
    }

    private func makeArtifactDirectory() throws -> URL {
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-artifacts", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func writeJSON(_ object: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    private func waitUntil(
        timeoutNanoseconds: UInt64 = 1_000_000_000,
        condition: @escaping () async -> Bool
    ) async throws {
        let deadline = DispatchTime.now().uptimeNanoseconds + timeoutNanoseconds
        while !(await condition()) {
            if DispatchTime.now().uptimeNanoseconds >= deadline {
                XCTFail("condition did not become true before timeout")
                return
            }
            try await Task.sleep(nanoseconds: 1_000_000)
        }
    }
}

private final class ScriptedDFlashDraft: DFlashDraftingModel {
    let config: DFlash2DraftConfiguration
    private let proposals: [Int]

    init(proposals: [Int]) throws {
        self.proposals = proposals
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-artifacts", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata: [String: Any] = [
            "architectures": ["MuseGlimmerAssistantModel"],
            "block_size": 4,
            "head_dim": 8,
            "hidden_size": 16,
            "intermediate_size": 32,
            "layer_types": ["sliding_attention"],
            "mask_token_id": 31,
            "max_position_embeddings": 4_096,
            "model_type": "muse_glimmer_assistant",
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "rms_norm_eps": 0.00001,
            "rope_parameters": ["rope_theta": 500_000.0],
            "sliding_window": 2_048,
            "target_layer_ids": [1],
        ]
        try JSONSerialization.data(withJSONObject: metadata).write(
            to: directory.appendingPathComponent("config.json"))
        self.config = try DFlash2DraftConfiguration.load(
            directory: directory.path, targetLayers: 2, vocabularySize: 32)
    }

    func callAsFunction(noiseEmbedding: MLXArray, targetHidden: MLXArray) -> MLXArray {
        MLXArray.zeros(noiseEmbedding.shape)
    }

    func select(hidden: MLXArray, logits: MLXArray, anchor: Int) -> [Int] {
        Array(proposals.prefix(hidden.dim(1)))
    }
}

private final class TrackingDFlashTarget: DFlash2Target {
    let dflash2HiddenSize = 16
    let dflash2LayerCount = 2
    let dflash2VocabularySize = 32
    private var position = 0
    private(set) var forwardLengths: [Int] = []
    private(set) var restoreCount = 0

    func dflash2NewCache() -> [any KVCache] {
        position = 0
        forwardLengths = []
        restoreCount = 0
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
        forwardLengths.append(length)
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
        restoreCount += 1
        position = snapshot as? Int ?? position
    }
}
