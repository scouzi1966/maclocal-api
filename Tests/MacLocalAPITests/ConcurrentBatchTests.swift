import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

@testable import AFMKit
@testable import AFMKitMLX
@testable import AFMServer

/// Unit tests for Phase 1 concurrent batching internals:
/// RequestSlot, StreamChunk, and BatchScheduler queuing logic.
struct ConcurrentBatchTests {
// dimensions: execution=batch

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - RequestSlot
    // ═══════════════════════════════════════════════════════════════════

    @Test("RequestSlot initializes with unique ID and zero content")
    func slotInitialization() {
        let slot = RequestSlot(promptTokens: 100)
        #expect(slot.promptTokens == 100)
        #expect(slot.fullContent == "")
        #expect(slot.elapsedTime >= 0)
    }

    @Test("RequestSlot IDs are unique across instances")
    func slotUniqueIDs() {
        let a = RequestSlot(promptTokens: 10)
        let b = RequestSlot(promptTokens: 20)
        #expect(a.id != b.id)
    }

    @Test("RequestSlot appendContent accumulates text")
    func slotAppendContent() {
        let slot = RequestSlot(promptTokens: 50)
        slot.appendContent("Hello")
        slot.appendContent(" ")
        slot.appendContent("world")
        #expect(slot.fullContent == "Hello world")
    }

    @Test("RequestSlot is thread-safe under concurrent writes")
    func slotConcurrentAppend() async {
        let slot = RequestSlot(promptTokens: 0)
        let iterations = 1000

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<iterations {
                group.addTask {
                    slot.appendContent("\(i),")
                }
            }
        }

        // All 1000 writes should be present (order may vary)
        let parts = slot.fullContent.split(separator: ",").compactMap { Int($0) }
        #expect(parts.count == iterations)
        // Every number 0..<1000 should appear exactly once
        let unique = Set(parts)
        #expect(unique.count == iterations)
    }

    @Test("RequestSlot elapsedTime increases over time")
    func slotElapsedTime() async throws {
        let slot = RequestSlot(promptTokens: 0)
        let t0 = slot.elapsedTime
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        let t1 = slot.elapsedTime
        #expect(t1 > t0)
        #expect(t1 >= 0.04) // at least ~40ms (allowing some slack)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - StreamChunk
    // ═══════════════════════════════════════════════════════════════════

    @Test("StreamChunk defaults: only text, everything else nil")
    func streamChunkDefaults() {
        let chunk = StreamChunk(text: "hello")
        #expect(chunk.text == "hello")
        #expect(chunk.logprobs == nil)
        #expect(chunk.toolCalls == nil)
        #expect(chunk.promptTokens == nil)
        #expect(chunk.completionTokens == nil)
        #expect(chunk.cachedTokens == nil)
        #expect(chunk.promptTime == nil)
        #expect(chunk.generateTime == nil)
    }

    @Test("StreamChunk carries timing info")
    func streamChunkWithInfo() {
        let chunk = StreamChunk(
            text: "",
            promptTokens: 100,
            completionTokens: 50,
            promptTime: 1.5,
            generateTime: 3.0
        )
        #expect(chunk.promptTokens == 100)
        #expect(chunk.completionTokens == 50)
        #expect(chunk.promptTime == 1.5)
        #expect(chunk.generateTime == 3.0)
    }

    @Test("StreamChunk carries cached token count")
    func streamChunkCachedTokens() {
        let chunk = StreamChunk(text: "", cachedTokens: 512)
        #expect(chunk.cachedTokens == 512)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MLXServiceError
    // ═══════════════════════════════════════════════════════════════════

    @Test("MLXServiceError.serviceShuttingDown has descriptive message")
    func serviceShuttingDownError() {
        let error = MLXServiceError.serviceShuttingDown
        #expect(error.localizedDescription.contains("shutting down"))
    }

    @Test("MLXServiceError.noModelLoaded has descriptive message")
    func noModelLoadedError() {
        let error = MLXServiceError.noModelLoaded
        #expect(error.localizedDescription.contains("No MLX model"))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - BatchScheduler constants
    // ═══════════════════════════════════════════════════════════════════

    @Test("BatchScheduler default max concurrent is 8")
    func defaultMaxConcurrent() {
        #expect(BatchScheduler.defaultMaxConcurrent == 8)
    }

    @Test("BatchScheduler admission window is enabled for burst fairness")
    func defaultAdmissionWindowEnabled() {
        #expect(BatchScheduler.defaultAdmissionWindowNanoseconds > 0)
        #expect(BatchScheduler.defaultAdmissionWindowNanoseconds <= 20_000_000)
    }

    @Test("Scheduler submission admission keeps reserved and unreserved capacity separate")
    func schedulerSubmissionAdmissionIsExplicit() {
        let admission = BatchSchedulerAdmissionState(maxConcurrent: 1)

        #expect(admission.reserveForUnreservedSubmission())
        #expect(!admission.reserveForUnreservedSubmission())
        #expect(admission.snapshot == .init(inFlightCount: 1, reservedCount: 0))

        admission.finish()
        let reservation = admission.tryReserve()
        #expect(reservation != nil)
        #expect(admission.snapshot == .init(inFlightCount: 1, reservedCount: 1))
        #expect(!admission.reserveForUnreservedSubmission())
        #expect(admission.consumeReservationForSubmission(reservation!))
        #expect(admission.snapshot == .init(inFlightCount: 1, reservedCount: 0))
    }

    @Test("Gemma 4 defers staggered arrivals until the active cohort drains")
    func gemma4DefersStaggeredAdmissions() {
        #expect(BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: true, activeSlotCount: 1))
        #expect(BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: true, activeSlotCount: 8))
        #expect(!BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: true, activeSlotCount: 0))
    }

    @Test("Other models continue admitting requests into active batches")
    func nonGemmaModelsAdmitStaggeredRequests() {
        #expect(!BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: false, activeSlotCount: 4))
        #expect(!BatchScheduler.shouldDeferStaggeredAdmissions(
            requiresFixedDecodeCohorts: false, activeSlotCount: 0))
    }

    @Test("Gemma text and vision models advertise fixed decode cohorts")
    func gemmaVariantsAdvertiseFixedDecodeCohorts() {
        #expect(BatchScheduler.requiresFixedDecodeCohorts(for: Gemma4Model.self))
        #expect(BatchScheduler.requiresFixedDecodeCohorts(for: Gemma4VLM.self))
        #expect(!BatchScheduler.requiresFixedDecodeCohorts(for: Qwen3Model.self))
    }

    @Test("BatchScheduler supports DeepSeek hybrid cache through its batch container")
    func supportsDeepseekHybridCacheBatching() {
        let cache = DeepseekV4Cache(slidingWindow: 128, compressRatio: 4)
        #expect(BatchScheduler.supportsDenseBatchMerge(cache))
    }

    @Test("DeepSeek batch cache preserves independent hybrid state")
    func deepseekBatchCachePreservesIndependentState() {
        func makeCache(offset: Int, poolRows: Int, bufferRows: Int) -> DeepseekV4Cache {
            let cache = DeepseekV4Cache(
                slidingWindow: 128,
                compressRatio: 4,
                poolQuantizationEnabled: false)
            if offset > 0 {
                _ = cache.update(
                    keys: MLXArray.zeros([1, 2, offset, 8], dtype: .float16),
                    values: MLXArray.zeros([1, 2, offset, 8], dtype: .float16))
            }
            cache.setPooled(
                .compressor,
                value: MLXArray.zeros([1, poolRows, 16], dtype: .float16))
            cache.setPooled(
                .indexer,
                value: MLXArray.zeros([1, poolRows + 1, 8], dtype: .float16))
            cache.setBuffers(
                .compressor,
                kv: MLXArray.zeros([1, bufferRows, 16], dtype: .float16),
                gate: MLXArray.zeros([1, bufferRows, 2], dtype: .float16))
            cache.setBuffers(
                .indexer,
                kv: MLXArray.zeros([1, bufferRows + 1, 8], dtype: .float16),
                gate: MLXArray.zeros([1, bufferRows + 1, 2], dtype: .float16))
            return cache
        }

        let first = makeCache(offset: 3, poolRows: 2, bufferRows: 1)
        let second = makeCache(offset: 7, poolRows: 5, bufferRows: 3)
        let third = makeCache(offset: 11, poolRows: 7, bufferRows: 2)
        let batch = BatchDeepseekV4Cache.merge([first, second])

        #expect(batch.count == 2)
        #expect(batch.offset == 7)
        #expect(batch.cache(at: 0).offset == 3)
        #expect(batch.cache(at: 1).offset == 7)
        #expect(batch.cache(at: 0).getPooled(.compressor)?.dim(1) == 2)
        #expect(batch.cache(at: 1).getPooled(.compressor)?.dim(1) == 5)
        #expect(batch.cache(at: 0).getBuffers(.compressor).kv?.dim(1) == 1)
        #expect(batch.cache(at: 1).getBuffers(.compressor).kv?.dim(1) == 3)

        batch.extend(with: third)
        #expect(batch.count == 3)
        #expect(batch.offset == 11)

        let extracted = batch.extract(1)
        #expect(extracted.offset == 7)
        #expect(extracted.getPooled(.indexer)?.dim(1) == 6)
        #expect(extracted.getBuffers(.indexer).kv?.dim(1) == 4)

        batch.filter([2, 0])
        #expect(batch.count == 2)
        #expect(batch.cache(at: 0).offset == 11)
        #expect(batch.cache(at: 1).offset == 3)
        #expect(batch.cache(at: 0).getPooled(.compressor)?.dim(1) == 7)
        #expect(batch.cache(at: 1).getPooled(.compressor)?.dim(1) == 2)
    }

    @Test("BatchScheduler preserves reusable partial prefixes for individual prefill")
    func reusablePartialPrefix() {
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 43,
            inputTokenCount: 59,
            hasRecurrentLayers: false,
            forcedSuffix: nil
        ) == 43)
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 10,
            inputTokenCount: 20,
            hasRecurrentLayers: false,
            forcedSuffix: nil
        ) == 4)
    }

    @Test("BatchScheduler bypasses unsafe recurrent exact replay")
    func recurrentExactReplayBypass() {
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 59,
            inputTokenCount: 59,
            hasRecurrentLayers: true,
            forcedSuffix: nil,
            sourceTokenCount: 59
        ) == 0)
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 59,
            inputTokenCount: 59,
            hasRecurrentLayers: true,
            forcedSuffix: 16
        ) == 43)
    }

    @Test("BatchScheduler rejects recurrent state captured beyond the matched prefix")
    func recurrentDescendantStateBypass() {
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 60,
            inputTokenCount: 95,
            hasRecurrentLayers: true,
            forcedSuffix: nil,
            sourceTokenCount: 69
        ) == 0)
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 60,
            inputTokenCount: 95,
            hasRecurrentLayers: true,
            forcedSuffix: nil,
            sourceTokenCount: 60
        ) == 60)
    }

    @Test("BatchScheduler restores exact DeepSeek prompt-minus-one boundary")
    func recurrentExactBoundaryRestore() {
        #expect(BatchScheduler.effectiveCachedPrefixLength(
            matchedPrefix: 58,
            inputTokenCount: 59,
            hasRecurrentLayers: true,
            forcedSuffix: nil,
            sourceTokenCount: 58
        ) == 58)
    }

    @Test("BatchScheduler snapshots generic recurrent caches at a replay boundary")
    func genericRecurrentBoundarySnapshot() {
        #expect(BatchScheduler.requiresReplayBoundarySnapshot([
            MambaCache(), KVCacheSimple()
        ]))
        #expect(!BatchScheduler.requiresReplayBoundarySnapshot([
            KVCacheSimple(), KVCacheSimple()
        ]))
    }

    @Test("BatchScheduler bounds recurrent prefix checkpoints")
    func recurrentCheckpointSelection() {
        #expect(BatchScheduler.recurrentCheckpointBoundaries(
            restoredPrefix: 0,
            finalBoundary: 870
        ) == [256, 512, 768])
        let long = BatchScheduler.recurrentCheckpointBoundaries(
            restoredPrefix: 0,
            finalBoundary: 32_767
        )
        #expect(long.count == 7)
        #expect(long.last! < 32_767)
        #expect(BatchScheduler.recurrentCheckpointBoundaries(
            restoredPrefix: 768,
            finalBoundary: 870
        ).isEmpty)
    }

    @Test("BatchScheduler cache snapshot owns independent MLX storage")
    func cacheSnapshotOwnsIndependentStorage() {
        let backing = MLXArray([Int32(10), Int32(20), Int32(30), Int32(40)])
        let snapshot = BatchScheduler.snapshotCacheState([backing])[0]
        MLX.eval([snapshot])

        backing[..<3] = MLXArray([Int32(90), Int32(91), Int32(92)])
        MLX.eval([backing])

        #expect(snapshot.asArray(Int32.self) == [10, 20, 30, 40])
    }

    @Test("BatchScheduler emits only completed tool calls from slot runtime events")
    func completedToolCallsFromEvents() {
        let placeholder = ResponseToolCall(
            index: 0,
            id: "call_placeholder",
            type: "function",
            function: ResponseToolCallFunction(name: "get_weather", arguments: "")
        )
        let final = ResponseToolCall(
            index: 0,
            id: "call_placeholder",
            type: "function",
            function: ResponseToolCallFunction(name: "get_weather", arguments: #"{"location":"Berlin"}"#)
        )
        let eagerAppend = ResponseToolCall(
            index: 1,
            id: "call_two",
            type: "function",
            function: ResponseToolCallFunction(name: "read_file", arguments: #"{"path":"README.md"}"#)
        )

        let emitted = BatchScheduler.completedToolCallsToEmit(from: [
            .started,
            .appendCollected(placeholder),
            .delta(StreamDeltaToolCall(index: 0, id: nil, type: nil, function: StreamDeltaFunction(name: nil, arguments: "{}"))),
            .replaceCollected(index: 0, toolCall: final),
            .appendCollected(eagerAppend),
        ])

        #expect(emitted.count == 2)
        #expect(emitted[0].function.name == "get_weather")
        #expect(emitted[0].function.arguments == #"{"location":"Berlin"}"#)
        #expect(emitted[1].function.name == "read_file")
        #expect(emitted[1].function.arguments == #"{"path":"README.md"}"#)
    }

    @Test("BatchScheduler emits incremental tool call deltas from slot runtime events")
    func deltaToolCallsFromEvents() {
        let deltas = BatchScheduler.deltaToolCallsToEmit(from: [
            .started,
            .appendCollected(ResponseToolCall(
                index: 0,
                id: "call_placeholder",
                type: "function",
                function: ResponseToolCallFunction(name: "get_weather", arguments: "")
            )),
            .delta(StreamDeltaToolCall(
                index: 0,
                id: "call_placeholder",
                type: "function",
                function: StreamDeltaFunction(name: "get_weather", arguments: "{\"location\":\"Berlin\"}")
            )),
            .replaceCollected(index: 0, toolCall: ResponseToolCall(
                index: 0,
                id: "call_placeholder",
                type: "function",
                function: ResponseToolCallFunction(name: "get_weather", arguments: #"{"location":"Berlin"}"#)
            )),
        ])

        #expect(deltas.count == 1)
        #expect(deltas[0].index == 0)
        #expect(deltas[0].id == "call_placeholder")
        #expect(deltas[0].function?.name == "get_weather")
        #expect(deltas[0].function?.arguments == "{\"location\":\"Berlin\"}")
    }

    @Test("BatchScheduler helper extraction keeps tool call streams isolated per event list")
    func helperExtractionKeepsToolStreamsIsolated() {
        let weatherEvents: [ToolCallStreamingEvent] = [
            .started,
            .delta(StreamDeltaToolCall(
                index: 0,
                id: "call_weather",
                type: "function",
                function: StreamDeltaFunction(name: "get_weather", arguments: "{\"location\":\"Berlin\"}")
            )),
            .replaceCollected(index: 0, toolCall: ResponseToolCall(
                index: 0,
                id: "call_weather",
                type: "function",
                function: ResponseToolCallFunction(name: "get_weather", arguments: #"{"location":"Berlin"}"#)
            )),
        ]
        let readEvents: [ToolCallStreamingEvent] = [
            .started,
            .delta(StreamDeltaToolCall(
                index: 0,
                id: "call_read",
                type: "function",
                function: StreamDeltaFunction(name: "read_file", arguments: "{\"path\":\"README.md\"}")
            )),
            .replaceCollected(index: 0, toolCall: ResponseToolCall(
                index: 0,
                id: "call_read",
                type: "function",
                function: ResponseToolCallFunction(name: "read_file", arguments: #"{"path":"README.md"}"#)
            )),
        ]

        let weatherDeltas = BatchScheduler.deltaToolCallsToEmit(from: weatherEvents)
        let weatherCompleted = BatchScheduler.completedToolCallsToEmit(from: weatherEvents)
        let readDeltas = BatchScheduler.deltaToolCallsToEmit(from: readEvents)
        let readCompleted = BatchScheduler.completedToolCallsToEmit(from: readEvents)

        #expect(weatherDeltas.count == 1)
        #expect(weatherCompleted.count == 1)
        #expect(weatherDeltas[0].function?.name == "get_weather")
        #expect(weatherCompleted[0].function.name == "get_weather")
        #expect(weatherDeltas[0].function?.name != readDeltas.first?.function?.name)
        #expect(weatherCompleted[0].function.name != readCompleted.first?.function.name)

        #expect(readDeltas.count == 1)
        #expect(readCompleted.count == 1)
        #expect(readDeltas[0].function?.name == "read_file")
        #expect(readCompleted[0].function.name == "read_file")
    }

    @Test("BatchScheduler preserves completed aggregate after argument deltas")
    func streamChunksPreserveCompletedAggregateAfterDeltas() {
        let events: [ToolCallStreamingEvent] = [
            .started,
            .appendCollected(ResponseToolCall(
                index: 0,
                id: "call_placeholder",
                type: "function",
                function: ResponseToolCallFunction(name: "get_weather", arguments: "")
            )),
            .delta(StreamDeltaToolCall(
                index: 0,
                id: "call_placeholder",
                type: "function",
                function: StreamDeltaFunction(name: "get_weather", arguments: "{\"location\":\"Berlin\"}")
            )),
            .replaceCollected(index: 0, toolCall: ResponseToolCall(
                index: 0,
                id: "call_placeholder",
                type: "function",
                function: ResponseToolCallFunction(name: "get_weather", arguments: #"{"location":"Berlin"}"#)
            )),
        ]

        let chunks = BatchScheduler.streamChunksToEmit(from: events)

        #expect(chunks.count == 2)
        #expect(chunks[0].toolCallDeltas?.count == 1)
        #expect(chunks[0].toolCalls == nil)
        #expect(chunks[1].toolCallDeltas == nil)
        #expect(chunks[1].toolCalls?.count == 1)
        #expect(chunks[1].toolCalls?.first?.function.name == "get_weather")
        #expect(chunks[1].toolCalls?.first?.function.arguments == #"{"location":"Berlin"}"#)
    }

    @Test("BatchScheduler stop helper emits stopped chunk on exact stop match")
    func stopHelperEmitsStoppedChunk() {
        var stopBuffer = ""
        var insideThink = false

        let result = BatchScheduler.stopChunksToEmit(
            from: "Hello\n\nUser:",
            stopBuffer: &stopBuffer,
            activeStops: ["\n\nUser:"],
            maxStopLength: "\n\nUser:".count,
            insideThink: &insideThink,
            thinkStartTag: nil,
            thinkEndTag: nil
        )

        #expect(result.stopped)
        #expect(result.chunks.count == 1)
        #expect(result.chunks[0].text == "Hello")
        #expect(result.chunks[0].stoppedBySequence == true)
    }

    @Test("BatchScheduler stop helper buffers partial stop across chunk boundaries")
    func stopHelperBuffersAcrossBoundaries() {
        var stopBuffer = ""
        var insideThink = false

        let first = BatchScheduler.stopChunksToEmit(
            from: "Hello\n\nUs",
            stopBuffer: &stopBuffer,
            activeStops: ["\n\nUser:"],
            maxStopLength: "\n\nUser:".count,
            insideThink: &insideThink,
            thinkStartTag: nil,
            thinkEndTag: nil
        )
        #expect(first.stopped == false)
        #expect(first.chunks.count == 1)
        #expect(first.chunks[0].text == "He")
        #expect(stopBuffer == "llo\n\nUs")

        let second = BatchScheduler.stopChunksToEmit(
            from: "er:",
            stopBuffer: &stopBuffer,
            activeStops: ["\n\nUser:"],
            maxStopLength: "\n\nUser:".count,
            insideThink: &insideThink,
            thinkStartTag: nil,
            thinkEndTag: nil
        )
        #expect(second.stopped)
        #expect(second.chunks.count == 1)
        #expect(second.chunks[0].text == "llo")
        #expect(second.chunks[0].stoppedBySequence == true)
    }

    @Test("BatchScheduler stop helper does not stop while still inside think block")
    func stopHelperDoesNotStopInsideThinkBlock() {
        var stopBuffer = ""
        var insideThink = false

        let result = BatchScheduler.stopChunksToEmit(
            from: "<think>plan\n\nUser:",
            stopBuffer: &stopBuffer,
            activeStops: ["\n\nUser:"],
            maxStopLength: "\n\nUser:".count,
            insideThink: &insideThink,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>"
        )

        #expect(result.stopped == false)
        #expect(result.chunks.count == 1)
        #expect(result.chunks[0].text == "<think>plan\n\nUser:")
        #expect(insideThink == true)
    }
}
