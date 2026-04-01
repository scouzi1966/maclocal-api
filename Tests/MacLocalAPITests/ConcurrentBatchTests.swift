import Foundation
import Testing

@testable import MacLocalAPI

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

    @Test("BatchScheduler emits delta chunks before completed tool call chunks")
    func streamChunksPreserveDeltaBeforeCompletedOrdering() {
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
        #expect(chunks[1].toolCalls?.count == 1)
        #expect(chunks[1].toolCallDeltas == nil)
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
