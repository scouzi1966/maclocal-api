import Foundation
import Testing

@testable import MacLocalAPI

/// Unit tests for Phase 1 concurrent batching internals:
/// RequestSlot, StreamChunk, and BatchScheduler queuing logic.
struct ConcurrentBatchTests {

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
}
