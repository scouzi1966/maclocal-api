import Foundation
import Testing
import MLX

@testable import MacLocalAPI
@testable import MLXLMCommon

/// Unit tests for logit processors in batch mode (1D and 2D logit tensors).
/// Verifies fix for issue #72: processors must handle both single-sequence
/// [vocabSize] and batched [B, vocabSize] logits without crashing.
struct LogitProcessorBatchTests {
// dimensions: execution=batch, sampling_params=top_k/min_p/presence_penalty/repetition_penalty

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - TopKProcessor
    // ═══════════════════════════════════════════════════════════════════

    @Test("TopKProcessor works with 1D logits [vocabSize]")
    func topK1D() {
        let proc = TopKProcessor(k: 2)
        let logits = MLXArray([1.0, 5.0, 3.0, 2.0, 4.0] as [Float])
        let result = proc.process(logits: logits)
        // Only top-2 (5.0, 4.0) should survive; rest should be -inf
        let values = result.asArray(Float.self)
        #expect(values[1] == 5.0)  // kept
        #expect(values[4] == 4.0)  // kept
        #expect(values[0] == -Float.infinity)  // masked
        #expect(values[2] == -Float.infinity)  // masked (3.0 is 3rd)
        #expect(values[3] == -Float.infinity)  // masked
    }

    @Test("TopKProcessor works with 2D logits [B, vocabSize]")
    func topK2D() {
        let proc = TopKProcessor(k: 2)
        // [2, 5] batch of logits
        let logits = MLXArray([1.0, 5.0, 3.0, 2.0, 4.0, 9.0, 1.0, 7.0, 3.0, 5.0] as [Float]).reshaped([2, 5])
        let result = proc.process(logits: logits)
        let flat = result.reshaped([-1]).asArray(Float.self)
        // Batch 0: top-2 are 5.0 (idx 1), 4.0 (idx 4)
        #expect(flat[1] == 5.0)
        #expect(flat[4] == 4.0)
        // Batch 1: top-2 are 9.0 (idx 0), 7.0 (idx 2)
        #expect(flat[5] == 9.0)
        #expect(flat[7] == 7.0)
    }

    @Test("TopKProcessor is no-op when k >= vocabSize")
    func topKNoOp() {
        let proc = TopKProcessor(k: 10)
        let logits = MLXArray([1.0, 2.0, 3.0] as [Float])
        let result = proc.process(logits: logits)
        #expect(result.asArray(Float.self) == [1.0, 2.0, 3.0])
    }

    @Test("TopKProcessor is no-op when k <= 0")
    func topKZero() {
        let proc = TopKProcessor(k: 0)
        let logits = MLXArray([1.0, 2.0, 3.0] as [Float])
        let result = proc.process(logits: logits)
        #expect(result.asArray(Float.self) == [1.0, 2.0, 3.0])
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MinPProcessor
    // ═══════════════════════════════════════════════════════════════════

    @Test("MinPProcessor works with 1D logits")
    func minP1D() {
        let proc = MinPProcessor(minP: 0.5)
        let logits = MLXArray([10.0, 1.0, 0.5, 9.5] as [Float])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        // threshold = max(10.0) + log(0.5) ≈ 10.0 - 0.693 = 9.307
        #expect(values[0] == 10.0)  // kept (10.0 >= 9.307)
        #expect(values[3] == 9.5)   // kept (9.5 >= 9.307)
        #expect(values[1] == -Float.infinity)  // masked
        #expect(values[2] == -Float.infinity)  // masked
    }

    @Test("MinPProcessor works with 2D logits")
    func minP2D() {
        let proc = MinPProcessor(minP: 0.5)
        let logits = MLXArray([10.0, 1.0, 9.5, 5.0, 4.0, 3.0] as [Float]).reshaped([2, 3])
        let result = proc.process(logits: logits)
        #expect(result.ndim == 2)
        #expect(result.shape == [2, 3])
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - PresenceContext
    // ═══════════════════════════════════════════════════════════════════

    @Test("PresenceContext works with 1D logits")
    func presence1D() {
        var proc = PresenceContext(presencePenalty: 2.0, contextSize: 10)
        proc.prompt(MLXArray([0, 1, 2] as [Int32]))
        let logits = MLXArray([10.0, 20.0, 30.0, 40.0, 50.0] as [Float])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        // Tokens 0, 1, 2 should be penalized by -2.0
        #expect(values[0] == 8.0)   // 10.0 - 2.0
        #expect(values[1] == 18.0)  // 20.0 - 2.0
        #expect(values[2] == 28.0)  // 30.0 - 2.0
        #expect(values[3] == 40.0)  // untouched
        #expect(values[4] == 50.0)  // untouched
    }

    @Test("PresenceContext works with 2D logits")
    func presence2D() {
        var proc = PresenceContext(presencePenalty: 1.0, contextSize: 10)
        proc.prompt(MLXArray([0, 2] as [Int32]))
        let logits = MLXArray([5.0, 10.0, 15.0] as [Float]).reshaped([1, 3])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        #expect(values[0] == 4.0)   // 5.0 - 1.0
        #expect(values[1] == 10.0)  // untouched
        #expect(values[2] == 14.0)  // 15.0 - 1.0
    }

    @Test("PresenceContext with empty tokens is no-op")
    func presenceEmpty() {
        let proc = PresenceContext(presencePenalty: 2.0, contextSize: 10)
        let logits = MLXArray([1.0, 2.0, 3.0] as [Float])
        let result = proc.process(logits: logits)
        #expect(result.asArray(Float.self) == [1.0, 2.0, 3.0])
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - RepetitionContext
    // ═══════════════════════════════════════════════════════════════════

    @Test("RepetitionContext works with 1D logits")
    func repetition1D() {
        var proc = RepetitionContext(repetitionPenalty: 2.0, repetitionContextSize: 10)
        proc.prompt(MLXArray([1] as [Int32]))
        let logits = MLXArray([10.0, 8.0, -4.0] as [Float])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        // Token 1 (value 8.0, positive) → divided by penalty: 8.0 / 2.0 = 4.0
        #expect(values[0] == 10.0)  // untouched
        #expect(values[1] == 4.0)   // 8.0 / 2.0
        #expect(values[2] == -4.0)  // untouched (not in context)
    }

    @Test("RepetitionContext works with 2D logits")
    func repetition2D() {
        var proc = RepetitionContext(repetitionPenalty: 2.0, repetitionContextSize: 10)
        proc.prompt(MLXArray([0] as [Int32]))
        let logits = MLXArray([6.0, 10.0, 3.0] as [Float]).reshaped([1, 3])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        #expect(values[0] == 3.0)   // 6.0 / 2.0 (positive, divided)
        #expect(values[1] == 10.0)  // untouched
        #expect(values[2] == 3.0)   // untouched
    }

    @Test("RepetitionContext applies multiplicative penalty to negative logits")
    func repetitionNegative() {
        var proc = RepetitionContext(repetitionPenalty: 2.0, repetitionContextSize: 10)
        proc.prompt(MLXArray([0] as [Int32]))
        let logits = MLXArray([-3.0, 5.0] as [Float])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        // Token 0 (value -3.0, negative) → multiplied by penalty: -3.0 * 2.0 = -6.0
        #expect(values[0] == -6.0)
        #expect(values[1] == 5.0)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - GrammarLogitProcessor
    // ═══════════════════════════════════════════════════════════════════

    @Test("GrammarLogitProcessor works with 1D logits")
    func grammar1D() {
        let proc = GrammarLogitProcessor()
        proc.tokenMask = MLXArray([0.0, -1e9, 0.0, -1e9] as [Float])
        let logits = MLXArray([5.0, 10.0, 3.0, 8.0] as [Float])
        let result = proc.process(logits: logits)
        let values = result.asArray(Float.self)
        #expect(values[0] == 5.0)
        #expect(values[1] < -1e8)  // effectively masked
        #expect(values[2] == 3.0)
        #expect(values[3] < -1e8)  // effectively masked
    }

    @Test("GrammarLogitProcessor works with 2D logits")
    func grammar2D() {
        let proc = GrammarLogitProcessor()
        proc.tokenMask = MLXArray([0.0, -1e9, 0.0] as [Float])
        let logits = MLXArray([5.0, 10.0, 3.0] as [Float]).reshaped([1, 3])
        let result = proc.process(logits: logits)
        // Broadcasting: [1,3] + [3] → [1,3]
        #expect(result.ndim == 2)
        let values = result.asArray(Float.self)
        #expect(values[0] == 5.0)
        #expect(values[1] < -1e8)
    }

    @Test("GrammarLogitProcessor with nil mask is passthrough")
    func grammarNilMask() {
        let proc = GrammarLogitProcessor()
        let logits = MLXArray([1.0, 2.0, 3.0] as [Float])
        let result = proc.process(logits: logits)
        #expect(result.asArray(Float.self) == [1.0, 2.0, 3.0])
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - CompositeLogitProcessor
    // ═══════════════════════════════════════════════════════════════════

    @Test("CompositeLogitProcessor chains processors correctly with 1D logits")
    func composite1D() {
        var presence = PresenceContext(presencePenalty: 1.0, contextSize: 10)
        presence.prompt(MLXArray([0] as [Int32]))
        let topK = TopKProcessor(k: 2)
        var composite = CompositeLogitProcessor([presence, topK])
        composite.prompt(MLXArray([0] as [Int32]))

        let logits = MLXArray([10.0, 5.0, 8.0] as [Float])
        let result = composite.process(logits: logits)
        // Step 1: presence penalty on token 0: [9.0, 5.0, 8.0]
        // Step 2: top-k=2 keeps top 2 (9.0, 8.0): [9.0, -inf, 8.0]
        let values = result.asArray(Float.self)
        #expect(values[0] == 9.0)
        #expect(values[1] == -Float.infinity)
        #expect(values[2] == 8.0)
    }

    @Test("CompositeLogitProcessor chains processors correctly with 2D logits")
    func composite2D() {
        var presence = PresenceContext(presencePenalty: 1.0, contextSize: 10)
        presence.prompt(MLXArray([0] as [Int32]))
        let topK = TopKProcessor(k: 2)
        var composite = CompositeLogitProcessor([presence, topK])
        composite.prompt(MLXArray([0] as [Int32]))

        let logits = MLXArray([10.0, 5.0, 8.0] as [Float]).reshaped([1, 3])
        let result = composite.process(logits: logits)
        #expect(result.ndim == 2)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Cross-Slot Isolation
    // ═══════════════════════════════════════════════════════════════════

    @Test("Separate processor instances have independent state")
    func processorIsolation() {
        // Simulate two slots with different parameters — use separate logit arrays
        // since PresenceContext mutates in-place
        var proc1 = PresenceContext(presencePenalty: 1.0, contextSize: 10)
        proc1.prompt(MLXArray([0] as [Int32]))

        var proc2 = PresenceContext(presencePenalty: 5.0, contextSize: 10)
        proc2.prompt(MLXArray([1] as [Int32]))

        let logits1 = MLXArray([10.0, 20.0, 30.0] as [Float])
        let logits2 = MLXArray([10.0, 20.0, 30.0] as [Float])

        let r1 = proc1.process(logits: logits1)
        let r2 = proc2.process(logits: logits2)

        let v1 = r1.asArray(Float.self)
        let v2 = r2.asArray(Float.self)

        // proc1: penalizes token 0 by -1.0
        #expect(v1[0] == 9.0)
        #expect(v1[1] == 20.0)

        // proc2: penalizes token 1 by -5.0
        #expect(v2[0] == 10.0)
        #expect(v2[1] == 15.0)
    }

    @Test("didSample updates only the called processor's state")
    func didSampleIsolation() {
        var proc1 = RepetitionContext(repetitionPenalty: 2.0, repetitionContextSize: 5)
        var proc2 = RepetitionContext(repetitionPenalty: 2.0, repetitionContextSize: 5)

        proc1.prompt(MLXArray([Int32]()))
        proc2.prompt(MLXArray([Int32]()))

        // Only proc1 sees token 0
        proc1.didSample(token: MLXArray(0))

        // Use separate logit arrays since RepetitionContext mutates in-place
        let logits1 = MLXArray([10.0, 5.0] as [Float])
        let logits2 = MLXArray([10.0, 5.0] as [Float])

        let r1 = proc1.process(logits: logits1)
        let r2 = proc2.process(logits: logits2)

        // proc1 should penalize token 0
        #expect(r1.asArray(Float.self)[0] == 5.0)  // 10.0 / 2.0
        // proc2 should not penalize anything
        #expect(r2.asArray(Float.self)[0] == 10.0)
    }
}
