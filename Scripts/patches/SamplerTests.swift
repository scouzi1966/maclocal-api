// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

/// Tests for the custom LogitProcessor implementations added via vendor patches:
/// TopKProcessor, MinPProcessor, PresenceContext, CompositeLogitProcessor,
/// and the updated GenerateParameters.processor() factory.
public class SamplerTests: XCTestCase {

    /// Helper: create a Float32 MLXArray from a [Float] literal, shaped [1, N].
    private func floatLogits(_ values: [Float]) -> MLXArray {
        MLXArray(values).reshaped(1, values.count)
    }

    // MARK: - TopKProcessor

    func testTopKBasic() {
        // logits: [1, 5, 3, 2, 4] — top-2 should keep indices 1 (5.0) and 4 (4.0)
        let logits = floatLogits([1, 5, 3, 2, 4])
        let processor = TopKProcessor(k: 2)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[1], 5.0, accuracy: 1e-5)
        XCTAssertEqual(values[4], 4.0, accuracy: 1e-5)
        XCTAssertEqual(values[0], -Float.infinity)
        XCTAssertEqual(values[2], -Float.infinity)
        XCTAssertEqual(values[3], -Float.infinity)
    }

    func testTopKDisabledWhenZero() {
        let logits = floatLogits([1, 2, 3])
        let processor = TopKProcessor(k: 0)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values, [1, 2, 3])
    }

    func testTopKDisabledWhenLargerThanVocab() {
        let logits = floatLogits([1, 2, 3])
        let processor = TopKProcessor(k: 10)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values, [1, 2, 3])
    }

    func testTopKEqualsVocabSize() {
        let logits = floatLogits([1, 2, 3])
        let processor = TopKProcessor(k: 3)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values, [1, 2, 3])
    }

    func testTopKOne() {
        let logits = floatLogits([1, 5, 3, 2, 4])
        let processor = TopKProcessor(k: 1)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[1], 5.0, accuracy: 1e-5)
        for i in [0, 2, 3, 4] {
            XCTAssertEqual(values[i], -Float.infinity, "Index \(i) should be -inf")
        }
    }

    func testTopKWithDuplicates() {
        // Ties at boundary: top-2 values are 5.0 and 3.0 → threshold = 3.0
        // All values >= 3.0 survive
        let logits = floatLogits([1, 3, 3, 2, 5])
        let processor = TopKProcessor(k: 2)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[4], 5.0, accuracy: 1e-5)
        XCTAssertEqual(values[1], 3.0, accuracy: 1e-5)
        XCTAssertEqual(values[2], 3.0, accuracy: 1e-5)
        XCTAssertEqual(values[0], -Float.infinity)
        XCTAssertEqual(values[3], -Float.infinity)
    }

    // MARK: - MinPProcessor

    func testMinPBasic() {
        // logits: [10, 1, 0] — max=10, threshold = 10 + log(0.1) ≈ 7.697
        // Only index 0 (10.0) should survive
        let logits = floatLogits([10, 1, 0])
        let processor = MinPProcessor(minP: 0.1)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[0], 10.0, accuracy: 1e-5)
        XCTAssertEqual(values[1], -Float.infinity)
        XCTAssertEqual(values[2], -Float.infinity)
    }

    func testMinPDisabledWhenZero() {
        let logits = floatLogits([1, 2, 3])
        let processor = MinPProcessor(minP: 0.0)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values, [1, 2, 3])
    }

    func testMinPDisabledWhenOne() {
        let logits = floatLogits([1, 2, 3])
        let processor = MinPProcessor(minP: 1.0)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values, [1, 2, 3])
    }

    func testMinPPermissive() {
        // Very small minP — threshold = 3 + log(0.001) ≈ -3.908 → all survive
        let logits = floatLogits([1, 2, 3])
        let processor = MinPProcessor(minP: 0.001)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        for v in values {
            XCTAssertNotEqual(v, -Float.infinity)
        }
    }

    func testMinPStrict() {
        // Very high minP — threshold = 10 + log(0.99) ≈ 9.99
        // Only index 2 (10.0) should survive
        let logits = floatLogits([0, 0, 10])
        let processor = MinPProcessor(minP: 0.99)
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[2], 10.0, accuracy: 1e-5)
        XCTAssertEqual(values[0], -Float.infinity)
        XCTAssertEqual(values[1], -Float.infinity)
    }

    // MARK: - PresenceContext

    func testPresenceBasic() {
        var processor = PresenceContext(presencePenalty: 1.5, contextSize: 10)
        processor.prompt(MLXArray([Int32(0), Int32(1), Int32(2)]))

        let logits = floatLogits([10, 10, 10, 10, 10])
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        // Tokens 0, 1, 2 penalized by -1.5
        XCTAssertEqual(values[0], 8.5, accuracy: 1e-4)
        XCTAssertEqual(values[1], 8.5, accuracy: 1e-4)
        XCTAssertEqual(values[2], 8.5, accuracy: 1e-4)
        // Tokens 3, 4 unchanged
        XCTAssertEqual(values[3], 10.0, accuracy: 1e-4)
        XCTAssertEqual(values[4], 10.0, accuracy: 1e-4)
    }

    func testPresenceTracksNewTokens() {
        var processor = PresenceContext(presencePenalty: 2.0, contextSize: 10)
        processor.prompt(MLXArray([Int32]())) // empty prompt

        processor.didSample(token: MLXArray(Int32(3)))

        let logits = floatLogits([5, 5, 5, 5, 5])
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[0], 5.0, accuracy: 1e-4)
        XCTAssertEqual(values[3], 3.0, accuracy: 1e-4) // 5.0 - 2.0
    }

    func testPresenceDisabledWhenZero() {
        var processor = PresenceContext(presencePenalty: 0.0, contextSize: 10)
        processor.prompt(MLXArray([Int32(0), Int32(1), Int32(2)]))

        let logits = floatLogits([10, 10, 10])
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[0], 10.0, accuracy: 1e-4)
        XCTAssertEqual(values[1], 10.0, accuracy: 1e-4)
        XCTAssertEqual(values[2], 10.0, accuracy: 1e-4)
    }

    func testPresenceSlidingWindow() {
        var processor = PresenceContext(presencePenalty: 1.0, contextSize: 2)
        processor.prompt(MLXArray([Int32]()))

        processor.didSample(token: MLXArray(Int32(0)))
        processor.didSample(token: MLXArray(Int32(1)))
        processor.didSample(token: MLXArray(Int32(2))) // evicts token 0

        let logits = floatLogits([10, 10, 10, 10])
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[0], 10.0, accuracy: 1e-4) // evicted
        XCTAssertEqual(values[1], 9.0, accuracy: 1e-4)
        XCTAssertEqual(values[2], 9.0, accuracy: 1e-4)
        XCTAssertEqual(values[3], 10.0, accuracy: 1e-4)
    }

    func testPresenceNegativeLogits() {
        var processor = PresenceContext(presencePenalty: 1.0, contextSize: 10)
        processor.prompt(MLXArray([Int32(0)]))

        let logits = floatLogits([-5, 5])
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[0], -6.0, accuracy: 1e-4) // -5 - 1 = -6
        XCTAssertEqual(values[1], 5.0, accuracy: 1e-4)
    }

    // MARK: - CompositeLogitProcessor

    func testCompositeChaining() {
        // TopK(2) then MinP(0.5)
        // [1, 5, 3, 2, 4] → TopK(2) → [−inf, 5, −inf, −inf, 4]
        // MinP(0.5): max=5, threshold=5+log(0.5)≈4.307 → only index 1 survives
        let logits = floatLogits([1, 5, 3, 2, 4])
        let composite = CompositeLogitProcessor([
            TopKProcessor(k: 2),
            MinPProcessor(minP: 0.5),
        ])
        let result = composite.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[1], 5.0, accuracy: 1e-5)
        for i in [0, 2, 3, 4] {
            XCTAssertEqual(values[i], -Float.infinity, "Index \(i) should be -inf")
        }
    }

    func testCompositePromptAndDidSample() {
        var composite = CompositeLogitProcessor([
            PresenceContext(presencePenalty: 1.0, contextSize: 10),
            TopKProcessor(k: 3),
        ])

        composite.prompt(MLXArray([Int32(0), Int32(1)]))
        composite.didSample(token: MLXArray(Int32(2)))

        let logits = floatLogits([10, 10, 10, 10, 10])
        let result = composite.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        // Tokens 3 and 4 should be unpenalized
        XCTAssertEqual(values[3], 10.0, accuracy: 1e-4)
        XCTAssertEqual(values[4], 10.0, accuracy: 1e-4)
    }

    // MARK: - GenerateParameters.processor()

    func testProcessorReturnsNilWhenAllDisabled() {
        let params = GenerateParameters(
            temperature: 0.6,
            topP: 1.0,
            repetitionPenalty: nil,
            repetitionContextSize: 20,
            topK: 0,
            minP: 0.0,
            presencePenalty: 0.0
        )
        XCTAssertNil(params.processor())
    }

    func testProcessorReturnsSingleWhenOneEnabled() {
        let params = GenerateParameters(topK: 20)
        let processor = params.processor()
        XCTAssertNotNil(processor)
        XCTAssertTrue(processor is TopKProcessor)
    }

    func testProcessorReturnsCompositeWhenMultipleEnabled() {
        let params = GenerateParameters(
            repetitionPenalty: 1.2,
            repetitionContextSize: 20,
            topK: 20,
            presencePenalty: 1.5
        )
        let processor = params.processor()
        XCTAssertNotNil(processor)
        XCTAssertTrue(processor is CompositeLogitProcessor)
    }

    func testProcessorRepetitionPenaltyOneIsDisabled() {
        let params = GenerateParameters(
            repetitionPenalty: 1.0,
            repetitionContextSize: 20
        )
        XCTAssertNil(params.processor())
    }

    func testProcessorMinPBoundary() {
        let params1 = GenerateParameters(minP: 1.0)
        XCTAssertNil(params1.processor())

        let params2 = GenerateParameters(minP: 0.99)
        XCTAssertNotNil(params2.processor())
        XCTAssertTrue(params2.processor() is MinPProcessor)
    }

    // MARK: - Integration: sampler chain order

    func testSamplerChainOrder() {
        // presence_penalty=2 penalizes token 0 → 8.0
        // topK=3: sorted [10,10,10,10,8] → 3rd largest = 10 → token 0 (8.0) → -inf
        let params = GenerateParameters(
            repetitionContextSize: 20,
            topK: 3,
            presencePenalty: 2.0
        )
        var processor = params.processor()!
        processor.prompt(MLXArray([Int32(0)]))

        let logits = floatLogits([10, 10, 10, 10, 10])
        let result = processor.process(logits: logits)
        let values = result.reshaped(-1).asArray(Float.self)

        XCTAssertEqual(values[0], -Float.infinity)
        let nonInfCount = values.filter { $0 != -Float.infinity }.count
        XCTAssertGreaterThanOrEqual(nonInfCount, 3)
    }

    // MARK: - Seed determinism

    func testSeedDeterminism() {
        // Same seed + same logits → same sampled token every time
        let logits = floatLogits([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        var results: [Int32] = []
        for _ in 0..<5 {
            let sampler = CategoricalSampler(temperature: 0.9, seed: 42)
            let token = sampler.sample(logits: logits)
            results.append(token.item(Int32.self))
        }

        // All 5 samples with the same seed must produce the same token
        let allSame = results.allSatisfy { $0 == results[0] }
        XCTAssertTrue(allSame, "Same seed should produce identical results, got: \(results)")
    }

    func testDifferentSeedsDiffer() {
        // Different seeds should (with very high probability) produce different tokens
        // Using very flat logits (uniform-ish) to maximize entropy
        let logits = floatLogits([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        var tokens = Set<Int32>()
        for seed in UInt64(100)..<UInt64(120) {
            let sampler = CategoricalSampler(temperature: 1.0, seed: seed)
            let token = sampler.sample(logits: logits)
            tokens.insert(token.item(Int32.self))
        }

        // With 10 uniform tokens and 20 different seeds, we should get at least 2 distinct values
        XCTAssertGreaterThan(tokens.count, 1, "Different seeds should produce varied tokens")
    }

    func testSeedNilIsNonDeterministic() {
        // Without a seed, repeated sampling should (usually) vary
        let logits = floatLogits([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        var tokens = Set<Int32>()
        for _ in 0..<20 {
            let sampler = CategoricalSampler(temperature: 1.0)
            let token = sampler.sample(logits: logits)
            tokens.insert(token.item(Int32.self))
        }

        // 20 samples from 10 uniform logits — should see variation
        XCTAssertGreaterThan(tokens.count, 1, "Unseeded sampling should produce varied tokens")
    }

    func testTopPSamplerSeedDeterminism() {
        let logits = floatLogits([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        var results: [Int32] = []
        for _ in 0..<5 {
            let sampler = TopPSampler(temperature: 0.9, topP: 0.9, seed: 42)
            let token = sampler.sample(logits: logits)
            results.append(token.item(Int32.self))
        }

        let allSame = results.allSatisfy { $0 == results[0] }
        XCTAssertTrue(allSame, "Same seed should produce identical TopP results, got: \(results)")
    }

    func testGenerateParametersSeedPassthrough() {
        // Verify seed is stored and affects sampler creation
        let params = GenerateParameters(temperature: 0.8, seed: 42)
        XCTAssertEqual(params.seed, 42)

        let paramsNoSeed = GenerateParameters(temperature: 0.8)
        XCTAssertNil(paramsNoSeed.seed)
    }
}
