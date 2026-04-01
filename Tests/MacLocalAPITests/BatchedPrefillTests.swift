import Foundation
import MLX
import MLXFast
import MLXLMCommon
import Testing

@testable import MacLocalAPI

/// Unit tests for batched prefill infrastructure:
/// - BatchKVCacheSimple mask creation with leftPadding
/// - MambaCache leftPadding mask creation
/// - Left-padding token construction and stacking
/// - Cache update with batched input
/// - Individual cache extraction from batch
/// - Merge batched prefill caches into decode batch
struct BatchedPrefillTests {
// dimensions: execution=batch, cache_type=KVCacheSimple/MambaCache/CacheList

    init() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - BatchKVCacheSimple Mask Tests
    // ═══════════════════════════════════════════════════════════════════

    @Test("BatchKVCacheSimple mask with uniform padding (same-length sequences)")
    func batchMaskUniformPadding() {
        // B=2, both sequences same length → leftPadding = [0, 0]
        let cache = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 0])
        let maskMode = cache.makeMask(n: 4, windowSize: nil, returnArray: false)

        // n=4 (prefill), _idx=0, totalLen=4
        // Should produce causal mask [B, 1, 4, 4]
        if case .array(let mask) = maskMode {
            #expect(mask.ndim == 4)
            #expect(mask.dim(0) == 2)  // B=2
            #expect(mask.dim(2) == 4)  // n=4
            #expect(mask.dim(3) == 4)  // totalLen=4

            // Both sequences have no padding, so mask is standard causal
            // Position 0 should attend to position 0 only
            // Position 3 should attend to all 4 positions
            let m0 = mask[0, 0].asArray(Bool.self)  // [4, 4]
            // Row 0: [true, false, false, false]
            #expect(m0[0] == true)
            #expect(m0[1] == false)
            // Row 3: [true, true, true, true]
            #expect(m0[12] == true)
            #expect(m0[13] == true)
            #expect(m0[14] == true)
            #expect(m0[15] == true)
        } else {
            Issue.record("Expected .array mask for batched prefill, got \(maskMode)")
        }
    }

    @Test("BatchKVCacheSimple mask with asymmetric padding (different-length sequences)")
    func batchMaskAsymmetricPadding() {
        // B=2: seq0 has 4 tokens (no padding), seq1 has 2 tokens (2 padding)
        let cache = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 2])
        let maskMode = cache.makeMask(n: 4, windowSize: nil, returnArray: false)

        if case .array(let mask) = maskMode {
            #expect(mask.dim(0) == 2)  // B
            #expect(mask.dim(2) == 4)  // n
            #expect(mask.dim(3) == 4)  // totalLen

            // Seq 0 (no padding): standard causal mask
            let m0 = mask[0, 0].asArray(Bool.self)
            #expect(m0[0] == true)   // pos 0 attends to pos 0
            #expect(m0[1] == false)  // pos 0 does NOT attend to pos 1

            // Seq 1 (padding=2): positions 0,1 are padding → masked out
            let m1 = mask[1, 0].asArray(Bool.self)
            // Row 0 of seq1: all false (position 0 is padding, can't attend anywhere valid)
            #expect(m1[0] == false)  // pos 0 → pos 0 (both padding)
            #expect(m1[1] == false)  // pos 0 → pos 1 (also padding)
            // Row 2 of seq1 (first real token): should attend to pos 2 only
            #expect(m1[8] == false)   // pos 2 → pos 0 (padding)
            #expect(m1[9] == false)   // pos 2 → pos 1 (padding)
            #expect(m1[10] == true)   // pos 2 → pos 2 (self)
            #expect(m1[11] == false)  // pos 2 → pos 3 (causal: future)
            // Row 3 of seq1 (second real token): should attend to pos 2 and 3
            #expect(m1[12] == false)  // pos 3 → pos 0 (padding)
            #expect(m1[13] == false)  // pos 3 → pos 1 (padding)
            #expect(m1[14] == true)   // pos 3 → pos 2
            #expect(m1[15] == true)   // pos 3 → pos 3 (self)
        } else {
            Issue.record("Expected .array mask")
        }
    }

    @Test("BatchKVCacheSimple decode mask with padding (n=1)")
    func batchMaskDecodeWithPadding() {
        // After prefill of 4 tokens, decode step (n=1)
        let cache = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 2])
        // Simulate prefill by writing 4 tokens
        let B = 2, heads = 2, headDim = 8
        let _ = cache.update(
            keys: MLXArray.ones([B, heads, 4, headDim]),
            values: MLXArray.ones([B, heads, 4, headDim])
        )
        #expect(cache.offset == 4)

        // Decode: n=1, totalLen=5
        let maskMode = cache.makeMask(n: 1, windowSize: nil, returnArray: false)
        if case .array(let mask) = maskMode {
            // [B, 1, 1, 5]
            #expect(mask.dim(0) == 2)
            #expect(mask.dim(2) == 1)
            #expect(mask.dim(3) == 5)

            // Seq 0 (no padding): attend to all 5 positions
            let m0 = mask[0, 0, 0].asArray(Bool.self)
            #expect(m0.allSatisfy { $0 == true })

            // Seq 1 (padding=2): attend to positions 2,3,4 only
            let m1 = mask[1, 0, 0].asArray(Bool.self)
            #expect(m1[0] == false)  // padding
            #expect(m1[1] == false)  // padding
            #expect(m1[2] == true)
            #expect(m1[3] == true)
            #expect(m1[4] == true)
        } else {
            Issue.record("Expected .array mask for batched decode")
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MambaCache Mask Tests
    // ═══════════════════════════════════════════════════════════════════

    @Test("MambaCache leftPadding creates correct SSM mask")
    func mambaCacheLeftPaddingMask() {
        let cache = MambaCache(leftPadding: [0, 2, 1])
        // N=4 (sequence length after padding)
        let mask = cache.makeMask(N: 4)
        #expect(mask != nil)

        if let mask {
            // Shape should be [B, N] = [3, 4]
            #expect(mask.dim(0) == 3)
            #expect(mask.dim(1) == 4)

            let m = mask.asArray(Bool.self)
            // Seq 0 (padding=0): [true, true, true, true]
            #expect(m[0] == true)
            #expect(m[1] == true)
            #expect(m[2] == true)
            #expect(m[3] == true)
            // Seq 1 (padding=2): [false, false, true, true]
            #expect(m[4] == false)
            #expect(m[5] == false)
            #expect(m[6] == true)
            #expect(m[7] == true)
            // Seq 2 (padding=1): [false, true, true, true]
            #expect(m[8] == false)
            #expect(m[9] == true)
            #expect(m[10] == true)
            #expect(m[11] == true)
        }
    }

    @Test("MambaCache without leftPadding returns nil mask")
    func mambaCacheNoLeftPadding() {
        let cache = MambaCache()
        let mask = cache.makeMask(N: 4)
        #expect(mask == nil)
    }

    @Test("MambaCache mask returns nil after state is set (non-prefill)")
    func mambaCacheMaskNilAfterStateSet() {
        let cache = MambaCache(leftPadding: [0, 1])
        // Simulate forward pass setting state
        cache[0] = MLXArray.ones([2, 16])
        // After state is set, makeMask should return nil (cache[0] != nil)
        let mask = cache.makeMask(N: 4)
        #expect(mask == nil)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Left-Padding Token Construction
    // ═══════════════════════════════════════════════════════════════════

    @Test("Left-padding stacks variable-length token arrays correctly")
    func leftPaddingTokenConstruction() {
        // Simulate 3 prompts of different lengths
        let prompts: [[Int]] = [
            [10, 20, 30, 40, 50],  // 5 tokens
            [100, 200, 300],        // 3 tokens
            [1, 2, 3, 4],           // 4 tokens
        ]

        let lengths = prompts.map { $0.count }
        let maxLen = lengths.max()!
        #expect(maxLen == 5)

        var leftPads: [Int] = []
        var paddedRows: [[Int32]] = []

        for tokens in prompts {
            let pad = maxLen - tokens.count
            leftPads.append(pad)
            paddedRows.append(Array(repeating: Int32(0), count: pad) + tokens.map { Int32($0) })
        }

        #expect(leftPads == [0, 2, 1])

        let batchTokens = stacked(paddedRows.map { MLXArray($0) })
        #expect(batchTokens.shape == [3, 5])

        // Verify padding positions are 0
        let row1 = batchTokens[1].asArray(Int32.self)
        #expect(row1[0] == 0)   // padding
        #expect(row1[1] == 0)   // padding
        #expect(row1[2] == 100) // real token
        #expect(row1[3] == 200)
        #expect(row1[4] == 300)

        // Verify last position is always the last real token
        for i in 0..<3 {
            let lastToken = batchTokens[i, -1].item(Int32.self)
            #expect(Int(lastToken) == prompts[i].last!)
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - BatchKVCacheSimple Update & Extract
    // ═══════════════════════════════════════════════════════════════════

    @Test("BatchKVCacheSimple update stores KV for all sequences")
    func batchCacheUpdate() {
        let B = 3, heads = 2, headDim = 4
        let cache = BatchKVCacheSimple(batchSize: B, leftPadding: [0, 1, 2])

        // Simulate prefill: 5 tokens
        let keys = MLXArray.ones([B, heads, 5, headDim])
        let values = MLXArray.ones([B, heads, 5, headDim]) * 2
        let (retK, retV) = cache.update(keys: keys, values: values)

        #expect(cache.offset == 5)
        #expect(retK.dim(0) == B)
        #expect(retK.dim(2) == 5)
        #expect(retV.dim(0) == B)
    }

    @Test("BatchKVCacheSimple extract returns correct per-sequence KV")
    func batchCacheExtract() {
        let B = 2, heads = 2, headDim = 4
        let cache = BatchKVCacheSimple(batchSize: B, leftPadding: [0, 2])

        // Write 4 tokens with distinguishable values
        var keys = MLXArray.zeros([B, heads, 4, headDim])
        keys[0] = MLXArray.ones([heads, 4, headDim]) * 1.0  // seq 0
        keys[1] = MLXArray.ones([heads, 4, headDim]) * 2.0  // seq 1
        var values = MLXArray.zeros([B, heads, 4, headDim])
        values[0] = MLXArray.ones([heads, 4, headDim]) * 10.0
        values[1] = MLXArray.ones([heads, 4, headDim]) * 20.0
        let _ = cache.update(keys: keys, values: values)

        // Extract seq 0: no padding, should get all 4 tokens
        let (k0, v0, tc0) = cache.extract(0)
        #expect(tc0 == 4)
        #expect(k0.dim(0) == 1)   // batch dim = 1
        #expect(k0.dim(2) == 4)   // seq len = 4
        // All key values should be ~1.0
        let k0val = k0[0, 0, 0, 0].item(Float.self)
        #expect(abs(k0val - 1.0) < 1e-5)

        // Extract seq 1: padding=2, should get 2 real tokens (positions 2,3)
        let (k1, v1, tc1) = cache.extract(1)
        #expect(tc1 == 2)
        #expect(k1.dim(0) == 1)
        #expect(k1.dim(2) == 2)
        // Key values should be ~2.0
        let k1val = k1[0, 0, 0, 0].item(Float.self)
        #expect(abs(k1val - 2.0) < 1e-5)
        // Value should be ~20.0
        let v1val = v1[0, 0, 0, 0].item(Float.self)
        #expect(abs(v1val - 20.0) < 1e-5)
    }

    @Test("Extracted cache can be loaded into KVCacheSimple for radix save")
    func extractedCacheToKVCacheSimple() {
        let B = 2, heads = 2, headDim = 4
        let cache = BatchKVCacheSimple(batchSize: B, leftPadding: [0, 1])
        let _ = cache.update(
            keys: MLXArray.ones([B, heads, 3, headDim]),
            values: MLXArray.ones([B, heads, 3, headDim])
        )

        // Extract seq 1 (padding=1 → 2 real tokens)
        let (k, v, tokenCount) = cache.extract(1)
        #expect(tokenCount == 2)

        // Load into KVCacheSimple
        let individual = KVCacheSimple()
        individual.state = [k, v]
        #expect(individual.offset == 2)
        #expect(individual.state[0].dim(2) == 2)

        // Verify isTrimmable and truncateToOffset work
        #expect(individual.isTrimmable)
        individual.truncateToOffset()
        #expect(individual.state[0].dim(2) == 2)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - MambaCache State Extraction
    // ═══════════════════════════════════════════════════════════════════

    @Test("MambaCache batched state can be sliced per-sequence")
    func mambaCacheStateExtraction() {
        let B = 3
        let cache = MambaCache(leftPadding: [0, 1, 2])

        // Simulate forward pass writing batched state
        cache[0] = MLXArray.ones([B, 16]) * MLXArray([Float(1), Float(2), Float(3)]).reshaped([B, 1])
        cache[1] = MLXArray.ones([B, 8]) * MLXArray([Float(10), Float(20), Float(30)]).reshaped([B, 1])

        let fullState = cache.state
        #expect(fullState.count == 2)
        #expect(fullState[0].dim(0) == B)

        // Extract per-sequence
        for i in 0..<B {
            let individual = MambaCache()
            individual.state = fullState.map { $0[i ..< i + 1] }
            let indState = individual.state
            #expect(indState.count == 2)
            #expect(indState[0].dim(0) == 1)

            // Verify values
            let expected = Float(i + 1)
            let actual = indState[0][0, 0].item(Float.self)
            #expect(abs(actual - expected) < 1e-5)
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Cache Type Classification
    // ═══════════════════════════════════════════════════════════════════

    @Test("KVCacheSimple and ArraysCache are batch-eligible")
    func cacheTypeClassificationEligible() {
        let caches: [KVCache] = [
            KVCacheSimple(),
            MambaCache(),
            KVCacheSimple(),
            MambaCache(),
        ]
        let canBatch = caches.allSatisfy { $0 is KVCacheSimple || $0 is ArraysCache }
        #expect(canBatch)
    }

    @Test("RotatingKVCache is NOT batch-eligible")
    func cacheTypeClassificationIneligible() {
        let caches: [KVCache] = [
            KVCacheSimple(),
            RotatingKVCache(maxSize: 1024),
            MambaCache(),
        ]
        let canBatch = caches.allSatisfy { $0 is KVCacheSimple || $0 is ArraysCache }
        #expect(!canBatch)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Merge Into Existing Decode Batch
    // ═══════════════════════════════════════════════════════════════════

    @Test("BatchKVCacheSimple extend merges prefill batch with existing decode batch")
    func batchCacheExtendMerge() {
        let heads = 2, headDim = 4

        // Existing decode batch: 1 sequence, 10 tokens decoded
        let existing = BatchKVCacheSimple(batchSize: 1, leftPadding: [0])
        let _ = existing.update(
            keys: MLXArray.ones([1, heads, 10, headDim]),
            values: MLXArray.ones([1, heads, 10, headDim])
        )
        #expect(existing.offset == 10)

        // New batch from prefill: 2 sequences, 5 tokens each
        let prefill = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 0])
        let _ = prefill.update(
            keys: MLXArray.ones([2, heads, 5, headDim]) * 2,
            values: MLXArray.ones([2, heads, 5, headDim]) * 2
        )

        // Merge: extract individual caches and extend one by one
        for i in 0..<2 {
            let (k, v, _) = prefill.extract(i)
            let single = BatchKVCacheSimple.merge([{
                let c = KVCacheSimple()
                c.state = [k, v]
                return c
            }()])
            existing.extend(with: single)
        }

        // After merge: B=3, offset = max(10, 5) = 10
        #expect(existing.offset == 10)
        // Verify state includes all 3 sequences (keys dim 0 should be 3)
        let state = existing.state
        #expect(state.count == 4)  // keys, values, perSeqOffset, leftPadding
        #expect(state[0].dim(0) == 3)  // B=3
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - createAttentionMask with BatchKVCacheSimple
    // ═══════════════════════════════════════════════════════════════════

    @Test("createAttentionMask delegates to BatchKVCacheSimple.makeMask")
    func createAttentionMaskWithBatchCache() {
        let cache = BatchKVCacheSimple(batchSize: 2, leftPadding: [0, 1])
        // h shape: [B, seqLen, hiddenDim] — seqLen=3
        let h = MLXArray.ones([2, 3, 8])

        let maskMode = createAttentionMask(h: h, cache: cache, windowSize: nil, returnArray: false)
        if case .array(let mask) = maskMode {
            // Should be [B, 1, 3, 3] (n=3, totalLen=0+3=3)
            #expect(mask.ndim == 4)
            #expect(mask.dim(0) == 2)
            #expect(mask.dim(2) == 3)
            #expect(mask.dim(3) == 3)
        } else {
            Issue.record("Expected .array mask from BatchKVCacheSimple")
        }
    }

    @Test("createSSMMask delegates to MambaCache.makeMask")
    func createSSMMaskWithBatchedMambaCache() {
        let cache = MambaCache(leftPadding: [0, 2])
        let h = MLXArray.ones([2, 4, 8])  // B=2, seqLen=4

        let mask = createSSMMask(h: h, cache: cache)
        #expect(mask != nil)
        if let mask {
            #expect(mask.dim(0) == 2)
            #expect(mask.dim(1) == 4)
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - End-to-End: Prefill Cache Creation Pattern
    // ═══════════════════════════════════════════════════════════════════

    @Test("Batched prefill cache creation matches expected pattern for mixed layers")
    func prefillCacheCreationPattern() {
        // Simulate Qwen3.5-MoE: 4 layers, 3 linear (MambaCache) + 1 attention (KVCacheSimple)
        let templateCaches: [KVCache] = [
            MambaCache(),       // layer 0: linear
            MambaCache(),       // layer 1: linear
            MambaCache(),       // layer 2: linear
            KVCacheSimple(),    // layer 3: attention (every 4th)
        ]

        let B = 4
        let leftPads = [0, 3, 1, 5]

        // Create batched caches following the prefillBatch pattern
        let batchedCaches: [KVCache] = templateCaches.map { layerCache in
            if layerCache is ArraysCache {
                return MambaCache(leftPadding: leftPads) as KVCache
            } else {
                return BatchKVCacheSimple(batchSize: B, leftPadding: leftPads) as KVCache
            }
        }

        // Verify types
        #expect(batchedCaches[0] is MambaCache)
        #expect(batchedCaches[1] is MambaCache)
        #expect(batchedCaches[2] is MambaCache)
        #expect(batchedCaches[3] is BatchKVCacheSimple)

        // Verify BatchKVCacheSimple has correct state via metaState
        let bkvc = batchedCaches[3] as! BatchKVCacheSimple
        let meta = bkvc.metaState
        #expect(meta[0] == "4")  // batchSize
        #expect(meta[1] == "0")  // _idx (empty cache)
    }

    @Test("Full extraction round-trip: batch → individual → KVCacheSimple")
    func fullExtractionRoundTrip() {
        let B = 3, heads = 2, headDim = 4, seqLen = 6
        let leftPads = [0, 2, 1]

        // Create and populate batched attention cache
        let batchCache = BatchKVCacheSimple(batchSize: B, leftPadding: leftPads)
        let keys = MLXArray.ones([B, heads, seqLen, headDim])
        let values = MLXArray.ones([B, heads, seqLen, headDim])
        let _ = batchCache.update(keys: keys, values: values)

        // Create and populate batched SSM cache
        let ssmCache = MambaCache(leftPadding: leftPads)
        ssmCache[0] = MLXArray.ones([B, 32])
        ssmCache[1] = MLXArray.ones([B, 16])

        // Extract per-request caches
        for i in 0..<B {
            // Attention
            let (k, v, tc) = batchCache.extract(i)
            let expectedLen = seqLen - leftPads[i]
            #expect(tc == expectedLen)
            #expect(k.dim(2) == expectedLen)

            let individual = KVCacheSimple()
            individual.state = [k, v]
            #expect(individual.offset == expectedLen)

            // SSM
            let ssmState = ssmCache.state
            let indSSM = MambaCache()
            indSSM.state = ssmState.map { $0[i ..< i + 1] }
            #expect(indSSM.state.count == 2)
            #expect(indSSM.state[0].dim(0) == 1)
        }
    }
}
