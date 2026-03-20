import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MacLocalAPI

struct KVCacheTruncateTests {

    @Test("truncateToOffset physically shrinks arrays to offset")
    func truncateShrinksArrays() {
        let cache = KVCacheSimple()
        let B = 1, heads = 4, headDim = 32
        let _ = cache.update(keys: MLXArray.ones([B, heads, 100, headDim]),
                             values: MLXArray.ones([B, heads, 100, headDim]))
        #expect(cache.offset == 100)
        let _ = cache.update(keys: MLXArray.ones([B, heads, 50, headDim]),
                             values: MLXArray.ones([B, heads, 50, headDim]))
        #expect(cache.offset == 150)
        cache.trim(50)
        #expect(cache.offset == 100)
        cache.truncateToOffset()
        let state = cache.state
        #expect(state.count == 2)
        #expect(state[0].dim(2) == 100)
        #expect(state[1].dim(2) == 100)
    }

    @Test("truncateToOffset is no-op when offset matches array size")
    func truncateNoOpWhenAligned() {
        let cache = KVCacheSimple()
        let _ = cache.update(keys: MLXArray.ones([1, 4, 50, 32]),
                             values: MLXArray.ones([1, 4, 50, 32]))
        #expect(cache.offset == 50)
        cache.truncateToOffset()
        let state = cache.state
        #expect(state[0].dim(2) == 50)
        #expect(state[1].dim(2) == 50)
    }

    @Test("truncateToOffset on empty cache is no-op")
    func truncateEmptyCache() {
        let cache = KVCacheSimple()
        cache.truncateToOffset()
        #expect(cache.offset == 0)
        #expect(cache.state.isEmpty)
    }

    @Test("state getter returns raw refs after truncateToOffset (fast path)")
    func stateReturnsRawRefsAfterTruncate() {
        let cache = KVCacheSimple()
        let _ = cache.update(keys: MLXArray.ones([1, 2, 80, 16]),
                             values: MLXArray.ones([1, 2, 80, 16]))
        let _ = cache.update(keys: MLXArray.ones([1, 2, 20, 16]),
                             values: MLXArray.ones([1, 2, 20, 16]))
        cache.trim(20)
        cache.truncateToOffset()
        let state = cache.state
        #expect(state[0].dim(2) == cache.offset)
        #expect(state[1].dim(2) == cache.offset)
    }

    @Test("truncateToOffset after multiple trim cycles")
    func truncateAfterMultipleTrimCycles() {
        let cache = KVCacheSimple()
        let _ = cache.update(keys: MLXArray.ones([1, 2, 200, 16]),
                             values: MLXArray.ones([1, 2, 200, 16]))
        let _ = cache.update(keys: MLXArray.ones([1, 2, 100, 16]),
                             values: MLXArray.ones([1, 2, 100, 16]))
        cache.trim(100)
        cache.truncateToOffset()
        #expect(cache.state[0].dim(2) == 200)

        let _ = cache.update(keys: MLXArray.ones([1, 2, 50, 16]),
                             values: MLXArray.ones([1, 2, 50, 16]))
        cache.trim(50)
        cache.truncateToOffset()
        #expect(cache.state[0].dim(2) == 200)
    }

    @Test("BaseKVCache truncateToOffset is safe no-op on MambaCache")
    func baseKVCacheFallback() {
        let cache = MambaCache()
        cache.truncateToOffset()  // should not crash
    }
}
