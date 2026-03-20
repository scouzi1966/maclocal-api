import Foundation
import MLX
import Testing

@testable import MacLocalAPI

struct RadixTreeCacheTests {

    /// Return placeholder layer states without creating real MLXArrays.
    /// RadixTreeCache only stores/returns layer states — it never reads array
    /// contents — so empty inner arrays are sufficient for testing the tree
    /// logic.  This avoids the Metal-library-not-found crash that occurs when
    /// `swift test` runs outside the GPU sandbox.
    private func fakeLayers(seqLen: Int) -> [[MLXArray]] {
        return [[], []]
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - KVCacheEntry
    // ═══════════════════════════════════════════════════════════════════

    @Test("KVCacheEntry stores tokens and layerStates")
    func cacheEntryInit() {
        let entry = KVCacheEntry(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(entry.tokens == [1, 2, 3])
        #expect(entry.layerStates.count == 2)
        #expect(entry.lastAccessTime > 0)
    }

    @Test("KVCacheEntry touch updates access time")
    func cacheEntryTouch() {
        let entry = KVCacheEntry(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        let t0 = entry.lastAccessTime
        for _ in 0..<10000 { _ = 1 + 1 }
        entry.touch()
        #expect(entry.lastAccessTime >= t0)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - RadixNode
    // ═══════════════════════════════════════════════════════════════════

    @Test("RadixNode isLeaf and hasCachedState")
    func nodeProperties() {
        let node = RadixNode(edgeTokens: [1, 2])
        #expect(node.isLeaf == true)
        #expect(node.hasCachedState == false)

        node.cacheEntry = KVCacheEntry(tokens: [1, 2], layerStates: [])
        #expect(node.hasCachedState == true)

        let child = RadixNode(edgeTokens: [3], parent: node)
        node.children[3] = child
        #expect(node.isLeaf == false)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - findPrefix: all branches
    // ═══════════════════════════════════════════════════════════════════

    @Test("findPrefix: exact match returns full prefix")
    func findExact() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        let (prefixLen, states, _) = cache.findPrefix([1, 2, 3, 4, 5])
        #expect(prefixLen == 5)
        #expect(states != nil)
        #expect(states!.count == 2)
    }

    @Test("findPrefix: query longer than cached — partial prefix hit")
    func findPartialPrefix() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        let (prefixLen, states, _) = cache.findPrefix([1, 2, 3, 4, 5])
        #expect(prefixLen == 3)
        #expect(states != nil)
    }

    @Test("findPrefix: no child for next token — miss at root")
    func findMissAtRoot() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        let (prefixLen, states, _) = cache.findPrefix([9, 8, 7])
        #expect(prefixLen == 0)
        #expect(states == nil)
    }

    @Test("findPrefix: empty cache — miss")
    func findMissEmptyCache() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        let (prefixLen, states, _) = cache.findPrefix([1, 2, 3])
        #expect(prefixLen == 0)
        #expect(states == nil)
    }

    @Test("findPrefix: partial edge match divergence mid-edge — returns cached state")
    func findPartialEdgeDivergence() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        let (prefixLen, states, _) = cache.findPrefix([1, 2, 3, 9, 9])
        #expect(prefixLen == 3)
        #expect(states != nil)
    }

    @Test("findPrefix: traverses nodes but no cached node found")
    func findTraversedButNoCachedNode() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3, 9, 8], layerStates: fakeLayers(seqLen: 5))
        let (prefixLen, states, _) = cache.findPrefix([1, 2, 3, 7, 7])
        #expect(prefixLen == 0)
        #expect(states == nil)
    }

    @Test("findPrefix: query shorter than cached edge")
    func findQueryShorterThanEdge() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        let (prefixLen, states, _) = cache.findPrefix([1, 2])
        #expect(prefixLen == 2)
        #expect(states != nil)
    }

    @Test("findPrefix: deepest cached node wins over shallower")
    func findDeepestCachedNode() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        let (prefixLen, _, _) = cache.findPrefix([1, 2, 3, 4, 5, 6])
        #expect(prefixLen == 5)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - findPrefix: debug logging branches
    // ═══════════════════════════════════════════════════════════════════

    @Test("findPrefix with debugLogging exercises all log branches")
    func findWithDebugLogging() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64, debugLogging: true)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        let _ = cache.findPrefix([1, 2, 3])
        let _ = cache.findPrefix([9])
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        let _ = cache.findPrefix([1, 2, 3, 4, 9])
    }

    @Test("findPrefix: miss with matched > 0 and debugLogging")
    func findMissMatchedWithDebug() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64, debugLogging: true)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3, 9, 8], layerStates: fakeLayers(seqLen: 5))
        let _ = cache.findPrefix([1, 2, 3, 7, 7])
    }

    @Test("findPrefix: miss with matched == 0 and debugLogging")
    func findMissNoMatchWithDebug() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64, debugLogging: true)
        cache.insert(tokens: [1, 2], layerStates: fakeLayers(seqLen: 2))
        let _ = cache.findPrefix([9, 9])
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - insert: all branches
    // ═══════════════════════════════════════════════════════════════════

    @Test("insert: empty tokens is no-op")
    func insertEmptyNoOp() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [], layerStates: fakeLayers(seqLen: 0))
        #expect(cache.count == 0)
    }

    @Test("insert: new edge (no matching child)")
    func insertNewEdge() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 1)
        let (len, _, _) = cache.findPrefix([1, 2, 3])
        #expect(len == 3)
    }

    @Test("insert: partial edge split with remaining tokens")
    func insertSplitWithRemaining() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3, 9, 8], layerStates: fakeLayers(seqLen: 5))
        #expect(cache.count == 2)
        let (len1, _, _) = cache.findPrefix([1, 2, 3, 4, 5])
        #expect(len1 == 5)
        let (len2, _, _) = cache.findPrefix([1, 2, 3, 9, 8])
        #expect(len2 == 5)
    }

    @Test("insert: partial edge split at exact split point")
    func insertSplitExact() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 2)
        let (len1, _, _) = cache.findPrefix([1, 2, 3])
        #expect(len1 == 3)
        let (len2, _, _) = cache.findPrefix([1, 2, 3, 4, 5])
        #expect(len2 == 5)
    }

    @Test("insert: exact match update — node already has cache")
    func insertExactUpdateExisting() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 1)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 1)
    }

    @Test("insert: exact match — node had no prior cache")
    func insertExactNewCache() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 2)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 2)
    }

    @Test("insert: multiple inserts grow the tree (multi-turn)")
    func insertMultiTurn() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3], layerStates: fakeLayers(seqLen: 3))
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3, 4, 5, 6, 7], layerStates: fakeLayers(seqLen: 7))
        let (len, _, _) = cache.findPrefix([1, 2, 3, 4, 5, 6, 7, 8, 9])
        #expect(len == 7)
        #expect(cache.count == 3)
    }

    @Test("insert with debugLogging exercises log branches")
    func insertWithDebugLogging() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 2, debugLogging: true)
        cache.insert(tokens: [1, 2], layerStates: fakeLayers(seqLen: 2))
        cache.insert(tokens: [1, 3], layerStates: fakeLayers(seqLen: 2))
        cache.insert(tokens: [5, 6], layerStates: fakeLayers(seqLen: 2))
        cache.insert(tokens: [5, 6], layerStates: fakeLayers(seqLen: 2))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - LRU Eviction
    // ═══════════════════════════════════════════════════════════════════

    @Test("LRU eviction at capacity evicts oldest")
    func lruEviction() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 3)
        cache.insert(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        cache.insert(tokens: [2], layerStates: fakeLayers(seqLen: 1))
        cache.insert(tokens: [3], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 3)
        cache.insert(tokens: [4], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 3)
        let (len1, _, _) = cache.findPrefix([1])
        #expect(len1 == 0)
        let (len4, s4, _) = cache.findPrefix([4])
        #expect(len4 == 1)
        #expect(s4 != nil)
    }

    @Test("Touch via findPrefix prevents eviction")
    func touchPreventsEviction() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 3)
        cache.insert(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        cache.insert(tokens: [2], layerStates: fakeLayers(seqLen: 1))
        cache.insert(tokens: [3], layerStates: fakeLayers(seqLen: 1))
        let _ = cache.findPrefix([1])
        cache.insert(tokens: [4], layerStates: fakeLayers(seqLen: 1))
        let (len1, s1, _) = cache.findPrefix([1])
        #expect(len1 == 1)
        #expect(s1 != nil)
        let (len2, _, _) = cache.findPrefix([2])
        #expect(len2 == 0)
    }

    @Test("Multiple evictions on single insert (maxEntries=1)")
    func multipleEvictions() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 1)
        cache.insert(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 1)
        cache.insert(tokens: [2], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 1)
        let (len1, _, _) = cache.findPrefix([1])
        #expect(len1 == 0)
        let (len2, _, _) = cache.findPrefix([2])
        #expect(len2 == 1)
    }

    @Test("Eviction with debugLogging")
    func evictionWithDebugLogging() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 1, debugLogging: true)
        cache.insert(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        cache.insert(tokens: [2], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 1)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - compactUpward
    // ═══════════════════════════════════════════════════════════════════

    @Test("compactUpward merges single-child node after sibling eviction")
    func compactMergesSingleChild() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 2)
        cache.insert(tokens: [1, 2, 3, 4, 5], layerStates: fakeLayers(seqLen: 5))
        cache.insert(tokens: [1, 2, 3, 9, 8], layerStates: fakeLayers(seqLen: 5))
        #expect(cache.count == 2)
        cache.insert(tokens: [7, 8, 9], layerStates: fakeLayers(seqLen: 3))
        #expect(cache.count == 2)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Invalidation and Count
    // ═══════════════════════════════════════════════════════════════════

    @Test("invalidateAll clears everything")
    func invalidateAll() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2], layerStates: fakeLayers(seqLen: 2))
        cache.insert(tokens: [3, 4], layerStates: fakeLayers(seqLen: 2))
        #expect(cache.count == 2)
        cache.invalidateAll()
        #expect(cache.count == 0)
        let (len, _, _) = cache.findPrefix([1, 2])
        #expect(len == 0)
    }

    @Test("invalidateAll with debugLogging")
    func invalidateAllWithDebug() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64, debugLogging: true)
        cache.insert(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        cache.invalidateAll()
        #expect(cache.count == 0)
    }

    @Test("Count tracks insertions, evictions, and invalidation")
    func countAccuracy() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 2)
        #expect(cache.count == 0)
        cache.insert(tokens: [1], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 1)
        cache.insert(tokens: [2], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 2)
        cache.insert(tokens: [3], layerStates: fakeLayers(seqLen: 1))
        #expect(cache.count == 2)
        cache.invalidateAll()
        #expect(cache.count == 0)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Conversation fork scenario
    // ═══════════════════════════════════════════════════════════════════

    @Test("Conversation fork: both branches survive and are findable")
    func conversationFork() {
        let cache = RadixTreeCache(modelID: "test", maxEntries: 64)
        cache.insert(tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     layerStates: fakeLayers(seqLen: 10))
        cache.insert(tokens: [1, 2, 3, 4, 5, 20, 21, 22],
                     layerStates: fakeLayers(seqLen: 8))
        let (lenA, _, _) = cache.findPrefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        #expect(lenA == 10)
        let (lenB, _, _) = cache.findPrefix([1, 2, 3, 4, 5, 20, 21, 22])
        #expect(lenB == 8)
        let (lenC, _, _) = cache.findPrefix([1, 2, 3, 4, 5, 99])
        #expect(lenC == 0)
    }
}
