// Sources/MacLocalAPI/Models/RadixTreeCache.swift
import Foundation
import MLX
import MLXLMCommon

/// A single block of cached KV state for all layers at a contiguous range of tokens.
final class KVCacheEntry: @unchecked Sendable {
    let tokens: [Int]
    /// Per-layer KV cache state arrays (saved via cache.state)
    var layerStates: [[MLXArray]]
    var lastAccessTime: UInt64

    init(tokens: [Int], layerStates: [[MLXArray]]) {
        self.tokens = tokens
        self.layerStates = layerStates
        self.lastAccessTime = mach_absolute_time()
    }

    func touch() { lastAccessTime = mach_absolute_time() }
}

/// Radix tree node. Each edge is labeled with a token subsequence.
final class RadixNode: @unchecked Sendable {
    var children: [Int: RadixNode] = [:]  // keyed by first token of edge
    var edgeTokens: [Int]                  // full edge label (token sequence)
    var cacheEntry: KVCacheEntry?          // non-nil for nodes with cached KV state
    weak var parent: RadixNode?

    init(edgeTokens: [Int] = [], parent: RadixNode? = nil) {
        self.edgeTokens = edgeTokens
        self.parent = parent
    }

    var isLeaf: Bool { children.isEmpty }
    var hasCachedState: Bool { cacheEntry != nil }
}

/// Radix tree for multi-slot KV cache prefix sharing.
/// Replaces PromptCacheBox with multi-request prefix matching.
/// NOT internally synchronized — callers must ensure serial access
/// (e.g., within container.perform {} which holds an async mutex).
final class RadixTreeCache: @unchecked Sendable {
    private let root = RadixNode()
    private let modelID: String
    private let maxEntries: Int
    private let debugLogging: Bool
    private var entryCount = 0

    init(modelID: String, maxEntries: Int = 64, debugLogging: Bool = false) {
        self.modelID = modelID
        self.maxEntries = maxEntries
        self.debugLogging = debugLogging
    }

    /// Find longest cached prefix for the given token sequence.
    /// Returns (matched token count, per-layer KV states for the matched prefix).
    func findPrefix(_ tokens: [Int]) -> (prefixLen: Int, layerStates: [[MLXArray]]?) {
        var node = root
        var matched = 0
        var lastCachedNode: RadixNode? = nil
        var lastCachedLen = 0

        while matched < tokens.count {
            let nextToken = tokens[matched]
            guard let child = node.children[nextToken] else {
                if debugLogging {
                    let childKeys = node.children.keys.sorted()
                    print("[PrefixCache] Radix traversal: matched \(matched) tokens, no child for token \(nextToken) at pos \(matched) (children: \(childKeys))")
                }
                break
            }

            // Match edge tokens
            let edge = child.edgeTokens
            var edgePos = 0
            while edgePos < edge.count && matched < tokens.count {
                if tokens[matched] != edge[edgePos] { break }
                edgePos += 1
                matched += 1
            }

            // Track cached state even on partial edge match — the caller
            // trims KV state to the matched prefix length, so we can use
            // a cache entry that covers more tokens than we matched.
            if child.hasCachedState && matched > lastCachedLen {
                lastCachedNode = child
                lastCachedLen = matched
            }

            if edgePos < edge.count {
                // Partial edge match — stop traversal but keep any cached state found above
                if debugLogging {
                    print("[PrefixCache] Radix traversal: matched \(matched) tokens, diverged at pos \(matched): input=\(tokens[matched]) vs cached=\(edge[edgePos])")
                }
                break
            }

            // Full edge matched
            node = child
        }

        if let cached = lastCachedNode {
            cached.cacheEntry?.touch()
            if debugLogging {
                let entryTokenCount = cached.cacheEntry?.tokens.count ?? 0
                print("[PrefixCache] Radix hit: \(lastCachedLen)/\(tokens.count) tokens matched (entry has \(entryTokenCount) tokens)")
            }
            return (lastCachedLen, cached.cacheEntry?.layerStates)
        }

        if debugLogging {
            if matched > 0 {
                print("[PrefixCache] Radix miss: traversed \(matched) tokens but no cached node (/\(tokens.count) input tokens)")
            } else {
                print("[PrefixCache] Radix miss for \(tokens.count) tokens (no prefix match)")
            }
        }
        return (0, nil)
    }

    /// Insert a cached prefix into the tree.
    /// layerStates: per-layer KV cache state (from cache[i].state).
    func insert(tokens: [Int], layerStates: [[MLXArray]]) {
        guard !tokens.isEmpty else { return }

        // Evict if at capacity
        while entryCount >= maxEntries {
            evictLRU()
        }

        var node = root
        var pos = 0

        while pos < tokens.count {
            let nextToken = tokens[pos]

            guard let child = node.children[nextToken] else {
                // No matching child — insert remaining tokens as new edge
                let newNode = RadixNode(edgeTokens: Array(tokens[pos...]), parent: node)
                newNode.cacheEntry = KVCacheEntry(tokens: tokens, layerStates: layerStates)
                node.children[nextToken] = newNode
                entryCount += 1
                if debugLogging {
                    print("[PrefixCache] Radix insert: \(tokens.count) tokens, \(layerStates.count) layers (entries: \(entryCount))")
                }
                return
            }

            // Match edge tokens
            let edge = child.edgeTokens
            var edgePos = 0
            while edgePos < edge.count && pos < tokens.count && tokens[pos] == edge[edgePos] {
                edgePos += 1
                pos += 1
            }

            if edgePos < edge.count {
                // Partial edge match — split the edge
                let splitNode = RadixNode(edgeTokens: Array(edge[..<edgePos]), parent: node)
                child.edgeTokens = Array(edge[edgePos...])
                child.parent = splitNode
                splitNode.children[edge[edgePos]] = child
                node.children[nextToken] = splitNode

                if pos < tokens.count {
                    // Remaining tokens go as a new child of splitNode
                    let newNode = RadixNode(edgeTokens: Array(tokens[pos...]), parent: splitNode)
                    newNode.cacheEntry = KVCacheEntry(tokens: tokens, layerStates: layerStates)
                    splitNode.children[tokens[pos]] = newNode
                    entryCount += 1
                } else {
                    // Exact split point — cache lives on splitNode
                    splitNode.cacheEntry = KVCacheEntry(tokens: tokens, layerStates: layerStates)
                    entryCount += 1
                }
                if debugLogging {
                    print("[PrefixCache] Radix insert (split): \(tokens.count) tokens, \(layerStates.count) layers (entries: \(entryCount))")
                }
                return
            }

            // Full edge matched, continue to next node
            node = child
        }

        // Exact match — update existing node
        if node.cacheEntry == nil { entryCount += 1 }
        node.cacheEntry = KVCacheEntry(tokens: tokens, layerStates: layerStates)
        if debugLogging {
            print("[PrefixCache] Radix insert (update): \(tokens.count) tokens, \(layerStates.count) layers (entries: \(entryCount))")
        }
    }

    /// Evict the least-recently-used cache entry.
    private func evictLRU() {
        var oldest: RadixNode? = nil
        var oldestTime: UInt64 = .max

        func walk(_ node: RadixNode) {
            if let entry = node.cacheEntry, entry.lastAccessTime < oldestTime {
                oldest = node
                oldestTime = entry.lastAccessTime
            }
            for child in node.children.values { walk(child) }
        }
        walk(root)

        if let victim = oldest {
            victim.cacheEntry = nil
            entryCount -= 1
            // Compact: remove leaf nodes with no cache and no children
            compactUpward(victim)
            if debugLogging {
                print("[PrefixCache] Radix evict LRU (entries: \(entryCount))")
            }
        }
    }

    /// Remove empty leaf nodes upward and merge single-child nodes.
    private func compactUpward(_ node: RadixNode) {
        guard node.isLeaf, !node.hasCachedState, let parent = node.parent else { return }
        parent.children.removeValue(forKey: node.edgeTokens.first!)
        if parent.children.count == 1 && !parent.hasCachedState && parent.parent != nil {
            // Merge single child into parent. The grandparent's key for this node
            // is still correct — it maps to the first token of parent.edgeTokens,
            // which doesn't change (we only append to it).
            let onlyChild = parent.children.values.first!
            parent.edgeTokens += onlyChild.edgeTokens
            parent.children = onlyChild.children
            parent.cacheEntry = onlyChild.cacheEntry
            for child in parent.children.values { child.parent = parent }
        }
    }

    /// Invalidate all entries (e.g., on model change).
    func invalidateAll() {
        root.children.removeAll()
        entryCount = 0
        if debugLogging {
            print("[PrefixCache] Radix invalidated all entries")
        }
    }

    /// Current number of cached entries.
    var count: Int { entryCount }
}
