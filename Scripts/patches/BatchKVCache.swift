// Copyright © 2024 Apple Inc.
// Batched KV cache for concurrent sequence generation.
// Ported from Python mlx-lm BatchKVCache (cache.py).

import Foundation
import MLX
import MLXFast

/// Batched KV cache storing keys/values for B sequences in shape [B, heads, seqLen, dim].
/// Uses left-padding to handle variable-length sequences in a dense array.
///
/// All sequences share the same write cursor (`_idx`). Per-sequence logical offsets
/// are tracked via `leftPadding` — sequences that started later have more left padding.
///
/// During decode, all sequences advance by exactly 1 token per step, so `_idx` stays
/// synchronized. The causal mask prevents attention to left-padding positions.
public class BatchKVCacheSimple: BaseKVCache {
    internal var keys: MLXArray?
    internal var values: MLXArray?

    /// Per-sequence left padding [B], int32
    var leftPadding: MLXArray

    /// Per-sequence logical offsets [B], int32 (initialized as -leftPadding)
    var perSeqOffset: MLXArray

    /// Global write cursor (same for all sequences)
    var _idx: Int = 0

    /// Pre-allocation step size
    var step: Int = 256

    /// Current batch size
    var batchSize: Int

    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }
    public override var maxSize: Int? { nil }
    public override var isTrimmable: Bool { true }

    public init(batchSize: Int, leftPadding: [Int]) {
        self.batchSize = batchSize
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.perSeqOffset = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let prev = _idx
        let newTokens = newKeys.dim(2)

        if self.keys == nil || (prev + newTokens) > self.keys!.dim(2) {
            let B = newKeys.dim(0)
            let kvHeads = newKeys.dim(1)
            let kHeadDim = newKeys.dim(3)
            let vHeadDim = newValues.dim(3)
            let nSteps = (step + newTokens - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            if let existingKeys = self.keys, let existingValues = self.values {
                var ek = existingKeys
                var ev = existingValues
                if prev % step != 0 {
                    ek = ek[.ellipsis, ..<prev, 0...]
                    ev = ev[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([ek, newK], axis: 2)
                self.values = concatenated([ev, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        perSeqOffset = perSeqOffset + Int32(newTokens)
        _idx += newTokens

        self.keys![.ellipsis, prev ..< _idx, 0...] = newKeys
        self.values![.ellipsis, prev ..< _idx, 0...] = newValues

        return (self.keys![.ellipsis, ..<_idx, 0...],
                self.values![.ellipsis, ..<_idx, 0...])
    }

    // MARK: - State

    public override var state: [MLXArray] {
        get {
            guard let k = keys, let v = values else { return [] }
            return [
                k[.ellipsis, ..<_idx, 0...],
                v[.ellipsis, ..<_idx, 0...],
                perSeqOffset,
                leftPadding,
            ]
        }
        set {
            guard newValue.count == 4 else {
                keys = nil; values = nil; _idx = 0
                return
            }
            keys = newValue[0]
            values = newValue[1]
            perSeqOffset = newValue[2]
            leftPadding = newValue[3]
            _idx = newValue[0].dim(2)
        }
    }

    public override var metaState: [String] {
        get { ["\(batchSize)", "\(_idx)"] }
        set { }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = Swift.min(_idx, n)
        _idx -= trimmed
        perSeqOffset = perSeqOffset - Int32(trimmed)
        return trimmed
    }

    // MARK: - Mask

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // makeMask is called BEFORE update() in the model's forward pass.
        // update() will add n tokens, so the total key length after update = _idx + n.
        // The mask must cover all _idx + n positions to match SDPA's K/V dimensions.
        let totalLen = _idx + n
        guard batchSize > 0 && totalLen > 0 else { return .none }

        // Key indices: [1, 1, totalLen]
        let keyIndices = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped([1, 1, totalLen])
        // Padding mask: attend only to non-padding positions
        // leftPadding: [B] → [B, 1, 1]
        let padMask = keyIndices .>= leftPadding.reshaped([batchSize, 1, 1])

        if n == 1 {
            // Decode: single new token — causal constraint is automatic
            // Shape: [B, 1, 1, totalLen]
            return .array(padMask.expandedDimensions(axis: 1))
        }

        // Prefill: need causal + padding
        // Query indices: [1, n, 1] — queries cover positions [_idx, _idx+n)
        let queryStart = Int32(_idx)
        let queryIndices = (MLXArray(Int32(0) ..< Int32(n)) + queryStart).reshaped([1, n, 1])
        var mask = (queryIndices .>= keyIndices) .&& padMask

        if let windowSize {
            mask = mask .&& (keyIndices .>= (queryIndices - Int32(windowSize)))
        }

        // [B, 1, n, totalLen]
        return .array(mask.expandedDimensions(axis: 1))
    }

    // MARK: - Batch Operations

    /// Remove sequences — keep only specified batch indices.
    public func filter(_ keepIndices: [Int]) {
        guard let k = keys, let v = values else { return }

        let indices = MLXArray(keepIndices.map { Int32($0) })
        keys = k[indices]
        values = v[indices]
        perSeqOffset = perSeqOffset[indices]
        leftPadding = leftPadding[indices]
        batchSize = keepIndices.count

        // Shift left to reduce padding waste
        let minPad = leftPadding.min().item(Int32.self)
        if minPad > 0 {
            let mp = Int(minPad)
            keys = keys![.ellipsis, mp..., 0...]
            values = values![.ellipsis, mp..., 0...]
            _idx -= mp
            leftPadding = leftPadding - minPad
        }
    }

    /// Merge another BatchKVCacheSimple into this one (extend batch).
    public func extend(with other: BatchKVCacheSimple) {
        guard let otherKFull = other.keys, let otherVFull = other.values else { return }
        guard let selfKFull = self.keys, let selfVFull = self.values else {
            self.keys = otherKFull[.ellipsis, ..<other._idx, 0...]
            self.values = otherVFull[.ellipsis, ..<other._idx, 0...]
            self.leftPadding = other.leftPadding
            self.perSeqOffset = other.perSeqOffset
            self._idx = other._idx
            self.batchSize = other.batchSize
            return
        }

        // Trim to actual used length — pre-allocated step padding would cause
        // shape mismatch after rightJustify (self and other would get different dim(2)).
        let selfK = selfKFull[.ellipsis, ..<_idx, 0...]
        let selfV = selfVFull[.ellipsis, ..<_idx, 0...]
        let otherK = otherKFull[.ellipsis, ..<other._idx, 0...]
        let otherV = otherVFull[.ellipsis, ..<other._idx, 0...]

        let maxIdx = Swift.max(_idx, other._idx)

        func rightJustify(_ k: MLXArray, _ v: MLXArray, _ pad: MLXArray, _ off: MLXArray, idx: Int)
            -> (MLXArray, MLXArray, MLXArray, MLXArray)
        {
            var rk = k, rv = v, rp = pad, ro = off
            let left = maxIdx - idx
            if left > 0 {
                let shape = [rk.dim(0), rk.dim(1), left, rk.dim(3)]
                rk = concatenated([MLXArray.zeros(shape, dtype: rk.dtype), rk], axis: 2)
                rv = concatenated([MLXArray.zeros(shape, dtype: rv.dtype), rv], axis: 2)
                rp = rp + Int32(left)
            }
            return (rk, rv, rp, ro)
        }

        let (rk1, rv1, rp1, ro1) = rightJustify(selfK, selfV, leftPadding, perSeqOffset, idx: _idx)
        let (rk2, rv2, rp2, ro2) = rightJustify(otherK, otherV, other.leftPadding, other.perSeqOffset, idx: other._idx)

        keys = concatenated([rk1, rk2], axis: 0)
        values = concatenated([rv1, rv2], axis: 0)
        leftPadding = concatenated([rp1, rp2], axis: 0)
        perSeqOffset = concatenated([ro1, ro2], axis: 0)
        _idx = maxIdx
        batchSize += other.batchSize
    }

    /// Extract a single sequence's cache (for saving to prefix cache or removal).
    /// Returns (keys, values, tokenCount) with shapes [1, heads, seqLen, dim].
    public func extract(_ index: Int) -> (keys: MLXArray, values: MLXArray, tokenCount: Int) {
        guard let k = keys, let v = values else {
            return (MLXArray([]), MLXArray([]), 0)
        }
        let padding = Int(leftPadding[index].item(Int32.self))
        let ek = MLX.contiguous(k[index ..< index + 1, 0..., padding ..< _idx, 0...])
        let ev = MLX.contiguous(v[index ..< index + 1, 0..., padding ..< _idx, 0...])
        return (ek, ev, _idx - padding)
    }

    /// Create a BatchKVCacheSimple by merging individual per-sequence KVCache objects.
    /// Each cache should have shape [1, heads, seqLen, dim].
    public static func merge(_ caches: [KVCache]) -> BatchKVCacheSimple {
        let B = caches.count
        var allKeys: [MLXArray] = []
        var allValues: [MLXArray] = []
        var lengths: [Int] = []

        for cache in caches {
            let st = cache.state
            if st.count >= 2 {
                allKeys.append(st[0])
                allValues.append(st[1])
                lengths.append(cache.offset)
            }
        }

        guard !allKeys.isEmpty else {
            return BatchKVCacheSimple(batchSize: B, leftPadding: Array(repeating: 0, count: B))
        }

        let maxLen = lengths.max()!
        var padding = [Int]()
        var paddedKeys = [MLXArray]()
        var paddedValues = [MLXArray]()

        for i in 0 ..< B {
            let padAmount = maxLen - lengths[i]
            padding.append(padAmount)

            if padAmount > 0 {
                let shape = [allKeys[i].dim(0), allKeys[i].dim(1), padAmount, allKeys[i].dim(3)]
                let kPad = MLXArray.zeros(shape, dtype: allKeys[i].dtype)
                let vPad = MLXArray.zeros(shape, dtype: allValues[i].dtype)
                paddedKeys.append(concatenated([kPad, allKeys[i]], axis: 2))
                paddedValues.append(concatenated([vPad, allValues[i]], axis: 2))
            } else {
                paddedKeys.append(allKeys[i])
                paddedValues.append(allValues[i])
            }
        }

        let batch = BatchKVCacheSimple(batchSize: B, leftPadding: padding)
        batch.keys = concatenated(paddedKeys, axis: 0)
        batch.values = concatenated(paddedValues, axis: 0)
        batch._idx = maxLen
        batch.perSeqOffset = MLXArray(lengths.map { Int32($0) })
        return batch
    }
}
