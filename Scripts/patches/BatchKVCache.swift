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

    /// Per-sequence logical offsets [B], int32
    var perSeqOffset: MLXArray

    /// Global write cursor (same for all sequences)
    var _idx: Int = 0

    /// Pre-allocation step size
    var step: Int = 256

    /// Current batch size
    var batchSize: Int

    /// Cached flags — set at merge/extend/filter time, never eval during decode.
    /// update() preserves both (uniform add keeps equality; padding unchanged).
    var _allOffsetsEqual: Bool?
    var _zeroPadding: Bool?

    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }
    public override var offsetArray: MLXArray? { perSeqOffset }
    public override var maxSize: Int? { nil }
    public override var isTrimmable: Bool { true }

    public override var allOffsetsEqual: Bool {
        if let cached = _allOffsetsEqual { return cached }
        let result = batchSize <= 1 || (perSeqOffset .== perSeqOffset[0]).all().item(Bool.self)
        _allOffsetsEqual = result
        return result
    }

    /// True when all leftPadding entries are zero (same-length sequences).
    var zeroPadding: Bool {
        if let cached = _zeroPadding { return cached }
        let result = batchSize <= 1 || leftPadding.max().item(Int32.self) == 0
        _zeroPadding = result
        return result
    }

    public override func syncPerSeqOffsets(_ offsets: MLXArray, allEqual: Bool? = nil) {
        perSeqOffset = offsets
        _allOffsetsEqual = allEqual  // propagate from source — no eval needed
    }

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
                if prev % step != 0 {
                    ek = ek[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([ek, newK], axis: 2)
                // Handle zero-width values (e.g. GLM-5 DSA indexer cache)
                if existingValues.dim(3) > 0 {
                    var ev = existingValues
                    if prev % step != 0 {
                        ev = ev[.ellipsis, ..<prev, 0...]
                    }
                    self.values = concatenated([ev, newV], axis: 2)
                } else {
                    self.values = MLXArray.zeros([newK.dim(0), newK.dim(1), self.keys!.dim(2), 0], dtype: existingValues.dtype)
                }
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        perSeqOffset = perSeqOffset + Int32(newTokens)
        _idx += newTokens

        self.keys![.ellipsis, prev ..< _idx, 0...] = newKeys
        if newValues.dim(3) > 0 {
            self.values![.ellipsis, prev ..< _idx, 0...] = newValues
        }

        let retKeys = self.keys![.ellipsis, ..<_idx, 0...]
        let retValues = self.values!.dim(3) > 0
            ? self.values![.ellipsis, ..<_idx, 0...]
            : MLXArray.zeros([retKeys.dim(0), self.values!.dim(1), _idx, 0], dtype: self.values!.dtype)
        return (retKeys, retValues)
    }

    // MARK: - State

    public override var state: [MLXArray] {
        get {
            guard let k = keys, let v = values else { return [] }
            let kSlice = k[.ellipsis, ..<_idx, 0...]
            let vSlice = v.dim(3) > 0
                ? v[.ellipsis, ..<_idx, 0...]
                : MLXArray.zeros([k.dim(0), v.dim(1), _idx, 0], dtype: v.dtype)
            return [
                kSlice,
                vSlice,
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
            _allOffsetsEqual = nil; _zeroPadding = nil
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

        if n == 1 {
            // Decode: single new token — causal constraint is automatic.
            // When all padding is zero (same-length sequences), return .none to match
            // the B=1 serial path, ensuring bit-identical SDPA computation.
            if zeroPadding { return .none }
            // Different-length sequences: need padding mask [B, 1, 1, totalLen]
            let keyIndices = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped([1, 1, totalLen])
            let padMask = keyIndices .>= leftPadding.reshaped([batchSize, 1, 1])
            return .array(padMask.expandedDimensions(axis: 1))
        }

        // Key indices: [1, 1, totalLen]
        let keyIndices = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped([1, 1, totalLen])
        // Padding mask: attend only to non-padding positions
        // leftPadding: [B] → [B, 1, 1]
        let padMask = keyIndices .>= leftPadding.reshaped([batchSize, 1, 1])

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
        _allOffsetsEqual = nil

        // Shift left to reduce padding waste
        let minPad = leftPadding.min().item(Int32.self)
        if minPad > 0 {
            let mp = Int(minPad)
            keys = keys![.ellipsis, mp..., 0...]
            // Handle zero-width values (e.g. GLM-5 DSA indexer cache)
            if let v = values, v.dim(3) > 0 {
                values = v[.ellipsis, mp..., 0...]
            }
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
                let kShape = [rk.dim(0), rk.dim(1), left, rk.dim(3)]
                let vShape = [rv.dim(0), rv.dim(1), left, rv.dim(3)]
                rk = concatenated([MLXArray.zeros(kShape, dtype: rk.dtype), rk], axis: 2)
                rv = concatenated([MLXArray.zeros(vShape, dtype: rv.dtype), rv], axis: 2)
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
        _allOffsetsEqual = nil
    }

    /// Extract a single sequence's cache (for saving to prefix cache or removal).
    /// Returns (keys, values, tokenCount) with shapes [1, heads, seqLen, dim].
    public func extract(_ index: Int) -> (keys: MLXArray, values: MLXArray, tokenCount: Int) {
        guard let k = keys, let v = values else {
            return (MLXArray([]), MLXArray([]), 0)
        }
        let padding = Int(leftPadding[index].item(Int32.self))
        let tokenCount = _idx - padding
        let ek = MLX.contiguous(k[index ..< index + 1, 0..., padding ..< _idx, 0...])
        // Handle zero-width values (e.g. GLM-5 DSA indexer cache stores keys only)
        let ev: MLXArray
        if v.dim(3) == 0 {
            ev = MLXArray.zeros([1, v.dim(1), tokenCount, 0], dtype: v.dtype)
        } else {
            ev = MLX.contiguous(v[index ..< index + 1, 0..., padding ..< _idx, 0...])
        }
        return (ek, ev, tokenCount)
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
                let kShape = [allKeys[i].dim(0), allKeys[i].dim(1), padAmount, allKeys[i].dim(3)]
                let vShape = [allValues[i].dim(0), allValues[i].dim(1), padAmount, allValues[i].dim(3)]
                let kPad = MLXArray.zeros(kShape, dtype: allKeys[i].dtype)
                let vPad = MLXArray.zeros(vShape, dtype: allValues[i].dtype)
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
        // Precompute flags at merge time — avoids eval during decode
        let allSameLen = Set(lengths).count <= 1
        batch._allOffsetsEqual = allSameLen
        batch._zeroPadding = padding.allSatisfy { $0 == 0 }
        return batch
    }
}

// MARK: - BatchRotatingKVCache

/// Batched rotating KV cache for concurrent generation with sliding-window attention.
/// Shape: [B, heads, seqLen, dim] with a fixed circular buffer of size `maxCacheSize`.
///
/// Uses the same left-padding strategy as `BatchKVCacheSimple` but wraps the write
/// cursor at `maxCacheSize` → `keep`, enabling batched decode for models like Gemma 4
/// that use `RotatingKVCache` on sliding-window attention layers.
///
/// Constraint: requires `keep == 0` so all sequences wrap simultaneously.
public class BatchRotatingKVCache: BaseKVCache {
    internal var keys: MLXArray?
    internal var values: MLXArray?

    /// Per-sequence left padding [B], int32
    var leftPadding: MLXArray

    /// Per-sequence logical offsets [B], int32 (initialized as -leftPadding)
    var perSeqOffset: MLXArray

    /// Global write cursor within the circular buffer (same for all sequences)
    var _idx: Int = 0

    /// Max logical offset across the batch. May exceed `maxCacheSize` after wrap.
    private var _offset: Int = 0

    /// Pre-allocation step size (used before buffer reaches maxCacheSize)
    public static let preAllocStep: Int = 256

    /// Fixed circular buffer size (sliding window)
    public let maxCacheSize: Int

    /// Sink tokens preserved at start of buffer (must be 0 for batching)
    public let keep: Int

    /// Current batch size
    public var batchSize: Int

    /// Cached flags — set at merge/extend/filter, never eval during decode.
    private var _allOffsetsEqual: Bool?
    private var _zeroPadding: Bool?

    public override var offset: Int {
        get { _offset }
        set { _offset = newValue }
    }
    /// Per-sequence RoPE offsets — tracks each sequence's real token count.
    public override var offsetArray: MLXArray? { perSeqOffset }
    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { _offset < maxCacheSize }

    public override var allOffsetsEqual: Bool {
        if let cached = _allOffsetsEqual { return cached }
        let result = batchSize <= 1 || (perSeqOffset .== perSeqOffset[0]).all().item(Bool.self)
        _allOffsetsEqual = result
        return result
    }

    var zeroPadding: Bool {
        if let cached = _zeroPadding { return cached }
        let result = batchSize <= 1 || leftPadding.max().item(Int32.self) == 0
        _zeroPadding = result
        return result
    }

    public override func syncPerSeqOffsets(_ offsets: MLXArray, allEqual: Bool? = nil) {
        perSeqOffset = offsets
        _allOffsetsEqual = allEqual
    }

    public init(batchSize: Int, leftPadding: [Int], maxCacheSize: Int, keep: Int = 0) {
        precondition(keep == 0, "BatchRotatingKVCache requires keep=0 for synchronized wrapping")
        self.batchSize = batchSize
        self.maxCacheSize = maxCacheSize
        self.keep = keep
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.perSeqOffset = MLXArray(leftPadding.map { -Int32($0) })
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    private struct SequenceState {
        let keys: MLXArray
        let values: MLXArray
        let storedCount: Int
        let actualOffset: Int32
    }

    /// Public accessor for current per-sequence padding (used by per-sequence SDPA path).
    public func currentPaddingArray() -> MLXArray { currentPadding() }

    /// Current per-sequence padding count.
    ///
    /// Pre-wrap, padding is the fixed left pad introduced during merge.
    /// Once the shared buffer reaches `maxCacheSize`, shorter sequences keep a
    /// rotating gap until they individually fill the window. At that point the
    /// pad count is derived from `perSeqOffset`, not from the original prefix pad.
    private func currentPadding(afterAdding tokens: Int = 0) -> MLXArray {
        guard _offset >= maxCacheSize else { return leftPadding }
        let totalOffsets = perSeqOffset + Int32(tokens)
        let stored = minimum(totalOffsets, Int32(maxCacheSize))
        return maximum(MLXArray(Int32(maxCacheSize)) - stored, MLXArray(Int32(0)))
    }

    private func materializeSequence(at index: Int, padding: MLXArray) -> SequenceState {
        guard let keys, let values else {
            return SequenceState(
                keys: MLXArray([]),
                values: MLXArray([]),
                storedCount: 0,
                actualOffset: Int32(0)
            )
        }

        let seqKeys = keys[index ..< index + 1]
        let seqValues = values[index ..< index + 1]
        let orderedKeys: MLXArray
        let orderedValues: MLXArray
        if _offset < maxCacheSize {
            orderedKeys = seqKeys[.ellipsis, ..<_idx, 0...]
            orderedValues = seqValues[.ellipsis, ..<_idx, 0...]
        } else {
            orderedKeys = batchTemporalOrder(seqKeys)
            orderedValues = batchTemporalOrder(seqValues)
        }

        let pad = Int(padding[index].item(Int32.self))
        let start = Swift.min(pad, orderedKeys.dim(2))
        let validCount = Swift.max(0, orderedKeys.dim(2) - start)
        let validKeys =
            validCount > 0
            ? MLX.contiguous(orderedKeys[.ellipsis, start..., 0...])
            : MLXArray.zeros([1, orderedKeys.dim(1), 0, orderedKeys.dim(3)], dtype: orderedKeys.dtype)
        let validValues =
            validCount > 0
            ? MLX.contiguous(orderedValues[.ellipsis, start..., 0...])
            : MLXArray.zeros([1, orderedValues.dim(1), 0, orderedValues.dim(3)], dtype: orderedValues.dtype)

        return SequenceState(
            keys: validKeys,
            values: validValues,
            storedCount: validCount,
            actualOffset: perSeqOffset[index].item(Int32.self)
        )
    }

    private func materializeAllSequences() -> [SequenceState] {
        guard keys != nil, values != nil, batchSize > 0 else { return [] }
        let padding = currentPadding()
        return (0..<batchSize).map { materializeSequence(at: $0, padding: padding) }
    }

    private func rebuildDenseLayout(from sequences: [SequenceState], targetLen: Int) {
        guard !sequences.isEmpty else {
            keys = nil
            values = nil
            leftPadding = MLXArray([])
            perSeqOffset = MLXArray([])
            _allOffsetsEqual = nil; _zeroPadding = nil
            _idx = 0
            _offset = 0
            batchSize = 0
            return
        }

        var rebuiltKeys = [MLXArray]()
        var rebuiltValues = [MLXArray]()
        var rebuiltPadding = [Int32]()
        var rebuiltOffsets = [Int32]()

        for sequence in sequences {
            let pad = Swift.max(0, targetLen - sequence.storedCount)
            rebuiltPadding.append(Int32(pad))
            rebuiltOffsets.append(sequence.actualOffset)

            if pad > 0 {
                let keyPad = MLXArray.zeros(
                    [sequence.keys.dim(0), sequence.keys.dim(1), pad, sequence.keys.dim(3)],
                    dtype: sequence.keys.dtype
                )
                let valuePad = MLXArray.zeros(
                    [sequence.values.dim(0), sequence.values.dim(1), pad, sequence.values.dim(3)],
                    dtype: sequence.values.dtype
                )
                rebuiltKeys.append(concatenated([keyPad, sequence.keys], axis: 2))
                rebuiltValues.append(concatenated([valuePad, sequence.values], axis: 2))
            } else {
                rebuiltKeys.append(sequence.keys)
                rebuiltValues.append(sequence.values)
            }
        }

        keys = concatenated(rebuiltKeys, axis: 0)
        values = concatenated(rebuiltValues, axis: 0)
        leftPadding = MLXArray(rebuiltPadding)
        perSeqOffset = MLXArray(rebuiltOffsets)
        _idx = targetLen
        let maxOffset = sequences.map { Swift.max(0, Int($0.actualOffset)) }.max() ?? 0
        _offset = Swift.max(targetLen, maxOffset)
        batchSize = sequences.count
        _allOffsetsEqual = nil
    }

    // MARK: - Update

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let S = newKeys.dim(2)
        if S == 1 {
            return updateInPlace(keys: newKeys, values: newValues)
        } else {
            return updateConcat(keys: newKeys, values: newValues)
        }
    }

    /// Single-token decode: circular in-place write at `_idx`, wraps at maxCacheSize.
    private func updateInPlace(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" && batchSize > 1 {
            print("[BatchRotatingKV] updateInPlace B=\(batchSize) newKeys=\(newKeys.shape) _idx=\(_idx) _offset=\(_offset) maxCache=\(maxCacheSize) existing=\(keys?.shape.description ?? "nil")")
        }
        let previousOffset = _offset
        let B = newKeys.dim(0)
        let nKVHeads = newKeys.dim(1)
        let kHeadDim = newKeys.dim(3)
        let vHeadDim = newValues.dim(3)

        // Grow buffer until maxCacheSize
        if self.keys == nil || (_idx >= (self.keys?.dim(2) ?? 0) && (self.keys?.dim(2) ?? 0) < maxCacheSize) {
            let newSize = Swift.min(Self.preAllocStep, maxCacheSize - _idx)
            let kZeros = MLXArray.zeros([B, nKVHeads, newSize, kHeadDim], dtype: newKeys.dtype)
            let vZeros = MLXArray.zeros([B, nKVHeads, newSize, vHeadDim], dtype: newValues.dtype)

            if let existingK = self.keys, let existingV = self.values {
                self.keys = concatenated([existingK, kZeros], axis: 2)
                self.values = concatenated([existingV, vZeros], axis: 2)
            } else {
                self.keys = kZeros
                self.values = vZeros
            }
        }

        // Trim to maxCacheSize if over-allocated
        if let k = self.keys, k.dim(2) > maxCacheSize {
            self.keys = k[.ellipsis, ..<maxCacheSize, 0...]
            self.values = self.values![.ellipsis, ..<maxCacheSize, 0...]
            _idx = maxCacheSize
        }

        // Wrap: all sequences wrap simultaneously
        if _idx >= maxCacheSize {
            _idx = keep
        }

        // Write at current position for all B sequences
        self.keys![.ellipsis, _idx ..< (_idx + 1), 0...] = newKeys
        self.values![.ellipsis, _idx ..< (_idx + 1), 0...] = newValues
        _offset = Swift.max(previousOffset + 1, _idx + 1)
        _idx += 1
        perSeqOffset = perSeqOffset + Int32(1)

        // Return appropriate slice
        if _offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<_offset, 0...],
                self.values![.ellipsis, ..<_offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    /// Multi-token prefill: identical allocate-and-write pattern as BatchKVCacheSimple.update().
    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" && batchSize > 1 {
            print("[BatchRotatingKV] updateConcat B=\(batchSize) newKeys=\(newKeys.shape) _idx=\(_idx) _offset=\(_offset) maxCache=\(maxCacheSize) existing=\(keys?.shape.description ?? "nil")")
        }
        let prev = _idx
        let newTokens = newKeys.dim(2)

        if self.keys == nil || (prev + newTokens) > self.keys!.dim(2) {
            let B = newKeys.dim(0)
            let kvHeads = newKeys.dim(1)
            let kHeadDim = newKeys.dim(3)
            let vHeadDim = newValues.dim(3)
            let nSteps = (Self.preAllocStep + newTokens - 1) / Self.preAllocStep
            let kShape = [B, kvHeads, nSteps * Self.preAllocStep, kHeadDim]
            let vShape = [B, kvHeads, nSteps * Self.preAllocStep, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            if let existingKeys = self.keys, let existingValues = self.values {
                var ek = existingKeys
                if prev % Self.preAllocStep != 0 {
                    ek = ek[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([ek, newK], axis: 2)
                var ev = existingValues
                if prev % Self.preAllocStep != 0 {
                    ev = ev[.ellipsis, ..<prev, 0...]
                }
                self.values = concatenated([ev, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        _offset += newTokens
        _idx = prev + newTokens
        perSeqOffset = perSeqOffset + Int32(newTokens)

        self.keys![.ellipsis, prev ..< _idx, 0...] = newKeys
        self.values![.ellipsis, prev ..< _idx, 0...] = newValues

        return (self.keys![.ellipsis, ..<_idx, 0...], self.values![.ellipsis, ..<_idx, 0...])
    }

    /// Reconstruct temporal order from circular buffer (operates on axis 2).
    private func batchTemporalOrder(_ array: MLXArray) -> MLXArray {
        if _idx == array.dim(2) {
            return array
        } else if _idx < _offset {
            // Buffer has wrapped: [keep..., after_wrap..., before_wrap...]
            return concatenated([
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, _idx..., 0...],
                array[.ellipsis, keep ..< _idx, 0...],
            ], axis: 2)
        } else {
            return array[.ellipsis, ..<_idx, 0...]
        }
    }

    /// Trim oldest non-keep tokens from the sequence dimension.
    private func batchTrim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append { toCat.append(append) }
        return concatenated(toCat, axis: 2)
    }

    // MARK: - State

    public override var state: [MLXArray] {
        get {
            guard let k = keys, let v = values else { return [] }
            let kSlice = _offset < k.dim(2) ? k[.ellipsis, ..<_offset, 0...] : k
            let vSlice = _offset < v.dim(2) ? v[.ellipsis, ..<_offset, 0...] : v
            return [kSlice, vSlice, perSeqOffset, currentPadding()]
        }
        set {
            guard newValue.count == 4 else {
                keys = nil; values = nil; _offset = 0; _idx = 0
                return
            }
            keys = newValue[0]
            values = newValue[1]
            perSeqOffset = newValue[2]
            leftPadding = newValue[3]
            _offset = newValue[0].dim(2)
            _idx = _offset
            _allOffsetsEqual = nil; _zeroPadding = nil
        }
    }

    public override var metaState: [String] {
        get { ["\(batchSize)", "\(_offset)", "\(_idx)", "\(maxCacheSize)", "\(keep)"] }
        set { }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard _offset < maxCacheSize else { return 0 }
        let trimmed = Swift.min(_offset, n)
        _offset -= trimmed
        _idx -= trimmed
        perSeqOffset = perSeqOffset - Int32(trimmed)
        return trimmed
    }

    // MARK: - Mask

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // makeMask is called BEFORE update().
        // Pre-wrap: key length = _idx + n (cache is growing).
        // Post-wrap: key length = maxCacheSize (buffer is full, _idx is circular).
        let totalLen = _offset >= maxCacheSize ? maxCacheSize : _idx + n
        guard batchSize > 0 && totalLen > 0 else { return .none }
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" && batchSize > 1 {
            print("[BatchRotatingKV] makeMask B=\(batchSize) n=\(n) _idx=\(_idx) _offset=\(_offset) totalLen=\(totalLen) window=\(windowSize?.description ?? "nil")")
        }

        if n == 1 {
            // Decode: single token
            if _offset < maxCacheSize {
                // Pre-wrap: zero padding → .none matches B=1 serial SDPA path
                if zeroPadding { return .none }
                let currentPad = currentPadding()
                let keyIndices = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped([1, 1, totalLen])
                let padMask = keyIndices .>= currentPad.reshaped([batchSize, 1, 1])
                return .array(padMask.expandedDimensions(axis: 1))
            }

            // Post-wrap: buffer is full, need rotation-aware masking
            let keyIndices = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped([1, 1, totalLen])
            let currentPad = currentPadding()
            var decodePadMask = keyIndices .>= currentPad.reshaped([batchSize, 1, 1])
            var writeIndex = _idx
            if writeIndex >= maxCacheSize { writeIndex = keep }
            let remainingPad = currentPadding(afterAdding: 1).reshaped([batchSize, 1, 1])
            let rotatedIndices =
                (keyIndices - Int32(writeIndex + 1) + Int32(maxCacheSize))
                % Int32(maxCacheSize)
            decodePadMask = rotatedIndices .>= remainingPad

            if let windowSize, maxCacheSize > windowSize {
                // Buffer has wrapped AND window < maxCacheSize: need rotation-aware mask
                let currentIdx = writeIndex
                let maskSize = maxCacheSize
                let windowMask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
                let rolledMask = roll(windowMask, shift: currentIdx + 1)
                    .reshaped([1, 1, maskSize])

                // Combine: per-sequence padding AND rotation window
                let combined = rolledMask .&& decodePadMask
                return .array(combined.expandedDimensions(axis: 1))
            }
            // Post-wrap, window == maxCacheSize: just padding mask
            return .array(decodePadMask.expandedDimensions(axis: 1))
        }

        let keyIndices = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped([1, 1, totalLen])
        let currentPad = currentPadding()
        let padMask = keyIndices .>= currentPad.reshaped([batchSize, 1, 1])

        // Prefill: causal + padding (+ optional window)
        let queryStart = Int32(_idx)
        let queryIndices = (MLXArray(Int32(0) ..< Int32(n)) + queryStart).reshaped([1, n, 1])
        let cappedOffset = Swift.min(maxCacheSize - 1, _offset)
        let cappedKeyIndices = MLXArray(Int32(0) ..< Int32(cappedOffset + n)).reshaped([1, 1, cappedOffset + n])

        var mask = (queryIndices .>= cappedKeyIndices)
        if let windowSize {
            mask = mask .&& (cappedKeyIndices .>= (queryIndices - Int32(windowSize)))
        }

        // Apply per-sequence padding
        if totalLen == cappedOffset + n {
            mask = mask .&& padMask
        }

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
        _allOffsetsEqual = nil

        // Shift left to reduce padding waste (only before buffer wraps)
        if _offset < maxCacheSize {
            let minPad = leftPadding.min().item(Int32.self)
            if minPad > 0 {
                let mp = Int(minPad)
                keys = keys![.ellipsis, mp..., 0...]
                values = values![.ellipsis, mp..., 0...]
                _idx -= mp
                _offset -= mp
                leftPadding = leftPadding - minPad
            }
        } else {
            let sequences = materializeAllSequences()
            rebuildDenseLayout(from: sequences, targetLen: maxCacheSize)
        }
    }

    /// Merge another BatchRotatingKVCache into this one (extend batch).
    public func extend(with other: BatchRotatingKVCache) {
        precondition(maxCacheSize == other.maxCacheSize && keep == other.keep,
                     "Cannot extend BatchRotatingKVCache with different maxCacheSize/keep")

        let otherSequences = other.materializeAllSequences()
        guard !otherSequences.isEmpty else { return }

        let selfSequences = materializeAllSequences()
        let combined = selfSequences + otherSequences
        let targetLen =
            (_offset >= maxCacheSize || other._offset >= maxCacheSize)
            ? maxCacheSize
            : (combined.map(\.storedCount).max() ?? 0)
        rebuildDenseLayout(from: combined, targetLen: targetLen)
    }

    /// Extract a single sequence's cache for prefix save.
    /// Returns (keys, values, tokenCount) with shapes [1, heads, seqLen, dim].
    public func extract(_ index: Int) -> (keys: MLXArray, values: MLXArray, tokenCount: Int) {
        guard keys != nil, values != nil else {
            return (MLXArray([]), MLXArray([]), 0)
        }
        let sequence = materializeSequence(at: index, padding: currentPadding())
        return (sequence.keys, sequence.values, sequence.storedCount)
    }

    /// Create a BatchRotatingKVCache by merging individual RotatingKVCache objects.
    public static func merge(_ caches: [KVCache]) -> BatchRotatingKVCache {
        let B = caches.count

        // Extract config from first cache
        guard let firstRC = caches.first as? RotatingKVCache else {
            fatalError("BatchRotatingKVCache.merge requires RotatingKVCache inputs")
        }
        let maxSize = firstRC.cacheSize
        let keepVal = firstRC.keepCount

        // Extract state per cache — nil for empty/fresh caches
        var perCacheState: [(keys: MLXArray, values: MLXArray, storedLength: Int, actualOffset: Int)?] = []
        for cache in caches {
            let st = cache.state
            if st.count >= 2, st[0].dim(2) > 0 {
                perCacheState.append((
                    keys: st[0],
                    values: st[1],
                    storedLength: st[0].dim(2),
                    actualOffset: cache.offset
                ))
            } else {
                perCacheState.append(nil)
            }
        }

        guard perCacheState.contains(where: { $0 != nil }) else {
            return BatchRotatingKVCache(
                batchSize: B, leftPadding: Array(repeating: 0, count: B),
                maxCacheSize: maxSize, keep: keepVal
            )
        }

        // Use first non-nil cache for dtype/shape reference
        let refState = perCacheState.first(where: { $0 != nil })!!
        let maxLen = Swift.min(perCacheState.compactMap { $0?.storedLength }.max()!, maxSize)
        var padding = [Int]()
        var paddedKeys = [MLXArray]()
        var paddedValues = [MLXArray]()

        for i in 0 ..< B {
            let seqLen: Int
            var k: MLXArray
            var v: MLXArray

            if let state = perCacheState[i] {
                seqLen = Swift.min(state.storedLength, maxSize)
                k = state.keys
                v = state.values
                if k.dim(2) > maxSize {
                    k = k[.ellipsis, (k.dim(2) - maxSize)..., 0...]
                    v = v[.ellipsis, (v.dim(2) - maxSize)..., 0...]
                }
            } else {
                // Empty cache — create zero arrays matching reference shape
                seqLen = 0
                k = MLXArray.zeros([1, refState.keys.dim(1), 0, refState.keys.dim(3)], dtype: refState.keys.dtype)
                v = MLXArray.zeros([1, refState.values.dim(1), 0, refState.values.dim(3)], dtype: refState.values.dtype)
            }

            let padAmount = maxLen - seqLen
            padding.append(padAmount)

            if padAmount > 0 {
                let kPad = MLXArray.zeros([k.dim(0), k.dim(1), padAmount, k.dim(3)], dtype: k.dtype)
                let vPad = MLXArray.zeros([v.dim(0), v.dim(1), padAmount, v.dim(3)], dtype: v.dtype)
                paddedKeys.append(concatenated([kPad, k], axis: 2))
                paddedValues.append(concatenated([vPad, v], axis: 2))
            } else {
                paddedKeys.append(k)
                paddedValues.append(v)
            }
        }

        let batch = BatchRotatingKVCache(
            batchSize: B, leftPadding: padding,
            maxCacheSize: maxSize, keep: keepVal
        )
        batch.keys = concatenated(paddedKeys, axis: 0)
        batch.values = concatenated(paddedValues, axis: 0)
        batch._idx = maxLen
        batch._offset = maxLen
        batch.perSeqOffset = MLXArray(perCacheState.map { Int32($0?.actualOffset ?? 0) })
        let offsets = perCacheState.compactMap { $0?.actualOffset }
        batch._allOffsetsEqual = Set(offsets).count <= 1
        batch._zeroPadding = padding.allSatisfy { $0 == 0 }
        return batch
    }
}

// MARK: - BatchCacheList

/// Batched wrapper for CacheList — holds a batched version of each sub-cache.
/// Models using CacheList (GLM-5, FalconH1, BaichuanM1) have multiple sub-caches
/// per layer (e.g. MLA attention + DSA indexer). This class batches each sub-cache
/// independently so the model's forward pass can access them via subscript.
///
/// Subclasses CacheList so that model code doing `cache as? CacheList` succeeds
/// and subscript access returns the batched sub-caches transparently.
public class BatchCacheList: CacheList {
    public private(set) var batchedCaches: [KVCache]

    public init(batchedCaches: [KVCache]) {
        self.batchedCaches = batchedCaches
        super.init(caches: batchedCaches)
    }

    /// Extract a single sequence's CacheList from the batched structure.
    public func extract(_ index: Int) -> CacheList {
        let extracted: [KVCache] = batchedCaches.map { subCache in
            if let bkv = subCache as? BatchKVCacheSimple {
                let (k, v, _) = bkv.extract(index)
                let individual = KVCacheSimple()
                individual.state = [k, v]
                return individual as KVCache
            } else if let ac = subCache as? ArraysCache {
                let individual = MambaCache()
                if !ac.state.isEmpty {
                    individual.state = ac.state.map { $0[index ..< index + 1] }
                }
                return individual as KVCache
            } else {
                return subCache
            }
        }
        return CacheList(caches: extracted)
    }

    /// Create a BatchCacheList from individual CacheLists.
    public static func merge(_ cacheLists: [KVCache], leftPadding: [Int]) -> BatchCacheList {
        guard let first = cacheLists.first as? CacheList else {
            fatalError("BatchCacheList.merge requires CacheList inputs")
        }
        let subCount = first.count
        let batchedSubs: [KVCache] = (0..<subCount).map { subIdx in
            let subCaches = cacheLists.map { ($0 as! CacheList)[subIdx] }
            if subCaches.allSatisfy({ $0 is ArraysCache }) {
                let batched = MambaCache(leftPadding: leftPadding)
                // Merge states: stack along batch dimension
                if let first = subCaches.first, !first.state.isEmpty {
                    batched.state = (0..<first.state.count).map { stateIdx in
                        concatenated(subCaches.map { $0.state[stateIdx] }, axis: 0)
                    }
                }
                return batched as KVCache
            } else {
                return BatchKVCacheSimple.merge(subCaches) as KVCache
            }
        }
        return BatchCacheList(batchedCaches: batchedSubs)
    }

    /// Create a BatchCacheList for prefill (empty, with left padding).
    public static func forPrefill(template: CacheList, batchSize: Int, leftPadding: [Int]) -> BatchCacheList {
        let batchedSubs: [KVCache] = (0..<template.count).map { subIdx in
            let subCache = template[subIdx]
            if subCache is ArraysCache {
                return MambaCache(leftPadding: leftPadding) as KVCache
            } else {
                return BatchKVCacheSimple(batchSize: batchSize, leftPadding: leftPadding) as KVCache
            }
        }
        return BatchCacheList(batchedCaches: batchedSubs)
    }
}
