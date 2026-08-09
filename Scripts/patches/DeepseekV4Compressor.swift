// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSV4 Compressor + Indexer + per-layer DeepseekV4Cache.
//
// These power the "compressed global context" path that augments the
// 128-token local sliding window with pooled summaries of older tokens.
// Every decoder layer with `compress_ratio > 0` (~41 of 43 layers in
// DSV4-Flash) carries a Compressor; layers with `compress_ratio == 4`
// ALSO carry an Indexer that picks the top-k most relevant pooled
// entries per query position.
//
// Reference:
//   - jang-tools/jang_tools/dsv4/mlx_model.py lines 410-489
//   - DSV4-RUNTIME-ARCHITECTURE.md §1 ("Compressor + Indexer")

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - DSV4 pool storage

/// Segmented affine storage for DSV4's compressor/indexer pools.
///
/// The first 2 MiB of each branch stays in its attention-ready dtype. Once a
/// branch crosses that byte threshold, rows are encoded independently in
/// bounded 64-row segments with UInt8 codes and fp16 scale/min metadata. The
/// full pool is materialized only as a temporary attention input; it is never
/// retained beside the encoded representation.
private final class DeepseekV4PoolStorage {
    static let hotByteLimit = 2 * 1024 * 1024
    static let segmentRows = 64
    static let groupSize = 32
    static let bits = 8

    let quantizationEnabled: Bool
    private(set) var hotPool: MLXArray?
    private(set) var quantizedSegments: [HybridPoolQuantizedSegment] = []
    private var emptySpec: (shape: [Int], dtype: DType)?

    init(quantizationEnabled: Bool) {
        self.quantizationEnabled = quantizationEnabled
    }

    var rowCount: Int {
        if let hotPool { return hotPool.dim(1) }
        return quantizedSegments.reduce(0) { $0 + $1.rowCount }
    }

    var retainedByteCount: Int {
        (hotPool?.nbytes ?? 0)
            + quantizedSegments.reduce(0) { $0 + $1.retainedByteCount }
    }

    var retainedArrays: [MLXArray] {
        if let hotPool { return [hotPool] }
        return quantizedSegments.flatMap { [$0.codes, $0.scales, $0.biases] }
    }

    func materialized() -> MLXArray? {
        if let hotPool { return hotPool }
        if quantizedSegments.isEmpty {
            if let emptySpec {
                return MLXArray.zeros(emptySpec.shape, dtype: emptySpec.dtype)
            }
            return nil
        }
        let parts = quantizedSegments.map(Self.dequantize)
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    func replace(with value: MLXArray?) {
        hotPool = nil
        quantizedSegments = []
        emptySpec = nil
        guard let value else { return }
        precondition(value.ndim == 3, "DSV4 pooled state must have shape [batch, rows, features]")
        if value.dim(1) == 0 {
            emptySpec = (value.shape, value.dtype)
        } else if !quantizationEnabled || value.nbytes <= Self.hotByteLimit {
            hotPool = value
        } else {
            replaceQuantized(with: value)
        }
    }

    func append(_ value: MLXArray) {
        guard value.ndim == 3, value.dim(1) > 0 else { return }
        emptySpec = nil

        if !quantizationEnabled {
            hotPool = hotPool.map { concatenated([$0, value], axis: 1) } ?? value
            return
        }

        if hotPool != nil || quantizedSegments.isEmpty {
            let combined = hotPool.map { concatenated([$0, value], axis: 1) } ?? value
            if combined.nbytes <= Self.hotByteLimit {
                hotPool = combined
            } else {
                replaceQuantized(with: combined)
            }
            return
        }

        var offset = 0
        let rows = value.dim(1)
        let tail = trailingPartialSegmentRows()
        if tail > 0 {
            let take = min(Self.segmentRows - tail, rows)
            quantizedSegments.append(
                Self.quantize(value[0..., 0..<take, 0...]))
            offset += take
            compactCompletedTail()
        }

        while rows - offset >= Self.segmentRows {
            quantizedSegments.append(
                Self.quantize(value[0..., offset..<(offset + Self.segmentRows), 0...]))
            offset += Self.segmentRows
        }
        if offset < rows {
            quantizedSegments.append(
                Self.quantize(value[0..., offset..<rows, 0...]))
        }
    }

    func trimTrailingRows(_ count: Int) {
        var remaining = max(0, count)
        guard remaining > 0 else { return }
        emptySpec = nil
        if let hotPool {
            let keep = max(0, hotPool.dim(1) - remaining)
            self.hotPool = keep == 0 ? nil : hotPool[0..., 0..<keep, 0...]
            return
        }

        while remaining > 0, let last = quantizedSegments.last {
            if last.rowCount <= remaining {
                quantizedSegments.removeLast()
                remaining -= last.rowCount
            } else {
                quantizedSegments[quantizedSegments.count - 1] = Self.slice(
                    last, keepingRows: last.rowCount - remaining)
                remaining = 0
            }
        }
    }

    func replace(withQuantizedSegments segments: [HybridPoolQuantizedSegment]) {
        hotPool = nil
        emptySpec = nil
        if quantizationEnabled {
            quantizedSegments = segments
        } else {
            quantizedSegments = []
            guard !segments.isEmpty else { return }
            let parts = segments.map(Self.dequantize)
            hotPool = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        }
    }

    func copy() -> DeepseekV4PoolStorage {
        let result = DeepseekV4PoolStorage(quantizationEnabled: quantizationEnabled)
        result.hotPool = hotPool?[.ellipsis]
        result.emptySpec = emptySpec
        result.quantizedSegments = quantizedSegments.map { segment in
            HybridPoolQuantizedSegment(
                codes: segment.codes[.ellipsis],
                scales: segment.scales[.ellipsis],
                biases: segment.biases[.ellipsis],
                originalShape: segment.originalShape,
                groupSize: segment.groupSize,
                bits: segment.bits,
                originalDType: segment.originalDType)
        }
        return result
    }

    private func replaceQuantized(with value: MLXArray) {
        hotPool = nil
        quantizedSegments = []
        var start = 0
        while start < value.dim(1) {
            let end = min(start + Self.segmentRows, value.dim(1))
            quantizedSegments.append(Self.quantize(value[0..., start..<end, 0...]))
            start = end
        }
    }

    private func trailingPartialSegmentRows() -> Int {
        var rows = 0
        for segment in quantizedSegments.reversed() {
            guard segment.rowCount < Self.segmentRows else { break }
            rows += segment.rowCount
        }
        return rows
    }

    private func compactCompletedTail() {
        var start = quantizedSegments.count
        var rows = 0
        while start > 0 {
            let segment = quantizedSegments[start - 1]
            guard segment.rowCount < Self.segmentRows else { break }
            start -= 1
            rows += segment.rowCount
        }
        guard rows == Self.segmentRows else { return }
        let parts = quantizedSegments[start...].map(Self.dequantize)
        let materialized = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        quantizedSegments.replaceSubrange(start..., with: [Self.quantize(materialized)])
    }

    private static func quantize(_ value: MLXArray) -> HybridPoolQuantizedSegment {
        let shape = value.shape
        let featureCount = shape.last ?? 0
        precondition(featureCount > 0, "DSV4 pool feature dimension must be non-zero")
        let effectiveGroupSize = featureCount % groupSize == 0 ? groupSize : featureCount
        let groupCount = featureCount / effectiveGroupSize
        let groupedShape = Array(shape.dropLast()) + [groupCount, effectiveGroupSize]
        let grouped = value.asType(.float32).reshaped(groupedShape)
        let minimums = grouped.min(axis: -1, keepDims: true)
        let maximums = grouped.max(axis: -1, keepDims: true)
        let qmax = Float((1 << bits) - 1)
        let rawScales = (maximums - minimums) / qmax
        let scales32 = MLX.where(
            rawScales .> Float(1e-8),
            rawScales,
            MLXArray.ones(like: rawScales))
        let codes = clip(
            MLX.round((grouped - minimums) / scales32),
            min: Float(0), max: qmax).asType(.uint8)
        let scales = scales32.asType(.float16)
        let biases = minimums.asType(.float16)
        MLX.eval([codes, scales, biases])
        return HybridPoolQuantizedSegment(
            codes: codes,
            scales: scales,
            biases: biases,
            originalShape: shape,
            groupSize: effectiveGroupSize,
            bits: bits,
            originalDType: value.dtype)
    }

    private static func dequantize(_ segment: HybridPoolQuantizedSegment) -> MLXArray {
        (segment.codes.asType(.float32) * segment.scales.asType(.float32)
            + segment.biases.asType(.float32))
            .reshaped(segment.originalShape)
            .asType(segment.originalDType)
    }

    private static func slice(
        _ segment: HybridPoolQuantizedSegment,
        keepingRows rows: Int
    ) -> HybridPoolQuantizedSegment {
        var shape = segment.originalShape
        shape[1] = rows
        let result = HybridPoolQuantizedSegment(
            codes: segment.codes[0..., 0..<rows, 0..., 0...],
            scales: segment.scales[0..., 0..<rows, 0..., 0...],
            biases: segment.biases[0..., 0..<rows, 0..., 0...],
            originalShape: shape,
            groupSize: segment.groupSize,
            bits: segment.bits,
            originalDType: segment.originalDType)
        MLX.eval([result.codes, result.scales, result.biases])
        return result
    }
}

// MARK: - DeepseekV4Cache
//
// Per-layer composite cache. Wraps a `RotatingKVCache` for the local
// sliding window plus persistent buffer state for the compressor and
// indexer. Multi-call stateful: on each prefill step it accumulates
// raw-token windows until a full `compress_ratio`-sized chunk is ready,
// then pools and stores. The pooled sequence grows across calls so
// turn 2 and beyond see the full history summary.
//
// For short prompts (L < compress_ratio) and no V4Cache provided, the
// attention forward takes a fast-path that skips the compressor
// entirely (Python mirror: `if v4_cache is None and L < compress_ratio
// → skip`).
public final class DeepseekV4Cache: QuantizedHybridPoolCache, CacheRetainedByteCountProviding {
    /// Lightweight checkpoint for speculative verification. Pool appends replace
    /// storage arrays instead of mutating them in place, so retaining these views
    /// is enough to restore the compressed context without copying the bounded
    /// rotating K/V window on every speculative round.
    public struct SpeculativeSnapshot {
        fileprivate let compPool: DeepseekV4PoolStorage
        fileprivate let idxPool: DeepseekV4PoolStorage
        fileprivate let compBufferKV: MLXArray?
        fileprivate let compBufferGate: MLXArray?
        fileprivate let idxBufferKV: MLXArray?
        fileprivate let idxBufferGate: MLXArray?
    }
    /// Expose the inner rotating cache so `TQDiskSerializer` and
    /// `restoreRotatingLayer` can round-trip the sliding-window state.
    /// Compressor/Indexer pool tensors and their incomplete-window
    /// buffers ALSO round-trip via `state` / `metaState` so prefix-cache
    /// reuse for multi-turn chat doesn't have to re-derive the pool
    /// from prompt tokens every turn.
    public var rotating: RotatingKVCache { local }
    /// Local sliding-window cache (compress_ratio-agnostic).
    public let local: RotatingKVCache
    public let slidingWindow: Int
    /// Per-layer compress_ratio. Required (no nil): the proportional
    /// pool-row truncation in `trim(_:)` needs it, the paged cache
    /// uses it as part of the block hash so `cr=4` and `cr=128` layer
    /// blocks never collide, and the disk serializer stamps it as a
    /// metaState entry.
    public let compressRatio: Int
    /// Compressor buffer state (raw kv/gate not yet ready to pool)
    /// and pooled summary so far.
    fileprivate var compBufferKV: MLXArray?
    fileprivate var compBufferGate: MLXArray?
    private var compPool: DeepseekV4PoolStorage
    /// Indexer's own buffer state (separate branch — compressor inside
    /// the indexer uses its own buffers).
    fileprivate var idxBufferKV: MLXArray?
    fileprivate var idxBufferGate: MLXArray?
    private var idxPool: DeepseekV4PoolStorage

    /// Family-native DSV4 pool quantization. This does not enable TurboQuant
    /// KV: the rotating local K/V cache remains in its model-selected dtype.
    public let hybridPoolQuantizationEnabled: Bool

    public init(
        slidingWindow: Int,
        compressRatio: Int,
        poolQuantizationEnabled: Bool? = nil
    ) {
        precondition(compressRatio > 0, "DeepseekV4Cache requires compressRatio > 0; cr=0 layers should use a plain RotatingKVCache.")
        let quantizationEnabled = poolQuantizationEnabled
            ?? Self.resolvePoolQuantizationDefault()
        self.slidingWindow = slidingWindow
        self.compressRatio = compressRatio
        self.hybridPoolQuantizationEnabled = quantizationEnabled
        self.compPool = DeepseekV4PoolStorage(quantizationEnabled: quantizationEnabled)
        self.idxPool = DeepseekV4PoolStorage(quantizationEnabled: quantizationEnabled)
        self.local = RotatingKVCache(maxSize: slidingWindow, keep: 0)
    }

    /// Pool quantization is the production default for DSV4 long context.
    /// `DSV4_POOL_QUANT=0|false|off|no` remains an explicit diagnostic A/B.
    public static func resolvePoolQuantizationDefault(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment["DSV4_POOL_QUANT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        else { return true }
        return !["0", "false", "off", "no"].contains(raw)
    }

    /// Conservative retained-byte projection for one batch of the complete
    /// DSV4 hybrid cache at a requested context. It includes UInt8 pool codes,
    /// fp16 affine metadata, every layer's bounded rotating K/V, and a full
    /// incomplete-window buffer allowance for both compressor branches.
    public static func projectedQuantizedCacheUpperBoundBytes(
        contextLength: Int,
        compressRatios: [Int],
        headDim: Int,
        indexerHeadDim: Int,
        slidingWindow: Int,
        activationBytes: Int = 2
    ) -> Int {
        precondition(contextLength >= 0)
        precondition(headDim > 0 && indexerHeadDim > 0)
        precondition(slidingWindow > 0 && activationBytes > 0)

        func encodedPoolBytes(rows: Int, features: Int) -> Int {
            guard rows > 0 else { return 0 }
            let group = features % DeepseekV4PoolStorage.groupSize == 0
                ? DeepseekV4PoolStorage.groupSize
                : features
            let elements = rows * features
            let affineGroups = rows * (features / group)
            // UInt8 code + fp16 scale + fp16 minimum per affine group.
            return elements + affineGroups * 2 * MemoryLayout<Float16>.size
        }

        var total = compressRatios.count
            * 2 * slidingWindow * headDim * activationBytes
        for ratio in compressRatios where ratio > 0 {
            let rows = (contextLength + ratio - 1) / ratio
            total += encodedPoolBytes(rows: rows, features: headDim)
            let compressorWidth = ratio == 4 ? 2 * headDim : headDim
            total += 2 * ratio * compressorWidth * activationBytes
            if ratio == 4 {
                total += encodedPoolBytes(rows: rows, features: indexerHeadDim)
                total += 2 * ratio * indexerHeadDim * activationBytes
            }
        }
        return total
    }

    // KVCache protocol implementation — delegate everything to `local`.
    public var offset: Int { local.offset }
    public var offsetArray: MLXArray? { local.offsetArray }
    public var maxSize: Int? { local.maxSize }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        local.update(keys: keys, values: values)
    }

    /// Round-trip layout (kept stable for `TQDiskSerializer`):
    ///   index 0..<localCount       — `local.state` (rotating window)
    ///   index localCount + 0       — compPooled (or empty zero-row tensor)
    ///   index localCount + 1       — compBufferKV (or empty)
    ///   index localCount + 2       — compBufferGate (or empty)
    ///   index localCount + 3       — idxPooled
    ///   index localCount + 4       — idxBufferKV
    ///   index localCount + 5       — idxBufferGate
    /// `metaState` carries the entry tag list + `compress_ratio` + the
    /// number of `local.state` arrays so deserialization knows where to
    /// split.
    public var state: [MLXArray] {
        get {
            let localState = local.state
            return localState
                + [serializableArray(compPool.materialized()),
                   serializableArray(compBufferKV),
                   serializableArray(compBufferGate),
                   serializableArray(idxPool.materialized()),
                   serializableArray(idxBufferKV),
                   serializableArray(idxBufferGate)]
        }
        set {
            // Last 6 slots are the DSV4 pool/buffer arrays; everything
            // before them is `local.state`.
            precondition(newValue.count >= 6,
                "DeepseekV4Cache.state setter expects at least 6 trailing pool slots")
            let split = newValue.count - 6
            local.state = Array(newValue[0..<split])
            compPool.replace(with: nullableFromArray(newValue[split + 0]))
            compBufferKV = nullableFromArray(newValue[split + 1])
            compBufferGate = nullableFromArray(newValue[split + 2])
            idxPool.replace(with: nullableFromArray(newValue[split + 3]))
            idxBufferKV = nullableFromArray(newValue[split + 4])
            idxBufferGate = nullableFromArray(newValue[split + 5])
        }
    }

    public var metaState: [String] {
        get {
            // First entries are `local.metaState`; we tack on dsv4-
            // specific scalars so the disk serializer + restore path
            // can reconstruct without guessing.
            local.metaState + [
                "dsv4_cache_v1",
                String(compressRatio),
                String(slidingWindow),
            ]
        }
        set {
            // Strip the DSV4 trailer if present and pass the rest to
            // local. We don't need to re-read compressRatio from the
            // trailer because it's an init-time invariant of the
            // surrounding layer's attention module — but accept the
            // legacy code path that didn't write it.
            if newValue.count >= 3,
               newValue[newValue.count - 3] == "dsv4_cache_v1"
            {
                local.metaState = Array(newValue[0..<(newValue.count - 3)])
            } else {
                local.metaState = newValue
            }
        }
    }

    public var isTrimmable: Bool { local.isTrimmable }

    public func captureSpeculativeSnapshot() -> SpeculativeSnapshot {
        SpeculativeSnapshot(
            compPool: compPool.copy(),
            idxPool: idxPool.copy(),
            compBufferKV: compBufferKV?[.ellipsis],
            compBufferGate: compBufferGate?[.ellipsis],
            idxBufferKV: idxBufferKV?[.ellipsis],
            idxBufferGate: idxBufferGate?[.ellipsis])
    }

    /// Roll back only rejected verifier rows. The local rotating cache can
    /// rewind its logical offset even after reaching the window size; restoring
    /// the checkpoint separately preserves every previously committed pool row.
    public func rollbackSpeculative(
        rejected: Int,
        to snapshot: SpeculativeSnapshot
    ) {
        guard rejected > 0 else { return }
        _ = local.trim(rejected)
        compPool = snapshot.compPool
        idxPool = snapshot.idxPool
        compBufferKV = snapshot.compBufferKV
        compBufferGate = snapshot.compBufferGate
        idxBufferKV = snapshot.idxBufferKV
        idxBufferGate = snapshot.idxBufferGate
    }

    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Proportional pool-row truncation matching `llama.cpp dsv4_clear_rows`.
    /// Pre-fix this only delegated to `local.trim(n)` — the contaminated
    /// pool rows survived multi-turn prefix-cache reuse and produced the
    /// "polite-assistant attractor loops" reported on /v1/chat/completions.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let rv = local.trim(n)
        // Incomplete-window buffers are start_pos-keyed and invalidated
        // by ANY trim — clear unconditionally (Python comment
        // mlx_model.py L497-501).
        compBufferKV = nil
        compBufferGate = nil
        idxBufferKV = nil
        idxBufferGate = nil
        if n <= 0 || compressRatio <= 0 {
            return rv
        }
        // Drop the trailing `max(1, n / compress_ratio)` rows from each
        // pool. Most-recently-appended row may have been computed from
        // a window overlapping output tokens — keeping it would
        // re-introduce contamination (Python L532-537).
        let rowsToDrop = max(1, n / compressRatio)
        compPool.trimTrailingRows(rowsToDrop)
        idxPool.trimTrailingRows(rowsToDrop)
        return rv
    }

    public func innerState() -> [MLXArray] {
        local.innerState()
            + compPool.retainedArrays
            + [compBufferKV, compBufferGate, idxBufferKV, idxBufferGate].compactMap { $0 }
            + idxPool.retainedArrays
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        local.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func truncateToOffset() {
        local.truncateToOffset()
    }

    public func copy() -> any KVCache {
        // Deep-copy: clone local KV plus all four pool/buffer slots so
        // the snapshot is fully independent of the original cache.
        // Pre-fix `copy()` constructed a fresh cache and only set
        // `state = local.state`, which silently dropped the pool
        // entirely — so any caller that relied on `copy()` to
        // checkpoint multi-turn state lost the long-context summary.
        let dup = DeepseekV4Cache(
            slidingWindow: slidingWindow,
            compressRatio: compressRatio,
            poolQuantizationEnabled: hybridPoolQuantizationEnabled)
        let localState = local.state
        if !localState.isEmpty {
            // RotatingKVCache writes decode tokens into its ring in place.
            // An ellipsis slice is only a view, so the retained prompt-minus-one
            // checkpoint would otherwise be overwritten before the synchronous
            // disk store at terminal state. Create independent buffers; the
            // snapshot caller batches their materialization in one MLX.eval.
            dup.local.state = localState.map { $0 * 1 }
        }
        dup.local.metaState = local.metaState
        dup.compPool = compPool.copy()
        dup.idxPool = idxPool.copy()
        dup.compBufferKV = compBufferKV?[.ellipsis]
        dup.compBufferGate = compBufferGate?[.ellipsis]
        dup.idxBufferKV = idxBufferKV?[.ellipsis]
        dup.idxBufferGate = idxBufferGate?[.ellipsis]
        return dup
    }

    public var retainedCacheByteCount: Int {
        local.state.reduce(0) { $0 + $1.nbytes }
            + compPool.retainedByteCount
            + idxPool.retainedByteCount
            + [compBufferKV, compBufferGate, idxBufferKV, idxBufferGate]
                .compactMap { $0 }
                .reduce(0) { $0 + $1.nbytes }
    }

    // MARK: - State (de)serialization helpers

    /// MLX state arrays cannot be `nil`, so we substitute a zero-row
    /// sentinel and recover it on the way back. The sentinel is shape
    /// (1, 0, 1) fp32 — distinguishable from any real pool tensor by
    /// having axis-1 size 0.
    private func serializableArray(_ arr: MLXArray?) -> MLXArray {
        guard let arr else {
            return MLXArray.zeros([1, 0, 1], dtype: .float32)
        }
        return arr
    }

    private func nullableFromArray(_ arr: MLXArray) -> MLXArray? {
        // Sentinel for nil: any tensor whose axis-1 dim is 0.
        if arr.ndim >= 2 && arr.dim(1) == 0 {
            return nil
        }
        return arr
    }

    // State accessors for Compressor/Indexer. Public so disk round-trip
    // tests and any future cache-inspection code can verify the
    // ephemeral buffers are cleared on restore (they recompute from
    // prompt tokens on the next prefill).
    public func getBuffers(_ key: BranchKey) -> (kv: MLXArray?, gate: MLXArray?) {
        switch key {
        case .compressor: return (compBufferKV, compBufferGate)
        case .indexer: return (idxBufferKV, idxBufferGate)
        }
    }

    public func setBuffers(_ key: BranchKey, kv: MLXArray?, gate: MLXArray?) {
        switch key {
        case .compressor:
            compBufferKV = kv
            compBufferGate = gate
        case .indexer:
            idxBufferKV = kv
            idxBufferGate = gate
        }
    }

    public func getPooled(_ key: BranchKey) -> MLXArray? {
        key == .compressor ? compPool.materialized() : idxPool.materialized()
    }

    public func setPooled(_ key: BranchKey, value: MLXArray) {
        if key == .compressor {
            compPool.replace(with: value)
        } else {
            idxPool.replace(with: value)
        }
    }

    public func appendPooled(_ key: BranchKey, value: MLXArray) -> MLXArray {
        let storage = key == .compressor ? compPool : idxPool
        storage.append(value)
        return storage.materialized()
            ?? MLXArray.zeros([value.dim(0), 0, value.dim(2)], dtype: value.dtype)
    }

    public enum BranchKey { case compressor, indexer }

    // MARK: - HybridPoolCache conformance
    //
    // 2026-05-04: forward the protocol API used by the disk serializer
    // and the paged cache to the existing `getPooled`/`setPooled` /
    // `getBuffers`/`setBuffers` accessors. The protocol exists so the
    // MLXLMCommon disk-cache subsystem can persist DSV4 layers without
    // an MLXLLM dependency.
    private static func bridge(_ branch: HybridPoolBranch) -> BranchKey {
        switch branch {
        case .compressor: return .compressor
        case .indexer: return .indexer
        }
    }

    public func hybridPool(branch: HybridPoolBranch) -> MLXArray? {
        getPooled(Self.bridge(branch))
    }

    public func setHybridPool(branch: HybridPoolBranch, value: MLXArray?) {
        let key = Self.bridge(branch)
        if let value {
            setPooled(key, value: value)
        } else {
            switch key {
            case .compressor: compPool.replace(with: nil)
            case .indexer: idxPool.replace(with: nil)
            }
        }
    }

    public func hybridBuffers(branch: HybridPoolBranch) -> (kv: MLXArray?, gate: MLXArray?) {
        getBuffers(Self.bridge(branch))
    }

    public func setHybridBuffers(branch: HybridPoolBranch, kv: MLXArray?, gate: MLXArray?) {
        setBuffers(Self.bridge(branch), kv: kv, gate: gate)
    }

    public func hybridPoolQuantizedSegments(
        branch: HybridPoolBranch
    ) -> [HybridPoolQuantizedSegment]? {
        let segments = branch == .compressor
            ? compPool.quantizedSegments
            : idxPool.quantizedSegments
        return segments.isEmpty ? nil : segments
    }

    public func setHybridPoolQuantizedSegments(
        branch: HybridPoolBranch,
        segments: [HybridPoolQuantizedSegment]
    ) {
        if branch == .compressor {
            compPool.replace(withQuantizedSegments: segments)
        } else {
            idxPool.replace(withQuantizedSegments: segments)
        }
    }

    public func hybridPoolRetainedByteCount(branch: HybridPoolBranch) -> Int {
        branch == .compressor
            ? compPool.retainedByteCount
            : idxPool.retainedByteCount
    }
}

// MARK: - Batched DeepSeek V4 cache

/// Ephemeral scheduler container for concurrent DeepSeek V4 generation.
///
/// The hybrid cache cannot be represented by the generic dense KV batch:
/// compressor/indexer pools and incomplete windows have request-specific row
/// counts and phases. Keep each request's complete cache independent while the
/// transformer continues to execute a batched hidden-state/MoE graph.
public final class BatchDeepseekV4Cache: KVCache, CacheRetainedByteCountProviding {
    private var caches: [DeepseekV4Cache]

    public init(caches: [DeepseekV4Cache]) {
        precondition(!caches.isEmpty, "BatchDeepseekV4Cache requires at least one cache")
        let first = caches[0]
        precondition(caches.allSatisfy {
            $0.slidingWindow == first.slidingWindow
                && $0.compressRatio == first.compressRatio
                && $0.hybridPoolQuantizationEnabled == first.hybridPoolQuantizationEnabled
        }, "DeepSeek V4 batch caches must have matching configurations")
        self.caches = caches
    }

    public var count: Int { caches.count }
    public var offset: Int { caches.map(\.offset).max() ?? 0 }
    public var offsetArray: MLXArray? {
        MLXArray(caches.map { Int32($0.offset) })
    }
    public var maxSize: Int? { caches.first?.maxSize }
    public var isTrimmable: Bool { caches.allSatisfy(\.isTrimmable) }
    public var retainedCacheByteCount: Int {
        caches.reduce(0) { $0 + $1.retainedCacheByteCount }
    }

    public func cache(at index: Int) -> DeepseekV4Cache {
        caches[index]
    }

    public static func merge(_ source: [KVCache]) -> BatchDeepseekV4Cache {
        BatchDeepseekV4Cache(caches: source.map { cache in
            guard let deepseek = cache as? DeepseekV4Cache else {
                preconditionFailure("BatchDeepseekV4Cache.merge requires DeepseekV4Cache inputs")
            }
            return deepseek
        })
    }

    public func extend(with cache: DeepseekV4Cache) {
        guard let first = caches.first else {
            caches = [cache]
            return
        }
        precondition(
            cache.slidingWindow == first.slidingWindow
                && cache.compressRatio == first.compressRatio
                && cache.hybridPoolQuantizationEnabled == first.hybridPoolQuantizationEnabled,
            "Cannot extend DeepSeek V4 batch with a different cache configuration")
        caches.append(cache)
    }

    public func filter(_ keepIndices: [Int]) {
        caches = keepIndices.map { caches[$0] }
    }

    public func extract(_ index: Int) -> DeepseekV4Cache {
        caches[index].copy() as! DeepseekV4Cache
    }

    /// Attention recognizes this container and updates each child directly.
    /// Keep a defensive implementation for generic cache inspection utilities.
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys.dim(0) == caches.count && values.dim(0) == caches.count)
        var keyRows: [MLXArray] = []
        var valueRows: [MLXArray] = []
        var lengths: [Int] = []
        for index in caches.indices {
            let (rowKeys, rowValues) = caches[index].update(
                keys: keys[index ..< index + 1],
                values: values[index ..< index + 1])
            keyRows.append(rowKeys)
            valueRows.append(rowValues)
            lengths.append(rowKeys.dim(2))
        }
        let maximum = lengths.max() ?? 0
        func pad(_ rows: [MLXArray]) -> MLXArray {
            concatenated(zip(rows, lengths).map { row, length in
                guard length < maximum else { return row }
                let shape = [row.dim(0), row.dim(1), maximum - length, row.dim(3)]
                return concatenated([MLXArray.zeros(shape, dtype: row.dtype), row], axis: 2)
            }, axis: 0)
        }
        return (pad(keyRows), pad(valueRows))
    }

    /// Batch containers are runtime-only and are never persisted as one cache.
    public var state: [MLXArray] {
        get { [] }
        set {
            precondition(newValue.isEmpty,
                "BatchDeepseekV4Cache state must be extracted per request")
        }
    }

    public var metaState: [String] {
        get { [] }
        set {
            precondition(newValue.isEmpty,
                "BatchDeepseekV4Cache metadata must be extracted per request")
        }
    }

    public func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        caches.map { $0.trim(n) }.min() ?? 0
    }

    public func truncateToOffset() {
        caches.forEach { $0.truncateToOffset() }
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        caches[0].makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        BatchDeepseekV4Cache(caches: caches.map { $0.copy() as! DeepseekV4Cache })
    }
}

// MARK: - Compressor
//
// Projects input x through `wkv` + `wgate`, accumulates raw windows
// until a full `compress_ratio`-chunk is ready, then pools the chunk
// via softmax(gate)-weighted sum. For compress_ratio=4 the output is
// 2× widened (overlap mode) so each pool spans TWO adjacent chunks —
// strictly increases context coverage. After pooling, applies the
// absolute position embedding (APE) and partial RoPE at the chunk
// positions. Updates the per-layer V4Cache pool if provided.
enum DeepseekV4CompressorQATKind {
    case attentionKV
    case indexer
}

public final class DeepseekV4Compressor: Module {
    let compressRatio: Int
    let headDim: Int
    let outDim: Int
    let overlap: Bool
    let rmsNormEps: Float
    let qatKind: DeepseekV4CompressorQATKind
    let activationQATEnabled: Bool

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    /// APE: (compress_ratio, out_dim) learned positional bias inside
    /// each pool window.
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(
        config: DeepseekV4Configuration,
        compressRatio: Int,
        headDim: Int,
        qatKind: DeepseekV4CompressorQATKind = .attentionKV
    ) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.rmsNormEps = config.rmsNormEps
        self.qatKind = qatKind
        self.activationQATEnabled = config.activationQATEnabled
        self.overlap = compressRatio == 4
        self.outDim = headDim * (overlap ? 2 : 1)
        self._wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._ape.wrappedValue = zeros([compressRatio, outDim])
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    /// Forward. Returns pooled summary of shape (B, pooled_count, headDim).
    /// When `v4Cache` is provided, pooled is appended to the cache pool
    /// and the full cached pool is returned.
    func callAsFunction(
        _ x: MLXArray,
        rope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int,
        branch: DeepseekV4Cache.BranchKey = .compressor
    ) -> MLXArray {
        let B = x.dim(0)
        // Official 0731 stages the compressor projections and pooling state
        // in fp32. This must happen before the linears, not merely after them.
        let projectionInput = x.asType(.float32)
        var kv = wkv(projectionInput)
        var gate = wgate(projectionInput)

        var poolBase = startPos
        let alreadyWindowed: Bool
        if let cache = v4Cache, overlap {
            let positions = MLXArray(
                Int32(startPos)..<Int32(startPos + gate.dim(1)))
            let apeRows = ape.asType(gate.dtype)[positions % Int32(compressRatio)]
            gate = gate + apeRows.expandedDimensions(axis: 0)
            let accumulated = accumulateOverlapWindows(
                kv: kv, gate: gate, cache: cache, branch: branch,
                ratio: compressRatio, startPos: startPos)
            kv = accumulated.kvRows
            gate = accumulated.gateRows
            poolBase = accumulated.poolBase
            alreadyWindowed = true
        } else {
            // Accumulate windows. When cache present, prepend unused-tail
            // buffers from prior calls.
            if let cache = v4Cache {
                let (bufKV, bufGate) = cache.getBuffers(branch)
                if let bKV = bufKV, bKV.dim(1) > 0, let bG = bufGate {
                    kv = concatenated([bKV, kv], axis: 1)
                    gate = concatenated([bG, gate], axis: 1)
                    poolBase -= bKV.dim(1)
                }
                let total = kv.dim(1)
                let usable = (total / compressRatio) * compressRatio
                // Stash the tail for the next call.
                let tailKV = usable < total ? kv[0..., usable..., 0...] : nil
                let tailGate = usable < total ? gate[0..., usable..., 0...] : nil
                cache.setBuffers(branch, kv: tailKV, gate: tailGate)
                kv = kv[0..., 0..<usable, 0...]
                gate = gate[0..., 0..<usable, 0...]
            } else {
                let total = kv.dim(1)
                let usable = (total / compressRatio) * compressRatio
                kv = kv[0..., 0..<usable, 0...]
                gate = gate[0..., 0..<usable, 0...]
            }
            alreadyWindowed = false
        }

        if kv.dim(1) == 0 {
            let empty = MLXArray.zeros([B, 0, headDim], dtype: x.dtype)
            if let cache = v4Cache {
                return cache.getPooled(branch) ?? empty
            }
            return empty
        }

        let W: Int
        var kvWin: MLXArray
        var gateWin: MLXArray
        if alreadyWindowed {
            W = kv.dim(1)
            kvWin = kv
            gateWin = gate
        } else {
            W = kv.dim(1) / compressRatio
            kvWin = kv.reshaped(B, W, compressRatio, outDim)
            gateWin =
                gate.reshaped(B, W, compressRatio, outDim) + ape.asType(gate.dtype)

            if overlap {
                kvWin = overlapTransform(kvWin, fillValue: 0.0)
                // For gate, the pre-allocated fill is -inf so softmax assigns
                // zero mass to the padding half.
                gateWin = overlapTransform(
                    gateWin, fillValue: -Float.infinity)
            }
        }

        let weights =
            softmax(gateWin.asType(.float32), axis: 2, precise: true).asType(
                kvWin.dtype)
        var pooled = (kvWin * weights).sum(axis: 2)
        pooled = norm(pooled.asType(x.dtype))

        // Apply RoPE at the chunk centers (position = chunk_idx * ratio
        // + pool_base).
        let positions =
            MLXArray(
                Int32(0)..<Int32(pooled.dim(1))
            ).asType(.float32) * Float(compressRatio) + Float(poolBase)
        // Build cos/sin at those positions.
        let angles =
            positions.expandedDimensions(axis: -1)
            * rope.invFreq.expandedDimensions(axis: 0)
        let cosP = cos(angles).expandedDimensions(axes: [0])
        let sinP = sin(angles).expandedDimensions(axes: [0])
        pooled = DeepseekV4Math.applyPartialRoPE(
            pooled, cos: cosP, sin: sinP, ropeDim: rope.dim)

        if activationQATEnabled {
            switch qatKind {
            case .attentionKV:
                let nopeDim = headDim - rope.dim
                if nopeDim >= 64 {
                    pooled = DeepseekV4Math.e4m3KVActivationRoundTrip(
                        pooled, ropeDim: rope.dim)
                }
            case .indexer:
                if headDim == 128 {
                    pooled = DeepseekV4Math.indexerActivationRoundTrip(pooled)
                }
            }
        }

        if let cache = v4Cache {
            return cache.appendPooled(branch, value: pooled)
        }
        return pooled
    }

    /// Ratio-4 overlap accumulation for the stateful decode path.
    ///
    /// DSV4's overlap compressor needs the previous complete window for the
    /// left half of the next pooled row. A plain remainder buffer works during
    /// large prefill chunks but corrupts single-token decode: every completed
    /// decode row would otherwise get a zero left half at the call boundary.
    func accumulateOverlapWindows(
        kv: MLXArray,
        gate: MLXArray,
        cache: DeepseekV4Cache,
        branch: DeepseekV4Cache.BranchKey,
        ratio: Int,
        startPos: Int
    ) -> (kvRows: MLXArray, gateRows: MLXArray, poolBase: Int) {
        let B = kv.dim(0)

        func emptyRows(_ poolBase: Int) -> (kvRows: MLXArray, gateRows: MLXArray, poolBase: Int) {
            (
                MLXArray.zeros([B, 0, 2 * ratio, headDim], dtype: kv.dtype),
                MLXArray.zeros([B, 0, 2 * ratio, headDim], dtype: gate.dtype),
                poolBase
            )
        }

        func makeRow(
            prevKV: MLXArray?,
            prevGate: MLXArray?,
            curKV: MLXArray,
            curGate: MLXArray
        ) -> (MLXArray, MLXArray) {
            let leftKV: MLXArray
            let leftGate: MLXArray
            if let prevKV, let prevGate {
                leftKV = prevKV[0..., 0..., 0..<headDim]
                leftGate = prevGate[0..., 0..., 0..<headDim]
            } else {
                leftKV = MLXArray.zeros([B, ratio, headDim], dtype: kv.dtype)
                leftGate = MLXArray.full(
                    [B, ratio, headDim],
                    values: MLXArray(-Float.infinity).asType(gate.dtype))
            }
            let rightKV = curKV[0..., 0..., headDim...]
            let rightGate = curGate[0..., 0..., headDim...]
            return (
                concatenated([leftKV, rightKV], axis: 1).expandedDimensions(axis: 1),
                concatenated([leftGate, rightGate], axis: 1).expandedDimensions(axis: 1)
            )
        }

        if startPos == 0 {
            let usable = (kv.dim(1) / ratio) * ratio
            let remainderKV = usable < kv.dim(1) ? kv[0..., usable..., 0...] : nil
            let remainderGate = usable < gate.dim(1) ? gate[0..., usable..., 0...] : nil
            if usable >= ratio {
                let lastKV = kv[0..., (usable - ratio)..<usable, 0...]
                let lastGate = gate[0..., (usable - ratio)..<usable, 0...]
                cache.setBuffers(
                    branch,
                    kv: remainderKV != nil
                        ? concatenated([lastKV, remainderKV!], axis: 1)
                        : lastKV,
                    gate: remainderGate != nil
                        ? concatenated([lastGate, remainderGate!], axis: 1)
                        : lastGate)
            } else {
                cache.setBuffers(branch, kv: remainderKV, gate: remainderGate)
            }
            guard usable > 0 else { return emptyRows(startPos) }
            let W = usable / ratio
            let fullKV = kv[0..., 0..<usable, 0...].reshaped(B, W, ratio, outDim)
            let fullGate = gate[0..., 0..<usable, 0...].reshaped(B, W, ratio, outDim)
            return (
                overlapTransform(fullKV, fillValue: 0.0),
                overlapTransform(fullGate, fillValue: -Float.infinity),
                startPos
            )
        }

        let (bufKV, bufGate) = cache.getBuffers(branch)
        var prevKV: MLXArray? = nil
        var prevGate: MLXArray? = nil
        var partialKV = bufKV
        var partialGate = bufGate
        if let bKV = bufKV, bKV.dim(1) >= ratio, let bGate = bufGate {
            prevKV = bKV[0..., 0..<ratio, 0...]
            prevGate = bGate[0..., 0..<ratio, 0...]
            partialKV = bKV.dim(1) > ratio ? bKV[0..., ratio..., 0...] : nil
            partialGate = bGate.dim(1) > ratio ? bGate[0..., ratio..., 0...] : nil
        }

        let priorPartialLen = partialKV?.dim(1) ?? 0
        var currentKV =
            if let partialKV, partialKV.dim(1) > 0 {
                concatenated([partialKV, kv], axis: 1)
            } else {
                kv
            }
        var currentGate =
            if let partialGate, partialGate.dim(1) > 0 {
                concatenated([partialGate, gate], axis: 1)
            } else {
                gate
            }

        var kvRows: [MLXArray] = []
        var gateRows: [MLXArray] = []
        while currentKV.dim(1) >= ratio {
            let curKV = currentKV[0..., 0..<ratio, 0...]
            let curGate = currentGate[0..., 0..<ratio, 0...]
            let row = makeRow(prevKV: prevKV, prevGate: prevGate, curKV: curKV, curGate: curGate)
            kvRows.append(row.0)
            gateRows.append(row.1)
            prevKV = curKV
            prevGate = curGate
            currentKV = currentKV.dim(1) > ratio ? currentKV[0..., ratio..., 0...] :
                MLXArray.zeros([B, 0, outDim], dtype: kv.dtype)
            currentGate = currentGate.dim(1) > ratio ? currentGate[0..., ratio..., 0...] :
                MLXArray.zeros([B, 0, outDim], dtype: gate.dtype)
        }

        if let prevKV, let prevGate {
            cache.setBuffers(
                branch,
                kv: currentKV.dim(1) > 0 ? concatenated([prevKV, currentKV], axis: 1) : prevKV,
                gate: currentGate.dim(1) > 0 ? concatenated([prevGate, currentGate], axis: 1) : prevGate)
        } else {
            cache.setBuffers(
                branch,
                kv: currentKV.dim(1) > 0 ? currentKV : nil,
                gate: currentGate.dim(1) > 0 ? currentGate : nil)
        }

        let poolBase = max(0, startPos - priorPartialLen)
        guard !kvRows.isEmpty else { return emptyRows(poolBase) }
        return (concatenated(kvRows, axis: 1), concatenated(gateRows, axis: 1), poolBase)
    }

    /// Overlap transform for compress_ratio=4. Expands (B, W, R, D) to
    /// (B, W, 2R, D) where the first R columns are the first half of
    /// the previous window's output and the last R columns are the
    /// current window's second half — gives each pool access to both
    /// chunks.
    private func overlapTransform(_ x: MLXArray, fillValue: Float) -> MLXArray {
        let B = x.dim(0)
        let W = x.dim(1)
        let R = x.dim(2)
        // Build output in two halves via concatenate; avoids
        // mutable-assignment patterns that mlx-swift Module doesn't
        // provide a clean API for.
        //
        // Layout:
        //   out[:, 0, :R]   = fill
        //   out[:, 1:, :R]  = x[:, :-1, :, :headDim]
        //   out[:,  :, R:]  = x[:, :, :, headDim:]
        let firstHalfAll = x[0..., 0..., 0..., 0..<headDim]  // (B, W, R, hd)
        // Shift: prepend a fill-window at position 0.
        let fillWindow = MLXArray.full(
            [B, 1, R, headDim], values: MLXArray(fillValue).asType(x.dtype))
        let shifted = concatenated(
            [fillWindow, firstHalfAll[0..., 0..<(W - 1), 0..., 0...]],
            axis: 1)  // (B, W, R, hd)
        let secondHalfAll = x[0..., 0..., 0..., headDim...]  // (B, W, R, hd)
        // Concat along the R axis: (B, W, 2R, hd).
        return concatenated([shifted, secondHalfAll], axis: 2)
    }
}

// MARK: - Indexer

/// Per-query top-k selector over the Compressor's pooled output.
/// Only present on compress_ratio=4 layers. Given `x` and `q_residual`
/// (the post-`q_norm` low-rank Q), projects Q into `n_heads`×`head_dim`,
/// scores against the Compressor's pooled keys, weights by a per-head
/// coefficient from `weights_proj`, and returns the top-`index_topk`
/// indices per query position.
public final class DeepseekV4Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let topK: Int
    let compressRatio: Int
    let scale: Float
    let activationQATEnabled: Bool

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor

    init(config: DeepseekV4Configuration, compressRatio: Int) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.topK = config.indexTopk
        self.compressRatio = compressRatio
        self.scale = 1.0 / sqrt(Float(headDim))
        self.activationQATEnabled = config.activationQATEnabled
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, nHeads * headDim, bias: false)
        self._weightsProj.wrappedValue = Linear(
            config.hiddenSize, nHeads, bias: false)
        self._compressor.wrappedValue = DeepseekV4Compressor(
            config: config, compressRatio: compressRatio, headDim: headDim,
            qatKind: .indexer)
    }

    /// Always advances the indexer's private compressor branch. Returns top-k
    /// indices shape (B, L, k) only after the pool exceeds `topK`; while every
    /// row remains visible it returns nil without running the score path.
    func callAsFunction(
        _ x: MLXArray,
        qResidual: MLXArray,
        rope: DeepseekV4RoPE,
        positionRope: DeepseekV4RoPE,
        v4Cache: DeepseekV4Cache?,
        startPos: Int
    ) -> MLXArray? {
        let pooled = compressor(
            x, rope: rope, v4Cache: v4Cache,
            startPos: startPos, branch: .indexer)
        let pooledLen = pooled.dim(1)
        if pooledLen == 0 { return nil }

        let B = x.dim(0)
        let L = x.dim(1)
        // Keep A3's score-path optimization without skipping the stateful
        // compressor above. Returning nil tells attention to use every pooled
        // row directly; no Q projection, score matmul, or arg-partition runs.
        // This branch must occur only after `compressor(...)` so incremental
        // 2048 -> 2052 token growth has all 512 historical index rows when row
        // 513 first activates learned top-k selection.
        if pooledLen <= topK {
            return nil
        }

        var q = wqB(qResidual)
            .reshaped(B, L, nHeads, headDim)
            .transposed(0, 2, 1, 3)
        // Partial RoPE on Q using the plain (non-compressor) RoPE.
        let (cosT, sinT) = positionRope.cosSin(offset: startPos, length: L)
        let cosQ = cosT.expandedDimensions(axes: [0, 1])
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(
            q, cos: cosQ, sin: sinQ, ropeDim: rope.dim)
        if activationQATEnabled && headDim == 128 {
            q = DeepseekV4Math.indexerActivationRoundTrip(q)
        }

        // scores: (B, nHeads, L, pooledLen). Match Python shape.
        // q is (B, nHeads, L, headDim); pooled is (B, pooledLen, headDim).
        // Expand pooled to (B, 1, pooledLen, headDim) for broadcast.
        let pooledBroad = pooled.expandedDimensions(axis: 1)
        var scores = q.asType(.float32).matmul(
            pooledBroad.asType(.float32).swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(0.0)) * MLXArray(scale)

        // weights: (B, L, nHeads) * n_heads^-0.5. Broadcast over the
        // pooled axis and sum over heads.
        let wRaw = weightsProj(x).asType(.float32)
            * MLXArray(1.0 / sqrt(Float(nHeads)))
        // Reshape scores sum axis: (B, 1, L, nHeads) multiply → sum.
        let wExpanded = wRaw.swappedAxes(-1, -2).expandedDimensions(axis: -1)
        scores = (scores * wExpanded).sum(axis: 1)  // (B, L, pooledLen)
        scores = DeepseekV4Math.causalMaskedIndexerScores(
            scores, offset: startPos, ratio: compressRatio)

        let k = min(topK, pooled.dim(1))
        let topIdx = argPartition(-scores, kth: k - 1, axis: -1)[
            .ellipsis, 0..<k]
        return topIdx
    }
}
