// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4 (DSV4-Flash / DSV4-Pro) — full model forward.
//
// Reference:
//   - jang/research/DSV4-RUNTIME-ARCHITECTURE.md §1-14
//   - jang/research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §1 (all 13 bug fixes)
//   - jang-tools/jang_tools/dsv4_prune/mlx_model.py (1128 LOC Python ref)
//
// Architecture vs DSV3 (all new, all non-negotiable):
//   • mHC residual stream (hc_mult=4 parallel copies, collapse/expand
//     per block using a Sinkhorn-normalized mixing matrix)
//   • MLA with head_dim=512, num_kv_heads=1 (single latent KV head
//     broadcast to all 64 Q heads via GQA), RoPE only on last
//     qk_rope_head_dim=64 dims
//   • Learned per-head `attn_sink` logit prepended pre-softmax
//   • Inverse RoPE on attention OUTPUT (strips positional info before
//     residual add-back)
//   • Grouped low-rank O projection: `bsgd,grd→bsgr` einsum with
//     o_groups=8, o_lora_rank=1024, then wo_b to hidden_size
//   • MoE routing via sqrtsoftplus instead of softmax
//   • Hash routing for first num_hash_layers=3 layers (tid2eid lookup)
//   • DSV4 SwiGLU with swiglu_limit=10.0 (clamp gate + up)
//   • Per-layer rope_theta: 10000 for compress_ratio=0 (no YaRN),
//     160000 for compress_ratio>0 (with YaRN)
//   • HyperHead reduce at the top of the model (mHC copies → hidden)
//
// Compressor + Indexer (for long-context attention with compress_ratio>0)
// are wired for the canonical DSV4-Flash SWA+CSA+HSA path. Layers with
// cr>0 use DeepseekV4Cache to preserve the local sliding window plus
// pooled global context across turns and disk-cache restores.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum DeepseekV4PerformanceProfile {
    static var enabled: Bool {
        guard let value = getenv("VMLX_DSV4_STAGE_PROFILE") else { return false }
        return String(cString: value) == "1"
    }
}

private enum DeepseekV4RuntimeOptions {
    static func enabled(_ name: String, default defaultValue: Bool) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name] else {
            return defaultValue
        }
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return value != "0" && value != "false" && value != "off"
    }

    static var dsparkNativeHeadEnabled: Bool {
        let environment = ProcessInfo.processInfo.environment
        let kernels = (environment["AFM_MLX_KERNELS"]
            ?? environment["VMLX_DSV4_KERNELS"] ?? "native")
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return kernels == "native"
            && enabled("VMLX_DSV4_DSPARK_HEAD_GEMV", default: true)
    }
}

private enum DeepseekV4NumericTrace {
    static let enabled =
        ProcessInfo.processInfo.environment["VMLX_DSV4_NUMERIC_TRACE"] == "1"

    static func tokens(_ inputs: MLXArray) {
        guard enabled else { return }
        let values = inputs.reshaped(-1).asArray(Int32.self)
        FileHandle.standardError.write(
            Data("[DSV4Numeric] tokens=\(values)\n".utf8))
    }

    static func tensor(_ label: String, _ value: MLXArray) {
        guard enabled else { return }
        let flat = value.asType(.float32).reshaped(-1)
        MLX.eval(flat)
        let count = min(4, flat.size)
        let sample = count > 0 ? flat[0..<count].asArray(Float.self) : []
        let mean = flat.mean().item(Float.self)
        let rms = sqrt((flat * flat).mean().item(Float.self))
        let absoluteMax = abs(flat).max().item(Float.self)
        let line = String(
            format: "[DSV4Numeric] %@ shape=%@ mean=%.9g rms=%.9g max=%.9g sample=%@\n",
            label, String(describing: value.shape), mean, rms, absoluteMax,
            String(describing: sample))
        FileHandle.standardError.write(Data(line.utf8))
    }
}

// MARK: - RoPE

/// DSV4 RoPE: YaRN scaling with `high = min(..., dim-1)` clamp (bug #10).
/// Per-layer theta — the layer chooses between `rope_theta=10000` (no
/// YaRN when compress_ratio=0) and `compress_rope_theta=160000` (with
/// YaRN scaling when compress_ratio>0).
class DeepseekV4RoPE: Module {
    let dim: Int
    let base: Float
    let factor: Float
    let origMaxPos: Int
    let betaFast: Float
    let betaSlow: Float
    // Precomputed half-dim inv-freq table.
    let invFreq: MLXArray

    init(
        dim: Int,
        base: Float,
        factor: Float = 1.0,
        origMaxPos: Int = 65536,
        betaFast: Float = 32,
        betaSlow: Float = 1
    ) {
        self.dim = dim
        self.base = base
        self.factor = factor
        self.origMaxPos = origMaxPos
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.invFreq = DeepseekV4Math.yarnInvFreq(
            dim: dim, base: base, maxPos: 0,
            origMaxPos: origMaxPos, factor: factor,
            betaFast: betaFast, betaSlow: betaSlow)
    }

    /// Compute cos/sin tables for positions `[offset, offset+L)`.
    /// Returned shape: `(L, dim/2)`.
    func cosSin(offset: Int, length: Int) -> (cos: MLXArray, sin: MLXArray) {
        let positions = MLXArray(Int32(offset)..<Int32(offset + length)).asType(.float32)
        // positions: (L,), invFreq: (dim/2,) → angles: (L, dim/2)
        let angles = positions.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        return (cos: cos(angles), sin: sin(angles))
    }
}

// MARK: - Attention (MLA with sinks + inverse RoPE + grouped O)

class DeepseekV4Attention: Module {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let qLoraRank: Int
    let oGroups: Int
    let oLoraRank: Int
    /// Per-layer compress_ratio ∈ {0, 4, 128}. 0 = no compressor, plain
    /// sliding-window attention. 4 or 128 = Compressor (+ Indexer at 4)
    /// augments local KV with pooled global context.
    let compressRatio: Int
    let scale: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    // wo_a operates on PER-GROUP features (numHeads*headDim // oGroups),
    // mapping them to oGroups*oLoraRank via einsum bsgd,grd→bsgr.
    // Python: Linear(n_heads*head_dim // o_groups, o_groups*o_lora_rank).
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear
    /// q_norm is on `q_lora_rank` (1024), NOT head_dim. Applied BEFORE wq_b.
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    /// Shape (num_heads,) — one learned sink logit per head.
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray

    // The official grouped output projection uses a dequantized wo_a tensor.
    // Keep the evaluated immutable tensor across tokens instead of rebuilding
    // all 43 layer weights on every decode step. The opt-out is retained for
    // constrained-memory deployments and performance regression testing.
    private var cachedDequantizedWoA: MLXArray?
    private let dequantizedWoALock = NSLock()

    private static let cacheDequantizedWoA: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV4_CACHE_WOA"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }()

    private static let compileAttentionPostDecode = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_COMPILE_ATTN_POST", default: true)

    private static let compileAttentionPreDecode = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_COMPILE_ATTN_PRE", default: true)

    private static let quantizedGroupedWoA = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_QUANTIZED_GROUPED_WOA", default: true)

    private static let quantizedGroupedWoAActivationQAT = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_QUANTIZED_GROUPED_WOA_QAT", default: false)

    private lazy var compiledAttentionPreDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] args in
            self.projectAttentionInputs(args[0], cos: args[1], sin: args[2])
        }
        return compile(shapeless: false, body)
    }()

    private lazy var compiledAttentionPostDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] args in
            [self.projectAttentionOutput(
                args[0], cos: args[1], sin: args[2], groupedWeight: args[3])]
        }
        return compile(shapeless: false, body)
    }()

    private lazy var compiledQuantizedAttentionPostDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] args in
            [self.projectAttentionOutputQuantized(
                args[0], cos: args[1], sin: args[2],
                weight: args[3], scales: args[4])]
        }
        return compile(shapeless: false, body)
    }()

    let rope: DeepseekV4RoPE

    // Compressor + Indexer (instantiated only when compressRatio > 0).
    // Swift can't have conditionally-present @ModuleInfo properties
    // cleanly, so we instantiate always and null the pooled path inside
    // forward when compressRatio == 0.
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?

    init(
        config: DeepseekV4Configuration,
        layerIdx: Int,
        compressRatioOverride: Int? = nil
    ) {
        self.config = config
        self.layerIdx = layerIdx
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.ropeDim = config.qkRopeHeadDim
        self.qLoraRank = config.qLoraRank
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = 1.0 / sqrt(Float(headDim))

        // Resolve per-layer compress_ratio. If config.compressRatios is
        // populated use it directly; otherwise fall back to the default
        // DSV4-Flash pattern (layer 0 and last → 0; middle: odd → 4,
        // even → 128 per layer index after accounting for layer 0).
        if let compressRatioOverride {
            self.compressRatio = compressRatioOverride
        } else if !config.compressRatios.isEmpty && layerIdx < config.compressRatios.count {
            self.compressRatio = config.compressRatios[layerIdx]
        } else {
            let n = config.numHiddenLayers
            if layerIdx == 0 || layerIdx == n - 1 {
                self.compressRatio = 0
            } else {
                let i = layerIdx - 1
                self.compressRatio = (i % 2 == 1) ? 4 : 128
            }
        }

        self._wqA.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: false)
        self._wqB.wrappedValue = Linear(qLoraRank, numHeads * headDim, bias: false)
        self._wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        // wo_a: per-group features (n_heads*head_dim // o_groups) →
        // o_groups * o_lora_rank. For DSV4-Flash: 4096 → 8192.
        self._woA.wrappedValue = Linear(
            numHeads * headDim / oGroups, oGroups * oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(
            oGroups * oLoraRank, config.hiddenSize, bias: false)
        // q_norm operates on q_lora_rank (1024), not head_dim.
        self._qNorm.wrappedValue = RMSNorm(
            dimensions: qLoraRank, eps: config.rmsNormEps)
        self._kvNorm.wrappedValue = RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._attnSink.wrappedValue = zeros([numHeads])

        // RoPE: compressRatio>0 → compress_rope_theta (160000) + YaRN.
        // compressRatio==0 → rope_theta (10000), NO YaRN.
        let theta =
            compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta
        let factor: Float =
            compressRatio > 0
            ? Float((config.ropeScaling?["factor"]?.asFloat()) ?? 16.0)
            : 1.0
        let origMax =
            Int(
                (config.ropeScaling?["original_max_position_embeddings"]?.asInt()) ?? 65536)
        self.rope = DeepseekV4RoPE(
            dim: ropeDim, base: theta, factor: factor,
            origMaxPos: origMax, betaFast: 32, betaSlow: 1)

        // Compressor + Indexer are attached ONLY on layers with a
        // non-zero compress_ratio — matches bundle weight keys.
        if compressRatio > 0 {
            self._compressor.wrappedValue = DeepseekV4Compressor(
                config: config, compressRatio: compressRatio, headDim: headDim)
            if compressRatio == 4 {
                self._indexer.wrappedValue = DeepseekV4Indexer(
                    config: config, compressRatio: compressRatio)
            }
        }
    }

    private func groupedOutputWeight(dtype: DType) -> MLXArray {
        let groupFeatures = (numHeads * headDim) / oGroups
        guard let quantized = woA as? QuantizedLinear else {
            return woA.weight.reshaped(oGroups, oLoraRank, groupFeatures)
        }

        if Self.cacheDequantizedWoA {
            dequantizedWoALock.lock()
            defer { dequantizedWoALock.unlock() }
            if let cachedDequantizedWoA {
                return cachedDequantizedWoA
            }
            let value = MLX.dequantized(
                quantized.weight, scales: quantized.scales,
                biases: quantized.biases, groupSize: quantized.groupSize,
                bits: quantized.bits, mode: quantized.mode, dtype: dtype
            ).reshaped(oGroups, oLoraRank, groupFeatures)
            MLX.eval(value)
            cachedDequantizedWoA = value
            return value
        }

        return MLX.dequantized(
            quantized.weight, scales: quantized.scales,
            biases: quantized.biases, groupSize: quantized.groupSize,
            bits: quantized.bits, mode: quantized.mode, dtype: dtype
        ).reshaped(oGroups, oLoraRank, groupFeatures)
    }

    private func projectAttentionOutput(
        _ attention: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        groupedWeight: MLXArray
    ) -> MLXArray {
        let batch = attention.dim(0)
        let length = attention.dim(2)
        let groupFeatures = (numHeads * headDim) / oGroups
        let derotated = DeepseekV4Math.applyPartialRoPE(
            attention,
            cos: cos.expandedDimensions(axes: [0, 1]),
            sin: sin.expandedDimensions(axes: [0, 1]),
            ropeDim: ropeDim,
            inverse: true)
        let grouped = derotated.transposed(0, 2, 1, 3)
            .reshaped(batch, length, oGroups, groupFeatures)
        let projectedA = einsum("bsgd,grd->bsgr", grouped, groupedWeight)
            .reshaped(batch, length, oGroups * oLoraRank)
        return woB(projectedA)
    }

    /// Preserve wo_a in its checkpoint MXFP8 representation and use MLX's
    /// batched quantized matmul across the eight output groups. This avoids
    /// reading the approximately 67 MB BF16-expanded wo_a tensor in every
    /// layer on every generated token.
    private func projectAttentionOutputQuantized(
        _ attention: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        weight: MLXArray,
        scales: MLXArray
    ) -> MLXArray {
        let batch = attention.dim(0)
        let length = attention.dim(2)
        let groupFeatures = (numHeads * headDim) / oGroups
        let derotated = DeepseekV4Math.applyPartialRoPE(
            attention,
            cos: cos.expandedDimensions(axes: [0, 1]),
            sin: sin.expandedDimensions(axes: [0, 1]),
            ropeDim: ropeDim,
            inverse: true)
        var grouped = derotated.transposed(0, 2, 1, 3)
            .reshaped(batch, length, oGroups, groupFeatures)
        if Self.quantizedGroupedWoAActivationQAT {
            grouped = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(
                grouped, mode: .mxfp8)
        }

        let packedWeight = weight.reshaped(oGroups, oLoraRank, weight.dim(-1))
        let groupedScales = scales.reshaped(oGroups, oLoraRank, scales.dim(-1))
        let projectedA = quantizedMM(
            grouped.expandedDimensions(axis: -2),
            packedWeight,
            scales: groupedScales,
            biases: nil,
            transpose: true,
            groupSize: 32,
            bits: 8,
            mode: .mxfp8
        ).squeezed(axis: -2).reshaped(batch, length, oGroups * oLoraRank)
        return woB(projectedA)
    }

    private func projectAttentionOutputSymmetricQ8(
        _ attention: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        quantized: DeepseekV4QuantizedLinear
    ) -> MLXArray {
        let batch = attention.dim(0)
        let length = attention.dim(2)
        let groupFeatures = (numHeads * headDim) / oGroups
        let derotated = DeepseekV4Math.applyPartialRoPE(
            attention,
            cos: cos.expandedDimensions(axes: [0, 1]),
            sin: sin.expandedDimensions(axes: [0, 1]),
            ropeDim: ropeDim,
            inverse: true)
        let grouped = derotated.transposed(0, 2, 1, 3)
            .reshaped(batch, length, oGroups, groupFeatures)
        let projectedA = quantized.symmetricQ8Grouped(grouped, outputGroups: oGroups)
            .reshaped(batch, length, oGroups * oLoraRank)
        return woB(projectedA)
    }

    private func projectAttentionInputs(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> [MLXArray] {
        let batch = x.dim(0)
        let length = x.dim(1)
        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual).reshaped(batch, length, numHeads, headDim)
        let qDType = q.dtype
        let qFloat = q.asType(.float32)
        q = (qFloat * rsqrt(
            (qFloat * qFloat).mean(axis: -1, keepDims: true)
                + config.rmsNormEps)).asType(qDType)
        q = q.transposed(0, 2, 1, 3)

        var kv = kvNorm(wkv(x))
            .reshaped(batch, length, 1, headDim)
            .transposed(0, 2, 1, 3)
        let cosExpanded = cos.expandedDimensions(axes: [0, 1])
        let sinExpanded = sin.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(
            q, cos: cosExpanded, sin: sinExpanded, ropeDim: ropeDim)
        kv = DeepseekV4Math.applyPartialRoPE(
            kv, cos: cosExpanded, sin: sinExpanded, ropeDim: ropeDim)
        let nopeDim = headDim - ropeDim
        if config.activationQATEnabled && nopeDim >= 64 {
            kv = DeepseekV4Math.e4m3KVActivationRoundTrip(kv, ropeDim: ropeDim)
        }
        return [q, kv, qResidual]
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let offset = cache?.offset ?? 0
        let (cosT, sinT) = rope.cosSin(offset: offset, length: L)

        if Self.compileAttentionPreDecode,
            L == 1,
            Device.defaultDevice().deviceType == .gpu,
            !DeepseekV4NumericTrace.enabled
        {
            let projected = compiledAttentionPreDecode([x, cosT, sinT])
            return attend(
                x: x,
                q: projected[0],
                kv: projected[1],
                qResidual: projected[2],
                cosT: cosT,
                sinT: sinT,
                mask: mask,
                cache: cache,
                offset: offset)
        }

        // --- Q projection ---
        // wq_a(x): (B, L, qLoraRank) → q_norm on qLoraRank → wq_b:
        // (B, L, numHeads*headDim). Keep the post-qnorm residual — the
        // Indexer uses it as its own Q source.
        let qA = wqA(x)
        let qResidual = qNorm(qA)
        var q = wqB(qResidual)
        if layerIdx == 0 && DeepseekV4NumericTrace.enabled {
            if let quantized = wqA as? QuantizedLinear {
                let activation = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(
                    x, mode: quantized.mode)
                let dequantized = MLX.dequantized(
                    quantized.weight, scales: quantized.scales,
                    biases: quantized.biases, groupSize: quantized.groupSize,
                    bits: quantized.bits, mode: quantized.mode, dtype: x.dtype)
                DeepseekV4NumericTrace.tensor(
                    "layer.0.attention.wq_a.activation", activation)
                DeepseekV4NumericTrace.tensor(
                    "layer.0.attention.wq_a.weight", dequantized)
                DeepseekV4NumericTrace.tensor(
                    "layer.0.attention.wq_a.direct", activation.matmul(dequantized.transposed()))
            }
            DeepseekV4NumericTrace.tensor("layer.0.attention.wq_a", qA)
            DeepseekV4NumericTrace.tensor("layer.0.attention.q_norm", qResidual)
            DeepseekV4NumericTrace.tensor("layer.0.attention.wq_b", q)
        }
        q = q.reshaped(B, L, numHeads, headDim)
        // The 0731 reference computes the statistic explicitly in fp32.
        // The fused RMSNorm changes MXFP8 attention numerics materially.
        let qDType = q.dtype
        let qFloat = q.asType(.float32)
        q = (qFloat * rsqrt(
            (qFloat * qFloat).mean(axis: -1, keepDims: true)
                + config.rmsNormEps)).asType(qDType)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attention.q_head_norm", q)
        }
        q = q.transposed(0, 2, 1, 3)

        // --- KV projection (single latent head) ---
        let kvProjected = wkv(x)
        var kv = kvNorm(kvProjected)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attention.wkv", kvProjected)
            DeepseekV4NumericTrace.tensor("layer.0.attention.kv_norm", kv)
        }
        kv = kv.reshaped(B, L, 1, headDim).transposed(0, 2, 1, 3)

        // --- Partial RoPE on last ropeDim dims of Q and K ---
        let cosQ = cosT.expandedDimensions(axes: [0, 1])
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(q, cos: cosQ, sin: sinQ, ropeDim: ropeDim)
        kv = DeepseekV4Math.applyPartialRoPE(kv, cos: cosQ, sin: sinQ, ropeDim: ropeDim)

        // Pinned 0731 graph contract: fake-quantize the 448 non-RoPE KV
        // dimensions in block-64 E4M3FN immediately after RoPE and before the
        // row enters any local or disk-backed cache. Tiny synthetic configs
        // use sub-64 heads and intentionally skip this production-only op.
        let nopeDim = headDim - ropeDim
        if config.activationQATEnabled && nopeDim >= 64 {
            kv = DeepseekV4Math.e4m3KVActivationRoundTrip(kv, ropeDim: ropeDim)
        }
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attention.q", q)
            DeepseekV4NumericTrace.tensor("layer.0.attention.kv", kv)
        }

        return attend(
            x: x,
            q: q,
            kv: kv,
            qResidual: qResidual,
            cosT: cosT,
            sinT: sinT,
            mask: mask,
            cache: cache,
            offset: offset)
    }

    private func attend(
        x: MLXArray,
        q: MLXArray,
        kv: MLXArray,
        qResidual: MLXArray,
        cosT: MLXArray,
        sinT: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // --- Cache update (sliding-window local) ---
        var keys = kv
        if let cache = cache {
            (keys, _) = cache.update(keys: kv, values: kv)
        }
        var fullKV = keys
        let windowLen = fullKV.dim(2)

        // --- Compressor + Indexer global context (compressRatio > 0 layers) ---
        //
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // Two paths now distinguished by query length:
        //
        //   * decode (L == 1): build `(B, 1, k, D)` of selected pool
        //     rows for the single query (or the whole pool if no topk
        //     gating fires) and concat onto `full_kv`. No mask needed —
        //     the only query is causally OK against every selected row
        //     because the indexer enforces `(k_idx + 1) * ratio <= q + 1`
        //     in scoring, and the compressor only emits pool rows whose
        //     summarized window has fully ended.
        //
        //   * prefill (L > 1): keep the pool flat at `(B, 1, P, D)` and
        //     build a 2-segment mask `[window_visibility | comp_visibility]`
        //     so each query sees only the local window keys it should
        //     AND only the pool rows whose summarized window ended at
        //     or before that query's position. ANDed with the indexer's
        //     selection mask on `cr=4` layers. The previous implementation
        //     padded the mask with all-ones, allowing query `q` to see
        //     pool rows summarizing tokens with positions > q, AND
        //     gathered `(B, 1, L*k, D)` — leaking query `i`'s selected
        //     rows into query `j`'s attention.
        var dsv4PrefillMask: MLXArray? = nil
        var poolEntries: Int = 0
        if compressRatio > 0 {
            let v4Cache = cache as? DeepseekV4Cache
            if v4Cache != nil || L >= compressRatio {
                if let comp = compressor {
                    var pooled = comp(x, rope: rope, v4Cache: v4Cache, startPos: offset)
                    // pooled shape: (B, W, headDim) where W = pooled count.
                    let W = pooled.dim(1)
                    var topK: MLXArray? = nil
                    if compressRatio == 4, let idx = indexer {
                        // The indexer's learned compressor owns a separate
                        // cache branch. Advance it even before either branch
                        // emits its first complete row, so partial windows and
                        // all pre-topK history are present when learned
                        // selection first activates.
                        topK = idx(
                            x, qResidual: qResidual, rope: rope,
                            positionRope: rope, v4Cache: v4Cache, startPos: offset)
                    }
                    if W > 0 {

                        if L == 1 {
                            // DECODE FAST PATH — gather only the topk
                            // rows for the single query (or all rows
                            // when topk == nil / W <= topK), shape
                            // `(B, 1, k, D)`.
                            if let tk = topK {
                                let k = tk.dim(-1)
                                // pooled: (B, W, D) → (B, 1, 1, W, D)
                                let expanded = pooled.expandedDimensions(axes: [1, 2])
                                let pooledBroad = broadcast(
                                    expanded, to: [B, 1, L, W, headDim])
                                // tk: (B, L=1, k) → (B, 1, L, k, 1)
                                let idxExp = tk.expandedDimensions(axes: [1, 4])
                                let idxBroad = broadcast(
                                    idxExp, to: [B, 1, L, k, headDim])
                                let gathered = takeAlong(
                                    pooledBroad, idxBroad, axis: 3)
                                // (B, 1, k, D)
                                pooled = gathered.reshaped(B, 1, k, headDim)
                            } else {
                                pooled = pooled.expandedDimensions(axis: 1)
                            }
                        } else {
                            // PREFILL PATH — flat pool, mask carries
                            // visibility.
                            pooled = pooled.expandedDimensions(axis: 1)
                            // local sliding-window visibility (B,1,L,windowLen)
                            let localMask = DeepseekV4Math.buildWindowMask(
                                batch: B, queryLen: L, offset: offset,
                                window: config.slidingWindow,
                                windowLen: windowLen)
                            // compressed-pool causal visibility (B,1,L,W)
                            var compMask = DeepseekV4Math.compressedVisibility(
                                batch: B, queryLen: L, offset: offset,
                                compressedLen: W, ratio: compressRatio)
                            if let tk = topK {
                                let sel = DeepseekV4Math.indexerSelectionMask(
                                    topk: tk, compressedLen: W)
                                compMask = MLX.logicalAnd(compMask, sel)
                            }
                            // Pre-broadcast both halves to the same query
                            // dim (already done by helpers); concat along
                            // last axis.
                            dsv4PrefillMask = concatenated(
                                [localMask, compMask], axis: -1)
                        }

                        if pooled.dim(2) > 0 {
                            poolEntries = pooled.dim(2)
                            fullKV = concatenated([fullKV, pooled], axis: 2)
                        }
                    }
                }
            }
        }

        // --- Resolve final attention mask ---
        // Three cases:
        //   (a) DSV4-built prefill mask present → use it directly.
        //   (b) Caller-provided array mask → trim/pad to `fullKV.dim(2)`
        //       (legacy code path, also triggered for `cr == 0` SWA-only
        //        layers that bypass DSV4 mask construction).
        //   (c) Bool-causal sentinel from `createAttentionMask` → leave it
        //       alone; SDPA will compute the causal mask itself.
        var adjustedMask = mask
        if let dsv4 = dsv4PrefillMask {
            adjustedMask = .array(dsv4)
        } else if case .array(let maskArr) = mask,
            poolEntries > 0
        {
            // Decode path: extend the mask with all-ones for the pool
            // entries (every selected row is causally valid for the
            // single query — see above).
            let padShape =
                Array(maskArr.shape.dropLast()) + [fullKV.dim(2) - maskArr.dim(-1)]
            let pad = MLXArray.ones(padShape, dtype: maskArr.dtype)
            adjustedMask = .array(concatenated([maskArr, pad], axis: -1))
        } else if case .array(let maskArr) = mask,
            fullKV.dim(2) != maskArr.dim(-1)
        {
            // Defensive: align array mask to actual key length.
            if maskArr.dim(-1) > fullKV.dim(2) {
                let trimmed = maskArr[.ellipsis, (-fullKV.dim(2))...]
                adjustedMask = .array(trimmed)
            } else {
                let padShape =
                    Array(maskArr.shape.dropLast()) + [fullKV.dim(2) - maskArr.dim(-1)]
                let pad = MLXArray.zeros(padShape, dtype: maskArr.dtype)
                adjustedMask = .array(concatenated([maskArr, pad], axis: -1))
            }
        }

        // --- SDPA with attention sinks ---
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: fullKV, values: fullKV,
            scale: scale, mask: adjustedMask,
            sinks: config.useAttnSink ? attnSink.asType(q.dtype) : nil)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attention.raw", output)
        }
        // output shape: (B, numHeads, L, headDim)

        if Self.quantizedGroupedWoA,
            let quantized = woA as? DeepseekV4QuantizedLinear,
            quantized.usesSymmetricQ8Storage
        {
            return projectAttentionOutputSymmetricQ8(
                output, cos: cosT, sin: sinT, quantized: quantized)
        }

        if Self.quantizedGroupedWoA,
            let quantized = woA as? QuantizedLinear,
            quantized.mode == .mxfp8,
            quantized.groupSize == 32,
            quantized.bits == 8,
            quantized.biases == nil
        {
            if Self.compileAttentionPostDecode,
                L == 1,
                Device.defaultDevice().deviceType == .gpu,
                !DeepseekV4NumericTrace.enabled
            {
                return compiledQuantizedAttentionPostDecode([
                    output, cosT, sinT, quantized.weight, quantized.scales,
                ])[0]
            }
            return projectAttentionOutputQuantized(
                output,
                cos: cosT,
                sin: sinT,
                weight: quantized.weight,
                scales: quantized.scales)
        }

        // Keep dequantization and its evaluated cache outside the compiled
        // graph. Only the stateless inverse-RoPE and grouped projection tail
        // is compiled, so mutable KV cache state remains explicit above.
        let groupedWeight = groupedOutputWeight(dtype: output.dtype)
        if Self.compileAttentionPostDecode,
            L == 1,
            Device.defaultDevice().deviceType == .gpu,
            !DeepseekV4NumericTrace.enabled
        {
            return compiledAttentionPostDecode([output, cosT, sinT, groupedWeight])[0]
        }

        if layerIdx == 0 {
            let derotated = DeepseekV4Math.applyPartialRoPE(
                output,
                cos: cosT.expandedDimensions(axes: [0, 1]),
                sin: sinT.expandedDimensions(axes: [0, 1]),
                ropeDim: ropeDim,
                inverse: true)
            DeepseekV4NumericTrace.tensor("layer.0.attention.derotated", derotated)
        }
        let projected = projectAttentionOutput(
            output,
            cos: cosT,
            sin: sinT,
            groupedWeight: groupedWeight)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attention.oproj_b", projected)
        }
        return projected
    }

    /// DSpARK attention persists target hidden-state KV only; draft KV is
    /// temporary to the current proposal block.
    func dsparkForward(
        _ x: MLXArray,
        mainX: MLXArray,
        cache: KVCache
    ) -> MLXArray {
        precondition(compressRatio == 0, "DSpARK attention must use local KV")
        let offset = cache.offset
        let B = mainX.dim(0)
        let mainLength = mainX.dim(1)
        var mainKV = kvNorm(wkv(mainX))
            .reshaped(B, mainLength, 1, headDim).transposed(0, 2, 1, 3)
        let mainRope = rope.cosSin(offset: offset, length: mainLength)
        mainKV = DeepseekV4Math.applyPartialRoPE(
            mainKV, cos: mainRope.cos.expandedDimensions(axes: [0, 1]),
            sin: mainRope.sin.expandedDimensions(axes: [0, 1]), ropeDim: ropeDim)
        if config.activationQATEnabled && headDim - ropeDim >= 64 {
            mainKV = DeepseekV4Math.e4m3KVActivationRoundTrip(mainKV, ropeDim: ropeDim)
        }
        let (persistentKV, _) = cache.update(keys: mainKV, values: mainKV)
        if offset == 0 { return x }

        let blockLength = x.dim(1)
        let draftOffset = offset + mainLength
        var q = wqB(qNorm(wqA(x))).reshaped(B, blockLength, numHeads, headDim)
        let qDType = q.dtype
        let qF32 = q.asType(.float32)
        q = (qF32 * rsqrt(
            (qF32 * qF32).mean(axis: -1, keepDims: true) + config.rmsNormEps
        )).asType(qDType).transposed(0, 2, 1, 3)
        var draftKV = kvNorm(wkv(x))
            .reshaped(B, blockLength, 1, headDim).transposed(0, 2, 1, 3)
        let draftRope = rope.cosSin(offset: draftOffset, length: blockLength)
        let draftCos = draftRope.cos.expandedDimensions(axes: [0, 1])
        let draftSin = draftRope.sin.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(q, cos: draftCos, sin: draftSin, ropeDim: ropeDim)
        draftKV = DeepseekV4Math.applyPartialRoPE(
            draftKV, cos: draftCos, sin: draftSin, ropeDim: ropeDim)
        if config.activationQATEnabled && headDim - ropeDim >= 64 {
            draftKV = DeepseekV4Math.e4m3KVActivationRoundTrip(draftKV, ropeDim: ropeDim)
        }
        let allKV = concatenated([persistentKV, draftKV], axis: 2)
        var output = MLXFast.scaledDotProductAttention(
            queries: q, keys: allKV, values: allKV, scale: scale, mask: .none,
            sinks: config.useAttnSink ? attnSink.asType(q.dtype) : nil)
        output = DeepseekV4Math.applyPartialRoPE(
            output, cos: draftCos, sin: draftSin, ropeDim: ropeDim, inverse: true)
        output = output.transposed(0, 2, 1, 3)
            .reshaped(B, blockLength, numHeads * headDim)
        let groupFeatures = (numHeads * headDim) / oGroups
        let grouped = output.reshaped(B, blockLength, oGroups, groupFeatures)
        let projectedA: MLXArray
        if let quantized = woA as? QuantizedLinear {
            let weight: MLXArray
            if Self.cacheDequantizedWoA {
                dequantizedWoALock.lock()
                if let cachedDequantizedWoA { weight = cachedDequantizedWoA } else {
                    let value = MLX.dequantized(
                        quantized.weight, scales: quantized.scales,
                        biases: quantized.biases, groupSize: quantized.groupSize,
                        bits: quantized.bits, mode: quantized.mode, dtype: output.dtype
                    ).reshaped(oGroups, oLoraRank, groupFeatures)
                    MLX.eval(value); cachedDequantizedWoA = value; weight = value
                }
                dequantizedWoALock.unlock()
            } else {
                weight = MLX.dequantized(
                    quantized.weight, scales: quantized.scales,
                    biases: quantized.biases, groupSize: quantized.groupSize,
                    bits: quantized.bits, mode: quantized.mode, dtype: output.dtype
                ).reshaped(oGroups, oLoraRank, groupFeatures)
            }
            projectedA = einsum("bsgd,grd->bsgr", grouped, weight)
                .reshaped(B, blockLength, oGroups * oLoraRank)
        } else {
            projectedA = einsum(
                "bsgd,grd->bsgr", grouped,
                woA.weight.reshaped(oGroups, oLoraRank, groupFeatures)
            ).reshaped(B, blockLength, oGroups * oLoraRank)
        }
        return woB(projectedA)
    }
}

// MARK: - Embedded DSpARK drafter

/// Low-rank token-transition prior used by the final DSpARK stage.
class DeepseekV4DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var markovW1: Embedding
    @ModuleInfo(key: "markov_w2") var markovW2: Linear

    init(config: DeepseekV4Configuration) {
        self._markovW1.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.dsparkMarkovRank)
        self._markovW2.wrappedValue = Linear(
            config.dsparkMarkovRank,
            config.vocabSize,
            bias: false)
    }

    func callAsFunction(_ tokenIds: MLXArray) -> (logits: MLXArray, embedding: MLXArray) {
        let embedding = markovW1(tokenIds)
        if DeepseekV4RuntimeOptions.dsparkNativeHeadEnabled,
           let logits = DeepseekV4Math.dsparkHeadFp32(embedding, linear: markovW2)
        {
            return (logits, embedding)
        }
        return (markovW2(embedding), embedding)
    }
}

class DeepseekV4DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(config: DeepseekV4Configuration) {
        self._projection.wrappedValue = Linear(
            config.hiddenSize + config.dsparkMarkovRank,
            1,
            bias: false)
    }

    func callAsFunction(_ hidden: MLXArray, markovEmbedding: MLXArray) -> MLXArray {
        projection(concatenated([hidden, markovEmbedding], axis: -1)).squeezed(axis: -1)
    }
}

public struct DeepseekV4DSparkProposal {
    public let tokenIds: MLXArray
    public let logits: MLXArray
    public let confidence: MLXArray
}

/// Optional checkpoint-backed DSpARK stage. Loading it does not alter the
/// ordinary autoregressive execution path.
class DeepseekV4DSparkStage: Module {
    @ModuleInfo(key: "attn") var attention: DeepseekV4Attention
    @ModuleInfo(key: "ffn") var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attentionHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "main_proj") var mainProjection: Linear?
    @ModuleInfo(key: "main_norm") var mainNorm: RMSNorm?
    @ModuleInfo(key: "norm") var outputNorm: RMSNorm?
    @ModuleInfo(key: "markov_head") var markovHead: DeepseekV4DSparkMarkovHead?
    @ModuleInfo(key: "confidence_head") var confidenceHead: DeepseekV4DSparkConfidenceHead?
    @ParameterInfo(key: "hc_head_fn") var headFn: MLXArray?
    @ParameterInfo(key: "hc_head_base") var headBase: MLXArray?
    @ParameterInfo(key: "hc_head_scale") var headScale: MLXArray?
    let stageIndex: Int
    let config: DeepseekV4Configuration

    init(config: DeepseekV4Configuration, stageIndex: Int) {
        self.config = config
        self.stageIndex = stageIndex
        self._attention.wrappedValue = DeepseekV4Attention(
            config: config, layerIdx: config.numHiddenLayers + stageIndex,
            compressRatioOverride: 0)
        self._ffn.wrappedValue = DeepseekV4MoE(
            config: config, layerIdx: config.numHiddenLayers + stageIndex)
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attentionHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        if stageIndex == 0 {
            self._mainProjection.wrappedValue = Linear(
                config.hiddenSize * config.dsparkTargetLayerIds.count,
                config.hiddenSize, bias: false)
            self._mainNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if stageIndex == config.dsparkStageCount - 1 {
            self._outputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._markovHead.wrappedValue = DeepseekV4DSparkMarkovHead(config: config)
            self._confidenceHead.wrappedValue = DeepseekV4DSparkConfidenceHead(config: config)
            self._headFn.wrappedValue = zeros([
                config.hcMult, config.hcMult * config.hiddenSize,
            ])
            self._headBase.wrappedValue = zeros([config.hcMult])
            self._headScale.wrappedValue = zeros([1])
        }
    }

    func prepareDraft(
        mainHidden: MLXArray,
        anchorTokenIds: MLXArray,
        embedding: Embedding
    ) -> (hidden: MLXArray, main: MLXArray) {
        guard let mainProjection, let mainNorm else {
            preconditionFailure("Only DSpARK stage zero can prepare a draft")
        }
        let batch = anchorTokenIds.dim(0)
        let anchors = anchorTokenIds.reshaped(batch, 1)
        let noise = MLXArray.ones(
            [batch, max(0, config.dsparkBlockSize - 1)], dtype: .int32)
            * MLXArray(Int32(config.dsparkNoiseTokenId))
        let draftIds = concatenated([anchors.asType(.int32), noise], axis: 1)
        var hidden = embedding(draftIds).expandedDimensions(axis: -2)
        hidden = repeated(hidden, count: config.hcMult, axis: -2)
        return (hidden, mainNorm(mainProjection(mainHidden)))
    }

    func forward(
        _ hidden: MLXArray,
        main: MLXArray,
        anchorTokenIds: MLXArray,
        cache: KVCache
    ) -> MLXArray {
        if cache.offset == 0 {
            _ = attention.dsparkForward(hidden, mainX: main, cache: cache)
            return hidden
        }
        let attentionResidual = hidden
        let (attentionInput, attentionPost, attentionComb) = attentionHC.collapse(hidden)
        let attentionOutput = attention.dsparkForward(
            attentionNorm(attentionInput), mainX: main, cache: cache)
        let afterAttention = attentionHC.expand(
            blockOut: attentionOutput, residual: attentionResidual,
            post: attentionPost, comb: attentionComb)
        let ffnResidual = afterAttention
        let (ffnInput, ffnPost, ffnComb) = ffnHC.collapse(afterAttention)
        ffn.currentInputIds = anchorTokenIds
        let ffnOutput = ffn(ffnNorm(ffnInput))
        ffn.currentInputIds = nil
        return ffnHC.expand(
            blockOut: ffnOutput, residual: ffnResidual,
            post: ffnPost, comb: ffnComb)
    }

    func makeProposal(
        _ hidden: MLXArray,
        anchorTokenIds: MLXArray,
        lmHead: Linear
    ) -> DeepseekV4DSparkProposal {
        guard let outputNorm, let markovHead, let confidenceHead,
            let headFn, let headBase, let headScale
        else {
            preconditionFailure("Only the final DSpARK stage can produce tokens")
        }
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let flattened = hidden.reshaped(
            batch, length, config.hcMult * config.hiddenSize)
        let normalized = MLXFast.rmsNorm(
            flattened.asType(.float32),
            weight: MLXArray.ones([config.hcMult * config.hiddenSize]),
            eps: config.rmsNormEps)
        let mixes = normalized.matmul(headFn.asType(.float32).transposed())
        let coefficients = sigmoid(
            mixes * headScale.asType(.float32) + headBase.asType(.float32))
            + MLXArray(config.hcEps)
        let reduced = (
            coefficients.asType(hidden.dtype).expandedDimensions(axis: -1) * hidden
        ).sum(axis: -2)
        let normalizedHead = outputNorm(reduced)
        let baseLogits: MLXArray
        if DeepseekV4RuntimeOptions.dsparkNativeHeadEnabled,
           let native = DeepseekV4Math.dsparkHeadFp32(normalizedHead, linear: lmHead)
        {
            baseLogits = native
        } else {
            baseLogits = DeepseekV4Math.lmHeadFp32(normalizedHead, lmHead: lmHead)
        }
        var outputIds = anchorTokenIds.reshaped(batch, 1).asType(.int32)
        var biasedLogits: [MLXArray] = []
        var markovEmbeddings: [MLXArray] = []
        for index in 0..<config.dsparkBlockSize {
            let current = outputIds[0..., index]
            let markov = markovHead(current)
            let logits = baseLogits[0..., index, 0...] + markov.logits
            biasedLogits.append(logits)
            markovEmbeddings.append(markov.embedding)
            let next = argMax(logits, axis: -1).asType(.int32).expandedDimensions(axis: 1)
            outputIds = concatenated([outputIds, next], axis: 1)
        }
        let markovStack = stacked(markovEmbeddings, axis: 1)
        return DeepseekV4DSparkProposal(
            tokenIds: outputIds,
            logits: stacked(biasedLogits, axis: 1),
            confidence: confidenceHead(reduced, markovEmbedding: markovStack))
    }
}

// MARK: - MoE gate (sqrtsoftplus + hash routing)

private struct DeepseekV4SelectorKey: Hashable {
    let topK: Int
    let normalize: Bool
}

private enum DeepseekV4CompiledSelectorCache {
    typealias Selector = @Sendable ([MLXArray]) -> [MLXArray]

    private static let lock = NSLock()
    nonisolated(unsafe) private static var selectors: [DeepseekV4SelectorKey: Selector] = [:]

    static func selector(topK: Int, normalize: Bool) -> Selector {
        let key = DeepseekV4SelectorKey(topK: topK, normalize: normalize)
        lock.lock()
        defer { lock.unlock() }
        if let cached = selectors[key] { return cached }

        // Mirrors the authoritative global `@mx.compile`
        // `sqrtsoftplus_select`. Parameters that alter graph structure are
        // captured in the cache key; the scaling factor remains an array input
        // so bundles with the same selector shape can share the safe stateless
        // trace without sharing model or cache state.
        let body: Selector = { args in
            let logits = args[0]
            let bias = args[1]
            let scalingFactor = args[2]
            let originalScores = sqrt(log1p(exp(logits)))
            let biasedScores = originalScores + bias
            let indices = argPartition(
                -biasedScores, kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
                .asType(.int32)
            var weights = takeAlong(originalScores, indices, axis: -1)
            if topK > 1 && normalize {
                weights = weights / weights.sum(axis: -1, keepDims: true)
            }
            weights = weights * scalingFactor
            return [indices, weights]
        }
        let compiled = compile(body)
        let nestedSafe: Selector = { args in
            if CompiledDecodeTrace.isActive {
                return body(args)
            }
            let result = compiled(args)
            return result.count == 2 ? result : body(args)
        }
        selectors[key] = nestedSafe
        return nestedSafe
    }
}

class DeepseekV4MoEGate: Module {
    let config: DeepseekV4Configuration
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let isHashLayer: Bool
    fileprivate let compiledSelector: DeepseekV4CompiledSelectorCache.Selector
    let scalingFactorArray: MLXArray
    private static let fusedRouterEnabled = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_FUSED_ROUTER", default: true)
    /// Gate projection weight: (nRoutedExperts, hiddenSize). Stored as a
    /// raw parameter (loaded via sanitize) rather than a Linear to allow
    /// the matmul to run in fp32 per the authoritative reference.
    @ParameterInfo(key: "weight") var weight: MLXArray
    /// Optional noaux bias added to scores for selection only. When
    /// absent the bias term is skipped.
    @ParameterInfo(key: "bias") var bias: MLXArray
    /// Hash routing lookup table (token_id → expert_id), shape (vocab,).
    /// Only populated for hash layers.
    @ParameterInfo(key: "tid2eid") var tid2eid: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.isHashLayer = config.isHashLayer(layerIdx)
        self.compiledSelector = DeepseekV4CompiledSelectorCache.selector(
            topK: config.numExpertsPerTok,
            normalize: config.normTopkProb)
        self.scalingFactorArray = MLXArray([config.routedScalingFactor])
        self._weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        self._bias.wrappedValue = zeros([nRoutedExperts])
        // Hash routing table: bundle ships (vocab, topK) — already
        // pre-stamped with which `topK` experts each token id should
        // route to, so the gate just gathers without computing scores.
        // Non-hash layers don't have this tensor, so we still allocate
        // a placeholder slot.
        self._tid2eid.wrappedValue =
            zeros([isHashLayer ? config.vocabSize : 1, isHashLayer ? topK : 1])
    }

    /// Returns (indices, weights) where indices has shape (B, L, topK)
    /// and weights has shape (B, L, topK).
    ///
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Hash layers now match the Python `Gate.__call__` reference
    /// (`jang_tools.dsv4.mlx_model.Gate.__call__`) — they gather the
    /// PER-TOKEN gate scores at the hash-selected expert ids instead
    /// of returning a synthetic uniform `routedScalingFactor / topK`.
    /// Without this fix every hash-routed layer collapsed all six
    /// selected experts to the same weight, throwing away the
    /// information the gate matmul + sqrtsoftplus produced and
    /// flattening the routing geometry the model was trained with.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> (MLXArray, MLXArray) {
        if isHashLayer, let ids = inputIds {
            // Hash routing: tid2eid is (vocab, topK) — pre-stamped at
            // convert time with which topK experts each token id
            // routes to. `tid2eid[ids]` for ids shape (B, L) returns
            // (B, L, topK) directly via fancy index.
            let indices = tid2eid[ids].asType(.int32)  // (B, L, topK)
            let logits = x.asType(.float32).matmul(
                weight.asType(.float32).transposed())
            let selectedLogits = takeAlong(logits, indices, axis: -1)
            // Gate the experts using their actual sqrtsoftplus score. This is
            // algebraically identical to scoring all experts then gathering.
            var weights = DeepseekV4Math.sqrtSoftplus(selectedLogits)
            if normTopkProb {
                let denom = weights.sum(axis: -1, keepDims: true) + 1e-20
                weights = weights / denom
            }
            weights = weights * routedScalingFactor
            return (indices.asType(.uint32), weights)
        }

        // Non-hash: the same stateless compiled sqrtsoftplus + noaux-biased
        // top-k microfunction used by the authoritative affine runtime.
        let logits = x.asType(.float32).matmul(weight.asType(.float32).transposed())
        if Self.fusedRouterEnabled,
           x.dim(1) == 1,
           let selected = DeepseekV4Math.fusedSqrtSoftplusSelect(
               logits: logits,
               bias: bias,
               k: topK,
               normalize: normTopkProb,
               scalingFactor: scalingFactorArray)
        {
            return (selected.indices.asType(.uint32), selected.weights)
        }
        let selected = compiledSelector([logits, bias, scalingFactorArray])
        let indices = selected[0]
        let weights = selected[1]
        return (indices.asType(.uint32), weights)
    }
}

// MARK: - MoE (SwitchGLU routed + shared expert)

class DeepseekV4MoE: Module, UnaryLayer {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let topK: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP
    /// Hack to thread the input token ids down into the gate when this
    /// layer is hash-routed. Set by the outer model before each layer
    /// call when hash routing applies.
    var currentInputIds: MLXArray? = nil
    private static let compileMoEDecode =
        ProcessInfo.processInfo.environment["VMLX_DSV4_COMPILE_MOE"] == "1"
    private static let stagedSelectorDecode = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_STAGED_SELECTOR", default: false)
    private lazy var compiledDecode: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        let body: (MLXArray, MLXArray) -> MLXArray = { [unowned self] x, ids in
            CompiledDecodeTrace.withActive {
                self.forward(x, inputIds: ids)
            }
        }
        return compile(shapeless: false, body)
    }()
    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.topK = config.numExpertsPerTok
        let limit = config.swigluLimit
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // Symmetric DSV4 limited-SwiGLU — `silu(min(gate, limit)) *
        // clip(up, -limit, +limit)`. We pass a 2-arg `glue` closure to
        // SwitchGLU (instead of a 1-arg `activation`) so BOTH gate and
        // up get clamped before the multiply. The Python reference
        // (`jang_tools.dsv4.mlx_model._dsv4_swiglu`) also runs the
        // multiply in fp32 before casting back to gate.dtype to avoid
        // per-layer precision drift across the 43 MoE layers; we mirror
        // that here.
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: MLXNN.silu,
            glue: { gate, up in
                DeepseekV4Math.dsv4SwiGLU(gate: gate, up: up, limit: limit)
            },
            scoredGlue: { gate, up, scores in
                DeepseekV4Math.dsv4ScoredSwiGLU(
                    gate: gate, up: up, scores: scores, limit: limit)
            },
            scoredSwiGLULimit: limit)
        self.gate = DeepseekV4MoEGate(config: config, layerIdx: layerIdx)
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: limit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Self.compileMoEDecode && x.dim(1) == 1 {
            let ids = currentInputIds
                ?? MLXArray.zeros([x.dim(0), x.dim(1)], dtype: .uint32)
            return compiledDecode(x, ids)
        }
        return forward(x, inputIds: currentInputIds)
    }

    fileprivate func forward(_ x: MLXArray, inputIds: MLXArray?) -> MLXArray {
        let profileStages =
            DeepseekV4PerformanceProfile.enabled && !CompiledDecodeTrace.isActive
        var stageStart = profileStages ? CFAbsoluteTimeGetCurrent() : 0
        func finishStage(_ name: String, _ arrays: [MLXArray]) {
            guard profileStages else { return }
            MLX.eval(arrays)
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(Data(String(format:
                "[DSV4MoEProfile] layer=%d stage=%@ ms=%.3f\n",
                layerIdx, name, (now - stageStart) * 1_000).utf8))
            stageStart = now
        }

        if Self.stagedSelectorDecode && !gate.isHashLayer && x.size == config.hiddenSize {
            let logits = x.asType(.float32).matmul(
                gate.weight.asType(.float32).transposed())
            let sharedGate = sharedExperts.gateProj as? DeepseekV4QuantizedLinear
            let sharedUp = sharedExperts.upProj as? DeepseekV4QuantizedLinear
            let sharedDown = sharedExperts.downProj as? DeepseekV4QuantizedLinear
            if let sharedGate, let sharedUp, let sharedDown,
               let output = switchMLP.deepseekV4FusedSelectingMoEWithSharedQ8(
                   x,
                   logits: logits,
                   bias: gate.bias.asType(.float32),
                   routeScale: gate.scalingFactorArray.asType(.float32),
                   sharedGate: sharedGate,
                   sharedUp: sharedUp,
                   sharedDown: sharedDown)
            {
                finishStage("router_routed_and_shared", [output])
                return output
            }
            if let routed = switchMLP.deepseekV4FusedSelectingMoE(
                x,
                logits: logits,
                bias: gate.bias.asType(.float32),
                routeScale: gate.scalingFactorArray.asType(.float32))
            {
                finishStage("router_and_routed", [routed])
                let shared = sharedExperts(x)
                finishStage("shared", [shared])
                let output = DeepseekV4Math.addSharedExpertFP32(
                    routed, shared: shared, outputDType: x.dtype)
                finishStage("add", [output])
                return output
            }
        }

        let (indices, scores) = gate(x, inputIds: inputIds)
        finishStage("gate", [indices, scores])
        let routed = switchMLP(x, indices, preDownScores: scores)
        var y = routed.ndim == x.ndim
            ? routed.asType(.float32)
            : DeepseekV4Math.reduceRoutedExpertsFP32(routed)
        finishStage("routed", [y])
        finishStage("route_reduce", [y])
        let shared = sharedExperts(x)
        finishStage("shared", [shared])
        y = DeepseekV4Math.addSharedExpertFP32(
            y, shared: shared, outputDType: x.dtype)
        finishStage("add", [y])
        return y
    }
}

// MARK: - Dense MLP (shared expert) with DSV4 SwiGLU clamp

class DeepseekV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.swigluLimit = swigluLimit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gateProj(x)
        let u = upProj(x)
        return downProj(DeepseekV4Math.dsv4SwiGLU(gate: g, up: u, limit: swigluLimit))
    }
}

// MARK: - mHC Hyper-Connection (per-block collapse + expand)

class DeepseekV4HyperConnection: Module {
    let hcMult: Int
    let hcIters: Int
    let hcEps: Float
    let hiddenSize: Int
    let mixHc: Int  // (2 + hcMult) * hcMult — bundle stores params at this width
    /// `hc_{attn,ffn}_fn`: shape `((2+hc)*hc, hc*hidden)`. Bundle stores
    /// `(24, 16384)` for hc=4, hidden=4096.
    @ParameterInfo(key: "fn") var fn: MLXArray
    /// `hc_{attn,ffn}_scale`: shape `(3,)` per-field scalar.
    @ParameterInfo(key: "scale") var scale: MLXArray
    /// `hc_{attn,ffn}_base`: shape `((2+hc)*hc,)` per-field bias.
    @ParameterInfo(key: "base") var base: MLXArray
    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hcIters = config.hcSinkhornIters
        self.hcEps = config.hcEps
        self.hiddenSize = config.hiddenSize
        self.mixHc = (2 + config.hcMult) * config.hcMult
        self._fn.wrappedValue = zeros([mixHc, hcMult * hiddenSize])
        self._scale.wrappedValue = zeros([3])
        self._base.wrappedValue = zeros([mixHc])
    }

    /// Collapse: `h` shape (B, L, hcMult, hiddenSize) → collapsed x
    /// (B, L, hiddenSize) plus `post` (B, L, hcMult) and `comb`
    /// (B, L, hcMult, hcMult) for the expand step.
    ///
    /// Mirrors Python `DeepseekV4DecoderLayer._hc_pre`:
    ///   x_flat   = flatten(h, axis=2)              # (B, L, hc*hidden)
    ///   x_normed = rms_norm(x_flat, ones, eps)
    ///   mixes    = x_normed @ fn.T                 # (B, L, mix_hc)
    ///   pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
    ///   y = sum(pre[..., None] * x_flat.reshape(B,L,hc,D), axis=2)
    func collapse(_ h: MLXArray) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        let dtype = h.dtype
        let B = h.dim(0)
        let L = h.dim(1)

        // Python keeps this entire path in fp32. It also applies the scalar
        // reciprocal RMS after the projection: `(x_flat @ fn.T) * rsqrt`.
        // Besides preserving its operation order, that avoids materializing a
        // normalized 16K-wide residual only to project it down to `mixHc`.
        let xFlat = h.reshaped(B, L, hcMult * hiddenSize).asType(.float32)
        let reciprocalRMS = rsqrt(
            (xFlat * xFlat).mean(axis: -1, keepDims: true) + hcEps)
        let mixes = xFlat.matmul(fn.asType(.float32).transposed()) * reciprocalRMS

        if DeepseekV4Math.fusedHC4Enabled && hcMult == 4 && L == 1
            && Device.defaultDevice().deviceType == .gpu
        {
            let fused = DeepseekV4Math.hcSplitSinkhornCollapse4(
                mixes: mixes,
                scale: scale,
                base: base,
                residual: xFlat.reshaped(B, L, hcMult, hiddenSize),
                hiddenSize: hiddenSize,
                iters: hcIters,
                eps: hcEps)
            return (
                x: fused.collapsed.asType(dtype),
                post: fused.post,
                comb: fused.comb)
        }

        let (pre, post, comb) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: hcIters, eps: hcEps)

        // y = sum(pre[..., None] * x_flat.reshape(B, L, hc, D), axis=2)
        let xReshape = xFlat.reshaped(B, L, hcMult, hiddenSize)
        let y = (pre.expandedDimensions(axis: -1) * xReshape).sum(axis: -2)
        return (x: y.asType(dtype), post: post, comb: comb)
    }

    /// Decode-only collapse plus the immediately following RMSNorm. The
    /// optimized kernel is deliberately gated while parity and throughput are
    /// qualified; all other shapes retain the ordinary MLX implementation.
    func collapseNormalized(
        _ h: MLXArray, normalization: RMSNorm
    ) -> (x: MLXArray, normalized: MLXArray, post: MLXArray, comb: MLXArray) {
        let dtype = h.dtype
        let B = h.dim(0)
        let L = h.dim(1)

        guard DeepseekV4Math.fusedHCNormEnabled,
            DeepseekV4Math.fusedHC4Enabled,
            hcMult == 4,
            L == 1,
            Device.defaultDevice().deviceType == .gpu,
            !DeepseekV4NumericTrace.enabled
        else {
            let ordinary = collapse(h)
            return (
                x: ordinary.x,
                normalized: normalization(ordinary.x),
                post: ordinary.post,
                comb: ordinary.comb)
        }

        let xFlat = h.reshaped(B, L, hcMult * hiddenSize).asType(.float32)
        let reciprocalRMS = rsqrt(
            (xFlat * xFlat).mean(axis: -1, keepDims: true) + hcEps)
        let mixes = xFlat.matmul(fn.asType(.float32).transposed()) * reciprocalRMS
        let fused = DeepseekV4Math.hcSplitSinkhornCollapseNorm4(
            mixes: mixes,
            scale: scale,
            base: base,
            residual: xFlat.reshaped(B, L, hcMult, hiddenSize),
            normWeight: normalization.weight,
            normEps: normalization.eps,
            hiddenSize: hiddenSize,
            iters: hcIters,
            hcEps: hcEps)
        return (
            x: fused.collapsed.asType(dtype),
            normalized: fused.normalized.asType(dtype),
            post: fused.post,
            comb: fused.comb)
    }

    /// Expand: given attn/ffn output `blockOut` (B, L, hiddenSize),
    /// residual (B, L, hcMult, hiddenSize), and the (post, comb) from
    /// the matching collapse, return new h (B, L, hcMult, hiddenSize).
    ///
    /// Mirrors Python `_hc_post`:
    ///   y = post[..., None] * x[..., None, :] + matmul(comb, residual)
    func expand(
        blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        let dtype = blockOut.dtype
        if DeepseekV4Math.fusedHC4Enabled && hcMult == 4 && residual.dim(1) == 1
            && Device.defaultDevice().deviceType == .gpu
        {
            return DeepseekV4Math.hcExpand4(
                blockOut: blockOut,
                residual: residual,
                post: post,
                comb: comb,
                hiddenSize: hiddenSize).asType(dtype)
        }
        // Match the 0731 MLX reference's broadcast/reduction axes exactly.
        let combResid = DeepseekV4Math.hcExpandResidual(
            comb: comb, residual: residual)
        // post: (B,L,hc) → (B,L,hc,1); blockOut: (B,L,D) → (B,L,1,D).
        let y = post.expandedDimensions(axis: -1)
            * blockOut.asType(.float32).expandedDimensions(axis: -2) + combResid
        return y.asType(dtype)
    }
}

// MARK: - HyperHead (top-of-model mHC reduce)

class DeepseekV4HyperHead: Module {
    let hcMult: Int
    let hiddenSize: Int
    let hcEps: Float
    /// Bundle stores `hc_head_fn` at `(hcMult, hcMult*hiddenSize)`.
    @ParameterInfo(key: "hc_head_fn") var fn: MLXArray
    @ParameterInfo(key: "hc_head_base") var base: MLXArray
    @ParameterInfo(key: "hc_head_scale") var scale: MLXArray
    /// Constant ones-vector for the RMS norm in `_hc_head_reduce`.
    let hcHeadRMSOnes: MLXArray

    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hiddenSize = config.hiddenSize
        self.hcEps = config.rmsNormEps
        self._fn.wrappedValue = zeros([hcMult, hcMult * hiddenSize])
        self._base.wrappedValue = zeros([hcMult])
        self._scale.wrappedValue = zeros([1])
        self.hcHeadRMSOnes = MLXArray.ones([config.hcMult * config.hiddenSize])
    }

    /// Reduce (B, L, hcMult, hiddenSize) → (B, L, hiddenSize). Mirrors
    /// Python `_hc_head_reduce`:
    ///   x_flat   = flatten(x, axis=2)            # (B, L, hc*hidden)
    ///   x_normed = rms_norm(x_flat, ones, eps)
    ///   mixes    = x_normed @ hc_head_fn.T       # (B, L, hc)
    ///   pre      = sigmoid(mixes * scale + base) + hc_eps
    ///   y        = sum(pre[..., None] * x_flat.reshape(B,L,hc,D), axis=2)
    /// NO sum-to-1 normalization — match the Python reference exactly.
    func reduce(_ h: MLXArray) -> MLXArray {
        let dtype = h.dtype
        let B = h.dim(0)
        let L = h.dim(1)
        let xFlat = h.reshaped(B, L, hcMult * hiddenSize)
        // Same dtype rule as `_hc_pre`: this RMS reduction spans
        // hcMult*hiddenSize (≈16K for DSV4-Flash). Apple GPUs differ in
        // implicit bf16 accumulation behavior, so keep the reduction and
        // the tiny gate projection in fp32, then cast the final mixed
        // residual back to the model dtype. This mirrors the jang-tools
        // HyperHead fix documented in DSV4-HC-PRE-FP32-CAST-FIX.
        let xNormed = MLXFast.rmsNorm(
            xFlat.asType(.float32),
            weight: hcHeadRMSOnes.asType(.float32),
            eps: hcEps)
        let mixes = xNormed.matmul(fn.asType(.float32).transposed())  // (B, L, hcMult)
        let pre = sigmoid(mixes * scale.asType(.float32) + base.asType(.float32))
            + MLXArray(hcEps)
        let xReshape = xFlat.reshaped(B, L, hcMult, hiddenSize)
        return (pre.asType(dtype).expandedDimensions(axis: -1) * xReshape).sum(axis: -2)
            .asType(dtype)
    }
}

// MARK: - Decoder layer (mHC wrap over attn + MoE)

class DeepseekV4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV4Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV4MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    let layerIdx: Int
    private static let compileAttentionHCDecode = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_COMPILE_ATTN_HC", default: true)
    private lazy var compiledAttentionHCDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] args in
            CompiledDecodeTrace.withActive {
                let collapsed = self.attnHC.collapseNormalized(
                    args[0], normalization: self.inputLayerNorm)
                return [
                    collapsed.x,
                    collapsed.normalized,
                    collapsed.post,
                    collapsed.comb,
                ]
            }
        }
        return compile(shapeless: false, body)
    }()
    private static let compileFFNDecode = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_COMPILE_FFN", default: true)
    private lazy var compiledLayerTailDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] args in
            CompiledDecodeTrace.withActive {
                let hA = self.attnHC.expand(
                    blockOut: args[0],
                    residual: args[1],
                    post: args[2],
                    comb: args[3])
                let residual = hA
                let collapsed = self.ffnHC.collapseNormalized(
                    hA, normalization: self.postAttentionLayerNorm)
                let output = self.mlp.forward(collapsed.normalized, inputIds: args[4])
                return [self.ffnHC.expand(
                    blockOut: output,
                    residual: residual,
                    post: collapsed.post,
                    comb: collapsed.comb)]
            }
        }
        return compile(shapeless: false, body)
    }()

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DeepseekV4MoE(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
    }

    /// Forward. `h` shape: (B, L, hcMult, hiddenSize).
    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray?
    ) -> MLXArray {
        // Explicit Release diagnostic only. Splitting the lazy graph at each
        // block boundary perturbs total throughput, but attributes the
        // remaining affine DSV4 gap without changing the production graph
        // when the variable is absent. The diagnostic intentionally accepts
        // every shape so AFM generation wrappers cannot silently bypass it.
        let profileStages =
            DeepseekV4PerformanceProfile.enabled
        var stageStart = profileStages ? CFAbsoluteTimeGetCurrent() : 0
        func finishStage(_ name: String, _ arrays: [MLXArray]) {
            guard profileStages else { return }
            MLX.eval(arrays)
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(Data(String(format:
                "[DSV4StageProfile] layer=%d shape=%@ stage=%@ ms=%.3f\n",
                layerIdx, String(describing: h.shape), name,
                (now - stageStart) * 1_000).utf8))
            stageStart = now
        }
        // ---- Attention HC ----
        let residualA = h
        let xA: MLXArray
        let normedA: MLXArray
        let postA: MLXArray
        let combA: MLXArray
        if Self.compileAttentionHCDecode && h.dim(1) == 1
            && !profileStages && !DeepseekV4NumericTrace.enabled
        {
            let collapsed = compiledAttentionHCDecode([h])
            xA = collapsed[0]
            normedA = collapsed[1]
            postA = collapsed[2]
            combA = collapsed[3]
        } else {
            let collapsed = attnHC.collapseNormalized(
                h, normalization: inputLayerNorm)
            xA = collapsed.x
            normedA = collapsed.normalized
            postA = collapsed.post
            combA = collapsed.comb
        }
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attn_hc_pre.x", xA)
            DeepseekV4NumericTrace.tensor("layer.0.attn_hc_pre.post", postA)
            DeepseekV4NumericTrace.tensor("layer.0.attn_hc_pre.comb", combA)
        }
        finishStage("attn_hc_pre", [xA, postA, combA])
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attn_norm", normedA)
        }
        finishStage("attn_norm", [normedA])
        let attnOut = selfAttn(normedA, mask: mask, cache: cache)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attention", attnOut)
        }
        finishStage("attention", [attnOut])
        if Self.compileFFNDecode && h.dim(1) == 1
            && !profileStages && !DeepseekV4NumericTrace.enabled
        {
            let ids = inputIds
                ?? MLXArray.zeros([h.dim(0), h.dim(1)], dtype: .uint32)
            return compiledLayerTailDecode([
                attnOut, residualA, postA, combA, ids,
            ])[0]
        }
        let hA = attnHC.expand(
            blockOut: attnOut, residual: residualA, post: postA, comb: combA)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.attn_hc_post", hA)
        }
        finishStage("attn_hc_post", [hA])

        // ---- FFN HC ----
        let residualF = hA
        let collapsedF = ffnHC.collapseNormalized(
            hA, normalization: postAttentionLayerNorm)
        let xF = collapsedF.x
        let postF = collapsedF.post
        let combF = collapsedF.comb
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.ffn_hc_pre.x", xF)
            DeepseekV4NumericTrace.tensor("layer.0.ffn_hc_pre.post", postF)
            DeepseekV4NumericTrace.tensor("layer.0.ffn_hc_pre.comb", combF)
        }
        finishStage("ffn_hc_pre", [xF, postF, combF])
        let normedF = collapsedF.normalized
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.ffn_norm", normedF)
        }
        finishStage("ffn_norm", [normedF])
        mlp.currentInputIds = inputIds
        let ffnOut = mlp(normedF)
        mlp.currentInputIds = nil
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.moe", ffnOut)
        }
        finishStage("moe", [ffnOut])
        let hF = ffnHC.expand(
            blockOut: ffnOut, residual: residualF, post: postF, comb: combF)
        if layerIdx == 0 {
            DeepseekV4NumericTrace.tensor("layer.0.ffn_hc_post", hF)
        }
        finishStage("ffn_hc_post", [hF])
        return hF
    }
}

// MARK: - Inner model

public class DeepseekV4ModelInner: Module {
    let config: DeepseekV4Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4DecoderLayer]
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map {
            DeepseekV4DecoderLayer(config: config, layerIdx: $0)
        }
        self._hcHead.wrappedValue = DeepseekV4HyperHead(config: config)
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        DeepseekV4NumericTrace.tokens(inputs)
        // embed: (B, L) → (B, L, hiddenSize)
        var h = embedTokens(inputs)
        DeepseekV4NumericTrace.tensor("embedding", h)
        // Tile to mHC copies: (B, L, hiddenSize) → (B, L, hcMult, hiddenSize).
        // Python tiles via broadcast; Swift uses `repeated` along axis -2.
        h = h.expandedDimensions(axis: -2)  // (B, L, 1, H)
        h = repeated(h, count: config.hcMult, axis: -2)  // (B, L, hcMult, H)

        let firstCache = cache?.first
        let hFlat2 = h.reshaped(h.dim(0), h.dim(1), -1)  // for createAttentionMask
        let mask = createAttentionMask(h: hFlat2, cache: firstCache)

        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                mask: mask,
                cache: cache?[i],
                inputIds: inputs)
            DeepseekV4NumericTrace.tensor("layer.\(i)", h)
        }

        // HyperHead reduce: (B, L, hcMult, H) → (B, L, H)
        var out = hcHead.reduce(h)
        DeepseekV4NumericTrace.tensor("hc_head", out)
        out = norm(out)
        DeepseekV4NumericTrace.tensor("norm", out)
        return out
    }

    /// Runs the ordinary verifier forward pass while retaining the mean-mHC
    /// states required by an embedded DSpARK drafter. This is deliberately a
    /// separate entry point so normal autoregressive generation keeps its
    /// existing graph and incurs no capture bookkeeping.
    public func forwardCapturingHiddenStates(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        layerIds: [Int]
    ) -> (hidden: MLXArray, captured: MLXArray) {
        precondition(!layerIds.isEmpty, "DSpARK capture requires at least one layer")
        precondition(
            layerIds.allSatisfy { $0 >= 0 && $0 < layers.count },
            "DSpARK capture layer is outside the base transformer")

        let requested = Set(layerIds)
        var h = embedTokens(inputs)
        h = h.expandedDimensions(axis: -2)
        h = repeated(h, count: config.hcMult, axis: -2)

        let firstCache = cache?.first
        let hFlat2 = h.reshaped(h.dim(0), h.dim(1), -1)
        let mask = createAttentionMask(h: hFlat2, cache: firstCache)
        var capturedByLayer: [Int: MLXArray] = [:]

        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                mask: mask,
                cache: cache?[i],
                inputIds: inputs)
            if requested.contains(i) {
                capturedByLayer[i] = h.mean(axis: -2)
            }
        }

        let ordered = layerIds.map { layerId -> MLXArray in
            guard let value = capturedByLayer[layerId] else {
                preconditionFailure("DSpARK capture missed layer \(layerId)")
            }
            return value
        }
        var out = hcHead.reduce(h)
        out = norm(out)
        return (hidden: out, captured: concatenated(ordered, axis: -1))
    }
}

// MARK: - Outer model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel,
    DeepseekV4SymmetricQ8Model
{
    public var usesDeepseekV4SymmetricQ8: Bool { config.afmSymmetricQ8 }
    public var kvHeads: [Int]
    var config: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    var mtp: [DeepseekV4DSparkStage]
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    private var cachedLMHeadF32: MLXArray?
    private var cachedLMHeadQ8: (weight: MLXArray, scales: MLXArray, biases: MLXArray?)?
    private let cachedLMHeadF32Lock = NSLock()

    private static let cacheLMHeadF32: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV4_CACHE_LM_HEAD"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }()

    private static let quantizeLMHeadQ8 = DeepseekV4RuntimeOptions.enabled(
        "VMLX_DSV4_Q8_LM_HEAD", default: false)

    public init(_ config: DeepseekV4Configuration) {
        self.config = config
        if DeepseekV4PerformanceProfile.enabled {
            fputs("[DSV4StageProfile] enabled\n", stderr)
        }
        if Self.quantizeLMHeadQ8 {
            fputs("[DSV4] output head: runtime affine Q8 (group size 32)\n", stderr)
        }
        // Single latent KV head per layer — report kvHeads as [1]*L so
        // the cache allocator sizes per-layer caches correctly.
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = DeepseekV4ModelInner(config: config)
        self.mtp = config.hasEmbeddedDSpark
            ? (0..<config.dsparkStageCount).map {
                DeepseekV4DSparkStage(config: config, stageIndex: $0)
            }
            : []
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Build per-layer caches.
    ///
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass — pure long-context):
    /// DSV4-Flash IS a hybrid SWA+CSA+HSA architecture by definition.
    /// The previous "long-ctx off" fallback (a plain `RotatingKVCache`
    /// for every layer) was producing measurably degraded output on
    /// any chat that exceeded `sliding_window=128` tokens, because the
    /// `cr>0` layers lost their compressed/indexed global context after
    /// the local window rotated. The toggle is removed; every layer
    /// now allocates the canonical cache for its `compress_ratio`:
    ///   - `cr == 0` (layers 0 and n-1) → `RotatingKVCache(window=128)`
    ///   - `cr > 0`  (every other layer) → `DeepseekV4Cache(window=128, cr=cr)`
    ///
    /// `DSV4_KV_MODE` env override is preserved for diagnostics so the
    /// host can deliberately pick the local KV sizing tradeoff:
    ///   - default (unset / "sliding"): rotating window + DeepseekV4Cache pool
    ///   - "full"  : plain KVCacheSimple on every layer (no compression,
    ///               no pool — for memory-permits long-reasoning runs
    ///               that don't need the hybrid path)
    ///   - "tq"    : KVCacheSimple, BatchEngine swaps to TurboQuantKVCache
    ///               once offset > min-tokens (caller must also set
    ///               `GenerateParameters.kvMode = .turboQuant(...)`)
    ///
    /// Caller-level `GenerateParameters.kvMode = .turboQuant` is
    /// intentionally NOT enough to switch DSV4 into `"tq"` mode. Osaurus
    /// can set global TQ defaults for ordinary KV models; DSV4 must keep
    /// its SWA+CSA+HSA hybrid cache unless the operator explicitly opts
    /// into the diagnostic/simple-cache override via `DSV4_KV_MODE=tq`.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let env = ProcessInfo.processInfo.environment
        let envMode = env["DSV4_KV_MODE"]?.lowercased()
        let mode: String = envMode ?? "sliding"

        return (0..<config.numHiddenLayers).map { layerIdx in
            switch mode {
            case "full", "tq":
                return KVCacheSimple()
            default:
                let cr =
                    config.compressRatios.count > layerIdx
                    ? config.compressRatios[layerIdx]
                    : Self.defaultCompressRatio(
                        layerIdx: layerIdx,
                        numLayers: config.numHiddenLayers)
                if cr > 0 {
                    return DeepseekV4Cache(
                        slidingWindow: config.slidingWindow,
                        compressRatio: cr)
                }
                return RotatingKVCache(
                    maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    /// Mirror of the per-layer compress_ratio default in
    /// `DeepseekV4Attention.init` for bundles whose
    /// `config.compressRatios` array isn't populated. Layers 0 and n-1
    /// are pure SWA (`cr=0`); middle layers alternate `4` (HSA+CSA)
    /// and `128` (CSA only).
    static func defaultCompressRatio(layerIdx: Int, numLayers: Int) -> Int {
        if layerIdx == 0 || layerIdx == numLayers - 1 { return 0 }
        let i = layerIdx - 1
        return (i % 2 == 1) ? 4 : 128
    }

    private func projectLogits(_ h: MLXArray) -> MLXArray {
        let logits: MLXArray
        if Self.quantizeLMHeadQ8, !(lmHead is QuantizedLinear) {
            cachedLMHeadF32Lock.lock()
            if cachedLMHeadQ8 == nil {
                let quantized = MLX.quantized(
                    lmHead.weight, groupSize: 32, bits: 8, mode: .affine)
                var arrays = [quantized.wq, quantized.scales]
                if let biases = quantized.biases { arrays.append(biases) }
                MLX.eval(arrays)
                cachedLMHeadQ8 = (quantized.wq, quantized.scales, quantized.biases)
            }
            let cached = cachedLMHeadQ8!
            cachedLMHeadF32Lock.unlock()
            var output = MLX.quantizedMM(
                h, cached.weight,
                scales: cached.scales, biases: cached.biases,
                transpose: true, groupSize: 32, bits: 8, mode: .affine)
            if let bias = lmHead.bias {
                output = output + bias.asType(output.dtype)
            }
            logits = output.asType(.float32)
        } else if Self.cacheLMHeadF32, !(lmHead is QuantizedLinear) {
            cachedLMHeadF32Lock.lock()
            if cachedLMHeadF32 == nil {
                let weight = lmHead.weight.asType(.float32)
                MLX.eval(weight)
                cachedLMHeadF32 = weight
            }
            let weight = cachedLMHeadF32!
            cachedLMHeadF32Lock.unlock()
            var output = h.asType(.float32).matmul(weight.transposed())
            if let bias = lmHead.bias {
                output = output + bias.asType(.float32)
            }
            logits = output
        } else {
            logits = DeepseekV4Math.lmHeadFp32(h, lmHead: lmHead)
        }
        DeepseekV4NumericTrace.tensor("logits", logits)
        return logits
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        projectLogits(model(inputs, cache: cache))
    }

    /// Authoritative target forward used by DSpARK verification. The logits
    /// follow the ordinary generation path exactly; capture only retains the
    /// configured hidden-state taps needed to advance the drafter context.
    public func forwardDSparkVerifier(
        _ inputs: MLXArray,
        cache: [KVCache]
    ) -> (logits: MLXArray, captured: MLXArray)? {
        guard !mtp.isEmpty else { return nil }
        let result = model.forwardCapturingHiddenStates(
            inputs, cache: cache, layerIds: config.dsparkTargetLayerIds)
        return (projectLogits(result.hidden), result.captured)
    }

    public func newDSparkCache() -> [KVCache] {
        mtp.map { _ in RotatingKVCache(maxSize: config.slidingWindow, keep: 0) }
    }

    public var supportsEmbeddedDSpark: Bool { !mtp.isEmpty }

    @discardableResult
    public func prefillDSpark(
        anchorTokenIds: MLXArray,
        capturedHidden: MLXArray,
        cache: [KVCache]
    ) -> Bool {
        guard !mtp.isEmpty, cache.count == mtp.count else { return false }
        let prepared = mtp[0].prepareDraft(
            mainHidden: capturedHidden, anchorTokenIds: anchorTokenIds,
            embedding: model.embedTokens)
        var hidden = prepared.hidden
        for (index, stage) in mtp.enumerated() {
            hidden = stage.forward(
                hidden, main: prepared.main,
                anchorTokenIds: anchorTokenIds, cache: cache[index])
        }
        return true
    }

    public func proposeDSpark(
        anchorTokenIds: MLXArray,
        capturedHidden: MLXArray,
        cache: [KVCache]
    ) -> DeepseekV4DSparkProposal? {
        guard !mtp.isEmpty, cache.count == mtp.count else { return nil }
        let prepared = mtp[0].prepareDraft(
            mainHidden: capturedHidden, anchorTokenIds: anchorTokenIds,
            embedding: model.embedTokens)
        var hidden = prepared.hidden
        for (index, stage) in mtp.enumerated() {
            hidden = stage.forward(
                hidden, main: prepared.main,
                anchorTokenIds: anchorTokenIds, cache: cache[index])
        }
        return mtp[mtp.count - 1].makeProposal(
            hidden, anchorTokenIds: anchorTokenIds, lmHead: lmHead)
    }

    /// Weight sanitize — remap DSV4 bundle key names to match module
    /// attribute paths, stack per-expert weights, and retain an embedded
    /// DSpARK drafter when the checkpoint advertises the complete contract.
    /// compressor/indexer keys.
    ///
    /// Remap rules (from §G of RUNTIME-ARCHITECTURE):
    ///   model.embed.weight            → model.embed_tokens.weight
    ///   layers.{L}.attn.*             → model.layers.{L}.self_attn.*
    ///   layers.{L}.ffn.*              → model.layers.{L}.mlp.*
    ///   layers.{L}.attn_norm.weight   → model.layers.{L}.input_layernorm.weight
    ///   layers.{L}.ffn_norm.weight    → model.layers.{L}.post_attention_layernorm.weight
    ///   layers.{L}.hc_attn_*          → model.layers.{L}.attn_hc.{fn,scale,base}
    ///   layers.{L}.hc_ffn_*           → model.layers.{L}.ffn_hc.{fn,scale,base}
    ///   hc_head_*                     → model.hc_head.{hc_head_fn,hc_head_base,hc_head_scale}
    ///   norm.weight                   → model.norm.weight
    ///   head.weight                   → lm_head.weight
    ///   ffn.experts.{E}.{w1|w2|w3}.*  → mlp.switch_mlp.{gate|down|up}_proj.* (stacked)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if config.afmNativeCheckpoint {
            return weights
        }

        var out: [String: MLXArray] = [:]
        let officialQuantizedBases = Set(weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scale") else { return nil }
            let base = String(key.dropLast(".scale".count))
            return weights["\(base).weight"] == nil ? nil : base
        })
        // First pass: direct rename. `mtp.*` is a complete inference drafter
        // in 0731 checkpoints, not a disposable training head.
        // Compressor + Indexer weights are KEPT — they're wired into
        // DeepseekV4Attention for long-context (L > sliding_window)
        // attention. Layers with compress_ratio == 0 carry no such
        // weights; layers with >0 carry `self_attn.compressor.*` and
        // (for ratio=4) `self_attn.indexer.*`.
        // Mirrors `Model.sanitize` in
        // jang-tools/jang_tools/dsv4/mlx_model.py:1124. Per-prefix
        // structural matching — avoids over-broad string replace bugs
        // (e.g. ".w1." colliding outside MLP contexts).
        let projForW = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]
        for (sourceKey, sourceValue) in weights {
            var rawKey = sourceKey
            var value = sourceValue

            // The official 0731 checkpoint is already packed in the same
            // MXFP4/MXFP8 bit layout used by MLX. Reinterpret the bytes rather
            // than dequantizing and requantizing them, and normalize DeepSeek's
            // singular scale key to MLX's plural convention. `view` adjusts
            // the final dimension as bytes are grouped into UInt32 words.
            if sourceKey.hasSuffix(".scale") {
                let base = String(sourceKey.dropLast(".scale".count))
                if officialQuantizedBases.contains(base) {
                    rawKey = "\(base).scales"
                    value = normalizeOfficialBlockScale(
                        sourceValue,
                        weight: weights["\(base).weight"]!)
                }
            } else if sourceKey.hasSuffix(".weight") {
                let base = String(sourceKey.dropLast(".weight".count))
                if officialQuantizedBases.contains(base) {
                    value = sourceValue.view(dtype: .uint32)
                }
            }

            if rawKey.hasPrefix("mtp.") {
                let afterMTP = rawKey.dropFirst("mtp.".count)
                guard let stageDot = afterMTP.firstIndex(of: "."),
                    Int(afterMTP[..<stageDot]) != nil
                else { continue }
                let stage = String(afterMTP[..<stageDot])
                let rest = String(afterMTP[afterMTP.index(after: stageDot)...])
                let pfx = "mtp.\(stage)"
                if rest.hasPrefix("main_proj.") || rest.hasPrefix("main_norm.")
                    || rest.hasPrefix("norm.") || rest.hasPrefix("markov_head.")
                    || rest.hasPrefix("confidence_head.") || rest.hasPrefix("hc_head_")
                    || rest == "attn_norm.weight" || rest == "ffn_norm.weight"
                    || rest.hasPrefix("attn.")
                {
                    out["\(pfx).\(rest)"] = value
                    continue
                }
                if rest.hasPrefix("hc_attn_") {
                    out["\(pfx).attn_hc.\(rest.dropFirst("hc_attn_".count))"] = value
                    continue
                }
                if rest.hasPrefix("hc_ffn_") {
                    out["\(pfx).ffn_hc.\(rest.dropFirst("hc_ffn_".count))"] = value
                    continue
                }
                if rest.hasPrefix("ffn.") {
                    let inner = String(rest.dropFirst("ffn.".count))
                    if inner.hasPrefix("gate.") {
                        out["\(pfx).ffn.\(inner)"] = value
                        continue
                    }
                    if inner.hasPrefix("shared_experts.") {
                        let field = String(inner.dropFirst("shared_experts.".count))
                        if let dot = field.firstIndex(of: "."),
                            let projection = projForW[String(field[..<dot])]
                        {
                            let suffix = field[field.index(after: dot)...]
                            out["\(pfx).ffn.shared_experts.\(projection).\(suffix)"] = value
                            continue
                        }
                    }
                    if inner.hasPrefix("experts.") {
                        let afterExperts = String(inner.dropFirst("experts.".count))
                        guard let expertDot = afterExperts.firstIndex(of: ".") else { continue }
                        let expert = String(afterExperts[..<expertDot])
                        let tail = String(afterExperts[afterExperts.index(after: expertDot)...])
                        if let projectionDot = tail.firstIndex(of: "."),
                            let projection = projForW[String(tail[..<projectionDot])]
                        {
                            let suffix = tail[tail.index(after: projectionDot)...]
                            out["\(pfx).ffn.experts.\(expert).\(projection).\(suffix)"] = value
                            continue
                        }
                    }
                    out["\(pfx).ffn.\(inner)"] = value
                    continue
                }
                out["\(pfx).\(rest)"] = value
                continue
            }

            // Top-level (no `layers.N.` prefix).
            if rawKey == "embed.weight" || rawKey == "embed.scales"
                || rawKey == "embed.biases"
            {
                let suffix = String(rawKey.dropFirst("embed.".count))
                out["model.embed_tokens.\(suffix)"] = value
                continue
            }
            if rawKey.hasPrefix("head.") {
                // head.{weight,scales,biases} → lm_head.*
                let suffix = String(rawKey.dropFirst("head.".count))
                out["lm_head.\(suffix)"] = value
                continue
            }
            if rawKey == "norm.weight" {
                out["model.norm.weight"] = value
                continue
            }
            if rawKey == "hc_head_fn" || rawKey == "hc_head_base"
                || rawKey == "hc_head_scale"
            {
                // `@ParameterInfo(key: "hc_head_*")` lives at
                // `model.hc_head.hc_head_*`.
                out["model.hc_head.\(rawKey)"] = value
                continue
            }

            // layers.N.{...} branch
            guard rawKey.hasPrefix("layers.") else {
                out["model.\(rawKey)"] = value
                continue
            }
            let afterLayers = rawKey.dropFirst("layers.".count)
            guard let dotIdx = afterLayers.firstIndex(of: ".") else { continue }
            let layerStr = String(afterLayers[..<dotIdx])
            guard Int(layerStr) != nil else { continue }
            let rest = String(afterLayers[afterLayers.index(after: dotIdx)...])
            let pfx = "model.layers.\(layerStr)"

            // Norms
            if rest == "attn_norm.weight" {
                out["\(pfx).input_layernorm.weight"] = value
                continue
            }
            if rest == "ffn_norm.weight" {
                out["\(pfx).post_attention_layernorm.weight"] = value
                continue
            }

            // mHC per-layer (hc_attn_*, hc_ffn_*).
            if rest.hasPrefix("hc_attn_") {
                let field = String(rest.dropFirst("hc_attn_".count))
                out["\(pfx).attn_hc.\(field)"] = value
                continue
            }
            if rest.hasPrefix("hc_ffn_") {
                let field = String(rest.dropFirst("hc_ffn_".count))
                out["\(pfx).ffn_hc.\(field)"] = value
                continue
            }

            // Attention subtree (q_norm / kv_norm / wq_a / wq_b / wkv /
            // wo_a / wo_b / attn_sink / compressor.* / indexer.*).
            if rest.hasPrefix("attn.") {
                let inner = String(rest.dropFirst("attn.".count))
                out["\(pfx).self_attn.\(inner)"] = value
                continue
            }

            // FFN subtree.
            if rest.hasPrefix("ffn.") {
                let inner = String(rest.dropFirst("ffn.".count))
                if inner.hasPrefix("gate.") {
                    let f = String(inner.dropFirst("gate.".count))
                    out["\(pfx).mlp.gate.\(f)"] = value
                    continue
                }
                if inner.hasPrefix("shared_experts.") {
                    let f = String(inner.dropFirst("shared_experts.".count))
                    if let firstDot = f.firstIndex(of: "."),
                        let proj = projForW[String(f[..<firstDot])]
                    {
                        let suffix = String(f[f.index(after: firstDot)...])
                        out["\(pfx).mlp.shared_experts.\(proj).\(suffix)"] = value
                        continue
                    }
                    out["\(pfx).mlp.shared_experts.\(f)"] = value
                    continue
                }
                if inner.hasPrefix("experts.") {
                    let after = String(inner.dropFirst("experts.".count))
                    guard let eDot = after.firstIndex(of: ".") else { continue }
                    let eStr = String(after[..<eDot])
                    let tail = String(after[after.index(after: eDot)...])
                    if let firstDot = tail.firstIndex(of: "."),
                        let proj = projForW[String(tail[..<firstDot])]
                    {
                        let suffix = String(tail[tail.index(after: firstDot)...])
                        out["\(pfx).mlp.experts.\(eStr).\(proj).\(suffix)"] = value
                        continue
                    }
                    out["\(pfx).mlp.experts.\(eStr).\(tail)"] = value
                    continue
                }
                out["\(pfx).mlp.\(inner)"] = value
                continue
            }

            out["\(pfx).\(rest)"] = value
        }

        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // DSV4-Flash JANGTQ bundles ship a pre-stacked
        // `jangtq_stacked.safetensors` overlay where the routed-expert
        // weights live at
        // `layers.{L}.mlp.switch_mlp.{gate,down,up}_proj.{packed,norms}`
        // — note the missing `tq_` prefix. Older Swift JANGTQ bundles
        // and the in-tree `TurboQuantSwitchLinear` use `tq_packed` /
        // `tq_norms`. Rewrite the un-prefixed names so the
        // `@ParameterInfo` keys match. Layout-preserving rename only.
        for layerIdx in 0..<config.numHiddenLayers {
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for (src, dst) in [("packed", "tq_packed"), ("norms", "tq_norms")] {
                    let from = "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(src)"
                    let to = "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(dst)"
                    if let v = out.removeValue(forKey: from), out[to] == nil {
                        out[to] = v
                    }
                }
            }
        }

        // Second pass: stack per-expert weights into switch_mlp.{gate,
        // up,down}_proj.*. Two formats supported:
        //
        // Affine (JANG_2L / JANG4): suffixes weight / scales / biases.
        //   Source per expert: (out, in) [+ (out, in/group)] [+ (out, in/group)]
        //   Stacked shape: (n_experts, ...).
        //
        // JANGTQ (JANGTQ2 / JANGTQ4 routed experts): suffixes
        // tq_packed / tq_norms. tq_bits is a per-tensor int constant
        // — we drop it (TurboQuantSwitchLinear configures bits at
        // construction time from the model_factory).
        //   Source per expert: tq_packed (out, packed_cols), tq_norms (out,)
        //   Stacked shape: (n_experts, out, packed_cols) / (n_experts, out)
        // The first pass already rewrote `.w1.` → `.gate_proj.` (etc.)
        // globally, so per-expert keys live at
        // `model.layers.L.mlp.experts.E.gate_proj.{suffix}`. Stack into
        // `model.layers.L.mlp.switch_mlp.{gate,down,up}_proj.{suffix}`.
        let suffixes = ["weight", "scales", "biases", "tq_packed", "tq_norms"]
        let streamJANGTQExperts = false
        for layerIdx in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx).mlp.experts"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in suffixes {
                    let first = "\(prefix).0.\(projName).\(suffix)"
                    guard out[first] != nil else { continue }
                    if streamJANGTQExperts && (suffix == "tq_packed" || suffix == "tq_norms") {
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(
                                forKey: "\(prefix).\(e).\(projName).\(suffix)")
                        }
                        continue
                    }
                    var tensors: [MLXArray] = []
                    for e in 0..<config.nRoutedExperts {
                        let key = "\(prefix).\(e).\(projName).\(suffix)"
                        guard let t = out[key] else {
                            tensors = []
                            break
                        }
                        tensors.append(t)
                    }
                    if tensors.count == config.nRoutedExperts {
                        let stackedKey =
                            "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(suffix)"
                        if out[stackedKey] == nil {
                            out[stackedKey] = stacked(tensors)
                        }
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(
                                forKey: "\(prefix).\(e).\(projName).\(suffix)")
                        }
                    }
                }
                // Drop per-expert + prestacked-switch_mlp tq_bits scalars
                // — TurboQuantSwitchLinear gets the bit width from the
                // JANGTQ config (`mxtq_bits.routed_expert`), not weights.
                // Legacy bundles ship `mlp.experts.{e}.{proj}.tq_bits`;
                // prestacked bundles (DSV4-Flash JANGTQ_K, etc.) ship a
                // single `mlp.switch_mlp.{proj}.tq_bits` per layer.
                // Drop both regardless of which layout.
                for e in 0..<config.nRoutedExperts {
                    out.removeValue(
                        forKey: "\(prefix).\(e).\(projName).tq_bits")
                }
                out.removeValue(
                    forKey: "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).tq_bits")
            }
        }
        for stageIdx in 0..<config.dsparkStageCount {
            let prefix = "mtp.\(stageIdx).ffn.experts"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard out["\(prefix).0.\(projection).\(suffix)"] != nil else {
                        continue
                    }
                    let tensors = (0..<config.nRoutedExperts).compactMap {
                        out["\(prefix).\($0).\(projection).\(suffix)"]
                    }
                    guard tensors.count == config.nRoutedExperts else { continue }
                    out["mtp.\(stageIdx).ffn.switch_mlp.\(projection).\(suffix)"] =
                        stacked(tensors)
                    for expert in 0..<config.nRoutedExperts {
                        out.removeValue(
                            forKey: "\(prefix).\(expert).\(projection).\(suffix)")
                    }
                }
            }
        }
        return out
    }

    /// Convert the official FP8 block-scale grid to MLX's per-row, per-32-value
    /// scale layout. Official FP4 expert tensors already use MLX's scale shape.
    public func normalizeOfficialBlockScale(
        _ scale: MLXArray,
        weight: MLXArray
    ) -> MLXArray {
        var result = scale.view(dtype: .uint8)
        guard weight.dtype == .uint8, weight.ndim >= 2, result.ndim == 2 else {
            return result
        }

        let expectedRows = weight.dim(-2)
        let expectedColumns = weight.dim(-1) / 32
        guard expectedRows % result.dim(0) == 0,
            expectedColumns % result.dim(1) == 0
        else { return result }

        let rowRepeat = expectedRows / result.dim(0)
        let columnRepeat = expectedColumns / result.dim(1)
        if rowRepeat > 1 {
            result = repeated(result, count: rowRepeat, axis: 0)
        }
        if columnRepeat > 1 {
            result = repeated(result, count: columnRepeat, axis: 1)
        }
        return result
    }

    public var loraLayers: [Module] {
        model.layers
    }
}


// MARK: - Embedded DSpARK speculative generator

/// Exact greedy speculative decoding for DeepSeek V4 checkpoints carrying an
/// embedded DSpARK drafter. The target model remains authoritative: only the
/// longest draft prefix matching target argmax is committed, followed by the
/// target's replacement/bonus token.
public final class DeepseekV4DSparkGenerator {
    private let model: DeepseekV4Model
    public let draftLimit: Int

    public init(model: DeepseekV4Model, draftLimit: Int? = nil) {
        self.model = model
        self.draftLimit = max(
            1, min(draftLimit ?? model.config.dsparkBlockSize,
                   model.config.dsparkBlockSize))
    }

    private func tokens(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
    }

    private func token(_ id: Int) -> MLXArray {
        MLXArray([Int32(id)]).reshaped([1, 1])
    }

    private func argmaxLast(_ logits: MLXArray) -> Int {
        argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
    }

    public func generate(
        promptIds: [Int],
        maxTokens: Int,
        eosIds: Set<Int> = [],
        onToken: ((Int) -> Bool)? = nil
    ) -> [Int] {
        guard !promptIds.isEmpty, maxTokens > 0, !model.mtp.isEmpty else { return [] }

        let verifierCache = model.newCache(parameters: nil)
        let drafterCache = model.newDSparkCache()
        guard let prefill = model.forwardDSparkVerifier(
            tokens(promptIds), cache: verifierCache)
        else { return [] }

        var pending = argmaxLast(prefill.logits)
        guard model.prefillDSpark(
            anchorTokenIds: token(pending),
            capturedHidden: prefill.captured,
            cache: drafterCache)
        else { return [] }

        var output: [Int] = []
        var rounds = 0
        var draftedTotal = 0
        var acceptedTotal = 0

        func emit(_ value: Int) -> Bool {
            output.append(value)
            if eosIds.contains(value) { return false }
            if let onToken, !onToken(value) { return false }
            return output.count < maxTokens
        }

        if !emit(pending) { return output }

        // The embedded checkpoint primes only its target-hidden KV during
        // prefill. Consume the first target token autoregressively to obtain
        // the real hidden row that starts stage execution (the reference
        // implementation likewise does not execute DSpARK blocks at offset 0).
        guard let warmup = model.forwardDSparkVerifier(
            token(pending), cache: verifierCache)
        else { return output }
        pending = argmaxLast(warmup.logits)
        var contextForDrafter = warmup.captured
        if !emit(pending) { return output }

        while output.count < maxTokens {
            guard let proposal = model.proposeDSpark(
                anchorTokenIds: token(pending),
                capturedHidden: contextForDrafter,
                cache: drafterCache)
            else { break }

            let available = proposal.tokenIds.dim(1) - 1
            let count = min(draftLimit, available, maxTokens - output.count)
            if count <= 0 { break }
            let draftArray = proposal.tokenIds[0..., 1..<(count + 1)].asType(.int32)
            let verifyIds = concatenated([token(pending), draftArray], axis: 1)
            let snapshots: [DeepseekV4Cache.SpeculativeSnapshot?] = verifierCache.map {
                ($0 as? DeepseekV4Cache)?.captureSpeculativeSnapshot()
            }
            guard let verified = model.forwardDSparkVerifier(
                verifyIds, cache: verifierCache)
            else { break }

            let targetArray = argMax(
                verified.logits[0, 0..., 0...], axis: -1).asType(.int32)
            MLX.eval(draftArray, targetArray)
            let drafts = draftArray.reshaped([count]).asArray(Int32.self).map(Int.init)
            let targets = targetArray.reshaped([count + 1]).asArray(Int32.self).map(Int.init)

            var accepted = 0
            while accepted < count && drafts[accepted] == targets[accepted] {
                accepted += 1
            }
            let rejected = count - accepted
            var committedCapture = verified.captured[
                0..., 0..<(accepted + 1), 0...]
            if rejected > 0 {
                // The compressed pool and local KV must be restored to the
                // same logical point. Restoring only the pre-verify pool while
                // retaining accepted local rows creates a hybrid history and
                // makes the next target token diverge from greedy decoding.
                // Rewind the complete verifier block, then replay only the
                // committed prefix. Rejects pay one short target forward;
                // fully accepted blocks remain single-forward.
                for (index, cache) in verifierCache.enumerated() {
                    if let deepseekCache = cache as? DeepseekV4Cache,
                       let snapshot = snapshots[index]
                    {
                        deepseekCache.rollbackSpeculative(
                            rejected: count + 1, to: snapshot)
                    } else {
                        _ = cache.trim(count + 1)
                    }
                }
                let replayIds = [pending] + Array(drafts.prefix(accepted))
                guard let replay = model.forwardDSparkVerifier(
                    tokens(replayIds), cache: verifierCache)
                else { break }
                committedCapture = replay.captured
            }

            rounds += 1
            draftedTotal += count
            acceptedTotal += accepted
            contextForDrafter = committedCapture

            var committed = accepted > 0 ? Array(drafts.prefix(accepted)) : []
            committed.append(targets[accepted])
            pending = committed.last!
            var keepGoing = true
            for value in committed.prefix(maxTokens - output.count) {
                if !emit(value) {
                    keepGoing = false
                    break
                }
            }
            if !keepGoing { break }
        }

        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" {
            let acceptance = draftedTotal > 0
                ? Double(acceptedTotal) / Double(draftedTotal) : 0
            let perRound = rounds > 0 ? Double(output.count) / Double(rounds) : 0
            print(String(format:
                "[DSpARK] rounds=%d accept=%d/%d (%.1f%%) tok/round=%.2f",
                rounds, acceptedTotal, draftedTotal, acceptance * 100, perRound))
        }
        return output
    }
}
