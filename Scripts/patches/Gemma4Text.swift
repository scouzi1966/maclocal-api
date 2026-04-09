//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py
//
//  Gemma 4 text model supporting all variants:
//    - Dense 31B: standard attention + MLP
//    - MoE 26B-A4B: mixed attention + SwitchGLU MoE (128 experts, top-8)
//    - E4B/E2B: per-layer input gating + KV sharing across layers
//
//  Key features: mixed sliding/full attention, K-eq-V, per-layer RoPE config,
//  optional MoE, logit softcapping, scaled embeddings, per-layer input gating,
//  KV cache sharing for small models.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Nested rope parameters per layer type
struct Gemma4RopeParams: Codable, Sendable {
    var ropeTheta: Float?
    var partialRotaryFactor: Float?
    var ropeType: String?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeType = "rope_type"
    }
}

/// Text-specific configuration for Gemma 4 models.
public struct Gemma4TextConfig: Codable, Sendable {
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var headDim: Int
    var globalHeadDim: Int
    var numKeyValueHeads: Int
    var numGlobalKeyValueHeads: Int?
    var rmsNormEps: Float
    var vocabSize: Int
    var slidingWindow: Int
    var layerTypes: [String]
    var ropeParameters: [String: Gemma4RopeParams]
    var enableMoeBlock: Bool
    var numExperts: Int?
    var topKExperts: Int?
    var moeIntermediateSize: Int?
    var finalLogitSoftcapping: Float?
    var attentionKEqV: Bool
    var tieWordEmbeddings: Bool
    var ropeTraditional: Bool
    // Per-layer input gating (E2B/E4B)
    var hiddenSizePerLayerInput: Int
    var numKvSharedLayers: Int
    var vocabSizePerLayerInput: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attentionKEqV = "attention_k_eq_v"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTraditional = "rope_traditional"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case numKvSharedLayers = "num_kv_shared_layers"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        numGlobalKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        ropeParameters = try c.decode([String: Gemma4RopeParams].self, forKey: .ropeParameters)
        enableMoeBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        finalLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        hiddenSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        numKvSharedLayers = try c.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        vocabSizePerLayerInput = try c.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? vocabSize
    }

    /// First layer index that shares KV from an earlier layer (0 means no sharing)
    var firstKvSharedLayerIdx: Int {
        numKvSharedLayers > 0 ? numHiddenLayers - numKvSharedLayers : numHiddenLayers
    }
}

/// Top-level configuration that reads text_config from nested config.json.
/// Supports both "gemma4" (nested text_config) and "gemma4_text" (flat) model types.
public struct Gemma4Configuration: Codable, Sendable {
    var textConfig: Gemma4TextConfig

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let nested = try? container.decode(Gemma4TextConfig.self, forKey: .textConfig) {
            self.textConfig = nested
        } else {
            // Flat config (model_type = "gemma4_text"): decode from top level
            self.textConfig = try Gemma4TextConfig(from: decoder)
        }
    }
}

// MARK: - RMSNorm Variants

/// Gemma-style RMSNorm with +1 weight offset (stored weights are shifted by -1).
class Gemma4RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

/// RMSNorm without learnable scale (scale_shift=0, no weight parameter).
class Gemma4RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = (x * x).mean(axis: -1, keepDims: true)
        return x * rsqrt(variance + eps)
    }
}

/// RMSNorm with zero-shift (weight used directly, no +1 offset). For per-layer projection.
class Gemma4RMSNormZeroShift: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Scaled Linear

/// Linear layer with output scaling (for per-layer model projection).
class Gemma4ScaledLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inputDims: Int, outputDims: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outputDims, inputDims])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return (x.matmul(weight.T)) * scalar
    }
}

// MARK: - MLP

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Router

/// Expert router: RMSNorm(no-scale) → scale → project → top-k → renormalize.
class Gemma4Router: Module {
    let numExperts: Int
    let topK: Int
    let rootSize: Float

    let norm: Gemma4RMSNormNoScale
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(hiddenSize: Int, numExperts: Int, topK: Int, eps: Float) {
        self.numExperts = numExperts
        self.topK = topK
        self.rootSize = pow(Float(hiddenSize), -0.5)

        self.norm = Gemma4RMSNormNoScale(eps: eps)
        _proj.wrappedValue = Linear(hiddenSize, numExperts, bias: false)
        _scale.wrappedValue = MLXArray.ones([hiddenSize])
        _perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = norm(x)
        h = h * rootSize
        h = h * scale

        let expertScores = proj(h)
        let routerProbs = softmax(expertScores, axis: -1)

        let topKIndices = MLX.argPartition(-expertScores, kth: topK - 1, axis: -1)[
            .ellipsis, ..<topK]

        var topKWeights = MLX.takeAlong(routerProbs, topKIndices, axis: -1)
        topKWeights = topKWeights / MLX.sum(topKWeights, axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (topKIndices, topKWeights)
    }
}

// MARK: - Experts (SwitchGLU wrapper)

/// Sparse MoE using SwitchGLU with GeGLU activation.
class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(hiddenSize: Int, moeIntermediateSize: Int, numExperts: Int) {
        _switchGLU.wrappedValue = SwitchGLU(
            inputDims: hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: { geluApproximate($0) },
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let shape = x.shape  // [B, S, H]
        let B = shape[0]
        let S = shape[1]
        let H = shape[2]
        let K = topKIndices.dim(-1)

        let xFlat = x.reshaped(B * S, H)
        let indicesFlat = topKIndices.reshaped(B * S, K)

        let expertOut = switchGLU(xFlat, indicesFlat)

        let weights = topKWeights.reshaped(B * S, K)[.ellipsis, .newAxis]
        return (expertOut * weights).sum(axis: -2).reshaped(B, S, H)
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let layerIdx: Int
    let isSliding: Bool
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let useKEqV: Bool

    // KV sharing (E2B/E4B)
    let isKvSharedLayer: Bool
    let kvSharedLayerIndex: Int?
    let storeFullLengthKV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let vNorm: Gemma4RMSNormNoScale

    let rope: RoPE

    /// Stored KV for sharing with later layers (set during forward pass)
    var lastKV: (MLXArray, MLXArray)?

    init(_ config: Gemma4TextConfig, layerIdx: Int) {
        self.layerIdx = layerIdx
        let layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // Full attention uses global_head_dim; sliding uses head_dim
        self.headDim =
            (!isSliding && config.globalHeadDim > 0) ? config.globalHeadDim : config.headDim

        self.nHeads = config.numAttentionHeads

        // K-eq-V only for full attention layers
        self.useKEqV = config.attentionKEqV && !isSliding
        if useKEqV, let globalKV = config.numGlobalKeyValueHeads {
            self.nKVHeads = globalKV
        } else {
            self.nKVHeads = config.numKeyValueHeads
        }

        // KV sharing (E2B/E4B): layers >= firstKvSharedLayerIdx reuse KV
        let firstShared = config.firstKvSharedLayerIdx
        self.isKvSharedLayer = layerIdx >= firstShared && firstShared < config.numHiddenLayers
        if isKvSharedLayer {
            // Find the last layer before the shared region with the same type
            let prevLayers = Array(config.layerTypes[..<firstShared])
            let targetType = config.layerTypes[layerIdx]
            self.kvSharedLayerIndex = prevLayers.lastIndex(of: targetType)
        } else {
            self.kvSharedLayerIndex = nil
        }

        // Determine if this layer should store its KV for later sharing
        if !isKvSharedLayer {
            let prevLayers = Array(config.layerTypes[..<firstShared])
            let targetType = config.layerTypes[layerIdx]
            self.storeFullLengthKV = layerIdx == prevLayers.lastIndex(of: targetType)
        } else {
            self.storeFullLengthKV = false
        }

        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        if !useKEqV {
            _vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        }
        _oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.vNorm = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        // RoPE configuration per layer type
        let layerKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters[layerKey]
        let ropeTheta = ropeParams?.ropeTheta ?? 10000.0
        let partialRotaryFactor = ropeParams?.partialRotaryFactor ?? 1.0
        let ropeDims = Int(Float(headDim) * partialRotaryFactor)

        self.rope = RoPE(
            dimensions: ropeDims,
            traditional: config.ropeTraditional,
            base: ropeTheta
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        sharedKV: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let shape = x.shape  // [B, L, _]
        let B = shape[0]
        let L = shape[1]

        // Read offset BEFORE cache update — keys and queries must use the same offset.
        // Batched caches provide per-sequence offsets for correct RoPE positions.
        // Only use array offset when sequences have different lengths. When all
        // per-seq offsets are equal, use the scalar path — this ensures bit-identical
        // RoPE computation between B=1 and B>=2. The `allOffsetsEqual` flag is cached
        // on the batch cache to avoid expensive eval() on every layer.
        let scalarOffset = cache?.offset ?? 0
        let batchedOffset: MLXArray? = {
            guard let arr = cache?.offsetArray, arr.dim(0) == B else { return nil }
            if (cache as? BaseKVCache)?.allOffsetsEqual == true { return nil }
            return arr
        }()

        var queries = qProj(x).reshaped(B, L, nHeads, headDim)
        queries = qNorm(queries)

        var keys: MLXArray
        var values: MLXArray

        if isKvSharedLayer, let (sharedK, sharedV) = sharedKV {
            // Reuse KV from earlier layer
            keys = sharedK
            values = sharedV
        } else {
            keys = kProj(x).reshaped(B, L, nKVHeads, headDim)

            // K-eq-V: values come from raw k_proj output (before k_norm)
            if useKEqV {
                values = keys
            } else {
                values = vProj!(x).reshaped(B, L, nKVHeads, headDim)
            }

            keys = kNorm(keys)
            values = vNorm(values)
            values = values.transposed(0, 2, 1, 3)

            keys = keys.transposed(0, 2, 1, 3)
            if let batchedOffset {
                keys = rope(keys, offset: batchedOffset)
            } else {
                keys = rope(keys, offset: scalarOffset)
            }

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        if storeFullLengthKV {
            lastKV = (keys, values)
        }

        queries = queries.transposed(0, 2, 1, 3)
        if let batchedOffset {
            queries = rope(queries, offset: batchedOffset)
        } else {
            queries = rope(queries, offset: scalarOffset)
        }

        let output: MLXArray

        // For decode (L=1) with no mask needed, use gatherMM-based attention
        // to avoid SDPA's mask-construction overhead entirely. gatherMM fuses
        // the GQA head mapping into the gather indices, doing Q@K^T and
        // attn@V each in ONE Metal dispatch vs the mask path's ~5 ops.
        if L == 1 && mask.mask == nil && B > 1 && nKVHeads != nHeads {
            // Build GQA gather indices: map each (batch, q_head) to the
            // corresponding (batch, kv_head) in the flat batch×heads space.
            let headsPerGroup = nHeads / nKVHeads
            // rhsIndices: [B, nHeads] where [b,h] = b * nKVHeads + h / headsPerGroup
            let batchIdx = MLXArray(0 ..< Int32(B)).reshaped([B, 1]) * Int32(nKVHeads)
            let headMap = MLXArray((0 ..< nHeads).map { Int32($0 / headsPerGroup) }).reshaped([1, nHeads])
            let rhsIdx = (batchIdx + headMap).reshaped([B * nHeads])

            // Q: [B, nHeads, 1, headDim] → scores = Q @ K^T
            let kT = keys.transposed(0, 1, 3, 2)  // [B, nKVHeads, headDim, T_kv]
            let scores = gatherMM(
                queries.reshaped([B * nHeads, 1, headDim]),
                kT.reshaped([B * nKVHeads, headDim, keys.dim(2)]),
                rhsIndices: rhsIdx
            )
            // scores: [B*nHeads, 1, T_kv] → softmax
            let attnWeights = softMax(scores * (1.0 / sqrt(Float(headDim))), axis: -1)

            // attn @ V: [B*nHeads, 1, T_kv] @ [B*nKVHeads, T_kv, headDim]
            let attnOut = gatherMM(
                attnWeights,
                values.reshaped([B * nKVHeads, values.dim(2), headDim]),
                rhsIndices: rhsIdx
            )
            // [B*nHeads, 1, headDim] → [B, nHeads, 1, headDim]
            output = attnOut.reshaped([B, nHeads, 1, headDim])
        } else {
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: 1.0, mask: mask)
        }

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Decoder Layer

class Gemma4DecoderLayer: Module {
    let layerType: String
    let enableMoe: Bool
    let hasPerLayerInput: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention

    // Standard MLP (always present for both dense and MoE)
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP

    // Norms shared by both paths
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // MoE-specific components (26B model)
    @ModuleInfo(key: "router") var router: Gemma4Router?
    @ModuleInfo(key: "experts") var experts: Gemma4Experts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?

    // Per-layer input gating (E2B/E4B)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scaling
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfig, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoe = config.enableMoeBlock
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0

        _selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = Gemma4MLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)

        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if enableMoe, let numExperts = config.numExperts, let topK = config.topKExperts,
            let moeIntermediate = config.moeIntermediateSize
        {
            _router.wrappedValue = Gemma4Router(
                hiddenSize: config.hiddenSize, numExperts: numExperts,
                topK: topK, eps: config.rmsNormEps)
            _experts.wrappedValue = Gemma4Experts(
                hiddenSize: config.hiddenSize,
                moeIntermediateSize: moeIntermediate,
                numExperts: numExperts)
            _postFeedforwardLayernorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _preFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hasPerLayerInput {
            let plDim = config.hiddenSizePerLayerInput
            _perLayerInputGate.wrappedValue = Linear(config.hiddenSize, plDim, bias: false)
            _perLayerProjection.wrappedValue = Linear(plDim, config.hiddenSize, bias: false)
            _postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        _layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        // Self-attention
        var residual = x
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache, sharedKV: sharedKV)
        h = postAttentionLayernorm(h)
        h = residual + h

        // Feedforward
        residual = h

        if enableMoe, let router, let experts,
            let norm1 = postFeedforwardLayernorm1,
            let norm2 = postFeedforwardLayernorm2,
            let preNorm2 = preFeedforwardLayernorm2
        {
            // MoE path: parallel dense MLP + expert MLP
            var h1 = preFeedforwardLayernorm(h)
            h1 = mlp(h1)
            h1 = norm1(h1)

            let (topKIndices, topKWeights) = router(h)
            var h2 = preNorm2(h)
            h2 = experts(h2, topKIndices: topKIndices, topKWeights: topKWeights)
            h2 = norm2(h2)

            h = h1 + h2
        } else {
            // Dense path: standard MLP
            h = preFeedforwardLayernorm(h)
            h = mlp(h)
        }

        h = postFeedforwardLayernorm(h)
        h = residual + h

        // Per-layer input gating (E2B/E4B)
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let plInput = perLayerInput
        {
            residual = h
            var g = gate(h)
            g = geluApproximate(g)
            g = g * plInput
            g = proj(g)
            g = norm(g)
            h = residual + g
        }

        // Per-layer scaling
        h = h * layerScalar

        return h
    }
}

// MARK: - Inner Text Model

public class Gemma4TextModelInner: Module {
    public let config: Gemma4TextConfig
    let windowSize: Int
    public let embedScale: Float

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding

    // Per-layer input embeddings (E2B/E4B) — public for VLM access
    @ModuleInfo(key: "embed_tokens_per_layer") public var embedTokensPerLayer: Embedding?

    let layers: [Gemma4DecoderLayer]
    let norm: RMSNorm

    // Per-layer input (E2B/E4B) — embedTokensPerLayer declared above
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Gemma4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNormZeroShift?
    let perLayerInputScale: Float

    init(_ config: Gemma4TextConfig) {
        self.config = config
        self.windowSize = config.slidingWindow
        self.embedScale = Float(config.hiddenSize).squareRoot()

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIdx: i)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input (E2B/E4B)
        let plDim = config.hiddenSizePerLayerInput
        if plDim > 0 {
            _embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * plDim)
            _perLayerModelProjection.wrappedValue = Gemma4ScaledLinear(
                inputDims: config.hiddenSize,
                outputDims: config.numHiddenLayers * plDim,
                scalar: pow(Float(config.hiddenSize), -0.5))
            _perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: plDim, eps: config.rmsNormEps)
            self.perLayerInputScale = pow(2.0, -0.5)
        } else {
            self.perLayerInputScale = 1.0
        }
    }

    /// Standard entry point: token IDs → embeddings → forward pass.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = embedTokens(inputs) * embedScale
        let perLayerInputs = computePerLayerInputs(inputIds: inputs, embeddings: h)
        return forward(h: h, perLayerInputs: perLayerInputs, cache: cache)
    }

    /// VLM entry point: pre-computed embeddings (with image features scattered in).
    /// perLayerInputTokens: masked token IDs for PLE (image positions zeroed out).
    public func callWithEmbeddings(
        _ inputEmbeddings: MLXArray,
        perLayerInputTokens: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var perLayerInputs: MLXArray?
        if let tokens = perLayerInputTokens {
            perLayerInputs = computePerLayerInputs(inputIds: tokens, embeddings: inputEmbeddings)
        }
        return forward(h: inputEmbeddings, perLayerInputs: perLayerInputs, cache: cache)
    }

    /// Compute per-layer input gating embeddings (E2B/E4B).
    private func computePerLayerInputs(inputIds: MLXArray, embeddings: MLXArray) -> MLXArray? {
        let plDim = config.hiddenSizePerLayerInput
        let numLayers = config.numHiddenLayers
        guard plDim > 0, let embedPL = embedTokensPerLayer, let projPL = perLayerModelProjection,
              let normPL = perLayerProjectionNorm else { return nil }

        let tokenPL = embedPL(inputIds) * Float(plDim).squareRoot()
        let tokenPLReshaped = tokenPL.reshaped(inputIds.shape + [numLayers, plDim])

        let modelPL = projPL(embeddings).reshaped(embeddings.shape.dropLast() + [numLayers, plDim])
        let modelPLNormed = normPL(modelPL)

        return (modelPLNormed + tokenPLReshaped) * perLayerInputScale
    }

    /// Core forward pass shared by both entry points.
    /// Known issue: Gemma 4 E4B crashes in MLX at B>=3 during batched prefill.
    private func forward(h: MLXArray, perLayerInputs: MLXArray?, cache: [KVCache]?) -> MLXArray {
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" && h.dim(0) > 1 {
            print("[Gemma4Forward] B=\(h.dim(0)) L=\(h.dim(1)) H=\(h.dim(2)) perLayerInputs=\(perLayerInputs?.shape.description ?? "nil")")
        }
        var h = h

        let cache: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)

        // Find representative cache indices for mask creation
        var globalCacheIdx: Int?
        var slidingCacheIdx: Int?
        for (i, layer) in layers.enumerated() {
            if layer.layerType == "full_attention" && globalCacheIdx == nil {
                globalCacheIdx = i
            } else if layer.layerType == "sliding_attention" && slidingCacheIdx == nil {
                slidingCacheIdx = i
            }
        }

        let globalMask = createAttentionMask(
            h: h, cache: globalCacheIdx.flatMap { cache[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: slidingCacheIdx.flatMap { cache[$0] }, windowSize: windowSize)

        // KV sharing store: layer_idx → (keys, values, scalarOffset, perSeqOffsets?, allEqual?)
        var sharedKVStore: [Int: ((MLXArray, MLXArray), Int, MLXArray?, Bool?)] = [:]

        // Flush the lazy graph every N layers during batched prefill (S>1).
        // RotatingKVCache creates ~20 lazy ops per layer; at 40 layers the
        // graph can overflow Metal's buffer limit. Flushing every 1 layer
        // serializes the GPU and kills throughput. Every 8 layers balances
        // graph size (~160 ops, safe) with pipelining (5 syncs not 40).
        // MUST NOT fire during decode (S=1).
        let seqLen = h.dim(1)
        let prefillFlushInterval = 8
        let isPrefill = h.dim(0) > 1 && seqLen > 1
        let debugBatch = ProcessInfo.processInfo.environment["AFM_DEBUG_PREFILL"] == "1" && h.dim(0) > 2
        for (i, (layer, c)) in zip(layers, cache).enumerated() {
            let isGlobal = layer.layerType == "full_attention"
            let mask = isGlobal ? globalMask : slidingMask

            // Per-layer input slice for this layer
            let plInput: MLXArray?
            if let pli = perLayerInputs {
                plInput = pli[0..., 0..., i, 0...]
            } else {
                plInput = nil
            }

            // KV sharing: check if this layer should reuse KV
            var layerSharedKV: (MLXArray, MLXArray)?
            let attn = layer.selfAttn
            if attn.isKvSharedLayer, let refIdx = attn.kvSharedLayerIndex,
                let (kv, refOffset, refOffsetArray, refAllEqual) = sharedKVStore[refIdx]
            {
                layerSharedKV = kv
                // Sync offset so RoPE positions match the shared KV.
                // Must sync BOTH scalar offset AND per-sequence offsets —
                // KV-shared layers skip cache.update(), so their batch cache
                // perSeqOffset stays at the stale merge-time value (e.g. [0,0]).
                // Without this, batch mode uses wrong RoPE positions for 67%
                // of Gemma 4 E4B layers.
                if let baseCache = c as? BaseKVCache {
                    baseCache.offset = refOffset
                    if let refArr = refOffsetArray {
                        baseCache.syncPerSeqOffsets(refArr, allEqual: refAllEqual)
                    }
                }
            }

            let preOffset = c?.offset ?? 0
            let preOffsetArray = c?.offsetArray
            let preAllEqual = (c as? BaseKVCache)?.allOffsetsEqual

            // Per-layer timing: force sync to measure actual GPU wall per layer.
            // Only fires in profile mode (AFM_PROFILE_LAYERS) to avoid serializing
            // the normal decode pipeline.
            let profileLayers = ProcessInfo.processInfo.environment["AFM_PROFILE_LAYERS"] == "1"
            let layerStart = profileLayers ? Date() : Date.distantPast

            h = layer(h, mask: mask, cache: c, perLayerInput: plInput, sharedKV: layerSharedKV)

            if profileLayers {
                _ = h.sum().item(Float.self)  // force GPU sync for timing
                let layerWall = Date().timeIntervalSince(layerStart) * 1000
                let layerTypeStr = layer.layerType
                let isShared = attn.isKvSharedLayer
                print("[Gemma4Layer] layer=\(i) type=\(layerTypeStr) shared=\(isShared) B=\(h.dim(0)) wall=\(String(format: "%.1f", layerWall))ms")
            }

            // Store KV for sharing if needed
            if attn.storeFullLengthKV, let kv = attn.lastKV {
                sharedKVStore[i] = (kv, preOffset, preOffsetArray, preAllEqual)
            }

            // Collapse the lazy graph at layer boundaries during batched prefill.
            if isPrefill && (i + 1) % prefillFlushInterval == 0 {
                MLX.eval(h)
                if let c = c { MLX.eval(c.innerState()) }
            }
            if debugBatch {
                print("[Gemma4Forward] layer \(i) B=\(h.dim(0)) OK h=\(h.shape)")
                fflush(stdout)
            }
        }

        return norm(h)
    }
}

// MARK: - Text Model (language_model level)

public class Gemma4TextModel: Module {
    public let config: Gemma4TextConfig

    @ModuleInfo(key: "model") public var model: Gemma4TextModelInner
    let finalLogitSoftcapping: Float?

    init(_ config: Gemma4TextConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        _model.wrappedValue = Gemma4TextModelInner(config)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)

        // Tied embeddings as LM head
        out = model.embedTokens.asLinear(out)

        // Logit softcapping
        if let cap = finalLogitSoftcapping, cap > 0 {
            out = tanh(out / cap) * cap
        }

        return out
    }
}

// MARK: - Top-Level Model

public class Gemma4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Gemma4TextModel
    let configuration: Gemma4Configuration

    public init(_ args: Gemma4Configuration) {
        self.configuration = args
        let textConfig = args.textConfig
        self.vocabularySize = textConfig.vocabSize

        self.kvHeads = (0 ..< textConfig.numHiddenLayers).map { i in
            let layerType = textConfig.layerTypes[i]
            let isSliding = layerType == "sliding_attention"
            let useKEqV = textConfig.attentionKEqV && !isSliding
            if useKEqV, let globalKV = textConfig.numGlobalKeyValueHeads {
                return globalKV
            }
            return textConfig.numKeyValueHeads
        }

        _languageModel.wrappedValue = Gemma4TextModel(textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let textConfig = configuration.textConfig
        return (0 ..< textConfig.numHiddenLayers).map { i in
            let layerType = textConfig.layerTypes[i]
            if layerType == "full_attention" {
                return KVCacheSimple() as any KVCache
            } else {
                return RotatingKVCache(maxSize: textConfig.slidingWindow, keep: 0) as any KVCache
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip rotary embedding weights
            if key.contains("self_attn.rotary_emb") { continue }
            // Skip vision/audio weights
            if key.hasPrefix("vision_tower.") || key.hasPrefix("embed_vision.")
                || key.hasPrefix("audio_tower.") || key.hasPrefix("embed_audio.")
            { continue }
            // Skip quantization metadata (vision/audio clipped linear markers)
            if key.contains("input_max") || key.contains("input_min")
                || key.contains("output_max") || key.contains("output_min")
            { continue }

            var newKey = key
            // Handle prefix mapping
            if key.hasPrefix("model.language_model.") {
                newKey = key.replacingOccurrences(
                    of: "model.language_model.", with: "language_model.model.")
            } else if !key.hasPrefix("language_model.") {
                newKey = "language_model." + key
            }

            sanitized[newKey] = value
        }

        // Remove tied lm_head if present
        if configuration.textConfig.tieWordEmbeddings {
            sanitized["language_model.lm_head.weight"] = nil
        }

        return sanitized
    }
}

// MARK: - LoRA

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
