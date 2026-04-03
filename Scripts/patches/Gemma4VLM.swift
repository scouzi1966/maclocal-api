//
//  Gemma4VLM.swift
//  mlx-swift-lm
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4
//
//  Gemma 4 Vision-Language Model: SigLIP2 vision encoder + Gemma 4 text model.
//  Supports all variants (E2B, E4B, 26B-A4B, 31B) with variable-resolution
//  image input, aspect-ratio preserving resize, and position-aware pooling.
//
//  FULLY SELF-CONTAINED — all text model components are duplicated here with
//  the Gemma4VL prefix so this file can live in MLXVLM without importing MLXLLM.
//

import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// ============================================================================
// MARK: - Text Configuration (duplicated from Gemma4Text.swift)
// ============================================================================

/// Nested rope parameters per layer type.
struct Gemma4VLRopeParams: Codable, Sendable {
    var ropeTheta: Float?
    var partialRotaryFactor: Float?
    var ropeType: String?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeType = "rope_type"
    }
}

/// Text-specific configuration for Gemma 4 models (VL copy).
public struct Gemma4VLTextConfig: Codable, Sendable {
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
    var ropeParameters: [String: Gemma4VLRopeParams]
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

    public init(from decoder: Swift.Decoder) throws {
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
        ropeParameters = try c.decode([String: Gemma4VLRopeParams].self, forKey: .ropeParameters)
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

    /// First layer index that shares KV from an earlier layer (0 means no sharing).
    var firstKvSharedLayerIdx: Int {
        numKvSharedLayers > 0 ? numHiddenLayers - numKvSharedLayers : numHiddenLayers
    }
}

// ============================================================================
// MARK: - Text RMSNorm Variants (duplicated from Gemma4Text.swift)
// ============================================================================

/// Gemma-style RMSNorm with +1 weight offset (stored weights are shifted by -1).
class Gemma4VLRMSNorm: Module, UnaryLayer {
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
class Gemma4VLRMSNormNoScale: Module {
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
class Gemma4VLRMSNormZeroShift: Module {
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

// ============================================================================
// MARK: - Text Scaled Linear (duplicated from Gemma4Text.swift)
// ============================================================================

/// Linear layer with output scaling (for per-layer model projection).
class Gemma4VLScaledLinear: Module {
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

// ============================================================================
// MARK: - Text MLP (duplicated from Gemma4Text.swift)
// ============================================================================

class Gemma4VLMLP: Module {
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

// ============================================================================
// MARK: - Text Router (duplicated from Gemma4Text.swift)
// ============================================================================

/// Expert router: RMSNorm(no-scale) -> scale -> project -> top-k -> renormalize.
class Gemma4VLRouter: Module {
    let numExperts: Int
    let topK: Int
    let rootSize: Float

    let norm: Gemma4VLRMSNormNoScale
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(hiddenSize: Int, numExperts: Int, topK: Int, eps: Float) {
        self.numExperts = numExperts
        self.topK = topK
        self.rootSize = pow(Float(hiddenSize), -0.5)

        self.norm = Gemma4VLRMSNormNoScale(eps: eps)
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

// ============================================================================
// MARK: - Text Experts (duplicated from Gemma4Text.swift)
// ============================================================================

/// Sparse MoE using SwitchGLU with GeGLU activation.
class Gemma4VLExperts: Module {
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

// ============================================================================
// MARK: - Text Attention (duplicated from Gemma4Text.swift)
// ============================================================================

class Gemma4VLAttention: Module {
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
    let vNorm: Gemma4VLRMSNormNoScale

    let rope: RoPE

    /// Stored KV for sharing with later layers (set during forward pass).
    var lastKV: (MLXArray, MLXArray)?

    init(_ config: Gemma4VLTextConfig, layerIdx: Int) {
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
        self.vNorm = Gemma4VLRMSNormNoScale(eps: config.rmsNormEps)

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

        // Read offset BEFORE cache update -- keys and queries must use the same offset
        let offset = cache?.offset ?? 0

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
            keys = rope(keys, offset: offset)

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }
        }

        if storeFullLengthKV {
            lastKV = (keys, values)
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1.0,  // Gemma4 uses scale=1.0 (scaling is in q/k norms)
            mask: mask
        )

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// ============================================================================
// MARK: - Text Decoder Layer (duplicated from Gemma4Text.swift)
// ============================================================================

class Gemma4VLDecoderLayer: Module {
    let layerType: String
    let enableMoe: Bool
    let hasPerLayerInput: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4VLAttention

    // Standard MLP (always present for both dense and MoE)
    @ModuleInfo(key: "mlp") var mlp: Gemma4VLMLP

    // Norms shared by both paths
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    // MoE-specific components (26B model)
    @ModuleInfo(key: "router") var router: Gemma4VLRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4VLExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?

    // Per-layer input gating (E2B/E4B)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scaling
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4VLTextConfig, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoe = config.enableMoeBlock
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0

        _selfAttn.wrappedValue = Gemma4VLAttention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = Gemma4VLMLP(
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
            _router.wrappedValue = Gemma4VLRouter(
                hiddenSize: config.hiddenSize, numExperts: numExperts,
                topK: topK, eps: config.rmsNormEps)
            _experts.wrappedValue = Gemma4VLExperts(
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

// ============================================================================
// MARK: - Text Model Inner (duplicated from Gemma4Text.swift)
// ============================================================================

class Gemma4VLTextModelInner: Module {
    let config: Gemma4VLTextConfig
    let windowSize: Int
    let embedScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    // Per-layer input embeddings (E2B/E4B)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?

    let layers: [Gemma4VLDecoderLayer]
    let norm: RMSNorm

    // Per-layer input projection
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Gemma4VLScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4VLRMSNormZeroShift?
    let perLayerInputScale: Float

    init(_ config: Gemma4VLTextConfig) {
        self.config = config
        self.windowSize = config.slidingWindow
        self.embedScale = Float(config.hiddenSize).squareRoot()

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers).map { i in
            Gemma4VLDecoderLayer(config, layerIdx: i)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Per-layer input (E2B/E4B)
        let plDim = config.hiddenSizePerLayerInput
        if plDim > 0 {
            _embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * plDim)
            _perLayerModelProjection.wrappedValue = Gemma4VLScaledLinear(
                inputDims: config.hiddenSize,
                outputDims: config.numHiddenLayers * plDim,
                scalar: pow(Float(config.hiddenSize), -0.5))
            _perLayerProjectionNorm.wrappedValue = Gemma4VLRMSNormZeroShift(
                dimensions: plDim, eps: config.rmsNormEps)
            self.perLayerInputScale = pow(2.0, -0.5)
        } else {
            self.perLayerInputScale = 1.0
        }
    }

    /// Standard entry point: token IDs -> embeddings -> forward pass.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = embedTokens(inputs) * embedScale
        let perLayerInputs = computePerLayerInputs(inputIds: inputs, embeddings: h)
        return forward(h: h, perLayerInputs: perLayerInputs, cache: cache)
    }

    /// VLM entry point: pre-computed embeddings (with image features scattered in).
    /// perLayerInputTokens: masked token IDs for PLE (image positions zeroed out).
    func callWithEmbeddings(
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
    private func forward(h: MLXArray, perLayerInputs: MLXArray?, cache: [KVCache]?) -> MLXArray {
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

        // KV sharing store: layer_idx -> (keys, values, offset)
        var sharedKVStore: [Int: ((MLXArray, MLXArray), Int)] = [:]

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
                let (kv, refOffset) = sharedKVStore[refIdx]
            {
                layerSharedKV = kv
                // Sync offset so RoPE positions match the shared KV
                if let baseCache = c as? BaseKVCache {
                    baseCache.offset = refOffset
                }
            }

            let preOffset = c?.offset ?? 0

            h = layer(h, mask: mask, cache: c, perLayerInput: plInput, sharedKV: layerSharedKV)

            // Store KV for sharing if needed
            if attn.storeFullLengthKV, let kv = attn.lastKV {
                sharedKVStore[i] = (kv, preOffset)
            }
        }

        return norm(h)
    }
}

// ============================================================================
// MARK: - Text Model (language_model level, duplicated from Gemma4Text.swift)
// ============================================================================

class Gemma4VLTextModel: Module {
    let config: Gemma4VLTextConfig

    @ModuleInfo(key: "model") var model: Gemma4VLTextModelInner
    let finalLogitSoftcapping: Float?

    init(_ config: Gemma4VLTextConfig) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        _model.wrappedValue = Gemma4VLTextModelInner(config)
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

// ============================================================================
// MARK: - Vision Configuration
// ============================================================================

public struct Gemma4VisionConfig: Codable, Sendable {
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var patchSize: Int
    var positionEmbeddingSize: Int
    var poolingKernelSize: Int
    var defaultOutputLength: Int
    var useClippedLinears: Bool
    var standardize: Bool
    var ropeParameters: Gemma4VisionRopeParams

    struct Gemma4VisionRopeParams: Codable, Sendable {
        var ropeTheta: Float
        var ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case defaultOutputLength = "default_output_length"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        useClippedLinears = try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false
        ropeParameters = try c.decodeIfPresent(Gemma4VisionRopeParams.self, forKey: .ropeParameters)
            ?? Gemma4VisionRopeParams(ropeTheta: 100.0, ropeType: "default")
    }

    var maxPatches: Int { defaultOutputLength * poolingKernelSize * poolingKernelSize }
}

// ============================================================================
// MARK: - VLM Configuration
// ============================================================================

public struct Gemma4VLMConfiguration: Codable, Sendable {
    var textConfig: Gemma4VLTextConfig
    var visionConfig: Gemma4VisionConfig
    var imageTokenId: Int
    var boiTokenId: Int
    var eoiTokenId: Int
    var visionSoftTokensPerImage: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(Gemma4VLTextConfig.self, forKey: .textConfig)
        visionConfig = try c.decode(Gemma4VisionConfig.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: .boiTokenId) ?? 255999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: .eoiTokenId) ?? 258882
        visionSoftTokensPerImage = try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage) ?? 280
    }
}

// ============================================================================
// MARK: - Vision RMSNorm (float32 computation)
// ============================================================================

/// RMSNorm with learned scale, computed in float32 (SigLIP2 pattern).
class VisionRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = (xf * xf).mean(axis: -1, keepDims: true)
        let normed = xf * rsqrt(variance + eps)
        return (normed * weight.asType(.float32)).asType(x.dtype)
    }
}

/// RMSNorm without learnable scale, computed in float32 (parameter-free).
class VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = (xf * xf).mean(axis: -1, keepDims: true)
        return (xf * rsqrt(variance + eps)).asType(x.dtype)
    }
}

// ============================================================================
// MARK: - Clippable Linear (SigLIP2 vision)
// ============================================================================

class ClippableLinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    let useClipping: Bool
    @ParameterInfo(key: "input_min") var inputMin: MLXArray?
    @ParameterInfo(key: "input_max") var inputMax: MLXArray?
    @ParameterInfo(key: "output_min") var outputMin: MLXArray?
    @ParameterInfo(key: "output_max") var outputMax: MLXArray?

    init(inputDims: Int, outputDims: Int, bias: Bool = false, useClipping: Bool = true) {
        self.useClipping = useClipping
        _linear.wrappedValue = Linear(inputDims, outputDims, bias: bias)
        if useClipping {
            _inputMin.wrappedValue = MLXArray(Float(-1e38))
            _inputMax.wrappedValue = MLXArray(Float(1e38))
            _outputMin.wrappedValue = MLXArray(Float(-1e38))
            _outputMax.wrappedValue = MLXArray(Float(1e38))
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if useClipping, let lo = inputMin, let hi = inputMax {
            h = MLX.clip(h, min: lo, max: hi)
        }
        h = linear(h)
        if useClipping, let lo = outputMin, let hi = outputMax {
            h = MLX.clip(h, min: lo, max: hi)
        }
        return h
    }
}

// ============================================================================
// MARK: - Multidimensional RoPE (SigLIP2 vision)
// ============================================================================

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// Apply 2D rotary position embeddings matching PyTorch's apply_multidimensional_rope.
/// Splits the head dimension into ndim parts and applies rotate_half independently
/// to each part (one per spatial dimension).
func applyMultidimensionalRope(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float = 100.0
) -> MLXArray {
    let headDim = inputs.dim(-1)
    let ndim = positions.dim(-1)  // 2 for 2D positions
    let channelsPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = channelsPerDim / 2

    var resultParts: [MLXArray] = []
    for d in 0 ..< ndim {
        let xPart = inputs[.ellipsis, (d * channelsPerDim) ..< ((d + 1) * channelsPerDim)]

        let freqExponents = (2.0 / Float(channelsPerDim))
            * MLXArray(0 ..< halfPerDim).asType(.float32)
        let timescale = pow(MLXArray(baseFrequency), freqExponents)
        let sinusoidInp = positions[.ellipsis, d ..< (d + 1)].asType(.float32) / timescale

        var cosD = cos(sinusoidInp)
        var sinD = sin(sinusoidInp)
        cosD = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        sinD = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        cosD = expandedDimensions(cosD, axis: 2)
        sinD = expandedDimensions(sinD, axis: 2)

        resultParts.append(xPart * cosD + rotateHalf(xPart) * sinD)
    }

    return concatenated(resultParts, axis: -1)
}

// ============================================================================
// MARK: - Vision Attention (SigLIP2 GQA with q/k/v norms, 2D RoPE)
// ============================================================================

class Gemma4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBaseFreq: Float

    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm
    let vNorm: VisionRMSNormNoScale

    init(_ config: Gemma4VisionConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.ropeBaseFreq = config.ropeParameters.ropeTheta
        let clip = config.useClippedLinears

        _qProj.wrappedValue = ClippableLinear(
            inputDims: config.hiddenSize, outputDims: numHeads * headDim, useClipping: clip)
        _kProj.wrappedValue = ClippableLinear(
            inputDims: config.hiddenSize, outputDims: numKVHeads * headDim, useClipping: clip)
        _vProj.wrappedValue = ClippableLinear(
            inputDims: config.hiddenSize, outputDims: numKVHeads * headDim, useClipping: clip)
        _oProj.wrappedValue = ClippableLinear(
            inputDims: numHeads * headDim, outputDims: config.hiddenSize, useClipping: clip)
        _qNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        _kNorm.wrappedValue = VisionRMSNorm(dimensions: headDim)
        self.vNorm = VisionRMSNormNoScale()
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let v = vNorm(vProj(x).reshaped(B, L, numKVHeads, headDim))

        q = qNorm(q)
        k = kNorm(k)

        // Apply 2D RoPE
        q = applyMultidimensionalRope(q, positions: positions, baseFrequency: ropeBaseFreq)
        k = applyMultidimensionalRope(k, positions: positions, baseFrequency: ropeBaseFreq)

        // Transpose to [B, H, L, D] for SDPA
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vt = v.transposed(0, 2, 1, 3)

        // Vision uses bidirectional attention (no causal mask), pass additive mask directly
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: vt, scale: 1.0, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// ============================================================================
// MARK: - Vision MLP (GeGLU)
// ============================================================================

class Gemma4VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: ClippableLinear

    init(_ config: Gemma4VisionConfig) {
        let clip = config.useClippedLinears
        _gateProj.wrappedValue = ClippableLinear(
            inputDims: config.hiddenSize, outputDims: config.intermediateSize, useClipping: clip)
        _upProj.wrappedValue = ClippableLinear(
            inputDims: config.hiddenSize, outputDims: config.intermediateSize, useClipping: clip)
        _downProj.wrappedValue = ClippableLinear(
            inputDims: config.intermediateSize, outputDims: config.hiddenSize, useClipping: clip)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// ============================================================================
// MARK: - Vision Transformer Block (4 RMSNorms per block)
// ============================================================================

class Gemma4VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4VisionAttention
    @ModuleInfo(key: "mlp") var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    init(_ config: Gemma4VisionConfig) {
        _selfAttn.wrappedValue = Gemma4VisionAttention(config)
        _mlp.wrappedValue = Gemma4VisionMLP(config)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let attnOut = postAttentionLayernorm(
            selfAttn(inputLayernorm(x), positions: positions, mask: mask))
        let h = x + attnOut
        let ffwOut = postFeedforwardLayernorm(mlp(preFeedforwardLayernorm(h)))
        return h + ffwOut
    }
}

// ============================================================================
// MARK: - Vision Patch Embedder (linear proj + factored 2D position table)
// ============================================================================

class Gemma4VisionPatchEmbedder: Module {
    let hiddenSize: Int
    let patchSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ParameterInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(_ config: Gemma4VisionConfig) {
        self.hiddenSize = config.hiddenSize
        self.patchSize = config.patchSize
        self.positionEmbeddingSize = config.positionEmbeddingSize

        _inputProj.wrappedValue = Linear(3 * patchSize * patchSize, hiddenSize, bias: false)
        _positionEmbeddingTable.wrappedValue = MLXArray.ones([2, positionEmbeddingSize, hiddenSize])
        super.init()
    }

    func callAsFunction(
        _ pixelValues: MLXArray, patchPositions: MLXArray, paddingPositions: MLXArray
    ) -> MLXArray {
        let patches = patchify(pixelValues)
        let posEmbed = positionEmbeddings(patchPositions, padding: paddingPositions)
        return patches + posEmbed
    }

    private func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let C = pixelValues.dim(1)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let p = patchSize
        let pH = H / p
        let pW = W / p

        // [B, C, pH, p, pW, p] -> [B, pH, pW, p, p, C] -> [B, pH*pW, C*p*p]
        var patches = pixelValues.reshaped(B, C, pH, p, pW, p)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(B, pH * pW, C * p * p)
        // Rescale: 2 * (x - 0.5) = 2x - 1
        patches = 2 * (patches - 0.5)
        return inputProj(patches.asType(inputProj.weight.dtype))
    }

    private func positionEmbeddings(_ positions: MLXArray, padding: MLXArray) -> MLXArray {
        // One-hot encode positions -> matmul with table -> sum over dimensions
        let oh = oneHot(positions, numClasses: positionEmbeddingSize)
        // [B, num_patches, 2, pos_size] -> [B, 2, num_patches, pos_size]
        let ohT = oh.transposed(0, 2, 1, 3).asType(positionEmbeddingTable.dtype)
        var pe = ohT.matmul(positionEmbeddingTable)
        pe = pe.sum(axis: 1)
        // Zero out padding positions
        pe = MLX.where(expandedDimensions(padding, axis: -1), MLXArray(Float(0)), pe)
        return pe
    }

    private func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
        (expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)).asType(.float32)
    }
}

// ============================================================================
// MARK: - Vision Pooler (position-aware average pooling)
// ============================================================================

class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let defaultOutputLength: Int
    let rootHiddenSize: Float

    init(_ config: Gemma4VisionConfig) {
        self.hiddenSize = config.hiddenSize
        self.defaultOutputLength = config.defaultOutputLength
        self.rootHiddenSize = Float(config.hiddenSize).squareRoot()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, patchPositions: MLXArray, paddingPositions: MLXArray,
        outputLength: Int? = nil
    ) -> (MLXArray, MLXArray) {
        let length = outputLength ?? defaultOutputLength
        let inputSeqLen = hiddenStates.dim(1)

        if inputSeqLen == length {
            return (hiddenStates * rootHiddenSize, paddingPositions)
        }

        // Position-aware average pooling
        let k = Int(Float(inputSeqLen / length).squareRoot())
        let kSquared = Float(k * k)

        let clamped = MLX.clip(patchPositions, min: 0)
        let maxX = clamped[.ellipsis, 0].max(axis: -1, keepDims: true) + 1
        var kernelIdxs = floor(clamped.asType(.float32) / Float(k)).asType(.int32)
        kernelIdxs = kernelIdxs[.ellipsis, 0] + (maxX / k) * kernelIdxs[.ellipsis, 1]

        // One-hot encode kernel indices -> weighted sum
        let oh = (expandedDimensions(kernelIdxs, axis: -1) .== MLXArray(0 ..< length))
            .asType(.float32)
        let weights = oh / kSquared

        // [B, L, length]^T x [B, L, D] -> [B, length, D]
        let output = weights.transposed(0, 2, 1).matmul(hiddenStates.asType(.float32))
            .asType(hiddenStates.dtype)

        let mask = MLX.logicalNot(MLX.all(weights .== Float(0), axis: 1))
        return (output * rootHiddenSize, mask)
    }
}

// ============================================================================
// MARK: - Vision Encoder (transformer layers)
// ============================================================================

class Gemma4VisionEncoder: Module {
    let layers: [Gemma4VisionBlock]

    init(_ config: Gemma4VisionConfig) {
        self.layers = (0 ..< config.numHiddenLayers).map { _ in Gemma4VisionBlock(config) }
    }

    func callAsFunction(_ h: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = h
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

// ============================================================================
// MARK: - Vision Tower (patch embed + transformer + pooler + optional standardize)
// ============================================================================

class Gemma4VisionTower: Module {
    let config: Gemma4VisionConfig
    let maxPatches: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: Gemma4VisionEncoder
    let pooler: Gemma4VisionPooler

    // Standardization (26B/31B)
    @ParameterInfo(key: "std_bias") var stdBias: MLXArray?
    @ParameterInfo(key: "std_scale") var stdScale: MLXArray?

    init(_ config: Gemma4VisionConfig) {
        self.config = config
        self.maxPatches = config.maxPatches

        _patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config)
        _encoder.wrappedValue = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

        if config.standardize {
            _stdBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            _stdScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }

        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        // Ensure channel-first [B, C, H, W]
        // Framework may pass various layouts; find the channel dim (size 3) and transpose
        var pv = pixelValues
        if pv.ndim == 3 {
            // [H, W, C] → [1, C, H, W]
            pv = expandedDimensions(pv.transposed(2, 0, 1), axis: 0)
        } else if pv.ndim == 4 {
            // Find which dim is channels (size 3) and move to dim(1)
            if pv.dim(1) != 3 {
                if pv.dim(3) == 3 {
                    // [B, H, W, C] → [B, C, H, W]
                    pv = pv.transposed(0, 3, 1, 2)
                } else if pv.dim(2) == 3 {
                    // [B, W, C, H] → [B, C, H, W]
                    pv = pv.transposed(0, 2, 3, 1)
                }
            }
        }
        let B = pv.dim(0)
        let H = pv.dim(2)
        let W = pv.dim(3)
        let patchSize = config.patchSize
        let numReal = (H / patchSize) * (W / patchSize)

        let (patchPositions, paddingPositions) = computePatchPositions(
            B: B, H: H, W: W, patchSize: patchSize)

        // Embed real patches
        var embeds = patchEmbedder(
            pv,
            patchPositions: patchPositions[0..., ..<numReal],
            paddingPositions: paddingPositions[0..., ..<numReal])

        // Pad to maxPatches
        let numPadding = maxPatches - numReal
        if numPadding > 0 {
            let padEmbeds = MLXArray.zeros([B, numPadding, config.hiddenSize])
                .asType(embeds.dtype)
            embeds = concatenated([embeds, padEmbeds], axis: 1)
        }

        // Build bidirectional attention mask [B, 1, L, L]
        let validMask = MLX.logicalNot(paddingPositions)
        let attnMask = expandedDimensions(validMask, axis: 1) * expandedDimensions(validMask, axis: 2)
        let attnMaskFloat = MLX.where(
            attnMask,
            MLXArray(Float(0)).asType(embeds.dtype),
            MLXArray(Float(-1e9)).asType(embeds.dtype))
        let attnMask4D = expandedDimensions(attnMaskFloat, axis: 1)

        // Run transformer
        let hidden = encoder(embeds, positions: patchPositions, mask: attnMask4D)

        // Pool
        let (pooled, poolMask) = pooler(
            hidden, patchPositions: patchPositions, paddingPositions: paddingPositions)

        // Strip padding: take only valid pooled tokens
        var result = pooled
        let nValid = poolMask.asType(.int32).sum(axis: -1)
        let validCount = nValid[0].item(Int.self)
        if validCount > 0 && validCount < pooled.dim(1) {
            result = pooled[0..., ..<validCount]
        }

        // Standardize (E4B: standardize=false, 26B/31B: standardize=true)
        if let bias = stdBias, let scale = stdScale {
            result = (result - bias) * scale
        }

        return result
    }

    private func computePatchPositions(B: Int, H: Int, W: Int, patchSize: Int) -> (
        MLXArray, MLXArray
    ) {
        let pH = H / patchSize
        let pW = W / patchSize
        let numReal = pH * pW
        let numPad = maxPatches - numReal

        // Grid positions [numReal, 2]
        var realPos = [[Int32]]()
        for y in 0 ..< pH {
            for x in 0 ..< pW {
                realPos.append([Int32(x), Int32(y)])
            }
        }

        var positions = realPos
        if numPad > 0 {
            for _ in 0 ..< numPad {
                positions.append([-1, -1])
            }
        }

        let posArray = MLXArray(positions.flatMap { $0 }).reshaped(1, maxPatches, 2)
        let batchPos = MLX.broadcast(posArray, to: [B, maxPatches, 2])

        var padding = [Bool](repeating: false, count: numReal)
        if numPad > 0 {
            padding += [Bool](repeating: true, count: numPad)
        }
        let padArray = MLXArray(padding).reshaped(1, maxPatches)
        let batchPad = MLX.broadcast(padArray, to: [B, maxPatches])

        return (batchPos, batchPad)
    }
}

// ============================================================================
// MARK: - Multimodal Embedder (Linear + RMSNormNoScale)
// ============================================================================

class Gemma4MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    let norm: Gemma4VLRMSNormNoScale

    init(visionDim: Int, textDim: Int, eps: Float = 1e-6) {
        _embeddingProjection.wrappedValue = Linear(visionDim, textDim, bias: false)
        self.norm = Gemma4VLRMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(embeddingProjection(x))
    }
}

// ============================================================================
// MARK: - Masked Scatter Helper
// ============================================================================

func maskedScatter(_ input: MLXArray, mask: MLXArray, source: MLXArray) -> MLXArray {
    let maskFlat = mask.flattened().asType(.int32)
    let indices = MLX.cumsum(maskFlat) - 1
    let aligned = source.flattened()[indices % source.size]
    return MLX.where(maskFlat, aligned, input.flattened()).reshaped(input.shape)
}

// ============================================================================
// MARK: - VLM Model
// ============================================================================

public class Gemma4VLM: Module, VLMModel, KVCacheDimensionProvider {
    public let config: Gemma4VLMConfiguration

    @ModuleInfo(key: "language_model") var languageModel: Gemma4VLTextModel
    @ModuleInfo(key: "vision_tower") var visionTower: Gemma4VisionTower
    @ModuleInfo(key: "embed_vision") var embedVision: Gemma4MultimodalEmbedder

    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] {
        (0 ..< config.textConfig.numHiddenLayers).map { i in
            let layerType = config.textConfig.layerTypes[i]
            let isSliding = layerType == "sliding_attention"
            let useKEqV = config.textConfig.attentionKEqV && !isSliding
            if useKEqV, let globalKV = config.textConfig.numGlobalKeyValueHeads {
                return globalKV
            }
            return config.textConfig.numKeyValueHeads
        }
    }

    public init(_ config: Gemma4VLMConfiguration) {
        self.config = config
        _languageModel.wrappedValue = Gemma4VLTextModel(config.textConfig)
        _visionTower.wrappedValue = Gemma4VisionTower(config.visionConfig)
        _embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            visionDim: config.visionConfig.hiddenSize,
            textDim: config.textConfig.hiddenSize,
            eps: config.visionConfig.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?)
        throws -> PrepareResult
    {
        let tokens = input.text.tokens
        let pixelValues = input.image?.pixels

        if pixelValues == nil {
            // Text-only: delegate to the standard text path
            let logits = languageModel(tokens, cache: cache)
            return .logits(LMOutput(logits: logits))
        }

        // Get text embeddings scaled by sqrt(hidden_size)
        // Ensure batch dimension [1, S] for tokens, [1, S, D] for embeds
        var batchTokens = tokens
        if batchTokens.ndim == 1 {
            batchTokens = expandedDimensions(batchTokens, axis: 0)
        }
        let embeds = languageModel.model.embedTokens(batchTokens) * languageModel.model.embedScale

        // Run vision tower — ensure channel-first [B, C, H, W]
        var pixels = pixelValues!
        if pixels.ndim == 4 && pixels.dim(3) == 3 {
            // [B, H, W, C] → [B, C, H, W]
            pixels = pixels.transposed(0, 3, 1, 2)
        }
        let imageFeatures = visionTower(pixels)
        let projectedFeatures = embedVision(imageFeatures).asType(embeds.dtype)

        // Scatter image features into text embeddings at image_token positions
        let imageMask = batchTokens .== config.imageTokenId
        let maskExpanded = MLX.broadcast(
            expandedDimensions(imageMask, axis: -1), to: embeds.shape)
        let finalEmbeds = maskedScatter(embeds, mask: maskExpanded, source: projectedFeatures)

        // Build per-layer input tokens for PLE (E2B/E4B)
        // Image positions are zeroed out so PLE doesn't try to embed image tokens
        var perLayerInputTokens: MLXArray? = nil
        if languageModel.model.config.hiddenSizePerLayerInput > 0 {
            let textMask = MLX.logicalNot(imageMask)
            perLayerInputTokens = MLX.where(textMask, batchTokens, MLXArray.zeros(like: batchTokens))
        }

        // Run through the text model with pre-computed embeddings
        var out = languageModel.model.callWithEmbeddings(
            finalEmbeds, perLayerInputTokens: perLayerInputTokens, cache: cache)

        // LM head (tied embeddings)
        out = languageModel.model.embedTokens.asLinear(out)

        // Logit softcapping
        if let cap = languageModel.finalLogitSoftcapping, cap > 0 {
            out = tanh(out / cap) * cap
        }

        return .logits(LMOutput(logits: out))
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let textConfig = config.textConfig
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
        let useClipped = config.visionConfig.useClippedLinears
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip rotary embedding weights
            if key.contains("self_attn.rotary_emb") { continue }
            // Skip audio weights
            if key.contains("audio_tower") || key.contains("embed_audio") { continue }
            // Skip clipping params when not used in vision; always skip for text
            if key.contains("input_max") || key.contains("input_min")
                || key.contains("output_max") || key.contains("output_min")
            {
                if key.contains("vision_tower") && !useClipped { continue }
                if !key.contains("vision_tower") { continue }
            }

            var newKey = key
            var v = value

            // Strip "model." prefix from HuggingFace checkpoint
            if newKey.hasPrefix("model.") {
                newKey = String(newKey.dropFirst("model.".count))
            }
            // Remap language_model.X -> language_model.model.X
            if newKey.hasPrefix("language_model.") && !newKey.hasPrefix("language_model.model.") {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model." + rest
            }

            // MoE: experts.down_proj -> experts.switch_glu.down_proj.weight
            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj", with: ".experts.switch_glu.down_proj.weight")
            }
            // MoE: experts.gate_up_proj -> split into switch_glu.gate_proj + switch_glu.up_proj
            if newKey.hasSuffix(".experts.gate_up_proj") {
                let gateKey = newKey.replacingOccurrences(
                    of: ".experts.gate_up_proj",
                    with: ".experts.switch_glu.gate_proj.weight")
                let upKey = newKey.replacingOccurrences(
                    of: ".experts.gate_up_proj",
                    with: ".experts.switch_glu.up_proj.weight")

                let swapped = v.swappedAxes(-1, -2)
                let midDim = swapped.dim(-1) / 2
                sanitized[gateKey] = swapped[.ellipsis, ..<midDim].swappedAxes(-1, -2)
                sanitized[upKey] = swapped[.ellipsis, midDim...].swappedAxes(-1, -2)
                continue
            }

            sanitized[newKey] = v
        }

        // Handle tied embeddings
        if config.textConfig.tieWordEmbeddings {
            sanitized["language_model.lm_head.weight"] = nil
        }

        return sanitized
    }
}

// ============================================================================
// MARK: - LoRA
// ============================================================================

extension Gemma4VLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// ============================================================================
// MARK: - Processor Configuration
// ============================================================================

public struct Gemma4VLMProcessorConfiguration: Codable, Sendable {
    var imageMean: [Float]?
    var imageStd: [Float]?
    var patchSize: Int?
    var maxSoftTokens: Int?
    var poolingKernelSize: Int?
    var imageSize: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case poolingKernelSize = "pooling_kernel_size"
        case imageSize = "size"
    }
}

// ============================================================================
// MARK: - Processor
// ============================================================================

public class Gemma4VLMProcessor: UserInputProcessor {
    let config: Gemma4VLMProcessorConfiguration
    let tokenizer: any Tokenizer

    let patchSize: Int
    let maxSoftTokens: Int
    let poolingKernelSize: Int
    let imageTokenId: Int
    let boiToken: String
    let eoiToken: String
    let imageToken: String

    public init(_ config: Gemma4VLMProcessorConfiguration, _ tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        self.patchSize = config.patchSize ?? 16
        self.maxSoftTokens = config.maxSoftTokens ?? 280
        self.poolingKernelSize = config.poolingKernelSize ?? 3
        // Token strings from tokenizer_config.json
        self.imageTokenId = 258880  // <|image|>
        self.boiToken = "<|image>"
        self.eoiToken = "<image|>"
        self.imageToken = "<|image|>"
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Use structured content generator that includes image references in messages
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        // Apply chat template to get token IDs
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: true,
            truncation: false, maxLength: nil,
            tools: input.tools, additionalContext: input.additionalContext)

        // Process images and expand BOI token to BOI + N*image_token + EOI
        var processedImage: LMInput.ProcessedImage?

        if !input.images.isEmpty {
            let boiTokenId = 255999   // <|image>
            let imageTokenId = 258880 // <|image|>
            let eoiTokenId = 258882   // <image|>

            var allPixels = [MLXArray]()
            var softTokenCounts = [Int]()

            for image in input.images {
                let processed = try processImage(image)
                allPixels.append(processed.pixels)
                softTokenCounts.append(processed.softTokens)
            }

            // Expand: each <|image|> token → <|image> + N*<|image|> + <image|>
            // The chat template emits a single <|image|> per image; expand to actual soft token count
            var expandedTokens = [Int]()
            var imageIdx = 0
            for token in promptTokens {
                if token == imageTokenId && imageIdx < softTokenCounts.count {
                    let n = softTokenCounts[imageIdx]
                    expandedTokens.append(boiTokenId)
                    expandedTokens.append(contentsOf: Array(repeating: imageTokenId, count: n))
                    expandedTokens.append(eoiTokenId)
                    imageIdx += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens

            if allPixels.count == 1 {
                processedImage = LMInput.ProcessedImage(pixels: allPixels[0])
            } else {
                processedImage = LMInput.ProcessedImage(
                    pixels: concatenated(allPixels, axis: 0))
            }
        }

        return LMInput(
            text: .init(tokens: MLXArray(promptTokens)),
            image: processedImage)
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let softTokens: Int
    }

    private func processImage(_ image: UserInput.Image) throws -> ProcessedImage {
        let ciImage = try image.asCIImage()

        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw VLMError.imageProcessingFailure("Failed to create CGImage")
        }

        let width = cgImage.width
        let height = cgImage.height

        // Aspect-ratio preserving resize
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
        let targetPx = maxPatches * patchSize * patchSize
        let factor = (Float(targetPx) / Float(width * height)).squareRoot()
        let sideMult = poolingKernelSize * patchSize

        var targetH = Int(floor(factor * Float(height) / Float(sideMult))) * sideMult
        var targetW = Int(floor(factor * Float(width) / Float(sideMult))) * sideMult

        if targetH == 0 && targetW == 0 {
            targetH = sideMult
            targetW = sideMult
        } else if targetH == 0 {
            targetH = sideMult
        } else if targetW == 0 {
            targetW = sideMult
        }

        // Resize using CoreImage bicubic resampling
        let targetSize = CGSize(width: targetW, height: targetH)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(ciImage)
        let resized = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)

        // Convert to MLXArray [1, H, W, C] then to [1, C, H, W]
        let pixelArray = MediaProcessing.asMLXArray(resized)
        // pixelArray is [1, H, W, C] from MediaProcessing; transpose to channel-first [1, C, H, W]
        let channelFirst = pixelArray.transposed(0, 3, 1, 2)

        // Rescale: MediaProcessing.asMLXArray already produces [0, 1] range float32

        let numPatches = (targetH / patchSize) * (targetW / patchSize)
        let softTokens = numPatches / (poolingKernelSize * poolingKernelSize)

        return ProcessedImage(pixels: channelFirst, softTokens: softTokens)
    }
}

// ============================================================================
// MARK: - String replacingOccurrences with closure
// ============================================================================

extension String {
    /// Replace occurrences of a substring, using a closure for each replacement.
    fileprivate func replacingOccurrences(
        of target: String, with replacement: (String) -> String
    ) -> String {
        var result = ""
        var remaining = self[startIndex...]
        while let range = remaining.range(of: target) {
            result += remaining[..<range.lowerBound]
            result += replacement(String(remaining[range]))
            remaining = remaining[range.upperBound...]
        }
        result += remaining
        return result
    }
}
