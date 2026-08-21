//
//  Qwen3Next.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen3NextConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_next"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int

    // Linear attention parameters
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int

    // MoE parameters
    var numExperts: Int
    var numExpertsPerToken: Int
    var decoderSparseStep: Int
    var sharedExpertIntermediateSize: Int
    var mlpOnlyLayers: [Int]
    var moeIntermediateSize: Int

    // Normalization and embedding
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float = 1_000_000
    var partialRotaryFactor: Float = 1.0
    var maxPositionEmbeddings: Int = 32768
    var fullAttentionInterval: Int = 4

    // Optional
    var normTopkProb: Bool = false
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case fullAttentionInterval = "full_attention_interval"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_next"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)

        self.linearNumValueHeads = try container.decode(Int.self, forKey: .linearNumValueHeads)
        self.linearNumKeyHeads = try container.decode(Int.self, forKey: .linearNumKeyHeads)
        self.linearKeyHeadDim = try container.decode(Int.self, forKey: .linearKeyHeadDim)
        self.linearValueHeadDim = try container.decode(Int.self, forKey: .linearValueHeadDim)
        self.linearConvKernelDim = try container.decode(Int.self, forKey: .linearConvKernelDim)

        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.decoderSparseStep = try container.decode(Int.self, forKey: .decoderSparseStep)
        self.sharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        self.mlpOnlyLayers = try container.decode([Int].self, forKey: .mlpOnlyLayers)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)

        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.normTopkProb =
            try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

// MARK: - RMSNormGated

class Qwen3NextRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(hiddenSize: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        if let gate {
            x = silu(gate) * x
        }
        return x
    }
}

// MARK: - Full Attention (every Nth layer)

class Qwen3NextAttention: Module {
    let args: Qwen3NextConfiguration
    let scale: Float
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: Qwen3NextConfiguration) {
        self.args = args
        self.numHeads = args.attentionHeads
        self.numKVHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        // Q projects to 2x (queries + gate)
        _qProj.wrappedValue = Linear(
            args.hiddenSize, numHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, numKVHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, numKVHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            numHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: ropeDims, traditional: false, base: args.ropeTheta, scale: ropeScale)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // Q projection -> split into queries and gate
        let qProjOutput = qProj(x)
        let qReshaped = qProjOutput.reshaped(B, L, numHeads, -1)
        let qSplits = MLX.split(qReshaped, parts: 2, axis: -1)
        var queries = qSplits[0]
        let gate = qSplits[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        // Apply Q/K norms and reshape
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, numKVHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        // RoPE
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Attention with cache
        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        // Apply output gating: output * sigmoid(gate)
        output = output * sigmoid(gate)

        return oProj(output)
    }
}

// MARK: - MLP

class Qwen3NextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Gated DeltaNet (Linear Attention)

class Qwen3NextGatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let layerNormEpsilon: Float
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkvz") var inProjQKVZ: Linear
    @ModuleInfo(key: "in_proj_ba") var inProjBA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen3NextConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.numVHeads = config.linearNumValueHeads
        self.numKHeads = config.linearNumKeyHeads
        self.headKDim = config.linearKeyHeadDim
        self.headVDim = config.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = config.linearConvKernelDim
        self.layerNormEpsilon = config.rmsNormEps
        self.convDim = keyDim * 2 + valueDim

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: false
        )

        // Projects to q, k, v, z (gate for output)
        _inProjQKVZ.wrappedValue = Linear(
            hiddenSize, keyDim * 2 + valueDim * 2, bias: false)
        // Projects to beta (b) and alpha (a)
        _inProjBA.wrappedValue = Linear(
            hiddenSize, numVHeads * 2, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLX.log(MLXArray.ones([numVHeads]) * 8.0)

        _norm.wrappedValue = Qwen3NextRMSNormGated(
            hiddenSize: headVDim, eps: config.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    /// Reshape interleaved projections into per-head tensors.
    private func fixQueryKeyValueOrdering(
        mixedQKVZ: MLXArray, mixedBA: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let nk = numKHeads
        let dn = headKDim
        let nv = numVHeads
        let dv = headVDim

        let batchShape = Array(mixedQKVZ.shape.dropLast())
        let qkvzReshaped = mixedQKVZ.reshaped(batchShape + [nk, -1])
        let baReshaped = mixedBA.reshaped(batchShape + [nk, -1])

        // Split qkvz along last dim: q[dn], k[dn], v[nv/nk * dv], z[nv/nk * dv]
        let vPerK = nv / nk
        let splitIndices = [dn, 2 * dn, 2 * dn + vPerK * dv]
        let qkvzSplits = MLX.split(qkvzReshaped, indices: splitIndices, axis: -1)
        let q = qkvzSplits[0]
        let k = qkvzSplits[1]
        let vRaw = qkvzSplits[2]
        let zRaw = qkvzSplits[3]

        // Split ba along last dim: b[nv/nk], a[nv/nk]
        let baSplits = MLX.split(baReshaped, indices: [vPerK], axis: -1)
        let bRaw = baSplits[0]
        let aRaw = baSplits[1]

        // Reshape v, z to [B, S, Hv, Dv] and b, a to [B, S, Hv]
        let v = vRaw.reshaped(batchShape + [nv, dv])
        let z = zRaw.reshaped(batchShape + [nv, dv])
        let b = bRaw.reshaped(batchShape + [nv])
        let a = aRaw.reshaped(batchShape + [nv])

        return (q, k, v, z, b, a)
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray? = nil, cache: ArraysCache? = nil
    ) -> MLXArray {
        let (B, S, _) = (inputs.dim(0), inputs.dim(1), inputs.dim(2))

        let (q, k, v, z, b, a) = fixQueryKeyValueOrdering(
            mixedQKVZ: inProjQKVZ(inputs),
            mixedBA: inProjBA(inputs)
        )

        // Conv state management
        let convState: MLXArray
        if let cache, let cached = cache[0] {
            convState = cached
        } else {
            convState = MLXArray.zeros(
                [B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        // Concatenate q, k, v for conv
        let mixedQKV = concatenated(
            [q.reshaped(B, S, -1), k.reshaped(B, S, -1), v.reshaped(B, S, -1)],
            axis: -1
        )

        var convInput: MLXArray
        if let mask {
            convInput = which(mask[.ellipsis, .newAxis], mixedQKV, 0)
        } else {
            convInput = mixedQKV
        }
        convInput = concatenated([convState, convInput], axis: 1)

        // Update conv cache
        if let cache {
            let nKeep = convKernelSize - 1
            cache[0] = convInput[0..., (convInput.dim(1) - nKeep)...]
        }

        // Apply conv + silu
        let convOut = silu(conv1d(convInput))

        // Split conv output back into q, k, v
        let convSplits = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var qConv = convSplits[0].reshaped(B, S, numKHeads, headKDim)
        var kConv = convSplits[1].reshaped(B, S, numKHeads, headKDim)
        let vConv = convSplits[2].reshaped(B, S, numVHeads, headVDim)

        // Q/K normalization (no learned weights)
        let invScale = pow(Float(headKDim), -0.5)
        qConv = (invScale * invScale) * manualRmsNorm(qConv, eps: 1e-6)
        kConv = invScale * manualRmsNorm(kConv, eps: 1e-6)

        // Recurrent state
        let state: MLXArray? = cache?[1]

        let (out, newState) = gatedDeltaUpdate(
            q: qConv, k: kConv, v: vConv,
            a: a, b: b,
            ALog: aLog, dtBias: dtBias,
            state: state, mask: mask,
            useKernel: true
        )

        // Update recurrent state cache
        if let cache {
            cache[1] = newState
        }

        // Apply gated norm and output projection
        let normed = norm(out, gate: z)
        return outProj(normed.reshaped(B, S, -1))
    }
}

/// Manual RMS norm without learned weights: x * rsqrt(mean(x^2) + eps)
private func manualRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(variance + eps)
}

// MARK: - Sparse MoE with Shared Expert

class Qwen3NextSparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen3NextConfiguration) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: numExperts
        )
        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Expert routing
        let gates = gate(x)
        let softGates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let inds = MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = MLX.takeAlong(softGates, inds, axis: -1)

        if normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let moeOutput = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        // Shared expert with gating
        let sharedY = sharedExpert(x)
        let gatedSharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return moeOutput + gatedSharedY
    }
}

// MARK: - Decoder Layer

class Qwen3NextDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3NextGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3NextAttention?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: Qwen3NextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen3NextGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen3NextAttention(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if !args.mlpOnlyLayers.contains(layerIdx),
            args.numExperts > 0, (layerIdx + 1) % args.decoderSparseStep == 0
        {
            self.mlp = Qwen3NextSparseMoeBlock(args)
        } else {
            self.mlp = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        faMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? ArraysCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: faMask, cache: cache)
        }
        let h = x + r
        let out = h + mlp(postAttentionLayerNorm(h))
        return out
    }
}

// MARK: - Inner Model

public class Qwen3NextModelInner: Module {
    let args: Qwen3NextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3NextDecoderLayer]
    let norm: RMSNorm

    /// Index of the first linear (GatedDeltaNet) layer for SSM mask creation
    let ssmIdx: Int
    /// Index of the first full attention layer for FA mask creation
    let faIdx: Int

    init(_ args: Qwen3NextConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            Qwen3NextDecoderLayer(args, layerIdx: i)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Find first linear and first attention layer indices
        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let cache: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)

        // Create masks from representative caches
        let faMask = createAttentionMask(h: h, cache: cache[faIdx])
        let ssmMask = createSSMMask(h: h, cache: cache[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, faMask: faMask, ssmMask: ssmMask, cache: cache[i])
        }

        return norm(h)
    }
}

// MARK: - Top-Level Model

public class Qwen3NextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3NextModelInner
    let configuration: Qwen3NextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3NextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        // Linear layers use ArraysCache (0 kvHeads), attention layers use KVCacheSimple
        self.kvHeads = (0 ..< args.hiddenLayers).map { i in
            let isLinear = (i + 1) % args.fullAttentionInterval != 0
            return isLinear ? 0 : args.kvHeads
        }
        self.model = Qwen3NextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        model.layers.map { layer in
            if layer.isLinear {
                // MambaCache is ArraysCache(size: 2), compatible with createSSMMask
                return MambaCache() as any KVCache
            } else {
                return KVCacheSimple() as any KVCache
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        // 1. Filter out mtp.* keys (multi-token prediction weights)
        sanitizedWeights = sanitizedWeights.filter { !$0.key.hasPrefix("mtp.") }

        // Handle tied embeddings
        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // Match Python: if experts are already stacked (quantized model), return early.
        // Quantized models already have norm +1.0 and conv1d transpose applied during quantization.
        guard sanitizedWeights["model.layers.0.mlp.experts.0.up_proj.weight"] != nil else {
            return sanitizedWeights
        }

        // 2. Stack MoE expert weights into SwitchGLU format
        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l).mlp"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                if sanitizedWeights["\(prefix).experts.0.\(n).weight"] != nil {
                    let toJoin = (0 ..< configuration.numExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).experts.\(e).\(n).weight")!
                    }
                    sanitizedWeights["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                }
            }
        }

        // 3. Transpose conv1d weights and 4. Add +1.0 to norm weights
        // Only for unquantized models - quantized models already have these applied.
        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (k, v) in sanitizedWeights {
            // Conv1d weight transpose: Python [out, in, kernel] -> Swift [out, kernel, in]
            if k.contains("conv1d.weight"), v.shape.last != 1 {
                sanitizedWeights[k] = v.swappedAxes(1, 2)
            }
            // Norm weights stored without +1 bias in safetensors
            if normSuffixes.contains(where: { k.hasSuffix($0) }) {
                if v.ndim == 1 {
                    sanitizedWeights[k] = v + 1.0
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension Qwen3NextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
