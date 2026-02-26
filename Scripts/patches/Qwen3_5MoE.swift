//
//  Qwen3_5MoE.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//  and https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//
//  Qwen3.5 is a hybrid architecture mixing GatedDeltaNet linear attention (3 of every 4
//  layers) with full gated attention, plus sparse MoE with a shared expert on every layer.
//  The key difference from Qwen3Next is the GatedDeltaNet projection layout: Qwen3.5 uses
//  4 separate projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) instead of
//  Qwen3Next's 2 combined ones (in_proj_qkvz, in_proj_ba).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen3_5MoETextConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_5_moe_text"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int = 14336
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int?

    // Linear attention parameters
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 128
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4

    // MoE parameters
    var numExperts: Int = 0
    var numExpertsPerToken: Int = 0
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var mlpOnlyLayers: [Int] = []

    // Normalization and embedding
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int
    var fullAttentionInterval: Int = 4

    // RoPE - extracted from rope_parameters
    var ropeTheta: Float = 100_000
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 262144

    // Optional
    var normTopkProb: Bool = true
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil

    /// Computed head dimension (hiddenSize / attentionHeads if not explicit)
    var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / attentionHeads)
    }

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
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case fullAttentionInterval = "full_attention_interval"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encodeIfPresent(headDim, forKey: .headDim)
        try container.encode(linearNumValueHeads, forKey: .linearNumValueHeads)
        try container.encode(linearNumKeyHeads, forKey: .linearNumKeyHeads)
        try container.encode(linearKeyHeadDim, forKey: .linearKeyHeadDim)
        try container.encode(linearValueHeadDim, forKey: .linearValueHeadDim)
        try container.encode(linearConvKernelDim, forKey: .linearConvKernelDim)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try container.encode(sharedExpertIntermediateSize, forKey: .sharedExpertIntermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(mlpOnlyLayers, forKey: .mlpOnlyLayers)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(fullAttentionInterval, forKey: .fullAttentionInterval)
        try container.encode(normTopkProb, forKey: .normTopkProb)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
    }

    /// Intermediate struct for decoding nested rope_parameters
    private struct RopeParameters: Codable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?
        var type: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case type
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5_moe_text"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)

        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4

        self.numExperts =
            try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 0
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.mlpOnlyLayers =
            try container.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []

        self.rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.normTopkProb =
            try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144

        // Extract rope_theta and partial_rotary_factor from nested rope_parameters
        if let ropeParams = try container.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        {
            self.ropeTheta = ropeParams.ropeTheta ?? 100_000
            self.partialRotaryFactor = ropeParams.partialRotaryFactor ?? 0.25
        } else {
            self.ropeTheta = 100_000
            self.partialRotaryFactor = 0.25
        }
    }
}

/// Wrapper configuration for qwen3_5_moe (has text_config)
public struct Qwen3_5MoEConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_5_moe"
    var textConfig: Qwen3_5MoETextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }
}

// MARK: - RMSNormGated (reused from Qwen3Next concept)

private class Qwen3_5RMSNormGated: Module {
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

private class Qwen3_5Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: Qwen3_5MoETextConfiguration) {
        self.numHeads = args.attentionHeads
        self.numKVHeads = args.kvHeads
        self.headDim = args.resolvedHeadDim
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

        self.rope = RoPE(
            dimensions: ropeDims, traditional: false, base: args.ropeTheta, scale: 1.0)
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

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, numKVHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

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

private class Qwen3_5MLP: Module, UnaryLayer {
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

// MARK: - GatedDeltaNet (Linear Attention) - Qwen3.5 variant with separate projections

private class Qwen3_5GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    // Qwen3.5 uses 4 separate projections (unlike Qwen3Next's combined 2)
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen3_5RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen3_5MoETextConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.numVHeads = config.linearNumValueHeads
        self.numKHeads = config.linearNumKeyHeads
        self.headKDim = config.linearKeyHeadDim
        self.headVDim = config.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = config.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: false
        )

        // Separate projections: qkv, z (gate), b (beta), a (alpha)
        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLX.log(MLXArray.ones([numVHeads]) * 8.0)

        _norm.wrappedValue = Qwen3_5RMSNormGated(
            hiddenSize: headVDim, eps: config.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray? = nil, cache: ArraysCache? = nil
    ) -> MLXArray {
        let (B, S, _) = (inputs.dim(0), inputs.dim(1), inputs.dim(2))

        let qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        // Conv state management
        let convState: MLXArray
        if let cache, let cached = cache[0] {
            convState = cached
        } else {
            convState = MLXArray.zeros(
                [B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        var convInput: MLXArray
        if let mask {
            convInput = which(mask[.ellipsis, .newAxis], qkv, 0)
        } else {
            convInput = qkv
        }
        convInput = concatenated([convState, convInput], axis: 1)

        // Update conv cache
        if let cache {
            let nKeep = convKernelSize - 1
            cache[0] = convInput[0..., (convInput.dim(1) - nKeep)...]
        }

        // Apply conv + silu
        let convOut = silu(conv1d(convInput))

        // Split conv output into q, k, v
        let convSplits = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = convSplits[0].reshaped(B, S, numKHeads, headKDim)
        var k = convSplits[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplits[2].reshaped(B, S, numVHeads, headVDim)

        // Q/K normalization (no learned weights)
        let invScale = pow(Float(headKDim), -0.5)
        q = (invScale * invScale) * qwen3_5ManualRmsNorm(q, eps: 1e-6)
        k = invScale * qwen3_5ManualRmsNorm(k, eps: 1e-6)

        // Recurrent state
        let state: MLXArray? = cache?[1]

        let (out, newState) = gatedDeltaUpdate(
            q: q, k: k, v: v,
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

/// Manual RMS norm without learned weights
private func qwen3_5ManualRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(variance + eps)
}

// MARK: - Sparse MoE with Shared Expert

private class Qwen3_5SparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3_5MLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen3_5MoETextConfiguration) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: numExperts
        )
        _sharedExpert.wrappedValue = Qwen3_5MLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
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

private class Qwen3_5DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3_5GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3_5Attention?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: Qwen3_5MoETextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen3_5GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen3_5Attention(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        // Qwen3.5: every layer gets MoE when num_experts > 0 (no decoder_sparse_step)
        if args.numExperts > 0 {
            self.mlp = Qwen3_5SparseMoeBlock(args)
        } else {
            self.mlp = Qwen3_5MLP(
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

class Qwen3_5TextModelInner: Module {
    let args: Qwen3_5MoETextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3_5DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen3_5MoETextConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            Qwen3_5DecoderLayer(args, layerIdx: i)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let cache: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)

        let faMask = createAttentionMask(h: h, cache: cache[faIdx])
        let ssmMask = createSSMMask(h: h, cache: cache[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, faMask: faMask, ssmMask: ssmMask, cache: cache[i])
        }

        return norm(h)
    }
}

// MARK: - Text Model (language_model level)

class Qwen3_5TextModel: Module {
    let args: Qwen3_5MoETextConfiguration

    @ModuleInfo(key: "model") var model: Qwen3_5TextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: Qwen3_5MoETextConfiguration) {
        self.args = args
        _model.wrappedValue = Qwen3_5TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }
}

// MARK: - Top-Level Model

public class Qwen3_5MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen3_5TextModel
    let configuration: Qwen3_5MoEConfiguration

    public init(_ args: Qwen3_5MoEConfiguration) {
        self.configuration = args
        let textArgs = args.textConfig
        self.vocabularySize = textArgs.vocabularySize
        // Linear layers use ArraysCache (0 kvHeads), attention layers use KVCacheSimple
        self.kvHeads = (0 ..< textArgs.hiddenLayers).map { i in
            let isLinear = (i + 1) % textArgs.fullAttentionInterval != 0
            return isLinear ? 0 : textArgs.kvHeads
        }
        _languageModel.wrappedValue = Qwen3_5TextModel(textArgs)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let textArgs = configuration.textConfig
        return (0 ..< textArgs.hiddenLayers).map { i in
            let isLinear = (i + 1) % textArgs.fullAttentionInterval != 0
            if isLinear {
                return MambaCache() as any KVCache
            } else {
                return KVCacheSimple() as any KVCache
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        let textArgs = configuration.textConfig

        // 1. Remap weight prefixes: handle models stored with different prefix conventions
        var remapped: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            var newKey = key
            if key.hasPrefix("model.language_model.") {
                // model.language_model.X -> language_model.model.X
                newKey = key.replacingOccurrences(
                    of: "model.language_model.", with: "language_model.model.")
            } else if key.hasPrefix("model.visual") || key.hasPrefix("vision_tower") {
                // Skip vision encoder weights
                continue
            } else if !key.hasPrefix("language_model.") {
                newKey = "language_model." + key
            }
            remapped[newKey] = value
        }
        sanitizedWeights = remapped

        // 2. Filter out mtp.* keys
        sanitizedWeights = sanitizedWeights.filter {
            !$0.key.contains("mtp.")
        }

        // 3. Handle tied embeddings
        if textArgs.tieWordEmbeddings {
            sanitizedWeights["language_model.lm_head.weight"] = nil
        }

        // 4. Split fused gate_up_proj if present (some checkpoints fuse gate+up)
        for l in 0 ..< textArgs.hiddenLayers {
            let prefix = "language_model.model.layers.\(l).mlp"
            let gateUpKey = "\(prefix).experts.gate_up_proj"
            if let gateUp = sanitizedWeights.removeValue(forKey: gateUpKey) {
                let mid = gateUp.dim(-2) / 2
                sanitizedWeights["\(prefix).switch_mlp.gate_proj.weight"] =
                    gateUp[.ellipsis, ..<mid, 0...]
                sanitizedWeights["\(prefix).switch_mlp.up_proj.weight"] =
                    gateUp[.ellipsis, mid..., 0...]
                if let downProj = sanitizedWeights.removeValue(
                    forKey: "\(prefix).experts.down_proj")
                {
                    sanitizedWeights["\(prefix).switch_mlp.down_proj.weight"] = downProj
                }
            }

            // Stack individual expert weights if present
            if sanitizedWeights["\(prefix).experts.0.up_proj.weight"] != nil {
                for n in ["up_proj", "down_proj", "gate_proj"] {
                    if sanitizedWeights["\(prefix).experts.0.\(n).weight"] != nil {
                        let toJoin = (0 ..< textArgs.numExperts).map { e in
                            sanitizedWeights.removeValue(
                                forKey: "\(prefix).experts.\(e).\(n).weight")!
                        }
                        sanitizedWeights["\(prefix).switch_mlp.\(n).weight"] =
                            MLX.stacked(toJoin)
                    }
                }
            }
        }

        // 5. Check if we need to apply conv1d transpose and norm +1.0 shift
        // Quantized models from mlx-community already have these applied.
        // Detect by checking if any conv1d weight has shape [..., kernelSize, 1] (already transposed)
        // vs [..., 1, kernelSize] (needs transpose).
        let hasMTPWeights = sanitizedWeights.keys.contains { $0.contains("mtp.") }
        let hasUnsanitizedConv1d = sanitizedWeights.contains { key, value in
            key.contains("conv1d.weight") && value.shape.last != 1
        }
        let shouldShiftNorms = hasMTPWeights || hasUnsanitizedConv1d

        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (k, v) in sanitizedWeights {
            // Conv1d weight transpose if needed
            if k.contains("conv1d.weight"), v.ndim == 3, v.shape.last != 1 {
                sanitizedWeights[k] = v.swappedAxes(1, 2)
            }
            // Norm weight +1.0 shift (only for unquantized/mtp models)
            if shouldShiftNorms,
                normSuffixes.contains(where: { k.hasSuffix($0) }),
                v.ndim == 1
            {
                sanitizedWeights[k] = v + 1.0
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension Qwen3_5MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
