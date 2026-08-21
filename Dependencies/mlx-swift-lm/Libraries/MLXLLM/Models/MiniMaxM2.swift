//
//  MiniMaxM2.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/minimax.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct MiniMaxM2Configuration: Codable, Sendable {
    var modelType: String = "minimax_m2"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var numLocalExperts: Int
    var numExpertsPerTok: Int
    var rmsNormEps: Float
    var ropeTheta: Float = 5_000_000
    var rotaryDim: Int
    var vocabularySize: Int
    var tieWordEmbeddings: Bool = false
    var scoringFunc: String = "sigmoid"
    var useQkNorm: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case scoringFunc = "scoring_func"
        case useQkNorm = "use_qk_norm"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m2"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)

        let hidden = self.hiddenSize
        let heads = self.attentionHeads
        self.headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / heads)

        self.numLocalExperts = try container.decode(Int.self, forKey: .numLocalExperts)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 5_000_000
        self.rotaryDim = try container.decode(Int.self, forKey: .rotaryDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.useQkNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
    }
}

// MARK: - Attention

class MiniMaxM2Attention: Module {
    let args: MiniMaxM2Configuration
    let scale: Float
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let useQkNorm: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ args: MiniMaxM2Configuration) {
        self.args = args
        self.numHeads = args.attentionHeads
        self.numKVHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.useQkNorm = args.useQkNorm

        _qProj.wrappedValue = Linear(args.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, args.hiddenSize, bias: false)

        // QK norm on full concatenated dimension (not per-head like Qwen3MoE)
        if useQkNorm {
            _qNorm.wrappedValue = RMSNorm(
                dimensions: headDim * numHeads, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(
                dimensions: headDim * numKVHeads, eps: args.rmsNormEps)
        }

        // Partial rotary: rotaryDim (64) < headDim (128)
        self.rope = RoPE(
            dimensions: args.rotaryDim, traditional: false, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Apply QK norm on flat [B, L, heads*headDim] before reshape
        if useQkNorm {
            queries = qNorm!(queries)
            keys = kNorm!(keys)
        }

        // Reshape to multi-head and transpose
        queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE (partial rotary)
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Sparse MoE Block

class MiniMaxM2SparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eCorrectionBias: MLXArray

    init(_ args: MiniMaxM2Configuration) {
        self.numExperts = args.numLocalExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.intermediateSize,
            numExperts: numExperts)
        _eCorrectionBias.wrappedValue = MLXArray.zeros([numExperts])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Sigmoid routing (not softmax)
        let gates = gate(x.asType(.float32))
        let scores = sigmoid(gates)
        let origScores = scores

        // Bias added for expert selection only
        let biasedScores = scores + eCorrectionBias

        let k = topK
        let inds = MLX.argPartition(-biasedScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]

        // Use original unbiased scores for combination weights
        var selectedScores = MLX.takeAlong(origScores, inds, axis: -1)
        selectedScores = selectedScores / (MLX.sum(selectedScores, axis: -1, keepDims: true) + 1e-20)
        selectedScores = selectedScores.asType(x.dtype)

        let y = switchMLP(x, inds)
        return (y * selectedScores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - Decoder Layer

class MiniMaxM2DecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: MiniMaxM2Attention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MiniMaxM2SparseMoeBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: MiniMaxM2Configuration) {
        _selfAttn.wrappedValue = MiniMaxM2Attention(args)
        _blockSparseMoe.wrappedValue = MiniMaxM2SparseMoeBlock(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let out = r + blockSparseMoe(postAttentionLayerNorm(r))
        return out
    }
}

// MARK: - Inner Model

public class MiniMaxM2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [MiniMaxM2DecoderLayer]
    let norm: RMSNorm

    init(_ args: MiniMaxM2Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in
            MiniMaxM2DecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Top-Level Model

public class MiniMaxM2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniMaxM2ModelInner
    let configuration: MiniMaxM2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: MiniMaxM2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = MiniMaxM2ModelInner(args)

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

    /// Predicate for casting: e_score_correction_bias should stay in float32
    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias")
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights

        // 1. Filter out mtp.* and FP8 scale keys
        sanitizedWeights = sanitizedWeights.filter {
            !$0.key.hasPrefix("mtp.") && !$0.key.contains("weight_scale_inv")
        }

        // Handle tied embeddings
        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        // 2. MoE expert weight stacking (for unquantized models)
        // Quantized models already have stacked weights — early return
        guard sanitizedWeights[
            "model.layers.0.block_sparse_moe.experts.0.w1.weight"] != nil
        else {
            return sanitizedWeights
        }

        let mapping = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]
        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l).block_sparse_moe"
            for (origName, newName) in mapping {
                if sanitizedWeights["\(prefix).experts.0.\(origName).weight"] != nil {
                    let toJoin = (0 ..< configuration.numLocalExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).experts.\(e).\(origName).weight")!
                    }
                    sanitizedWeights["\(prefix).switch_mlp.\(newName).weight"] = MLX.stacked(
                        toJoin)
                }
            }
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension MiniMaxM2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
