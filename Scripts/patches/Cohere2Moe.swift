import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

//  Cohere2MoE (model_type: "cohere2_moe", arch: "Cohere2MoeForCausalLM")
//
//  Cohere's North family of MoE models (e.g. CohereLabs/North-Mini-Code-1.0, 30B-A3B).
//  Ported from mlx-vlm `mlx_vlm/models/cohere2_moe/language.py`.
//
//  Architecture = Cohere2 base + DeepSeek-style MoE:
//   - Parallel block: a single `input_layernorm`, with attention and the FF/MoE branch
//     both reading the same normed hidden, summed with the residual (attn + ff + x).
//   - Interleaved attention: full-attention every Nth layer, sliding-window otherwise
//     (driven by `layer_types`). Full-attention layers use NoPE (no RoPE); sliding layers
//     use traditional RoPE. The dense "prefix" layers (first_k_dense_replace) force RoPE
//     when prefix_dense_sliding_window_pattern == 1.
//   - MoE: sigmoid (or softmax) router + SwitchGLU experts + optional shared experts,
//     with `first_k_dense_replace` leading dense MLP layers (DeepSeek-V3 pattern).
//   - Tied input/output embeddings, logits scaled by `logit_scale`.
//
//  Note: this implementation targets the RMSNorm variant (rms_norm_eps set, as in the
//  North checkpoints). A LayerNorm variant (rms_norm_eps == null) is not yet handled.

// MARK: - Configuration

public struct Cohere2MoeConfiguration: Codable, Sendable {
    var modelType: String = "cohere2_moe"
    var hiddenSize: Int = 2048
    var headDim: Int = 128
    var numHiddenLayers: Int = 49
    var intermediateSize: Int = 768
    var numAttentionHeads: Int = 32
    var numKeyValueHeads: Int = 4
    var ropeTheta: Float = 50000.0
    var vocabSize: Int = 262144
    var layerNormEps: Float = 1e-5
    var rmsNormEps: Float? = nil
    var logitScale: Float = 1.0
    var attentionBias: Bool = false
    var layerNormBias: Bool = false
    var slidingWindow: Int = 4096
    var slidingWindowPattern: Int = 4
    var numExperts: Int = 128
    var numExpertsPerTok: Int = 8
    var normTopkProb: Bool = false
    var numSharedExperts: Int? = nil
    var moeGateAct: String = "sigmoid"
    var expertSelectionFn: String? = nil
    var sharedExpertCombinationStrategy: String = "average"
    var firstKDenseReplace: Int = 0
    var prefixDenseIntermediateSize: Int? = nil
    var prefixDenseSlidingWindowPattern: Int = 1
    var layerTypes: [String]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case rmsNormEps = "rms_norm_eps"
        case logitScale = "logit_scale"
        case attentionBias = "attention_bias"
        case layerNormBias = "layer_norm_bias"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case numSharedExperts = "num_shared_experts"
        case moeGateAct = "moe_gate_act"
        case expertSelectionFn = "expert_selection_fn"
        case sharedExpertCombinationStrategy = "shared_expert_combination_strategy"
        case firstKDenseReplace = "first_k_dense_replace"
        case prefixDenseIntermediateSize = "prefix_dense_intermediate_size"
        case prefixDenseSlidingWindowPattern = "prefix_dense_sliding_window_pattern"
        case layerTypes = "layer_types"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func d<T: Decodable>(_ k: CodingKeys, _ def: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? def
        }
        modelType = try d(.modelType, "cohere2_moe")
        hiddenSize = try d(.hiddenSize, 2048)
        headDim = try d(.headDim, 128)
        numHiddenLayers = try d(.numHiddenLayers, 49)
        intermediateSize = try d(.intermediateSize, 768)
        numAttentionHeads = try d(.numAttentionHeads, 32)
        numKeyValueHeads = try d(.numKeyValueHeads, 4)
        ropeTheta = try d(.ropeTheta, 50000.0)
        vocabSize = try d(.vocabSize, 262144)
        layerNormEps = try d(.layerNormEps, 1e-5)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
        logitScale = try d(.logitScale, 1.0)
        attentionBias = try d(.attentionBias, false)
        layerNormBias = try d(.layerNormBias, false)
        slidingWindow = try d(.slidingWindow, 4096)
        slidingWindowPattern = try d(.slidingWindowPattern, 4)
        numExperts = try d(.numExperts, 128)
        numExpertsPerTok = try d(.numExpertsPerTok, 8)
        normTopkProb = try d(.normTopkProb, false)
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts)
        moeGateAct = try d(.moeGateAct, "sigmoid")
        expertSelectionFn = try c.decodeIfPresent(String.self, forKey: .expertSelectionFn)
        sharedExpertCombinationStrategy = try d(.sharedExpertCombinationStrategy, "average")
        firstKDenseReplace = try d(.firstKDenseReplace, 0)
        prefixDenseIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .prefixDenseIntermediateSize)
        prefixDenseSlidingWindowPattern = try d(.prefixDenseSlidingWindowPattern, 1)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
    }

    // Mirror the Python `__post_init__` derived values.
    var effectiveSharedExperts: Int { numSharedExperts ?? 0 }
    var effectiveGateAct: String { expertSelectionFn ?? moeGateAct }
    var effectivePrefixDenseIntermediateSize: Int { prefixDenseIntermediateSize ?? intermediateSize }

    func isPrefixDenseLayer(_ i: Int) -> Bool { i < firstKDenseReplace }

    func isSlidingLayer(_ i: Int) -> Bool {
        if isPrefixDenseLayer(i) { return false }
        if let types = layerTypes { return types[i] == "sliding_attention" }
        return (i + 1) % slidingWindowPattern != 0
    }

    func forceRope(_ i: Int) -> Bool {
        isPrefixDenseLayer(i) && prefixDenseSlidingWindowPattern == 1
    }

    func appliesRope(_ i: Int) -> Bool { isSlidingLayer(i) || forceRope(i) }

    func makeNorm() -> RMSNorm {
        RMSNorm(dimensions: hiddenSize, eps: rmsNormEps ?? layerNormEps)
    }
}

// MARK: - Attention

private class Cohere2MoeAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let usesRope: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: Cohere2MoeConfiguration, layerIdx: Int) {
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.usesRope = config.appliesRope(layerIdx)

        let dim = config.hiddenSize
        let bias = config.attentionBias
        self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: bias)
        self._kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: bias)
        self._vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: bias)
        self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: bias)

        self.rope = RoPE(dimensions: headDim, traditional: true, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // Full-attention layers use NoPE (Cohere2/Command-A pattern); only sliding
        // (and force-rope prefix-dense) layers apply RoPE.
        if usesRope {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Dense MLP (prefix-dense layers + shared experts)

private class Cohere2MoeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Sparse MoE block

private class Cohere2MoeSparseBlock: Module, UnaryLayer {
    let topK: Int
    let normTopkProb: Bool
    let useSigmoid: Bool
    let combination: String

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Cohere2MoeMLP?

    init(_ config: Cohere2MoeConfiguration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.useSigmoid = config.effectiveGateAct == "sigmoid"
        self.combination = config.sharedExpertCombinationStrategy

        self._gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.numExperts,
            activation: silu)

        if config.effectiveSharedExperts > 0 {
            let sharedIntermediate = config.intermediateSize * config.effectiveSharedExperts
            self._sharedExperts.wrappedValue = Cohere2MoeMLP(
                hiddenSize: config.hiddenSize, intermediateSize: sharedIntermediate)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x).asType(.float32)
        gates = useSigmoid ? sigmoid(gates) : softmax(gates, axis: -1)

        let inds = argPartition(-gates, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / maximum(scores.sum(axis: -1, keepDims: true), 1e-12)
        }

        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)

        if let shared = sharedExperts {
            y = combination == "average" ? (y + shared(x)) / 2 : y + shared(x)
        }
        return y
    }
}

// MARK: - Decoder layer (parallel block)

private class Cohere2MoeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Cohere2MoeAttention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm

    init(_ config: Cohere2MoeConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Cohere2MoeAttention(config, layerIdx: layerIdx)
        if config.isPrefixDenseLayer(layerIdx) {
            self.mlp = Cohere2MoeMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.effectivePrefixDenseIntermediateSize)
        } else {
            self.mlp = Cohere2MoeSparseBlock(config)
        }
        self._inputLayernorm.wrappedValue = config.makeNorm()
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = inputLayernorm(x)
        let attnH = selfAttn(h, mask: mask, cache: cache)
        let ffH = mlp(h)
        return attnH + ffH + x
    }
}

// MARK: - Inner model

private class Cohere2MoeModelInner: Module {
    let config: Cohere2MoeConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate var layers: [Cohere2MoeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Cohere2MoeConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            Cohere2MoeDecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = config.makeNorm()
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let c = cache?[i]
            let mask = createAttentionMask(
                h: h, cache: c, windowSize: config.isSlidingLayer(i) ? config.slidingWindow : nil)
            h = layer(h, mask: mask, cache: c)
        }
        return norm(h)
    }
}

// MARK: - Top-level model

public class Cohere2MoeModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let kvHeads: [Int]
    let config: Cohere2MoeConfiguration
    fileprivate let model: Cohere2MoeModelInner

    public init(_ config: Cohere2MoeConfiguration) {
        self.config = config
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self.model = Cohere2MoeModelInner(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        // Tied embeddings + logit scaling (Cohere).
        return model.embedTokens.asLinear(out) * config.logitScale
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        (0 ..< config.numHiddenLayers).map { i in
            config.isSlidingLayer(i)
                ? RotatingKVCache(maxSize: config.slidingWindow, keep: 0) as any KVCache
                : KVCacheSimple() as any KVCache
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights

        // Some conversions (e.g. mlx-community North-Mini-Code) nest all weights under a
        // `language_model.` prefix (VLM-style). Strip it so keys match this text model's
        // `model.…` module layout.
        if w.keys.contains(where: { $0.hasPrefix("language_model.") }) {
            var remapped: [String: MLXArray] = [:]
            for (k, v) in w {
                remapped[k.hasPrefix("language_model.") ? String(k.dropFirst("language_model.".count)) : k] = v
            }
            w = remapped
        }

        // Stack per-expert MoE weights into the SwitchGLU layout (no-op if the checkpoint
        // already ships them pre-stacked as `…mlp.switch_mlp.*`).
        for l in 0 ..< config.numHiddenLayers where !config.isPrefixDenseLayer(l) {
            let prefix = "model.layers.\(l).mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let first = "\(prefix).experts.0.\(proj).\(suffix)"
                    guard w[first] != nil else { continue }
                    let joined = (0 ..< config.numExperts).compactMap {
                        w.removeValue(forKey: "\(prefix).experts.\($0).\(proj).\(suffix)")
                    }
                    w["\(prefix).switch_mlp.\(proj).\(suffix)"] = stacked(joined)
                }
            }
        }

        // Drop weights this implementation does not use.
        for key in Array(w.keys) {
            if key.contains("rotary_emb.inv_freq") {
                w[key] = nil
            } else if key.hasSuffix(".bias") {
                if key.contains(".mlp.") {
                    w[key] = nil
                } else if key.contains(".self_attn.") && !config.attentionBias {
                    w[key] = nil
                } else if key.lowercased().contains("layernorm") && !config.layerNormBias {
                    w[key] = nil
                }
            }
        }

        // Tied embeddings: no separate lm_head.
        w["lm_head.weight"] = nil
        w["model.lm_head.weight"] = nil
        return w
    }

    public var loraLayers: [Module] { model.layers }
}
