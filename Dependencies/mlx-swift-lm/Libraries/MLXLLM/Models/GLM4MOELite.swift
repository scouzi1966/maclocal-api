//
//  GLM4MoeLite.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe_lite.py
//  MLA (Multi-head Latent Attention) with MoE architecture for GLM-4.7-Flash

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MultiLinear

/// Per-head linear layer with weight shape [numHeads, outputDims, inputDims].
/// Supports both regular and quantized weights.
class GLM4MoeLiteMultiLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray?
    @ParameterInfo(key: "biases") var biases: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads

        // Use scalar placeholders for all parameters. The actual tensors come from
        // the checkpoint via update(parameters:). Quantized models store packed
        // weights with different shapes than logical dims.
        self._weight.wrappedValue = MLXArray(Float(0))
        self._scales.wrappedValue = MLXArray(Float(0))
        self._biases.wrappedValue = MLXArray(Float(0))

        super.init()
    }

    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        if let scales, let biases, scales.size > 1 {
            // Quantized path
            // Quantization is always along the last weight dim (= inputDims)
            let dims = inputDims
            let bits = (weight.dim(-1) * 32) / dims
            let groupSize = dims / scales.dim(-1)
            return quantizedMatmul(
                x, weight, scales: scales, biases: biases,
                transpose: transpose, groupSize: groupSize, bits: bits)
        } else {
            // Regular path
            if transpose {
                return matmul(x, weight.swappedAxes(-1, -2))
            } else {
                return matmul(x, weight)
            }
        }
    }
}

// MARK: - Attention (MLA)

class GLM4MoeLiteAttention: Module {
    let config: GLM4MoeLiteConfiguration
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: GLM4MoeLiteMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: GLM4MoeLiteMultiLinear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: GLM4MoeLiteConfiguration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank = config.qLoraRank {
            self._qAProj.wrappedValue = Linear(
                config.hiddenSize, qLoraRank, bias: config.attentionBias)
            self._qALayerNorm.wrappedValue = RMSNorm(
                dimensions: qLoraRank, eps: config.rmsNormEps)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(
                config.hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: kvLoraRank, eps: config.rmsNormEps)

        self._embedQ.wrappedValue = GLM4MoeLiteMultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        self._unembedOut.wrappedValue = GLM4MoeLiteMultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)

        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        // Handle rope_scaling mscale for scale adjustment
        if let ropeScaling = config.ropeScaling,
           case .float(let mscaleAllDim) = ropeScaling["mscale_all_dim"],
           mscaleAllDim != 0,
           case .float(let scalingFactor) = ropeScaling["factor"],
           scalingFactor > 1
        {
            let s = 0.1 * mscaleAllDim * log(scalingFactor) + 1.0
            self.scale = self.scale * s * s
        }

        self.rope = RoPE(
            dimensions: qkRopeHeadDim,
            traditional: true,
            base: config.ropeTheta)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Compute Q
        var q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = qSplit[0]  // [B, numHeads, L, qkNopeHeadDim]
        var qPe = qSplit[1]    // [B, numHeads, L, qkRopeHeadDim]

        // Compress KV
        let compressedKvFull = kvAProjWithMqa(x)
        let kvSplit = split(compressedKvFull, indices: [kvLoraRank], axis: -1)
        let compressedKv = kvSplit[0]  // [B, L, kvLoraRank]
        var kPe = kvSplit[1]           // [B, L, qkRopeHeadDim]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kvLatent = kvALayerNorm(compressedKv)

        // Apply RoPE
        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)

        // Expand kv_latent to [B, 1, L, kvLoraRank]
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        // Update cache: stores kv_latent as "keys" and k_pe as "values"
        if let cache {
            let (updatedKvLatent, updatedKPe) = cache.update(keys: kvLatent, values: kPe)
            kvLatent = updatedKvLatent
            kPe = updatedKPe
        }

        // Compute positional attention scores
        // pe_scores: [B, numHeads, L, L_total]
        var peScores = matmul(qPe * scale, kPe.swappedAxes(-1, -2))
        if let mask {
            peScores = MLX.where(
                mask, peScores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: peScores.dtype))
        }

        // MLA attention with two-path optimization
        let output: MLXArray
        if L == 1 {
            // Decode path: work in latent space (cheaper)
            // Transform q_nope to latent space
            qNope = embedQ(qNope)  // [B, numHeads, 1, qkNopeHeadDim] → [B, numHeads, 1, kvLoraRank]
            // Attention in latent space
            let attnOut = mlaAttention(
                queries: qNope, keys: kvLatent, values: kvLatent,
                scale: scale, peMask: peScores)
            // Project output back from latent space
            output = unembedOut(attnOut)  // [B, numHeads, 1, kvLoraRank] → [B, numHeads, 1, vHeadDim]
        } else {
            // Prefill path: project K and V from latent space
            let k = embedQ(kvLatent, transpose: false)    // [B, 1, L, kvLoraRank] → [B, numHeads, L, qkNopeHeadDim]
            let v = unembedOut(kvLatent)                   // [B, 1, L, kvLoraRank] → [B, numHeads, L, vHeadDim]
            // Standard attention with PE scores as additive mask
            output = mlaAttention(
                queries: qNope, keys: k, values: v,
                scale: scale, peMask: peScores)
        }

        let reshaped = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(reshaped)
    }

    /// Manual scaled dot-product attention with additive PE mask
    private func mlaAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, peMask: MLXArray
    ) -> MLXArray {
        // scores = (Q @ K^T) * scale + pe_scores
        var scores = matmul(queries, keys.swappedAxes(-1, -2)) * scale
        scores = scores + peMask
        let weights = softmax(scores, axis: -1)
        return matmul(weights, values)
    }
}

// MARK: - MLP

class GLM4MoeLiteMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoeLiteConfiguration, intermediateSize: Int? = nil) {
        let intermediate = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Gate

class GLM4MoeLiteMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoeLiteConfiguration) {
        self.topK = config.numExpertsPerTok
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb

        self._weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts!, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts!])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = x.matmul(weight.T)
        let origScores = sigmoid(gates.asType(.float32))
        var scores = origScores + eScoreCorrectionBias

        if nGroup > 1 {
            scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
            let groupScores = sorted(scores, axis: -1)[.ellipsis, ..<2]
                .sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            var groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            let numExperts = weight.dim(0)
            let batchShape = scores.shape.dropLast(2)
            var broadcastShape = Array(batchShape)
            broadcastShape.append(k)
            broadcastShape.append(numExperts / nGroup)
            groupIdx = broadcast(groupIdx, to: broadcastShape)
            scores = putAlong(scores, groupIdx, values: MLXArray(0.0), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }

        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var finalScores = takeAlong(origScores, inds, axis: -1)

        if topK > 1, normTopkProb {
            let denominator = finalScores.sum(axis: -1, keepDims: true) + 1e-20
            finalScores = finalScores / denominator
        }
        finalScores = finalScores * routedScalingFactor

        return (inds, finalScores)
    }
}

// MARK: - MoE

class GLM4MoeLiteMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: GLM4MoeLiteMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoeLiteMLP?

    init(_ config: GLM4MoeLiteConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok

        self._gate.wrappedValue = GLM4MoeLiteMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts!)

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            self._sharedExperts.wrappedValue = GLM4MoeLiteMLP(
                config, intermediateSize: intermediateSize)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder Layer

class GLM4MoeLiteDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoeLiteAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM4MoeLiteConfiguration, layerIdx: Int) {
        self._attention.wrappedValue = GLM4MoeLiteAttention(config)

        let useMoe = config.nRoutedExperts != nil
            && layerIdx >= config.firstKDenseReplace
            && layerIdx % config.moeLayerFreq == 0
        self.mlp = useMoe ? GLM4MoeLiteMoE(config) : GLM4MoeLiteMLP(config)

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

// MARK: - Model Inner

class GLM4MoeLiteModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLM4MoeLiteDecoderLayer]
    let norm: RMSNorm
    let numHiddenLayers: Int

    init(_ config: GLM4MoeLiteConfiguration) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers).map {
            GLM4MoeLiteDecoderLayer(config, layerIdx: $0)
        }
        self.numHiddenLayers = config.numHiddenLayers
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Create attention mask (boolean, True = attend, False = mask out)
        let mask: MLXArray? = createBoolMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    /// Create a boolean causal attention mask compatible with MLA's manual attention.
    /// Returns True for positions to attend, False for positions to mask.
    private func createBoolMask(h: MLXArray, cache: KVCache?) -> MLXArray? {
        let T = h.dim(1)
        if T == 1 {
            return nil  // Single token decode, no mask needed
        }
        let offset = cache?.offset ?? 0
        // Create causal mask: positions can attend to earlier positions
        let rowIdx = MLXArray(0 ..< T).reshaped(T, 1) + offset
        let colIdx = MLXArray(0 ..< (T + offset)).reshaped(1, T + offset)
        return rowIdx .>= colIdx
    }
}

// MARK: - Top-level Model

public class GLM4MoeLiteModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: GLM4MoeLiteModelInner
    let configuration: GLM4MoeLiteConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] {
        model.layers
    }

    public init(_ config: GLM4MoeLiteConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        // KV cache uses 1 head (stores compressed latent)
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in 1 }

        self._model.wrappedValue = GLM4MoeLiteModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        // Filter out MTP (multi-token prediction) layers
        let mptPrefix = "model.layers.\(configuration.numHiddenLayers)"
        sanitized = sanitized.filter { !$0.key.hasPrefix(mptPrefix) }

        // Stack MoE experts
        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if sanitized[firstKey] != nil, let nExperts = configuration.nRoutedExperts {
                        let toJoin = (0 ..< nExperts).compactMap {
                            sanitized.removeValue(
                                forKey: "\(prefix).mlp.experts.\($0).\(projName).\(key)")
                        }
                        if !toJoin.isEmpty {
                            sanitized["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = MLX.stacked(
                                toJoin)
                        }
                    }
                }
            }

            // Handle kv_b_proj → embed_q + unembed_out splitting
            let attnPrefix = "model.layers.\(l).self_attn"
            if let kvBWeight = sanitized["\(attnPrefix).kv_b_proj.weight"] {
                let isQuantized = sanitized["\(attnPrefix).kv_b_proj.scales"] != nil
                let headDim = configuration.qkNopeHeadDim + configuration.vHeadDim
                let numHeads = configuration.numAttentionHeads

                var v: MLXArray
                if isQuantized {
                    let kvBScales = sanitized.removeValue(
                        forKey: "\(attnPrefix).kv_b_proj.scales")!
                    let kvBBiases = sanitized.removeValue(
                        forKey: "\(attnPrefix).kv_b_proj.biases")!
                    let dims = configuration.kvLoraRank
                    let bits = (kvBWeight.dim(-1) * 32) / dims
                    let groupSize = dims / kvBScales.dim(-1)
                    v = dequantized(
                        kvBWeight, scales: kvBScales, biases: kvBBiases,
                        groupSize: groupSize, bits: bits)
                } else {
                    v = kvBWeight
                }
                sanitized.removeValue(forKey: "\(attnPrefix).kv_b_proj.weight")

                // Reshape to [numHeads, headDim, kvLoraRank] and split
                v = v.reshaped(numHeads, headDim, -1)
                let wk = contiguous(
                    v[0..., ..<configuration.qkNopeHeadDim, 0...]
                        .swappedAxes(-1, -2))
                let wv = contiguous(
                    v[0..., configuration.qkNopeHeadDim..., 0...])

                if isQuantized {
                    let qBits = 4  // Standard 4-bit quantization
                    let qGroupSize = 64  // Standard group size
                    let (qWk, sWk, bWk) = MLX.quantized(wk, groupSize: qGroupSize, bits: qBits)
                    let (qWv, sWv, bWv) = MLX.quantized(wv, groupSize: qGroupSize, bits: qBits)
                    sanitized["\(attnPrefix).embed_q.weight"] = qWk
                    sanitized["\(attnPrefix).embed_q.scales"] = sWk
                    sanitized["\(attnPrefix).embed_q.biases"] = bWk
                    sanitized["\(attnPrefix).unembed_out.weight"] = qWv
                    sanitized["\(attnPrefix).unembed_out.scales"] = sWv
                    sanitized["\(attnPrefix).unembed_out.biases"] = bWv
                } else {
                    sanitized["\(attnPrefix).embed_q.weight"] = wk
                    sanitized["\(attnPrefix).unembed_out.weight"] = wv
                }
            }
        }

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized
    }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias")
        }
    }
}

// MARK: - Configuration

public struct GLM4MoeLiteConfiguration: Codable, Sendable {
    var modelType: String = "glm4_moe_lite"
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var numExpertsPerTok: Int
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool
    var partialRotaryFactor: Float
    var tieWordEmbeddings: Bool
    var topkMethod: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case topkMethod = "topk_method"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm4_moe_lite"
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 202752
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
    }
}
