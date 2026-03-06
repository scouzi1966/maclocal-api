//
//  Qwen3.swift
//  LLM
//
//  Created by John Mai on 2025/4/28.
//
//  Patched: fused QKV attention, fused gate+up MLP with fusedSiluMul kernel,
//  MLXFast.rmsNorm for Q/K norms.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py

class Qwen3Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    // Fused QKV weights (lazily created)
    private var fusedQKVWeight: MLXArray?
    private var fusedQKVScales: MLXArray?
    private var fusedQKVBiases: MLXArray?
    private var fusedQKVGroupSize: Int = 0
    private var fusedQKVBits: Int = 0
    private var fusedQKVMode: QuantizationMode = .affine
    private var fusedQKVAttempted = false
    private var fusedQKVSplitIndices: [Int] = []

    // Pre-computed Q/K norm weights for MLXFast.rmsNorm
    private var qNormWeight: MLXArray?
    private var kNormWeight: MLXArray?
    private var normsPrepared = false

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

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
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    /// Lazily fuse Q+K+V projections into a single quantized matmul
    private func tryFuseQKV() {
        guard !fusedQKVAttempted else { return }
        fusedQKVAttempted = true

        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim

        guard let qQ = wq as? QuantizedLinear,
              let qK = wk as? QuantizedLinear,
              let qV = wv as? QuantizedLinear,
              qQ.groupSize == qK.groupSize,
              qQ.groupSize == qV.groupSize,
              qQ.bits == qK.bits,
              qQ.bits == qV.bits,
              qQ.mode == qK.mode,
              qQ.mode == qV.mode
        else { return }

        fusedQKVSplitIndices = [heads * headDim, heads * headDim + kvHeads * headDim]
        fusedQKVWeight = concatenated([qQ.weight, qK.weight, qV.weight], axis: 0)
        fusedQKVScales = concatenated([qQ.scales, qK.scales, qV.scales], axis: 0)
        if let b1 = qQ.biases, let b2 = qK.biases, let b3 = qV.biases {
            fusedQKVBiases = concatenated([b1, b2, b3], axis: 0)
        }
        fusedQKVGroupSize = qQ.groupSize
        fusedQKVBits = qQ.bits
        fusedQKVMode = qQ.mode

        if let fw = fusedQKVWeight, let fs = fusedQKVScales {
            var toMaterialize: [MLXArray] = [fw, fs]
            if let fb = fusedQKVBiases { toMaterialize.append(fb) }
            MLX.eval(toMaterialize)
        }
    }

    /// Lazily prepare Q/K norm weights for MLXFast.rmsNorm
    private func prepareNormWeights(_ dtype: DType) {
        guard !normsPrepared else { return }
        normsPrepared = true

        let qw = qNorm.weight.asType(dtype)
        let kw = kNorm.weight.asType(dtype)
        MLX.eval([qw, kw])
        qNormWeight = qw
        kNormWeight = kw
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        tryFuseQKV()
        prepareNormWeights(x.dtype)

        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray

        if let fWeight = fusedQKVWeight, let fScales = fusedQKVScales {
            // Fused path: single quantized matmul for Q+K+V
            let combined = MLX.quantizedMatmul(
                x, fWeight, scales: fScales, biases: fusedQKVBiases,
                transpose: true, groupSize: fusedQKVGroupSize,
                bits: fusedQKVBits, mode: fusedQKVMode)
            let parts = MLX.split(combined, indices: fusedQKVSplitIndices, axis: -1)
            queries = parts[0]
            keys = parts[1]
            values = parts[2]
        } else {
            queries = wq(x)
            keys = wk(x)
            values = wv(x)
        }

        // prepare the queries, keys and values for the attention computation
        // Use MLXFast.rmsNorm (1 C++ kernel) instead of RMSNorm module (4 graph ops)
        queries = MLXFast.rmsNorm(
            queries.reshaped(B, L, args.attentionHeads, -1),
            weight: qNormWeight!, eps: args.rmsNormEps
        ).transposed(0, 2, 1, 3)
        keys = MLXFast.rmsNorm(
            keys.reshaped(B, L, args.kvHeads, -1),
            weight: kNormWeight!, eps: args.rmsNormEps
        ).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE positioning
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Use the automatic attention router that handles both quantized and regular caches
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

        return wo(output)
    }
}

class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    let hiddenDimensions: Int

    // Fused gate+up quantized weights (lazily created)
    private var fusedWeight: MLXArray?
    private var fusedScales: MLXArray?
    private var fusedBiases: MLXArray?
    private var fusedGroupSize: Int = 0
    private var fusedBits: Int = 0
    private var fusedMode: QuantizationMode = .affine
    private var fusionAttempted = false

    public init(dimensions: Int, hiddenDimensions: Int) {
        self.hiddenDimensions = hiddenDimensions
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    /// Lazily fuse gate+up weights for quantized models.
    private func tryFuseGateUp() {
        guard !fusionAttempted else { return }
        fusionAttempted = true

        guard let qGate = gate as? QuantizedLinear,
              let qUp = up as? QuantizedLinear,
              qGate.groupSize == qUp.groupSize,
              qGate.bits == qUp.bits,
              qGate.mode == qUp.mode
        else { return }

        // Concatenate along output dimension (axis 0): [N, K_packed] -> [2N, K_packed]
        fusedWeight = concatenated([qGate.weight, qUp.weight], axis: 0)
        fusedScales = concatenated([qGate.scales, qUp.scales], axis: 0)
        if let gBiases = qGate.biases, let uBiases = qUp.biases {
            fusedBiases = concatenated([gBiases, uBiases], axis: 0)
        }
        fusedGroupSize = qGate.groupSize
        fusedBits = qGate.bits
        fusedMode = qGate.mode

        // Force evaluate so the concat is materialized once
        if let fw = fusedWeight, let fs = fusedScales {
            var toMaterialize: [MLXArray] = [fw, fs]
            if let fb = fusedBiases { toMaterialize.append(fb) }
            MLX.eval(toMaterialize)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        tryFuseGateUp()

        if let fWeight = fusedWeight, let fScales = fusedScales {
            // Fused path: single quantizedMatmul for gate+up, then fusedSiluMul kernel
            let gateUp = MLX.quantizedMatmul(
                x,
                fWeight,
                scales: fScales,
                biases: fusedBiases,
                transpose: true,
                groupSize: fusedGroupSize,
                bits: fusedBits,
                mode: fusedMode
            )
            return down(fusedSiluMul(gateUp, hiddenDims: hiddenDimensions))
        } else {
            // Fallback: separate dispatches (non-quantized)
            return down(silu(gate(x)) * up(x))
        }
    }
}

class Qwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    let mlp: Qwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(args)
        self.mlp = Qwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Qwen3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3ModelInner
    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3ModelInner(args)

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

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

public struct Qwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension Qwen3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
