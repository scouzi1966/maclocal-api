//
//  GLM5MoeDsa.swift
//  mlx-swift-lm
//
//  Port of deepseek_v32.py + glm_moe_dsa.py wrapper
//  MLA (Multi-head Latent Attention) with MoE and Dynamic Sparse Attention (DSA)
//  for GLM-5

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MultiLinear

/// Per-head linear layer with weight shape [numHeads, outputDims, inputDims].
/// Supports both regular and quantized weights.
class GLM5MoeDsaMultiLinear: Module {
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
        // weights with different shapes than logical dims, so verify: [.all] shape
        // checks would fail with full-sized init values.
        self._weight.wrappedValue = MLXArray(Float(0))
        self._scales.wrappedValue = MLXArray(Float(0))
        self._biases.wrappedValue = MLXArray(Float(0))

        super.init()
    }

    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        if let scales, let biases, scales.size > 1 {
            // Quantization is always along the last weight dim (= inputDims)
            let dims = inputDims
            let bits = (weight.dim(-1) * 32) / dims
            let groupSize = dims / scales.dim(-1)
            return quantizedMatmul(
                x, weight, scales: scales, biases: biases,
                transpose: transpose, groupSize: groupSize, bits: bits)
        } else {
            if transpose {
                return matmul(x, weight.swappedAxes(-1, -2))
            } else {
                return matmul(x, weight)
            }
        }
    }
}

// MARK: - Indexer (Dynamic Sparse Attention)

/// Computes top-k KV indices for sparse attention selection.
/// Uses separate Q/K projections with RoPE to score all cached KV positions,
/// then selects the most relevant ones for each query position.
class GLM5MoeDsaIndexer: Module {
    let dim: Int
    let nHeads: Int
    let headDim: Int
    let ropeHeadDim: Int
    let indexTopk: Int
    let qLoraRank: Int
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    let rope: RoPE

    init(_ config: GLM5MoeDsaConfiguration) {
        self.dim = config.hiddenSize
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.indexTopk = config.indexTopk
        self.qLoraRank = config.qLoraRank!
        self.softmaxScale = pow(Float(config.indexHeadDim), -0.5)

        self._wqB.wrappedValue = Linear(qLoraRank, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, headDim, bias: false)
        self._kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        self._weightsProj.wrappedValue = Linear(dim, nHeads, bias: false)

        self.rope = RoPE(
            dimensions: config.qkRopeHeadDim,
            traditional: true,
            base: config.ropeTheta)

        super.init()
    }

    /// Returns top-k indices for sparse attention, or nil if cache is too small.
    func callAsFunction(
        _ x: MLXArray, qr: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let b = x.dim(0)
        let s = x.dim(1)

        // Q projection from compressed q representation
        var q = wqB(qr)
        q = q.reshaped(b, s, nHeads, headDim).swappedAxes(1, 2)
        let qSplit = split(q, indices: [ropeHeadDim], axis: -1)
        var qPe = qSplit[0]
        let qNope = qSplit[1]

        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        q = concatenated([qPe, qNope], axis: -1)

        // K projection (single head)
        var k = wk(x)
        k = kNorm(k)
        k = k.reshaped(b, 1, s, headDim)
        let kSplit = split(k, indices: [ropeHeadDim], axis: -1)
        var kPe = kSplit[0]
        let kNope = kSplit[1]
        kPe = rope(kPe, offset: offset)
        k = concatenated([kPe, kNope], axis: -1)

        // Update indexer cache (k only, values are empty)
        if let cache {
            let (cachedK, _) = cache.update(keys: k, values: MLXArray.zeros([b, 1, s, 0]))
            k = cachedK
        }

        // Don't apply sparse selection if not enough cached entries
        if k.dim(2) <= indexTopk {
            return nil
        }

        // Compute indexer scores: [B, nHeads, s, totalSeq]
        var scores = matmul(q, k.swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(Float(0)))

        // Weight scores per head and aggregate
        var weights = weightsProj(x) * (pow(Float(nHeads), -0.5) * softmaxScale)
        weights = weights.swappedAxes(-1, -2)[.ellipsis, .newAxis]
        scores = scores * weights
        scores = scores.sum(axis: 1, keepDims: true)  // [B, 1, s, totalSeq]

        // Apply mask
        if let mask {
            scores = MLX.where(mask, scores, MLXArray(-Float.infinity))
        }

        // Return top-k indices (largest scores)
        let totalSeq = scores.dim(-1)
        let kthIdx = totalSeq - indexTopk
        return argPartition(scores, kth: kthIdx, axis: -1)[.ellipsis, kthIdx...]
    }
}

// MARK: - Attention (MLA + DSA)

class GLM5MoeDsaAttention: Module {
    let config: GLM5MoeDsaConfiguration
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
    @ModuleInfo(key: "embed_q") var embedQ: GLM5MoeDsaMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: GLM5MoeDsaMultiLinear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: GLM5MoeDsaIndexer

    let rope: RoPE

    init(_ config: GLM5MoeDsaConfiguration) {
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
                dimensions: qLoraRank, eps: 1e-6)
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
            dimensions: kvLoraRank, eps: 1e-6)

        self._embedQ.wrappedValue = GLM5MoeDsaMultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        self._unembedOut.wrappedValue = GLM5MoeDsaMultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)

        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        self._indexer.wrappedValue = GLM5MoeDsaIndexer(config)

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

        // Access sub-caches from CacheList
        let cacheList = cache as? CacheList
        let mainCache = cacheList?[0]
        let indexerCache = cacheList?[1]

        // Compute Q (compressed)
        let qr: MLXArray
        var q: MLXArray
        if qLoraRank == nil {
            qr = x  // unused for indexer in non-LoRA path, but needed for type
            q = qProj!(x)
        } else {
            let qrComputed = qALayerNorm!(qAProj!(x))
            qr = qrComputed
            q = qBProj!(qrComputed)
        }
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = qSplit[0]  // [B, numHeads, L, qkNopeHeadDim]
        var qPe = qSplit[1]    // [B, numHeads, L, qkRopeHeadDim]

        // Compress KV
        let compressedKvFull = kvAProjWithMqa(x)
        let kvSplit = split(compressedKvFull, indices: [kvLoraRank], axis: -1)
        let compressedKv = kvSplit[0]
        var kPe = kvSplit[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        // Apply RoPE
        let offset = mainCache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)

        // Expand kv_latent to [B, 1, L, kvLoraRank]
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        // Update main cache
        if let mainCache {
            let (updatedKvLatent, updatedKPe) = mainCache.update(keys: kvLatent, values: kPe)
            kvLatent = updatedKvLatent
            kPe = updatedKPe
        }

        // Dynamic Sparse Attention via indexer
        var currentMask = mask
        let topkIndices = indexer(x, qr: qr, mask: currentMask, cache: indexerCache)

        if let topkIndices {
            if L == 1 {
                // Decode mode: physically gather top-k KV entries
                // topkIndices shape: [B, 1, 1, topk]
                var idx = topkIndices.squeezed(axis: 2)  // [B, 1, topk]
                idx = expandedDimensions(idx, axis: -1)  // [B, 1, topk, 1]

                // Gather from kv_latent: [B, 1, totalSeq, kvLoraRank] → [B, 1, topk, kvLoraRank]
                let latentBcastShape = [idx.dim(0), idx.dim(1), idx.dim(2), kvLatent.dim(-1)]
                let latentIdx = broadcast(idx, to: latentBcastShape)
                kvLatent = takeAlong(kvLatent, latentIdx, axis: 2)

                // Gather from k_pe: [B, 1, totalSeq, qkRopeHeadDim] → [B, 1, topk, qkRopeHeadDim]
                let peBcastShape = [idx.dim(0), idx.dim(1), idx.dim(2), kPe.dim(-1)]
                let peIdx = broadcast(idx, to: peBcastShape)
                kPe = takeAlong(kPe, peIdx, axis: 2)

                currentMask = nil
            } else {
                // Prefill mode: create sparse boolean mask
                // topkIndices shape: [B, 1, L, topk]
                var shape = topkIndices.shape
                shape[shape.count - 1] = kvLatent.dim(2)  // replace topk with totalSeq
                var sparseMask = MLXArray.zeros(shape).asType(Bool.self)
                sparseMask = putAlong(sparseMask, topkIndices, values: MLXArray(true), axis: -1)
                if let currentMask {
                    sparseMask = sparseMask * currentMask
                }
                currentMask = sparseMask
            }
        }

        // Create dependency edge to keep indexer cache in the computation graph
        // This prevents the graph from growing too large during generation
        if var mainCacheMut = cacheList?[0], let indexerCache = cacheList?[1] {
            var mainState = mainCacheMut.state
            if mainState.count >= 1, indexerCache.state.count >= 2 {
                mainState[0] = depends(
                    input: mainState[0],
                    dependencies: [indexerCache.state[0], indexerCache.state[1]])
                mainCacheMut.state = mainState
            }
        }

        // Compute positional attention scores
        var peScores = matmul(qPe * scale, kPe.swappedAxes(-1, -2))
        if let currentMask {
            peScores = MLX.where(
                currentMask, peScores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: peScores.dtype))
        }

        // MLA attention with two-path optimization
        let output: MLXArray
        if L == 1 {
            // Decode path: work in latent space (cheaper)
            qNope = embedQ(qNope)
            let attnOut = mlaAttention(
                queries: qNope, keys: kvLatent, values: kvLatent,
                scale: scale, peMask: peScores)
            output = unembedOut(attnOut)
        } else {
            // Prefill path: project K and V from latent space
            let k = embedQ(kvLatent, transpose: false)
            let v = unembedOut(kvLatent)
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
        var scores = matmul(queries, keys.swappedAxes(-1, -2)) * scale
        scores = scores + peMask
        let weights = softmax(scores, axis: -1)
        return matmul(weights, values)
    }
}

// MARK: - MLP

class GLM5MoeDsaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM5MoeDsaConfiguration, intermediateSize: Int? = nil) {
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

class GLM5MoeDsaMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM5MoeDsaConfiguration) {
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

class GLM5MoeDsaMoE: Module, UnaryLayer {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "gate") var gate: GLM5MoeDsaMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5MoeDsaMLP?

    init(_ config: GLM5MoeDsaConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok

        self._gate.wrappedValue = GLM5MoeDsaMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts!)

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            self._sharedExperts.wrappedValue = GLM5MoeDsaMLP(
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

class GLM5MoeDsaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM5MoeDsaAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: GLM5MoeDsaConfiguration, layerIdx: Int) {
        self._attention.wrappedValue = GLM5MoeDsaAttention(config)

        let useMoe = config.nRoutedExperts != nil
            && layerIdx >= config.firstKDenseReplace
            && layerIdx % config.moeLayerFreq == 0
        self.mlp = useMoe ? GLM5MoeDsaMoE(config) : GLM5MoeDsaMLP(config)

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

class GLM5MoeDsaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLM5MoeDsaDecoderLayer]
    let norm: RMSNorm
    let numHiddenLayers: Int

    init(_ config: GLM5MoeDsaConfiguration) {
        precondition(config.vocabSize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0 ..< config.numHiddenLayers).map {
            GLM5MoeDsaDecoderLayer(config, layerIdx: $0)
        }
        self.numHiddenLayers = config.numHiddenLayers
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Create attention mask using first layer's main cache offset
        let firstMainCache: KVCache? = {
            guard let first = cache?.first as? CacheList else { return nil }
            return first[0]
        }()
        let mask: MLXArray? = createBoolMask(h: h, cache: firstMainCache)

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
        let rowIdx = MLXArray(0 ..< T).reshaped(T, 1) + offset
        let colIdx = MLXArray(0 ..< (T + offset)).reshaped(1, T + offset)
        return rowIdx .>= colIdx
    }
}

// MARK: - Top-level Model

public class GLM5MoeDsaModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: GLM5MoeDsaModelInner
    let configuration: GLM5MoeDsaConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] {
        model.layers
    }

    public init(_ config: GLM5MoeDsaConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        // kvHeads used for framework info; actual cache creation overridden in newCache
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in 1 }

        self._model.wrappedValue = GLM5MoeDsaModelInner(config)

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

    /// Override cache creation to use CacheList (two KVCaches per layer:
    /// [0] = main MLA attention cache, [1] = DSA indexer cache)
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return (0 ..< configuration.numHiddenLayers).map { _ in
            CacheList(KVCacheSimple(), KVCacheSimple())
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        // Filter out MTP (multi-token prediction) layers
        let mptLayer = configuration.numHiddenLayers
        sanitized = sanitized.filter { key, _ in
            let parts = key.split(separator: ".")
            if parts.count >= 3, parts[1] == "layers",
               let layerIdx = Int(parts[2]), layerIdx >= mptLayer
            {
                return false
            }
            return true
        }

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
                    let qBits = 4
                    let qGroupSize = 64
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

public struct GLM5MoeDsaConfiguration: Codable, Sendable {
    var modelType: String = "glm_moe_dsa"
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
    var tieWordEmbeddings: Bool
    var topkMethod: String

    // DSA-specific fields
    var indexHeadDim: Int
    var indexNHeads: Int
    var indexTopk: Int

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
        case tieWordEmbeddings = "tie_word_embeddings"
        case topkMethod = "topk_method"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopk = "index_topk"
    }

    /// Separate key for GLM-5's rope_parameters field (not a stored property)
    private enum ExtraCodingKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm_moe_dsa"
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
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 202752
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"

        // DSA-specific fields
        indexHeadDim = try c.decode(Int.self, forKey: .indexHeadDim)
        indexNHeads = try c.decode(Int.self, forKey: .indexNHeads)
        indexTopk = try c.decode(Int.self, forKey: .indexTopk)

        // Handle rope_parameters → ropeScaling + ropeTheta mapping (GLM-5 specific)
        // GLM-5 uses "rope_parameters" in config.json instead of "rope_scaling"/"rope_theta"
        let extra = try decoder.container(keyedBy: ExtraCodingKeys.self)
        if let ropeParams = try extra.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters) {
            self.ropeScaling = ropeParams
            if case .float(let theta) = ropeParams["rope_theta"] {
                self.ropeTheta = theta
            } else {
                self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
            }
        } else {
            self.ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        }
    }
}
