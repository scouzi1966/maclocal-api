//
//  Qwen4Exp.swift
//  mlx-swift-lm
//
//  Text-generation port for Qwen/Qwen3.8-Flash-Next (model_type=qwen4_exp).
//  The model combines Gated DeltaNet, sparse MoE, hyper-connections, and a
//  sharded hashed n-gram PLE table, and QSA block-indexed attention.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen4ExpTextConfiguration: Decodable, Sendable {
    var modelType: String = "qwen4_exp_text"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var numExpertsPerToken: Int
    var numExperts: Int
    var layerTypes: [String]
    var rmsNormEps: Float
    var vocabularySize: Int
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var hcCount: Int
    var hcLowRank: Int
    var pleLayerIDs: [Int]
    var pleEmbedDim: Int
    var pleConvKernelSize: Int
    var ngramSize: Int
    var headsPerNgram: Int
    var ngramVocabularySizeBase: Int
    var ngramVocabularyDivisor: Int
    var splitNgramParts: Int
    var seed: Int
    var indexerHeads: Int
    var indexerKVHeads: Int
    var indexerHeadDim: Int
    var indexerBudget: Int
    var indexerCompressRatio: Int
    var normTopKProbability: Bool
    var outputGateType: String
    var eosTokenID: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var mropeSection: [Int]

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
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case numExpertsPerToken = "num_experts_per_tok"
        case numExperts = "num_experts"
        case layerTypes = "layer_types"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case hcCount = "hc_count"
        case hcLowRank = "hc_lowrank"
        case pleLayerIDs = "ple_layer_ids"
        case pleEmbedDim = "ple_embed_dim"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabularySizeBase = "ngram_vocab_size_base"
        case ngramVocabularyDivisor = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case seed
        case indexerHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case normTopKProbability = "norm_topk_prob"
        case outputGateType = "output_gate_type"
        case eosTokenID = "eos_token_id"
        case ropeParameters = "rope_parameters"
    }

    private struct RopeParameters: Codable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?
        var mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case mropeSection = "mrope_section"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        linearNumValueHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 32
        linearNumKeyHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        linearKeyHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        linearValueHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerToken)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        hcCount = try c.decodeIfPresent(Int.self, forKey: .hcCount) ?? 4
        hcLowRank = try c.decodeIfPresent(Int.self, forKey: .hcLowRank) ?? 320
        pleLayerIDs = try c.decodeIfPresent([Int].self, forKey: .pleLayerIDs) ?? []
        pleEmbedDim = try c.decodeIfPresent(Int.self, forKey: .pleEmbedDim) ?? hiddenSize
        pleConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
        ngramSize = try c.decodeIfPresent(Int.self, forKey: .ngramSize) ?? 3
        headsPerNgram = try c.decodeIfPresent(Int.self, forKey: .headsPerNgram) ?? 8
        ngramVocabularySizeBase = try c.decodeIfPresent(Int.self, forKey: .ngramVocabularySizeBase) ?? 20_000_000
        ngramVocabularyDivisor = try c.decodeIfPresent(Int.self, forKey: .ngramVocabularyDivisor) ?? 128
        splitNgramParts = try c.decodeIfPresent(Int.self, forKey: .splitNgramParts) ?? 128
        seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1234
        indexerHeads = try c.decodeIfPresent(Int.self, forKey: .indexerHeads) ?? 4
        indexerKVHeads = try c.decodeIfPresent(Int.self, forKey: .indexerKVHeads) ?? 1
        indexerHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexerHeadDim) ?? 128
        indexerBudget = try c.decodeIfPresent(Int.self, forKey: .indexerBudget) ?? 2048
        indexerCompressRatio = try c.decodeIfPresent(Int.self, forKey: .indexerCompressRatio) ?? 4
        normTopKProbability = try c.decodeIfPresent(Bool.self, forKey: .normTopKProbability) ?? true
        outputGateType = try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? "silu"
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 0
        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta ?? 10_000_000
        partialRotaryFactor = rope?.partialRotaryFactor ?? 0.25
        mropeSection = rope?.mropeSection ?? [11, 11, 10]
    }
}

public struct Qwen4ExpConfiguration: Decodable, Sendable {
    var modelType: String
    var textConfig: Qwen4ExpTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(modelType: String = "qwen4_exp", textConfig: Qwen4ExpTextConfiguration) {
        self.modelType = modelType
        self.textConfig = textConfig
    }
}

// MARK: - Normalization and hyper-connections

private final class Qwen4ExpZeroCenteredRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let groupSize: Int
    let eps: Float

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        self.groupSize = groupSize ?? dimensions
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalShape = x.shape
        let grouped = x.reshaped(Array(originalShape.dropLast()) + [-1, groupSize])
            .asType(.float32)
        let normalized = grouped * rsqrt(
            (grouped * grouped).mean(axis: -1, keepDims: true) + eps)
        let groupedWeight = (weight + 1).asType(.float32).reshaped(-1, groupSize)
        return (normalized * groupedWeight).asType(x.dtype).reshaped(originalShape)
    }
}

private final class Qwen4ExpGatedNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let sigmoidGate: Bool

    init(dimensions: Int, eps: Float, gateType: String) {
        self.eps = eps
        self.sigmoidGate = gateType == "sigmoid"
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let normalized = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        return normalized * (sigmoidGate ? sigmoid(gate) : silu(gate))
    }
}

private final class Qwen4ExpGatedResidual: Module {
    let hcCount: Int
    let hiddenSize: Int
    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixWeightUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInjectWeight: Linear?

    init(_ config: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        hcCount = config.hcCount
        hiddenSize = config.hiddenSize
        let width = hcCount * hiddenSize
        _hcNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _inputMixWeightDown.wrappedValue = Linear(width, config.hcLowRank, bias: false)
        _inputMixWeightUp.wrappedValue = Linear(config.hcLowRank, width, bias: false)
        if useCombine {
            _blockInjectWeight.wrappedValue = Linear(width, hcCount, bias: false)
        }
    }

    func mix(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let normalized = hcNorm(input)
        let weights = sigmoid(inputMixWeightUp(silu(inputMixWeightDown(normalized) / Float(hcCount))))
        let shape = Array(input.shape.dropLast())
        let mixed = (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
        let injection = 2 * sigmoid(blockInjectWeight!(normalized) / Float(hcCount))
        return (mixed, input, injection)
    }

    func combine(_ input: MLXArray) -> MLXArray {
        let normalized = hcNorm(input)
        let weights = sigmoid(inputMixWeightUp(silu(inputMixWeightDown(normalized) / Float(hcCount))))
        let shape = Array(input.shape.dropLast())
        return (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
    }

    func inject(_ output: MLXArray, residual: MLXArray, weights: MLXArray) -> MLXArray {
        let shape = Array(output.shape.dropLast())
        let injection = expandedDimensions(output, axis: -2)
            * expandedDimensions(weights, axis: -1)
        return residual + injection.reshaped(shape + [hcCount * hiddenSize])
    }
}

// MARK: - Attention

private final class Qwen4ExpMultimodalRoPE {
    private let invFreq: MLXArray
    private let mropeSection: [Int]

    init(dimensions: Int, base: Float, mropeSection: [Int]) {
        let frequency = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            / Float(dimensions)
        invFreq = 1 / pow(MLXArray(base), frequency)
        self.mropeSection = mropeSection
    }

    private func interleave(_ frequencies: MLXArray) -> MLXArray {
        let dimensions = frequencies.dim(-1)
        let triplets = min(
            min(mropeSection[1] * 3, dimensions),
            min(mropeSection[2] * 3, dimensions)) / 3
        guard triplets > 0 else { return frequencies[0, 0..., 0..., 0...] }

        let temporal = MLXArray(stride(from: 0, to: triplets * 3, by: 3).map(Int32.init))
        let height = MLXArray(stride(from: 1, to: triplets * 3, by: 3).map(Int32.init))
        let width = MLXArray(stride(from: 2, to: triplets * 3, by: 3).map(Int32.init))
        let selected = stacked([
            take(frequencies[0, 0..., 0..., 0...], temporal, axis: -1),
            take(frequencies[1, 0..., 0..., 0...], height, axis: -1),
            take(frequencies[2, 0..., 0..., 0...], width, axis: -1),
        ], axis: -1).reshaped(frequencies.dim(1), frequencies.dim(2), triplets * 3)

        let consumed = triplets * 3
        guard consumed < dimensions else { return selected }
        return concatenated([selected, frequencies[0, 0..., 0..., consumed...]], axis: -1)
    }

    private func frequencies(positionIDs: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        var positions = positionIDs
        if positions.ndim == 2 {
            positions = tiled(positions[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }
        let frequency = positions.asType(.float32)[0..., 0..., 0..., .newAxis]
            * invFreq[.newAxis, .newAxis, .newAxis, 0...]
        let interleaved = interleave(frequency)
        let embedding = concatenated([interleaved, interleaved], axis: -1)
        return (cos(embedding).asType(dtype), sin(embedding).asType(dtype))
    }

    func apply(_ tensor: MLXArray, positionIDs: MLXArray) -> MLXArray {
        let dimensions = invFreq.dim(0) * 2
        let rotated = tensor[0..., 0..., 0..., ..<dimensions]
        let tail = tensor[0..., 0..., 0..., dimensions...]
        let halves = MLX.split(rotated, parts: 2, axis: -1)
        let rotatedHalf = concatenated([-halves[1], halves[0]], axis: -1)
        let (cosine, sine) = frequencies(positionIDs: positionIDs, dtype: tensor.dtype)
        let result = rotated * cosine[0..., .newAxis, 0..., 0...]
            + rotatedHalf * sine[0..., .newAxis, 0..., 0...]
        return tail.size == 0 ? result : concatenated([result, tail], axis: -1)
    }
}

private final class Qwen4ExpAttentionCache: KVCache {
    var offset = 0
    var offsetArray: MLXArray? { nil }
    var maxSize: Int? { nil }
    private var keys: MLXArray?
    private var values: MLXArray?
    private var indexKeys: MLXArray?
    private var indexPositionIDs: MLXArray?

    func updateIndexKeys(_ newKeys: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
        indexKeys = indexKeys.map { concatenated([$0, newKeys], axis: 1) } ?? newKeys
        let axis = positionIDs.ndim - 1
        indexPositionIDs = indexPositionIDs.map {
            concatenated([$0, positionIDs], axis: axis)
        } ?? positionIDs
        return (indexKeys!, indexPositionIDs!)
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        keys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        values = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        offset += newKeys.dim(2)
        return (keys!, values!)
    }

    var state: [MLXArray] {
        get { [keys, values, indexKeys, indexPositionIDs].compactMap { $0 } }
        set {
            precondition((2 ... 4).contains(newValue.count))
            keys = newValue[0]
            values = newValue[1]
            indexKeys = newValue.count >= 3 ? newValue[2] : nil
            indexPositionIDs = newValue.count == 4 ? newValue[3] : nil
            offset = newValue[0].dim(2)
        }
    }

    var metaState: [String] {
        get { [] }
        set { precondition(newValue.isEmpty) }
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let amount = min(offset, n)
        offset -= amount
        if let keys { self.keys = keys[.ellipsis, ..<offset, 0...] }
        if let values { self.values = values[.ellipsis, ..<offset, 0...] }
        if let indexKeys { self.indexKeys = indexKeys[0..., ..<offset, 0...] }
        if let positions = indexPositionIDs {
            indexPositionIDs = positions.ndim == 2
                ? positions[0..., ..<offset]
                : positions[0..., 0..., ..<offset]
        }
        return amount
    }

    func truncateToOffset() {}

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func innerState() -> [MLXArray] {
        [keys, values, indexKeys, indexPositionIDs].compactMap { $0 }
    }
}

private final class Qwen4ExpQSAIndexer: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopK: Int
    let rope: Qwen4ExpMultimodalRoPE
    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: Qwen4ExpZeroCenteredRMSNorm

    init(_ config: Qwen4ExpTextConfiguration) {
        heads = config.indexerHeads
        kvHeads = config.indexerKVHeads
        headDim = config.indexerHeadDim
        tokenBudget = config.indexerBudget
        compressRatio = config.indexerCompressRatio
        blockTopK = config.indexerBudget / config.indexerCompressRatio
        _indexQKProj.wrappedValue = Linear(
            config.hiddenSize,
            (config.indexerHeads + config.indexerKVHeads) * config.indexerHeadDim,
            bias: false)
        _qLayerNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: config.indexerHeadDim, eps: config.rmsNormEps)
        _kLayerNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: config.indexerHeadDim, eps: config.rmsNormEps)
        let rotaryDimensions = Int(Float(config.indexerHeadDim) * config.partialRotaryFactor)
        rope = Qwen4ExpMultimodalRoPE(
            dimensions: rotaryDimensions, base: config.ropeTheta,
            mropeSection: config.mropeSection)
    }

    func callAsFunction(
        _ hidden: MLXArray,
        positionIDs providedPositionIDs: MLXArray?,
        cache: Qwen4ExpAttentionCache?
    ) -> MLXArray? {
        let (batch, length) = (hidden.dim(0), hidden.dim(1))
        let previousOffset = cache?.offset ?? 0
        let positionIDs = providedPositionIDs ?? tiled(
            MLXArray(Int32(previousOffset) ..< Int32(previousOffset + length))[.newAxis, 0...],
            repetitions: [batch, 1])
        let qk = indexQKProj(hidden)
        let splitPoint = heads * headDim
        let parts = MLX.split(qk, indices: [splitPoint], axis: -1)
        var queries = qLayerNorm(parts[0].reshaped(batch, length, heads, headDim))
            .transposed(0, 2, 1, 3)
        queries = rope.apply(queries, positionIDs: positionIDs)
        let currentKeys = parts[1].reshaped(batch, length, kvHeads, headDim)
            .mean(axis: 2)
        let (allKeys, allPositionIDs) = cache?.updateIndexKeys(
            currentKeys, positionIDs: positionIDs) ?? (currentKeys, positionIDs)
        let totalLength = allKeys.dim(1)

        guard totalLength > tokenBudget else { return nil }

        let completeBlocks = totalLength / compressRatio
        let pooled = allKeys[0..., ..<(completeBlocks * compressRatio), 0...]
            .reshaped(batch, completeBlocks, compressRatio, headDim)
            .asType(.float32).mean(axis: 2).asType(allKeys.dtype)
        var blockKeys = kLayerNorm(pooled)
        let blockIndices = MLXArray(
            stride(from: 0, to: completeBlocks * compressRatio, by: compressRatio).map(Int32.init))
        let positionAxis = allPositionIDs.ndim - 1
        let blockPositionIDs = take(allPositionIDs, blockIndices, axis: positionAxis)
        blockKeys = rope.apply(
            expandedDimensions(blockKeys, axis: 1), positionIDs: blockPositionIDs
        ).squeezed(axis: 1)
        let tokenPositions = MLXArray(Int32(0) ..< Int32(totalLength))
        let tokenBlockIDs = tokenPositions.floorDivide(compressRatio)
        var rows = [MLXArray]()

        for queryIndex in 0 ..< length {
            let visibleCount = previousOffset + queryIndex + 1
            let visibleBlocks = visibleCount / compressRatio
            if visibleBlocks <= blockTopK {
                rows.append((tokenPositions .< visibleCount)[.newAxis, 0...])
                continue
            }

            let query = queries[0..., 0..., queryIndex, 0...]
            let visibleBlockKeys = blockKeys[0..., ..<visibleBlocks, 0...]
            let scores = maximum(
                (expandedDimensions(query, axis: -2)
                    * expandedDimensions(visibleBlockKeys, axis: 1))
                    .sum(axis: -1),
                0
            ).sum(axis: 1) / sqrt(Float(headDim))
            let selectedBlocks = MLX.argPartition(
                -scores, kth: blockTopK - 1, axis: -1)[0..., ..<blockTopK]
            let selectedTokens = (
                expandedDimensions(tokenBlockIDs, axes: [0, 1])
                    .== expandedDimensions(selectedBlocks, axis: -1)
            ).asType(.int32).sum(axis: 1) .> 0
            let tailStart = visibleBlocks * compressRatio
            let tail = (tokenPositions .>= tailStart) .&& (tokenPositions .< visibleCount)
            rows.append(selectedTokens .|| tail[.newAxis, 0...])
        }

        return stacked(rows, axis: 1)[.ellipsis, .newAxis, 0..., 0...]
    }
}

private final class Qwen4ExpAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: Qwen4ExpMultimodalRoPE
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo var indexer: Qwen4ExpQSAIndexer

    init(_ config: Qwen4ExpTextConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim * 2, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(config)
        rope = Qwen4ExpMultimodalRoPE(
            dimensions: Int(Float(headDim) * config.partialRotaryFactor),
            base: config.ropeTheta, mropeSection: config.mropeSection)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIDs providedPositionIDs: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        let positionIDs = providedPositionIDs ?? tiled(
            MLXArray(Int32(offset) ..< Int32(offset + l))[.newAxis, 0...],
            repetitions: [b, 1])
        let qsaMask = indexer(
            x, positionIDs: positionIDs, cache: cache as? Qwen4ExpAttentionCache)
        let qParts = MLX.split(qProj(x).reshaped(b, l, heads, headDim * 2), parts: 2, axis: -1)
        var q = qNorm(qParts[0]).transposed(0, 2, 1, 3)
        let gate = qParts[1].reshaped(b, l, -1)
        var k = kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope.apply(q, positionIDs: positionIDs)
        k = rope.apply(k, positionIDs: positionIDs)
        var output = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale,
            mask: qsaMask.map { .array($0) } ?? mask)
            .transposed(0, 2, 1, 3).reshaped(b, l, -1)
        output = output * sigmoid(gate)
        return oProj(output)
    }
}

// MARK: - Gated DeltaNet

private final class Qwen4ExpGatedDeltaNet: Module {
    let keyDim: Int
    let valueDim: Int
    let keyHeads: Int
    let valueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let convKernel: Int
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo var norm: Qwen4ExpGatedNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen4ExpTextConfiguration) {
        keyHeads = config.linearNumKeyHeads
        valueHeads = config.linearNumValueHeads
        keyHeadDim = config.linearKeyHeadDim
        valueHeadDim = config.linearValueHeadDim
        keyDim = keyHeads * keyHeadDim
        valueDim = valueHeads * valueHeadDim
        convKernel = config.linearConvKernelDim
        let convDim = keyDim * 2 + valueDim
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim,
            kernelSize: convKernel, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(config.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([valueHeads])
        _aLog.wrappedValue = MLX.log(MLXArray.ones([valueHeads]) * 8)
        _norm.wrappedValue = Qwen4ExpGatedNorm(
            dimensions: valueHeadDim, eps: config.rmsNormEps, gateType: config.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let projected = inProjQKV(x)
        let prior = cache?[0] ?? MLXArray.zeros(
            [b, convKernel - 1, keyDim * 2 + valueDim], dtype: x.dtype)
        let convInput = concatenated([prior, projected], axis: 1)
        cache?[0] = convInput[0..., (convInput.dim(1) - convKernel + 1)...]
        let mixed = silu(conv1d(convInput))
        let pieces = MLX.split(mixed, indices: [keyDim, keyDim * 2], axis: -1)
        var q = pieces[0].reshaped(b, l, keyHeads, keyHeadDim)
        var k = pieces[1].reshaped(b, l, keyHeads, keyHeadDim)
        let v = pieces[2].reshaped(b, l, valueHeads, valueHeadDim)
        q = q * rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
            * pow(Float(keyHeadDim), -0.5)
        k = k * rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)
        let (output, state) = gatedDeltaUpdate(
            q: q, k: k, v: v,
            a: inProjA(x), b: inProjB(x),
            ALog: aLog, dtBias: dtBias,
            state: cache?[1], useKernel: true)
        cache?[1] = state
        let z = inProjZ(x).reshaped(b, l, valueHeads, valueHeadDim)
        return outProj(norm(output, gate: z).reshaped(b, l, valueDim))
    }
}

// MARK: - MoE

private final class Qwen4ExpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class Qwen4ExpSparseMoE: Module, UnaryLayer {
    let topK: Int
    let normalize: Bool
    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen4ExpMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ config: Qwen4ExpTextConfiguration) {
        topK = config.numExpertsPerToken
        normalize = config.normTopKProbability
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts)
        _sharedExpert.wrappedValue = Qwen4ExpMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x)
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let indices = MLX.argPartition(-probabilities, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        if normalize { scores = scores / scores.sum(axis: -1, keepDims: true) }
        let routed = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

// MARK: - PLE n-gram embedding

private let qwen4SplitMixGamma: UInt64 = 0x9E3779B97F4A7C15
private let qwen4SplitMixM1: UInt64 = 0xBF58476D1CE4E5B9
private let qwen4SplitMixM2: UInt64 = 0x94D049BB133111EB

private func qwen4SplitMix64(_ input: UInt64) -> UInt64 {
    var value = input &+ qwen4SplitMixGamma
    value = (value ^ (value >> 30)) &* qwen4SplitMixM1
    value = (value ^ (value >> 27)) &* qwen4SplitMixM2
    return value ^ (value >> 31)
}

private func qwen4IsPrime(_ value: Int) -> Bool {
    if value < 2 { return false }
    if value % 2 == 0 { return value == 2 }
    var divisor = 3
    while divisor * divisor <= value {
        if value % divisor == 0 { return false }
        divisor += 2
    }
    return true
}

private func qwen4NthPrime(after start: Int, count: Int) -> Int {
    var result = start
    for _ in 0 ..< count {
        result += 1
        while !qwen4IsPrime(result) { result += 1 }
    }
    return result
}

private final class Qwen4ExpShardedEmbedding: Module {
    let rowsPerShard: Int
    @ModuleInfo var shards: [Embedding]

    init(rows: Int, dimensions: Int, parts: Int) {
        precondition(rows % parts == 0)
        let shardRows = rows / parts
        rowsPerShard = shardRows
        _shards.wrappedValue = (0 ..< parts).map { _ in
            Embedding(embeddingCount: shardRows, dimensions: dimensions)
        }
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let shardIDs = ids.floorDivide(rowsPerShard)
        let localIDs = ids % rowsPerShard
        var result: MLXArray?
        for (index, shard) in shards.enumerated() {
            let safeIDs = which(shardIDs .== index, localIDs, 0)
            let values = shard(safeIDs) * (shardIDs .== index)[.ellipsis, .newAxis]
            result = result.map { $0 + values } ?? values
        }
        return result!
    }
}

private final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let headsPerNgram: Int
    let eosTokenID: Int
    let contextLength: Int
    let headSizes: MLXArray
    let headOffsets: MLXArray
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ModuleInfo(key: "ngram_embedding") var ngramEmbedding: Qwen4ExpShardedEmbedding

    init(_ config: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        ngramSize = config.ngramSize
        headsPerNgram = config.headsPerNgram
        eosTokenID = config.eosTokenID
        contextLength = ngramSize - 1
        let heads = contextLength * headsPerNgram
        var sizes = [Int]()
        var offsets = [Int]()
        var total = 0
        for head in 0 ..< heads {
            let globalHead = pleLayerIndex * heads + head
            let size = qwen4NthPrime(after: config.ngramVocabularySizeBase - 1, count: globalHead + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        let padded = ((total + config.ngramVocabularyDivisor - 1) / config.ngramVocabularyDivisor)
            * config.ngramVocabularyDivisor
        headSizes = MLXArray(sizes)
        headOffsets = MLXArray(offsets)
        let maxMultiplier = Int64.max / Int64(max(config.vocabularySize, 1))
        let bound = UInt64(max(1, maxMultiplier / 2))
        let base = UInt64(config.seed + 10_007 * pleLayerIndex)
        let multipliers: [Int64] = (0 ..< ngramSize).map { index in
            let value = base &+ qwen4SplitMixGamma &* UInt64(index + 1)
            return Int64(2 * (qwen4SplitMix64(value) % bound) + 1)
        }
        _layerMultipliers.wrappedValue = MLXArray(multipliers)
        _ngramEmbedding.wrappedValue = Qwen4ExpShardedEmbedding(
            rows: padded,
            dimensions: config.pleEmbedDim / heads,
            parts: config.splitNgramParts)
    }

    private func shifted(_ ids: MLXArray, by shift: Int) -> MLXArray {
        if shift == 0 { return ids }
        let length = ids.dim(1)
        let positions = MLXArray(0 ..< length)
        let eosPositions = which(ids .== eosTokenID, positions[.newAxis, 0...], -1)
        let inclusive = eosPositions.cummax(axis: 1)
        let priorEOS = concatenated([
            MLXArray.full([ids.dim(0), 1], values: MLXArray(-1), dtype: ids.dtype),
            inclusive[0..., ..<(length - 1)],
        ], axis: 1)
        let source = positions - shift
        let gather = maximum(source, 0)[.newAxis, 0...]
        let candidate = MLX.takeAlong(ids, gather, axis: 1)
        let valid = (positions[.newAxis, 0...] - (priorEOS + 1) .>= shift)
            .&& (source[.newAxis, 0...] .>= 0)
        return which(valid, candidate, eosTokenID)
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: ArraysCache?) -> MLXArray {
        let ids = inputIDs.asType(.int64)
        let previous = cache?[3] ?? MLXArray.full(
            [ids.dim(0), contextLength], values: MLXArray(eosTokenID), dtype: .int64)
        let history = concatenated([previous, ids], axis: 1)
        cache?[3] = history[0..., (history.dim(1) - contextLength)...]
        let shiftedIDs = (0 ..< ngramSize).map { shifted(history, by: $0) }
        var blocks = [MLXArray]()
        for order in 2 ... ngramSize {
            let start = (order - 2) * headsPerNgram
            let end = start + headsPerNgram
            var mixed = shiftedIDs[0] * layerMultipliers[0]
            for position in 1 ..< order {
                mixed = MLX.bitwiseXOr(mixed, shiftedIDs[position] * layerMultipliers[position])
            }
            let sizes = headSizes[start ..< end]
            let offsets = headOffsets[start ..< end]
            blocks.append((mixed[.ellipsis, .newAxis] % sizes) + offsets)
        }
        let allNgramIDs = concatenated(blocks, axis: -1)
        let outputStart = allNgramIDs.dim(1) - ids.dim(1)
        let ngramIDs = allNgramIDs[0..., outputStart...]
        return ngramEmbedding(ngramIDs).flattened(start: -2)
    }
}

private final class Qwen4ExpPLE: Module {
    let hiddenSize: Int
    let hcCount: Int
    let shortStateLength: Int
    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo var conv1d: Conv1d

    init(_ config: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        hiddenSize = config.hiddenSize
        hcCount = config.hcCount
        let width = hiddenSize * hcCount
        shortStateLength = (config.pleConvKernelSize - 1) * config.ngramSize
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(config, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(config.pleEmbedDim, width, bias: false)
        _valueProj.wrappedValue = Linear(config.pleEmbedDim, hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width,
            kernelSize: config.pleConvKernelSize,
            dilation: config.ngramSize, groups: width, bias: false)
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let prior = cache?[2] ?? MLXArray.zeros(
            [x.dim(0), shortStateLength, x.dim(2)], dtype: x.dtype)
        let input = concatenated([prior, x], axis: 1)
        cache?[2] = input[0..., (input.dim(1) - shortStateLength)...]
        return silu(conv1d(input))
    }

    func callAsFunction(_ hidden: MLXArray, inputIDs: MLXArray, cache: ArraysCache?) -> MLXArray {
        let embedding = pleEmbedding(inputIDs, cache: cache)
        let shape = Array(hidden.shape.dropLast())
        let key = normKey(keyProj(embedding)).reshaped(shape + [hcCount, hiddenSize])
        let query = normQuery(hidden).reshaped(shape + [hcCount, hiddenSize])
        var gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(Float(hiddenSize))
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
        let value = expandedDimensions(valueProj(embedding), axis: -2)
        let gated = (sigmoid(gate) * value).reshaped(shape + [hcCount * hiddenSize])
        return gated + shortConv(normConv(gated), cache: cache)
    }
}

// MARK: - Decoder and model

private final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "linear_attn") var linearAttention: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttention: Qwen4ExpAttention?
    @ModuleInfo var mlp: Qwen4ExpSparseMoE
    @ModuleInfo var ple: Qwen4ExpPLE?
    @ModuleInfo(key: "attn_hyper_connection") var attentionHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    init(_ config: Qwen4ExpTextConfiguration, layerIndex: Int) {
        isLinear = config.layerTypes[layerIndex] == "linear_attention"
        if isLinear { _linearAttention.wrappedValue = Qwen4ExpGatedDeltaNet(config) }
        else { _selfAttention.wrappedValue = Qwen4ExpAttention(config) }
        _mlp.wrappedValue = Qwen4ExpSparseMoE(config)
        if let pleIndex = config.pleLayerIDs.firstIndex(of: layerIndex + 1) {
            _ple.wrappedValue = Qwen4ExpPLE(config, pleLayerIndex: pleIndex)
        }
        _attentionHyperConnection.wrappedValue = Qwen4ExpGatedResidual(config)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(config)
    }

    func callAsFunction(
        _ input: MLXArray,
        inputIDs: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIDs: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let arrayCache = cache as? ArraysCache
        var hidden = input
        if let ple { hidden = hidden + ple(hidden, inputIDs: inputIDs, cache: arrayCache) }
        var mixed: MLXArray
        var residual: MLXArray
        var injection: MLXArray
        (mixed, residual, injection) = attentionHyperConnection.mix(hidden)
        let attended = isLinear
            ? linearAttention!(mixed, cache: arrayCache)
            : selfAttention!(
                mixed, mask: attentionMask, positionIDs: positionIDs, cache: cache)
        hidden = attentionHyperConnection.inject(attended, residual: residual, weights: injection)
        (mixed, residual, injection) = mlpHyperConnection.mix(hidden)
        return mlpHyperConnection.inject(mlp(mixed), residual: residual, weights: injection)
    }
}

private final class Qwen4ExpModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    init(_ config: Qwen4ExpTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Qwen4ExpDecoderLayer(config, layerIndex: $0)
        }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(config, useCombine: false)
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        var hidden = MLX.tiled(
            inputEmbeddings ?? embedTokens(inputIDs),
            repetitions: [1, 1, hyperConnectionMixer.hcCount])
        let layerCaches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let attentionIndex = layers.firstIndex { !$0.isLinear }
        let mask = attentionIndex.map { createAttentionMask(h: hidden, cache: layerCaches[$0]) } ?? .none
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden, inputIDs: inputIDs,
                attentionMask: mask, positionIDs: positionIDs, cache: layerCaches[index])
        }
        return hyperConnectionMixer.combine(hidden)
    }
}

public final class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    @ModuleInfo(key: "model") private var model: Qwen4ExpModelInner
    let configuration: Qwen4ExpTextConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ wrapper: Qwen4ExpConfiguration) {
        let config = wrapper.textConfig
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = config.layerTypes.map { $0 == "linear_attention" ? 0 : config.kvHeads }
        _model.wrappedValue = Qwen4ExpModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    public func forward(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        let hidden = model(
            inputIDs, inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType -> KVCache in
            if layerType == "linear_attention" {
                return ArraysCache(size: 4)
            }
            return Qwen4ExpAttentionCache()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        let zeroCenteredNormSuffixes = [
            ".hc_norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".norm_key.weight",
            ".norm_query.weight",
            ".norm_conv.weight",
        ]
        for (originalKey, value) in weights {
            guard originalKey.hasPrefix("language_model.") else { continue }
            var key = String(originalKey.dropFirst("language_model.".count))
            guard !key.hasPrefix("mtp.") else { continue }
            guard !key.hasSuffix(".ngram_heads_offsets"),
                  !key.hasSuffix(".ngram_heads_vocab_sizes") else { continue }
            if key.contains(".ple.ple_embedding.ngram_embedding.shard_") {
                key = key.replacingOccurrences(
                    of: ".ple.ple_embedding.ngram_embedding.shard_",
                    with: ".ple.ple_embedding.ngram_embedding.shards.")
            }
            result[key] = zeroCenteredNormSuffixes.contains(where: key.hasSuffix) ? value - 1 : value
        }
        if configuration.tieWordEmbeddings { result["lm_head.weight"] = nil }
        return result
    }
}

extension Qwen4ExpModel: LoRAModel {
    public var loraLayers: [Module] { model.layers.map { $0 as Module } }
}
