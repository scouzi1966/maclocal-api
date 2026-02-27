//
//  Qwen3_5MoEVL.swift
//  mlx-swift-lm
//
//  Qwen3.5 MoE VLM — hybrid GatedDeltaNet + full attention with MRoPE,
//  sparse MoE, and Qwen3-VL vision tower.
//
//  Self-contained in MLXVLM (no MLXLLM dependency). GatedDeltaNet uses
//  pure-MLX sequential scan instead of the Metal kernel in MLXLLM.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen3_5MoEVLConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_5_moe"
    var textConfig: Qwen3_5MoEVLTextConfiguration
    var visionConfig: Qwen3VLConfiguration.VisionConfiguration

    // Token IDs (defaults match Qwen3-VL)
    private var _imageTokenId: Int?
    var imageTokenId: Int { _imageTokenId ?? 151_655 }
    private var _videoTokenId: Int?
    var videoTokenId: Int { _videoTokenId ?? 151_656 }
    private var _imageTokenIndex: Int?
    var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private var _videoTokenIndex: Int?
    var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private var _visionStartTokenId: Int?
    var visionStartTokenId: Int { _visionStartTokenId ?? 151_652 }
    private var _visionEndTokenId: Int?
    var visionEndTokenId: Int { _visionEndTokenId ?? 151_653 }
    private var _visionTokenId: Int?
    var visionTokenId: Int { _visionTokenId ?? 151_654 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _visionTokenId = "vision_token_id"
    }
}

public struct Qwen3_5MoEVLTextConfiguration: Codable, Sendable {
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

    // RoPE
    var ropeTheta: Float = 100_000
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 262144

    // MRoPE
    var ropeScaling: Qwen3VLConfiguration.RoPEScaling? = nil

    // Optional
    var normTopkProb: Bool = true
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false

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
        case ropeScaling = "rope_scaling"
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
        try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
    }

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
        self.ropeScaling =
            try container.decodeIfPresent(Qwen3VLConfiguration.RoPEScaling.self, forKey: .ropeScaling)

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

// MARK: - RMSNormGated

private class Qwen3_5VLRMSNormGated: Module {
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

// MARK: - MLP

private class Qwen3_5VLMLP: Module, UnaryLayer {
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

// MARK: - GatedDeltaNet (with Metal kernel for fast inference)

private func vlManualRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(variance + eps)
}

private func vlComputeG(_ ALog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let result = MLX.exp(
        -MLX.exp(ALog.asType(.float32)) * softplus(a + dtBias)
    )
    return result.asType(ALog.dtype)
}

// MARK: - Metal Kernel for GatedDeltaNet

private func vlMakeGatedDeltaKernel(hasMask: Bool, vectorized: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let gComment: String
    let gSetup: String
    let gAccess: String
    let gAdvance: String

    if vectorized {
        gComment = "// g: [B, T, Hv, Dk]"
        gSetup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        gAccess = "g_[s_idx]"
        gAdvance = "g_ += Hv * Dk;"
    } else {
        gComment = "// g: [B, T, Hv]"
        gSetup = "auto g_ = g + b_idx * T * Hv;"
        gAccess = "g_[hv_idx]"
        gAdvance = "g_ += Hv;"
    }

    let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        \(gComment)
        \(gSetup)
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (\(maskSource)) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * \(gAccess);
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          \(gAdvance)
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    var suffix = ""
    if vectorized { suffix += "_vec" }
    if hasMask { suffix += "_mask" }

    return MLXFast.metalKernel(
        name: "vl_gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class VLGatedDeltaKernelManager: Sendable {
    static let shared = VLGatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelVec: MLXFast.MLXFastKernel?
    let kernelVecMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = vlMakeGatedDeltaKernel(hasMask: false, vectorized: false)
        kernelMasked = vlMakeGatedDeltaKernel(hasMask: true, vectorized: false)
        kernelVec = vlMakeGatedDeltaKernel(hasMask: false, vectorized: true)
        kernelVecMasked = vlMakeGatedDeltaKernel(hasMask: true, vectorized: true)
    }
}

private func vlGatedDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (B, T, Hk, Dk) = k.shape4
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype

    let kernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray]

    if g.ndim == 4 {
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask {
            kernel = VLGatedDeltaKernelManager.shared.kernelVecMasked
            inputs.append(mask)
        } else {
            kernel = VLGatedDeltaKernelManager.shared.kernelVec
        }
    } else {
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask {
            kernel = VLGatedDeltaKernelManager.shared.kernelMasked
            inputs.append(mask)
        } else {
            kernel = VLGatedDeltaKernelManager.shared.kernel
        }
    }

    guard let kernel else {
        // Metal kernel unavailable — fall back to pure MLX ops
        return vlGatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, inputType]
    )

    return (outputs[0], outputs[1])
}

private func vlGatedDeltaStepOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state

    let decay: MLXArray
    if g.ndim == 2 {
        decay = g[.ellipsis, .newAxis, .newAxis]
    } else {
        decay = g[.ellipsis, .newAxis, 0...]
    }
    var newState = state * decay

    let kvMem = (newState * k[.ellipsis, .newAxis, 0...]).sum(axis: -1)
    let delta = (v - kvMem) * beta[.ellipsis, .newAxis]
    newState = newState + k[.ellipsis, .newAxis, 0...] * delta[.ellipsis, .newAxis]
    let y = (newState * q[.ellipsis, .newAxis, 0...]).sum(axis: -1)

    if let mask {
        let expandedMask = expandedDimensions(
            expandedDimensions(
                expandedDimensions(mask, axis: 1),
                axis: 2),
            axis: 3)
        newState = which(expandedMask, newState, oldState)
    }

    return (y, newState)
}

private func vlGatedDeltaOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var currentState = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    let repeatFactor = Hv / Hk
    var q = q
    var k = k
    if repeatFactor > 1 {
        q = MLX.repeated(q, count: repeatFactor, axis: 2)
        k = MLX.repeated(k, count: repeatFactor, axis: 2)
    }

    var ys: [MLXArray] = []
    for t in 0 ..< T {
        let stepMask: MLXArray? = mask != nil ? mask![0..., t] : nil
        let (y, newState) = vlGatedDeltaStepOps(
            q: q[0..., t],
            k: k[0..., t],
            v: v[0..., t],
            g: g[0..., t],
            beta: beta[0..., t],
            state: currentState,
            mask: stepMask
        )
        currentState = newState
        ys.append(y)
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, currentState)
}

private func vlGatedDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray,
    a: MLXArray, b: MLXArray,
    ALog: MLXArray, dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = vlComputeG(ALog, a, dtBias)

    var currentState = state
    if currentState == nil {
        let B = q.dim(0)
        let Hk = q.dim(2)
        let Dk = q.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        currentState = MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)
    }

    // Use Metal kernel for fast inference; fall back to ops if kernel unavailable
    return vlGatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: currentState!, mask: mask)
}

private class Qwen3_5VLGatedDeltaNet: Module {
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

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen3_5VLRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen3_5MoEVLTextConfiguration) {
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

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = MLX.log(MLXArray.ones([numVHeads]) * 8.0)

        _norm.wrappedValue = Qwen3_5VLRMSNormGated(
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

        if let cache {
            let nKeep = convKernelSize - 1
            cache[0] = convInput[0..., (convInput.dim(1) - nKeep)...]
        }

        let convOut = silu(conv1d(convInput))

        let convSplits = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = convSplits[0].reshaped(B, S, numKHeads, headKDim)
        var k = convSplits[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplits[2].reshaped(B, S, numVHeads, headVDim)

        let invScale = pow(Float(headKDim), -0.5)
        q = (invScale * invScale) * vlManualRmsNorm(q, eps: 1e-6)
        k = invScale * vlManualRmsNorm(k, eps: 1e-6)

        let state: MLXArray? = cache?[1]

        let (out, newState) = vlGatedDeltaUpdate(
            q: q, k: k, v: v,
            a: a, b: b,
            ALog: aLog, dtBias: dtBias,
            state: state, mask: mask
        )

        if let cache {
            cache[1] = newState
        }

        let normed = norm(out, gate: z)
        return outProj(normed.reshaped(B, S, -1))
    }
}

// MARK: - Full Attention with MRoPE

private class Qwen3_5VLAttention: Module {
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

    let rotaryEmbedding: Qwen3VLLanguage.RotaryEmbedding

    init(_ args: Qwen3_5MoEVLTextConfiguration) {
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

        rotaryEmbedding = Qwen3VLLanguage.RotaryEmbedding(
            headDim: ropeDims,
            base: Double(args.ropeTheta),
            ropeScaling: args.ropeScaling)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

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

        // MRoPE: compute position IDs and apply multimodal rotary embedding
        var kvSequenceLength = keys.dim(-2)
        var positionIds = positionIds

        if positionIds == nil {
            let offset = cache?.offset ?? 0
            kvSequenceLength += offset + 1
            var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
            base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
            positionIds = base[.newAxis, 0..., 0...]
            positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
        } else {
            if let cache {
                kvSequenceLength += cache.offset + 1
            }
        }

        let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

        // Apply MRoPE to only the rotary dimensions, leave the rest unchanged
        let ropeDims = cosValues.dim(-1)
        let fullDim = headDim

        if ropeDims < fullDim {
            let qRot = queries[0..., 0..., 0..., 0 ..< ropeDims]
            let qPass = queries[0..., 0..., 0..., ropeDims...]
            let kRot = keys[0..., 0..., 0..., 0 ..< ropeDims]
            let kPass = keys[0..., 0..., 0..., ropeDims...]

            let (qRotated, kRotated) = Qwen3VLLanguage.applyMultimodalRotary(
                q: qRot, k: kRot, cos: cosValues, sin: sinValues)

            queries = concatenated([qRotated, qPass], axis: -1)
            keys = concatenated([kRotated, kPass], axis: -1)
        } else {
            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)
        }

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
            attentionMask = .array(slicedMask)
        } else {
            attentionMask = .none
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        output = output * sigmoid(gate)

        return oProj(output)
    }
}

// MARK: - Sparse MoE with Shared Expert

private class Qwen3_5VLSparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3_5VLMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen3_5MoEVLTextConfiguration) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: numExperts
        )
        _sharedExpert.wrappedValue = Qwen3_5VLMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = MLX.softmax(gate(x), axis: -1, precise: true)

        let k = topK
        // Use negative kth to avoid negating the array (saves 1 dispatch per layer)
        let inds = MLX.argPartition(gates, kth: -k, axis: -1)[.ellipsis, (-k)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)

        if normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let moeOutput = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        let sharedY = sharedExpert(x)
        let gatedSharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return moeOutput + gatedSharedY
    }
}

// MARK: - Decoder Layer

private class Qwen3_5VLDecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen3_5VLGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3_5VLAttention?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: Qwen3_5MoEVLTextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen3_5VLGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen3_5VLAttention(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if args.numExperts > 0 {
            self.mlp = Qwen3_5VLSparseMoeBlock(args)
        } else {
            self.mlp = Qwen3_5VLMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        ssmMask: MLXArray?,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? ArraysCache)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
        }
        let h = x + r
        let out = h + mlp(postAttentionLayerNorm(h))
        return out
    }
}

// MARK: - Inner Model

private class Qwen3_5VLTextModelInner: Module {
    let args: Qwen3_5MoEVLTextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3_5VLDecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen3_5MoEVLTextConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { i in
            Qwen3_5VLDecoderLayer(args, layerIdx: i)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1
    }

    func callAsFunction(
        _ inputIds: MLXArray?,
        cache: [KVCache]?,
        inputEmbeddings: MLXArray?,
        mask: MLXArray?,
        positionIds: MLXArray?,
        visualMask: MLXArray?,
        deepstackEmbeds: [MLXArray]?
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings {
            h = inputEmbeddings
        } else if let inputIds {
            h = embedTokens(inputIds)
        } else {
            fatalError("Either input ids or embeddings must be provided")
        }

        let cache: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)

        let ssmMask = createSSMMask(h: h, cache: cache[ssmIdx] as? MambaCache)

        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, ssmMask: ssmMask, cache: cache[index], positionIds: positionIds)

            if let embeds = deepstackEmbeds, index < embeds.count,
                let visualMask
            {
                h = applyDeepstack(
                    hiddenStates: h,
                    visualMask: visualMask,
                    visualEmbeds: embeds[index])
            }
        }

        return norm(h)
    }

    private func applyDeepstack(
        hiddenStates: MLXArray,
        visualMask: MLXArray,
        visualEmbeds: MLXArray
    ) -> MLXArray {
        let indices = maskIndices(visualMask)
        guard !indices.isEmpty else { return hiddenStates }

        let indexArray = MLXArray(indices.map { UInt32($0) })

        let result = hiddenStates
        result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds

        return result
    }

    private func maskIndices(_ mask: MLXArray) -> [Int] {
        let bools = mask.asType(.bool).asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(bools.count)
        for (idx, value) in bools.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }
}

// MARK: - Text Model (language_model level)

private class Qwen3_5VLTextModel: Module, KVCacheDimensionProvider {
    let args: Qwen3_5MoEVLTextConfiguration
    let config: Qwen3_5MoEVLConfiguration
    var kvHeads: [Int]

    @ModuleInfo(key: "model") var model: Qwen3_5VLTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    private var ropeDeltas: MLXArray? = nil

    init(_ config: Qwen3_5MoEVLConfiguration) {
        let args = config.textConfig
        self.args = args
        self.config = config
        self.kvHeads = (0 ..< args.hiddenLayers).map { i in
            let isLinear = (i + 1) % args.fullAttentionInterval != 0
            return isLinear ? 0 : args.kvHeads
        }
        _model.wrappedValue = Qwen3_5VLTextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(
        _ inputIds: MLXArray?,
        cache: [KVCache]?,
        inputEmbeddings: MLXArray?,
        mask: MLXArray?,
        positionIds providedPositionIds: MLXArray?,
        visualMask: MLXArray?,
        deepstackEmbeds: [MLXArray]?,
        pixelValues: MLXArray?,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?
    ) -> LMOutput {
        if pixelValues != nil {
            ropeDeltas = nil
        }

        var positionIds = providedPositionIds

        if positionIds == nil && (mask == nil || mask?.ndim == 2) {
            if (cache?.first?.offset ?? 0) == 0 || ropeDeltas == nil || cache == nil {
                if let inputIds {
                    let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                        inputIds: inputIds,
                        imageGridTHW: imageGridTHW,
                        videoGridTHW: videoGridTHW,
                        spatialMergeSize: config.visionConfig.spatialMergeSize,
                        imageTokenId: config.imageTokenIndex,
                        videoTokenId: config.videoTokenIndex,
                        visionStartTokenId: config.visionStartTokenId,
                        attentionMask: mask)

                    positionIds = computed
                    ropeDeltas = deltas
                } else if let cache, ropeDeltas == nil {
                    let batch = inputEmbeddings!.dim(0)
                    let seqLength = inputEmbeddings!.dim(1)
                    let currentOffset = cache.first?.offset ?? 0

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                    let offsetValue = MLXArray(currentOffset).asType(.int32)
                    base = base + offsetValue

                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = tiled(positionIds!, repetitions: [3, batch, seqLength])
                }
            } else if let cache, let ropeDeltas {
                let batch = (inputIds ?? inputEmbeddings!).dim(0)
                let seqLength = (inputIds ?? inputEmbeddings!).dim(1)

                let lastCacheOffset = cache.last?.offset ?? 0

                var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                var base = MLXArray(0 ..< seqLength).asType(.int32)
                base = base[.newAxis, 0...]
                base = broadcast(base, to: [batch, seqLength])

                if delta.dim(0) == 1 && batch > 1 {
                    delta = repeated(delta, count: batch, axis: 0)
                }

                base = base + delta

                positionIds = base[.newAxis, 0..., 0...]
                positionIds = broadcast(positionIds!, to: [3, batch, seqLength])
            }
        }

        // For full attention layers, compute attention mask
        var attentionMask = mask
        if attentionMask == nil {
            // Find the full attention cache to compute mask from
            let faIdx = args.fullAttentionInterval - 1
            let faCache: [KVCache]? = cache != nil && faIdx < cache!.count ? [cache![faIdx]] : nil
            attentionMask = createAttentionMask(h: (inputIds ?? inputEmbeddings!), cache: faCache)
        }

        var output = model(
            inputIds,
            cache: cache,
            inputEmbeddings: inputEmbeddings,
            mask: attentionMask,
            positionIds: positionIds,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds)

        if let lmHead {
            output = lmHead(output)
        } else {
            output = model.embedTokens.asLinear(output)
        }

        return LMOutput(logits: output)
    }
}

// MARK: - Top-Level VLM

public final class Qwen3_5MoEVL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3_5VLTextModel

    public let config: Qwen3_5MoEVLConfiguration

    public init(_ config: Qwen3_5MoEVLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfig)
        _languageModel.wrappedValue = Qwen3_5VLTextModel(config)
    }

    public var vocabularySize: Int { config.textConfig.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let textArgs = config.textConfig
        return (0 ..< textArgs.hiddenLayers).map { i in
            let isLinear = (i + 1) % textArgs.fullAttentionInterval != 0
            if isLinear {
                return MambaCache() as any KVCache
            } else {
                return KVCacheSimple() as any KVCache
            }
        }
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = (imageMask .|| videoMask)

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            fatalError(
                "Feature token mismatch: expected \(nImageTokens) tokens, got \(nImageFeatures) features"
            )
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)

        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?
    ) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        let allFrames = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)

        if let pixelValues, !allFrames.isEmpty {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: allFrames)
            let mergeSize = config.visionConfig.spatialMergeSize
            let splits = allFrames.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(textEmbeds.dtype)

            let (mergedEmbeds, mask) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex)

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let splitIndices = cumulativeSplitIndices(from: splits)
                    let slices = layerFeatures.split(indices: splitIndices)
                    let concatenatedSlices = concatenated(slices).asType(textEmbeds.dtype)
                    return concatenatedSlices
                }
            }
        }

        let languageOutput = languageModel(
            inputIds,
            cache: cache,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIds: nil,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames)

        return .logits(languageOutput)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let result = languageModel(
            inputs,
            cache: cache,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        ).logits
        return result
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = weights
        let textArgs = config.textConfig

        // 1. Remap weight prefixes
        var remapped: [String: MLXArray] = [:]
        for (key, value) in sanitizedWeights {
            var newKey = key
            if key.hasPrefix("model.language_model.") {
                newKey = key.replacingOccurrences(
                    of: "model.language_model.", with: "language_model.model.")
            } else if key.hasPrefix("model.visual") {
                newKey = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
            } else if key.hasPrefix("lm_head") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if !key.hasPrefix("language_model.") && !key.hasPrefix("vision_tower.") {
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

        // 4. Vision patch_embed transpose
        for (key, value) in sanitizedWeights {
            if key.contains("patch_embed.proj.weight") && key.contains("vision_tower") {
                if value.ndim == 5 && value.dim(-1) != config.visionConfig.inChannels {
                    sanitizedWeights[key] = value.transposed(0, 2, 3, 4, 1)
                }
            }
        }

        // 5. Stack individual expert weights if present, split fused gate_up_proj
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

        // 6. Conv1d transpose and norm shift
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
            if k.contains("conv1d.weight"), v.ndim == 3, v.shape.last != 1 {
                sanitizedWeights[k] = v.swappedAxes(1, 2)
            }
            if shouldShiftNorms,
                normSuffixes.contains(where: { k.hasSuffix($0) }),
                v.ndim == 1
            {
                sanitizedWeights[k] = v + 1.0
            }
        }

        // 7. Filter position_ids from vision
        sanitizedWeights = sanitizedWeights.filter {
            !$0.key.contains("position_ids")
        }

        return sanitizedWeights
    }
}

// MARK: - LoRA

extension Qwen3_5MoEVL: LoRAModel {}
