//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G (decay factor)

/// Compute gating decay factor: exp(-exp(A_log) * softplus(a + dt_bias))
func computeG(_ ALog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let result = MLX.exp(
        -MLX.exp(ALog.asType(.float32)) * softplus(a + dtBias)
    )
    return result.asType(ALog.dtype)
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool, vectorized: Bool) -> MLXFast.MLXFastKernel? {
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
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

// MARK: - Kernel Manager (Singleton)

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelVec: MLXFast.MLXFastKernel?
    let kernelVecMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false, vectorized: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true, vectorized: false)
        kernelVec = makeGatedDeltaKernel(hasMask: false, vectorized: true)
        kernelVecMasked = makeGatedDeltaKernel(hasMask: true, vectorized: true)
    }
}

// MARK: - Ops-Based Fallback (Single Step)

/// Single recurrent step using array operations.
///
/// - q, k: [B, H, Dk]
/// - v: [B, H, Dv]
/// - g: [B, H] or [B, H, Dk]
/// - beta: [B, H]
/// - state: [B, H, Dv, Dk]
private func gatedDeltaStepOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state

    // Decay
    let decay: MLXArray
    if g.ndim == 2 {
        decay = g[.ellipsis, .newAxis, .newAxis]
    } else {
        // g.ndim == 3: [B, H, Dk]
        decay = g[.ellipsis, .newAxis, 0...]
    }
    var newState = state * decay

    // kv_mem = sum(state * k[..., None, :], axis=-1) -> [B, H, Dv]
    let kvMem = (newState * k[.ellipsis, .newAxis, 0...]).sum(axis: -1)

    // delta = (v - kv_mem) * beta[..., None] -> [B, H, Dv]
    let delta = (v - kvMem) * beta[.ellipsis, .newAxis]

    // state = state + k[..., None, :] * delta[..., None]
    newState = newState + k[.ellipsis, .newAxis, 0...] * delta[.ellipsis, .newAxis]

    // y = sum(state * q[..., None, :], axis=-1) -> [B, H, Dv]
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

// MARK: - Ops-Based Loop (Prefill)

/// Multi-token ops-based loop for prompt prefill.
///
/// - q, k: [B, T, Hk, Dk]
/// - v: [B, T, Hv, Dv]
/// - g: [B, T, Hv] or [B, T, Hv, Dk]
/// - beta: [B, T, Hv]
/// - state: [B, Hv, Dv, Dk]
func gatedDeltaOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (B, T, Hk, Dk) = q.shape4
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
        let (y, newState) = gatedDeltaStepOps(
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

// MARK: - Metal Kernel Dispatch

/// Dispatch gated delta recurrence to Metal kernel.
func gatedDeltaKernel(
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
        // Vectorized gating
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask {
            kernel = GatedDeltaKernelManager.shared.kernelVecMasked
            inputs.append(mask)
        } else {
            kernel = GatedDeltaKernelManager.shared.kernelVec
        }
    } else {
        // Scalar gating
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask {
            kernel = GatedDeltaKernelManager.shared.kernelMasked
            inputs.append(mask)
        } else {
            kernel = GatedDeltaKernelManager.shared.kernel
        }
    }

    guard let kernel else {
        fatalError("Gated delta Metal kernel not available")
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

// MARK: - Entry Point

/// Main entry point for gated delta recurrence.
///
/// Computes beta and g from raw inputs, then dispatches to kernel or ops.
///
/// - Parameters:
///   - q, k: [B, T, Hk, Dk]
///   - v: [B, T, Hv, Dv]
///   - a, b: [B, T, Hv] raw gating/beta inputs
///   - ALog: [Hv] log of decay constants
///   - dtBias: [Hv] bias for gating
///   - state: [B, Hv, Dv, Dk] recurrent state (nil on first call)
///   - mask: [B, T] optional SSM mask
///   - useKernel: whether to use Metal kernel (false for training)
func gatedDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray,
    a: MLXArray, b: MLXArray,
    ALog: MLXArray, dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    useKernel: Bool = true
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = computeG(ALog, a, dtBias)

    var currentState = state
    if currentState == nil {
        let (B, _, Hk, Dk) = q.shape4
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        currentState = MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)
    }

    if !useKernel {
        return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: currentState, mask: mask)
    }

    return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: currentState!, mask: mask)
}
