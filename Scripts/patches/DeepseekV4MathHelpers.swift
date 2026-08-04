// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pure-math building blocks for the DeepSeek-V4 forward pass.
// Each helper is pure (no module state, no cache) so it's trivially
// unit-testable with synthetic tensors.
//
// Reference:
//   - `jang/research/DSV4-RUNTIME-ARCHITECTURE.md` §2 (per-layer forward)
//   - `jang-tools/jang_tools/dsv4_prune/mlx_model.py` —
//       * `_hc_split_sinkhorn_ops` (lines 79-110)
//       * `_apply_partial_rope` (lines 355-362)
//       * `_dsv4_swiglu` (lines 799-814)
//       * `sqrtsoftplus_select` (lines 736-757)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private let _deepseekV4SwiGLUClampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, limit in
    let outputType = gate.dtype
    let gateF32 = minimum(gate.asType(.float32), limit)
    let upF32 = minimum(maximum(up.asType(.float32), -limit), limit)
    return (silu(gateF32) * upF32).asType(outputType)
}

private let _deepseekV4SwiGLUUnclampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, _ in
    let outputType = gate.dtype
    return (silu(gate.asType(.float32)) * up.asType(.float32)).asType(outputType)
}

private let _deepseekV4ScoredSwiGLUClampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, scores, limit in
    let outputType = gate.dtype
    let gateF32 = minimum(gate.asType(.float32), limit)
    let upF32 = minimum(maximum(up.asType(.float32), -limit), limit)
    let scoreF32 = scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
    return (silu(gateF32) * upF32 * scoreF32).asType(outputType)
}

private let _deepseekV4ScoredSwiGLUUnclampedBody:
    @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
{ gate, up, scores, _ in
    let outputType = gate.dtype
    let scoreF32 = scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
    return (silu(gate.asType(.float32)) * up.asType(.float32) * scoreF32)
        .asType(outputType)
}

private let _deepseekV4ScoredSwiGLUClampedArrayBody:
    @Sendable ([MLXArray]) -> [MLXArray] =
{ args in
    [_deepseekV4ScoredSwiGLUClampedBody(args[0], args[1], args[2], args[3])]
}

private let _deepseekV4ScoredSwiGLUUnclampedArrayBody:
    @Sendable ([MLXArray]) -> [MLXArray] =
{ args in
    [_deepseekV4ScoredSwiGLUUnclampedBody(args[0], args[1], args[2], args[3])]
}

// The authoritative affine runtime compiles this exact stateless activation.
// Keep it separate from whole-model compiled decode: it captures no model or
// cache state, and the wrapper below avoids an illegal nested compile trace.
private let _compiledDeepseekV4SwiGLUClamped =
    compile(shapeless: true, _deepseekV4SwiGLUClampedBody)
private let _compiledDeepseekV4SwiGLUUnclamped =
    compile(shapeless: true, _deepseekV4SwiGLUUnclampedBody)
private let _compiledDeepseekV4ScoredSwiGLUClamped =
    compile(shapeless: true, _deepseekV4ScoredSwiGLUClampedArrayBody)
private let _compiledDeepseekV4ScoredSwiGLUUnclamped =
    compile(shapeless: true, _deepseekV4ScoredSwiGLUUnclampedArrayBody)

public enum DeepseekV4Math {
    public static let fusedHC4Enabled =
        ProcessInfo.processInfo.environment["VMLX_DSV4_FUSED_HC4"] != "0"
    public static let fusedHCNormEnabled =
        ProcessInfo.processInfo.environment["VMLX_DSV4_FUSED_HC_NORM"] == "1"

    private static let fusedSqrtSoftplusTopKKernel = MLXFast.metalKernel(
        name: "deepseek_v4_sqrtsoftplus_topk",
        inputNames: ["logits", "bias", "scale"],
        outputNames: ["indices", "weights"],
        source: """
            const uint lane = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.x;
            threadgroup float original[NEXPERTS];
            threadgroup float ranked[NEXPERTS];

            if (lane < NEXPERTS) {
                const uint offset = row * NEXPERTS + lane;
                const float value = static_cast<float>(logits[offset]);
                const float magnitude = metal::abs(value);
                const float softplus = metal::max(value, 0.0f)
                    + metal::fast::log(1.0f + metal::fast::exp(-magnitude));
                const float score = metal::sqrt(softplus);
                original[lane] = score;
                ranked[lane] = score + static_cast<float>(bias[lane]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                float selected_scores[TOPK];
                int selected_indices[TOPK];
                for (int slot = 0; slot < TOPK; ++slot) {
                    float best = -INFINITY;
                    int best_index = 0;
                    for (int expert = 0; expert < NEXPERTS; ++expert) {
                        const float candidate = ranked[expert];
                        if (candidate > best) {
                            best = candidate;
                            best_index = expert;
                        }
                    }
                    selected_indices[slot] = best_index;
                    selected_scores[slot] = original[best_index];
                    ranked[best_index] = -INFINITY;
                }

                float denominator = 0.0f;
                if (NORMALIZE != 0) {
                    for (int slot = 0; slot < TOPK; ++slot) {
                        denominator += selected_scores[slot];
                    }
                    denominator += 1.0e-20f;
                }
                const float route_scale = static_cast<float>(scale[0]);
                const uint output_base = row * TOPK;
                for (int slot = 0; slot < TOPK; ++slot) {
                    indices[output_base + slot] = selected_indices[slot];
                    const float normalized = NORMALIZE != 0
                        ? selected_scores[slot] / denominator
                        : selected_scores[slot];
                    weights[output_base + slot] = normalized * route_scale;
                }
            }
        """)

    private static let e4m3KVActivationRoundTripKernel = MLXFast.metalKernel(
        name: "deepseek_v4_e4m3_kv_activation_roundtrip",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint group = gid >> 6;
            const uint block = group % NBT;
            const uint row = group / NBT;
            const uint idx = row * N + block * 64 + lane;

            if (block >= NBQ) {
                y[idx] = static_cast<outT>(x[idx]);
            } else {
                threadgroup float scratch[64];
                const float input_value = static_cast<float>(x[idx]);
                scratch[lane] = metal::abs(input_value);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = 32; stride > 0; stride >>= 1) {
                    if (lane < stride) {
                        scratch[lane] = metal::max(
                            scratch[lane], scratch[lane + stride]);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                const float amax = metal::max(scratch[0], 1.0e-4f);
                const float raw_scale = amax / 448.0f;
                const uint raw_bits = as_type<uint>(raw_scale);
                const int raw_exp = int((raw_bits >> 23) & 0xffu) - 127;
                const bool has_mantissa = (raw_bits & 0x7fffffu) != 0u;
                const int scale_exp = raw_exp + int(has_mantissa);
                const float scale = as_type<float>(uint(scale_exp + 127) << 23);

                const float normalized = metal::clamp(
                    input_value / scale, -448.0f, 448.0f);
                const float sign = normalized < 0.0f ? -1.0f : 1.0f;
                const float absolute = metal::min(metal::abs(normalized), 448.0f);
                int low = 0;
                int high = 126;
                while (low < high) {
                    const int middle = (low + high + 1) >> 1;
                    const int exponent = (middle >> 3) & 0x0f;
                    const int mantissa = middle & 0x07;
                    const float candidate = exponent == 0
                        ? float(mantissa) * 0.001953125f
                        : (1.0f + float(mantissa) * 0.125f)
                            * metal::fast::exp2(float(exponent - 7));
                    if (candidate <= absolute) low = middle;
                    else high = middle - 1;
                }

                int best = low;
                const int best_exponent = (best >> 3) & 0x0f;
                const int best_mantissa = best & 0x07;
                float best_value = best_exponent == 0
                    ? float(best_mantissa) * 0.001953125f
                    : (1.0f + float(best_mantissa) * 0.125f)
                        * metal::fast::exp2(float(best_exponent - 7));
                if (best < 126) {
                    const int next = best + 1;
                    const int next_exponent = (next >> 3) & 0x0f;
                    const int next_mantissa = next & 0x07;
                    const float next_value = next_exponent == 0
                        ? float(next_mantissa) * 0.001953125f
                        : (1.0f + float(next_mantissa) * 0.125f)
                            * metal::fast::exp2(float(next_exponent - 7));
                    const float best_diff = metal::abs(absolute - best_value);
                    const float next_diff = metal::abs(absolute - next_value);
                    if (next_diff < best_diff ||
                        (next_diff == best_diff && (next & 1) == 0 && (best & 1) != 0)) {
                        best_value = next_value;
                    }
                }
                y[idx] = static_cast<outT>(sign * best_value * scale);
            }
        """)

    private static let indexerActivationRoundTripKernel = MLXFast.metalKernel(
        name: "deepseek_v4_indexer_hadamard128_e2m1_roundtrip",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint row = gid >> 7;
            const uint idx = row * 128 + lane;
            threadgroup float values[128];
            threadgroup float magnitudes[128];

            values[lane] = static_cast<float>(x[idx]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < 128; stride <<= 1) {
                if (lane < 64) {
                    const uint block = lane / stride;
                    const uint offset = lane % stride;
                    const uint low_idx = block * 2 * stride + offset;
                    const uint high_idx = low_idx + stride;
                    const float low = values[low_idx];
                    const float high = values[high_idx];
                    values[low_idx] = low + high;
                    values[high_idx] = low - high;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float rotated = values[lane] * 0.08838834764831845f;
            magnitudes[lane] = metal::abs(rotated);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 16; stride > 0; stride >>= 1) {
                if ((lane & 31u) < stride) {
                    magnitudes[lane] = metal::max(
                        magnitudes[lane], magnitudes[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint block_start = lane & ~31u;
            const float amax = metal::max(
                magnitudes[block_start], 7.052966104933725e-38f);
            const float raw_scale = amax / 6.0f;
            const uint raw_bits = as_type<uint>(raw_scale);
            const int raw_exp = int((raw_bits >> 23) & 0xffu) - 127;
            const bool has_mantissa = (raw_bits & 0x7fffffu) != 0u;
            const int scale_exp = raw_exp + int(has_mantissa);
            const float scale = as_type<float>(uint(scale_exp + 127) << 23);

            const float normalized = metal::clamp(rotated / scale, -6.0f, 6.0f);
            const float absolute = metal::abs(normalized);
            constexpr float codebook[8] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
            };
            int best = 0;
            float best_diff = absolute;
            for (int code = 1; code < 8; ++code) {
                const float diff = metal::abs(absolute - codebook[code]);
                if (diff < best_diff ||
                    (diff == best_diff && (code & 1) == 0 && (best & 1) != 0)) {
                    best = code;
                    best_diff = diff;
                }
            }
            const float sign = normalized < 0.0f ? -1.0f : 1.0f;
            y[idx] = static_cast<outT>(sign * codebook[best] * scale);
        """)

    /// Apply the pinned 0731 post-RoPE KV activation-QAT graph while copying
    /// the RoPE suffix exactly. `ropeDim` and the non-RoPE prefix are both
    /// 64-aligned in the released model (512 = 448 + 64).
    public static func e4m3KVActivationRoundTrip(
        _ x: MLXArray, ropeDim: Int
    ) -> MLXArray {
        let width = x.dim(-1)
        let nopeWidth = width - ropeDim
        precondition(
            width > 0 && ropeDim >= 0 && nopeWidth >= 0
                && width.isMultiple(of: 64) && nopeWidth.isMultiple(of: 64),
            "DSV4 E4M3 KV QAT requires 64-aligned full and non-RoPE widths")
        if nopeWidth == 0 { return x }
        let input = contiguous(x)
        let rows = input.size / width
        return e4m3KVActivationRoundTripKernel(
            [input],
            template: [
                ("N", width),
                ("NBQ", nopeWidth / 64),
                ("NBT", width / 64),
                ("outT", input.dtype),
            ],
            grid: (rows * width, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
    }

    /// Pinned 0731 indexer activation graph: normalized Hadamard-128 followed
    /// by block-32 power-of-two E2M1 fake quantization.
    public static func indexerActivationRoundTrip(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) == 128, "DSV4 indexer QAT requires 128-wide rows")
        let input = contiguous(x)
        let rows = input.size / 128
        return indexerActivationRoundTripKernel(
            [input],
            template: [("outT", input.dtype)],
            grid: (rows * 128, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
    }

    /// Routed expert outputs already include their per-route weights before
    /// the quantized down projection. Accumulate the expert axis in fp32; the
    /// caller adds the shared expert in fp32 and casts only once afterward.
    public static func reduceRoutedExpertsFP32(_ routed: MLXArray) -> MLXArray {
        routed.asType(.float32).sum(axis: -2)
    }

    public static func addSharedExpertFP32(
        _ routed: MLXArray, shared: MLXArray, outputDType: DType
    ) -> MLXArray {
        (routed.asType(.float32) + shared.asType(.float32)).asType(outputDType)
    }

    /// Official 0731 hyper-connection expansion. Keep the broadcast and
    /// reduction axes identical to the working MLX reference:
    /// `(comb[..., None] * residual[..., :, None, :]).sum(axis: -3)`.
    public static func hcExpandResidual(
        comb: MLXArray, residual: MLXArray
    ) -> MLXArray {
        (
            comb.asType(.float32).expandedDimensions(axis: -1)
                * residual.asType(.float32).expandedDimensions(axis: -2)
        ).sum(axis: -3)
    }

    // MARK: - Fused mHC split-Sinkhorn
    //
    // DSV4 executes this twice per transformer layer. Expressing twenty
    // Sinkhorn iterations as ordinary MLX ops creates more than forty graph
    // nodes per call (roughly 3,400 nodes per token across 43 layers). The
    // reference DSV4 MLX runtime uses one Metal dispatch instead. Keep the
    // same fp32 arithmetic and exact normalization order here.
    private static let hcSplitSinkhornKernel = MLXFast.metalKernel(
        name: "deepseek_v4_hc_split_sinkhorn",
        inputNames: ["mixes", "scale", "base", "eps"],
        outputNames: ["pre", "post", "comb"],
        source: """
            uint idx = thread_position_in_grid.x;
            constexpr int MIX = (2 + HC) * HC;
            float epsv = static_cast<float>(eps[0]);

            auto mix = mixes + idx * MIX;
            auto pre_out = pre + idx * HC;
            auto post_out = post + idx * HC;
            auto comb_out = comb + idx * HC * HC;

            float pre_scale = static_cast<float>(scale[0]);
            float post_scale = static_cast<float>(scale[1]);
            float comb_scale = static_cast<float>(scale[2]);

            for (int i = 0; i < HC; ++i) {
                float z = static_cast<float>(mix[i]) * pre_scale
                    + static_cast<float>(base[i]);
                pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
            }
            for (int i = 0; i < HC; ++i) {
                int off = HC + i;
                float z = static_cast<float>(mix[off]) * post_scale
                    + static_cast<float>(base[off]);
                post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
            }

            float c[HC * HC];
            for (int i = 0; i < HC; ++i) {
                float row_max = -INFINITY;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    int off = 2 * HC + cidx;
                    float v = static_cast<float>(mix[off]) * comb_scale
                        + static_cast<float>(base[off]);
                    c[cidx] = v;
                    row_max = metal::max(row_max, v);
                }
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    float v = metal::fast::exp(c[cidx] - row_max);
                    c[cidx] = v;
                    row_sum += v;
                }
                float inv_sum = 1.0f / row_sum;
                for (int j = 0; j < HC; ++j) {
                    int cidx = i * HC + j;
                    c[cidx] = c[cidx] * inv_sum + epsv;
                }
            }

            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }

            for (int iter = 1; iter < ITERS; ++iter) {
                for (int i = 0; i < HC; ++i) {
                    float row_sum = 0.0f;
                    for (int j = 0; j < HC; ++j) {
                        row_sum += c[i * HC + j];
                    }
                    float inv_denom = 1.0f / (row_sum + epsv);
                    for (int j = 0; j < HC; ++j) {
                        c[i * HC + j] *= inv_denom;
                    }
                }
                for (int j = 0; j < HC; ++j) {
                    float col_sum = 0.0f;
                    for (int i = 0; i < HC; ++i) {
                        col_sum += c[i * HC + j];
                    }
                    float inv_denom = 1.0f / (col_sum + epsv);
                    for (int i = 0; i < HC; ++i) {
                        c[i * HC + j] *= inv_denom;
                    }
                }
            }

            for (int i = 0; i < HC * HC; ++i) {
                comb_out[i] = c[i];
            }
        """)

    /// Decode-oriented HC=4 fusion used by the native MLX path. One
    /// threadgroup computes a token's Sinkhorn coefficients and immediately
    /// consumes the pre weights to collapse its four residual streams. This
    /// avoids materializing `pre` and removes the broadcast/multiply/reduce
    /// graph that otherwise follows every HC split.
    private static let hcSplitSinkhornCollapse4Kernel = MLXFast.metalKernel(
        name: "deepseek_v4_hc_split_sinkhorn_collapse4",
        inputNames: ["mixes", "scale", "base", "residual", "eps"],
        outputNames: ["post", "comb", "collapsed"],
        source: """
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint ntg = threads_per_threadgroup.x;
            threadgroup float pre_shared[4];

            auto mix = mixes + row * 24;
            auto post_out = post + row * 4;
            auto comb_out = comb + row * 16;
            float epsv = static_cast<float>(eps[0]);

            if (tid == 0) {
                float pre_scale = static_cast<float>(scale[0]);
                float post_scale = static_cast<float>(scale[1]);
                float comb_scale = static_cast<float>(scale[2]);

                for (int i = 0; i < 4; ++i) {
                    float z = static_cast<float>(mix[i]) * pre_scale
                        + static_cast<float>(base[i]);
                    pre_shared[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
                }
                for (int i = 0; i < 4; ++i) {
                    int off = 4 + i;
                    float z = static_cast<float>(mix[off]) * post_scale
                        + static_cast<float>(base[off]);
                    post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
                }

                float c[16];
                for (int i = 0; i < 4; ++i) {
                    float row_max = -INFINITY;
                    for (int j = 0; j < 4; ++j) {
                        int cidx = i * 4 + j;
                        int off = 8 + cidx;
                        float v = static_cast<float>(mix[off]) * comb_scale
                            + static_cast<float>(base[off]);
                        c[cidx] = v;
                        row_max = metal::max(row_max, v);
                    }
                    float row_sum = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        int cidx = i * 4 + j;
                        float v = metal::fast::exp(c[cidx] - row_max);
                        c[cidx] = v;
                        row_sum += v;
                    }
                    float inv_sum = 1.0f / row_sum;
                    for (int j = 0; j < 4; ++j) {
                        int cidx = i * 4 + j;
                        c[cidx] = c[cidx] * inv_sum + epsv;
                    }
                }

                for (int j = 0; j < 4; ++j) {
                    float col_sum = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        col_sum += c[i * 4 + j];
                    }
                    float inv_denom = 1.0f / (col_sum + epsv);
                    for (int i = 0; i < 4; ++i) {
                        c[i * 4 + j] *= inv_denom;
                    }
                }

                for (int iter = 1; iter < ITERS; ++iter) {
                    for (int i = 0; i < 4; ++i) {
                        float row_sum = 0.0f;
                        for (int j = 0; j < 4; ++j) {
                            row_sum += c[i * 4 + j];
                        }
                        float inv_denom = 1.0f / (row_sum + epsv);
                        for (int j = 0; j < 4; ++j) {
                            c[i * 4 + j] *= inv_denom;
                        }
                    }
                    for (int j = 0; j < 4; ++j) {
                        float col_sum = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            col_sum += c[i * 4 + j];
                        }
                        float inv_denom = 1.0f / (col_sum + epsv);
                        for (int i = 0; i < 4; ++i) {
                            c[i * 4 + j] *= inv_denom;
                        }
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    comb_out[i] = c[i];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto x = residual + row * 4 * D;
            auto y = collapsed + row * D;
            for (uint d = tid; d < D; d += ntg) {
                float value = static_cast<float>(x[d]) * pre_shared[0];
                value += static_cast<float>(x[D + d]) * pre_shared[1];
                value += static_cast<float>(x[2 * D + d]) * pre_shared[2];
                value += static_cast<float>(x[3 * D + d]) * pre_shared[3];
                y[d] = value;
            }
        """)

    /// Decode-only HC=4 collapse followed by weighted RMSNorm. This mirrors
    /// DeepSeek's operation order while retaining the collapsed row for
    /// numerical diagnostics. The row stays in threadgroup memory between
    /// collapse and normalization, removing a device round-trip and a second
    /// Metal dispatch.
    private static let hcSplitSinkhornCollapseNorm4Kernel = MLXFast.metalKernel(
        name: "deepseek_v4_hc_split_sinkhorn_collapse_norm4",
        inputNames: [
            "mixes", "scale", "base", "residual", "norm_weight",
            "hc_eps", "norm_eps",
        ],
        outputNames: ["post", "comb", "collapsed", "normalized"],
        source: """
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint ntg = threads_per_threadgroup.x;
            uint simd_index = simdgroup_index_in_threadgroup;
            uint simd_lane = thread_index_in_simdgroup;
            threadgroup float row_shared[D];
            threadgroup float pre_shared[4];
            threadgroup float simd_sums[32];
            threadgroup float inv_rms_shared[1];

            auto mix = mixes + row * 24;
            auto post_out = post + row * 4;
            auto comb_out = comb + row * 16;
            float epsv = static_cast<float>(hc_eps[0]);

            if (tid == 0) {
                float pre_scale = static_cast<float>(scale[0]);
                float post_scale = static_cast<float>(scale[1]);
                float comb_scale = static_cast<float>(scale[2]);

                for (int i = 0; i < 4; ++i) {
                    float z = static_cast<float>(mix[i]) * pre_scale
                        + static_cast<float>(base[i]);
                    pre_shared[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
                }
                for (int i = 0; i < 4; ++i) {
                    int off = 4 + i;
                    float z = static_cast<float>(mix[off]) * post_scale
                        + static_cast<float>(base[off]);
                    post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
                }

                float c[16];
                for (int i = 0; i < 4; ++i) {
                    float row_max = -INFINITY;
                    for (int j = 0; j < 4; ++j) {
                        int cidx = i * 4 + j;
                        int off = 8 + cidx;
                        float v = static_cast<float>(mix[off]) * comb_scale
                            + static_cast<float>(base[off]);
                        c[cidx] = v;
                        row_max = metal::max(row_max, v);
                    }
                    float row_sum = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        int cidx = i * 4 + j;
                        float v = metal::fast::exp(c[cidx] - row_max);
                        c[cidx] = v;
                        row_sum += v;
                    }
                    float inv_sum = 1.0f / row_sum;
                    for (int j = 0; j < 4; ++j) {
                        int cidx = i * 4 + j;
                        c[cidx] = c[cidx] * inv_sum + epsv;
                    }
                }

                for (int j = 0; j < 4; ++j) {
                    float col_sum = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        col_sum += c[i * 4 + j];
                    }
                    float inv_denom = 1.0f / (col_sum + epsv);
                    for (int i = 0; i < 4; ++i) {
                        c[i * 4 + j] *= inv_denom;
                    }
                }

                for (int iter = 1; iter < ITERS; ++iter) {
                    for (int i = 0; i < 4; ++i) {
                        float row_sum = 0.0f;
                        for (int j = 0; j < 4; ++j) {
                            row_sum += c[i * 4 + j];
                        }
                        float inv_denom = 1.0f / (row_sum + epsv);
                        for (int j = 0; j < 4; ++j) {
                            c[i * 4 + j] *= inv_denom;
                        }
                    }
                    for (int j = 0; j < 4; ++j) {
                        float col_sum = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            col_sum += c[i * 4 + j];
                        }
                        float inv_denom = 1.0f / (col_sum + epsv);
                        for (int i = 0; i < 4; ++i) {
                            c[i * 4 + j] *= inv_denom;
                        }
                    }
                }
                for (int i = 0; i < 16; ++i) {
                    comb_out[i] = c[i];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto x = residual + row * 4 * D;
            float sum_squares = 0.0f;
            for (uint d = tid; d < D; d += ntg) {
                float value = static_cast<float>(x[d]) * pre_shared[0];
                value += static_cast<float>(x[D + d]) * pre_shared[1];
                value += static_cast<float>(x[2 * D + d]) * pre_shared[2];
                value += static_cast<float>(x[3 * D + d]) * pre_shared[3];
                row_shared[d] = value;
                collapsed[row * D + d] = value;
                sum_squares += value * value;
            }

            sum_squares = simd_sum(sum_squares);
            if (simd_lane == 0) {
                simd_sums[simd_index] = sum_squares;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint simd_count = (ntg + 31) / 32;
            float total = tid < simd_count ? simd_sums[tid] : 0.0f;
            total = simd_sum(total);
            if (tid == 0) {
                float arg = total / static_cast<float>(D)
                    + static_cast<float>(norm_eps[0]);
                inv_rms_shared[0] = metal::precise::rsqrt(arg);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float inv_rms = inv_rms_shared[0];
            for (uint d = tid; d < D; d += ntg) {
                normalized[row * D + d] = row_shared[d] * inv_rms
                    * static_cast<float>(norm_weight[d]);
            }
        """)

    /// HC=4 expansion fusion. One thread handles one embedding coordinate and
    /// emits all four output streams, reusing the residual and block values.
    private static let hcExpand4Kernel = MLXFast.metalKernel(
        name: "deepseek_v4_hc_expand4_v2",
        inputNames: ["block", "residual", "post", "comb"],
        outputNames: ["expanded"],
        source: """
            uint gid = thread_position_in_grid.x;
            uint row = gid / D;
            uint d = gid - row * D;
            float block_value = static_cast<float>(block[row * D + d]);
            auto x = residual + row * 4 * D;
            float r0 = static_cast<float>(x[d]);
            float r1 = static_cast<float>(x[D + d]);
            float r2 = static_cast<float>(x[2 * D + d]);
            float r3 = static_cast<float>(x[3 * D + d]);
            auto p = post + row * 4;
            auto c = comb + row * 16;
            auto y = expanded + row * 4 * D;

            for (int dst = 0; dst < 4; ++dst) {
                float value = block_value * static_cast<float>(p[dst]);
                value += static_cast<float>(c[dst]) * r0;
                value += static_cast<float>(c[4 + dst]) * r1;
                value += static_cast<float>(c[8 + dst]) * r2;
                value += static_cast<float>(c[12 + dst]) * r3;
                y[dst * D + d] = value;
            }
        """)

    private static let scalarArrayLock = NSLock()
    nonisolated(unsafe) private static var scalarArrays: [Float: MLXArray] = [:]

    private static func scalarArray(_ value: Float) -> MLXArray {
        scalarArrayLock.lock()
        defer { scalarArrayLock.unlock() }
        if let cached = scalarArrays[value] { return cached }
        let array = MLXArray([value])
        scalarArrays[value] = array
        return array
    }

    // MARK: - Per-head Q RMSNorm ones cache
    //
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // The DSV4 attention applies a unit-weight RMSNorm-like rescale per
    // head before partial RoPE: `q * rsqrt((q^2).mean(-1) + eps)`. The
    // Python reference fuses this into a single `mx.fast.rms_norm` kernel
    // with a cached ones tensor. The cache is keyed on `(headDim, dtype)`
    // and shared across all 64 attention heads × 43 layers. NSLock-guarded
    // so concurrent SwitchGLU/attention forwards never race the cache.
    private static let qNormOnesLock = NSLock()
    nonisolated(unsafe) private static var qNormOnesCache: [String: MLXArray] = [:]

    public static func qNormOnes(headDim: Int, dtype: DType) -> MLXArray {
        let key = "\(headDim)|\(dtype)"
        qNormOnesLock.lock(); defer { qNormOnesLock.unlock() }
        if let w = qNormOnesCache[key] { return w }
        let w = MLXArray.ones([headDim], dtype: dtype)
        qNormOnesCache[key] = w
        return w
    }


    // MARK: - mHC split-Sinkhorn (collapse matrices)
    //
    // Given `mixes` of shape (..., 3*hcMult) and per-block scale/base
    // parameters, produce the three matrices needed by the HC collapse
    // kernel:
    //
    //   pre   = sigmoid(mixes * scale[0] + base[:hcMult]) + eps
    //           (no normalization — used to weight residual copies)
    //
    //   post  = 2 * sigmoid(mixes * scale[1] + base[hcMult:2*hcMult])
    //           (no eps — used to scale block output before add-back)
    //
    //   comb  = softmax(mixes * scale[2] + base[2*hcMult:3*hcMult], axis=-1) + eps
    //           col-normalize
    //           repeat (iters-1)× { row-normalize; col-normalize }
    //
    // `comb` is the sinkhorn doubly-stochastic mixing matrix that
    // preserves residual norm when used for the `expand` step.
    //
    // Shape contract:
    //   mixes: (..., 3*hcMult)
    //   scale: (3,)       one learned scalar per field
    //   base:  (3*hcMult,) learned bias concatenated across fields
    //   → pre:  (..., hcMult)
    //   → post: (..., hcMult)
    //   → comb: (..., hcMult, hcMult)
    public static func hcSplitSinkhorn(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        iters: Int = 20,
        eps: Float = 1e-6
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        if Device.defaultDevice().deviceType == .gpu {
            let leadShape = Array(mixes.shape.dropLast())
            let rows = mixes.size / ((2 + hcMult) * hcMult)
            let outputs = hcSplitSinkhornKernel(
                [mixes, scale, base, scalarArray(eps)],
                template: [("HC", hcMult), ("ITERS", iters)],
                grid: (rows, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [
                    leadShape + [hcMult],
                    leadShape + [hcMult],
                    leadShape + [hcMult, hcMult],
                ],
                outputDTypes: [.float32, .float32, .float32])
            return (pre: outputs[0], post: outputs[1], comb: outputs[2])
        }
        return hcSplitSinkhornOps(
            mixes: mixes,
            scale: scale,
            base: base,
            hcMult: hcMult,
            iters: iters,
            eps: eps)
    }

    public static func hcSplitSinkhornCollapse4(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        residual: MLXArray,
        hiddenSize: Int,
        iters: Int = 20,
        eps: Float = 1e-6
    ) -> (collapsed: MLXArray, post: MLXArray, comb: MLXArray) {
        let leadShape = Array(mixes.shape.dropLast())
        let rows = mixes.size / 24
        let outputs = hcSplitSinkhornCollapse4Kernel(
            [
                mixes.asType(.float32),
                scale.asType(.float32),
                base.asType(.float32),
                residual.asType(.float32),
                scalarArray(eps),
            ],
            template: [("D", hiddenSize), ("ITERS", iters)],
            grid: (rows * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [
                leadShape + [4],
                leadShape + [4, 4],
                leadShape + [hiddenSize],
            ],
            outputDTypes: [.float32, .float32, .float32])
        return (collapsed: outputs[2], post: outputs[0], comb: outputs[1])
    }

    public static func hcSplitSinkhornCollapseNorm4(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        residual: MLXArray,
        normWeight: MLXArray,
        normEps: Float,
        hiddenSize: Int,
        iters: Int = 20,
        hcEps: Float = 1e-6
    ) -> (collapsed: MLXArray, normalized: MLXArray, post: MLXArray, comb: MLXArray) {
        let leadShape = Array(mixes.shape.dropLast())
        let rows = mixes.size / 24
        let outputs = hcSplitSinkhornCollapseNorm4Kernel(
            [
                mixes.asType(.float32),
                scale.asType(.float32),
                base.asType(.float32),
                residual.asType(.float32),
                normWeight.asType(.float32),
                scalarArray(hcEps),
                scalarArray(normEps),
            ],
            template: [("D", hiddenSize), ("ITERS", iters)],
            grid: (rows * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [
                leadShape + [4],
                leadShape + [4, 4],
                leadShape + [hiddenSize],
                leadShape + [hiddenSize],
            ],
            outputDTypes: [.float32, .float32, .float32, .float32])
        return (
            collapsed: outputs[2],
            normalized: outputs[3],
            post: outputs[0],
            comb: outputs[1])
    }

    public static func hcExpand4(
        blockOut: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        comb: MLXArray,
        hiddenSize: Int
    ) -> MLXArray {
        let rows = blockOut.size / hiddenSize
        return hcExpand4Kernel(
            [
                contiguous(blockOut.asType(.float32)),
                contiguous(residual.asType(.float32)),
                contiguous(post.asType(.float32)),
                contiguous(comb.asType(.float32)),
            ],
            template: [("D", hiddenSize)],
            grid: (rows * hiddenSize, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [residual.shape],
            outputDTypes: [.float32])[0]
    }

    /// Pure-op reference and non-Metal fallback for the fused kernel above.
    public static func hcSplitSinkhornOps(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        iters: Int = 20,
        eps: Float = 1e-6
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        // Match Python `_hc_split_sinkhorn_ops` exactly. Mixes width is
        // `(2 + hcMult) * hcMult`, not `3 * hcMult`. The first hcMult
        // elements form `pre`, the next hcMult form `post`, and the
        // remaining `hcMult * hcMult` are reshaped into the (hc, hc)
        // doubly-stochastic mixing matrix `comb`.
        let mh = hcMult
        let mixHc = (2 + mh) * mh
        precondition(
            mixes.shape.last == mixHc,
            "mixes last dim must be (2+hcMult)*hcMult = \(mixHc), got \(mixes.shape.last ?? -1)")

        // Bring everything to fp32 for numerical stability — the
        // sinkhorn iterations are sensitive to fp16 underflow on the
        // post-softmax row/col normalizations.
        let mixesF = mixes.asType(.float32)
        let scaleF = scale.asType(.float32)
        let baseF = base.asType(.float32)

        let preScale = scaleF[0]
        let postScale = scaleF[1]
        let combScale = scaleF[2]

        let basePre = baseF[0..<mh]
        let basePost = baseF[mh..<(2 * mh)]
        let baseComb = baseF[(2 * mh)...]  // length mh*mh

        let mixPre = mixesF[.ellipsis, 0..<mh]
        let mixPost = mixesF[.ellipsis, mh..<(2 * mh)]
        let mixCombFlat = mixesF[.ellipsis, (2 * mh)...]  // (..., mh*mh)

        let pre = sigmoid(mixPre * preScale + basePre) + eps
        let post = 2.0 * sigmoid(mixPost * postScale + basePost)

        // Reshape last axis (mh*mh) into (mh, mh) and add bias also
        // reshaped to (mh, mh).
        let leadShape = Array(mixCombFlat.shape.dropLast())
        var combRaw = mixCombFlat * combScale
        combRaw = combRaw.reshaped(leadShape + [mh, mh])
            + baseComb.reshaped([mh, mh])
        var comb = softmax(combRaw, axis: -1) + eps
        // Initial col-normalize, then (iters-1) × {row, col}.
        comb = sinkhornColNormalize(comb, eps: eps)
        for _ in 0..<max(iters - 1, 0) {
            comb = sinkhornRowNormalize(comb, eps: eps)
            comb = sinkhornColNormalize(comb, eps: eps)
        }

        return (pre: pre, post: post, comb: comb)
    }

    private static func sinkhornRowNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let rowSum = x.sum(axis: -1, keepDims: true)
        return x / (rowSum + eps)
    }

    private static func sinkhornColNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let colSum = x.sum(axis: -2, keepDims: true)
        return x / (colSum + eps)
    }

    // MARK: - Partial RoPE
    //
    // DSV4 applies rotary ONLY to the last `ropeDim` (default 64) of
    // the head-dim=512 Q/K vector — the first 448 dims are "no-position".
    // Forward (token -> position-rotated): standard RoPE rotate.
    // Inverse (position-rotated -> token, used on attention OUTPUT):
    //   undo the rotation via negative-angle cos/sin, so the residual
    //   stream contribution is position-agnostic.
    public static func applyPartialRoPE(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        ropeDim: Int,
        inverse: Bool = false
    ) -> MLXArray {
        let headDim = x.shape.last!
        precondition(ropeDim <= headDim, "ropeDim must be ≤ headDim")
        let noPoseDim = headDim - ropeDim
        if noPoseDim == 0 {
            return rotateHalf(x, cos: cos, sin: sin, inverse: inverse)
        }
        // Split last axis: [..., :noPoseDim] keep; [..., noPoseDim:] rotate.
        let nope = x[.ellipsis, 0..<noPoseDim]
        let pe = x[.ellipsis, noPoseDim...]
        let rotated = rotateHalf(pe, cos: cos, sin: sin, inverse: inverse)
        return concatenated([nope, rotated], axis: -1)
    }

    /// Apply traditional/interleaved RoPE — DSV4 uses
    /// `traditional=True` (mx.fast.rope) which rotates ADJACENT pairs:
    /// `(x[…,0], x[…,1])`, `(x[…,2], x[…,3])`, etc. NOT split-half
    /// `(x[…,:D/2], x[…,D/2:])`. Mirror Python `_call_manual` in
    /// jang_tools/dsv4/mlx_model.py:DeepseekV4RoPE — using the wrong
    /// convention scrambles positional information across the head
    /// dim and the model decodes a repeating-token loop (verified
    /// 2026-04-24).
    ///
    /// `cos`/`sin` shape must broadcast over the leading axes and
    /// match `(L, ropeDim/2)`. `inverse=true` flips sin sign
    /// (equivalent to multiplying by conjugate of the rotation).
    private static func rotateHalf(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, inverse: Bool
    ) -> MLXArray {
        // The frequency table is intentionally built in fp32, but the
        // authoritative DSV4 runtime casts cos/sin to the activation dtype
        // before applying RoPE. Without this cast Swift promotes Q/K to fp32
        // in layer 0; the fp32 attention result then promotes the entire mHC
        // residual stream. Every later affine expert call consequently casts
        // its 67-134 MiB fp16 scale/bias tensors to fp32 on every token.
        // Preserve the model dtype here, matching DeepseekV4RoPE.__call__ in
        // jang_tools/dsv4/mlx_model.py.
        let c = cos.asType(x.dtype)
        let sinForDirection = inverse ? -sin : sin
        let s = sinForDirection.asType(x.dtype)
        let lastDim = x.shape.last!
        let halfDim = lastDim / 2
        // Reshape last axis from D to (D/2, 2) so the trailing pair
        // is the (real, imag) tuple of each rotation.
        let xPaired = x.reshaped(x.shape.dropLast() + [halfDim, 2])
        let x0 = xPaired[.ellipsis, 0]  // (..., D/2)
        let x1 = xPaired[.ellipsis, 1]  // (..., D/2)
        let r0 = x0 * c - x1 * s
        let r1 = x0 * s + x1 * c
        // Stack along a new last axis (D/2, 2) then collapse → D.
        let stacked = stacked([r0, r1], axis: -1)
        return stacked.reshaped(x.shape)
    }

    // MARK: - DSV4 SwiGLU activation with `limit`
    //
    // silu(min(gate, limit)) * clip(up, -limit, +limit). The authoritative
    // affine runtime evaluates the clamp, SiLU, and multiply in fp32 and only
    // then casts back to the incoming dtype. This is both a precision contract
    // across the 43-layer MoE stack and a compiled one-dispatch microkernel.
    public static func dsv4SwiGLU(
        gate: MLXArray,
        up: MLXArray,
        limit: Float
    ) -> MLXArray {
        let body = limit > 0
            ? _deepseekV4SwiGLUClampedBody
            : _deepseekV4SwiGLUUnclampedBody
        let limitArray = scalarArray(limit)
        if CompiledDecodeTrace.isActive {
            return body(gate, up, limitArray)
        }
        let compiled = limit > 0
            ? _compiledDeepseekV4SwiGLUClamped
            : _compiledDeepseekV4SwiGLUUnclamped
        return compiled(gate, up, limitArray)
    }

    /// Routed DSV4 expert activation. The score is multiplied while the
    /// limited-SwiGLU result is still fp32, then cast to the expert activation
    /// dtype immediately before the quantized down projection.
    public static func dsv4ScoredSwiGLU(
        gate: MLXArray,
        up: MLXArray,
        scores: MLXArray,
        limit: Float
    ) -> MLXArray {
        let body = limit > 0
            ? _deepseekV4ScoredSwiGLUClampedArrayBody
            : _deepseekV4ScoredSwiGLUUnclampedArrayBody
        let limitArray = scalarArray(limit)
        let args = [gate, up, scores, limitArray]
        // Keep this in the surrounding lazy graph. A separate shapeless compile
        // boundary adds decode specialization overhead and can mis-specialize
        // sorted prefill route counts (for example 78 distinct experts versus
        // 96 token routes).
        return body(args)[0]
    }

    // MARK: - sqrtsoftplus (MoE gate scoring)
    //
    // scores = sqrt(log1p(exp(logits))) — replaces softmax for DSV4's
    // routing. Monotonic, smoother gradient in the tail than softmax,
    // and doesn't require the sum-to-1 constraint that makes hash
    // routing incompatible.
    //
    // Numerical guard: log1p(exp(x)) is `softplus(x)` — mlx exposes
    // it directly and handles the overflow branch for large x.
    public static func sqrtSoftplus(_ logits: MLXArray) -> MLXArray {
        sqrt(logAddExp(logits, MLXArray(0.0)))
    }

    /// Decode-only fused selector for the released DSV4 routing contract.
    /// The caller owns metadata gating; unsupported shapes keep using the
    /// compiled MLX implementation below.
    public static func fusedSqrtSoftplusSelect(
        logits: MLXArray,
        bias: MLXArray,
        k: Int,
        normalize: Bool,
        scalingFactor: MLXArray
    ) -> (indices: MLXArray, weights: MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu,
              logits.dim(-1) > 0,
              logits.dim(-1) <= 256,
              bias.size == logits.dim(-1),
              scalingFactor.size == 1,
              k > 0,
              k <= logits.dim(-1)
        else { return nil }

        let experts = logits.dim(-1)
        let rows = logits.size / experts
        let outputShape = Array(logits.shape.dropLast()) + [k]
        let result = fusedSqrtSoftplusTopKKernel(
            [
                contiguous(logits.asType(.float32)),
                contiguous(bias.asType(.float32)),
                contiguous(scalingFactor.asType(.float32)),
            ],
            template: [
                ("NEXPERTS", experts),
                ("TOPK", k),
                ("NORMALIZE", normalize ? 1 : 0),
            ],
            grid: (rows * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [outputShape, outputShape],
            outputDTypes: [.int32, .float32])
        return (indices: result[0], weights: result[1])
    }

    /// Projects one decode token against only the experts already selected by
    /// a hash-routing table. The official graph needs the selected FP32 logits,
    /// not the other expert outputs, so a gathered matrix multiply avoids
    /// computing the full expert-width gate before discarding almost all of it.
    public static func selectedExpertLogits(
        input: MLXArray,
        weight: MLXArray,
        indices: MLXArray
    ) -> MLXArray {
        let hiddenSize = input.dim(-1)
        precondition(input.size == hiddenSize, "selected gate projection is decode-only")
        let inputMatrix = input.asType(.float32).reshaped(1, hiddenSize)
        let expertMatrices = weight.asType(.float32).expandedDimensions(axis: -1)
        return gatherMM(
            inputMatrix,
            expertMatrices,
            rhsIndices: indices.flattened()
        ).reshaped(indices.shape)
    }

    // MARK: - Top-k over sqrtsoftplus with bias + norm
    //
    // Production gate path (for non-hash layers):
    //   biased = scores + noauxBias
    //   topKIdx = argpartition(-biased, k)[:k]
    //   topKWeights = take_along_axis(scores, topKIdx)   — UNBIASED!
    //   normalized = topKWeights / sum(topKWeights) * routedScalingFactor
    //
    // Critical: `noauxBias` is used ONLY to pick the indices — once
    // picked, the UNBIASED score is what gets used as the expert
    // weight. This was bug #6 in the DSV-EXHAUSTIVE-VARIABLES-GUIDE;
    // using biased weights broke coherence.
    public static func sqrtSoftplusSelect(
        scores: MLXArray,
        noauxBias: MLXArray?,
        k: Int,
        normalize: Bool,
        scalingFactor: Float
    ) -> (indices: MLXArray, weights: MLXArray) {
        let biased = noauxBias != nil ? (scores + noauxBias!) : scores
        // argpartition returns unordered top-k; sort indices for
        // determinism (matters for cache-hit byte equivalence).
        let topKIdx = argPartition(-biased, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        // Gather the UNBIASED scores at those indices.
        let gathered = takeAlong(scores, topKIdx, axis: -1)
        var weights = gathered
        if normalize {
            let denom = weights.sum(axis: -1, keepDims: true) + 1e-20
            weights = weights / denom * scalingFactor
        } else {
            weights = weights * scalingFactor
        }
        return (indices: topKIdx, weights: weights)
    }

    // MARK: - YaRN RoPE freq table
    //
    // `rope_factor=16`, `original_seq_len=65536`, `beta_fast=32`,
    // `beta_slow=1` are the DSV4 defaults when compress_ratio>0.
    // Layers with compress_ratio==0 use plain (non-YaRN) RoPE with
    // `rope_theta=10000`.
    public static func yarnInvFreq(
        dim: Int,
        base: Float,
        maxPos: Int,
        origMaxPos: Int,
        factor: Float,
        betaFast: Float,
        betaSlow: Float
    ) -> MLXArray {
        _ = maxPos  // reserved for future extrapolation logic

        // Standard inv-freq table.
        let dimF = Float(dim)
        let halfDim = dim / 2
        var invFreq = [Float]()
        invFreq.reserveCapacity(halfDim)
        for i in 0..<halfDim {
            let exponent = Float(2 * i) / dimF
            invFreq.append(1.0 / pow(base, exponent))
        }
        let invFreqArr = MLXArray(invFreq)

        if factor == 1.0 {
            return invFreqArr
        }

        // YaRN: ramp mask smooths the transition between full and
        // scaled frequencies for dims that correspond to wavelengths
        // between betaFast and betaSlow. Match the pinned DSV4 source's
        // `find_correction_range(beta_fast, beta_slow, ...)`: betaFast owns
        // the low index and betaSlow owns the high index. Reversing them
        // collapses the intended ramp into a clamped step for the production
        // 0731 parameters. `high = min(..., dim - 1)` follows the source.
        let twoPi = Float.pi * 2
        func correctionDim(_ beta: Float) -> Float {
            dimF * log(Float(origMaxPos) / (beta * twoPi)) / (2.0 * log(base))
        }
        let low = max(0.0, floor(correctionDim(betaFast)))
        let high = min(Float(dim - 1), ceil(correctionDim(betaSlow)))
        let rangeWidth = max(high - low, 0.001)

        var ramp = [Float]()
        ramp.reserveCapacity(halfDim)
        for i in 0..<halfDim {
            let t = (Float(i) - low) / rangeWidth
            ramp.append(max(0.0, min(1.0, t)))
        }
        let rampArr = MLXArray(ramp)
        let smooth = MLXArray(1.0) - rampArr
        let scaled = invFreqArr / factor
        return scaled * (MLXArray(1.0) - smooth) + invFreqArr * smooth
    }

    // MARK: - LM head fp32 matmul
    //
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // DeepSeek-V4 reference inference (`inference/model.py`
    // `ParallelHead.get_logits`) explicitly performs the lm_head matmul
    // in fp32 — the weights are stored as fp32 and the activations are
    // cast to fp32 before `F.linear`. Hidden contraction is 4096 wide,
    // and bf16/fp16 accumulation drops ~0.5 ULP per logit which
    // empirically flips arithmetic-style next-token answers. Mirror
    // that here: dequantize quantized lm_heads to fp32, cast `h` to
    // fp32, then matmul.
    public static func lmHeadFp32(_ h: MLXArray, lmHead: Linear) -> MLXArray {
        if let q = lmHead as? QuantizedLinear {
            let wF32 = MLX.dequantized(
                q.weight, scales: q.scales, biases: q.biases,
                groupSize: q.groupSize, bits: q.bits
            ).asType(.float32)
            let hF32 = h.asType(.float32)
            var out = hF32.matmul(wF32.transposed())
            if let b = q.bias {
                out = out + b.asType(.float32)
            }
            return out
        }
        let wF32 = lmHead.weight.asType(.float32)
        let hF32 = h.asType(.float32)
        var out = hF32.matmul(wF32.transposed())
        if let b = lmHead.bias {
            out = out + b.asType(.float32)
        }
        return out
    }

    // MARK: - Compressor + Indexer attention masks (PR #1195 port)
    //
    // The DSV4 paper §9-13 attention path is a hybrid of:
    //   1. A LOCAL sliding-window over the last `window` raw tokens
    //      (kept in a RotatingKVCache).
    //   2. A GLOBAL pooled context over compressor chunks (kept as
    //      a single (B, P, head_dim) tensor in an ArraysCache slot).
    //
    // Visibility from query at raw position `q` to a key:
    //
    //   - Window key at raw position `r`:
    //       q - window < r <= q
    //
    //   - Compressed key at chunk index `k` covering raw range
    //     [k*ratio, (k+1)*ratio):
    //       (k + 1) * ratio <= q + 1
    //
    // For compress_ratio==4 layers the Indexer adds a top-k
    // selection — only the K compressed chunks the indexer scored
    // highest are visible, ANDed with the causal staircase above.
    //
    // Both helpers return 4D bool arrays of shape (B, 1, S, L_kv)
    // that broadcast onto SDPA attention scores (B, H, S, L_kv).
    // Building 4D directly avoids the SDPA broadcast bugs the
    // previous staircase attempts hit.

    /// Per-query visibility into the local sliding-window cache.
    ///
    /// `windowLen` is the number of slots currently filled in the
    /// RotatingKVCache (== window once the buffer wraps). The
    /// trailing `windowLen` raw positions in the cache map to raw
    /// token indices `(offset + S) - windowLen + i` for slot `i`.
    /// Returns shape `(B, 1, S, windowLen)`.
    public static func buildWindowMask(
        batch B: Int, queryLen S: Int,
        offset: Int, window: Int, windowLen: Int
    ) -> MLXArray {
        // q_pos: (B, S) — broadcasted absolute raw positions of each
        // query slot. The PR #1195 Python builds (B, S) by broadcasting
        // (1, S) to (B, S); we do the same.
        let qPos =
            MLXArray(Int32(offset)..<Int32(offset + S))
            .expandedDimensions(axis: 0)        // (1, S)
            .reshaped(1, S)
        // raw_pos_at_k: (windowLen,) → (1, 1, windowLen)
        let cacheK = MLXArray(Int32(0)..<Int32(windowLen))
        let rawPosAtK = MLXArray(Int32((offset + S) - windowLen)) + cacheK
        // qPos: (1, S) → (1, S, 1) then broadcast against (1, 1, windowLen)
        let qPos3 = qPos.expandedDimensions(axis: -1)
        let raw3 = rawPosAtK.expandedDimensions(axes: [0, 1])
        let lower = raw3 .> (qPos3 - MLXArray(Int32(window)))
        let upper = raw3 .<= qPos3
        let visible = MLX.logicalAnd(lower, upper)
        // (1, S, windowLen) → broadcast to (B, 1, S, windowLen)
        let v4 = visible.expandedDimensions(axis: 1)
        let bArr = MLX.broadcast(v4, to: [B, 1, S, windowLen])
        return bArr
    }

    /// Per-query causal visibility into the compressor's pooled
    /// chunks. Chunk `k` covers raw positions `[k*ratio, (k+1)*ratio)`
    /// and is visible to query `q` once that whole chunk has been
    /// observed: `(k+1)*ratio <= q+1`.
    /// Returns shape `(B, 1, S, compressedLen)`.
    public static func compressedVisibility(
        batch B: Int, queryLen S: Int,
        offset: Int, compressedLen: Int, ratio: Int
    ) -> MLXArray {
        let qPos =
            MLXArray(Int32(offset)..<Int32(offset + S))
            .expandedDimensions(axis: 0)
            .reshaped(1, S)
        let k = MLXArray(Int32(0)..<Int32(compressedLen))
        // (k+1) * ratio <= (qPos + 1)
        let lhs =
            (k + MLXArray(Int32(1))) * MLXArray(Int32(ratio))
        let rhs = qPos + MLXArray(Int32(1))
        // lhs: (compressedLen,) → (1, 1, compressedLen)
        let lhs3 = lhs.expandedDimensions(axes: [0, 1])
        // rhs: (1, S) → (1, S, 1)
        let rhs3 = rhs.expandedDimensions(axis: -1)
        let visible = lhs3 .<= rhs3
        let v4 = visible.expandedDimensions(axis: 1)
        return MLX.broadcast(v4, to: [B, 1, S, compressedLen])
    }

    /// Apply DSV4's block-causal compressed-pool visibility before
    /// HSA indexer top-k selection. The later SDPA mask is still the
    /// final authority, but masking here prevents argpartition from
    /// spending the whole top-k budget on future chunks that are later
    /// filtered out.
    ///
    /// `scores` shape: `(B, S, compressedLen)`.
    public static func causalMaskedIndexerScores(
        _ scores: MLXArray, offset: Int, ratio: Int
    ) -> MLXArray {
        let B = scores.dim(0)
        let S = scores.dim(1)
        let compressedLen = scores.dim(2)
        guard S > 1, compressedLen > 0 else { return scores }

        let visible = compressedVisibility(
            batch: B, queryLen: S, offset: offset,
            compressedLen: compressedLen, ratio: ratio
        ).squeezed(axis: 1)
        let negLarge = MLXArray(Float(-1.0e30), dtype: scores.dtype)
        return MLX.where(visible, scores, negLarge)
    }

    /// AND the per-query indexer top-k selection onto a compressed
    /// visibility mask. `topk` is the indexer's `(B, S, K)` int array
    /// of selected chunk indices; returns `(B, 1, S, compressedLen)`
    /// bool — true only when chunk index `c` appears in `topk[b, s, :]`.
    public static func indexerSelectionMask(
        topk: MLXArray, compressedLen: Int
    ) -> MLXArray {
        // topk: (B, S, K) → (B, S, K, 1)
        let topk4 = topk.expandedDimensions(axis: -1)
        // k_range: (compressedLen,) → (1, 1, 1, compressedLen)
        let kRange =
            MLXArray(Int32(0)..<Int32(compressedLen))
            .expandedDimensions(axes: [0, 1, 2])
        // (B, S, K, compressedLen) → (B, S, compressedLen) via any over K
        let eq = topk4 .== kRange
        let selected = eq.any(axis: -2)  // (B, S, compressedLen)
        return selected.expandedDimensions(axis: 1)  // (B, 1, S, compressedLen)
    }
}
