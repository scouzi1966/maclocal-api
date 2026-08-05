import Foundation
import MLX
import MLXFast
import MLXNN

/// Capability exposed by model configurations whose affine-Q8 metadata maps
/// to AFM's signed symmetric Q8 storage rather than MLX's affine encoding.
public protocol DeepseekV4SymmetricQ8Model {
    var usesDeepseekV4SymmetricQ8: Bool { get }
}

/// Activation fake-quantization used by the official DeepSeek-V4 0731 MXFP
/// inference path before every MXFP quantized matmul.
public enum DeepseekV4ActivationQuant {
    private static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV4_ACTIVATION_QAT"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }()

    private static let e4m3ActivationRoundTripKernel = MLXFast.metalKernel(
        name: "deepseek_v4_e4m3_activation_roundtrip",
        inputNames: ["x"],
        outputNames: ["y"],
        source: """
            const uint gid = thread_position_in_grid.x;
            const uint lane = thread_position_in_threadgroup.x;
            const uint idx = gid;

            threadgroup float scratch[128];
            const float input_value = static_cast<float>(x[idx]);
            scratch[lane] = metal::abs(input_value);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 64; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    scratch[lane] = metal::max(scratch[lane], scratch[lane + stride]);
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

            const float normalized = metal::clamp(input_value / scale, -448.0f, 448.0f);
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
        """)

    public static func isMXFP(_ mode: QuantizationMode) -> Bool {
        mode == .mxfp4 || mode == .mxfp8
    }

    public static func e4m3RoundTripIfNeeded(
        _ x: MLXArray, mode: QuantizationMode, blockSize: Int = 128
    ) -> MLXArray {
        guard enabled, isMXFP(mode), blockSize == 128, x.size > 0,
            x.dim(-1).isMultiple(of: 128)
        else {
            return x
        }
        let input = contiguous(x)
        return e4m3ActivationRoundTripKernel(
            [input],
            template: [("outT", input.dtype)],
            grid: (input.size, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
    }
}

open class DeepseekV4QuantizedLinear: QuantizedLinear {
    private let dequantizedWeightLock = NSLock()
    private var cachedDequantizedWeight: MLXArray?

    private static let nativeMXFP8Enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["VMLX_DSV4_NATIVE_MXFP8"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }()

    private static let symmetricQ8ActivationLog: Void = {
        fputs("[DSV4Path] symmetric-q8 active\n", stderr)
    }()

    private static let symmetricQ8Kernel = MLXFast.metalKernel(
        name: "deepseek_v4_symmetric_q8_matvec",
        inputNames: ["x", "weight", "scales"],
        outputNames: ["y"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const uint rowBase = threadgroup_position_in_grid.x * 2u;
            const uint inputRow = threadgroup_position_in_grid.y;
            const uint laneGroup = lane >> 2u;
            const uint laneOffset = (lane & 3u) * 8u;
            const device char *packed =
                reinterpret_cast<const device char *>(weight);
            const device inT *input = x + inputRow * INPUT;

            float rowSum[2] = {0.0f, 0.0f};
            for (uint group = simd * 8u + laneGroup;
                 group < GROUPS;
                 group += 32u) {
                const uint inputBase = group * 32u + laneOffset;
                float values[8];
                for (uint i = 0u; i < 8u; ++i) {
                    values[i] = static_cast<float>(input[inputBase + i]);
                }

                for (uint row = 0u; row < 2u && rowBase + row < OUTPUT; ++row) {
                    const uint outputRow = rowBase + row;
                    const uint weightBase = outputRow * INPUT + inputBase;
                    float dot = 0.0f;
                    for (uint i = 0u; i < 8u; ++i) {
                        dot += static_cast<float>(packed[weightBase + i]) * values[i];
                    }
                    rowSum[row] += dot * static_cast<float>(
                        scales[outputRow * GROUPS + group]);
                }
            }

            threadgroup float partial[8];
            for (uint row = 0u; row < 2u; ++row) {
                const float reduced = simd_sum(rowSum[row]);
                if (lane == 0u) partial[row * 4u + simd] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd == 0u && lane < 2u && rowBase + lane < OUTPUT) {
                const uint offset = lane * 4u;
                const float total = partial[offset] + partial[offset + 1u]
                    + partial[offset + 2u] + partial[offset + 3u];
                y[inputRow * OUTPUT + rowBase + lane] = static_cast<outT>(total);
            }
        """,
        ensureRowContiguous: false)

    private static let symmetricQ8GroupedKernel = MLXFast.metalKernel(
        name: "deepseek_v4_symmetric_q8_grouped_matvec",
        inputNames: ["x", "weight", "scales"],
        outputNames: ["y"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const uint localRowBase = threadgroup_position_in_grid.x * 2u;
            const uint inputRow = threadgroup_position_in_grid.y;
            const uint outputGroup = inputRow % OUTPUT_GROUPS;
            const uint weightRowBase = outputGroup * OUTPUT_PER_GROUP + localRowBase;
            const uint laneGroup = lane >> 2u;
            const uint laneOffset = (lane & 3u) * 8u;
            const device char *packed =
                reinterpret_cast<const device char *>(weight);
            const device inT *input = x + inputRow * INPUT;

            float rowSum[2] = {0.0f, 0.0f};
            for (uint group = simd * 8u + laneGroup;
                 group < GROUPS;
                 group += 32u) {
                const uint inputBase = group * 32u + laneOffset;
                float values[8];
                for (uint i = 0u; i < 8u; ++i) {
                    values[i] = static_cast<float>(input[inputBase + i]);
                }

                for (uint row = 0u; row < 2u && localRowBase + row < OUTPUT_PER_GROUP;
                     ++row) {
                    const uint weightRow = weightRowBase + row;
                    const uint weightBase = weightRow * INPUT + inputBase;
                    float dot = 0.0f;
                    for (uint i = 0u; i < 8u; ++i) {
                        dot += static_cast<float>(packed[weightBase + i]) * values[i];
                    }
                    rowSum[row] += dot * static_cast<float>(
                        scales[weightRow * GROUPS + group]);
                }
            }

            threadgroup float partial[8];
            for (uint row = 0u; row < 2u; ++row) {
                const float reduced = simd_sum(rowSum[row]);
                if (lane == 0u) partial[row * 4u + simd] = reduced;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd == 0u && lane < 2u && localRowBase + lane < OUTPUT_PER_GROUP) {
                const uint offset = lane * 4u;
                const float total = partial[offset] + partial[offset + 1u]
                    + partial[offset + 2u] + partial[offset + 3u];
                y[inputRow * OUTPUT_PER_GROUP + localRowBase + lane] =
                    static_cast<outT>(total);
            }
        """,
        ensureRowContiguous: false)

    public var usesSymmetricQ8Storage: Bool {
        mode == .affine && bits == 8 && groupSize == 32 && biases == nil
            && weight.dtype == .uint32
    }

    public func symmetricQ8Grouped(_ x: MLXArray, outputGroups: Int) -> MLXArray {
        precondition(usesSymmetricQ8Storage)
        let input = contiguous(x)
        let inputDims = input.dim(-1)
        let groups = inputDims / groupSize
        let totalOutputDims = scales.size / groups
        precondition(totalOutputDims % outputGroups == 0)
        precondition(input.dim(-2) == outputGroups)
        return MLXFast.deepseekV4SymmetricQ8Matvec(
            input,
            weight: contiguous(weight),
            scales: contiguous(scales),
            outputGroups: outputGroups)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(x, mode: mode)
        let y: MLXArray
        if usesSymmetricQ8Storage,
           activation.dim(-1) == weight.dim(-1) * 4
        {
            _ = Self.symmetricQ8ActivationLog
            y = MLXFast.deepseekV4SymmetricQ8Matvec(
                contiguous(activation),
                weight: contiguous(weight),
                scales: contiguous(scales))
        } else if mode == .mxfp8 && !Self.nativeMXFP8Enabled {
            // Diagnostic fallback for comparing the former BF16-expanded path
            // against mlx-swift 0.31.x's native MXFP8 implementation.
            dequantizedWeightLock.lock()
            if cachedDequantizedWeight == nil {
                let dequantized = MLX.dequantized(
                    weight, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode,
                    dtype: activation.dtype)
                MLX.eval(dequantized)
                cachedDequantizedWeight = dequantized
            }
            let dequantized = cachedDequantizedWeight!
            dequantizedWeightLock.unlock()
            y = activation.matmul(dequantized.transposed())
        } else {
            y = quantizedMM(
                activation,
                weight,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }
        var output = y
        if let bias {
            output = output + bias
        }
        return output
    }
}
