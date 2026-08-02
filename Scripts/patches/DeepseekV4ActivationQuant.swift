import Foundation
import MLX
import MLXFast
import MLXNN

/// Activation fake-quantization used by the official DeepSeek-V4 0731 MXFP
/// inference path before every MXFP quantized matmul.
public enum DeepseekV4ActivationQuant {
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
        guard isMXFP(mode), blockSize == 128, x.size > 0, x.dim(-1).isMultiple(of: 128) else {
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

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(x, mode: mode)
        let y: MLXArray
        if mode == .mxfp8 {
            // mlx-swift 0.30.3's MXFP8 quantizedMM scales 0731 checkpoint
            // outputs incorrectly. Dequantization itself is correct, so cache
            // the bf16 weight and use the reference-equivalent dense matmul.
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
