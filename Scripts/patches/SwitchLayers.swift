import Foundation
import MLX
import MLXFast
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

// GELU approximate without the Power primitive (x ** 3). Uses x * x * x which
// decomposes to Multiply ops with proper output_shapes support.
// On M3+: compiled with compile(shapeless: true) for fused Metal dispatch.
// On M1/M2: runs as plain closure (compile(shapeless: true) crashes on Tahoe — MLX #3329).
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    guard HardwareInfo.isCompiledDecodeSupported else { return body }
    let compiled = compile(shapeless: true, body)
    // When this activation is invoked *inside* the outer compiled-decode trace
    // (`setupCompiledDecode` → `CompiledDecodeTrace.withActive`), calling a
    // separately-compiled function is a nested compile — illegal, exactly like
    // `eval` during a trace (see the `!CompiledDecodeTrace.isActive` guards in
    // Gemma4Text). The inner `compileState.call` returns an empty result and
    // `[0]` traps (Transforms+Compile.swift). Run the plain body while tracing:
    // its ops are captured into the outer graph and fused there, so there is no
    // throughput loss — the inner compile was both illegal and redundant.
    return { x in CompiledDecodeTrace.isActive ? body(x) : compiled(x) }
}()

/// Drop-in replacement for MLXNN.GELU that avoids the Power primitive crash.
/// Use this anywhere `GELU(approximation: .precise)` or `.tanh` would be used.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compatibility helper for patched MoE models that expect a fused gate/up
/// activation utility. Keep this as ordinary MLX ops; DSV4 correctness depends
/// on avoiding the previous custom Metal fused path during bring-up.
public func fusedSiluMul(_ gateUp: MLXArray, hiddenDims: Int) -> MLXArray {
    let gate = gateUp[.ellipsis, ..<hiddenDims]
    let up = gateUp[.ellipsis, hiddenDims...]
    return silu(gate) * up
}

// Compiled activation kernels — fuses gate activation + element-wise multiply into
// a single Metal dispatch. Matches Python's @partial(mx.compile, shapeless=True).
// Guarded by HardwareInfo: M1/M2 + macOS Tahoe crashes with compile(shapeless: true).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, x: MLXArray) -> MLXArray in
        silu(gate) * x
    }
    guard HardwareInfo.isCompiledDecodeSupported else { return body }
    let compiled = compile(shapeless: true, body)
    // Fall back to the plain body inside the outer compiled-decode trace to
    // avoid an illegal nested compile (see `safeGeluApproximate`).
    return { g, x in CompiledDecodeTrace.isActive ? body(g, x) : compiled(g, x) }
}()

private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, x: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * x
    }
    guard HardwareInfo.isCompiledDecodeSupported else { return body }
    let compiled = compile(shapeless: true, body)
    // Fall back to the plain body inside the outer compiled-decode trace to
    // avoid an illegal nested compile (see `safeGeluApproximate`).
    return { g, x in CompiledDecodeTrace.isActive ? body(g, x) : compiled(g, x) }
}()

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

private enum SwitchGLUKernelEngine {
    static var ds4Enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        let raw = (env["AFM_MLX_KERNELS"] ?? env["VMLX_DSV4_KERNELS"] ?? "native")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "ds4"
    }

    static var nativeDeepseekMXFP4Enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        let raw = (env["AFM_MLX_KERNELS"] ?? env["VMLX_DSV4_KERNELS"] ?? "native")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let override = (env["VMLX_DSV4_NATIVE_MXFP4"] ?? "1")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "native" && override != "0" && override != "false"
    }

    static var specializedDeepseekMXFP4Enabled: Bool {
        ds4Enabled || nativeDeepseekMXFP4Enabled
    }

    static var deepseekMXFP4StagedMoEEnabled: Bool {
        let raw = (ProcessInfo.processInfo.environment["VMLX_DSV4_STAGED_MOE"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return nativeDeepseekMXFP4Enabled && (raw == "1" || raw == "true")
    }

    static var deepseekMXFP4StagedSelectorEnabled: Bool {
        let raw = (ProcessInfo.processInfo.environment["VMLX_DSV4_STAGED_SELECTOR"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return deepseekMXFP4StagedMoEEnabled && (raw == "1" || raw == "true")
    }

    static var deepseekMXFP4RowsPerSIMD: Int {
        constrainedInteger("VMLX_DSV4_MXFP4_ROWS_PER_SIMD", allowed: [1, 2, 4], default: 2)
    }

    static var deepseekMXFP4SIMDGroupsPerThreadgroup: Int {
        constrainedInteger("VMLX_DSV4_MXFP4_SIMD_GROUPS", allowed: [1, 2, 4, 8], default: 2)
    }

    static var deepseekMXFP4CooperativeDownEnabled: Bool {
        let raw = (ProcessInfo.processInfo.environment["VMLX_DSV4_COOPERATIVE_DOWN"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "1" || raw == "true"
    }

    private static var deepseekMXFP4HalfMultiplyMode: String {
        (ProcessInfo.processInfo.environment["VMLX_DSV4_HALF_MULTIPLY"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    static var deepseekMXFP4GateUpHalfMultiplyEnabled: Bool {
        let mode = deepseekMXFP4HalfMultiplyMode
        return mode == "1" || mode == "true" || mode == "all" || mode == "gate_up"
    }

    static var deepseekMXFP4DownHalfMultiplyEnabled: Bool {
        let mode = deepseekMXFP4HalfMultiplyMode
        return mode == "1" || mode == "true" || mode == "all" || mode == "down"
    }

    private static func constrainedInteger(
        _ name: String, allowed: Set<Int>, default defaultValue: Int
    ) -> Int {
        guard let raw = ProcessInfo.processInfo.environment[name],
              let value = Int(raw),
              allowed.contains(value)
        else { return defaultValue }
        return value
    }
}

private enum DeepseekV4DS4Kernels {
    private static let routeLimit = 6
    private static let supportedInputDims = 4096
    private static let supportedHiddenDims = 2048
    private static let supportedExperts = 256
    private static let supportedGroupSize = 32
    private static let stagedMoEActivationLog: Void = {
        fputs("[DSV4Path] staged-moe active\n", stderr)
    }()
    private static let stagedSelectorActivationLog: Void = {
        fputs("[DSV4Path] staged-selector active\n", stderr)
    }()
    private static let stagedSharedQ8ActivationLog: Void = {
        fputs("[DSV4Path] staged-shared-q8 active\n", stderr)
    }()
    static func fusedRoutedMoE(
        input: MLXArray,
        indices: MLXArray,
        scores: MLXArray,
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear,
        limit: Float
    ) -> MLXArray? {
        guard SwitchGLUKernelEngine.deepseekMXFP4StagedMoEEnabled,
              input.size == supportedInputDims,
              input.dim(-1) == supportedInputDims,
              gate.inputDims == supportedInputDims,
              gate.outputDims == supportedHiddenDims,
              gate.numExperts == supportedExperts,
              up.inputDims == supportedInputDims,
              up.outputDims == supportedHiddenDims,
              up.numExperts == supportedExperts,
              down.inputDims == supportedHiddenDims,
              down.outputDims == supportedInputDims,
              down.numExperts == supportedExperts,
              indices.size == routeLimit,
              gate.groupSize == supportedGroupSize,
              up.groupSize == supportedGroupSize,
              down.groupSize == supportedGroupSize,
              gate.bits == 4, up.bits == 4, down.bits == 4,
              gate.mode == .mxfp4, up.mode == .mxfp4, down.mode == .mxfp4,
              gate.weight.dtype == .uint32,
              up.weight.dtype == .uint32,
              down.weight.dtype == .uint32,
              gate.scales.dtype == .uint8,
              up.scales.dtype == .uint8,
              down.scales.dtype == .uint8,
              gate.bias == nil, up.bias == nil, down.bias == nil,
              gate.biases == nil, up.biases == nil, down.biases == nil
        else { return nil }

        _ = stagedMoEActivationLog
        return MLXFast.deepseekV4MXFP4MoE(
            contiguous(input),
            gateWeight: contiguous(gate.weight),
            gateScales: contiguous(gate.scales),
            upWeight: contiguous(up.weight),
            upScales: contiguous(up.scales),
            downWeight: contiguous(down.weight),
            downScales: contiguous(down.scales),
            indices: contiguous(indices.flattened()),
            scores: contiguous(scores.flattened()),
            activationLimit: limit)
    }

    static func fusedRoutedMoESelecting(
        input: MLXArray,
        logits: MLXArray,
        bias: MLXArray,
        routeScale: MLXArray,
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear,
        limit: Float
    ) -> MLXArray? {
        guard SwitchGLUKernelEngine.deepseekMXFP4StagedSelectorEnabled,
              input.size == supportedInputDims,
              input.dim(-1) == supportedInputDims,
              logits.size == supportedExperts,
              logits.dtype == .float32,
              bias.size == supportedExperts,
              bias.dtype == .float32,
              routeScale.size == 1,
              routeScale.dtype == .float32,
              gate.inputDims == supportedInputDims,
              gate.outputDims == supportedHiddenDims,
              gate.numExperts == supportedExperts,
              up.inputDims == supportedInputDims,
              up.outputDims == supportedHiddenDims,
              up.numExperts == supportedExperts,
              down.inputDims == supportedHiddenDims,
              down.outputDims == supportedInputDims,
              down.numExperts == supportedExperts,
              gate.groupSize == supportedGroupSize,
              up.groupSize == supportedGroupSize,
              down.groupSize == supportedGroupSize,
              gate.bits == 4, up.bits == 4, down.bits == 4,
              gate.mode == .mxfp4, up.mode == .mxfp4, down.mode == .mxfp4,
              gate.weight.dtype == .uint32,
              up.weight.dtype == .uint32,
              down.weight.dtype == .uint32,
              gate.scales.dtype == .uint8,
              up.scales.dtype == .uint8,
              down.scales.dtype == .uint8,
              gate.bias == nil, up.bias == nil, down.bias == nil,
              gate.biases == nil, up.biases == nil, down.biases == nil
        else { return nil }

        _ = stagedSelectorActivationLog
        return MLXFast.deepseekV4MXFP4MoESelecting(
            contiguous(input),
            gateWeight: contiguous(gate.weight),
            gateScales: contiguous(gate.scales),
            upWeight: contiguous(up.weight),
            upScales: contiguous(up.scales),
            downWeight: contiguous(down.weight),
            downScales: contiguous(down.scales),
            logits: contiguous(logits),
            bias: contiguous(bias),
            routeScale: contiguous(routeScale),
            activationLimit: limit)
    }

    static func fusedRoutedMoESelectingWithSharedQ8(
        input: MLXArray,
        logits: MLXArray,
        bias: MLXArray,
        routeScale: MLXArray,
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear,
        sharedGate: DeepseekV4QuantizedLinear,
        sharedUp: DeepseekV4QuantizedLinear,
        sharedDown: DeepseekV4QuantizedLinear,
        limit: Float
    ) -> MLXArray? {
        guard ProcessInfo.processInfo.environment["VMLX_DSV4_SHARED_Q8_STAGE"] == "1",
              SwitchGLUKernelEngine.deepseekMXFP4StagedSelectorEnabled,
              input.size == supportedInputDims,
              input.dim(-1) == supportedInputDims,
              logits.size == supportedExperts,
              logits.dtype == .float32,
              bias.size == supportedExperts,
              bias.dtype == .float32,
              routeScale.size == 1,
              routeScale.dtype == .float32,
              gate.inputDims == supportedInputDims,
              gate.outputDims == supportedHiddenDims,
              gate.numExperts == supportedExperts,
              up.inputDims == supportedInputDims,
              up.outputDims == supportedHiddenDims,
              up.numExperts == supportedExperts,
              down.inputDims == supportedHiddenDims,
              down.outputDims == supportedInputDims,
              down.numExperts == supportedExperts,
              gate.groupSize == supportedGroupSize,
              up.groupSize == supportedGroupSize,
              down.groupSize == supportedGroupSize,
              gate.bits == 4, up.bits == 4, down.bits == 4,
              gate.mode == .mxfp4, up.mode == .mxfp4, down.mode == .mxfp4,
              gate.weight.dtype == .uint32,
              up.weight.dtype == .uint32,
              down.weight.dtype == .uint32,
              gate.scales.dtype == .uint8,
              up.scales.dtype == .uint8,
              down.scales.dtype == .uint8,
              gate.bias == nil, up.bias == nil, down.bias == nil,
              gate.biases == nil, up.biases == nil, down.biases == nil,
              sharedGate.usesSymmetricQ8Storage,
              sharedUp.usesSymmetricQ8Storage,
              sharedDown.usesSymmetricQ8Storage,
              sharedGate.weight.size == supportedHiddenDims * supportedInputDims / 4,
              sharedUp.weight.size == supportedHiddenDims * supportedInputDims / 4,
              sharedDown.weight.size == supportedInputDims * supportedHiddenDims / 4
        else { return nil }

        _ = stagedSharedQ8ActivationLog
        return MLXFast.deepseekV4MXFP4MoESelectingWithSharedQ8(
            contiguous(input),
            gateWeight: contiguous(gate.weight),
            gateScales: contiguous(gate.scales),
            upWeight: contiguous(up.weight),
            upScales: contiguous(up.scales),
            downWeight: contiguous(down.weight),
            downScales: contiguous(down.scales),
            logits: contiguous(logits),
            bias: contiguous(bias),
            routeScale: contiguous(routeScale),
            sharedGateWeight: contiguous(sharedGate.weight),
            sharedGateScales: contiguous(sharedGate.scales),
            sharedUpWeight: contiguous(sharedUp.weight),
            sharedUpScales: contiguous(sharedUp.scales),
            sharedDownWeight: contiguous(sharedDown.weight),
            sharedDownScales: contiguous(sharedDown.scales),
            activationLimit: limit).asType(input.dtype)
    }

    private static let fusedGateUpScoredKernel = MLXFast.metalKernel(
        name: "deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu",
        inputNames: [
            "x", "gateW", "gateS", "upW", "upS", "indices", "scores",
        ],
        outputNames: ["activated"],
        source: """
            constexpr uint ROWS = ROWS_PER_SIMD;
            const uint linear = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = linear / 32;
            const uint tile = simd % ((HIDDEN + ROWS - 1u) / ROWS);
            const uint route = simd / ((HIDDEN + ROWS - 1u) / ROWS);
            const uint hidden = tile * ROWS;

            if (route >= ROUTES || hidden >= HIDDEN) {
                return;
            }

            const uint expert = static_cast<uint>(indices[route]);
            if (expert >= EXPERTS) {
                return;
            }

            float gateSum[ROWS] = {0.0f};
            float upSum[ROWS] = {0.0f};

            const uint ix = lane >> 1u;
            const uint halfLane = lane & 1u;
            for (uint group = ix; group < GROUPS; group += 16u) {
                const uint activationBase = group * GROUP_SIZE + halfLane * 16u;
                const float4 x0(
                    static_cast<float>(x[activationBase]),
                    static_cast<float>(x[activationBase + 1u]),
                    static_cast<float>(x[activationBase + 2u]),
                    static_cast<float>(x[activationBase + 3u]));
                const float4 x1(
                    static_cast<float>(x[activationBase + 4u]),
                    static_cast<float>(x[activationBase + 5u]),
                    static_cast<float>(x[activationBase + 6u]),
                    static_cast<float>(x[activationBase + 7u]));
                const float4 x2(
                    static_cast<float>(x[activationBase + 8u]),
                    static_cast<float>(x[activationBase + 9u]),
                    static_cast<float>(x[activationBase + 10u]),
                    static_cast<float>(x[activationBase + 11u]));
                const float4 x3(
                    static_cast<float>(x[activationBase + 12u]),
                    static_cast<float>(x[activationBase + 13u]),
                    static_cast<float>(x[activationBase + 14u]),
                    static_cast<float>(x[activationBase + 15u]));
                for (uint row = 0u; row < ROWS; ++row) {
                    const uint output = hidden + row;
                    const uint rowBase = (expert * HIDDEN + output) * PACKED_IN;
                    const uint scaleBase = (expert * HIDDEN + output) * GROUPS;
                    const float gateScale = dsv4_e8m0(gateS[scaleBase + group]);
                    const float upScale = dsv4_e8m0(upS[scaleBase + group]);
                    const uint wordBase = rowBase + group * WORDS_PER_GROUP + halfLane * 2u;
                    const uint gate0 = gateW[wordBase];
                    const uint gate1 = gateW[wordBase + 1u];
                    const uint up0 = upW[wordBase];
                    const uint up1 = upW[wordBase + 1u];
                    float gateDot;
                    float upDot;
                    if constexpr (HALF_MULTIPLY) {
                        const half4 gateProduct0 = half4(x0) * dsv4_fp4x4_half(gate0);
                        const half4 gateProduct1 = half4(x1) * dsv4_fp4x4_half(gate0 >> 16);
                        const half4 gateProduct2 = half4(x2) * dsv4_fp4x4_half(gate1);
                        const half4 gateProduct3 = half4(x3) * dsv4_fp4x4_half(gate1 >> 16);
                        const half4 upProduct0 = half4(x0) * dsv4_fp4x4_half(up0);
                        const half4 upProduct1 = half4(x1) * dsv4_fp4x4_half(up0 >> 16);
                        const half4 upProduct2 = half4(x2) * dsv4_fp4x4_half(up1);
                        const half4 upProduct3 = half4(x3) * dsv4_fp4x4_half(up1 >> 16);
                        gateDot = dot(float4(gateProduct0), float4(1.0f))
                            + dot(float4(gateProduct1), float4(1.0f))
                            + dot(float4(gateProduct2), float4(1.0f))
                            + dot(float4(gateProduct3), float4(1.0f));
                        upDot = dot(float4(upProduct0), float4(1.0f))
                            + dot(float4(upProduct1), float4(1.0f))
                            + dot(float4(upProduct2), float4(1.0f))
                            + dot(float4(upProduct3), float4(1.0f));
                    } else {
                        gateDot = dot(x0, dsv4_fp4x4(gate0))
                            + dot(x1, dsv4_fp4x4(gate0 >> 16))
                            + dot(x2, dsv4_fp4x4(gate1))
                            + dot(x3, dsv4_fp4x4(gate1 >> 16));
                        upDot = dot(x0, dsv4_fp4x4(up0))
                            + dot(x1, dsv4_fp4x4(up0 >> 16))
                            + dot(x2, dsv4_fp4x4(up1))
                            + dot(x3, dsv4_fp4x4(up1 >> 16));
                    }
                    gateSum[row] += gateScale * gateDot;
                    upSum[row] += upScale * upDot;
                }
            }

            for (uint row = 0u; row < ROWS; ++row) {
                gateSum[row] = simd_sum(gateSum[row]);
                upSum[row] = simd_sum(upSum[row]);
            }

            if (lane == 0) {
                const float activationLimit = static_cast<float>(LIMIT);
                for (uint row = 0u; row < ROWS && hidden + row < HIDDEN; ++row) {
                    const float gate = gateSum[row];
                    const float up = upSum[row];
                    const float limitedGate = metal::min(gate, activationLimit);
                    const float siluGate = limitedGate / (1.0f + metal::fast::exp(-limitedGate));
                    const float clippedUp = metal::clamp(up, -activationLimit, activationLimit);
                    const float routed = siluGate * clippedUp * static_cast<float>(scores[route]);
                    activated[route * HIDDEN + hidden + row] = static_cast<outT>(routed);
                }
            }
        """,
        header: """
            constant float dsv4_fp4_lut[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
            };

            constant half dsv4_fp4_half_lut[16] = {
                0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
                -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
            };

            static inline float4 dsv4_fp4x4(uint packed) {
                return float4(
                    dsv4_fp4_lut[packed & 0xfu],
                    dsv4_fp4_lut[(packed >> 4) & 0xfu],
                    dsv4_fp4_lut[(packed >> 8) & 0xfu],
                    dsv4_fp4_lut[(packed >> 12) & 0xfu]);
            }

            static inline half4 dsv4_fp4x4_half(uint packed) {
                return half4(
                    dsv4_fp4_half_lut[packed & 0xfu],
                    dsv4_fp4_half_lut[(packed >> 4) & 0xfu],
                    dsv4_fp4_half_lut[(packed >> 8) & 0xfu],
                    dsv4_fp4_half_lut[(packed >> 12) & 0xfu]);
            }

            static inline float dsv4_e8m0(uchar exponent) {
                const uint bits = exponent == 0
                    ? 0x00400000u
                    : (uint(exponent) << 23);
                return as_type<float>(bits);
            }
        """)

    private static let fusedDownSum6Kernel = MLXFast.metalKernel(
        name: "deepseek_v4_native_mxfp4_down_sum6",
        inputNames: ["activated", "downW", "downS", "indices"],
        outputNames: ["reduced"],
        source: """
            constexpr uint ROWS = ROWS_PER_SIMD;
            const uint linear = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = linear / 32u;
            const uint hidden = simd * ROWS;

            if (hidden >= OUTPUT) {
                return;
            }

            float total[ROWS] = {0.0f};
            for (uint route = 0u; route < ROUTES; ++route) {
                const uint expert = static_cast<uint>(indices[route]);
                if (expert >= EXPERTS) {
                    continue;
                }
                float routeSum[ROWS] = {0.0f};
                const uint ix = lane >> 1u;
                const uint halfLane = lane & 1u;
                for (uint group = ix; group < GROUPS; group += 16u) {
                    const uint activationBase =
                        route * INPUT + group * GROUP_SIZE + halfLane * 16u;
                    const float4 x0(
                            static_cast<float>(activated[activationBase]),
                            static_cast<float>(activated[activationBase + 1u]),
                            static_cast<float>(activated[activationBase + 2u]),
                            static_cast<float>(activated[activationBase + 3u]));
                    const float4 x1(
                            static_cast<float>(activated[activationBase + 4u]),
                            static_cast<float>(activated[activationBase + 5u]),
                            static_cast<float>(activated[activationBase + 6u]),
                            static_cast<float>(activated[activationBase + 7u]));
                    const float4 x2(
                            static_cast<float>(activated[activationBase + 8u]),
                            static_cast<float>(activated[activationBase + 9u]),
                            static_cast<float>(activated[activationBase + 10u]),
                            static_cast<float>(activated[activationBase + 11u]));
                    const float4 x3(
                            static_cast<float>(activated[activationBase + 12u]),
                            static_cast<float>(activated[activationBase + 13u]),
                            static_cast<float>(activated[activationBase + 14u]),
                            static_cast<float>(activated[activationBase + 15u]));

                    for (uint row = 0u; row < ROWS; ++row) {
                        const uint output = hidden + row;
                        const uint rowBase = (expert * OUTPUT + output) * PACKED_IN;
                        const uint scaleBase = (expert * OUTPUT + output) * GROUPS;
                        const float scale = dsv4_down_e8m0(downS[scaleBase + group]);
                        const uint wordBase =
                            rowBase + group * WORDS_PER_GROUP + halfLane * 2u;
                        const uint packed0 = downW[wordBase];
                        const uint packed1 = downW[wordBase + 1u];
                        float value;
                        if constexpr (HALF_MULTIPLY) {
                            const half4 product0 = half4(x0) * dsv4_down_fp4x4_half(packed0);
                            const half4 product1 = half4(x1) * dsv4_down_fp4x4_half(packed0 >> 16);
                            const half4 product2 = half4(x2) * dsv4_down_fp4x4_half(packed1);
                            const half4 product3 = half4(x3) * dsv4_down_fp4x4_half(packed1 >> 16);
                            value = dot(float4(product0), float4(1.0f))
                                + dot(float4(product1), float4(1.0f))
                                + dot(float4(product2), float4(1.0f))
                                + dot(float4(product3), float4(1.0f));
                        } else {
                            value = dot(x0, dsv4_down_fp4x4(packed0))
                                + dot(x1, dsv4_down_fp4x4(packed0 >> 16))
                                + dot(x2, dsv4_down_fp4x4(packed1))
                                + dot(x3, dsv4_down_fp4x4(packed1 >> 16));
                        }
                        routeSum[row] += scale * value;
                    }
                }

                for (uint row = 0u; row < ROWS; ++row) {
                    const float projected = simd_sum(routeSum[row]);
                    total[row] += static_cast<float>(static_cast<outT>(projected));
                }
            }

            if (lane == 0u) {
                for (uint row = 0u; row < ROWS && hidden + row < OUTPUT; ++row) {
                    reduced[hidden + row] = total[row];
                }
            }
        """,
        header: """
            constant float dsv4_down_fp4_lut[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
            };

            constant half dsv4_down_fp4_half_lut[16] = {
                0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
                -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
            };

            static inline float4 dsv4_down_fp4x4(uint packed) {
                return float4(
                    dsv4_down_fp4_lut[packed & 0xfu],
                    dsv4_down_fp4_lut[(packed >> 4) & 0xfu],
                    dsv4_down_fp4_lut[(packed >> 8) & 0xfu],
                    dsv4_down_fp4_lut[(packed >> 12) & 0xfu]);
            }

            static inline half4 dsv4_down_fp4x4_half(uint packed) {
                return half4(
                    dsv4_down_fp4_half_lut[packed & 0xfu],
                    dsv4_down_fp4_half_lut[(packed >> 4) & 0xfu],
                    dsv4_down_fp4_half_lut[(packed >> 8) & 0xfu],
                    dsv4_down_fp4_half_lut[(packed >> 12) & 0xfu]);
            }

            static inline float dsv4_down_e8m0(uchar exponent) {
                const uint bits = exponent == 0
                    ? 0x00400000u
                    : (uint(exponent) << 23);
                return as_type<float>(bits);
            }
        """)

    private static let cooperativeDownSum6Kernel = MLXFast.metalKernel(
        name: "deepseek_v4_native_mxfp4_cooperative_down_sum6",
        inputNames: ["activated", "downW", "downS", "indices"],
        outputNames: ["reduced"],
        source: """
            constexpr uint ROWS = ROWS_PER_SIMD;
            const uint lane = thread_index_in_simdgroup;
            const uint route = simdgroup_index_in_threadgroup;
            const uint hidden = threadgroup_position_in_grid.x * ROWS;
            if (hidden >= OUTPUT) {
                return;
            }

            threadgroup float partial[ROUTES * ROWS];
            float routeSum[ROWS] = {0.0f};
            const uint expert = static_cast<uint>(indices[route]);
            if (expert < EXPERTS) {
                const uint ix = lane >> 1u;
                const uint halfLane = lane & 1u;
                for (uint group = ix; group < GROUPS; group += 16u) {
                    const uint activationBase =
                        route * INPUT + group * GROUP_SIZE + halfLane * 16u;
                    const float4 x0(
                            static_cast<float>(activated[activationBase]),
                            static_cast<float>(activated[activationBase + 1u]),
                            static_cast<float>(activated[activationBase + 2u]),
                            static_cast<float>(activated[activationBase + 3u]));
                    const float4 x1(
                            static_cast<float>(activated[activationBase + 4u]),
                            static_cast<float>(activated[activationBase + 5u]),
                            static_cast<float>(activated[activationBase + 6u]),
                            static_cast<float>(activated[activationBase + 7u]));
                    const float4 x2(
                            static_cast<float>(activated[activationBase + 8u]),
                            static_cast<float>(activated[activationBase + 9u]),
                            static_cast<float>(activated[activationBase + 10u]),
                            static_cast<float>(activated[activationBase + 11u]));
                    const float4 x3(
                            static_cast<float>(activated[activationBase + 12u]),
                            static_cast<float>(activated[activationBase + 13u]),
                            static_cast<float>(activated[activationBase + 14u]),
                            static_cast<float>(activated[activationBase + 15u]));

                    for (uint row = 0u; row < ROWS && hidden + row < OUTPUT; ++row) {
                        const uint output = hidden + row;
                        const uint rowBase = (expert * OUTPUT + output) * PACKED_IN;
                        const uint scaleBase = (expert * OUTPUT + output) * GROUPS;
                        const float scale = dsv4_coop_down_e8m0(downS[scaleBase + group]);
                        const uint wordBase =
                            rowBase + group * WORDS_PER_GROUP + halfLane * 2u;
                        const uint packed0 = downW[wordBase];
                        const uint packed1 = downW[wordBase + 1u];
                        const float value =
                            dot(x0, dsv4_coop_down_fp4x4(packed0))
                            + dot(x1, dsv4_coop_down_fp4x4(packed0 >> 16))
                            + dot(x2, dsv4_coop_down_fp4x4(packed1))
                            + dot(x3, dsv4_coop_down_fp4x4(packed1 >> 16));
                        routeSum[row] += scale * value;
                    }
                }
            }

            for (uint row = 0u; row < ROWS; ++row) {
                const float projected = simd_sum(routeSum[row]);
                if (lane == 0u) {
                    partial[route * ROWS + row] =
                        static_cast<float>(static_cast<outT>(projected));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (route == 0u && lane == 0u) {
                for (uint row = 0u; row < ROWS && hidden + row < OUTPUT; ++row) {
                    float total = 0.0f;
                    for (uint sourceRoute = 0u; sourceRoute < ROUTES; ++sourceRoute) {
                        total += partial[sourceRoute * ROWS + row];
                    }
                    reduced[hidden + row] = total;
                }
            }
        """,
        header: """
            constant float dsv4_coop_down_fp4_lut[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
            };

            static inline float4 dsv4_coop_down_fp4x4(uint packed) {
                return float4(
                    dsv4_coop_down_fp4_lut[packed & 0xfu],
                    dsv4_coop_down_fp4_lut[(packed >> 4) & 0xfu],
                    dsv4_coop_down_fp4_lut[(packed >> 8) & 0xfu],
                    dsv4_coop_down_fp4_lut[(packed >> 12) & 0xfu]);
            }

            static inline float dsv4_coop_down_e8m0(uchar exponent) {
                const uint bits = exponent == 0
                    ? 0x00400000u
                    : (uint(exponent) << 23);
                return as_type<float>(bits);
            }
        """)

    static func fusedGateUpScoredSwiGLU(
        input: MLXArray,
        indices: MLXArray,
        scores: MLXArray,
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        limit: Float
    ) -> MLXArray? {
        guard SwitchGLUKernelEngine.specializedDeepseekMXFP4Enabled,
              input.dim(-1) == supportedInputDims,
              gate.inputDims == supportedInputDims,
              gate.outputDims == supportedHiddenDims,
              gate.numExperts == supportedExperts,
              up.inputDims == supportedInputDims,
              up.outputDims == supportedHiddenDims,
              up.numExperts == supportedExperts,
              indices.size == routeLimit,
              gate.groupSize == supportedGroupSize,
              up.groupSize == supportedGroupSize,
              gate.bits == 4,
              up.bits == 4,
              gate.mode == .mxfp4,
              up.mode == .mxfp4,
              gate.weight.dtype == .uint32,
              up.weight.dtype == .uint32,
              gate.scales.dtype == .uint8,
              up.scales.dtype == .uint8,
              gate.bias == nil,
              up.bias == nil,
              gate.biases == nil,
              up.biases == nil
        else {
            return nil
        }

        let activation = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(
            contiguous(input), mode: gate.mode
        )
        let flatActivation = contiguous(activation.flattened())
        let flatIndices = contiguous(indices.flattened())
        let flatScores = contiguous(scores.flattened())
        let outputShape = indices.shape + [1, supportedHiddenDims]
        let rowsPerSIMD = SwitchGLUKernelEngine.deepseekMXFP4RowsPerSIMD
        let simdGroups = SwitchGLUKernelEngine.deepseekMXFP4SIMDGroupsPerThreadgroup
        let halfMultiply = SwitchGLUKernelEngine.deepseekMXFP4GateUpHalfMultiplyEnabled ? 1 : 0
        let output = fusedGateUpScoredKernel(
            [
                flatActivation,
                contiguous(gate.weight),
                contiguous(gate.scales),
                contiguous(up.weight),
                contiguous(up.scales),
                flatIndices,
                flatScores,
            ],
            template: [
                ("outT", input.dtype),
                ("ROWS_PER_SIMD", rowsPerSIMD),
                ("ROUTES", routeLimit),
                ("EXPERTS", supportedExperts),
                ("HIDDEN", supportedHiddenDims),
                ("GROUP_SIZE", supportedGroupSize),
                ("GROUPS", supportedInputDims / supportedGroupSize),
                ("WORDS_PER_GROUP", supportedGroupSize / 8),
                ("PACKED_IN", supportedInputDims / 8),
                ("LIMIT", Int(limit)),
                ("HALF_MULTIPLY", halfMultiply),
            ],
            grid: (
                32 * routeLimit * ((supportedHiddenDims + rowsPerSIMD - 1) / rowsPerSIMD),
                1, 1),
            threadGroup: (32 * simdGroups, 1, 1),
            outputShapes: [outputShape],
            outputDTypes: [input.dtype]
        )[0]
        return output
    }

    static func fusedDownSum6(
        activated: MLXArray,
        indices: MLXArray,
        down: QuantizedSwitchLinear
    ) -> MLXArray? {
        guard SwitchGLUKernelEngine.specializedDeepseekMXFP4Enabled,
              down.inputDims == supportedHiddenDims,
              down.outputDims == supportedInputDims,
              down.numExperts == supportedExperts,
              indices.size == routeLimit,
              down.groupSize == supportedGroupSize,
              down.bits == 4,
              down.mode == .mxfp4,
              down.weight.dtype == .uint32,
              down.scales.dtype == .uint8,
              down.bias == nil,
              down.biases == nil
        else {
            return nil
        }

        let prepared = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(
            activated, mode: down.mode)
        let flatActivated = contiguous(prepared.flattened())
        let flatIndices = contiguous(indices.flattened())
        // This kernel has already reduced all six routes. Return the same rank
        // as the input activation so DeepSeek V4 can skip the generic
        // route-axis reduction instead of summing a synthetic size-one axis.
        let outputShape = Array(indices.shape.dropLast()) + [1, supportedInputDims]
        let rowsPerSIMD = SwitchGLUKernelEngine.deepseekMXFP4RowsPerSIMD
        if SwitchGLUKernelEngine.deepseekMXFP4CooperativeDownEnabled {
            return cooperativeDownSum6Kernel(
                [
                    flatActivated,
                    contiguous(down.weight),
                    contiguous(down.scales),
                    flatIndices,
                ],
                template: [
                    ("outT", activated.dtype),
                    ("ROWS_PER_SIMD", rowsPerSIMD),
                    ("ROUTES", routeLimit),
                    ("EXPERTS", supportedExperts),
                    ("INPUT", supportedHiddenDims),
                    ("OUTPUT", supportedInputDims),
                    ("GROUP_SIZE", supportedGroupSize),
                    ("GROUPS", supportedHiddenDims / supportedGroupSize),
                    ("WORDS_PER_GROUP", supportedGroupSize / 8),
                    ("PACKED_IN", supportedHiddenDims / 8),
                ],
                grid: (
                    32 * routeLimit
                        * ((supportedInputDims + rowsPerSIMD - 1) / rowsPerSIMD),
                    1, 1),
                threadGroup: (32 * routeLimit, 1, 1),
                outputShapes: [outputShape],
                outputDTypes: [.float32]
            )[0]
        }
        let simdGroups = SwitchGLUKernelEngine.deepseekMXFP4SIMDGroupsPerThreadgroup
        let halfMultiply = SwitchGLUKernelEngine.deepseekMXFP4DownHalfMultiplyEnabled ? 1 : 0
        return fusedDownSum6Kernel(
            [
                flatActivated,
                contiguous(down.weight),
                contiguous(down.scales),
                flatIndices,
            ],
            template: [
                ("outT", activated.dtype),
                ("ROWS_PER_SIMD", rowsPerSIMD),
                ("ROUTES", routeLimit),
                ("EXPERTS", supportedExperts),
                ("INPUT", supportedHiddenDims),
                ("OUTPUT", supportedInputDims),
                ("GROUP_SIZE", supportedGroupSize),
                ("GROUPS", supportedHiddenDims / supportedGroupSize),
                ("WORDS_PER_GROUP", supportedGroupSize / 8),
                ("PACKED_IN", supportedHiddenDims / 8),
                ("HALF_MULTIPLY", halfMultiply),
            ],
            grid: (32 * ((supportedInputDims + rowsPerSIMD - 1) / rowsPerSIMD), 1, 1),
            threadGroup: (32 * simdGroups, 1, 1),
            outputShapes: [outputShape],
            outputDTypes: [.float32]
        )[0]
    }
}

// MARK: - SwitchGLU

public protocol SwitchGLULayer: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
}

public class SwitchGLU: Module, SwitchGLULayer {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let isSiluActivation: Bool
    let isGeluActivation: Bool
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Optional 2-argument GLU closure that takes `(gate, up)` and returns
    /// the activated `gate * up` result. When non-nil, this OVERRIDES
    /// the standard `activation(gate) * up` path (and the compiled
    /// SwiGLU/GeGLU fast-paths) so DSV4 can apply
    /// `silu(min(gate, 10)) * clip(up, -10, 10)` — symmetric clamping
    /// of BOTH gate and up that the one-arg `activation` API can only
    /// express on `gate`. Every other caller passes `nil` and gets the
    /// historical bit-for-bit-identical fast paths.
    let glue: ((MLXArray, MLXArray) -> MLXArray)?
    /// Optional model-specific activation that applies a per-route score
    /// before the down projection. DSV4-0731 requires this ordering because
    /// its down projection is quantized; scaling the projection output later
    /// is not the checkpoint graph.
    let scoredGlue: ((MLXArray, MLXArray, MLXArray) -> MLXArray)?
    let scoredSwiGLULimit: Float?

    // Lazy fused gate+up gatherQuantizedMM cache.
    //
    // When both gate_proj and up_proj are QuantizedSwitchLinear with
    // matching (groupSize, bits, mode), we concatenate their weight,
    // scales and biases along the output axis once on first forward
    // and run a single `gatherQuantizedMM` for gate+up instead of two.
    // The compiled SwiGLU/GeGLU then splits the result and multiplies.
    //
    // Why: the standard 4-bit Qwen 3.5 / MiniMax / GLM4 MoE path dispatches
    // 3 separate gatherQuantizedMM Metal kernels per layer (gate, up, down).
    // At 40 layers × 100 tok/s that is 12,000 dispatches/sec just for MoE.
    // Halving the gate+up dispatches to one wider matmul saves one
    // Metal dispatch per layer per step, and the wider matmul has better
    // GPU occupancy because more output tiles share the same input read.
    //
    // Matches the `gate_up_proj` fusion mlx-community models sometimes
    // pre-bake into weights, and the JANGTQ fused gate_up SwiGLU kernel
    // we already ship for the TurboQuant path. See the optimization plan
    // doc § 6 "Int4 — Batched multi-expert gather for MoE".
    //
    // Disabled via `BENCH_NO_FUSED_GATE_UP=1` env var for A/B.
    private var fusedGateUpWeight: MLXArray? = nil
    private var fusedGateUpScales: MLXArray? = nil
    private var fusedGateUpBiases: MLXArray? = nil
    private var fusedGroupSize: Int = 64
    private var fusedBits: Int = 4
    private var fusedMode: QuantizationMode = .affine
    private var fusionAttempted: Bool = false

    private static var profileStages: Bool {
        ProcessInfo.processInfo.environment["VMLX_DSV4_STAGE_PROFILE"] == "1"
    }

    private static var sharedGateUpActivationEnabled: Bool {
        let raw = ProcessInfo.processInfo.environment["VMLX_SHARED_GATE_UP_ACTIVATION"] ?? "1"
        return raw != "0" && raw.lowercased() != "false"
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false,
        glue: ((MLXArray, MLXArray) -> MLXArray)? = nil,
        scoredGlue: ((MLXArray, MLXArray, MLXArray) -> MLXArray)? = nil,
        scoredSwiGLULimit: Float? = nil
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.glue = glue
        self.scoredGlue = scoredGlue
        self.scoredSwiGLULimit = scoredSwiGLULimit
        // Detect common activation types for compiled fast path.
        // Use safeGeluApproximate for comparison to avoid MLXNN's compiledGeluApproximate
        // which uses the Power primitive (x ** 3) and crashes on some Metal GPUs during
        // model load time — see comment on safeGeluApproximate above.
        let testInput = MLXArray([Float(1.0)])
        let testOutput = activation(testInput)
        let siluOutput = silu(testInput)
        let geluOutput = safeGeluApproximate(testInput)
        self.isSiluActivation = (testOutput .== siluOutput).all().item(Bool.self)
        self.isGeluActivation = !isSiluActivation && (testOutput .== geluOutput).all().item(Bool.self)

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Populate the fused gate+up weight cache on first forward. Safe to
    /// call multiple times — guarded by `fusionAttempted` so the work runs
    /// exactly once per SwitchGLU instance.
    private func ensureFusedGateUp() {
        if fusionAttempted { return }
        fusionAttempted = true

        // Feature flag — opt out for A/B comparison.
        if ProcessInfo.processInfo.environment["BENCH_NO_FUSED_GATE_UP"] == "1" {
            return
        }

        guard let g = gateProj as? QuantizedSwitchLinear,
              let u = upProj as? QuantizedSwitchLinear,
              g.groupSize == u.groupSize,
              g.bits == u.bits,
              g.mode == u.mode
        else {
            // Non-quantized or mismatched quantization params — can't fuse.
            return
        }

        let fusedBytes =
            g.weight.nbytes + u.weight.nbytes
            + g.scales.nbytes + u.scales.nbytes
            + (g.biases?.nbytes ?? 0) + (u.biases?.nbytes ?? 0)
        let cacheLimit = fusedGateUpCacheByteLimit()
        if cacheLimit >= 0 && fusedBytes > cacheLimit {
            return
        }

        // Concatenate along output axis. Quantized SwitchLinear weights are
        // shaped `[E, out, in_packed]`, so axis -2 stacks gate and up along
        // the output dimension, giving `[E, 2*hidden, in_packed]`. scales
        // and biases track the same output axis at group granularity.
        let fusedW = concatenated([g.weight, u.weight], axis: -2)
        let fusedS = concatenated([g.scales, u.scales], axis: -2)
        var fusedB: MLXArray? = nil
        if let gb = g.biases, let ub = u.biases {
            fusedB = concatenated([gb, ub], axis: -2)
        }

        // Force materialization now so the first forward pass doesn't pay
        // the concat cost mid-generation.
        var toMaterialize: [MLXArray] = [fusedW, fusedS]
        if let fb = fusedB { toMaterialize.append(fb) }
        MLX.eval(toMaterialize)

        self.fusedGateUpWeight = fusedW
        self.fusedGateUpScales = fusedS
        self.fusedGateUpBiases = fusedB
        self.fusedGroupSize = g.groupSize
        self.fusedBits = g.bits
        self.fusedMode = g.mode
    }

    private func fusedGateUpCacheByteLimit() -> Int {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["VMLX_FUSED_GATE_UP_CACHE_LIMIT_BYTES"],
            let bytes = Int(raw)
        {
            return bytes
        }
        if let raw = env["VMLX_FUSED_GATE_UP_CACHE_LIMIT_MB"],
            let mb = Int(raw)
        {
            return mb < 0 ? -1 : mb * 1024 * 1024
        }
        // Keep the decode micro-fusion for normal-sized MoE layers, but do
        // not let it duplicate giant routed expert banks. Ling MXFP4's fused
        // gate+up tensor is ~1 GiB per layer, which doubled production
        // footprint without being required for correctness.
        return 512 * 1024 * 1024
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        callAsFunction(x, indices, preDownScores: nil)
    }

    /// Decode-only DeepSeek V4 experiment that keeps route selection and the
    /// routed MXFP4 kernels in one primitive. Metadata gates the exact tensor
    /// contract; all other model geometries and quantizations return `nil`.
    public func deepseekV4FusedSelectingMoE(
        _ input: MLXArray,
        logits: MLXArray,
        bias: MLXArray,
        routeScale: MLXArray
    ) -> MLXArray? {
        guard let limit = scoredSwiGLULimit,
              let gate = gateProj as? QuantizedSwitchLinear,
              let up = upProj as? QuantizedSwitchLinear,
              let down = downProj as? QuantizedSwitchLinear
        else { return nil }
        return DeepseekV4DS4Kernels.fusedRoutedMoESelecting(
            input: input,
            logits: logits,
            bias: bias,
            routeScale: routeScale,
            gate: gate,
            up: up,
            down: down,
            limit: limit)
    }

    public func deepseekV4FusedSelectingMoEWithSharedQ8(
        _ input: MLXArray,
        logits: MLXArray,
        bias: MLXArray,
        routeScale: MLXArray,
        sharedGate: DeepseekV4QuantizedLinear,
        sharedUp: DeepseekV4QuantizedLinear,
        sharedDown: DeepseekV4QuantizedLinear
    ) -> MLXArray? {
        guard let limit = scoredSwiGLULimit,
              let gate = gateProj as? QuantizedSwitchLinear,
              let up = upProj as? QuantizedSwitchLinear,
              let down = downProj as? QuantizedSwitchLinear
        else { return nil }
        return DeepseekV4DS4Kernels.fusedRoutedMoESelectingWithSharedQ8(
            input: input,
            logits: logits,
            bias: bias,
            routeScale: routeScale,
            gate: gate,
            up: up,
            down: down,
            sharedGate: sharedGate,
            sharedUp: sharedUp,
            sharedDown: sharedDown,
            limit: limit)
    }

    /// Variant for model graphs that weight each routed activation before its
    /// expert down projection. The score tensor has the same leading shape as
    /// `indices`; sorting keeps scores aligned with the expert dispatch rows.
    public func callAsFunction(
        _ input: MLXArray,
        _ indices: MLXArray,
        preDownScores: MLXArray?
    ) -> MLXArray {
        ensureFusedGateUp()

        let profileStages = Self.profileStages && indices.size <= 32
        var stageStart = profileStages ? CFAbsoluteTimeGetCurrent() : 0
        func finishStage(_ name: String, _ arrays: [MLXArray]) {
            guard profileStages else { return }
            MLX.eval(arrays)
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(Data(String(format:
                "[SwitchGLUProfile] routes=%d stage=%@ ms=%.3f\n",
                indices.size, name, (now - stageStart) * 1_000).utf8))
            stageStart = now
        }

        // Fused gate+up is a net win for DECODE (single-token forward pass,
        // compute-bound per-expert matmul) but a net LOSS for PREFILL
        // (multi-token batches are memory-bandwidth bound, and the single
        // wider matmul has worse cache locality than two narrower ones).
        //
        // Decide per-call which path to take. indices.size is the number
        // of (token, expert) dispatches: at decode with B=1 and top_k=8
        // it's 8; at prefill with 512 tokens and top_k=8 it's 4096. The
        // threshold (32 by default) admits single-token + a few prompt
        // tokens as "decode-shaped" and bounces large prefill chunks to
        // the two-call path. Override via BENCH_FUSED_GATE_UP_THRESHOLD.
        let decodeThreshold: Int =
            Int(ProcessInfo.processInfo.environment["BENCH_FUSED_GATE_UP_THRESHOLD"] ?? "32") ?? 32
        let useFused =
            (fusedGateUpWeight != nil)
            && (indices.size <= decodeThreshold)

        let inputDType = input.dtype
        var x = MLX.expandedDimensions(input, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()
        var alignedScores = preDownScores

        if doSort {
            if let scores = alignedScores {
                let scoreOrder = argSort(indices.flattened())
                alignedScores = scores.flattened()[scoreOrder]
            }
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        func activate(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
            if let scores = alignedScores, let scoredGlue {
                return scoredGlue(gate, up, scores)
            }
            if let glue {
                return glue(gate, up)
            }
            if isSiluActivation {
                return compiledSwiGLU(gate, up)
            }
            if isGeluActivation {
                return compiledGeGLU(gate, up)
            }
            return activation(gate) * up
        }

        if !doSort,
           let scores = alignedScores,
           let limit = scoredSwiGLULimit,
           let gate = gateProj as? QuantizedSwitchLinear,
           let up = upProj as? QuantizedSwitchLinear,
           let down = downProj as? QuantizedSwitchLinear,
           let reduced = DeepseekV4DS4Kernels.fusedRoutedMoE(
               input: input,
               indices: idx,
               scores: scores,
               gate: gate,
               up: up,
               down: down,
               limit: limit)
        {
            finishStage("native_mxfp4_staged_moe", [reduced])
            return reduced
        }

        var activated: MLXArray
        if !doSort,
           let scores = alignedScores,
           let limit = scoredSwiGLULimit,
           let gate = gateProj as? QuantizedSwitchLinear,
           let up = upProj as? QuantizedSwitchLinear,
           let ds4Activated = DeepseekV4DS4Kernels.fusedGateUpScoredSwiGLU(
               input: input,
               indices: idx,
               scores: scores,
               gate: gate,
               up: up,
               limit: limit
           )
        {
            activated = ds4Activated
            finishStage("ds4_gate_up_scored_swiglu", [activated])
        } else if useFused, let fusedW = fusedGateUpWeight, let fusedS = fusedGateUpScales {
            // FUSED PATH — single gatherQuantizedMM for gate+up, then
            // split along output axis and apply compiled SwiGLU.
            // Decode-only per the threshold check above.
            let quantizedInput = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(x, mode: fusedMode)
            let combined = MLX.gatherQuantizedMM(
                quantizedInput, fusedW,
                scales: fusedS, biases: fusedGateUpBiases,
                rhsIndices: idx, transpose: true,
                groupSize: fusedGroupSize, bits: fusedBits, mode: fusedMode,
                sortedIndices: doSort)
            let splits = MLX.split(combined, parts: 2, axis: -1)
            let xGate = splits[0]
            let xUp = splits[1]
            finishStage("gate_up", [xGate, xUp])
            activated = activate(xGate, xUp)
        } else {
            // FALLBACK — original two-call path for non-quantized models,
            // prefill batches (indices.size > threshold), or when the
            // feature flag is off.
            let xUp: MLXArray
            let xGate: MLXArray
            if Self.sharedGateUpActivationEnabled,
               let gate = gateProj as? QuantizedSwitchLinear,
               let up = upProj as? QuantizedSwitchLinear,
               gate.groupSize == up.groupSize,
               gate.bits == up.bits,
               gate.mode == up.mode,
               DeepseekV4ActivationQuant.isMXFP(gate.mode)
            {
                let prepared = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(
                    x, mode: gate.mode)
                finishStage("gate_up_activation", [prepared])
                xUp = up.projectPreparedActivation(
                    prepared, idx, sortedIndices: doSort)
                finishStage("up", [xUp])
                xGate = gate.projectPreparedActivation(
                    prepared, idx, sortedIndices: doSort)
                finishStage("gate", [xGate])
            } else {
                xUp = upProj(x, idx, sortedIndices: doSort)
                finishStage("up", [xUp])
                xGate = gateProj(x, idx, sortedIndices: doSort)
                finishStage("gate", [xGate])
            }
            activated = activate(xGate, xUp)
        }

        finishStage("activation", [activated])

        // Generic fallback for a caller that supplies pre-down scores without
        // a fused scored activation. DSV4 supplies `scoredGlue`, so its clamp,
        // SiLU, route weighting, and cast execute in the exact official order.
        if let scores = alignedScores, scoredGlue == nil {
            activated = (
                activated.asType(.float32)
                    * scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
            ).asType(inputDType)
        }

        if !doSort,
           let down = downProj as? QuantizedSwitchLinear,
           let reduced = DeepseekV4DS4Kernels.fusedDownSum6(
               activated: activated,
               indices: idx,
               down: down)
        {
            finishStage("native_mxfp4_down_sum6", [reduced])
            return MLX.squeezed(reduced, axis: -2)
        }

        x = downProj(activated, idx, sortedIndices: doSort)
        finishStage("down", [x])

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    /// Initializer for already-quantized checkpoint tensors.
    ///
    /// Loading a pre-quantized safetensors bundle should not quantize the
    /// randomly initialized `SwitchLinear` placeholder just to replace it
    /// with file weights a few lines later. This initializer lets the loader
    /// swap in the quantized module using the real checkpoint arrays
    /// immediately, which avoids a full throwaway routed-MoE allocation.
    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray? = nil,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias)
        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let activation = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(x, mode: mode)
        return projectPreparedActivation(activation, indices, sortedIndices: sortedIndices)
    }

    /// Projects an activation that has already passed through the checkpoint's
    /// MXFP activation preparation. This lets compatible gate/up projections
    /// share that work without combining or duplicating their weight banks.
    public func projectPreparedActivation(
        _ activation: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            activation,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
