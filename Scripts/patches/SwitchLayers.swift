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
}

private enum DeepseekV4DS4Kernels {
    private static let routeLimit = 6
    private static let supportedInputDims = 4096
    private static let supportedHiddenDims = 2048
    private static let supportedExperts = 256
    private static let supportedGroupSize = 32

    private static let fusedGateUpScoredKernel = MLXFast.metalKernel(
        name: "deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu",
        inputNames: [
            "x", "gateW", "gateS", "upW", "upS", "indices", "scores",
        ],
        outputNames: ["activated"],
        source: """
            const uint linear = thread_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = linear / 32;
            const uint hidden = simd % HIDDEN;
            const uint route = simd / HIDDEN;
            if (route >= ROUTES || hidden >= HIDDEN) {
                return;
            }

            const uint expert = static_cast<uint>(indices[route]);
            if (expert >= EXPERTS) {
                return;
            }

            float gateSum = 0.0f;
            float upSum = 0.0f;
            const uint rowBase = (expert * HIDDEN + hidden) * PACKED_IN;
            const uint scaleBase = (expert * HIDDEN + hidden) * GROUPS;

            for (uint group = lane; group < GROUPS; group += 32u) {
                const float gateScale = dsv4_e8m0(gateS[scaleBase + group]);
                const float upScale = dsv4_e8m0(upS[scaleBase + group]);
                const uint wordBase = rowBase + group * WORDS_PER_GROUP;
                const uint activationBase = group * GROUP_SIZE;

                for (uint word = 0; word < WORDS_PER_GROUP; ++word) {
                    uint gatePacked = gateW[wordBase + word];
                    uint upPacked = upW[wordBase + word];
                    for (uint nib = 0; nib < 8u; ++nib) {
                        const uint inputOffset = activationBase + word * 8u + nib;
                        const float xv = static_cast<float>(x[inputOffset]);
                        gateSum += xv * gateScale * dsv4_fp4_value(gatePacked & 0xfu);
                        upSum += xv * upScale * dsv4_fp4_value(upPacked & 0xfu);
                        gatePacked >>= 4;
                        upPacked >>= 4;
                    }
                }
            }

            gateSum = simd_sum(gateSum);
            upSum = simd_sum(upSum);

            if (lane == 0) {
                const float activationLimit = static_cast<float>(LIMIT);
                const float limitedGate = metal::min(gateSum, activationLimit);
                const float siluGate = limitedGate / (1.0f + metal::fast::exp(-limitedGate));
                const float clippedUp = metal::clamp(upSum, -activationLimit, activationLimit);
                const float routed = siluGate * clippedUp * static_cast<float>(scores[route]);
                activated[route * HIDDEN + hidden] = static_cast<outT>(routed);
            }
        """,
        header: """
            static inline float dsv4_fp4_value(uint nibble) {
                switch (nibble & 0xfu) {
                case 0u: return 0.0f;
                case 1u: return 0.5f;
                case 2u: return 1.0f;
                case 3u: return 1.5f;
                case 4u: return 2.0f;
                case 5u: return 3.0f;
                case 6u: return 4.0f;
                case 7u: return 6.0f;
                case 8u: return -0.0f;
                case 9u: return -0.5f;
                case 10u: return -1.0f;
                case 11u: return -1.5f;
                case 12u: return -2.0f;
                case 13u: return -3.0f;
                case 14u: return -4.0f;
                default: return -6.0f;
                }
            }

            static inline float dsv4_e8m0(uchar exponent) {
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
        guard SwitchGLUKernelEngine.ds4Enabled,
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
                ("ROUTES", routeLimit),
                ("EXPERTS", supportedExperts),
                ("HIDDEN", supportedHiddenDims),
                ("GROUP_SIZE", supportedGroupSize),
                ("GROUPS", supportedInputDims / supportedGroupSize),
                ("WORDS_PER_GROUP", supportedGroupSize / 8),
                ("PACKED_IN", supportedInputDims / 8),
                ("LIMIT", Int(limit)),
            ],
            grid: (32 * routeLimit * supportedHiddenDims, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [outputShape],
            outputDTypes: [input.dtype]
        )[0]
        return output
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
            let xUp = upProj(x, idx, sortedIndices: doSort)
            finishStage("up", [xUp])
            let xGate = gateProj(x, idx, sortedIndices: doSort)
            finishStage("gate", [xGate])
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
