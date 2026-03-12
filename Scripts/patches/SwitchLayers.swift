import Foundation
import MLX
import MLXFast
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py
// Patched: gate+up weight fusion + fused silu_mul Metal kernel

// MARK: - Fused SiLU-Multiply Metal Kernel

/// Fused kernel: given concatenated [gate, up] tensor of shape [..., 2*H],
/// computes silu(gate) * up → [..., H] in a single dispatch.
/// Replaces 4 graph nodes (slice + slice + silu + multiply) with 1 kernel.
private func makeFusedSiluMulKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        uint tid = thread_position_in_grid.x;
        if (tid >= total_elems) return;

        uint h = tid % HIDDEN;
        uint batch = tid / HIDDEN;
        uint base = batch * (2 * HIDDEN);

        float gate_val = static_cast<float>(gate_up[base + h]);
        float up_val = static_cast<float>(gate_up[base + HIDDEN + h]);

        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        out[tid] = static_cast<InT>(silu_gate * up_val);
    """

    return MLXFast.metalKernel(
        name: "fused_silu_mul",
        inputNames: ["gate_up", "total_elems"],
        outputNames: ["out"],
        source: source
    )
}

private final class FusedSiluMulKernelManager: @unchecked Sendable {
    static let shared = FusedSiluMulKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeFusedSiluMulKernel()
    }
}

/// Apply fused silu-multiply on a concatenated gate+up tensor.
/// Input shape: [..., 2*hiddenDims], output shape: [..., hiddenDims]
public func fusedSiluMul(_ gateUp: MLXArray, hiddenDims: Int) -> MLXArray {
    guard let kernel = FusedSiluMulKernelManager.shared.kernel else {
        // Fallback to standard ops
        let g = gateUp[.ellipsis, ..<hiddenDims]
        let u = gateUp[.ellipsis, hiddenDims...]
        return silu(g) * u
    }

    let shape = gateUp.shape
    // Output shape: same as input but last dim halved
    var outShape = shape
    outShape[outShape.count - 1] = hiddenDims

    let totalElems = outShape.reduce(1, *)
    let threadsPerGroup = min(256, totalElems)
    let numGroups = (totalElems + threadsPerGroup - 1) / threadsPerGroup

    let outputs = kernel(
        [gateUp, MLXArray(Int32(totalElems))],
        template: [
            ("InT", gateUp.dtype),
            ("HIDDEN", hiddenDims),
        ],
        grid: (numGroups * threadsPerGroup, 1, 1),
        threadGroup: (threadsPerGroup, 1, 1),
        outputShapes: [outShape],
        outputDTypes: [gateUp.dtype]
    )
    return outputs[0]
}


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

// MARK: - SwitchGLU (with fused gate+up dispatch)

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    // Fused gate+up quantized weights (lazily created on first forward pass)
    private var fusedWeight: MLXArray?
    private var fusedScales: MLXArray?
    private var fusedBiases: MLXArray?
    private var fusedGroupSize: Int = 0
    private var fusedBits: Int = 0
    private var fusedMode: QuantizationMode = .affine
    private var fusionAttempted = false

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Lazily fuse gate_proj and up_proj weights for quantized models.
    /// Concatenates along the output dimension so a single gatherQuantizedMM
    /// replaces two separate dispatches.
    /// Skipped when memory headroom is tight (< 15% free) to avoid OOM on huge models.
    private func tryFuseGateUp() {
        guard !fusionAttempted else { return }
        fusionAttempted = true

        guard let qGate = gateProj as? QuantizedSwitchLinear,
              let qUp = upProj as? QuantizedSwitchLinear,
              qGate.groupSize == qUp.groupSize,
              qGate.bits == qUp.bits,
              qGate.mode == qUp.mode
        else { return }

        // Check memory headroom: fusion duplicates gate+up weights, so skip when tight.
        // The fused tensor is as large as gate+up combined, held alongside the originals.
        // For huge models (GLM-5 at 390GB on 512GB), this extra memory causes OOM.
        let snap = Memory.snapshot()
        let maxWorkingSet = GPU.deviceInfo().maxRecommendedWorkingSetSize
        let headroom = maxWorkingSet > 0
            ? Double(Int(maxWorkingSet) - snap.activeMemory) / Double(maxWorkingSet)
            : 1.0
        if headroom < 0.20 {
            // Not enough headroom — fall back to separate dispatches
            return
        }

        // Concatenate along output dimension (axis 1): [E, N, K_packed] → [E, 2N, K_packed]
        fusedWeight = concatenated([qGate.weight, qUp.weight], axis: 1)
        fusedScales = concatenated([qGate.scales, qUp.scales], axis: 1)
        if let gBiases = qGate.biases, let uBiases = qUp.biases {
            fusedBiases = concatenated([gBiases, uBiases], axis: 1)
        }
        fusedGroupSize = qGate.groupSize
        fusedBits = qGate.bits
        fusedMode = qGate.mode

        // Force evaluate the fused tensors so they're materialized once
        if let fw = fusedWeight, let fs = fusedScales {
            var toEval: [MLXArray] = [fw, fs]
            if let fb = fusedBiases { toEval.append(fb) }
            eval(toEval)
        }
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        tryFuseGateUp()

        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let result: MLXArray
        if let fWeight = fusedWeight, let fScales = fusedScales {
            // Fused path: single gatherQuantizedMM for gate+up
            let gateUp = MLX.gatherQuantizedMM(
                x,
                fWeight,
                scales: fScales,
                biases: fusedBiases,
                rhsIndices: idx,
                transpose: true,
                groupSize: fusedGroupSize,
                bits: fusedBits,
                mode: fusedMode,
                sortedIndices: doSort
            )

            // Fused silu-multiply: replaces slice+slice+silu+mul with 1 Metal kernel
            let activated = fusedSiluMul(gateUp, hiddenDims: hiddenDims)
            result = downProj(activated, idx, sortedIndices: doSort)
        } else {
            // Fallback: separate dispatches (non-quantized models)
            let xUp = upProj(x, idx, sortedIndices: doSort)
            let xGate = gateProj(x, idx, sortedIndices: doSort)
            result = downProj(
                activation(xGate) * xUp,
                idx,
                sortedIndices: doSort)
        }

        var out = result
        if doSort {
            out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(out, axis: -2)
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
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

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

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
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
