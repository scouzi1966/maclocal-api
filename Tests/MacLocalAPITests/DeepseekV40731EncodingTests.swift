import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest
@testable import AFMKitMLX

final class DeepseekV40731EncodingTests: XCTestCase {
    override func setUpWithError() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    func testChatPromptMatchesOfficial0731Encoder() throws {
        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "user", "content": "Reply exactly: OK"]],
            tools: nil,
            additionalContext: ["enable_thinking": false],
            addGenerationPrompt: true
        )

        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Reply exactly: OK<｜Assistant｜></think>"
        )
    }

    func testHighReasoningPromptMatchesOfficial0731Encoder() throws {
        let prompt = try DeepseekV4ChatEncoder.renderOpenAIChat(
            messages: [["role": "user", "content": "Reply exactly: OK"]],
            tools: nil,
            additionalContext: ["reasoning_effort": "high"],
            addGenerationPrompt: true
        )

        XCTAssertTrue(prompt.hasPrefix(
            "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        ))
        XCTAssertTrue(prompt.hasSuffix(
            "\n<｜User｜>Reply exactly: OK<｜Assistant｜><think>"
        ))
    }

    func testScoredSwiGLUPreservesSortedPrefillRouteCount() {
        let tokenCount = 16
        let topK = 6
        let routeCount = tokenCount * topK
        let hiddenSize = 8

        // Match the failing 0731 prefill geometry: 96 route scores whose
        // expert ids contain 78 distinct values after global sorting.
        let routeIDs = (0..<routeCount).map { UInt32($0 % 78) }
        let indices = MLXArray(routeIDs).reshaped(1, tokenCount, topK)
        let input = MLXArray.ones([1, tokenCount, hiddenSize])
        let expanded = MLX.expandedDimensions(input, axes: [-2, -3])
        let (sortedInput, sortedIndices, _) = gatherSort(
            x: expanded, indices: indices)

        let weight = MLXArray.ones([78, hiddenSize, hiddenSize])
        let projected = MLX.gatherMM(
            sortedInput,
            weight.swappedAxes(-1, -2),
            rhsIndices: sortedIndices,
            sortedIndices: true)
        let scoreOrder = argSort(indices.flattened())
        let scores = MLXArray.ones([routeCount])[scoreOrder]

        let output = DeepseekV4Math.dsv4ScoredSwiGLU(
            gate: projected,
            up: projected,
            scores: scores,
            limit: 10)
        MLX.eval(output)

        XCTAssertEqual(output.dim(0), routeCount)
        XCTAssertEqual(output.shape.last, hiddenSize)
    }

    func testScoredSwiGLUDecodeMatchesReference() {
        let gate = MLXArray([Float](arrayLiteral:
            -12, -2, 0, 2, 12,
            8, -8, 4, -4, 1,
            0.5, 10, -10, 3, -3,
            7, -7, 0.25, -0.25, 11,
            6, -6, 5, -5, 9,
            2.5, -2.5, 1.5, -1.5, 0,
        )).reshaped(6, 1, 5)
        let up = MLXArray([Float](arrayLiteral:
            12, -12, 2, -2, 0.5,
            -11, 11, -4, 4, 1,
            9, -9, 3, -3, 0.25,
            -8, 8, -5, 5, 2,
            7, -7, 6, -6, 10,
            -1, 1, -0.5, 0.5, 12,
        )).reshaped(6, 1, 5)
        let scores = MLXArray([Float](arrayLiteral: 0.05, 0.1, 0.15, 0.2, 0.23, 0.27))
        let limit = MLXArray(Float(10))

        let actual = DeepseekV4Math.dsv4ScoredSwiGLU(
            gate: gate, up: up, scores: scores, limit: 10)
        let gate32 = gate.asType(.float32)
        let up32 = up.asType(.float32)
        let reference = (
            MLXNN.silu(MLX.minimum(gate32, limit))
                * MLX.clip(up32, min: -10, max: 10)
                * scores.asType(.float32)[.ellipsis, .newAxis, .newAxis]
        ).asType(gate.dtype)
        MLX.eval(actual, reference)

        XCTAssertEqual(actual.shape, [6, 1, 5])
        XCTAssertTrue(
            MLX.allClose(actual, reference, rtol: 1e-5, atol: 1e-6).item(Bool.self)
        )
    }

    func testSharedMXFP4ActivationPreparationMatchesIndependentProjection() {
        let inputDims = 32
        let outputDims = 8
        let experts = 3
        let routes = 6
        let weight = MLXArray((0..<(experts * outputDims * inputDims)).map {
            Float(($0 % 31) - 15) * 0.004
        }).reshaped(experts, outputDims, inputDims)
        let base = SwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: experts,
            weight: weight)
        let projection = QuantizedSwitchLinear(
            base, groupSize: 32, bits: 4, mode: .mxfp4)
        let input = MLXArray((0..<(routes * inputDims)).map {
            Float(($0 % 37) - 18) * 0.013
        }).reshaped(routes, 1, inputDims)
        let indices = MLXArray([UInt32](arrayLiteral: 0, 0, 1, 1, 2, 2))
        let prepared = DeepseekV4ActivationQuant.e4m3RoundTripIfNeeded(
            input, mode: .mxfp4)

        for sorted in [false, true] {
            let independent = projection(
                input, indices, sortedIndices: sorted)
            let shared = projection.projectPreparedActivation(
                prepared, indices, sortedIndices: sorted)
            MLX.eval(independent, shared)
            XCTAssertTrue(
                MLX.allClose(independent, shared, rtol: 0, atol: 0).item(Bool.self),
                "shared activation mismatch with sortedIndices=\(sorted)")
        }
    }

    func testSelectedHashExpertLogitsMatchFullFP32Gate() {
        let hiddenSize = 32
        let experts = 17
        let input = MLXArray((0..<hiddenSize).map {
            Float(($0 % 13) - 6) * 0.019
        }).reshaped(1, 1, hiddenSize)
        let weight = MLXArray((0..<(experts * hiddenSize)).map {
            Float(($0 % 23) - 11) * 0.007
        }).reshaped(experts, hiddenSize)
        let indices = MLXArray([Int32](arrayLiteral: 1, 4, 7, 9, 13, 16))
            .reshaped(1, 1, 6)

        let full = input.asType(.float32).matmul(weight.asType(.float32).transposed())
        let expected = takeAlong(full, indices, axis: -1)
        let selected = DeepseekV4Math.selectedExpertLogits(
            input: input,
            weight: weight,
            indices: indices)
        MLX.eval(expected, selected)

        XCTAssertEqual(selected.shape, indices.shape)
        XCTAssertTrue(
            MLX.allClose(selected, expected, rtol: 1e-5, atol: 1e-6).item(Bool.self),
            "selected hash gate max error: \(MLX.max(MLX.abs(selected - expected)).item(Float.self))")
    }

    func testFusedRouterMatchesGenericExpertSetAndWeights() throws {
        let experts = 256
        let topK = 6
        let logits = MLXArray((0..<experts).map { index in
            sin(Float(index) * 0.173) * 3.7 + Float(index) * 0.00031
        }).reshaped(1, 1, experts)
        let bias = MLXArray((0..<experts).map { index in
            cos(Float(index) * 0.097) * 0.11
        })
        let scale = MLXArray([Float(2.5)])

        let original = DeepseekV4Math.sqrtSoftplus(logits)
        let genericIndices = argPartition(
            -(original + bias), kth: topK - 1, axis: -1)[.ellipsis, 0..<topK]
            .asType(.int32)
        let genericWeights = takeAlong(original, genericIndices, axis: -1)
        let normalizedGeneric = genericWeights
            / (genericWeights.sum(axis: -1, keepDims: true) + 1e-20) * scale
        let fused = try XCTUnwrap(DeepseekV4Math.fusedSqrtSoftplusSelect(
            logits: logits,
            bias: bias,
            k: topK,
            normalize: true,
            scalingFactor: scale))
        MLX.eval(genericIndices, normalizedGeneric, fused.indices, fused.weights)

        let expectedIndices = genericIndices.asArray(Int32.self)
        let expectedWeights = normalizedGeneric.asArray(Float.self)
        let actualIndices = fused.indices.asArray(Int32.self)
        let actualWeights = fused.weights.asArray(Float.self)
        XCTAssertEqual(Set(actualIndices), Set(expectedIndices))

        let expectedByExpert = Dictionary(uniqueKeysWithValues:
            zip(expectedIndices, expectedWeights))
        for (expert, weight) in zip(actualIndices, actualWeights) {
            let expected = try XCTUnwrap(expectedByExpert[expert])
            XCTAssertEqual(weight, expected, accuracy: 2e-5)
        }
    }

    func testFusedRouterFallsBackForUnsupportedExpertWidth() {
        let logits = MLXArray.zeros([1, 1, 257])
        let bias = MLXArray.zeros([257])
        let result = DeepseekV4Math.fusedSqrtSoftplusSelect(
            logits: logits,
            bias: bias,
            k: 6,
            normalize: true,
            scalingFactor: MLXArray([Float(1)]))
        XCTAssertNil(result)
    }

    func testDSparkHeadKernelMatchesFP32ProjectionForProposalBlock() throws {
        let rows = 5
        let inputDimensions = 128
        let outputDimensions = 4096
        let input = MLXArray((0..<(rows * inputDimensions)).map {
            Float(($0 % 37) - 18) * 0.013
        }).reshaped(1, rows, inputDimensions).asType(.bfloat16)
        let weight = MLXArray((0..<(outputDimensions * inputDimensions)).map {
            Float(($0 % 43) - 21) * 0.003
        }).reshaped(outputDimensions, inputDimensions).asType(.bfloat16)
        let linear = Linear(weight: weight)

        let actual = try XCTUnwrap(DeepseekV4Math.dsparkHeadFp32(input, linear: linear))
        let reference = input.asType(.float32)
            .matmul(weight.asType(.float32).transposed())
        MLX.eval(actual, reference)

        XCTAssertEqual(actual.shape, [1, rows, outputDimensions])
        XCTAssertEqual(actual.dtype, .float32)
        XCTAssertTrue(
            MLX.allClose(actual, reference, rtol: 2e-4, atol: 2e-4).item(Bool.self),
            "max DSpark head error: \(MLX.max(MLX.abs(actual - reference)).item(Float.self))")
    }

    func testDSparkHeadKernelRejectsUnsupportedShape() {
        let input = MLXArray.zeros([1, 5, 64], dtype: .bfloat16)
        let linear = Linear(weight: MLXArray.zeros([4096, 64], dtype: .bfloat16))
        XCTAssertNil(DeepseekV4Math.dsparkHeadFp32(input, linear: linear))
    }

    func testFusedHC4DecodeMatchesGenericReference() {
        assertFusedHC4MatchesReference(rows: 1, hiddenSize: 16)
    }

    func testFusedHC4MultiTokenMatchesGenericReference() {
        assertFusedHC4MatchesReference(rows: 3, hiddenSize: 16)
    }

    private func assertFusedHC4MatchesReference(rows: Int, hiddenSize: Int) {
        let mixes = MLXArray((0..<(rows * 24)).map {
            Float(($0 % 19) - 9) * 0.071
        }).reshaped(1, rows, 24)
        let scale = MLXArray([Float](arrayLiteral: 0.73, 1.19, 0.41))
        let base = MLXArray((0..<24).map { Float(($0 % 7) - 3) * 0.037 })
        let residual = MLXArray((0..<(rows * 4 * hiddenSize)).map {
            Float(($0 % 29) - 14) * 0.023
        }).reshaped(1, rows, 4, hiddenSize)
        let block = MLXArray((0..<(rows * hiddenSize)).map {
            Float(($0 % 17) - 8) * 0.031
        }).reshaped(1, rows, hiddenSize)

        let reference = DeepseekV4Math.hcSplitSinkhornOps(
            mixes: mixes, scale: scale, base: base, hcMult: 4)
        let collapsedReference = (
            residual.asType(.float32)
                * reference.pre[.ellipsis, .newAxis]
        ).sum(axis: -2)
        let expandedReference = DeepseekV4Math.hcExpandResidual(
            comb: reference.comb, residual: residual)
            + MLX.expandedDimensions(block, axis: -2)
                * MLX.expandedDimensions(reference.post, axis: -1)

        let fused = DeepseekV4Math.hcSplitSinkhornCollapse4(
            mixes: mixes,
            scale: scale,
            base: base,
            residual: residual,
            hiddenSize: hiddenSize)
        let normWeight = MLXArray((0..<hiddenSize).map {
            Float(($0 % 11) + 3) * 0.071
        })
        let normEps: Float = 1e-6
        let normalizedReference = MLXFast.rmsNorm(
            collapsedReference, weight: normWeight, eps: normEps)
        let fusedNorm = DeepseekV4Math.hcSplitSinkhornCollapseNorm4(
            mixes: mixes,
            scale: scale,
            base: base,
            residual: residual,
            normWeight: normWeight,
            normEps: normEps,
            hiddenSize: hiddenSize)
        let expanded = DeepseekV4Math.hcExpand4(
            blockOut: block,
            residual: residual,
            post: fused.post,
            comb: fused.comb,
            hiddenSize: hiddenSize)
        MLX.eval(
            reference.post, reference.comb, collapsedReference,
            fused.post, fused.comb, fused.collapsed,
            fusedNorm.post, fusedNorm.comb, fusedNorm.collapsed,
            normalizedReference, fusedNorm.normalized,
            expandedReference, expanded)

        let expansionMatches = MLX.allClose(
            expanded, expandedReference, rtol: 2e-5, atol: 2e-6).item(Bool.self)

        XCTAssertTrue(
            MLX.allClose(fused.post, reference.post, rtol: 2e-5, atol: 2e-6)
                .item(Bool.self))
        XCTAssertTrue(
            MLX.allClose(fused.comb, reference.comb, rtol: 2e-5, atol: 2e-6)
                .item(Bool.self))
        XCTAssertTrue(
            MLX.allClose(fused.collapsed, collapsedReference, rtol: 2e-5, atol: 2e-6)
                .item(Bool.self))
        XCTAssertTrue(
            MLX.allClose(fusedNorm.post, reference.post, rtol: 2e-5, atol: 2e-6)
                .item(Bool.self))
        XCTAssertTrue(
            MLX.allClose(fusedNorm.comb, reference.comb, rtol: 2e-5, atol: 2e-6)
                .item(Bool.self))
        XCTAssertTrue(
            MLX.allClose(fusedNorm.collapsed, collapsedReference, rtol: 2e-5, atol: 2e-6)
                .item(Bool.self))
        XCTAssertTrue(
            MLX.allClose(
                fusedNorm.normalized, normalizedReference,
                rtol: 2e-5, atol: 2e-6).item(Bool.self),
            "max fused norm error: \(MLX.max(MLX.abs(fusedNorm.normalized - normalizedReference)).item(Float.self))")
        XCTAssertTrue(
            expansionMatches,
            "max expansion error: \(MLX.max(MLX.abs(expanded - expandedReference)).item(Float.self))")
    }
}
