import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class DeepseekV40731EncodingTests: XCTestCase {
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
}
