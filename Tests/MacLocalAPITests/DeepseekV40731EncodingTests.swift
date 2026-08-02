import MLX
import MLXLLM
import MLXLMCommon
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
}
