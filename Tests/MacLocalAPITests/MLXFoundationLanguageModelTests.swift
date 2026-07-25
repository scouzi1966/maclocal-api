#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKit
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class MLXFoundationLanguageModelTests: XCTestCase {
    func testExecutorConfigurationIncludesModelAndRuntimeIdentity() {
        let first = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )
        let same = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )
        let otherModel = MLXLanguageModel(
            modelID: "mlx-community/model-b",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )

        XCTAssertEqual(first.executorConfiguration, same.executorConfiguration)
        XCTAssertNotEqual(first.executorConfiguration, otherModel.executorConfiguration)
        XCTAssertTrue(first.capabilities.contains(.reasoning))
        XCTAssertTrue(first.capabilities.contains(.toolCalling))
        XCTAssertTrue(first.capabilities.contains(.guidedGeneration))
    }

    func testTranscriptTranslationPreservesMultiTurnRoles() throws {
        let transcript = Transcript(entries: [
            .instructions(
                .init(
                    segments: [.text(.init(content: "Be concise."))],
                    toolDefinitions: []
                )
            ),
            .prompt(
                .init(segments: [.text(.init(content: "First question"))])
            ),
            .response(
                .init(
                    metadata: [:],
                    segments: [.text(.init(content: "First answer"))]
                )
            ),
            .prompt(
                .init(segments: [.text(.init(content: "Follow-up question"))])
            )
        ])

        let messages = try MLXLanguageModelExecutor.messages(from: transcript)

        XCTAssertEqual(messages.map(\.role), ["system", "user", "assistant", "user"])
        XCTAssertEqual(
            messages.map(\.textContent),
            ["Be concise.", "First question", "First answer", "Follow-up question"]
        )
    }

    func testTranscriptTranslationPreservesToolCallsAndOutputs() throws {
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "weather",
            arguments: try GeneratedContent(json: #"{"city":"Toronto"}"#)
        )
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Weather?"))])),
            .toolCalls(.init([call])),
            .toolOutput(
                .init(
                    id: "call_1",
                    toolName: "weather",
                    segments: [.text(.init(content: "Sunny"))]
                )
            )
        ])

        let messages = try MLXLanguageModelExecutor.messages(from: transcript)

        XCTAssertEqual(messages.map(\.role), ["user", "assistant", "tool"])
        XCTAssertEqual(messages[1].toolCalls?.first?.id, "call_1")
        XCTAssertEqual(messages[1].toolCalls?.first?.function.name, "weather")
        let arguments = try XCTUnwrap(
            messages[1].toolCalls?.first?.function.arguments.data(using: .utf8)
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: arguments) as? [String: String]
        )
        XCTAssertEqual(object, ["city": "Toronto"])
        XCTAssertEqual(messages[2].toolCallId, "call_1")
        XCTAssertEqual(messages[2].name, "weather")
        XCTAssertEqual(messages[2].textContent, "Sunny")
    }
}
#endif
