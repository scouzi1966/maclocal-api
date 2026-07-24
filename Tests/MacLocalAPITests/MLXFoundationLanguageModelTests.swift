#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKit
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
            defaultMaximumResponseTokens: 4_096
        )
        let same = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096
        )
        let otherModel = MLXLanguageModel(
            modelID: "mlx-community/model-b",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096
        )

        XCTAssertEqual(first.executorConfiguration, same.executorConfiguration)
        XCTAssertNotEqual(first.executorConfiguration, otherModel.executorConfiguration)
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
}
#endif
