#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationStructuredResponseCompleterTests: XCTestCase {
    func testCompletesRenderedContentToolSnapshotsAndTelemetry() throws {
        let clock = ContinuousClock()
        let startedAt = clock.now
        let completedAt = startedAt.advanced(by: .milliseconds(250))
        let rawContent = GeneratedContent(properties: [
            "label": "science",
            "confidence": 91
        ])
        let call = Transcript.ToolCall(
            id: "call-1",
            toolName: "lookup",
            arguments: try GeneratedContent(json: #"{"query":"sky"}"#)
        )
        let output = Transcript.ToolOutput(
            id: "call-1",
            toolName: "lookup",
            segments: [.text(Transcript.TextSegment(content: "Rayleigh scattering"))]
        )

        let result = try AFMFoundationStructuredResponseCompleter.complete(
            rawContent: rawContent,
            label: "Classification",
            usage: AFMFoundationGenerationUsage(
                inputTokens: 12,
                cachedInputTokens: 3,
                outputTokens: 7,
                reasoningTokens: 2
            ),
            transcriptEntries: [
                Transcript.Entry.toolCalls(Transcript.ToolCalls([call])),
                Transcript.Entry.toolOutput(output)
            ],
            contextAction: "trimmed",
            startedAt: startedAt,
            completedAt: completedAt
        ) { content in
            [
                AFMFoundationGeneratedContentReader.string("label", in: content),
                AFMFoundationGeneratedContentReader.number("confidence", in: content).map {
                    "\(Int($0))%"
                }
            ]
            .compactMap { $0 }
            .joined(separator: " ")
        }

        XCTAssertEqual(result.renderedContent, "science 91%")
        XCTAssertEqual(result.toolInvocations.count, 1)
        XCTAssertEqual(result.toolInvocations[0].name, "lookup")
        XCTAssertEqual(result.toolInvocations[0].status, .completed)
        XCTAssertEqual(result.toolInvocations[0].outputPreview, "Rayleigh scattering")
        XCTAssertEqual(result.telemetry.inputTokens, 12)
        XCTAssertEqual(result.telemetry.cachedInputTokens, 3)
        XCTAssertEqual(result.telemetry.outputTokens, 7)
        XCTAssertEqual(result.telemetry.reasoningTokens, 2)
        XCTAssertEqual(result.telemetry.toolNames, ["lookup"])
        XCTAssertEqual(result.telemetry.contextAction, "trimmed")
        XCTAssertEqual(result.telemetry.streamChunkCount, 1)
        XCTAssertEqual(try XCTUnwrap(result.telemetry.timeToFirstTokenMilliseconds), 250, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(result.telemetry.streamingMilliseconds), 0, accuracy: 0.001)
        XCTAssertNil(result.telemetry.tokensPerSecond)
    }

    func testRejectsEmptyRenderedContent() throws {
        XCTAssertThrowsError(
            try AFMFoundationStructuredResponseCompleter.complete(
                rawContent: GeneratedContent(properties: ["label": ""]),
                label: "Empty",
                usage: AFMFoundationGenerationUsage(
                    inputTokens: 1,
                    cachedInputTokens: 0,
                    outputTokens: 0,
                    reasoningTokens: 0
                ),
                transcriptEntries: [] as [Transcript.Entry],
                contextAction: nil,
                startedAt: ContinuousClock.now,
                completedAt: ContinuousClock.now,
                render: { _ in "  " }
            )
        ) { error in
            XCTAssertEqual(error as? AFMFoundationStructuredResponseError, .emptyRenderedContent(label: "Empty"))
        }
    }
}
#endif
