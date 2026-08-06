#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationTranscriptSnapshotParserTests: XCTestCase {
    func testRecordsToolArgumentsAndCompletedOutput() throws {
        let call = Transcript.ToolCall(
            id: "call-1",
            toolName: "lookup",
            arguments: try GeneratedContent(json: #"{"query":"selected"}"#)
        )
        let output = Transcript.ToolOutput(
            id: "call-1",
            toolName: "lookup",
            segments: [.text(Transcript.TextSegment(content: "Provider: Apple On-Device"))]
        )

        let invocations = AFMFoundationTranscriptSnapshotParser.toolInvocations(
            from: [
                Transcript.Entry.toolCalls(Transcript.ToolCalls([call])),
                Transcript.Entry.toolOutput(output)
            ]
        )

        XCTAssertEqual(invocations.count, 1)
        XCTAssertEqual(invocations[0].id, "call-1")
        XCTAssertEqual(invocations[0].name, "lookup")
        XCTAssertTrue(invocations[0].argumentsJSON?.contains("selected") == true)
        XCTAssertEqual(invocations[0].outputPreview, "Provider: Apple On-Device")
        XCTAssertEqual(invocations[0].status, .completed)
    }

    func testOutputWithoutMatchingIDUsesLatestRequestedToolWithSameName() throws {
        let first = Transcript.ToolCall(
            id: "call-1",
            toolName: "lookup",
            arguments: try GeneratedContent(json: #"{"query":"first"}"#)
        )
        let second = Transcript.ToolCall(
            id: "call-2",
            toolName: "lookup",
            arguments: try GeneratedContent(json: #"{"query":"second"}"#)
        )
        let output = Transcript.ToolOutput(
            id: "provider-generated-output-id",
            toolName: "lookup",
            segments: [.text(Transcript.TextSegment(content: "Second result"))]
        )

        let invocations = AFMFoundationTranscriptSnapshotParser.toolInvocations(
            from: [
                Transcript.Entry.toolCalls(Transcript.ToolCalls([first, second])),
                Transcript.Entry.toolOutput(output)
            ]
        )

        XCTAssertEqual(invocations.map(\.id), ["call-1", "call-2"])
        XCTAssertNil(invocations[0].outputPreview)
        XCTAssertEqual(invocations[1].outputPreview, "Second result")
        XCTAssertEqual(invocations[1].status, .completed)
    }

    func testReasoningContentReturnsReasoningAfterLatestPrompt() {
        let oldPrompt = Transcript.Entry.prompt(
            Transcript.Prompt(
                id: "prompt-old",
                segments: [.text(Transcript.TextSegment(id: "p-old", content: "Old prompt"))]
            )
        )
        let oldReasoning = Transcript.Entry.reasoning(
            Transcript.Reasoning(
                id: "reasoning-old",
                segments: [.text(Transcript.TextSegment(id: "r-old", content: "Old reasoning"))]
            )
        )
        let currentPrompt = Transcript.Entry.prompt(
            Transcript.Prompt(
                id: "prompt-current",
                segments: [.text(Transcript.TextSegment(id: "p-current", content: "Current prompt"))]
            )
        )
        let currentReasoning = Transcript.Entry.reasoning(
            Transcript.Reasoning(
                id: "reasoning-current",
                segments: [.text(Transcript.TextSegment(id: "r-current", content: "Current reasoning"))]
            )
        )

        let reasoning = AFMFoundationTranscriptSnapshotParser.reasoningContent(
            from: [oldPrompt, oldReasoning, currentPrompt, currentReasoning]
        )

        XCTAssertEqual(reasoning, "Current reasoning")
    }

    func testMarkingPendingUpdatesOnlyRequestedInvocations() throws {
        let requested = AFMFoundationToolInvocationSnapshot(
            id: "call-1",
            name: "lookup",
            argumentsJSON: #"{"query":"CoreAI"}"#,
            outputPreview: nil,
            status: .requested
        )
        let completed = AFMFoundationToolInvocationSnapshot(
            id: "call-2",
            name: "lookup",
            argumentsJSON: #"{"query":"done"}"#,
            outputPreview: "Done",
            status: .completed
        )

        let updated = AFMFoundationTranscriptSnapshotParser.markingPending(
            [requested, completed],
            as: .failed,
            failurePreview: "Search unavailable",
            previewLimit: 8
        )

        XCTAssertEqual(updated[0].status, .failed)
        XCTAssertEqual(updated[0].failurePreview, "Search u…")
        XCTAssertEqual(updated[1], completed)
    }
}
#endif
