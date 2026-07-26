#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
@MainActor
final class FoundationTranscriptWindowPlannerTests: XCTestCase {
    func testTrimsOldestPromptTurnsAndPreservesInstructions() async throws {
        let instructions = Self.instructions("System")
        let firstPrompt = Self.prompt("p1", "First")
        let firstResponse = Self.response("r1", "One")
        let secondPrompt = Self.prompt("p2", "Second")
        let secondResponse = Self.response("r2", "Two")
        let latestPrompt = Self.prompt("p3", "Latest")

        let plan = try await AFMFoundationTranscriptWindowPlanner.trimmingOldestPromptTurns(
            [instructions, firstPrompt, firstResponse, secondPrompt, secondResponse, latestPrompt],
            maxTokenCount: 40
        ) { entries in
            entries.count * 10
        }

        XCTAssertEqual(plan.entries, [instructions, secondPrompt, secondResponse, latestPrompt])
        XCTAssertEqual(plan.removedPromptTurns, 1)
        XCTAssertEqual(plan.originalPromptCount, 3)
        XCTAssertEqual(plan.finalTokenCount, 40)
    }

    func testThrowsWhenCurrentTurnCannotFit() async {
        do {
            _ = try await AFMFoundationTranscriptWindowPlanner.trimmingOldestPromptTurns(
                [Self.instructions("System"), Self.prompt("p", "Only prompt")],
                maxTokenCount: 20
            ) { entries in
                entries.count * 30
            }
            XCTFail("Expected planner to throw when the remaining turn cannot fit.")
        } catch let error as AFMFoundationTranscriptWindowPlannerError {
            XCTAssertEqual(error, .currentTurnExceedsWindow(requiredTokens: 60, maxTokens: 20))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    private static func instructions(_ text: String) -> Transcript.Entry {
        .instructions(
            Transcript.Instructions(
                id: "instructions-\(text)",
                segments: [.text(Transcript.TextSegment(id: "instructions-text-\(text)", content: text))],
                toolDefinitions: []
            )
        )
    }

    private static func prompt(_ id: String, _ text: String) -> Transcript.Entry {
        .prompt(
            Transcript.Prompt(
                id: id,
                segments: [.text(Transcript.TextSegment(id: "\(id)-text", content: text))]
            )
        )
    }

    private static func response(_ id: String, _ text: String) -> Transcript.Entry {
        .response(
            Transcript.Response(
                id: id,
                metadata: [:],
                segments: [.text(Transcript.TextSegment(id: "\(id)-text", content: text))]
            )
        )
    }
}
#endif
