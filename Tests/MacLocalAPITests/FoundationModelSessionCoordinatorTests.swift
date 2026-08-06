#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
@MainActor
final class FoundationModelSessionCoordinatorTests: XCTestCase {
    func testDynamicProfileSessionReusesExactProviderAndSignature() {
        let coordinator = AFMFoundationModelSessionCoordinator<String>()
        let first = coordinator.dynamicProfileSession(
            for: "apple",
            signature: "system-default",
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        let second = coordinator.dynamicProfileSession(
            for: "apple",
            signature: "system-default",
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        XCTAssertTrue(first === second)
    }

    func testDynamicProfileSessionRecreatesWhenProviderChanges() {
        let coordinator = AFMFoundationModelSessionCoordinator<String>()
        let first = coordinator.dynamicProfileSession(
            for: "apple",
            signature: "shared",
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        let second = coordinator.dynamicProfileSession(
            for: "pcc",
            signature: "shared",
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        XCTAssertFalse(first === second)
    }

    func testSimpleSessionReusesExactProviderAndSignature() {
        let coordinator = AFMFoundationModelSessionCoordinator<String>()
        let first = coordinator.simpleSession(
            for: "mlx",
            signature: "model-a",
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        let second = coordinator.simpleSession(
            for: "mlx",
            signature: "model-a",
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        XCTAssertTrue(first === second)
    }

    func testPrewarmRequiresMatchingExistingSession() {
        let coordinator = AFMFoundationModelSessionCoordinator<String>()

        XCTAssertFalse(coordinator.prewarm(promptPrefix: "Hello", for: "apple"))
    }

    func testDynamicProfileSnapshotIsSafeOffMainActor() async {
        let state = AFMFoundationDynamicProfileState(
            model: SystemLanguageModel.default,
            tools: [],
            instructions: "System"
        )

        let instructions = await Task.detached {
            state.snapshot().instructions
        }.value

        XCTAssertEqual(instructions, "System")
    }

    func testHistoryTransformDropsOrphanResponseBeforeFirstPrompt() {
        let instructions = Transcript.Entry.instructions(
            Transcript.Instructions(
                id: "instructions",
                segments: [.text(Transcript.TextSegment(id: "i", content: "System"))],
                toolDefinitions: []
            )
        )
        let orphan = Transcript.Entry.response(
            Transcript.Response(
                id: "orphan",
                metadata: [:],
                segments: [.text(Transcript.TextSegment(id: "o", content: "Orphan"))]
            )
        )
        let prompt = Transcript.Entry.prompt(
            Transcript.Prompt(
                id: "prompt",
                segments: [.text(Transcript.TextSegment(id: "p", content: "Hello"))]
            )
        )

        let transformed = AFMFoundationHistoryTransform.normalized([instructions, orphan, prompt])

        XCTAssertEqual(transformed, [instructions, prompt])
    }

    func testHistoryTransformKeepsConversationStartingAtFirstPrompt() {
        let prompt = Transcript.Entry.prompt(
            Transcript.Prompt(
                id: "prompt",
                segments: [.text(Transcript.TextSegment(id: "p", content: "Hello"))]
            )
        )
        let response = Transcript.Entry.response(
            Transcript.Response(
                id: "response",
                metadata: [:],
                segments: [.text(Transcript.TextSegment(id: "r", content: "Hi"))]
            )
        )

        XCTAssertEqual(
            AFMFoundationHistoryTransform.normalized([prompt, response]),
            [prompt, response]
        )
    }
}
#endif
