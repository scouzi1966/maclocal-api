import XCTest
@preconcurrency import MLXLMCommon
@testable import AFMKitMLX

final class AFMMLXChatHistoryPolicyTests: XCTestCase {
    func testResetCreatesSystemMessage() {
        let history = AFMMLXChatHistoryPolicy.reset(systemPrompt: "System A")

        XCTAssertEqual(history.count, 1)
        XCTAssertEqual(history[0].role, .system)
        XCTAssertEqual(history[0].content, "System A")
    }

    func testPreparedSnapshotInitializesEmptyHistory() {
        let prepared = AFMMLXChatHistoryPolicy.preparedSnapshot(
            history: [],
            userMessage: "Hello",
            images: [],
            systemPrompt: "System B"
        )

        XCTAssertEqual(prepared.history.count, 2)
        XCTAssertEqual(prepared.history[0].role, .system)
        XCTAssertEqual(prepared.history[0].content, "System B")
        XCTAssertEqual(prepared.history[1].role, .user)
        XCTAssertEqual(prepared.history[1].content, "Hello")
        XCTAssertEqual(prepared.snapshot.count, prepared.history.count)
    }

    func testPreparedSnapshotReplacesSingleMessageHistory() {
        let prepared = AFMMLXChatHistoryPolicy.preparedSnapshot(
            history: [.system("Old")],
            userMessage: "Next",
            images: [],
            systemPrompt: "New"
        )

        XCTAssertEqual(prepared.history.count, 2)
        XCTAssertEqual(prepared.history[0].content, "New")
        XCTAssertEqual(prepared.history[1].content, "Next")
    }

    func testPreparedSnapshotUpdatesSystemPromptAndPreservesConversation() {
        let prepared = AFMMLXChatHistoryPolicy.preparedSnapshot(
            history: [
                .system("Old"),
                .user("First"),
                .assistant("Second")
            ],
            userMessage: "Third",
            images: [],
            systemPrompt: "New"
        )

        XCTAssertEqual(prepared.history.map(\.role), [.system, .user, .assistant, .user])
        XCTAssertEqual(prepared.history.map(\.content), ["New", "First", "Second", "Third"])
    }

    func testAppendingAssistantHistoryTextSkipsEmptyText() {
        let history = [
            Chat.Message.system("System"),
            Chat.Message.user("Question")
        ]

        let updated = AFMMLXChatHistoryPolicy.appendingAssistantHistoryText("", to: history)

        XCTAssertEqual(updated.count, history.count)
        XCTAssertEqual(updated.map(\.content), history.map(\.content))
    }

    func testAppendingAssistantHistoryTextAddsAssistantMessage() {
        let history = [
            Chat.Message.system("System"),
            Chat.Message.user("Question")
        ]

        let updated = AFMMLXChatHistoryPolicy.appendingAssistantHistoryText("Answer", to: history)

        XCTAssertEqual(updated.count, 3)
        XCTAssertEqual(updated[2].role, .assistant)
        XCTAssertEqual(updated[2].content, "Answer")
    }
}
