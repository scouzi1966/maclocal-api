import XCTest
@testable import MacLocalAPI

final class TelegramBridgeTests: XCTestCase {
// dimensions: entry_mode=server
    func testParseAllowedUserIDsDeduplicates() throws {
        let ids = try TelegramConfiguration.parseAllowedUserIDs("12345, 67890,12345")
        XCTAssertEqual(ids, [12345, 67890])
    }

    func testParseAllowedUserIDsRejectsInvalidValue() {
        XCTAssertThrowsError(try TelegramConfiguration.parseAllowedUserIDs("abc,123"))
    }

    func testParseCommandIgnoresNonPrefixedText() {
        XCTAssertEqual(TelegramPolicy.parseCommand("hello", requiredPrefix: "/afm"), .none)
    }

    func testParseCommandExtractsPrompt() {
        XCTAssertEqual(TelegramPolicy.parseCommand("/afm summarize this", requiredPrefix: "/afm"), .prompt("summarize this"))
    }

    func testParseCommandRecognizesNewConversationCommand() {
        XCTAssertEqual(TelegramPolicy.parseCommand("/afm new", requiredPrefix: "/afm"), .newConversation)
        XCTAssertEqual(TelegramPolicy.parseCommand("/afm new", requiredPrefix: nil), .newConversation)
    }

    func testParseCommandAcceptsPlainTextWhenPrefixNotRequired() {
        XCTAssertEqual(TelegramPolicy.parseCommand("hello", requiredPrefix: nil), .prompt("hello"))
    }

    func testPrivateChatPolicy() {
        XCTAssertTrue(TelegramPolicy.isPrivateChat("private"))
        XCTAssertFalse(TelegramPolicy.isPrivateChat("group"))
    }

    func testAllowedUserPolicy() {
        XCTAssertTrue(TelegramPolicy.isAllowedUser(123, allowedUserIDs: [123]))
        XCTAssertFalse(TelegramPolicy.isAllowedUser(999, allowedUserIDs: [123]))
    }

    func testDefaultPromptReflectsAttachmentCount() {
        XCTAssertEqual(TelegramPolicy.defaultPrompt(for: 1), "Please analyze the attached image.")
        XCTAssertEqual(TelegramPolicy.defaultPrompt(for: 2), "Please analyze the attached images.")
    }

    func testMarkdownFormatUsesHTMLParseMode() {
        XCTAssertEqual(TelegramReplyFormat.markdown.apiParseMode, "HTML")
    }

    func testMarkdownRendererPreservesBasicFormatting() {
        let rendered = TelegramReplyFormat.markdown.render("**bold** and `code` and [link](https://example.com)")
        XCTAssertEqual(rendered, "<b>bold</b> and <code>code</code> and <a href=\"https://example.com\">link</a>")
    }

    func testChunkResponsePrefersParagraphAndWordBoundaries() {
        let text = "First paragraph line one.\n\nSecond paragraph line two with more words."
        let chunks = TelegramPolicy.chunkResponse(text, limit: 28)

        XCTAssertEqual(chunks, [
            "First paragraph line one.",
            "Second paragraph line two",
            "with more words."
        ])
    }

    func testConversationStoreReturnsAlternatingHistoryMessages() async {
        let store = TelegramConversationStore()
        await store.append(chatID: 42, userPrompt: "hello", assistantReply: "hi")
        await store.append(chatID: 42, userPrompt: "how are you?", assistantReply: "good")

        let messages = await store.historyMessages(chatID: 42)
        XCTAssertEqual(messages.map(\.role), ["user", "assistant", "user", "assistant"])
        XCTAssertEqual(messages.map(\.textContent), ["hello", "hi", "how are you?", "good"])
    }

    func testConversationStoreRespectsTurnLimit() async {
        let store = TelegramConversationStore()
        for index in 0..<(TelegramPolicy.maxConversationTurns + 3) {
            await store.append(chatID: 7, userPrompt: "u\(index)", assistantReply: "a\(index)")
        }

        let messages = await store.historyMessages(chatID: 7)
        XCTAssertEqual(messages.count, TelegramPolicy.maxConversationTurns * 2)
        XCTAssertEqual(messages.first?.textContent, "u3")
        XCTAssertEqual(messages.last?.textContent, "a10")
    }

    func testConversationStoreResetClearsHistory() async {
        let store = TelegramConversationStore()
        await store.append(chatID: 99, userPrompt: "hello", assistantReply: "hi")
        await store.reset(chatID: 99)
        let messages = await store.historyMessages(chatID: 99)
        XCTAssertTrue(messages.isEmpty)
    }

    func testStateFileNameDoesNotLeakTelegramToken() {
        let token = "123456789:AAExampleTokenValueExample123456789"
        let fileName = TelegramStateStore.stateFileName(for: token)

        XCTAssertTrue(fileName.hasPrefix("state-"))
        XCTAssertTrue(fileName.hasSuffix(".json"))
        XCTAssertFalse(fileName.contains(token))
        XCTAssertFalse(fileName.contains("123456789"))
        XCTAssertFalse(fileName.contains("AAExampleTokenValueExample123456789"))
        XCTAssertEqual(fileName, TelegramStateStore.stateFileName(for: token))
    }

    func testActiveRequestStoreReplaceCancelsPreviousTask() async {
        let store = TelegramActiveRequestStore()

        let firstCanceled = expectation(description: "first task canceled")
        let firstTask = Task {
            do {
                try await Task.sleep(nanoseconds: 30_000_000_000)
            } catch is CancellationError {
                firstCanceled.fulfill()
            } catch {
                XCTFail("Unexpected error: \(error)")
            }
        }

        _ = await store.replace(chatID: 1, task: firstTask)

        let secondTask = Task {}
        _ = await store.replace(chatID: 1, task: secondTask)

        await fulfillment(of: [firstCanceled], timeout: 1.0)
        secondTask.cancel()
        _ = await secondTask.result
    }

    func testActiveRequestStoreCancelStopsTrackedTask() async {
        let store = TelegramActiveRequestStore()

        let canceled = expectation(description: "tracked task canceled")
        let task = Task {
            do {
                try await Task.sleep(nanoseconds: 30_000_000_000)
            } catch is CancellationError {
                canceled.fulfill()
            } catch {
                XCTFail("Unexpected error: \(error)")
            }
        }

        _ = await store.replace(chatID: 42, task: task)
        let didCancel = await store.cancel(chatID: 42)

        XCTAssertTrue(didCancel)
        await fulfillment(of: [canceled], timeout: 1.0)
        _ = await task.result
        let canceledAgain = await store.cancel(chatID: 42)
        XCTAssertFalse(canceledAgain)
    }

    func testActiveRequestStoreFinishDoesNotClearReplacedRequest() async {
        let store = TelegramActiveRequestStore()

        let firstTask = Task {}
        let firstID = await store.replace(chatID: 9, task: firstTask)

        let secondTask = Task {}
        _ = await store.replace(chatID: 9, task: secondTask)

        await store.finish(chatID: 9, id: firstID)

        let didCancel = await store.cancel(chatID: 9)
        XCTAssertTrue(didCancel)

        firstTask.cancel()
        secondTask.cancel()
        _ = await firstTask.result
        _ = await secondTask.result
    }
}
