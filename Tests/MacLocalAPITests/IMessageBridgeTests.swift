import XCTest
@testable import MacLocalAPI

final class IMessageBridgeTests: XCTestCase {
    func testParseAllowedSendersNormalizesAndDeduplicates() throws {
        let senders = try IMessageConfiguration.parseAllowedSenders(" Alice@example.com ,mailto:alice@example.com,tel:+15551234567 ")
        XCTAssertEqual(senders, ["alice@example.com", "+15551234567"])
    }

    func testParseCommandIgnoresNonPrefixedText() {
        XCTAssertEqual(IMessagePolicy.parseCommand("hello"), .none)
    }

    func testParseCommandExtractsPrompt() {
        XCTAssertEqual(IMessagePolicy.parseCommand("/afm summarize this"), .prompt("summarize this"))
    }

    func testParseCommandExtractsPairCode() {
        XCTAssertEqual(IMessagePolicy.parseCommand("/afm pair 123456"), .pair("123456"))
    }

    func testDirectChatPolicyRequiresSingleParticipant() {
        XCTAssertTrue(IMessagePolicy.isDirectChat(participantCount: 1))
        XCTAssertFalse(IMessagePolicy.isDirectChat(participantCount: 2))
    }

    func testAllowedSenderPolicyNormalizesInput() {
        XCTAssertTrue(IMessagePolicy.isAllowedSender("MAILTO:Alice@Example.com", allowedSenders: ["alice@example.com"]))
        XCTAssertFalse(IMessagePolicy.isAllowedSender("bob@example.com", allowedSenders: ["alice@example.com"]))
    }

    func testDefaultPromptReflectsAttachmentCount() {
        let single = IMessagePolicy.defaultPrompt(for: [URL(fileURLWithPath: "/tmp/one.png")])
        let multiple = IMessagePolicy.defaultPrompt(for: [URL(fileURLWithPath: "/tmp/one.png"), URL(fileURLWithPath: "/tmp/two.png")])
        XCTAssertEqual(single, "Please analyze the attached image.")
        XCTAssertEqual(multiple, "Please analyze the attached images.")
    }

    func testChunkResponseSplitsLongMessages() {
        let chunks = IMessagePolicy.chunkResponse(String(repeating: "a", count: 7000), limit: 3200)
        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks.map(\.count), [3200, 3200, 600])
    }
}
