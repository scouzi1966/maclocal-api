import XCTest
@testable import AFMServer

final class MessagesControllerThinkingTests: XCTestCase {
    func testDisabledThinkingReachesChatTemplateWithoutReasoningEffort() throws {
        let request = try translatedRequest(thinking: "disabled")

        XCTAssertEqual(request["chat_template_kwargs"]?["enable_thinking"]?.boolValue, false)
        XCTAssertNil(request["reasoning_effort"])
    }

    func testEnabledThinkingKeepsExistingReasoningEffort() throws {
        let request = try translatedRequest(thinking: "enabled")

        XCTAssertEqual(request["reasoning_effort"]?.stringValue, "low")
        XCTAssertNil(request["chat_template_kwargs"])
    }

    func testOmittedThinkingPreservesProviderDefault() throws {
        let request = try translatedRequest(thinking: nil)

        XCTAssertNil(request["reasoning_effort"])
        XCTAssertNil(request["chat_template_kwargs"])
    }

    private func translatedRequest(thinking: String?) throws -> ResponsesJSON {
        var object: [String: ResponsesJSON] = [:]
        if let thinking {
            object["thinking"] = .object(["type": .string(thinking)])
        }
        return try MessagesController.makeChatRequest(
            object: object,
            sourceMessages: [.object([
                "role": .string("user"), "content": .string("Say hello")
            ])],
            maxTokens: 12,
            defaultModel: "test-model"
        )
    }
}
