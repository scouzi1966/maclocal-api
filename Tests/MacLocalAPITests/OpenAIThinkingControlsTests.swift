import AFMOpenAICompat
import XCTest

final class OpenAIThinkingControlsTests: XCTestCase {
    func testTopLevelReasoningEffortIsNormalizedIntoChatTemplateKwargs() throws {
        let request = try decode("""
        {
          "model": "deepseek",
          "messages": [{"role": "user", "content": "hi"}],
          "reasoning_effort": "high"
        }
        """)

        XCTAssertEqual(stringValue(request.effectiveChatTemplateKwargs?["reasoning_effort"]), "high")
    }

    func testThinkingBudgetAliasIsNormalizedIntoChatTemplateKwargs() throws {
        let request = try decode("""
        {
          "model": "deepseek",
          "messages": [{"role": "user", "content": "hi"}],
          "thinking_budget": "max"
        }
        """)

        XCTAssertEqual(stringValue(request.effectiveChatTemplateKwargs?["reasoning_effort"]), "max")
    }

    func testTopLevelReasoningEffortOverridesNestedKwarg() throws {
        let request = try decode("""
        {
          "model": "deepseek",
          "messages": [{"role": "user", "content": "hi"}],
          "chat_template_kwargs": {"reasoning_effort": "low"},
          "reasoning_effort": "high"
        }
        """)

        XCTAssertEqual(stringValue(request.effectiveChatTemplateKwargs?["reasoning_effort"]), "high")
    }

    private func decode(_ json: String) throws -> ChatCompletionRequest {
        let data = try XCTUnwrap(json.data(using: .utf8))
        return try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
    }

    private func stringValue(_ value: AnyCodable?) -> String? {
        guard case .string(let string)? = value?.value else { return nil }
        return string
    }
}
