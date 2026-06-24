import Foundation
import Testing

@testable import AFMKit
@testable import AFMServer

/// Unit tests for the Tier-1 agent-friendly changes (PR-1).
struct AgentFriendlyTier1Tests {

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.1 — Request ID middleware contract
    // ═══════════════════════════════════════════════════════════════════

    @Test("T1.1 minted request IDs use the documented req_<uuid12> shape")
    func requestIDMintFormat() {
        let id = RequestIDMiddleware.mint()
        #expect(id.hasPrefix("req_"))
        #expect(id.count == 16) // "req_" + 12 hex chars
        let suffix = String(id.dropFirst(4))
        #expect(suffix.count == 12)
        // 12 lowercase hex chars (uuid stripped of '-')
        #expect(suffix.allSatisfy { $0.isHexDigit && (!$0.isLetter || $0.isLowercase) })
    }

    @Test("T1.1 OpenAIError exposes request_id field for correlation")
    func openAIErrorCarriesRequestID() throws {
        let err = OpenAIError(
            message: "boom",
            type: "internal_error",
            code: "boom_code",
            requestId: "req_abc123def456"
        )
        let data = try JSONEncoder().encode(err)
        let json = String(data: data, encoding: .utf8) ?? ""
        #expect(json.contains("\"request_id\":\"req_abc123def456\""))
        #expect(json.contains("\"message\":\"boom\""))
    }

    @Test("T1.1 OpenAIError request_id is omitted when not provided")
    func openAIErrorRequestIDOptional() throws {
        let err = OpenAIError(message: "boom")
        let data = try JSONEncoder().encode(err)
        let json = String(data: data, encoding: .utf8) ?? ""
        // request_id absent when nil
        #expect(!json.contains("\"request_id\""))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.2 — stream_options.include_usage
    // ═══════════════════════════════════════════════════════════════════

    @Test("T1.2 default is to include usage when stream_options absent")
    func includeUsageDefaultsTrue() throws {
        let json = """
        {"messages":[{"role":"user","content":"hi"}]}
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(req.includeStreamingUsage == true)
        #expect(req.streamOptions == nil)
    }

    @Test("T1.2 include_usage=false suppresses the usage chunk")
    func includeUsageFalse() throws {
        let json = """
        {"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":false}}
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(req.includeStreamingUsage == false)
    }

    @Test("T1.2 include_usage=true is honored explicitly")
    func includeUsageTrue() throws {
        let json = """
        {"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(req.includeStreamingUsage == true)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.3 — parallel_tool_calls: false truncation
    // ═══════════════════════════════════════════════════════════════════

    private func mkToolCall(_ name: String, index: Int) -> ResponseToolCall {
        ResponseToolCall(
            index: index,
            id: "call_\(name)_\(index)",
            type: "function",
            function: ResponseToolCallFunction(name: name, arguments: "{}")
        )
    }

    @Test("T1.3 parallel_tool_calls=false truncates to first tool call")
    func parallelToolCallsFalseTruncates() {
        let multi = [mkToolCall("a", index: 0), mkToolCall("b", index: 1), mkToolCall("c", index: 2)]
        let turn = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "",
            toolCalls: multi,
            toolChoice: nil,
            parallelToolCalls: false,
            extractThinking: false,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 1,
            maxTokens: 100,
            sanitizeContent: { $0 }
        )
        #expect(turn.finishReason == "tool_calls")
        #expect(turn.toolCalls?.count == 1)
        #expect(turn.toolCalls?.first?.function.name == "a")
    }

    @Test("T1.3 parallel_tool_calls=true (default) preserves all calls")
    func parallelToolCallsTrueKeepsAll() {
        let multi = [mkToolCall("a", index: 0), mkToolCall("b", index: 1)]
        let turn = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "",
            toolCalls: multi,
            toolChoice: nil,
            parallelToolCalls: true,
            extractThinking: false,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 1,
            maxTokens: 100,
            sanitizeContent: { $0 }
        )
        #expect(turn.toolCalls?.count == 2)
    }

    @Test("T1.3 parallel_tool_calls=nil leaves prior behavior untouched")
    func parallelToolCallsNilKeepsAll() {
        let multi = [mkToolCall("a", index: 0), mkToolCall("b", index: 1)]
        let turn = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "",
            toolCalls: multi,
            toolChoice: nil,
            parallelToolCalls: nil,
            extractThinking: false,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 1,
            maxTokens: 100,
            sanitizeContent: { $0 }
        )
        #expect(turn.toolCalls?.count == 2)
    }

    @Test("T1.3 parallel_tool_calls=false with single call is a no-op")
    func parallelToolCallsFalseSingleCall() {
        let single = [mkToolCall("only", index: 0)]
        let turn = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "",
            toolCalls: single,
            toolChoice: nil,
            parallelToolCalls: false,
            extractThinking: false,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 1,
            maxTokens: 100,
            sanitizeContent: { $0 }
        )
        #expect(turn.toolCalls?.count == 1)
        #expect(turn.toolCalls?.first?.function.name == "only")
    }

    @Test("T1.3 parallel_tool_calls field round-trips through JSON")
    func parallelToolCallsJSONRoundTrip() throws {
        let json = """
        {"messages":[{"role":"user","content":"hi"}],"parallel_tool_calls":false}
        """
        let req = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        #expect(req.parallelToolCalls == false)
    }
}
