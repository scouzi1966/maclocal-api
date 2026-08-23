import Foundation
import Testing

@testable import AFMKit
@testable import AFMServer

/// Regression tests for fixed issues. Each case names the issue it covers.
struct IssueRegressionTests {

    @Test("MLX max_tokens fallback applies when request and CLI omit it")
    func mlxDefaultMaxTokensFallback() {
        #expect(MLXChatCompletionsController.resolveEffectiveMaxTokens(requested: nil, serverDefault: nil) == MLXChatCompletionsController.defaultMaxCompletionTokens)
        #expect(MLXChatCompletionsController.resolveEffectiveMaxTokens(requested: 0, serverDefault: nil) == MLXChatCompletionsController.defaultMaxCompletionTokens)
        #expect(MLXChatCompletionsController.defaultMaxCompletionTokens == 8192)
        #expect(MLXModelService.defaultMaximumResponseTokens == 8192)
        #expect(MLXChatCompletionsController.resolveEffectiveMaxTokens(requested: nil, serverDefault: 1024) == 1024)
        #expect(MLXChatCompletionsController.resolveEffectiveMaxTokens(requested: 2048, serverDefault: 1024) == 2048)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - #103 — Foundation server `--guided-json` fallback
    // ═══════════════════════════════════════════════════════════════════

    @Test("#103 server-level guided-json applies when request omits response_format")
    func issue103_serverDefaultUsedWhenRequestOmitsFormat() {
        let serverDefault = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "guided",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )
        let resolved = ChatCompletionsController.resolveStrictJsonSchema(
            requestFormat: nil,
            serverDefault: serverDefault
        )
        #expect(resolved != nil)
        #expect(resolved?.name == "guided")
        #expect(resolved?.strict == true)
    }

    @Test("#103 per-request response_format wins over server default")
    func issue103_requestFormatWinsOverServerDefault() {
        let serverDefault = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "guided",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )
        let perRequest = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "request_schema",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )
        let resolved = ChatCompletionsController.resolveStrictJsonSchema(
            requestFormat: perRequest,
            serverDefault: serverDefault
        )
        #expect(resolved?.name == "request_schema")
    }

    @Test("#103 returns nil when neither side supplies a schema")
    func issue103_nilWhenNoSchema() {
        #expect(ChatCompletionsController.resolveStrictJsonSchema(
            requestFormat: nil,
            serverDefault: nil
        ) == nil)
    }

    @Test("#103 non-strict schemas are not surfaced (matches existing semantics)")
    func issue103_strictFalseFiltered() {
        let nonStrict = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "guided",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: false
            )
        )
        #expect(ChatCompletionsController.resolveStrictJsonSchema(
            requestFormat: nil,
            serverDefault: nonStrict
        ) == nil)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - #99 — pre-opened <think> handled by streaming extractor
    // ═══════════════════════════════════════════════════════════════════
    //
    // Mechanism: when a chat template appends `<think>\n` to the prompt, the
    // service prepends a synthetic `<think>` chunk to the stream so the
    // controller's existing extractor latches `insideThinkBlock = true` on
    // first token. These tests exercise the extractor in that exact mode.

    @Test("#99 pre-opened think: synthetic <think> prefix routes body to reasoning")
    func issue99_preopenedThinkRoutesReasoning() {
        var buffer = "<think>"
        var inside = false
        var totalReasoning = ""
        var totalContent = ""

        // The synthetic chunk arrives first and flips state.
        let r0 = MLXChatCompletionsController.extractThinkTags(
            buffer: &buffer,
            insideThinkBlock: &inside
        )
        if let r = r0.reasoning { totalReasoning += r }
        if let c = r0.content { totalContent += c }
        #expect(inside == true)

        // Body arrives — most should flush as reasoning; a small tail is
        // retained for boundary detection (matching the controller's design).
        buffer += "Thinking through Rayleigh scattering."
        let r1 = MLXChatCompletionsController.extractThinkTags(
            buffer: &buffer,
            insideThinkBlock: &inside
        )
        if let r = r1.reasoning { totalReasoning += r }
        if let c = r1.content { totalContent += c }
        #expect(r1.reasoning != nil)

        // Closing tag plus visible content.
        buffer += "</think>The sky is blue."
        let r2 = MLXChatCompletionsController.extractThinkTags(
            buffer: &buffer,
            insideThinkBlock: &inside
        )
        if let r = r2.reasoning { totalReasoning += r }
        if let c = r2.content { totalContent += c }
        #expect(inside == false)

        // Final flush mirrors what the controller does after the stream loop:
        // any text still in the buffer is emitted based on the closing state.
        if !buffer.isEmpty {
            if inside {
                totalReasoning += buffer
            } else {
                totalContent += buffer
            }
        }

        #expect(totalReasoning == "Thinking through Rayleigh scattering.")
        #expect(totalContent == "The sky is blue.")
    }

    @Test("#99 whole-text extractThinkContent on already-open synthetic prefix")
    func issue99_extractThinkContentWithSyntheticPrefix() {
        let raw = "<think>reasoning trace</think>final answer"
        let (content, reasoning) = MLXChatCompletionsController.extractThinkContent(from: raw)
        #expect(reasoning == "reasoning trace")
        #expect(content == "final answer")
    }

    @Test("proxy non-streaming response renames reasoning field")
    func proxyNormalizeRenamesReasoningField() throws {
        let normalized = try normalizeProxyResponse([
            "choices": [[
                "message": [
                    "role": "assistant",
                    "content": "Visible answer",
                    "reasoning": "Hidden reasoning",
                ],
            ]],
        ])

        let message = try #require(firstMessage(from: normalized))
        #expect(message["reasoning"] == nil)
        #expect(message["reasoning_content"] as? String == "Hidden reasoning")
        #expect(message["content"] as? String == "Visible answer")
    }

    @Test("proxy non-streaming response extracts think tags")
    func proxyNormalizeExtractsThinkTags() throws {
        let normalized = try normalizeProxyResponse([
            "choices": [[
                "message": [
                    "role": "assistant",
                    "content": "<think>Plan answer</think>\nFinal answer",
                ],
            ]],
        ])

        let message = try #require(firstMessage(from: normalized))
        #expect(message["reasoning_content"] as? String == "Plan answer")
        #expect(message["content"] as? String == "Final answer")
    }

    @Test("proxy non-streaming response handles orphan closing think tag")
    func proxyNormalizeHandlesOrphanClosingThinkTag() throws {
        let normalized = try normalizeProxyResponse([
            "choices": [[
                "message": [
                    "role": "assistant",
                    "content": "Plan without opener</think>\nFinal answer",
                ],
            ]],
        ])

        let message = try #require(firstMessage(from: normalized))
        #expect(message["reasoning_content"] as? String == "Plan without opener")
        #expect(message["content"] as? String == "Final answer")
    }

    private func normalizeProxyResponse(_ object: [String: Any]) throws -> [String: Any] {
        let data = try JSONSerialization.data(withJSONObject: object)
        let normalizedData = BackendProxyService.normalizeReasoningField(in: data)
        return try #require(JSONSerialization.jsonObject(with: normalizedData) as? [String: Any])
    }

    private func firstMessage(from object: [String: Any]) -> [String: Any]? {
        guard let choices = object["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any] else {
            return nil
        }
        return message
    }
}
