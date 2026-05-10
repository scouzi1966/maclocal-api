import Foundation
import Testing

@testable import MacLocalAPI

/// Tests for T1.6 (tokenize / count_tokens request decoding) and T1.7
/// (`/openapi.json` integrity).
struct TokenizeAndOpenAPITests {

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.6 — request shape decoding
    // ═══════════════════════════════════════════════════════════════════

    @Test("T1.6 tokenize request decodes `text` field")
    func decodeTextField() throws {
        let json = #"{"model":"m","text":"hello world"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.model == "m")
        #expect(req.text == "hello world")
        #expect(req.prompt == nil)
        #expect(req.effectiveText == "hello world")
    }

    @Test("T1.6 tokenize request decodes vLLM-style `prompt` alias")
    func decodePromptField() throws {
        let json = #"{"model":"m","prompt":"hi"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.text == nil)
        #expect(req.prompt == "hi")
        #expect(req.effectiveText == "hi")
    }

    @Test("T1.6 effectiveText prefers text over prompt when both supplied")
    func textWinsOverPrompt() throws {
        let json = #"{"text":"a","prompt":"b"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.effectiveText == "a")
    }

    @Test("T1.6 effectiveText is empty when neither field is supplied")
    func effectiveTextEmpty() throws {
        let json = #"{"model":"m"}"#
        let req = try JSONDecoder().decode(TokenizeRequest.self, from: Data(json.utf8))
        #expect(req.effectiveText == "")
    }

    @Test("T1.6 TokenizeResponse round-trips with snake_case max_model_len")
    func tokenizeResponseEncoding() throws {
        let resp = TokenizeResponse(tokens: [1, 2, 3], count: 3, model: "m", maxModelLen: 32768)
        let data = try JSONEncoder().encode(resp)
        let json = String(data: data, encoding: .utf8) ?? ""
        #expect(json.contains("\"tokens\":[1,2,3]"))
        #expect(json.contains("\"count\":3"))
        #expect(json.contains("\"max_model_len\":32768"))
    }

    @Test("T1.6 CountTokensResponse uses Anthropic input_tokens key")
    func countResponseEncoding() throws {
        let resp = CountTokensResponse(inputTokens: 42, model: "m")
        let data = try JSONEncoder().encode(resp)
        let json = String(data: data, encoding: .utf8) ?? ""
        #expect(json.contains("\"input_tokens\":42"))
        #expect(json.contains("\"model\":\"m\""))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - T1.7 — OpenAPI spec integrity
    // ═══════════════════════════════════════════════════════════════════

    @Test("T1.7 openapi.json is valid JSON and reports OpenAPI 3.1")
    func openAPISpecParses() throws {
        let data = Data(OpenAPIController.specJSON.utf8)
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(parsed != nil)
        #expect((parsed?["openapi"] as? String) == "3.1.0")
        #expect((parsed?["info"] as? [String: Any])?["title"] != nil)
    }

    @Test("T1.7 openapi.json declares the agent-relevant endpoints")
    func openAPICoversAgentEndpoints() throws {
        let data = Data(OpenAPIController.specJSON.utf8)
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let paths = parsed?["paths"] as? [String: Any] ?? [:]
        let expected = [
            "/v1/chat/completions",
            "/v1/chat/completions/{id}/cancel",
            "/v1/tokenize",
            "/v1/count_tokens",
            "/v1/embeddings",
            "/v1/audio/transcriptions",
            "/v1/audio/speech",
            "/v1/ocr",
            "/v1/batch/completions",
            "/v1/files",
            "/v1/models",
            "/health",
            "/metrics"
        ]
        for path in expected {
            #expect(paths[path] != nil, "missing path in OpenAPI spec: \(path)")
        }
    }

    @Test("T1.7 docs page references /openapi.json on same origin")
    func docsHTMLReferencesSpec() {
        let html = OpenAPIController.docsHTML
        #expect(html.contains("/openapi.json"))
        #expect(html.contains("scalar"))
    }
}
