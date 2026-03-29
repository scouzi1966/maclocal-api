import Testing
import Foundation
@testable import MacLocalAPI

// Integration tests — require a running AFM server on localhost:9998
// Run with: swift test --filter AFMClientTests
// Start server first: MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache .build/arm64-apple-macosx/release/afm mlx -m mlx-community/SmolLM3-3B-4bit --port 9998

@Suite("AFMClient Integration Tests")
struct AFMClientTests {
    let client = AFMClient(baseURL: "http://localhost:9998")

    @Test("Health endpoint returns status ok")
    func health() async throws {
        let health = try await client.health()
        #expect(health.status == "ok" || health.status == "healthy")
        #expect(health.version != nil)
        print("  ✅ Health: status=\(health.status), version=\(health.version ?? "?")")
    }

    @Test("Models endpoint returns at least one model")
    func models() async throws {
        let models = try await client.models()
        #expect(models.data.count > 0)
        let first = models.data[0]
        #expect(!first.id.isEmpty)
        print("  ✅ Models: \(models.data.count) model(s), first=\(first.id)")
    }

    @Test("Non-streaming chat completion returns content")
    func chatNonStreaming() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = [AFMClient.userMessage("Say hello in exactly 3 words.")]
        req.maxTokens = 50
        req.temperature = 0.0

        let resp = try await client.chatCompletion(req)
        #expect(resp.object == "chat.completion")
        #expect(resp.choices.count > 0)
        let content = resp.choices[0].message.content ?? ""
        let reasoning = resp.choices[0].message.reasoningContent ?? ""
        // Thinking models may put everything in reasoningContent with empty content
        #expect(!content.isEmpty || !reasoning.isEmpty, "Expected content or reasoningContent")
        #expect(resp.usage != nil)
        #expect(resp.usage!.completionTokens > 0)
        print("  ✅ Chat: content=\"\(content.prefix(60))\" reasoning=\(reasoning.count) chars (\(resp.usage!.completionTokens) tokens)")
    }

    @Test("Streaming chat completion yields chunks")
    func chatStreaming() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = [AFMClient.userMessage("Count from 1 to 5.")]
        req.maxTokens = 100
        req.temperature = 0.0

        let stream = try await client.chatCompletionStream(req)
        var chunks = 0
        var assembled = ""
        var reasoning = ""
        for try await chunk in stream {
            chunks += 1
            if let content = chunk.choices.first?.delta.content, !content.isEmpty {
                assembled += content
            }
            if let r = chunk.choices.first?.delta.reasoningContent, !r.isEmpty {
                reasoning += r
            }
        }
        #expect(chunks > 0)
        // Thinking models may put everything in reasoning with empty content
        #expect(!assembled.isEmpty || !reasoning.isEmpty, "Expected content or reasoning")
        print("  ✅ Stream: \(chunks) chunks, content=\"\(assembled.prefix(60))\" reasoning=\(reasoning.count) chars")
    }

    @Test("Logprobs returned when requested")
    func logprobs() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = [AFMClient.userMessage("What is 1+1?")]
        req.maxTokens = 20
        req.temperature = 0.0
        req.logprobs = true
        req.topLogprobs = 3

        let resp = try await client.chatCompletion(req)
        let lp = resp.choices[0].logprobs
        #expect(lp != nil)
        #expect(lp!.content != nil)
        #expect(lp!.content!.count > 0)
        let first = lp!.content![0]
        #expect(first.logprob <= 0)
        #expect(first.topLogprobs != nil)
        #expect(first.topLogprobs!.count <= 3)
        print("  ✅ Logprobs: \(lp!.content!.count) tokens, first token=\"\(first.token)\" logprob=\(first.logprob)")
    }

    @Test("Stop sequence stops generation")
    func stopSequence() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = [AFMClient.userMessage("Count from 1 to 10, one number per line.")]
        req.maxTokens = 200
        req.temperature = 0.0
        req.stop = ["5"]

        let resp = try await client.chatCompletion(req)
        let content = resp.choices[0].message.content ?? ""
        #expect(!content.contains("5"))
        #expect(resp.choices[0].finishReason == "stop")
        print("  ✅ Stop: finish_reason=stop, content doesn't contain '5'")
    }

    @Test("Error on empty messages returns 400")
    func errorHandling() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = []

        do {
            _ = try await client.chatCompletion(req)
            Issue.record("Expected error for empty messages")
        } catch let error as AFMClient.AFMClientError {
            switch error {
            case .httpError(let code, _):
                #expect(code == 400)
                print("  ✅ Error: got HTTP \(code) for empty messages")
            default:
                Issue.record("Expected httpError, got \(error)")
            }
        }
    }

    @Test("Usage includes timing fields")
    func usageTiming() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = [AFMClient.userMessage("Hi")]
        req.maxTokens = 10
        req.temperature = 0.0

        let resp = try await client.chatCompletion(req)
        let usage = resp.usage!
        #expect(usage.promptTokens > 0)
        #expect(usage.completionTokens > 0)
        #expect(usage.totalTokens == usage.promptTokens + usage.completionTokens)
        print("  ✅ Usage: prompt=\(usage.promptTokens) completion=\(usage.completionTokens) total=\(usage.totalTokens)")
        if let tps = usage.completionTokensPerSecond {
            print("       tok/s=\(String(format: "%.1f", tps))")
        }
    }

    @Test("Convenience builders produce correct messages")
    func messageBuilders() async throws {
        let sys = AFMClient.systemMessage("You are helpful.")
        #expect(sys.role == "system")

        let user = AFMClient.userMessage("Hello")
        #expect(user.role == "user")

        let assist = AFMClient.assistantMessage("Hi there")
        #expect(assist.role == "assistant")

        let tool = AFMClient.toolResult(callId: "call_123", content: "{\"result\": 42}")
        #expect(tool.role == "tool")
        #expect(tool.toolCallId == "call_123")

        print("  ✅ Message builders: system, user, assistant, tool all correct")
    }

    @Test("Streaming usage chunk appears at end")
    func streamingUsage() async throws {
        var req = AFMClient.ChatRequest()
        req.messages = [AFMClient.userMessage("Say one word.")]
        req.maxTokens = 10
        req.temperature = 0.0

        let stream = try await client.chatCompletionStream(req)
        var usageChunk: AFMClient.ChatCompletionChunk?
        for try await chunk in stream {
            if chunk.usage != nil {
                usageChunk = chunk
            }
        }
        // A chunk with usage should appear
        #expect(usageChunk != nil, "Expected a chunk with usage")
        if let usage = usageChunk?.usage {
            #expect(usage.totalTokens > 0)
            print("  ✅ Stream usage: total=\(usage.totalTokens)")
        }
    }
}
