import Vapor
import Foundation

struct ChatCompletionResponse: Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
    let systemFingerprint: String?

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case model
        case choices
        case usage
        case systemFingerprint = "system_fingerprint"
    }

    private static func fingerprint(for model: String) -> String {
        if model == "foundation" {
            return "afm_apple_foundation"
        } else {
            let sanitized = model.replacingOccurrences(of: "/", with: "__").replacingOccurrences(of: " ", with: "_")
            return "afm_mlx__\(sanitized)"
        }
    }

    init(id: String = UUID().uuidString, model: String, content: String, reasoningContent: String? = nil, logprobs: ChoiceLogprobs? = nil, promptTokens: Int = 0, completionTokens: Int = 0) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: ResponseMessage(role: "assistant", content: content, reasoningContent: reasoningContent),
                logprobs: logprobs,
                finishReason: "stop"
            )
        ]
        self.usage = Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens
        )
        self.systemFingerprint = Self.fingerprint(for: model)
    }

    init(id: String = UUID().uuidString, model: String, toolCalls: [ResponseToolCall], logprobs: ChoiceLogprobs? = nil, promptTokens: Int = 0, completionTokens: Int = 0) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: ResponseMessage(role: "assistant", content: nil, toolCalls: toolCalls),
                logprobs: logprobs,
                finishReason: "tool_calls"
            )
        ]
        self.usage = Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens
        )
        self.systemFingerprint = Self.fingerprint(for: model)
    }
}

struct Choice: Content {
    let index: Int
    let message: ResponseMessage
    let logprobs: ChoiceLogprobs?
    let finishReason: String

    enum CodingKeys: String, CodingKey {
        case index
        case message
        case logprobs
        case finishReason = "finish_reason"
    }
}

struct ChoiceLogprobs: Content {
    let content: [TokenLogprobContent]?
}

struct TokenLogprobContent: Content {
    let token: String
    let logprob: Double
    let bytes: [Int]?
    let topLogprobs: [TopLogprobEntry]

    enum CodingKeys: String, CodingKey {
        case token
        case logprob
        case bytes
        case topLogprobs = "top_logprobs"
    }
}

struct TopLogprobEntry: Content {
    let token: String
    let logprob: Double
    let bytes: [Int]?
}

struct ResponseMessage: Content {
    let role: String
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ResponseToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    init(role: String, content: String?, reasoningContent: String? = nil, toolCalls: [ResponseToolCall]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }

    /// Always encode `content` (as null when nil) — Vercel AI SDK expects the key to be present.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encodeIfPresent(reasoningContent, forKey: .reasoningContent)
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
    }
}

// MARK: - Tool call response types

struct ResponseToolCall: Content {
    let id: String           // "call_<random>"
    let type: String         // "function"
    let function: ResponseToolCallFunction
}

struct ResponseToolCallFunction: Content {
    let name: String
    let arguments: String    // JSON string
}

struct Usage: Content {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

struct ChatCompletionStreamResponse: Content {
    let id: String
    let object: String
    let created: Int
    let model: String
    let systemFingerprint: String?
    let choices: [StreamChoice]
    let usage: StreamUsage?
    let timings: StreamTimings?

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage, timings
        case systemFingerprint = "system_fingerprint"
    }

    private static func fingerprint(for model: String) -> String {
        if model == "foundation" {
            return "afm_apple_foundation"
        } else {
            let sanitized = model.replacingOccurrences(of: "/", with: "__").replacingOccurrences(of: " ", with: "_")
            return "afm_mlx__\(sanitized)"
        }
    }

    init(id: String = UUID().uuidString, model: String, content: String, reasoningContent: String? = nil, logprobs: ChoiceLogprobs? = nil, isFinished: Bool = false, isFirst: Bool = false, finishReason: String? = nil, usage: StreamUsage? = nil, timings: StreamTimings? = nil) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.usage = usage
        self.timings = timings
        self.systemFingerprint = Self.fingerprint(for: model)

        if isFinished {
            self.choices = [
                StreamChoice(
                    index: 0,
                    delta: StreamDelta(content: nil),
                    logprobs: nil,
                    finishReason: finishReason ?? "stop"
                )
            ]
        } else if isFirst {
            self.choices = [
                StreamChoice(
                    index: 0,
                    delta: StreamDelta(role: "assistant", content: content, reasoningContent: reasoningContent),
                    logprobs: logprobs,
                    finishReason: nil
                )
            ]
        } else {
            self.choices = [
                StreamChoice(
                    index: 0,
                    delta: StreamDelta(content: content, reasoningContent: reasoningContent),
                    logprobs: logprobs,
                    finishReason: nil
                )
            ]
        }
    }

    /// Init for streaming tool call deltas
    init(id: String, model: String, toolCalls: [StreamDeltaToolCall], finishReason: String? = nil) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.usage = nil
        self.timings = nil
        self.systemFingerprint = Self.fingerprint(for: model)
        self.choices = [
            StreamChoice(
                index: 0,
                delta: StreamDelta(toolCalls: toolCalls),
                logprobs: nil,
                finishReason: finishReason
            )
        ]
    }
}

struct StreamTimings: Content {
    let prompt_n: Int
    let prompt_ms: Double
    let predicted_n: Int
    let predicted_ms: Double
}

struct StreamChoice: Content {
    let index: Int
    let delta: StreamDelta
    let logprobs: ChoiceLogprobs?
    let finishReason: String?

    enum CodingKeys: String, CodingKey {
        case index
        case delta
        case logprobs
        case finishReason = "finish_reason"
    }
}

struct StreamDelta: Content {
    let role: String?
    let content: String?
    let reasoningContent: String?
    let toolCalls: [StreamDeltaToolCall]?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    init(role: String? = nil, content: String?, reasoningContent: String? = nil, toolCalls: [StreamDeltaToolCall]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }

    init(toolCalls: [StreamDeltaToolCall]) {
        self.role = nil
        self.content = nil
        self.reasoningContent = nil
        self.toolCalls = toolCalls
    }
}

struct StreamDeltaToolCall: Content {
    let index: Int
    let id: String?
    let type: String?
    let function: StreamDeltaFunction?
}

struct StreamDeltaFunction: Content {
    let name: String?
    let arguments: String?
}

struct StreamUsage: Content {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int
    let completionTime: Double?
    let promptTime: Double?
    let totalTime: Double?
    let promptTokensPerSecond: Double?
    let completionTokensPerSecond: Double?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case completionTime = "completion_time"
        case promptTime = "prompt_time"
        case totalTime = "total_time"
        case promptTokensPerSecond = "prompt_tokens_per_second"
        case completionTokensPerSecond = "completion_tokens_per_second"
    }

    init(promptTokens: Int, completionTokens: Int, completionTime: Double, promptTime: Double = 0.0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
        self.completionTime = (completionTime * 100).rounded() / 100
        self.promptTime = (promptTime * 100).rounded() / 100
        self.totalTime = ((promptTime + completionTime) * 100).rounded() / 100

        if promptTime > 0 {
            let promptRate = Double(promptTokens) / promptTime
            self.promptTokensPerSecond = (promptRate * 100).rounded() / 100
        } else {
            self.promptTokensPerSecond = nil
        }

        if completionTime > 0 {
            let completionRate = Double(completionTokens) / completionTime
            self.completionTokensPerSecond = (completionRate * 100).rounded() / 100
        } else {
            self.completionTokensPerSecond = nil
        }
    }
}

struct OpenAIError: Content, Error {
    let error: ErrorDetail

    struct ErrorDetail: Content {
        let message: String
        let type: String
        let code: String?
    }

    init(message: String, type: String = "invalid_request_error", code: String? = nil) {
        self.error = ErrorDetail(message: message, type: type, code: code)
    }
}
