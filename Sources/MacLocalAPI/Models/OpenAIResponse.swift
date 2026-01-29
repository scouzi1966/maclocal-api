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
    
    init(id: String = UUID().uuidString, model: String, content: String, promptTokens: Int = 0, completionTokens: Int = 0) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: ResponseMessage(role: "assistant", content: content),
                finishReason: "stop"
            )
        ]
        self.usage = Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens
        )
        self.systemFingerprint = "fp_apple_foundation"
    }
}

struct Choice: Content {
    let index: Int
    let message: ResponseMessage
    let finishReason: String
    
    enum CodingKeys: String, CodingKey {
        case index
        case message
        case finishReason = "finish_reason"
    }
}

struct ResponseMessage: Content {
    let role: String
    let content: String
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
    let choices: [StreamChoice]
    let usage: StreamUsage?
    let timings: StreamTimings?

    init(id: String = UUID().uuidString, model: String, content: String, isFinished: Bool = false, isFirst: Bool = false, usage: StreamUsage? = nil, timings: StreamTimings? = nil) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.usage = usage
        self.timings = timings

        if isFinished {
            self.choices = [
                StreamChoice(
                    index: 0,
                    delta: StreamDelta(content: nil),
                    finishReason: "stop"
                )
            ]
        } else if isFirst {
            self.choices = [
                StreamChoice(
                    index: 0,
                    delta: StreamDelta(role: "assistant", content: content),
                    finishReason: nil
                )
            ]
        } else {
            self.choices = [
                StreamChoice(
                    index: 0,
                    delta: StreamDelta(content: content),
                    finishReason: nil
                )
            ]
        }
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
    let finishReason: String?
    
    enum CodingKeys: String, CodingKey {
        case index
        case delta
        case finishReason = "finish_reason"
    }
}

struct StreamDelta: Content {
    let role: String?
    let content: String?
    
    init(role: String? = nil, content: String?) {
        self.role = role
        self.content = content
    }
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