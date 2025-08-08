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