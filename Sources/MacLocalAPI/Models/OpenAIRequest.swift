import Vapor
import Foundation

struct ChatCompletionRequest: Content {
    let model: String
    let messages: [Message]
    let temperature: Double?
    let maxTokens: Int?
    let topP: Double?
    let frequencyPenalty: Double?
    let presencePenalty: Double?
    let stop: [String]?
    let stream: Bool?
    let user: String?
    
    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case temperature
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case stop
        case stream
        case user
    }
}

struct Message: Content {
    let role: String
    let content: String
    
    init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}