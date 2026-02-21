import Vapor
import Foundation

struct ChatCompletionRequest: Content {
    let model: String?
    let messages: [Message]
    let temperature: Double?
    let maxTokens: Int?
    let maxCompletionTokens: Int?
    let topP: Double?
    let repetitionPenalty: Double?
    let repeatPenalty: Double?
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
        case maxCompletionTokens = "max_completion_tokens"
        case topP = "top_p"
        case repetitionPenalty = "repetition_penalty"
        case repeatPenalty = "repeat_penalty"
        case frequencyPenalty = "frequency_penalty"
        case presencePenalty = "presence_penalty"
        case stop
        case stream
        case user
    }

    var effectiveMaxTokens: Int? {
        maxTokens ?? maxCompletionTokens
    }

    var effectiveRepetitionPenalty: Double? {
        repetitionPenalty ?? repeatPenalty
    }
}

/// Content can be a simple string or an array of content parts (multimodal)
enum MessageContent: Codable {
    case text(String)
    case parts([ContentPart])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            self = .text(string)
        } else if let parts = try? container.decode([ContentPart].self) {
            self = .parts(parts)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Content must be string or array of content parts"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let string):
            try container.encode(string)
        case .parts(let parts):
            try container.encode(parts)
        }
    }
}

struct ContentPart: Codable {
    let type: String        // "text" or "image_url"
    let text: String?       // For type="text"
    let image_url: ImageURL? // For type="image_url"
}

struct ImageURL: Codable {
    let url: String  // "data:image/png;base64,..." or URL
    let detail: String?  // "auto", "low", "high" (optional)
}

struct Message: Content {
    let role: String
    let content: MessageContent

    /// Convenience initializer for simple text messages
    init(role: String, content: String) {
        self.role = role
        self.content = .text(content)
    }

    init(role: String, content: MessageContent) {
        self.role = role
        self.content = content
    }

    /// Get combined text content from all text parts
    var textContent: String {
        switch content {
        case .text(let str):
            return str
        case .parts(let parts):
            return parts.compactMap { $0.text }.joined(separator: "\n")
        }
    }
}
