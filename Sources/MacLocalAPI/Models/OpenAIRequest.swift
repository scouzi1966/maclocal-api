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
    let topK: Int?
    let minP: Double?
    let seed: Int?
    let logprobs: Bool?
    let topLogprobs: Int?
    let stop: [String]?
    let stream: Bool?
    let user: String?
    let tools: [RequestTool]?
    let toolChoice: ToolChoice?
    let responseFormat: ResponseFormat?

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
        case topK = "top_k"
        case minP = "min_p"
        case seed
        case logprobs
        case topLogprobs = "top_logprobs"
        case stop
        case stream
        case user
        case tools
        case toolChoice = "tool_choice"
        case responseFormat = "response_format"
    }

    var effectiveMaxTokens: Int? {
        maxTokens ?? maxCompletionTokens
    }

    var effectiveRepetitionPenalty: Double? {
        repetitionPenalty ?? repeatPenalty
    }
}

// MARK: - Response format

struct ResponseFormat: Content {
    let type: String                    // "text", "json_object", "json_schema"
    let jsonSchema: ResponseJsonSchema? // only for type="json_schema"

    enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }
}

struct ResponseJsonSchema: Content {
    let name: String?
    let description: String?
    let schema: AnyCodable?
    let strict: Bool?
}

// MARK: - Tool definitions

struct RequestTool: Content {
    let type: String          // "function"
    let function: RequestToolFunction
}

struct RequestToolFunction: Content {
    let name: String
    let description: String?
    let parameters: AnyCodable?
}

enum ToolChoice: Codable {
    case mode(String)                        // "auto", "none", "required"
    case function(ToolChoiceFunction)         // {"type":"function","function":{"name":"..."}}

    struct ToolChoiceFunction: Codable {
        let type: String
        let function: ToolChoiceFunctionName
    }

    struct ToolChoiceFunctionName: Codable {
        let name: String
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .mode(str)
        } else if let fn = try? container.decode(ToolChoiceFunction.self) {
            self = .function(fn)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "tool_choice must be a string or {\"type\":\"function\",\"function\":{\"name\":\"...\"}}"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .mode(let str):
            try container.encode(str)
        case .function(let fn):
            try container.encode(fn)
        }
    }
}

// MARK: - Message content

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
    let content: MessageContent?    // optional — null for tool-call assistant messages
    let toolCalls: [MessageToolCall]?
    let toolCallId: String?
    let name: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
        case name
    }

    /// Convenience initializer for simple text messages
    init(role: String, content: String) {
        self.role = role
        self.content = .text(content)
        self.toolCalls = nil
        self.toolCallId = nil
        self.name = nil
    }

    init(role: String, content: MessageContent?) {
        self.role = role
        self.content = content
        self.toolCalls = nil
        self.toolCallId = nil
        self.name = nil
    }

    /// Get combined text content from all text parts
    var textContent: String {
        guard let content else { return "" }
        switch content {
        case .text(let str):
            return str
        case .parts(let parts):
            return parts.compactMap { $0.text }.joined(separator: "\n")
        }
    }
}

// MARK: - Tool call messages (multi-turn)

struct MessageToolCall: Content {
    let id: String
    let type: String        // "function"
    let function: MessageToolCallFunction
}

struct MessageToolCallFunction: Content {
    let name: String
    let arguments: String   // JSON string
}

// MARK: - AnyCodable

/// Wraps arbitrary JSON values for decoding/encoding tool parameter schemas.
struct AnyCodable: Codable, Sendable {
    let value: AnyCodableValue

    init(_ value: Any) {
        self.value = AnyCodableValue.from(value)
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.value = try AnyCodableValue(from: decoder)
        _ = container  // suppress warning
    }

    func encode(to encoder: Encoder) throws {
        try value.encode(to: encoder)
    }

    /// Convert to a dictionary suitable for ToolSpec ([String: any Sendable])
    func toSendable() -> Any {
        value.toAny()
    }
}

/// Recursive enum representing arbitrary JSON values.
enum AnyCodableValue: Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AnyCodableValue])
    case object([String: AnyCodableValue])

    static func from(_ value: Any) -> AnyCodableValue {
        switch value {
        case is NSNull:
            return .null
        case let b as Bool:
            return .bool(b)
        case let i as Int:
            return .int(i)
        case let d as Double:
            return .double(d)
        case let s as String:
            return .string(s)
        case let arr as [Any]:
            return .array(arr.map { from($0) })
        case let dict as [String: Any]:
            return .object(dict.mapValues { from($0) })
        default:
            return .string(String(describing: value))
        }
    }

    func toAny() -> Any {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let arr): return arr.map { $0.toAny() }
        case .object(let dict): return dict.mapValues { $0.toAny() }
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let arr = try? container.decode([AnyCodableValue].self) {
            self = .array(arr)
        } else if let obj = try? container.decode([String: AnyCodableValue].self) {
            self = .object(obj)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let b): try container.encode(b)
        case .int(let i): try container.encode(i)
        case .double(let d): try container.encode(d)
        case .string(let s): try container.encode(s)
        case .array(let arr): try container.encode(arr)
        case .object(let obj): try container.encode(obj)
        }
    }
}
