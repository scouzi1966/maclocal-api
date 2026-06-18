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
    let streamOptions: StreamOptions?
    let user: String?
    let tools: [RequestTool]?
    let toolChoice: ToolChoice?
    let parallelToolCalls: Bool?
    let responseFormat: ResponseFormat?
    let chatTemplateKwargs: [String: AnyCodable]?

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
        case streamOptions = "stream_options"
        case user
        case tools
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case responseFormat = "response_format"
        case chatTemplateKwargs = "chat_template_kwargs"
    }

    /// Whether the final SSE chunk should carry a `usage` block. Mirrors OpenAI's
    /// `stream_options.include_usage`. Default true preserves existing behavior. (T1.2)
    var includeStreamingUsage: Bool {
        streamOptions?.includeUsage ?? true
    }

    var effectiveMaxTokens: Int? {
        maxTokens ?? maxCompletionTokens
    }

    var effectiveRepetitionPenalty: Double? {
        repetitionPenalty ?? repeatPenalty
    }
}

/// OpenAI-compatible `stream_options`. Currently models `include_usage`. (T1.2)
struct StreamOptions: Content {
    let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

// MARK: - Response format

public struct ResponseFormat: Content {
    public let type: String                    // "text", "json_object", "json_schema"
    public let jsonSchema: ResponseJsonSchema? // only for type="json_schema"

    public init(type: String, jsonSchema: ResponseJsonSchema? = nil) {
        self.type = type; self.jsonSchema = jsonSchema
    }

    enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }
}

public struct ResponseJsonSchema: Content {
    public let name: String?
    public let description: String?
    public let schema: AnyCodable?
    public let strict: Bool?
    public init(name: String?, description: String?, schema: AnyCodable?, strict: Bool?) {
        self.name = name; self.description = description; self.schema = schema; self.strict = strict
    }
}

// MARK: - Tool definitions

public struct RequestTool: Content {
    public let type: String          // "function"
    public let function: RequestToolFunction
}

public struct RequestToolFunction: Content {
    public let name: String
    public let description: String?
    public let parameters: AnyCodable?
    public let strict: Bool?
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

    public func encode(to encoder: Encoder) throws {
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
public enum MessageContent: Codable, Sendable {
    case text(String)
    case parts([ContentPart])

    public init(from decoder: Decoder) throws {
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

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let string):
            try container.encode(string)
        case .parts(let parts):
            try container.encode(parts)
        }
    }
}

public struct ContentPart: Codable, Sendable {
    public let type: String        // "text", "image_url", or "input_audio"
    public let text: String?       // For type="text"
    public let image_url: ImageURL? // For type="image_url"
    public let input_audio: InputAudio? // For type="input_audio"

    public init(type: String, text: String? = nil, image_url: ImageURL? = nil, input_audio: InputAudio? = nil) {
        self.type = type
        self.text = text
        self.image_url = image_url
        self.input_audio = input_audio
    }
}

public struct ImageURL: Codable, Sendable {
    public let url: String  // "data:image/png;base64,..." or URL
    public let detail: String?  // "auto", "low", "high" (optional)
    public init(url: String, detail: String?) { self.url = url; self.detail = detail }
}

public struct InputAudio: Codable, Sendable {
    public let data: String   // base64-encoded audio
    public let format: String // "wav", "mp3", etc.
    public let language: String? // locale for transcription (e.g. "en-US", "ja-JP")
}

public struct Message: Content {
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
    public init(role: String, content: String) {
        self.role = role
        self.content = .text(content)
        self.toolCalls = nil
        self.toolCallId = nil
        self.name = nil
    }

    public init(role: String, content: MessageContent?) {
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

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        // Accept arguments as either a JSON string or a JSON object/array
        if let str = try? container.decode(String.self, forKey: .arguments) {
            arguments = str
        } else if let obj = try? container.decode(AnyCodableValue.self, forKey: .arguments) {
            let data = try JSONSerialization.data(withJSONObject: obj.toAny(), options: [.sortedKeys])
            arguments = String(data: data, encoding: .utf8) ?? "{}"
        } else {
            arguments = "{}"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case name, arguments
    }
}

// MARK: - AnyCodable

/// Wraps arbitrary JSON values for decoding/encoding tool parameter schemas.
public struct AnyCodable: Codable, Sendable {
    let value: AnyCodableValue

    public init(_ value: Any) {
        self.value = AnyCodableValue.from(value)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.value = try AnyCodableValue(from: decoder)
        _ = container  // suppress warning
    }

    public func encode(to encoder: Encoder) throws {
        try value.encode(to: encoder)
    }

    /// Convert to a dictionary suitable for ToolSpec ([String: any Sendable])
    func toSendable() -> any Sendable {
        value.toAny()
    }

    /// Convert to a type hierarchy compatible with Jinja Value.init(any:).
    /// Strips null values from dicts (Jinja can't handle NSNull or boxed Optional<Any>).
    /// JSON Schema nulls (e.g. "default": null) are semantically equivalent when omitted.
    func toJinjaCompatible() -> any Sendable {
        value.toJinjaCompatible()
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

    func toAny() -> any Sendable {
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

    /// Convert to a type hierarchy compatible with Jinja Value.init(any:).
    /// Returns non-optional Any so it can be stored in [String: any Sendable] (ToolSpec).
    /// Null values are stripped from dicts because Jinja can't handle NSNull and
    /// Optional<Any> boxed in Any. JSON Schema nulls ("default": null) are semantically
    /// equivalent when omitted — templates use `is defined` checks, not null comparisons.
    /// Arrays filter out nulls. Standalone nulls become empty string (shouldn't occur in
    /// practice since null only appears as dict values or array elements in JSON Schema).
    func toJinjaCompatible() -> any Sendable {
        switch self {
        case .null: return "" // Standalone null fallback; dict/array nulls are stripped
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let arr):
            return arr.compactMap { element -> (any Sendable)? in
                if case .null = element { return nil }
                return element.toJinjaCompatible()
            }
        case .object(let dict):
            var result: [String: any Sendable] = [:]
            for (key, value) in dict {
                if case .null = value { continue } // Strip null-valued keys
                result[key] = value.toJinjaCompatible()
            }
            // Strip $schema if present (not used by any chat template)
            result.removeValue(forKey: "$schema")

            // Flatten anyOf/oneOf nullable patterns for Jinja template compatibility.
            // e.g. {"anyOf": [{"type": "string"}, {"type": "null"}]} → {"type": "string"}
            // Templates like Gemma 4 do `value['type'] | upper` which crashes on anyOf dicts.
            if result["type"] == nil, let anyOf = result["anyOf"] as? [[String: any Sendable]] ?? result["oneOf"] as? [[String: any Sendable]] {
                let nonNull = anyOf.filter { ($0["type"] as? String) != "null" }
                if nonNull.count == 1, let single = nonNull.first {
                    var flattened = result
                    flattened.removeValue(forKey: "anyOf")
                    flattened.removeValue(forKey: "oneOf")
                    for (k, v) in single { flattened[k] = v }
                    return flattened
                }
            }
            return result
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

// MARK: - Batch API Types

/// A single request entry in the SSE multiplex batch.
struct BatchRequestItem: Content {
    let customId: String
    let body: ChatCompletionRequest

    enum CodingKeys: String, CodingKey {
        case customId = "custom_id"
        case body
    }
}

/// Request body for POST /v1/batch/completions (SSE multiplex).
struct BatchCompletionRequest: Content {
    let requests: [BatchRequestItem]
}

/// Request body for POST /v1/batches (OpenAI-compatible).
struct BatchCreateRequest: Content {
    let inputFileId: String
    let endpoint: String
    let completionWindow: String?

    enum CodingKeys: String, CodingKey {
        case inputFileId = "input_file_id"
        case endpoint
        case completionWindow = "completion_window"
    }
}

/// A single line in the input JSONL file for batch processing.
struct BatchInputLine: Codable {
    let customId: String
    let method: String
    let url: String
    let body: ChatCompletionRequest

    enum CodingKeys: String, CodingKey {
        case customId = "custom_id"
        case method, url, body
    }
}

// MARK: - Embeddings API Types

enum EmbeddingInput: Content {
    case string(String)
    case array([String])
    case tokenIDs([Int])
    case arrayTokenIDs([[Int]])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let array = try? container.decode([String].self) {
            self = .array(array)
        } else if let tokenIDs = try? container.decode([Int].self) {
            self = .tokenIDs(tokenIDs)
        } else if let arrayTokenIDs = try? container.decode([[Int]].self) {
            self = .arrayTokenIDs(arrayTokenIDs)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Embedding input must be a string, array of strings, array of token ids, or array of token-id arrays"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let string):
            try container.encode(string)
        case .array(let array):
            try container.encode(array)
        case .tokenIDs(let tokenIDs):
            try container.encode(tokenIDs)
        case .arrayTokenIDs(let arrayTokenIDs):
            try container.encode(arrayTokenIDs)
        }
    }

    var strings: [String] {
        switch self {
        case .string(let string):
            return [string]
        case .array(let array):
            return array
        case .tokenIDs, .arrayTokenIDs:
            return []
        }
    }

    var tokenIDArrays: [[Int]] {
        switch self {
        case .string, .array:
            return []
        case .tokenIDs(let tokenIDs):
            return [tokenIDs]
        case .arrayTokenIDs(let arrayTokenIDs):
            return arrayTokenIDs
        }
    }

    var isEmpty: Bool {
        switch self {
        case .string(let string):
            return string.isEmpty
        case .array(let array):
            return array.isEmpty
        case .tokenIDs(let tokenIDs):
            return tokenIDs.isEmpty
        case .arrayTokenIDs(let arrayTokenIDs):
            return arrayTokenIDs.isEmpty
        }
    }

    var isTokenized: Bool {
        switch self {
        case .tokenIDs, .arrayTokenIDs:
            return true
        case .string, .array:
            return false
        }
    }
}

enum EmbeddingEncodingFormat: String, Content {
    case float
    case base64
}

struct EmbeddingsRequest: Content {
    let input: EmbeddingInput
    let model: String?
    let encodingFormat: EmbeddingEncodingFormat?
    let dimensions: Int?
    let user: String?

    enum CodingKeys: String, CodingKey {
        case input
        case model
        case encodingFormat = "encoding_format"
        case dimensions
        case user
    }

    var resolvedEncodingFormat: EmbeddingEncodingFormat {
        encodingFormat ?? .float
    }
}
