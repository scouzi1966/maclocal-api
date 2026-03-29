// MARK: - AFM Client Types
//
// Wire-compatible with the server types in OpenAIRequest.swift / OpenAIResponse.swift
// but defined independently so the client can be extracted as a standalone package.
// Every property uses explicit CodingKeys with snake_case JSON mapping.
//
// All types are nested inside `AFMClient` to avoid name collisions with the
// server-side types in the same module. Users reference them as e.g.
// `AFMClient.ChatRequest`, `AFMClient.Usage`, etc.

import Foundation

// ============================================================================
// MARK: - Request Types
// ============================================================================

extension AFMClient {

    // MARK: ChatRequest

    /// Chat completion request body for `POST /v1/chat/completions`.
    public struct ChatRequest: Encodable, Sendable {
        public var model: String?
        public var messages: [ChatMessage]
        public var temperature: Double?
        public var maxTokens: Int?
        public var maxCompletionTokens: Int?
        public var topP: Double?
        public var repetitionPenalty: Double?
        public var frequencyPenalty: Double?
        public var presencePenalty: Double?
        public var topK: Int?
        public var minP: Double?
        public var seed: Int?
        public var logprobs: Bool?
        public var topLogprobs: Int?
        public var stop: [String]?
        public var stream: Bool?
        public var tools: [Tool]?
        public var toolChoice: ToolChoice?
        public var responseFormat: ResponseFormat?
        public var chatTemplateKwargs: [String: JSONValue]?

        public init(
            model: String? = nil,
            messages: [ChatMessage] = [],
            temperature: Double? = nil,
            maxTokens: Int? = nil,
            maxCompletionTokens: Int? = nil,
            topP: Double? = nil,
            repetitionPenalty: Double? = nil,
            frequencyPenalty: Double? = nil,
            presencePenalty: Double? = nil,
            topK: Int? = nil,
            minP: Double? = nil,
            seed: Int? = nil,
            logprobs: Bool? = nil,
            topLogprobs: Int? = nil,
            stop: [String]? = nil,
            stream: Bool? = nil,
            tools: [Tool]? = nil,
            toolChoice: ToolChoice? = nil,
            responseFormat: ResponseFormat? = nil,
            chatTemplateKwargs: [String: JSONValue]? = nil
        ) {
            self.model = model
            self.messages = messages
            self.temperature = temperature
            self.maxTokens = maxTokens
            self.maxCompletionTokens = maxCompletionTokens
            self.topP = topP
            self.repetitionPenalty = repetitionPenalty
            self.frequencyPenalty = frequencyPenalty
            self.presencePenalty = presencePenalty
            self.topK = topK
            self.minP = minP
            self.seed = seed
            self.logprobs = logprobs
            self.topLogprobs = topLogprobs
            self.stop = stop
            self.stream = stream
            self.tools = tools
            self.toolChoice = toolChoice
            self.responseFormat = responseFormat
            self.chatTemplateKwargs = chatTemplateKwargs
        }

        enum CodingKeys: String, CodingKey {
            case model
            case messages
            case temperature
            case maxTokens = "max_tokens"
            case maxCompletionTokens = "max_completion_tokens"
            case topP = "top_p"
            case repetitionPenalty = "repetition_penalty"
            case frequencyPenalty = "frequency_penalty"
            case presencePenalty = "presence_penalty"
            case topK = "top_k"
            case minP = "min_p"
            case seed
            case logprobs
            case topLogprobs = "top_logprobs"
            case stop
            case stream
            case tools
            case toolChoice = "tool_choice"
            case responseFormat = "response_format"
            case chatTemplateKwargs = "chat_template_kwargs"
        }
    }

    // MARK: ChatMessage

    /// A single message in a chat conversation.
    public struct ChatMessage: Codable, Sendable {
        public var role: String
        public var content: MessageContent?
        public var toolCalls: [MessageToolCall]?
        public var toolCallId: String?
        public var name: String?

        public init(
            role: String,
            content: MessageContent? = nil,
            toolCalls: [MessageToolCall]? = nil,
            toolCallId: String? = nil,
            name: String? = nil
        ) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
            self.name = name
        }

        /// Convenience: create a simple text message.
        public init(role: String, text: String) {
            self.role = role
            self.content = .text(text)
            self.toolCalls = nil
            self.toolCallId = nil
            self.name = nil
        }

        enum CodingKeys: String, CodingKey {
            case role
            case content
            case toolCalls = "tool_calls"
            case toolCallId = "tool_call_id"
            case name
        }
    }

    // MARK: MessageContent

    /// Content can be a simple string or an array of content parts (multimodal).
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
                    debugDescription: "Content must be a string or array of content parts"
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

    // MARK: ContentPart

    /// A single part of a multimodal message.
    public enum ContentPart: Codable, Sendable {
        case text(String)
        case imageURL(url: String, detail: ImageDetail?)
        case imageData(base64: String, mediaType: String, detail: ImageDetail?)

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "text":
                let text = try container.decode(String.self, forKey: .text)
                self = .text(text)
            case "image_url":
                let imageURL = try container.decode(ImageURLPayload.self, forKey: .imageUrl)
                let detail = imageURL.detail.flatMap { ImageDetail(rawValue: $0) }
                if imageURL.url.hasPrefix("data:") {
                    // Parse data URI: data:<mediaType>;base64,<data>
                    let stripped = imageURL.url.dropFirst(5) // remove "data:"
                    if let semicolonIdx = stripped.firstIndex(of: ";"),
                       let commaIdx = stripped.firstIndex(of: ",") {
                        let mediaType = String(stripped[stripped.startIndex..<semicolonIdx])
                        let base64 = String(stripped[stripped.index(after: commaIdx)...])
                        self = .imageData(base64: base64, mediaType: mediaType, detail: detail)
                    } else {
                        self = .imageURL(url: imageURL.url, detail: detail)
                    }
                } else {
                    self = .imageURL(url: imageURL.url, detail: detail)
                }
            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: container,
                    debugDescription: "Unknown content part type: \(type)"
                )
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .text(let text):
                try container.encode("text", forKey: .type)
                try container.encode(text, forKey: .text)
            case .imageURL(let url, let detail):
                try container.encode("image_url", forKey: .type)
                try container.encode(
                    ImageURLPayload(url: url, detail: detail?.rawValue),
                    forKey: .imageUrl
                )
            case .imageData(let base64, let mediaType, let detail):
                try container.encode("image_url", forKey: .type)
                let dataURI = "data:\(mediaType);base64,\(base64)"
                try container.encode(
                    ImageURLPayload(url: dataURI, detail: detail?.rawValue),
                    forKey: .imageUrl
                )
            }
        }

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case imageUrl = "image_url"
        }
    }

    /// Payload for image_url content parts.
    public struct ImageURLPayload: Codable, Sendable {
        public var url: String
        public var detail: String?

        public init(url: String, detail: String? = nil) {
            self.url = url
            self.detail = detail
        }
    }

    /// Image detail level for vision requests.
    public enum ImageDetail: String, Codable, Sendable {
        case auto
        case low
        case high
    }

    // MARK: MessageToolCall (for multi-turn conversations)

    /// A tool call in a message (used for assistant messages in multi-turn).
    public struct MessageToolCall: Codable, Sendable {
        public var id: String
        public var type: String
        public var function: MessageToolCallFunction

        public init(id: String, type: String = "function", function: MessageToolCallFunction) {
            self.id = id
            self.type = type
            self.function = function
        }
    }

    /// Function details of a message tool call.
    public struct MessageToolCallFunction: Codable, Sendable {
        public var name: String
        public var arguments: String

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }
    }

    // MARK: Tool

    /// Tool definition for function calling.
    public struct Tool: Encodable, Sendable {
        public var type: String
        public var function: ToolFunction

        public init(type: String = "function", function: ToolFunction) {
            self.type = type
            self.function = function
        }
    }

    /// Function definition within a tool.
    public struct ToolFunction: Encodable, Sendable {
        public var name: String
        public var description: String?
        public var parameters: JSONValue?
        public var strict: Bool?

        public init(name: String, description: String? = nil, parameters: JSONValue? = nil, strict: Bool? = nil) {
            self.name = name
            self.description = description
            self.parameters = parameters
            self.strict = strict
        }
    }

    // MARK: ToolChoice

    /// Controls which tool the model should call.
    public enum ToolChoice: Codable, Sendable {
        /// A string mode: "auto", "none", "required"
        case mode(String)
        /// Force a specific function by name.
        case function(name: String)

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .mode(str)
            } else if let fn = try? container.decode(ToolChoiceFunctionWrapper.self) {
                self = .function(name: fn.function.name)
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
            case .function(let name):
                try container.encode(
                    ToolChoiceFunctionWrapper(
                        type: "function",
                        function: ToolChoiceFunctionName(name: name)
                    )
                )
            }
        }
    }

    struct ToolChoiceFunctionWrapper: Codable, Sendable {
        var type: String
        var function: ToolChoiceFunctionName
    }

    struct ToolChoiceFunctionName: Codable, Sendable {
        var name: String
    }

    // MARK: ResponseFormat

    /// Controls the response output format.
    public struct ResponseFormat: Encodable, Sendable {
        public var type: String
        public var jsonSchema: JSONSchemaSpec?

        public init(type: String, jsonSchema: JSONSchemaSpec? = nil) {
            self.type = type
            self.jsonSchema = jsonSchema
        }

        /// Convenience: plain text format.
        public static var text: ResponseFormat { ResponseFormat(type: "text") }

        /// Convenience: JSON object format.
        public static var jsonObject: ResponseFormat { ResponseFormat(type: "json_object") }

        /// Convenience: JSON schema format.
        public static func jsonSchema(_ schema: JSONSchemaSpec) -> ResponseFormat {
            ResponseFormat(type: "json_schema", jsonSchema: schema)
        }

        enum CodingKeys: String, CodingKey {
            case type
            case jsonSchema = "json_schema"
        }
    }

    /// JSON Schema specification for structured output.
    public struct JSONSchemaSpec: Encodable, Sendable {
        public var name: String?
        public var description: String?
        public var schema: JSONValue?
        public var strict: Bool?

        public init(name: String? = nil, description: String? = nil, schema: JSONValue? = nil, strict: Bool? = nil) {
            self.name = name
            self.description = description
            self.schema = schema
            self.strict = strict
        }
    }

    // MARK: BatchRequestItem (for SSE multiplex)

    /// A single request in the SSE multiplex batch endpoint.
    public struct BatchRequestItem: Encodable, Sendable {
        public var customId: String
        public var body: ChatRequest

        public init(customId: String, body: ChatRequest) {
            self.customId = customId
            self.body = body
        }

        enum CodingKeys: String, CodingKey {
            case customId = "custom_id"
            case body
        }
    }
}

// ============================================================================
// MARK: - Response Types
// ============================================================================

extension AFMClient {

    // MARK: ChatCompletionResponse

    /// Non-streaming chat completion response.
    public struct ChatCompletionResponse: Decodable, Sendable {
        public var id: String
        public var object: String
        public var created: Int
        public var model: String
        public var choices: [Choice]
        public var usage: Usage?
        public var timings: Timings?
        public var systemFingerprint: String?
        public var afmProfile: AFMProfileData?
        public var afmProfileExtended: AFMProfileExtendedData?

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case created
            case model
            case choices
            case usage
            case timings
            case systemFingerprint = "system_fingerprint"
            case afmProfile = "afm_profile"
            case afmProfileExtended = "afm_profile_extended"
        }

        /// Convenience: extract the first choice's text content.
        public var text: String? {
            choices.first?.message.content
        }

        /// Convenience: extract reasoning content from the first choice.
        public var reasoningContent: String? {
            choices.first?.message.reasoningContent
        }

        /// Convenience: extract tool calls from the first choice.
        public var toolCalls: [ResponseToolCall]? {
            choices.first?.message.toolCalls
        }
    }

    // MARK: Choice

    /// A single choice in a non-streaming response.
    public struct Choice: Decodable, Sendable {
        public var index: Int
        public var message: ResponseMessage
        public var logprobs: ChoiceLogprobs?
        public var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case message
            case logprobs
            case finishReason = "finish_reason"
        }
    }

    // MARK: ResponseMessage

    /// The assistant's message in a non-streaming response.
    public struct ResponseMessage: Decodable, Sendable {
        public var role: String?
        public var content: String?
        public var reasoningContent: String?
        public var toolCalls: [ResponseToolCall]?

        enum CodingKeys: String, CodingKey {
            case role
            case content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }
    }

    // MARK: ResponseToolCall

    /// A tool call emitted by the model.
    public struct ResponseToolCall: Decodable, Sendable {
        public var index: Int?
        public var id: String?
        public var type: String?
        public var function: ResponseToolCallFunction?

        enum CodingKeys: String, CodingKey {
            case index
            case id
            case type
            case function
        }
    }

    /// Function details in a tool call response.
    public struct ResponseToolCallFunction: Decodable, Sendable {
        public var name: String?
        public var arguments: String?

        enum CodingKeys: String, CodingKey {
            case name
            case arguments
        }
    }

    // MARK: Logprobs

    /// Logprob data attached to a choice.
    public struct ChoiceLogprobs: Decodable, Sendable {
        public var content: [TokenLogprob]?

        enum CodingKeys: String, CodingKey {
            case content
        }
    }

    /// Token-level logprob information.
    public struct TokenLogprob: Decodable, Sendable {
        public var token: String
        public var logprob: Double
        public var bytes: [Int]?
        public var topLogprobs: [TopLogprob]?

        enum CodingKeys: String, CodingKey {
            case token
            case logprob
            case bytes
            case topLogprobs = "top_logprobs"
        }
    }

    /// A single entry in the top-logprobs array.
    public struct TopLogprob: Decodable, Sendable {
        public var token: String
        public var logprob: Double
        public var bytes: [Int]?

        enum CodingKeys: String, CodingKey {
            case token
            case logprob
            case bytes
        }
    }

    // MARK: Usage

    /// Token usage statistics.
    public struct Usage: Decodable, Sendable {
        public var promptTokens: Int
        public var completionTokens: Int
        public var totalTokens: Int
        public var promptTokensDetails: PromptTokensDetails?
        public var completionTime: Double?
        public var promptTime: Double?
        public var totalTime: Double?
        public var completionTokensPerSecond: Double?
        public var promptTokensPerSecond: Double?
        public var peakMemoryGib: Double?

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
            case promptTokensDetails = "prompt_tokens_details"
            case completionTime = "completion_time"
            case promptTime = "prompt_time"
            case totalTime = "total_time"
            case completionTokensPerSecond = "completion_tokens_per_second"
            case promptTokensPerSecond = "prompt_tokens_per_second"
            case peakMemoryGib = "peak_memory_gib"
        }
    }

    /// Prompt token breakdown details.
    public struct PromptTokensDetails: Decodable, Sendable {
        public var cachedTokens: Int?

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }
    }

    // MARK: Timings

    /// llama.cpp-style timing information.
    public struct Timings: Decodable, Sendable {
        public var promptN: Int?
        public var promptMs: Double?
        public var predictedN: Int?
        public var predictedMs: Double?

        enum CodingKeys: String, CodingKey {
            case promptN = "prompt_n"
            case promptMs = "prompt_ms"
            case predictedN = "predicted_n"
            case predictedMs = "predicted_ms"
        }
    }
}

// ============================================================================
// MARK: - Streaming Types
// ============================================================================

extension AFMClient {

    /// A single chunk in a streaming chat completion response.
    public struct ChatCompletionChunk: Decodable, Sendable {
        public var id: String
        public var object: String
        public var created: Int
        public var model: String
        public var choices: [StreamChoice]
        public var usage: Usage?
        public var timings: Timings?
        public var systemFingerprint: String?
        public var afmProfile: AFMProfileData?
        public var afmProfileExtended: AFMProfileExtendedData?

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case created
            case model
            case choices
            case usage
            case timings
            case systemFingerprint = "system_fingerprint"
            case afmProfile = "afm_profile"
            case afmProfileExtended = "afm_profile_extended"
        }

        /// Convenience: extract text delta from the first choice.
        public var text: String? {
            choices.first?.delta.content
        }

        /// Convenience: extract reasoning delta from the first choice.
        public var reasoningText: String? {
            choices.first?.delta.reasoningContent
        }

        /// Convenience: check if this chunk signals completion.
        public var isFinished: Bool {
            choices.first?.finishReason != nil
        }

        /// Convenience: the finish reason, if present.
        public var finishReason: String? {
            choices.first?.finishReason
        }
    }

    /// A single choice in a streaming chunk.
    public struct StreamChoice: Decodable, Sendable {
        public var index: Int
        public var delta: StreamDelta
        public var logprobs: ChoiceLogprobs?
        public var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case logprobs
            case finishReason = "finish_reason"
        }
    }

    /// The delta payload in a streaming choice.
    public struct StreamDelta: Decodable, Sendable {
        public var role: String?
        public var content: String?
        public var reasoningContent: String?
        public var toolCalls: [ResponseToolCall]?

        enum CodingKeys: String, CodingKey {
            case role
            case content
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
        }
    }
}

// ============================================================================
// MARK: - GPU Profiling Types
// ============================================================================

extension AFMClient {

    /// GPU profiling summary returned with `X-AFM-Profile: true`.
    public struct AFMProfileData: Decodable, Sendable {
        public var gpuPowerAvgW: Double?
        public var gpuPowerPeakW: Double?
        public var gpuSamples: Int?
        public var memoryWeightsGib: Double?
        public var memoryKvGib: Double?
        public var memoryPeakGib: Double?
        public var prefillTokS: Double?
        public var decodeTokS: Double?
        public var chip: String?
        public var theoreticalBwGbs: Double?
        public var estBandwidthGbs: Double?

        enum CodingKeys: String, CodingKey {
            case gpuPowerAvgW = "gpu_power_avg_w"
            case gpuPowerPeakW = "gpu_power_peak_w"
            case gpuSamples = "gpu_samples"
            case memoryWeightsGib = "memory_weights_gib"
            case memoryKvGib = "memory_kv_gib"
            case memoryPeakGib = "memory_peak_gib"
            case prefillTokS = "prefill_tok_s"
            case decodeTokS = "decode_tok_s"
            case chip
            case theoreticalBwGbs = "theoretical_bw_gbs"
            case estBandwidthGbs = "est_bandwidth_gbs"
        }
    }

    /// Extended GPU profile with time-series samples.
    /// Returned with `X-AFM-Profile: extended`.
    public struct AFMProfileExtendedData: Decodable, Sendable {
        public var summary: AFMProfileData
        public var samples: [AFMProfileSampleData]

        enum CodingKeys: String, CodingKey {
            case summary
            case samples
        }
    }

    /// A single GPU profiling sample (300ms interval).
    public struct AFMProfileSampleData: Decodable, Sendable {
        public var t: Double
        public var bwGbs: Double?
        public var gpuPct: Double?
        public var gpuPowerW: Double?
        public var dramPowerW: Double?

        enum CodingKeys: String, CodingKey {
            case t
            case bwGbs = "bw_gbs"
            case gpuPct = "gpu_pct"
            case gpuPowerW = "gpu_power_w"
            case dramPowerW = "dram_power_w"
        }
    }
}

// ============================================================================
// MARK: - Batch API Types
// ============================================================================

extension AFMClient {

    /// OpenAI File object.
    public struct FileObject: Decodable, Sendable {
        public var id: String
        public var object: String
        public var bytes: Int
        public var createdAt: Int
        public var filename: String
        public var purpose: String

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case bytes
            case createdAt = "created_at"
            case filename
            case purpose
        }
    }

    /// Response for `DELETE /v1/files/{id}`.
    public struct FileDeleteResponse: Decodable, Sendable {
        public var id: String
        public var object: String
        public var deleted: Bool

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case deleted
        }
    }

    /// OpenAI Batch object.
    public struct BatchObject: Decodable, Sendable {
        public var id: String
        public var object: String
        public var endpoint: String
        public var status: String
        public var inputFileId: String
        public var completionWindow: String?
        public var createdAt: Int
        public var completedAt: Int?
        public var outputFileId: String?
        public var requestCounts: BatchRequestCounts?

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case endpoint
            case status
            case inputFileId = "input_file_id"
            case completionWindow = "completion_window"
            case createdAt = "created_at"
            case completedAt = "completed_at"
            case outputFileId = "output_file_id"
            case requestCounts = "request_counts"
        }
    }

    /// Request counts within a batch.
    public struct BatchRequestCounts: Decodable, Sendable {
        public var total: Int
        public var completed: Int
        public var failed: Int

        enum CodingKeys: String, CodingKey {
            case total
            case completed
            case failed
        }
    }

    /// Response for `GET /v1/batches` (list).
    public struct BatchListResponse: Decodable, Sendable {
        public var object: String
        public var data: [BatchObject]
        public var hasMore: Bool

        enum CodingKeys: String, CodingKey {
            case object
            case data
            case hasMore = "has_more"
        }
    }
}

// ============================================================================
// MARK: - Error, Health, Models Types
// ============================================================================

extension AFMClient {

    /// API error response body.
    public struct APIError: Decodable, Sendable {
        public var error: APIErrorDetail

        enum CodingKeys: String, CodingKey {
            case error
        }
    }

    /// Detail within an API error response.
    public struct APIErrorDetail: Decodable, Sendable {
        public var message: String
        public var type: String
        public var code: String?

        enum CodingKeys: String, CodingKey {
            case message
            case type
            case code
        }
    }

    /// Response from `GET /health`.
    public struct HealthResponse: Decodable, Sendable {
        public var status: String
        public var timestamp: Double?
        public var version: String?

        enum CodingKeys: String, CodingKey {
            case status
            case timestamp
            case version
        }
    }

    /// Response from `GET /v1/models`.
    public struct ModelsResponse: Decodable, Sendable {
        public var object: String
        public var data: [ModelInfo]

        enum CodingKeys: String, CodingKey {
            case object
            case data
        }
    }

    /// A single model entry.
    public struct ModelInfo: Decodable, Sendable {
        public var id: String
        public var object: String
        public var created: Int?
        public var ownedBy: String?

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case created
            case ownedBy = "owned_by"
        }
    }

    /// Errors thrown by `AFMClient`.
    public enum AFMClientError: Error, LocalizedError, Sendable {
        /// Server returned a non-2xx HTTP status.
        case httpError(statusCode: Int, detail: APIErrorDetail?)
        /// JSON decoding failed.
        case decodingError(Error)
        /// Network-level failure (DNS, timeout, connection refused, etc.).
        case connectionError(Error)
        /// SSE stream parse failure.
        case streamError(String)
        /// Invalid URL construction.
        case invalidURL(String)

        public var errorDescription: String? {
            switch self {
            case .httpError(let code, let detail):
                if let detail {
                    return "HTTP \(code): \(detail.message)"
                }
                return "HTTP \(code)"
            case .decodingError(let error):
                return "Decoding error: \(error.localizedDescription)"
            case .connectionError(let error):
                return "Connection error: \(error.localizedDescription)"
            case .streamError(let message):
                return "Stream error: \(message)"
            case .invalidURL(let url):
                return "Invalid URL: \(url)"
            }
        }
    }

    /// Level of GPU profiling to request via the `X-AFM-Profile` header.
    public enum ProfileLevel: String, Sendable {
        case basic = "true"
        case extended = "extended"
    }
}
