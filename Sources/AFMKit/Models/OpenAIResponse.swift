import Foundation
import MLX

/// AFM-specific GPU profiling data, returned when client sends `X-AFM-Profile: true` header.
public struct AFMProfile: Codable, Sendable {
    public let gpuPowerAvgW: Double?
    public let gpuPowerPeakW: Double?
    public let gpuSamples: Int?
    public let memoryWeightsGiB: Double?
    public let memoryKvGiB: Double?
    public let memoryPeakGiB: Double?
    public let prefillTokS: Double?
    public let decodeTokS: Double?
    public let chip: String?
    public let theoreticalBwGbs: Double?
    public let estBandwidthGbs: Double?

    public enum CodingKeys: String, CodingKey {
        case gpuPowerAvgW = "gpu_power_avg_w"
        case gpuPowerPeakW = "gpu_power_peak_w"
        case gpuSamples = "gpu_samples"
        case memoryWeightsGiB = "memory_weights_gib"
        case memoryKvGiB = "memory_kv_gib"
        case memoryPeakGiB = "memory_peak_gib"
        case prefillTokS = "prefill_tok_s"
        case decodeTokS = "decode_tok_s"
        case chip
        case theoreticalBwGbs = "theoretical_bw_gbs"
        case estBandwidthGbs = "est_bandwidth_gbs"
    }
    public init(gpuPowerAvgW: Double?, gpuPowerPeakW: Double?, gpuSamples: Int?, memoryWeightsGiB: Double?, memoryKvGiB: Double?, memoryPeakGiB: Double?, prefillTokS: Double?, decodeTokS: Double?, chip: String?, theoreticalBwGbs: Double?, estBandwidthGbs: Double?) {
        self.gpuPowerAvgW = gpuPowerAvgW
        self.gpuPowerPeakW = gpuPowerPeakW
        self.gpuSamples = gpuSamples
        self.memoryWeightsGiB = memoryWeightsGiB
        self.memoryKvGiB = memoryKvGiB
        self.memoryPeakGiB = memoryPeakGiB
        self.prefillTokS = prefillTokS
        self.decodeTokS = decodeTokS
        self.chip = chip
        self.theoreticalBwGbs = theoreticalBwGbs
        self.estBandwidthGbs = estBandwidthGbs
    }
}

/// A single IOReport GPU sample (300ms interval).
public struct AFMProfileSample: Codable, Sendable {
    public let t: Double
    public let bwGbs: Double?
    public let gpuPct: Double
    public let gpuPowerW: Double
    public let dramPowerW: Double

    public enum CodingKeys: String, CodingKey {
        case t
        case bwGbs = "bw_gbs"
        case gpuPct = "gpu_pct"
        case gpuPowerW = "gpu_power_w"
        case dramPowerW = "dram_power_w"
    }
    public init(t: Double, bwGbs: Double?, gpuPct: Double, gpuPowerW: Double, dramPowerW: Double) {
        self.t = t
        self.bwGbs = bwGbs
        self.gpuPct = gpuPct
        self.gpuPowerW = gpuPowerW
        self.dramPowerW = dramPowerW
    }
}

/// Extended profile with time-series samples.
/// Returned when client sends `X-AFM-Profile: extended` header.
public struct AFMProfileExtended: Codable, Sendable {
    public let summary: AFMProfile
    public let samples: [AFMProfileSample]
    public init(summary: AFMProfile, samples: [AFMProfileSample]) {
        self.summary = summary
        self.samples = samples
    }
}

public struct ChatCompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [Choice]
    public let usage: Usage
    public let timings: StreamTimings?
    public let systemFingerprint: String?
    public let afmProfile: AFMProfile?
    public let afmProfileExtended: AFMProfileExtended?

    public enum CodingKeys: String, CodingKey {
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

    /// Custom encoder: omit afmProfile/afmProfileExtended when nil (no null pollution).
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(object, forKey: .object)
        try container.encode(created, forKey: .created)
        try container.encode(model, forKey: .model)
        try container.encode(choices, forKey: .choices)
        try container.encode(usage, forKey: .usage)
        try container.encodeIfPresent(timings, forKey: .timings)
        try container.encodeIfPresent(systemFingerprint, forKey: .systemFingerprint)
        try container.encodeIfPresent(afmProfile, forKey: .afmProfile)
        try container.encodeIfPresent(afmProfileExtended, forKey: .afmProfileExtended)
    }

    private static func fingerprint(for model: String) -> String {
        if model == "foundation" {
            return "afm_apple_foundation"
        } else {
            let sanitized = model.replacingOccurrences(of: "/", with: "__").replacingOccurrences(of: " ", with: "_")
            return "afm_mlx__\(sanitized)"
        }
    }

    public init(id: String = UUID().uuidString, model: String, content: String, reasoningContent: String? = nil, logprobs: ChoiceLogprobs? = nil, finishReason: String = "stop", promptTokens: Int = 0, completionTokens: Int = 0, cachedTokens: Int? = nil, completionTime: Double? = nil, promptTime: Double? = nil, timings: StreamTimings? = nil, afmProfile: AFMProfile? = nil, afmProfileExtended: AFMProfileExtended? = nil) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.choices = [
            Choice(
                index: 0,
                message: ResponseMessage(role: "assistant", content: content, reasoningContent: reasoningContent),
                logprobs: logprobs,
                finishReason: finishReason
            )
        ]
        self.usage = Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens,
            cachedTokens: cachedTokens,
            completionTime: completionTime,
            promptTime: promptTime
        )
        self.timings = timings
        self.systemFingerprint = Self.fingerprint(for: model)
        self.afmProfile = afmProfile
        self.afmProfileExtended = afmProfileExtended
    }

    public init(id: String = UUID().uuidString, model: String, toolCalls: [ResponseToolCall], logprobs: ChoiceLogprobs? = nil, promptTokens: Int = 0, completionTokens: Int = 0, cachedTokens: Int? = nil, completionTime: Double? = nil, promptTime: Double? = nil, timings: StreamTimings? = nil, afmProfile: AFMProfile? = nil, afmProfileExtended: AFMProfileExtended? = nil) {
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
            totalTokens: promptTokens + completionTokens,
            cachedTokens: cachedTokens,
            completionTime: completionTime,
            promptTime: promptTime
        )
        self.timings = timings
        self.systemFingerprint = Self.fingerprint(for: model)
        self.afmProfile = afmProfile
        self.afmProfileExtended = afmProfileExtended
    }
}

public struct Choice: Codable, Sendable {
    public let index: Int
    public let message: ResponseMessage
    public let logprobs: ChoiceLogprobs?
    public let finishReason: String

    public enum CodingKeys: String, CodingKey {
        case index
        case message
        case logprobs
        case finishReason = "finish_reason"
    }
    public init(index: Int, message: ResponseMessage, logprobs: ChoiceLogprobs?, finishReason: String) {
        self.index = index
        self.message = message
        self.logprobs = logprobs
        self.finishReason = finishReason
    }
}

public struct ChoiceLogprobs: Codable, Sendable {
    public let content: [TokenLogprobContent]?
    public init(content: [TokenLogprobContent]?) {
        self.content = content
    }
}

public struct TokenLogprobContent: Codable, Sendable {
    public let token: String
    public let logprob: Double
    public let bytes: [Int]?
    public let topLogprobs: [TopLogprobEntry]

    public enum CodingKeys: String, CodingKey {
        case token
        case logprob
        case bytes
        case topLogprobs = "top_logprobs"
    }
    public init(token: String, logprob: Double, bytes: [Int]?, topLogprobs: [TopLogprobEntry]) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.topLogprobs = topLogprobs
    }
}

public struct TopLogprobEntry: Codable, Sendable {
    public let token: String
    public let logprob: Double
    public let bytes: [Int]?
    public init(token: String, logprob: Double, bytes: [Int]?) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
    }
}

public struct ResponseMessage: Codable, Sendable {
    public let role: String
    public let content: String?
    public let reasoningContent: String?
    public let toolCalls: [ResponseToolCall]?

    public enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    public init(role: String, content: String?, reasoningContent: String? = nil, toolCalls: [ResponseToolCall]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }

    /// Always encode `content` (as null when nil) — Vercel AI SDK expects the key to be present.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encodeIfPresent(reasoningContent, forKey: .reasoningContent)
        try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
    }
}

// MARK: - Tool call response types

public struct ResponseToolCall: Codable, Sendable {
    public let index: Int?
    public let id: String           // "call_<random>"
    public let type: String         // "function"
    public let function: ResponseToolCallFunction
    public init(index: Int?, id: String, type: String, function: ResponseToolCallFunction) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct ResponseToolCallFunction: Codable, Sendable {
    public let name: String
    public let arguments: String    // JSON string
    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }
}

public struct PromptTokensDetails: Codable, Sendable {
    public let cachedTokens: Int

    public enum CodingKeys: String, CodingKey {
        case cachedTokens = "cached_tokens"
    }
    public init(cachedTokens: Int) {
        self.cachedTokens = cachedTokens
    }
}

public struct Usage: Codable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails?
    public let completionTime: Double?
    public let promptTime: Double?
    public let totalTime: Double?
    public let completionTokensPerSecond: Double?
    public let promptTokensPerSecond: Double?
    public let peakMemoryGib: Double?

    public enum CodingKeys: String, CodingKey {
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

    public init(promptTokens: Int, completionTokens: Int, totalTokens: Int, cachedTokens: Int? = nil, completionTime: Double? = nil, promptTime: Double? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = cachedTokens.map { PromptTokensDetails(cachedTokens: $0) }

        if let ct = completionTime {
            self.completionTime = (ct * 100).rounded() / 100
        } else {
            self.completionTime = nil
        }
        if let pt = promptTime {
            self.promptTime = (pt * 100).rounded() / 100
        } else {
            self.promptTime = nil
        }
        if let ct = completionTime {
            let pt = promptTime ?? 0
            self.totalTime = (((pt + ct) * 100).rounded()) / 100
        } else {
            self.totalTime = nil
        }

        if let ct = completionTime, ct > 0 {
            self.completionTokensPerSecond = (Double(completionTokens) / ct * 100).rounded() / 100
        } else {
            self.completionTokensPerSecond = nil
        }
        if let pt = promptTime, pt > 0 {
            self.promptTokensPerSecond = (Double(promptTokens) / pt * 100).rounded() / 100
        } else {
            self.promptTokensPerSecond = nil
        }

        self.peakMemoryGib = Self.safePeakMemoryGib()
    }

    private static func safePeakMemoryGib() -> Double? {
        do {
            try MLXMetalLibrary.ensureAvailable(verbose: false)
        } catch {
            return nil
        }

        let gib = 1024.0 * 1024.0 * 1024.0
        return (Double(Memory.snapshot().peakMemory) / gib * 10).rounded() / 10
    }
}

public struct ChatCompletionStreamResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let systemFingerprint: String?
    public let choices: [StreamChoice]
    public let usage: StreamUsage?
    public let timings: StreamTimings?

    public enum CodingKeys: String, CodingKey {
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

    public init(id: String = UUID().uuidString, model: String, content: String, reasoningContent: String? = nil, logprobs: ChoiceLogprobs? = nil, isFinished: Bool = false, isFirst: Bool = false, finishReason: String? = nil, usage: StreamUsage? = nil, timings: StreamTimings? = nil) {
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

    /// Init for the final stream usage summary chunk.
    /// OpenAI-compatible usage chunks carry top-level usage with an empty
    /// choices array so downstream parsers can distinguish them from content
    /// deltas and terminal finish_reason chunks.
    public init(id: String, model: String, usage: StreamUsage, timings: StreamTimings? = nil) {
        self.id = "chatcmpl-\(id.prefix(8))"
        self.object = "chat.completion.chunk"
        self.created = Int(Date().timeIntervalSince1970)
        self.model = model
        self.usage = usage
        self.timings = timings
        self.systemFingerprint = Self.fingerprint(for: model)
        self.choices = []
    }

    /// Init for streaming tool call deltas
    public init(id: String, model: String, toolCalls: [StreamDeltaToolCall], finishReason: String? = nil) {
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

public struct StreamTimings: Codable, Sendable {
    public let prompt_n: Int
    public let prompt_ms: Double
    public let predicted_n: Int
    public let predicted_ms: Double
    public init(prompt_n: Int, prompt_ms: Double, predicted_n: Int, predicted_ms: Double) {
        self.prompt_n = prompt_n
        self.prompt_ms = prompt_ms
        self.predicted_n = predicted_n
        self.predicted_ms = predicted_ms
    }
}

public struct StreamChoice: Codable, Sendable {
    public let index: Int
    public let delta: StreamDelta
    public let logprobs: ChoiceLogprobs?
    public let finishReason: String?

    public enum CodingKeys: String, CodingKey {
        case index
        case delta
        case logprobs
        case finishReason = "finish_reason"
    }
    public init(index: Int, delta: StreamDelta, logprobs: ChoiceLogprobs?, finishReason: String?) {
        self.index = index
        self.delta = delta
        self.logprobs = logprobs
        self.finishReason = finishReason
    }
}

public struct StreamDelta: Codable, Sendable {
    public let role: String?
    public let content: String?
    public let reasoningContent: String?
    public let toolCalls: [StreamDeltaToolCall]?

    public enum CodingKeys: String, CodingKey {
        case role
        case content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    public init(role: String? = nil, content: String?, reasoningContent: String? = nil, toolCalls: [StreamDeltaToolCall]? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }

    public init(toolCalls: [StreamDeltaToolCall]) {
        self.role = nil
        self.content = nil
        self.reasoningContent = nil
        self.toolCalls = toolCalls
    }
}

public struct StreamDeltaToolCall: Codable, Sendable {
    public let index: Int
    public let id: String?
    public let type: String?
    public let function: StreamDeltaFunction?
    public init(index: Int, id: String?, type: String?, function: StreamDeltaFunction?) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }
}

public struct StreamDeltaFunction: Codable, Sendable {
    public let name: String?
    public let arguments: String?
    public init(name: String?, arguments: String?) {
        self.name = name
        self.arguments = arguments
    }
}

public struct StreamUsage: Codable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails?
    public let completionTime: Double?
    public let promptTime: Double?
    public let totalTime: Double?
    public let promptTokensPerSecond: Double?
    public let completionTokensPerSecond: Double?

    public enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTime = "completion_time"
        case promptTime = "prompt_time"
        case totalTime = "total_time"
        case promptTokensPerSecond = "prompt_tokens_per_second"
        case completionTokensPerSecond = "completion_tokens_per_second"
    }

    public init(promptTokens: Int, completionTokens: Int, completionTime: Double, promptTime: Double = 0.0, cachedTokens: Int? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = promptTokens + completionTokens
        self.promptTokensDetails = cachedTokens.map { PromptTokensDetails(cachedTokens: $0) }
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

public struct OpenAIError: Error, Codable, Sendable {
    public let error: ErrorDetail

    public struct ErrorDetail: Codable, Sendable {
        public let message: String
        public let type: String
        public let code: String?
        /// Correlates with the `X-Request-ID`/`OpenAI-Request-ID` response header. (T1.1)
        public let requestId: String?

        public enum CodingKeys: String, CodingKey {
            case message, type, code
            case requestId = "request_id"
        }
        public init(message: String, type: String, code: String?, requestId: String?) {
            self.message = message
            self.type = type
            self.code = code
            self.requestId = requestId
        }
    }

    public init(message: String, type: String = "invalid_request_error", code: String? = nil, requestId: String? = nil) {
        self.error = ErrorDetail(message: message, type: type, code: code, requestId: requestId)
    }
}

// MARK: - Batch API Response Types

/// OpenAI File object for /v1/files endpoints.
public struct FileObject: Codable, Sendable {
    public let id: String
    public let object: String
    public let bytes: Int
    public let createdAt: Int
    public let filename: String
    public let purpose: String

    public enum CodingKeys: String, CodingKey {
        case id, object, bytes
        case createdAt = "created_at"
        case filename, purpose
    }

    public init(id: String, bytes: Int, createdAt: Int, filename: String, purpose: String) {
        self.id = id
        self.object = "file"
        self.bytes = bytes
        self.createdAt = createdAt
        self.filename = filename
        self.purpose = purpose
    }
}

/// Response for DELETE /v1/files/{id}.
public struct FileDeleteResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let deleted: Bool

    public init(id: String) {
        self.id = id
        self.object = "file"
        self.deleted = true
    }
}

/// Batch request counts.
public struct BatchRequestCounts: Codable, Sendable {
    public var total: Int
    public var completed: Int
    public var failed: Int
    public init(total: Int, completed: Int, failed: Int) {
        self.total = total
        self.completed = completed
        self.failed = failed
    }
}

/// OpenAI Batch object for /v1/batches endpoints.
public struct BatchObject: Codable, Sendable {
    public let id: String
    public let object: String
    public let endpoint: String
    public let inputFileId: String
    public let completionWindow: String
    public var status: String
    public let createdAt: Int
    public var completedAt: Int?
    public var outputFileId: String?
    public var requestCounts: BatchRequestCounts

    public enum CodingKeys: String, CodingKey {
        case id, object, endpoint, status
        case inputFileId = "input_file_id"
        case completionWindow = "completion_window"
        case createdAt = "created_at"
        case completedAt = "completed_at"
        case outputFileId = "output_file_id"
        case requestCounts = "request_counts"
    }
    public init(id: String, object: String, endpoint: String, inputFileId: String, completionWindow: String, status: String, createdAt: Int, completedAt: Int?, outputFileId: String?, requestCounts: BatchRequestCounts) {
        self.id = id
        self.object = object
        self.endpoint = endpoint
        self.inputFileId = inputFileId
        self.completionWindow = completionWindow
        self.status = status
        self.createdAt = createdAt
        self.completedAt = completedAt
        self.outputFileId = outputFileId
        self.requestCounts = requestCounts
    }
}

/// A single result line in the output JSONL file.
public struct BatchResultLine: Codable, Sendable {
    public let id: String
    public let customId: String
    public let response: BatchResultResponse?
    public let error: BatchError?

    public enum CodingKeys: String, CodingKey {
        case id
        case customId = "custom_id"
        case response, error
    }
    public init(id: String, customId: String, response: BatchResultResponse?, error: BatchError?) {
        self.id = id
        self.customId = customId
        self.response = response
        self.error = error
    }
}

/// The response wrapper inside a batch result line.
public struct BatchResultResponse: Codable, Sendable {
    public let statusCode: Int
    public let requestId: String
    public let body: ChatCompletionResponse

    public enum CodingKeys: String, CodingKey {
        case statusCode = "status_code"
        case requestId = "request_id"
        case body
    }
    public init(statusCode: Int, requestId: String, body: ChatCompletionResponse) {
        self.statusCode = statusCode
        self.requestId = requestId
        self.body = body
    }
}

/// Error object for batch results and SSE error events.
public struct BatchError: Codable, Sendable {
    public let message: String
    public let type: String
    public init(message: String, type: String) {
        self.message = message
        self.type = type
    }
}

/// Wrapper for OpenAI list responses (GET /v1/batches).
public struct BatchListResponse: Codable, Sendable {
    public let object: String
    public let data: [BatchObject]
    public let hasMore: Bool

    public enum CodingKeys: String, CodingKey {
        case object, data
        case hasMore = "has_more"
    }

    public init(data: [BatchObject]) {
        self.object = "list"
        self.data = data
        self.hasMore = false
    }
}

// MARK: - Embeddings API Response Types

public enum EmbeddingVectorPayload: Codable, Sendable {
    case float([Float])
    case base64(String)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let floatVector = try? container.decode([Float].self) {
            self = .float(floatVector)
        } else if let base64 = try? container.decode(String.self) {
            self = .base64(base64)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Embedding payload must be a float array or base64 string"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .float(let vector):
            try container.encode(vector)
        case .base64(let string):
            try container.encode(string)
        }
    }
}

public struct EmbeddingDataItem: Codable, Sendable {
    public let object: String
    public let index: Int
    public let embedding: EmbeddingVectorPayload

    public init(index: Int, embedding: EmbeddingVectorPayload) {
        self.object = "embedding"
        self.index = index
        self.embedding = embedding
    }
}

/// Usage accounting for the /v1/embeddings endpoint.
///
/// Scoped separately from the chat `Usage` struct so the Apple-only embeddings
/// path does not pull in `MLXMetalLibrary.ensureAvailable()`, which mutates the
/// process working directory when it initializes the MLX Metal library.
public struct EmbeddingsUsage: Codable, Sendable {
    public let promptTokens: Int
    public let totalTokens: Int

    public enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }
    public init(promptTokens: Int, totalTokens: Int) {
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
    }
}

public struct EmbeddingsResponse: Codable, Sendable {
    public let object: String
    public let data: [EmbeddingDataItem]
    public let model: String
    public let usage: EmbeddingsUsage

    public init(model: String, data: [EmbeddingDataItem], promptTokens: Int) {
        self.object = "list"
        self.data = data
        self.model = model
        self.usage = EmbeddingsUsage(promptTokens: promptTokens, totalTokens: promptTokens)
    }
}
