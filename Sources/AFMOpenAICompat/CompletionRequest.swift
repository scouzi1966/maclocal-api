import Foundation

/// OpenAI's legacy completion endpoint accepts either one prompt or an array.
/// AFM decodes both shapes so the HTTP layer can reject arrays with a stable
/// protocol error before provider admission.
public enum CompletionPrompt: Codable, Sendable {
    case text(String)
    case array

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let prompt = try? value.decode(String.self) {
            self = .text(prompt)
            return
        }
        if (try? value.decode([AnyCodable].self)) != nil {
            self = .array
            return
        }
        throw DecodingError.typeMismatch(
            CompletionPrompt.self,
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "prompt must be a string or array"
            )
        )
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .text(let prompt):
            try value.encode(prompt)
        case .array:
            try value.encode([AnyCodable]())
        }
    }
}

/// A legacy completion stop value accepts either one string or an array.
public enum CompletionStop: Codable, Sendable {
    case text(String)
    case array([String])

    public var sequences: [String] {
        switch self {
        case .text(let value): [value]
        case .array(let values): values
        }
    }

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if let stop = try? value.decode(String.self) {
            self = .text(stop)
        } else {
            self = .array(try value.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .text(let stop): try value.encode(stop)
        case .array(let stops): try value.encode(stops)
        }
    }
}

public struct CompletionRequest: Codable, Sendable {
    public let model: String?
    public let prompt: CompletionPrompt
    public let maxTokens: Int?
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let repetitionPenalty: Double?
    public let presencePenalty: Double?
    public let seed: Int?
    public let stop: CompletionStop?
    public let stream: Bool?
    public let streamOptions: StreamOptions?
    public let ignoreEOS: Bool?
    public let echo: Bool?
    public let logprobs: Int?
    public let n: Int?
    public let bestOf: Int?
    public let user: String?

    public enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, seed, stop, stream, echo, logprobs, n, user
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case streamOptions = "stream_options"
        case ignoreEOS = "ignore_eos"
        case bestOf = "best_of"
    }

    /// Issue #192 accepts this GuideLLM option but emits only final exact usage.
    public var includeStreamingUsage: Bool {
        streamOptions?.includeUsage ?? false
    }

    public init(
        model: String? = nil,
        prompt: CompletionPrompt,
        maxTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        stop: CompletionStop? = nil,
        stream: Bool? = nil,
        streamOptions: StreamOptions? = nil,
        ignoreEOS: Bool? = nil,
        echo: Bool? = nil,
        logprobs: Int? = nil,
        n: Int? = nil,
        bestOf: Int? = nil,
        user: String? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.stop = stop
        self.stream = stream
        self.streamOptions = streamOptions
        self.ignoreEOS = ignoreEOS
        self.echo = echo
        self.logprobs = logprobs
        self.n = n
        self.bestOf = bestOf
        self.user = user
    }
}
