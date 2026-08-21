import Foundation

public struct CompletionChoice: Codable, Sendable {
    public let text: String
    public let index: Int
    public let logprobs: AnyCodable?
    public let finishReason: String?

    public enum CodingKeys: String, CodingKey {
        case text, index, logprobs
        case finishReason = "finish_reason"
    }

    public init(
        text: String,
        index: Int = 0,
        logprobs: AnyCodable? = nil,
        finishReason: String? = nil
    ) {
        self.text = text
        self.index = index
        self.logprobs = logprobs
        self.finishReason = finishReason
    }
}

/// Shared shape for legacy completion responses and SSE chunks.
public struct CompletionResponse: Codable, Sendable {
    public let id: String
    public let object: String
    public let created: Int
    public let model: String
    public let choices: [CompletionChoice]
    public let usage: Usage?

    public init(
        id: String,
        created: Int,
        model: String,
        choices: [CompletionChoice],
        usage: Usage? = nil
    ) {
        self.id = id
        self.object = "text_completion"
        self.created = created
        self.model = model
        self.choices = choices
        self.usage = usage
    }
}
