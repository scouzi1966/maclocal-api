import Foundation

public struct AFMRawTextGenerationRequest: Hashable, Sendable {
    public var prompt: String
    public var modelID: AFMModelID
    public var maximumOutputTokens: Int?
    public var stopSequences: [String]
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var seed: Int?
    public var ignoreEndOfSequence: Bool

    public init(
        prompt: String,
        modelID: AFMModelID,
        maximumOutputTokens: Int? = nil,
        stopSequences: [String] = [],
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        ignoreEndOfSequence: Bool = false
    ) {
        self.prompt = prompt
        self.modelID = modelID
        self.maximumOutputTokens = maximumOutputTokens
        self.stopSequences = stopSequences
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.ignoreEndOfSequence = ignoreEndOfSequence
    }
}

public struct AFMRawTextGenerationResult: Hashable, Sendable {
    public var finishReason: AFMInferenceFinishReason
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int

    public init(
        finishReason: AFMInferenceFinishReason,
        promptTokens: Int,
        completionTokens: Int,
        totalTokens: Int
    ) {
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
    }
}

public enum AFMRawTextGenerationEvent: Hashable, Sendable {
    case textDelta(text: String, tokenID: Int?, timestamp: Double)
    case completed(AFMRawTextGenerationResult)
    case failed(reason: AFMInferenceFailureReason, message: String)
}

public protocol AFMRawTextGenerating: Sendable {
    func rawTextGenerationEvents(
        for request: AFMRawTextGenerationRequest
    ) -> AsyncStream<AFMRawTextGenerationEvent>
}

public struct AnyAFMRawTextGenerator: AFMRawTextGenerating, Sendable {
    private let operation:
        @Sendable (AFMRawTextGenerationRequest) -> AsyncStream<AFMRawTextGenerationEvent>

    public init(_ generator: any AFMRawTextGenerating) {
        operation = { request in
            generator.rawTextGenerationEvents(for: request)
        }
    }

    public init(
        rawTextGenerationEvents:
            @escaping @Sendable (AFMRawTextGenerationRequest) -> AsyncStream<AFMRawTextGenerationEvent>
    ) {
        operation = rawTextGenerationEvents
    }

    public func rawTextGenerationEvents(
        for request: AFMRawTextGenerationRequest
    ) -> AsyncStream<AFMRawTextGenerationEvent> {
        operation(request)
    }
}
