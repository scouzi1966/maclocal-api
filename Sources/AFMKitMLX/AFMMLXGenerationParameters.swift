import Foundation
@preconcurrency import MLXLMCommon

public struct AFMMLXGenerationParameterRequest: Equatable, Sendable {
    public let maxTokens: Int?
    public let maxKVSize: Int?
    public let kvBits: Int?
    public let kvGroupSize: Int
    public let quantizedKVStart: Int
    public let temperature: Double
    public let topP: Double
    public let repetitionPenalty: Double
    public let repetitionContextSize: Int
    public let topK: Int
    public let minP: Double
    public let presencePenalty: Double
    public let prefillStepSize: Int

    public init(
        maxTokens: Int?,
        maxKVSize: Int?,
        kvBits: Int?,
        kvGroupSize: Int,
        quantizedKVStart: Int,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        repetitionContextSize: Int,
        topK: Int,
        minP: Double,
        presencePenalty: Double,
        prefillStepSize: Int
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.prefillStepSize = prefillStepSize
    }
}

public enum AFMMLXGenerationParameterFactory {
    public static func make(
        maxTokens: Int?,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Double,
        topP: Double,
        repetitionPenalty: Double,
        repetitionContextSize: Int = 64,
        topK: Int = 0,
        minP: Double = 0.0,
        presencePenalty: Double = 0.0,
        prefillStepSize: Int
    ) -> MLXLMCommon.GenerateParameters {
        make(
            AFMMLXGenerationParameterRequest(
                maxTokens: maxTokens,
                maxKVSize: maxKVSize,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                quantizedKVStart: quantizedKVStart,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize,
                topK: topK,
                minP: minP,
                presencePenalty: presencePenalty,
                prefillStepSize: prefillStepSize
            )
        )
    }

    public static func make(
        _ request: AFMMLXGenerationParameterRequest
    ) -> MLXLMCommon.GenerateParameters {
        MLXLMCommon.GenerateParameters(
            maxTokens: request.maxTokens,
            maxKVSize: request.maxKVSize,
            kvBits: request.kvBits,
            kvGroupSize: request.kvGroupSize,
            quantizedKVStart: request.quantizedKVStart,
            temperature: Float(request.temperature),
            topP: Float(request.topP),
            repetitionPenalty: effectiveRepetitionPenalty(request.repetitionPenalty),
            repetitionContextSize: request.repetitionContextSize,
            topK: request.topK,
            minP: Float(request.minP),
            presencePenalty: Float(request.presencePenalty),
            prefillStepSize: request.prefillStepSize
        )
    }

    private static func effectiveRepetitionPenalty(_ value: Double) -> Float? {
        value == 1.0 ? nil : Float(value)
    }
}
