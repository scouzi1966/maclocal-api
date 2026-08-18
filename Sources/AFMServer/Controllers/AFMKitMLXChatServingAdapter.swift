import Foundation
import AFMKit
import AFMKitMLX

/// Bridges the OpenAI-compatible HTTP controllers onto AFMKit's neutral model
/// and event contracts. Provider scheduler and parser internals stay inside
/// AFMKitMLX.
final class AFMKitMLXChatServingAdapter: AFMChatServing, AFMTextTokenizing, @unchecked Sendable {
    private let fixedModel: AnyAFMModel
    private let fixedModelID: String
    private let fixedServingConfiguration: AFMChatServingConfiguration
    private let defaultChatTemplateKwargs: [String: AnyCodable]?
    private let forceDisableThinking: Bool
    private let fixedMaxConcurrent: Int

    init(
        model: AFMMLXModel,
        defaultChatTemplateKwargs: [String: AnyCodable]? = nil,
        forceDisableThinking: Bool = false
    ) {
        fixedModel = AnyAFMModel(model)
        fixedModelID = model.descriptor.modelID.rawValue
        fixedServingConfiguration = Self.configuration(for: model.descriptor)
        self.defaultChatTemplateKwargs = defaultChatTemplateKwargs
        self.forceDisableThinking = forceDisableThinking
        fixedMaxConcurrent = Self.maximumConcurrency(for: model.descriptor)
    }

    init(
        model: AnyAFMModel,
        modelID: String,
        defaultChatTemplateKwargs: [String: AnyCodable]? = nil,
        forceDisableThinking: Bool = false
    ) {
        fixedModel = model
        fixedModelID = modelID
        fixedServingConfiguration = Self.configuration(for: model.descriptor)
        self.defaultChatTemplateKwargs = defaultChatTemplateKwargs
        self.forceDisableThinking = forceDisableThinking
        fixedMaxConcurrent = Self.maximumConcurrency(for: model.descriptor)
    }

    var maxConcurrent: Int { fixedMaxConcurrent }
    var servingConfiguration: AFMChatServingConfiguration { fixedServingConfiguration }
    var defaultGuidedJsonSchema: ResponseFormat? { nil }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        requestFormat
    }

    func normalizeModel(_ raw: String) -> String {
        fixedModelID
    }

    func resolvedToolCallParser(logBypass: Bool) -> String? {
        nil
    }

    /// Admission is provider-owned. The HTTP layer no longer reserves MLX
    /// scheduler slots independently before asking the model to execute.
    func tryReserveSlot() -> Bool {
        return true
    }

    func waitForSlot(timeout: TimeInterval) async -> Bool {
        !Task.isCancelled
    }

    func releaseSlot() {}

    func tokenize(text: String) async throws -> [Int] {
        try await fixedModel.tokenize(text: text)
    }

    func ensureBatchMode(concurrency: Int) async throws {}

    func releaseBatchReference() {}

    func cancelBatchSlots(ids: Set<UUID>) async {}

    func startAPIProfile() {}

    func stopAPIProfile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile {
        return Self.profile(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            promptTime: promptTime,
            generateTime: generateTime)
    }

    func stopAPIProfileExtended(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfileExtended {
        return AFMProfileExtended(
            summary: Self.profile(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                promptTime: promptTime,
                generateTime: generateTime),
            samples: [])
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMChatGenerationResult {
        try await generate(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            toolChoice: nil,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs
        )
    }

    func generate(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> AFMChatGenerationResult {
        let request = try AFMRequest(
            openAIMessages: messages,
            generationConfig: generationConfig(
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                topK: topK,
                minP: minP,
                presencePenalty: presencePenalty,
                seed: seed,
                logprobs: logprobs,
                topLogprobs: topLogprobs,
                tools: tools,
                toolChoice: toolChoice,
                parallelToolCalls: parallelToolCalls,
                stop: stop,
                responseFormat: responseFormat,
                chatTemplateKwargs: chatTemplateKwargs
            )
        )
        let response = try await afmModel(for: model).respond(to: request)
        let metadata = response.metadata
        let promptTime = metadata.double("promptTime") ?? 0
        let generateTime = metadata.double("generateTime") ?? 0
        let stoppedBySequence = metadata.bool("stoppedBySequence") ?? false
        let responseModelID = metadata.string("modelID") ?? normalizeModel(model)
        return (
            modelID: responseModelID,
            content: rawContent(
                text: response.text,
                reasoning: response.reasoning,
                startTag: thinkStartTag,
                endTag: thinkEndTag
            ),
            promptTokens: response.usage.inputTokens,
            completionTokens: response.usage.outputTokens,
            tokenLogprobs: response.tokenLogprobs?.map(\.resolvedLogprob),
            toolCalls: response.toolCalls.isEmpty ? nil : response.toolCalls.enumerated().map {
                $0.element.responseToolCall(index: $0.offset)
            },
            cachedTokens: response.usage.cachedInputTokens,
            promptTime: promptTime,
            generateTime: generateTime,
            stoppedBySequence: stoppedBySequence
        )
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMChatStreamingResult {
        try await generateStreaming(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            tools: tools,
            toolChoice: nil,
            parallelToolCalls: parallelToolCalls,
            stop: stop,
            responseFormat: responseFormat,
            chatTemplateKwargs: chatTemplateKwargs,
            preserveStructuralTags: preserveStructuralTags,
            requestId: requestId
        )
    }

    func generateStreaming(
        model: String,
        messages: [Message],
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?,
        preserveStructuralTags: Bool,
        requestId: String?
    ) async throws -> AFMChatStreamingResult {
        let request = try AFMRequest(
            openAIMessages: messages,
            generationConfig: generationConfig(
                temperature: temperature,
                maxTokens: maxTokens,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                topK: topK,
                minP: minP,
                presencePenalty: presencePenalty,
                seed: seed,
                logprobs: logprobs,
                topLogprobs: topLogprobs,
                tools: tools,
                toolChoice: toolChoice,
                parallelToolCalls: parallelToolCalls,
                stop: stop,
                responseFormat: responseFormat,
                chatTemplateKwargs: chatTemplateKwargs
            )
        )
        let modelID = normalizeModel(model)
        let eventStream = afmModel(for: model).streamResponse(to: request)
        let startTag = thinkStartTag
        let endTag = thinkEndTag

        let stream = AsyncThrowingStream<AFMServerStreamChunk, Error> { continuation in
            let task = Task {
                var promptTokens = 0
                var completionTokens = 0
                var cachedTokens = 0
                var promptTime: Double = 0
                var generateTime: Double = 0
                var stoppedBySequence = false
                var insideReasoning = false
                var toolIndices = [String: Int]()

                do {
                    for try await event in eventStream {
                        switch event {
                        case .responseText(_, let text, _):
                            if insideReasoning {
                                continuation.yield(AFMServerStreamChunk(text: endTag ?? ""))
                                insideReasoning = false
                            }
                            continuation.yield(AFMServerStreamChunk(text: text))
                        case .reasoningText(_, let text, _):
                            if !insideReasoning {
                                continuation.yield(AFMServerStreamChunk(text: startTag ?? ""))
                                insideReasoning = true
                            }
                            continuation.yield(AFMServerStreamChunk(text: text))
                        case .tokenLogprobs(let values):
                            continuation.yield(
                                AFMServerStreamChunk(
                                    text: "",
                                    logprobs: values.map(\.resolvedLogprob)
                                )
                            )
                        case .toolCall(let call, let stage):
                            let index = toolIndices[call.id] ?? toolIndices.count
                            toolIndices[call.id] = index
                            emitToolCall(
                                call,
                                stage: stage,
                                index: index,
                                continuation: continuation
                            )
                        case .usage(let usage):
                            promptTokens = usage.inputTokens
                            completionTokens = usage.outputTokens
                            cachedTokens = usage.cachedInputTokens
                            continuation.yield(
                                AFMServerStreamChunk(
                                    text: "",
                                    promptTokens: promptTokens,
                                    completionTokens: completionTokens,
                                    cachedTokens: cachedTokens
                                )
                            )
                        case .metadata(let metadata):
                            promptTime = metadata.double("promptTime") ?? promptTime
                            generateTime = metadata.double("generateTime") ?? generateTime
                            stoppedBySequence =
                                metadata.bool("stoppedBySequence") ?? stoppedBySequence
                        case .completed:
                            if insideReasoning {
                                continuation.yield(AFMServerStreamChunk(text: endTag ?? ""))
                                insideReasoning = false
                            }
                            continuation.yield(
                                AFMServerStreamChunk(
                                    text: "",
                                    promptTokens: promptTokens,
                                    completionTokens: completionTokens,
                                    cachedTokens: cachedTokens,
                                    promptTime: promptTime,
                                    generateTime: generateTime,
                                    stoppedBySequence: stoppedBySequence
                                )
                            )
                        case .custom:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }

        return (
            modelID: modelID,
            stream: stream,
            promptTokens: 0,
            toolCallStartTag: nil,
            toolCallEndTag: nil,
            thinkStartTag: startTag,
            thinkEndTag: endTag
        )
    }

    private func afmModel(for model: String) -> AnyAFMModel {
        fixedModel
    }

    private static func maximumConcurrency(for descriptor: AFMModelDescriptor) -> Int {
        guard case .integer(let value)? = descriptor.metadata["maxConcurrent"] else {
            return 1
        }
        return max(1, value)
    }

    private static func configuration(
        for descriptor: AFMModelDescriptor
    ) -> AFMChatServingConfiguration {
        let reasoning = descriptor.capabilities.contains(.reasoning)
        return AFMChatServingConfiguration(
            thinkStartTag: reasoning ? "<think>" : nil,
            thinkEndTag: reasoning ? "</think>" : nil
        )
    }

    private static func profile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile {
        AFMProfile(
            gpuPowerAvgW: nil,
            gpuPowerPeakW: nil,
            gpuSamples: nil,
            memoryWeightsGiB: nil,
            memoryKvGiB: nil,
            memoryPeakGiB: nil,
            prefillTokS: promptTime > 0 ? Double(promptTokens) / promptTime : nil,
            decodeTokS: generateTime > 0 ? Double(completionTokens) / generateTime : nil,
            chip: nil,
            theoreticalBwGbs: nil,
            estBandwidthGbs: nil)
    }

    private func generationConfig(
        temperature: Double?,
        maxTokens: Int?,
        topP: Double?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        logprobs: Bool?,
        topLogprobs: Int?,
        tools: [RequestTool]?,
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool?,
        stop: [String]?,
        responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) -> GenerationConfig {
        var metadata = [String: AFMJSONValue]()
        switch toolChoice {
        case .mode(let mode) where mode == "required":
            metadata["toolCallingMode"] = .string("required")
        case .function(let choice):
            metadata["toolCallingMode"] = .string("required")
            metadata["requiredToolName"] = .string(choice.function.name)
        default:
            break
        }
        if let parallelToolCalls {
            metadata["parallelToolCalls"] = .bool(parallelToolCalls)
        }
        if let chatTemplateKwargs = mergedChatTemplateKwargs(request: chatTemplateKwargs) {
            metadata["chatTemplateKwargs"] = .object(
                chatTemplateKwargs.mapValues(\.afmJSONValue)
            )
        }
        return GenerationConfig(
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            seed: seed,
            logprobs: logprobs,
            topLogprobs: topLogprobs,
            stop: stop,
            tools: tools,
            responseFormat: responseFormat,
            metadata: metadata
        )
    }

    private func mergedChatTemplateKwargs(
        request: [String: AnyCodable]?
    ) -> [String: AnyCodable]? {
        var merged = defaultChatTemplateKwargs ?? [:]
        if let request {
            merged.merge(request) { _, requestValue in requestValue }
        }
        if forceDisableThinking {
            merged["enable_thinking"] = AnyCodable(false)
            merged.removeValue(forKey: "reasoning_effort")
        }
        return merged.isEmpty ? nil : merged
    }

    private func rawContent(
        text: String,
        reasoning: String?,
        startTag: String?,
        endTag: String?
    ) -> String {
        guard let reasoning, !reasoning.isEmpty else {
            return text
        }
        return "\(startTag ?? "")\(reasoning)\(endTag ?? "")\(text)"
    }

    private func emitToolCall(
        _ call: AFMToolCall,
        stage: AFMToolCallStage,
        index: Int,
        continuation: AsyncThrowingStream<AFMServerStreamChunk, Error>.Continuation
    ) {
        switch stage {
        case .started:
            continuation.yield(
                AFMServerStreamChunk(
                    text: "",
                    toolCallDeltas: [
                        StreamDeltaToolCall(
                            index: index,
                            id: call.id,
                            type: "function",
                            function: StreamDeltaFunction(
                                name: call.name,
                                arguments: nil
                            )
                        )
                    ]
                )
            )
        case .argumentsDelta(let delta):
            continuation.yield(
                AFMServerStreamChunk(
                    text: "",
                    toolCallDeltas: [
                        StreamDeltaToolCall(
                            index: index,
                            id: nil,
                            type: nil,
                            function: StreamDeltaFunction(
                                name: nil,
                                arguments: delta
                            )
                        )
                    ]
                )
            )
        case .completed:
            continuation.yield(
                AFMServerStreamChunk(
                    text: "",
                    toolCalls: [call.responseToolCall(index: index)]
                )
            )
        case .retracted:
            break
        }
    }
}

private extension AFMToolCall {
    func responseToolCall(index: Int) -> ResponseToolCall {
        ResponseToolCall(
            index: index,
            id: id,
            type: "function",
            function: ResponseToolCallFunction(
                name: name,
                arguments: arguments
            )
        )
    }
}

private extension AFMTokenLogProbability {
    var resolvedLogprob: AFMServerResolvedLogprob {
        AFMServerResolvedLogprob(
            token: token,
            tokenId: tokenID,
            logprob: logprob,
            topTokens: topTokens.map {
                (
                    token: $0.token,
                    tokenId: $0.tokenID,
                    logprob: $0.logprob
                )
            }
        )
    }
}

private extension AnyCodable {
    var afmJSONValue: AFMJSONValue {
        value.afmJSONValue
    }
}

private extension AnyCodableValue {
    var afmJSONValue: AFMJSONValue {
        switch self {
        case .null:
            return .null
        case .bool(let value):
            return .bool(value)
        case .int(let value):
            return .integer(value)
        case .double(let value):
            return .number(value)
        case .string(let value):
            return .string(value)
        case .array(let values):
            return .array(values.map(\.afmJSONValue))
        case .object(let values):
            return .object(values.mapValues(\.afmJSONValue))
        }
    }
}

private extension Dictionary where Key == String, Value == AFMJSONValue {
    func string(_ key: String) -> String? {
        guard case .string(let value)? = self[key] else {
            return nil
        }
        return value
    }

    func bool(_ key: String) -> Bool? {
        guard case .bool(let value)? = self[key] else {
            return nil
        }
        return value
    }

    func double(_ key: String) -> Double? {
        switch self[key] {
        case .number(let value):
            return value
        case .integer(let value):
            return Double(value)
        default:
            return nil
        }
    }
}
