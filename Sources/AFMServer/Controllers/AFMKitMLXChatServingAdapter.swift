import Foundation
import AFMKit
import AFMKitMLX

/// Bridges the existing OpenAI-compatible HTTP controllers onto AFMKit while
/// preserving their current `AFMMLXOpenAIChatServing` contract.
final class AFMKitMLXChatServingAdapter: AFMMLXOpenAIChatServing, AFMTextTokenizing, @unchecked Sendable {
    private let service: MLXModelService?
    private let resolver: MLXCacheResolver
    private let fixedModel: AnyAFMModel?
    private let fixedModelID: String?
    private let slotLock = NSLock()
    private let fixedMaxConcurrent: Int
    private var fixedSlotsReserved = 0

    init(service: MLXModelService, resolver: MLXCacheResolver = .init()) {
        self.service = service
        self.resolver = resolver
        fixedModel = nil
        fixedModelID = nil
        fixedMaxConcurrent = 1
    }

    init(model: AnyAFMModel, modelID: String) {
        service = nil
        resolver = .init()
        fixedModel = model
        fixedModelID = modelID
        if case .integer(let value) = model.descriptor.metadata["maxConcurrent"] {
            fixedMaxConcurrent = max(1, value)
        } else {
            fixedMaxConcurrent = 1
        }
    }

    var maxConcurrent: Int { service?.maxConcurrent ?? fixedMaxConcurrent }
    var servingConfiguration: AFMMLXServingConfiguration {
        service?.servingConfiguration ?? .init()
    }
    var defaultGuidedJsonSchema: ResponseFormat? { service?.defaultGuidedJsonSchema }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        service?.effectiveResponseFormat(requestFormat: requestFormat) ?? requestFormat
    }

    func normalizeModel(_ raw: String) -> String {
        service?.normalizeModel(raw) ?? fixedModelID ?? raw
    }

    func resolvedToolCallParser(logBypass: Bool) -> String? {
        service?.resolvedToolCallParser(logBypass: logBypass)
    }

    func tryReserveSlot() -> Bool {
        if let service { return service.tryReserveSlot() }
        slotLock.lock()
        defer { slotLock.unlock() }
        guard fixedSlotsReserved < fixedMaxConcurrent else { return false }
        fixedSlotsReserved += 1
        return true
    }

    func waitForSlot(timeout: TimeInterval) async -> Bool {
        if let service { return await service.waitForSlot(timeout: timeout) }
        if timeout <= 0 { return tryReserveSlot() }
        let deadline = ContinuousClock.now + .seconds(timeout)
        while ContinuousClock.now < deadline {
            if Task.isCancelled { return false }
            if tryReserveSlot() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return false
    }

    func releaseSlot() {
        if let service {
            service.releaseSlot()
            return
        }
        slotLock.lock()
        fixedSlotsReserved = max(0, fixedSlotsReserved - 1)
        slotLock.unlock()
    }

    func tokenize(text: String) async throws -> [Int] {
        do {
            guard let service else {
                throw AFMError.unsupportedCapability("tokenization for this provider")
            }
            return try await service.tokenize(text: text)
        } catch MLXServiceError.noModelLoaded {
            throw AFMError.unsupportedCapability("tokenization without a loaded model")
        }
    }

    func ensureBatchMode(concurrency: Int) async throws {
        if let service {
            try await service.ensureBatchMode(concurrency: concurrency)
        } else if concurrency > fixedMaxConcurrent {
            throw AFMError.unsupportedCapability("concurrent batch generation for this provider")
        }
    }

    func releaseBatchReference() {
        service?.releaseBatchReference()
    }

    func cancelBatchSlots(ids: Set<UUID>) async {
        if let service { await service.cancelBatchSlots(ids: ids) }
    }

    func startAPIProfile() {
        service?.startAPIProfile()
    }

    func stopAPIProfile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile {
        if let service {
            return service.stopAPIProfile(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                promptTime: promptTime,
                generateTime: generateTime)
        }
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
        if let service {
            return service.stopAPIProfileExtended(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                promptTime: promptTime,
                generateTime: generateTime)
        }
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
    ) async throws -> AFMMLXChatGenerationResult {
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
    ) async throws -> AFMMLXChatGenerationResult {
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
    ) async throws -> AFMMLXChatStreamingResult {
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
    ) async throws -> AFMMLXChatStreamingResult {
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

        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
            let task = Task {
                defer {
                    if self.service == nil {
                        self.releaseSlot()
                    }
                }
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
                                continuation.yield(StreamChunk(text: endTag ?? ""))
                                insideReasoning = false
                            }
                            continuation.yield(StreamChunk(text: text))
                        case .reasoningText(_, let text, _):
                            if !insideReasoning {
                                continuation.yield(StreamChunk(text: startTag ?? ""))
                                insideReasoning = true
                            }
                            continuation.yield(StreamChunk(text: text))
                        case .tokenLogprobs(let values):
                            continuation.yield(
                                StreamChunk(
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
                                StreamChunk(
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
                                continuation.yield(StreamChunk(text: endTag ?? ""))
                                insideReasoning = false
                            }
                            continuation.yield(
                                StreamChunk(
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
        if let fixedModel { return fixedModel }
        return AnyAFMModel(
            AFMMLXModel(
                modelID: AFMModelID(rawValue: model),
                resolver: resolver,
                attachedService: service!))
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
        if let chatTemplateKwargs {
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
        continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    ) {
        switch stage {
        case .started:
            continuation.yield(
                StreamChunk(
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
                StreamChunk(
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
                StreamChunk(
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
    var resolvedLogprob: ResolvedLogprob {
        ResolvedLogprob(
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
