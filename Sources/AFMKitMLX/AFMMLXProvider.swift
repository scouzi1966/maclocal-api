import Foundation
import AFMKitCore
import AFMOpenAICompat

public struct AFMMLXProviderFactory: AFMProviderFactory {
    public static let providerID: AFMProviderID = "mlx"

    private let resolver: MLXCacheResolver

    public init(resolver: MLXCacheResolver = .init()) {
        self.resolver = resolver
    }

    public var descriptor: AFMProviderDescriptor {
        AFMProviderDescriptor(
            id: Self.providerID,
            displayName: "MLX",
            privacyBoundary: .device,
            configurationKeys: [
                "kvBits",
                "enablePrefixCaching",
                "mtpEnabled",
                "mtpDepth",
                "mtpModelID",
                "eagle3DrafterPath",
                "maxConcurrent",
                "toolCallParser",
                "enableGrammarConstraints",
                "prefillStepSize",
                "kvEvictionPolicy",
                "fixToolArguments",
                "forceVLM",
                "cacheProfilePath",
                "trace",
                "gpuCapturePath",
                "gpuTraceDuration",
                "gpuProfile",
                "gpuProfileBandwidth"
            ],
            metadata: ["runtime": .string("mlx-swift")]
        )
    }

    public func modelDescriptors() async throws -> [AFMModelDescriptor] {
        let service = MLXModelService(resolver: resolver)
        return try service.revalidateRegistry().map {
            AFMMLXModelDescriptor.describe(modelID: $0, resolver: resolver)
        }
    }

    public func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel {
        AnyAFMModel(
            AFMMLXModel(
                modelID: id,
                configuration: configuration,
                resolver: resolver
            )
        )
    }
}

public final class AFMMLXModel: AFMModel, AFMTextTokenizing, @unchecked Sendable {
    public let descriptor: AFMModelDescriptor

    private let runtime: AFMMLXRuntime
    private let service: MLXModelService
    private let modelID: String
    private let schedulerAdmissionOwnership: AFMMLXSchedulerAdmissionOwnership
    private let streamingLoadOverride: (@Sendable () async throws -> Void)?

    public init(
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration = .init(),
        resolver: MLXCacheResolver = .init(),
        service providedService: MLXModelService? = nil
    ) {
        let runtime = AFMMLXRuntime(
            modelID: modelID.rawValue,
            providerConfiguration: configuration,
            resolver: resolver,
            service: providedService
        )

        self.runtime = runtime
        self.service = runtime.service
        self.modelID = runtime.modelID
        self.schedulerAdmissionOwnership = .model
        self.streamingLoadOverride = nil
        self.descriptor = runtime.descriptor
    }

    /// Wrap a host-owned service without mutating its established runtime settings.
    public init(
        modelID: AFMModelID,
        resolver: MLXCacheResolver = .init(),
        attachedService service: MLXModelService,
        schedulerAdmissionOwnership: AFMMLXSchedulerAdmissionOwnership = .model
    ) {
        let runtime = AFMMLXRuntime(
            modelID: modelID.rawValue,
            attaching: service,
            resolver: resolver
        )

        self.runtime = runtime
        self.service = service
        self.modelID = runtime.modelID
        self.schedulerAdmissionOwnership = schedulerAdmissionOwnership
        self.streamingLoadOverride = nil
        self.descriptor = runtime.descriptor
    }

    init(
        modelID: AFMModelID,
        resolver: MLXCacheResolver = .init(),
        attachedService service: MLXModelService,
        schedulerAdmissionOwnership: AFMMLXSchedulerAdmissionOwnership,
        testingStreamingLoad: @escaping @Sendable () async throws -> Void
    ) {
        let runtime = AFMMLXRuntime(
            modelID: modelID.rawValue,
            attaching: service,
            resolver: resolver
        )

        self.runtime = runtime
        self.service = service
        self.modelID = runtime.modelID
        self.schedulerAdmissionOwnership = schedulerAdmissionOwnership
        self.streamingLoadOverride = testingStreamingLoad
        self.descriptor = runtime.descriptor
    }

    public func availability() async -> AFMModelAvailability {
        .available
    }

    public func load(
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> AFMModelDescriptor {
        do {
            return try await runtime.load(
                progress: { progress?($0.fractionCompleted) }
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw AFMError.loadingFailed(error.localizedDescription)
        }
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        let callerAdmission: AFMMLXSchedulerAdmission? = switch schedulerAdmissionOwnership {
        case .model:
            nil
        case .caller(let admission):
            admission
        }
        guard callerAdmission?.isAdmitted != false else {
            throw AFMError.unavailable("MLX scheduler is at capacity")
        }
        var callerReservation = callerAdmission?.reservation
        defer {
            if let callerReservation { service.releaseSlot(callerReservation) }
        }
        _ = try await load(progress: nil)
        do {
            if AFMMLXGenerationRoute.resolve(maxConcurrent: service.maxConcurrent)
                == .schedulerStream {
                var accumulator = AFMMLXResponseAccumulator(modelID: modelID)
                let stream = streamResponse(to: request)
                callerReservation = nil
                for try await event in stream {
                    accumulator.consume(event)
                }
                return accumulator.response
            }
            let tools = request.effectiveOpenAITools()
            let result = try await service.generateWithTelemetry(
                model: modelID,
                messages: try request.openAIMessages(),
                temperature: request.options.temperature,
                maxTokens: request.options.maximumResponseTokens,
                topP: request.options.topP,
                repetitionPenalty: request.options.repetitionPenalty,
                topK: request.options.topK,
                minP: request.options.minP,
                presencePenalty: request.options.presencePenalty,
                seed: request.options.seed,
                logprobs: request.options.logprobs,
                topLogprobs: request.options.topLogprobs,
                tools: tools,
                parallelToolCalls: request.parallelToolCalls,
                stop: request.options.stopSequences,
                responseFormat: request.openAIResponseFormat(),
                chatTemplateKwargs: request.chatTemplateKwargs(),
                speculativeDecoding: request.speculativeDecodingOptions()
            )
            let normalized = Self.normalizedGeneratedResponse(
                result.content,
                startTag: service.thinkStartTag,
                endTag: service.thinkEndTag
            )
            let toolCalls = (result.toolCalls ?? []).map {
                AFMToolCall(
                    id: $0.id,
                    name: Self.sanitizedToolName($0.function.name),
                    arguments: $0.function.arguments
                )
            }
            try AFMMLXToolPolicy.validateCompletedToolCalls(
                toolCalls,
                for: request
            )
            let finishReason = Self.finishReason(
                toolCalls: result.toolCalls,
                stoppedBySequence: result.stoppedBySequence,
                completionTokens: result.completionTokens,
                maximumResponseTokens: request.options.maximumResponseTokens
            )
            var metadata: [String: AFMJSONValue] = [
                "modelID": .string(result.modelID),
                "promptTime": .number(result.promptTime),
                "generateTime": .number(result.generateTime),
                "stoppedBySequence": .bool(result.stoppedBySequence),
            ]
            if let telemetry = result.speculativeTelemetry {
                metadata[AFMMLXSpeculativeTelemetry.metadataKey] = telemetry.metadataValue
            }
            return AFMModelResponse(
                text: normalized.text,
                reasoning: normalized.reasoning,
                toolCalls: toolCalls,
                usage: AFMUsage(
                    inputTokens: result.promptTokens,
                    cachedInputTokens: result.cachedTokens,
                    outputTokens: result.completionTokens
                ),
                finishReason: finishReason,
                tokenLogprobs: result.tokenLogprobs?.map {
                    AFMTokenLogProbability(
                        token: $0.token,
                        tokenID: $0.tokenId,
                        logprob: $0.logprob,
                        topTokens: $0.topTokens.map {
                            AFMTopLogProbability(
                                token: $0.token,
                                tokenID: $0.tokenId,
                                logprob: $0.logprob
                            )
                        }
                    )
                },
                metadata: metadata
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AFMError {
            throw error
        } catch {
            throw AFMError.generationFailed(error.localizedDescription)
        }
    }

    public func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var reservation: AFMMLXSchedulerReservation? =
                    switch schedulerAdmissionOwnership {
                    case .model:
                        nil
                    case .caller(let admission):
                        admission.reservation
                    }
                defer {
                    if let reservation { service.releaseSlot(reservation) }
                }
                do {
                    if let streamingLoadOverride {
                        try await streamingLoadOverride()
                    } else {
                        _ = try await load(progress: nil)
                    }
                    let schedulerAdmission: AFMMLXSchedulerAdmission
                    switch schedulerAdmissionOwnership {
                    case .model:
                        schedulerAdmission = await service.waitForSlot(timeout: 30)
                    case .caller(let admission):
                        schedulerAdmission = admission
                    }
                    guard schedulerAdmission.isAdmitted else {
                        throw AFMError.unavailable("MLX scheduler is at capacity")
                    }
                    reservation = schedulerAdmission.reservation
                    let submissionAdmission: BatchSchedulerSubmissionAdmission =
                        switch schedulerAdmission {
                        case .serial:
                            .unreserved
                        case .reserved(let reservation):
                            .reserved(reservation)
                        case .unavailable:
                            .unreserved
                        }
                    let tools = request.effectiveOpenAITools()
                    let result = try await service.generateStreamingWithSchedulerAdmission(
                        model: modelID,
                        messages: try request.openAIMessages(),
                        temperature: request.options.temperature,
                        maxTokens: request.options.maximumResponseTokens,
                        topP: request.options.topP,
                        repetitionPenalty: request.options.repetitionPenalty,
                        topK: request.options.topK,
                        minP: request.options.minP,
                        presencePenalty: request.options.presencePenalty,
                        seed: request.options.seed,
                        logprobs: request.options.logprobs,
                        topLogprobs: request.options.topLogprobs,
                        tools: tools,
                        parallelToolCalls: request.parallelToolCalls,
                        stop: request.options.stopSequences,
                        responseFormat: request.openAIResponseFormat(),
                        chatTemplateKwargs: request.chatTemplateKwargs(),
                        speculativeDecoding: request.speculativeDecodingOptions(),
                        preserveStructuralTags: request.preserveStructuralTags,
                        requestId: nil,
                        admission: submissionAdmission
                    )
                    reservation = nil
                    var translator = MLXStreamEventTranslator(
                        thinkStartTag: result.thinkStartTag,
                        thinkEndTag: result.thinkEndTag,
                        maximumResponseTokens: request.options.maximumResponseTokens,
                        tools: tools
                    )
                    let streamService = service
                    var rawToolFallback = AFMMLXRawToolStreamFallback(
                        toolCallStartTag: result.toolCallStartTag,
                        toolCallEndTag: result.toolCallEndTag,
                        toolCallParser: streamService.resolvedToolCallParser(logBypass: false),
                        tools: tools,
                        applyFixToolArgs: { toolCall in
                            streamService.coerceToolCallArguments(
                                streamService.remapToolCallArguments(toolCall, tools: tools),
                                tools: tools
                            )
                        },
                        remapSingleKey: { key, toolName in
                            let remapped = streamService.remapArgumentKeys(
                                [key: ""],
                                toolName: toolName,
                                tools: tools
                            )
                            return remapped.keys.first ?? key
                        }
                    )
                    var completedToolCalls: [AFMToolCall] = []
                    for try await chunk in result.stream {
                        try Task.checkCancellation()
                        for normalizedChunk in rawToolFallback.consume(chunk) {
                            for event in translator.consume(normalizedChunk) {
                                let event = Self.sanitizedToolCallEvent(event)
                                if case .toolCall(let call, .completed) = event {
                                    completedToolCalls.append(call)
                                }
                                continuation.yield(event)
                            }
                        }
                    }
                    try Task.checkCancellation()
                    for normalizedChunk in rawToolFallback.finish() {
                        for event in translator.consume(normalizedChunk) {
                            let event = Self.sanitizedToolCallEvent(event)
                            if case .toolCall(let call, .completed) = event {
                                completedToolCalls.append(call)
                            }
                            continuation.yield(event)
                        }
                    }
                    let finalEvents = translator.finish().map(Self.sanitizedToolCallEvent)
                    for event in finalEvents {
                        if case .toolCall(let call, .completed) = event {
                            completedToolCalls.append(call)
                        }
                    }
                    try AFMMLXToolPolicy.validateCompletedToolCalls(
                        completedToolCalls,
                        for: request
                    )
                    for event in finalEvents {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func unload() async {
        await runtime.unload()
    }

    public func tokenize(text: String) async throws -> [Int] {
        _ = try await load(progress: nil)
        return try await service.tokenize(text: text)
    }

    private static func splitReasoning(
        _ value: String,
        startTag: String?,
        endTag: String?
    ) -> (text: String, reasoning: String?) {
        guard let startTag, let endTag else { return (value, nil) }
        var translator = MLXStreamEventTranslator(
            thinkStartTag: startTag,
            thinkEndTag: endTag,
            maximumResponseTokens: nil
        )
        let events = translator.consume(StreamChunk(text: value)) + translator.finish()
        var text = ""
        var reasoning = ""
        for event in events {
            switch event {
            case .responseText(_, let delta, _):
                text += delta
            case .reasoningText(_, let delta, _):
                reasoning += delta
            default:
                break
            }
        }
        return (text, reasoning.isEmpty ? nil : reasoning)
    }

    static func normalizedGeneratedResponse(
        _ value: String,
        startTag: String?,
        endTag: String?
    ) -> (text: String, reasoning: String?) {
        let split = splitReasoning(value, startTag: startTag, endTag: endTag)
        return normalizeResponse(text: split.text, reasoning: split.reasoning)
    }

    static func normalizeResponse(
        text: String,
        reasoning: String?
    ) -> (text: String, reasoning: String?) {
        let normalizedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedReasoning = reasoning?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (
            normalizedText,
            normalizedReasoning?.isEmpty == false ? normalizedReasoning : nil)
    }

    static func sanitizedToolName(_ value: String) -> String {
        let withoutTag = value.range(of: "</").map {
            String(value[..<$0.lowerBound])
        } ?? value
        return withoutTag.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func sanitizedToolCallEvent(
        _ event: AFMGenerationEvent
    ) -> AFMGenerationEvent {
        guard case .toolCall(let call, let stage) = event else { return event }
        let name = sanitizedToolName(call.name)
        guard name != call.name else { return event }
        return .toolCall(
            call: AFMToolCall(id: call.id, name: name, arguments: call.arguments),
            stage: stage
        )
    }

    private static func finishReason(
        toolCalls: [ResponseToolCall]?,
        stoppedBySequence: Bool,
        completionTokens: Int,
        maximumResponseTokens: Int?
    ) -> AFMFinishReason {
        if toolCalls?.isEmpty == false {
            return .toolCalls
        }
        if stoppedBySequence {
            return .stop
        }
        if let maximumResponseTokens,
           maximumResponseTokens > 0,
           completionTokens >= maximumResponseTokens {
            return .length
        }
        return .stop
    }
}

struct AFMMLXResponseAccumulator {
    private var text = ""
    private var reasoning = ""
    private var toolOrder: [String] = []
    private var toolCalls: [String: AFMToolCall] = [:]
    private var usage = AFMUsage()
    private var finishReason = AFMFinishReason.stop
    private var tokenLogprobs: [AFMTokenLogProbability] = []
    private var metadata: [String: AFMJSONValue]

    init(modelID: String) {
        self.metadata = ["modelID": .string(modelID)]
    }

    mutating func consume(_ event: AFMGenerationEvent) {
        switch event {
        case .responseText(let action, let value, _):
            Self.apply(action, value: value, to: &text)
        case .reasoningText(let action, let value, _):
            Self.apply(action, value: value, to: &reasoning)
        case .tokenLogprobs(let values):
            tokenLogprobs.append(contentsOf: values)
        case .toolCall(let call, let stage):
            switch stage {
            case .retracted:
                toolCalls.removeValue(forKey: call.id)
            case .started, .argumentsDelta, .completed:
                if !toolOrder.contains(call.id) {
                    toolOrder.append(call.id)
                }
                toolCalls[call.id] = call
            }
        case .usage(let value):
            usage = value
        case .metadata(let values):
            metadata.merge(values) { _, new in new }
        case .completed(let reason):
            finishReason = reason
        case .custom:
            break
        }
    }

    var response: AFMModelResponse {
        let normalized = AFMMLXModel.normalizeResponse(
            text: text,
            reasoning: reasoning.isEmpty ? nil : reasoning)
        return AFMModelResponse(
            text: normalized.text,
            reasoning: normalized.reasoning,
            toolCalls: toolOrder.compactMap { toolCalls[$0] },
            usage: usage,
            finishReason: finishReason,
            tokenLogprobs: tokenLogprobs.isEmpty ? nil : tokenLogprobs,
            metadata: metadata
        )
    }

    private static func apply(
        _ action: AFMTextUpdateAction,
        value: String,
        to destination: inout String
    ) {
        switch action {
        case .append: destination += value
        case .replace: destination = value
        }
    }
}

/// Converts raw model tool syntax into the same structured chunks produced by
/// the scheduler. This is a fallback for attached hosts that preserve tool tags
/// but do not install the scheduler's parser.
struct AFMMLXRawToolStreamFallback {
    private static let defaultDeepseekToolCallStartTag = "<｜DSML｜tool_calls>"
    private static let defaultDeepseekToolCallEndTag = "</｜DSML｜tool_calls>"

    private let toolCallStartTag: String?
    private let runtime: ToolCallStreamingRuntime?

    init(
        toolCallStartTag: String?,
        toolCallEndTag: String?,
        toolCallParser: String?,
        tools: [RequestTool]?,
        applyFixToolArgs: @escaping @Sendable (ResponseToolCall) -> ResponseToolCall,
        remapSingleKey: @escaping @Sendable (String, String) -> String
    ) {
        let startTag = toolCallStartTag ?? Self.defaultDeepseekToolCallStartTag
        let endTag = toolCallEndTag ?? Self.defaultDeepseekToolCallEndTag
        self.toolCallStartTag = startTag
        if tools?.isEmpty == false {
            self.runtime = ToolCallStreamingRuntime(
                toolCallStartTag: startTag,
                toolCallEndTag: endTag,
                toolCallParser: toolCallParser,
                tools: tools,
                applyFixToolArgs: applyFixToolArgs,
                remapSingleKey: remapSingleKey
            )
        } else {
            self.runtime = nil
        }
    }

    mutating func consume(_ chunk: StreamChunk) -> [StreamChunk] {
        guard let runtime,
              chunk.toolCallDeltas?.isEmpty != false,
              chunk.toolCalls?.isEmpty != false else {
            return [chunk]
        }

        var chunks: [StreamChunk] = []
        var toolPiece = chunk.text
        if !runtime.inToolCall,
           let toolCallStartTag,
           let range = toolPiece.range(of: toolCallStartTag),
           range.lowerBound != toolPiece.startIndex {
            chunks.append(StreamChunk(text: String(toolPiece[..<range.lowerBound])))
            toolPiece = String(toolPiece[range.lowerBound...])
        }

        let output = runtime.process(piece: toolPiece)
        guard output.handled else { return [chunk] }

        if let passthroughText = output.passthroughText, !passthroughText.isEmpty {
            chunks.append(StreamChunk(text: passthroughText))
        }
        chunks.append(contentsOf: BatchScheduler.streamChunksToEmit(from: output.events))
        if chunk.logprobs != nil || chunk.promptTokens != nil ||
            chunk.completionTokens != nil || chunk.cachedTokens != nil ||
            chunk.promptTime != nil || chunk.generateTime != nil ||
            chunk.stoppedBySequence != nil {
            chunks.append(
                StreamChunk(
                    text: "",
                    logprobs: chunk.logprobs,
                    promptTokens: chunk.promptTokens,
                    completionTokens: chunk.completionTokens,
                    cachedTokens: chunk.cachedTokens,
                    promptTime: chunk.promptTime,
                    generateTime: chunk.generateTime,
                    stoppedBySequence: chunk.stoppedBySequence
                )
            )
        }
        return chunks
    }

    mutating func finish() -> [StreamChunk] {
        guard let runtime else { return [] }
        return BatchScheduler.streamChunksToEmit(
            from: runtime.finishIncompleteToolCall()
        )
    }
}

enum AFMMLXToolPolicy {
    static func validateCompletedToolCalls(
        _ calls: [AFMToolCall],
        for request: AFMRequest
    ) throws {
        guard request.requiresToolCall else { return }
        guard !request.tools.isEmpty else {
            throw AFMError.invalidRequest(
                "Tool calling is required, but no tools are enabled."
            )
        }
        guard !calls.isEmpty else {
            throw AFMError.generationFailed(
                "The model returned no tool call while tool calling was required."
            )
        }
    }
}

public enum AFMMLXModelDescriptor {
    public static func describe(
        modelID: String,
        resolver: MLXCacheResolver = .init()
    ) -> AFMModelDescriptor {
        let directory = resolver.localModelDirectory(repoId: modelID)
        let config = directory.flatMap {
            jsonObject(at: $0.appendingPathComponent("config.json"))
        }
        let tokenizer = directory.flatMap {
            jsonObject(at: $0.appendingPathComponent("tokenizer_config.json"))
        }
        let generation = directory.flatMap {
            jsonObject(at: $0.appendingPathComponent("generation_config.json"))
        }
        let template = tokenizer?["chat_template"] as? String ?? ""
        let lowerID = modelID.lowercased()

        var capabilities: AFMModelCapabilities = [
            .text, .streaming, .structuredOutput, .prefixCaching
        ]
        if config.map(isVisionModelConfiguration) == true {
            capabilities.insert(.vision)
        }
        let reasoningPatterns = [
            "qwen3", "deepseek-r", "glm-4", "glm-5", "kimi", "qwq",
            "marco-o1", "skywork-o1", "ling-", "nemotron", "minimax", "gpt-oss"
        ]
        if template.contains("<think>")
            || generation?["enable_thinking"] as? Bool == true
            || reasoningPatterns.contains(where: lowerID.contains) {
            capabilities.insert(.reasoning)
        }
        if template.contains("tools") || template.contains("tool_call") {
            capabilities.insert(.toolCalling)
        }
        if let directory,
           FileManager.default.fileExists(
               atPath: directory.appendingPathComponent("mtp.safetensors").path
           ) {
            capabilities.insert(.speculativeDecoding)
        }

        let textConfig = config?["text_config"] as? [String: Any]
        let contextWindow = config?["max_position_embeddings"] as? Int
            ?? textConfig?["max_position_embeddings"] as? Int
        let displayName = modelID.split(separator: "/").last.map(String.init) ?? modelID
        return AFMModelDescriptor(
            providerID: AFMMLXProviderFactory.providerID,
            modelID: AFMModelID(rawValue: modelID),
            displayName: displayName,
            capabilities: capabilities,
            contextWindow: contextWindow,
            privacyBoundary: .device,
            requiresNetwork: directory == nil,
            metadata: [
                "runtime": .string("mlx-swift"),
                "defaultMaximumResponseTokens": .integer(8_192)
            ]
        )
    }

    /// Returns whether a decoded MLX `config.json` describes a vision-language
    /// model. Some families use the same top-level `model_type` for text and
    /// VLM variants, so this checks architectures and nested vision fields
    /// rather than relying on one key.
    public static func isVisionModelConfiguration(_ config: [String: Any]) -> Bool {
        if let architectures = config["architectures"] as? [String] {
            for architecture in architectures {
                let lower = architecture.lowercased()
                if lower.contains("vlm")
                    || lower.contains("vision")
                    || lower.contains("qwen2vl")
                    || lower.contains("qwenvl")
                    || lower.contains("llava")
                    || lower.contains("pixtral") {
                    return true
                }
            }
        }

        let modelType = (config["model_type"] as? String ?? "").lowercased()
        if modelType.contains("vl")
            || modelType.contains("vision")
            || modelType.contains("qwen2_vl")
            || modelType.contains("llava") {
            return true
        }

        if config["text_config"] != nil && config["vision_config"] != nil {
            return true
        }

        if config["vision_config"] != nil {
            return true
        }

        if config["visual"] != nil {
            return true
        }

        return false
    }

    public static func isVisionModelConfiguration(in modelDirectory: URL) -> Bool {
        guard let config = jsonObject(at: modelDirectory.appendingPathComponent("config.json")) else {
            return false
        }

        return isVisionModelConfiguration(config)
    }

    /// Returns true when the MLX configuration describes a VLM layout that
    /// should be loaded through the VLM factory instead of the LLM factory.
    /// Some multimodal configs store text architecture fields only in
    /// `text_config`; the generic LLM factory can fill unsafe defaults when
    /// those fields are absent at both levels.
    public static func requiresVisionModelFactory(_ config: [String: Any]) -> Bool {
        guard let textConfig = config["text_config"] as? [String: Any],
              config["vision_config"] != nil else {
            return false
        }

        let hasTopLevelHeads = config["num_attention_heads"] != nil
        let hasNestedHeads = textConfig["num_attention_heads"] != nil
        return !hasTopLevelHeads && !hasNestedHeads
    }

    public static func requiresVisionModelFactory(in modelDirectory: URL) -> Bool {
        guard let config = jsonObject(at: modelDirectory.appendingPathComponent("config.json")) else {
            return false
        }

        return requiresVisionModelFactory(config)
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}

extension AFMRequest {
    var preserveStructuralTags: Bool {
        guard case .bool(let value)? =
            metadata[AFMMLXRequestMetadata.preserveStructuralTags]
        else { return false }
        return value
    }

    func speculativeDecodingOptions() -> SpeculativeDecodingOptions? {
        guard case .object(let values)? = metadata["speculativeDecoding"] else {
            return nil
        }
        func string(_ key: String) -> String? {
            guard case .string(let value)? = values[key] else { return nil }
            return value
        }
        func integer(_ key: String) -> Int? {
            guard case .integer(let value)? = values[key] else { return nil }
            return value
        }
        return SpeculativeDecodingOptions(
            mode: string("mode"),
            requirement: string("requirement"),
            drafter: string("drafter"),
            maxDraftTokens: integer("maxDraftTokens"),
            forceAutoregressiveReason: string("forceAutoregressiveReason")
        )
    }

    var requiresToolCall: Bool {
        metadata["toolCallingMode"] == .string("required")
    }

    var requiredToolName: String? {
        guard case .string(let value)? = metadata["requiredToolName"] else {
            return nil
        }
        return value
    }

    var includeSchemaInPrompt: Bool {
        guard case .bool(let value)? = metadata["includeSchemaInPrompt"] else {
            return true
        }
        return value
    }

    var parallelToolCalls: Bool? {
        guard case .bool(let value)? = metadata["parallelToolCalls"] else {
            return nil
        }
        return value
    }

    func effectiveOpenAITools() -> [RequestTool]? {
        if case .string("disallowed")? = metadata["toolCallingMode"] {
            return nil
        }
        return openAITools()
    }

    func chatTemplateKwargs() -> [String: AnyCodable]? {
        var result: [String: AnyCodable] = [
            "afm_include_schema_in_prompt": AnyCodable(includeSchemaInPrompt)
        ]
        if requiresToolCall {
            if let requiredToolName {
                result["tool_choice"] = AnyCodable([
                    "type": "function",
                    "function": ["name": requiredToolName]
                ])
            } else {
                result["tool_choice"] = AnyCodable("required")
            }
        }
        if case .object(let values)? = metadata["chatTemplateKwargs"] {
            result.merge(
                values.mapValues { AnyCodable($0.foundationValue) }
            ) { _, new in new }
        }
        return result
    }

    func openAIMessages() throws -> [Message] {
        var result = try messages.map { message in
            let content: MessageContent?
            if message.content.isEmpty {
                content = nil
            } else {
                content = .parts(
                    try message.content.map { try $0.openAIContentPart }
                )
            }
            return Message(
                role: message.role.rawValue,
                content: content,
                toolCalls: message.toolCalls.isEmpty ? nil : message.toolCalls.map {
                    MessageToolCall(
                        id: $0.id,
                        type: "function",
                        function: MessageToolCallFunction(
                            name: $0.name,
                            arguments: $0.arguments
                        )
                    )
                },
                toolCallId: message.toolCallID,
                name: message.name
            )
        }
        if requiresToolCall, !tools.isEmpty {
            let instruction: String
            if let requiredToolName {
                instruction = "You must call the \(requiredToolName) tool. Do not answer with text."
            } else {
                instruction = "You must call one of the available tools. Do not answer with text."
            }
            result.insert(Message(role: "system", content: .text(instruction)), at: 0)
        }
        return result
    }

    func openAITools() -> [RequestTool]? {
        guard !tools.isEmpty else { return nil }
        return tools.map {
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: $0.name,
                    description: $0.description,
                    parameters: AnyCodable($0.inputSchema.foundationValue),
                    strict: true
                )
            )
        }
    }

    func openAIResponseFormat() -> ResponseFormat? {
        switch options.responseConstraint {
        case .none:
            return nil
        case .jsonObject:
            return ResponseFormat(type: "json_object")
        case .jsonSchema(let name, let schema, let strict):
            return ResponseFormat(
                type: "json_schema",
                jsonSchema: ResponseJsonSchema(
                    name: name,
                    description: nil,
                    schema: AnyCodable(schema.foundationValue),
                    strict: strict
                )
            )
        case .grammar:
            return nil
        }
    }
}

private extension AFMContentPart {
    var openAIContentPart: ContentPart {
        get throws {
            switch self {
            case .text(let text):
                return ContentPart(type: "text", text: text)
            case .data(let mimeType, let value):
                if mimeType.hasPrefix("audio/") {
                    return ContentPart(
                        type: "input_audio",
                        input_audio: InputAudio(
                            data: value.base64EncodedString(),
                            format: String(mimeType.dropFirst("audio/".count)),
                            language: nil
                        )
                    )
                }
                return ContentPart(
                    type: "image_url",
                    image_url: ImageURL(
                        url: "data:\(mimeType);base64,\(value.base64EncodedString())",
                        detail: nil
                    )
                )
            case .reference(let url):
                return ContentPart(
                    type: "image_url",
                    image_url: ImageURL(url: url.absoluteString, detail: nil)
                )
            case .custom(let type, _):
                throw AFMError.unsupportedCapability("custom content '\(type)'")
            }
        }
    }
}

private extension AFMJSONValue {
    var foundationValue: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .integer(let value): return value
        case .number(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map(\.foundationValue)
        case .object(let values): return values.mapValues(\.foundationValue)
        }
    }
}
