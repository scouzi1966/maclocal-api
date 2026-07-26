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
        } catch {
            throw AFMError.loadingFailed(error.localizedDescription)
        }
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        _ = try await load(progress: nil)
        do {
            let tools = request.effectiveOpenAITools()
            let result = try await service.generate(
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
                chatTemplateKwargs: request.chatTemplateKwargs()
            )
            let split = Self.splitReasoning(
                result.content,
                startTag: service.thinkStartTag,
                endTag: service.thinkEndTag
            )
            let toolCalls = (result.toolCalls ?? []).map {
                AFMToolCall(
                    id: $0.id,
                    name: $0.function.name,
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
            return AFMModelResponse(
                text: split.text,
                reasoning: split.reasoning,
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
                metadata: [
                    "modelID": .string(result.modelID),
                    "promptTime": .number(result.promptTime),
                    "generateTime": .number(result.generateTime),
                    "stoppedBySequence": .bool(result.stoppedBySequence)
                ]
            )
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
                do {
                    _ = try await load(progress: nil)
                    let tools = request.effectiveOpenAITools()
                    let result = try await service.generateStreaming(
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
                        requestId: nil
                    )
                    var translator = MLXStreamEventTranslator(
                        thinkStartTag: result.thinkStartTag,
                        thinkEndTag: result.thinkEndTag,
                        maximumResponseTokens: request.options.maximumResponseTokens
                    )
                    var completedToolCalls: [AFMToolCall] = []
                    for try await chunk in result.stream {
                        try Task.checkCancellation()
                        for event in translator.consume(chunk) {
                            if case .toolCall(let call, .completed) = event {
                                completedToolCalls.append(call)
                            }
                            continuation.yield(event)
                        }
                    }
                    try Task.checkCancellation()
                    let finalEvents = translator.finish()
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
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedReasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        return (trimmedText, trimmedReasoning.isEmpty ? nil : trimmedReasoning)
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
        if config?["vision_config"] != nil || config?["visual"] != nil {
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

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}

private extension AFMRequest {
    var requiresToolCall: Bool {
        metadata["toolCallingMode"] == .string("required")
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
        if case .object(let values)? = metadata["chatTemplateKwargs"] {
            result.merge(
                values.mapValues { AnyCodable($0.foundationValue) }
            ) { _, new in new }
        }
        return result
    }

    func openAIMessages() throws -> [Message] {
        try messages.map { message in
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
