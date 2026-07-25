#if canImport(FoundationModels)
import Foundation
import AFMKit
import AFMOpenAICompat
import FoundationModels
import ImageIO
import UniformTypeIdentifiers

/// An MLX-backed model that participates in the macOS 27 Foundation Models
/// `LanguageModelSession` API.
@available(macOS 27.0, *)
public struct MLXLanguageModel: LanguageModel, Sendable {
    public typealias Executor = MLXLanguageModelExecutor

    public let modelID: String
    public let engineConfig: MLXLanguageModelExecutor.Configuration

    public init(
        modelID: String,
        kvBits: Int? = nil,
        enablePrefixCaching: Bool = true,
        mtpEnabled: Bool = false,
        mtpDepth: Int = 3,
        eagle3DrafterPath: String? = nil,
        maxConcurrent: Int = 0,
        defaultMaximumResponseTokens: Int = 2_048,
        supportsVision: Bool = false,
        supportsReasoning: Bool = false,
        supportsToolCalling: Bool = false,
        supportsGuidedGeneration: Bool = false
    ) {
        self.modelID = modelID
        self.engineConfig = .init(
            modelID: modelID,
            kvBits: kvBits,
            enablePrefixCaching: enablePrefixCaching,
            mtpEnabled: mtpEnabled,
            mtpDepth: mtpDepth,
            eagle3DrafterPath: eagle3DrafterPath,
            maxConcurrent: maxConcurrent,
            defaultMaximumResponseTokens: defaultMaximumResponseTokens,
            supportsVision: supportsVision,
            supportsReasoning: supportsReasoning,
            supportsToolCalling: supportsToolCalling,
            supportsGuidedGeneration: supportsGuidedGeneration
        )
    }

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        if engineConfig.supportsVision {
            capabilities.append(.vision)
        }
        if engineConfig.supportsReasoning {
            capabilities.append(.reasoning)
        }
        if engineConfig.supportsToolCalling {
            capabilities.append(.toolCalling)
        }
        if engineConfig.supportsGuidedGeneration {
            capabilities.append(.guidedGeneration)
        }
        return LanguageModelCapabilities(capabilities)
    }

    public var executorConfiguration: MLXLanguageModelExecutor.Configuration {
        engineConfig
    }
}

/// Executes macOS 27 Foundation Models requests through `AFMEngine`.
@available(macOS 27.0, *)
public final class MLXLanguageModelExecutor: LanguageModelExecutor, @unchecked Sendable {
    public typealias Model = MLXLanguageModel

    public struct Configuration: Hashable, Sendable {
        public let modelID: String
        public let kvBits: Int?
        public let enablePrefixCaching: Bool
        public let mtpEnabled: Bool
        public let mtpDepth: Int
        public let eagle3DrafterPath: String?
        public let maxConcurrent: Int
        public let defaultMaximumResponseTokens: Int
        public let supportsVision: Bool
        public let supportsReasoning: Bool
        public let supportsToolCalling: Bool
        public let supportsGuidedGeneration: Bool

        public init(
            modelID: String,
            kvBits: Int? = nil,
            enablePrefixCaching: Bool = true,
            mtpEnabled: Bool = false,
            mtpDepth: Int = 3,
            eagle3DrafterPath: String? = nil,
            maxConcurrent: Int = 0,
            defaultMaximumResponseTokens: Int = 2_048,
            supportsVision: Bool = false,
            supportsReasoning: Bool = false,
            supportsToolCalling: Bool = false,
            supportsGuidedGeneration: Bool = false
        ) {
            self.modelID = modelID
            self.kvBits = kvBits
            self.enablePrefixCaching = enablePrefixCaching
            self.mtpEnabled = mtpEnabled
            self.mtpDepth = mtpDepth
            self.eagle3DrafterPath = eagle3DrafterPath
            self.maxConcurrent = maxConcurrent
            self.defaultMaximumResponseTokens = defaultMaximumResponseTokens
            self.supportsVision = supportsVision
            self.supportsReasoning = supportsReasoning
            self.supportsToolCalling = supportsToolCalling
            self.supportsGuidedGeneration = supportsGuidedGeneration
        }
    }

    private let runtime: MLXLanguageModelRuntime

    public init(configuration: Configuration) throws {
        self.runtime = MLXLanguageModelRuntime(
            engine: AFMEngine(
                backend: .mlx(modelID: configuration.modelID),
                config: EngineConfig(
                    kvBits: configuration.kvBits,
                    enablePrefixCaching: configuration.enablePrefixCaching,
                    mtpEnabled: configuration.mtpEnabled,
                    mtpDepth: configuration.mtpDepth,
                    eagle3DrafterPath: configuration.eagle3DrafterPath,
                    maxConcurrent: configuration.maxConcurrent
                )
            )
        )
    }

    deinit {
        let runtime = runtime
        Task {
            await runtime.unload()
        }
    }

    public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
        let runtime = runtime
        Task {
            _ = try? await runtime.preparedEngine()
        }
    }

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if request.schema != nil && !model.engineConfig.supportsGuidedGeneration {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "MLX guided generation is not wired to Foundation Models yet."
                )
            )
        }
        if !request.enabledToolDefinitions.isEmpty && !model.engineConfig.supportsToolCalling {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .toolCalling,
                    debugDescription: "MLX tool calling is not wired to Foundation Models yet."
                )
            )
        }

        let messages = try Self.messages(from: request.transcript)
        guard !messages.isEmpty else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: Array(request.transcript),
                    debugDescription: "The MLX provider could not convert the transcript to text messages."
                )
            )
        }

        let engine = try await runtime.preparedEngine()
        let options = try Self.generationConfig(from: request, model: model)

        var sentUsage = false
        var streamedTokens = 0
        for try await event in engine.streamEvents(to: messages, options) {
            switch event {
            case .text(let text, let tokenCount):
                streamedTokens += tokenCount
                await channel.send(
                    .response(action: .appendText(text, tokenCount: tokenCount))
                )
            case .usage(let promptTokens, let completionTokens, let cachedTokens):
                sentUsage = true
                await channel.send(
                    .response(
                        action: .updateUsage(
                            input: .init(
                                totalTokenCount: promptTokens,
                                cachedTokenCount: cachedTokens
                            ),
                            output: .init(
                                totalTokenCount: completionTokens,
                                reasoningTokenCount: 0
                            )
                        )
                    )
                )
            case .reasoning(let text, let tokenCount):
                await channel.send(
                    .reasoning(action: .appendText(text, tokenCount: tokenCount))
                )
            case .tokenLogprobs:
                continue
            case .toolCall(let call, let stage):
                switch stage {
                case .started:
                    await channel.send(
                        .toolCalls(
                            action: .toolCall(
                                id: call.id,
                                name: call.name,
                                action: .appendArguments("", tokenCount: 0)
                            )
                        )
                    )
                case .argumentsDelta(let delta):
                    await channel.send(
                        .toolCalls(
                            action: .toolCall(
                                id: call.id,
                                name: call.name,
                                action: .appendArguments(delta, tokenCount: 0)
                            )
                        )
                    )
                case .completed, .retracted:
                    continue
                }
            case .metadata(let values):
                await channel.send(
                    .response(action: .updateMetadata(Self.foundationMetadata(values)))
                )
            case .custom(let type, let payload):
                await channel.send(
                    .response(
                        action: .updateMetadata([
                            "afm.custom.\(type)": payload.base64EncodedString()
                        ])
                    )
                )
            case .completed(let reason):
                await channel.send(
                    .response(
                        action: .updateMetadata([
                            "afm.finishReason": String(describing: reason)
                        ])
                    )
                )
            }
        }

        if !sentUsage {
            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: 0, cachedTokenCount: 0),
                        output: .init(
                            totalTokenCount: streamedTokens,
                            reasoningTokenCount: 0
                        )
                    )
                )
            )
        }
    }

    static func messages(from transcript: Transcript) throws -> [Message] {
        var messages: [Message] = []

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                if let content = try messageContent(from: instructions.segments) {
                    messages.append(Message(role: "system", content: content))
                }
            case .prompt(let prompt):
                if let content = try messageContent(from: prompt.segments) {
                    messages.append(Message(role: "user", content: content))
                }
            case .response(let response):
                if let content = try messageContent(from: response.segments) {
                    messages.append(Message(role: "assistant", content: content))
                }
            case .reasoning(let reasoning):
                if let content = try messageContent(from: reasoning.segments) {
                    messages.append(Message(role: "assistant", content: content))
                }
            case .toolCalls(let toolCalls):
                messages.append(
                    Message(
                        role: "assistant",
                        content: nil,
                        toolCalls: toolCalls.map { call in
                            MessageToolCall(
                                id: call.id,
                                type: "function",
                                function: MessageToolCallFunction(
                                    name: call.toolName,
                                    arguments: call.arguments.jsonString
                                )
                            )
                        }
                    )
                )
            case .toolOutput(let output):
                messages.append(
                    Message(
                        role: "tool",
                        content: try messageContent(from: output.segments),
                        toolCallId: output.id,
                        name: output.toolName
                    )
                )
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [entry],
                        debugDescription: "The transcript contains an unknown entry type."
                    )
                )
            }
        }

        return messages
    }

    static func tools(
        from definitions: [Transcript.ToolDefinition]
    ) throws -> [RequestTool]? {
        guard !definitions.isEmpty else { return nil }
        return try definitions.map { definition in
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: definition.name,
                    description: definition.description,
                    parameters: try anyCodable(from: definition.parameters),
                    strict: true
                )
            )
        }
    }

    static func responseFormat(from schema: GenerationSchema?) throws -> ResponseFormat? {
        guard let schema else { return nil }
        return ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: schema.name,
                description: nil,
                schema: try anyCodable(from: schema),
                strict: true
            )
        )
    }

    static func generationConfig(
        from request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws -> GenerationConfig {
        var temperature = request.generationOptions.temperature
        var topP: Double?
        var topK: Int?
        var seed: Int?
        if let kind = request.generationOptions.samplingMode?.kind {
            switch kind {
            case .greedy:
                temperature = 0
            case .randomTopK(let value, let randomSeed):
                topK = value
                seed = randomSeed.flatMap { Int(exactly: $0) }
            case .randomProbabilityThreshold(let value, let randomSeed):
                topP = value
                seed = randomSeed.flatMap { Int(exactly: $0) }
            @unknown default:
                break
            }
        }

        let toolCallingMode: String?
        switch request.generationOptions.toolCallingMode?.kind {
        case .allowed: toolCallingMode = "allowed"
        case .required: toolCallingMode = "required"
        case .disallowed: toolCallingMode = "disallowed"
        case nil: toolCallingMode = nil
        @unknown default: toolCallingMode = nil
        }

        var metadata: [String: AFMJSONValue] = [
            "includeSchemaInPrompt": .bool(
                request.contextOptions.includeSchemaInPrompt ?? true
            )
        ]
        if let toolCallingMode {
            metadata["toolCallingMode"] = .string(toolCallingMode)
        }
        if let reasoningLevel = request.contextOptions.reasoningLevel {
            switch reasoningLevel {
            case .light: metadata["reasoningLevel"] = .string("light")
            case .moderate: metadata["reasoningLevel"] = .string("moderate")
            case .deep: metadata["reasoningLevel"] = .string("deep")
            case .custom(let value):
                metadata["reasoningLevel"] = .string(value)
            @unknown default:
                break
            }
        }
        metadata.merge(Self.afmMetadata(request.metadata)) { _, new in new }

        let definitions = request.generationOptions.toolCallingMode?.kind == .disallowed
            ? []
            : request.enabledToolDefinitions
        return GenerationConfig(
            temperature: temperature,
            maxTokens: request.generationOptions.maximumResponseTokens
                ?? model.engineConfig.defaultMaximumResponseTokens,
            topP: topP,
            topK: topK,
            seed: seed,
            tools: try Self.tools(from: definitions),
            responseFormat: try Self.responseFormat(from: request.schema),
            metadata: metadata
        )
    }

    private static func anyCodable(from schema: GenerationSchema) throws -> AnyCodable {
        let data = try JSONEncoder().encode(schema)
        let object = try JSONSerialization.jsonObject(with: data)
        return AnyCodable(object)
    }

    private static func messageContent(
        from segments: [Transcript.Segment]
    ) throws -> MessageContent? {
        let parts = try segments.flatMap { segment -> [ContentPart] in
            switch segment {
            case .text(let text):
                return [ContentPart(type: "text", text: text.content)]
            case .structure(let structure):
                return [ContentPart(type: "text", text: structure.content.jsonString)]
            case .attachment(let attachment):
                var result: [ContentPart] = []
                if let label = attachment.label, !label.isEmpty {
                    result.append(ContentPart(type: "text", text: label))
                }
                switch attachment.content {
                case .image(let image):
                    result.append(
                        ContentPart(
                            type: "image_url",
                            image_url: ImageURL(
                                url: try imageDataURL(image),
                                detail: nil
                            )
                        )
                    )
                @unknown default:
                    throw LanguageModelError.unsupportedTranscriptContent(
                        .init(
                            unsupportedContent: [],
                            debugDescription: "The transcript contains an unknown attachment type."
                        )
                    )
                }
                return result
            case .custom(let custom):
                return [
                    ContentPart(
                        type: "text",
                        text: customContent(custom)
                    )
                ]
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [],
                        debugDescription: "The transcript contains an unknown segment type."
                    )
                )
            }
        }
        guard !parts.isEmpty else { return nil }
        if parts.allSatisfy({ $0.type == "text" }) {
            return .text(parts.compactMap(\.text).joined(separator: "\n"))
        }
        return .parts(parts)
    }

    private static func customContent<C: Transcript.CustomSegment>(
        _ custom: C
    ) -> String {
        if let data = try? JSONEncoder().encode(custom.content),
           let value = String(data: data, encoding: .utf8) {
            return value
        }
        return custom.description
    }

    private static func imageDataURL(
        _ image: Transcript.ImageAttachment
    ) throws -> String {
        if let url = image.url {
            return url.absoluteString
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: [],
                    debugDescription: "Could not create a PNG image destination."
                )
            )
        }
        CGImageDestinationAddImage(destination, image.cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: [],
                    debugDescription: "Could not encode the image attachment as PNG."
                )
            )
        }
        return "data:image/png;base64,\((data as Data).base64EncodedString())"
    }

    private static func afmMetadata(
        _ values: [String: any Sendable & Codable & Equatable]
    ) -> [String: AFMJSONValue] {
        values.reduce(into: [:]) { result, item in
            switch item.value {
            case let value as Bool: result[item.key] = .bool(value)
            case let value as Int: result[item.key] = .integer(value)
            case let value as Double: result[item.key] = .number(value)
            case let value as String: result[item.key] = .string(value)
            default: result[item.key] = .string(String(describing: item.value))
            }
        }
    }

    private static func foundationMetadata(
        _ values: [String: AFMJSONValue]
    ) -> [String: any Sendable & Codable & Equatable] {
        values.reduce(into: [:]) { result, item in
            switch item.value {
            case .null: result[item.key] = "null"
            case .bool(let value): result[item.key] = value
            case .integer(let value): result[item.key] = value
            case .number(let value): result[item.key] = value
            case .string(let value): result[item.key] = value
            case .array, .object:
                result[item.key] = String(describing: item.value)
            }
        }
    }
}

@available(macOS 27.0, *)
private actor MLXLanguageModelRuntime {
    let engine: AFMEngine
    private var loadTask: Task<String, Error>?

    init(engine: AFMEngine) {
        self.engine = engine
    }

    func preparedEngine() async throws -> AFMEngine {
        if let loadTask {
            _ = try await loadTask.value
            return engine
        }

        let engine = engine
        let task = Task {
            try await engine.load()
        }
        loadTask = task
        do {
            _ = try await task.value
            return engine
        } catch {
            loadTask = nil
            throw error
        }
    }

    func unload() async {
        loadTask?.cancel()
        loadTask = nil
        await engine.unload()
    }
}
#endif
