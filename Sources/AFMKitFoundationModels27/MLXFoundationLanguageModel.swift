#if canImport(FoundationModels)
import Foundation
import AFMKit
import AFMOpenAICompat
import FoundationModels

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
            supportsReasoning: supportsReasoning,
            supportsToolCalling: supportsToolCalling,
            supportsGuidedGeneration: supportsGuidedGeneration
        )
    }

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
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
public struct MLXLanguageModelExecutor: LanguageModelExecutor {
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
            self.supportsReasoning = supportsReasoning
            self.supportsToolCalling = supportsToolCalling
            self.supportsGuidedGeneration = supportsGuidedGeneration
        }
    }

    private let engine: AFMEngine

    public init(configuration: Configuration) throws {
        self.engine = AFMEngine(
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
    }

    public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
        Task {
            _ = try? await engine.load()
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

        _ = try await engine.load()
        let options = GenerationConfig(
            temperature: request.generationOptions.temperature,
            maxTokens: request.generationOptions.maximumResponseTokens
                ?? model.engineConfig.defaultMaximumResponseTokens,
            tools: try Self.tools(from: request.enabledToolDefinitions),
            responseFormat: try Self.responseFormat(from: request.schema)
        )

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
            case .metadata, .custom, .completed:
                continue
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
                let content = try textContent(from: instructions.segments)
                if !content.isEmpty {
                    messages.append(Message(role: "system", content: content))
                }
            case .prompt(let prompt):
                let content = try textContent(from: prompt.segments)
                if !content.isEmpty {
                    messages.append(Message(role: "user", content: content))
                }
            case .response(let response):
                let content = try textContent(from: response.segments)
                if !content.isEmpty {
                    messages.append(Message(role: "assistant", content: content))
                }
            case .reasoning:
                continue
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
                        content: .text(try textContent(from: output.segments)),
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

    private static func anyCodable(from schema: GenerationSchema) throws -> AnyCodable {
        let data = try JSONEncoder().encode(schema)
        let object = try JSONSerialization.jsonObject(with: data)
        return AnyCodable(object)
    }

    private static func textContent(from segments: [Transcript.Segment]) throws -> String {
        try segments.compactMap { segment in
            switch segment {
            case .text(let text):
                return text.content
            case .structure(let structure):
                return structure.content.jsonString
            case .attachment, .custom:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [],
                        debugDescription: "MLX currently supports text transcript segments only."
                    )
                )
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [],
                        debugDescription: "The transcript contains an unknown segment type."
                    )
                )
            }
        }
        .joined(separator: "\n")
    }
}
#endif
