import Foundation
import AFMKitCore
import AFMOpenAICompat
import AFMKitMLX
import AFMKitFoundationModels
import AFMKitServices

// MARK: - Public configuration

/// Which model backend an ``AFMEngine`` drives.
public enum AFMBackend: Sendable {
    /// Apple's on-device Foundation Models (macOS 26+).
    case foundationModels
    /// An MLX model from Hugging Face (or local cache), addressed by its id.
    case mlx(modelID: String)
    /// A model supplied by an open AFM provider registry.
    case provider(providerID: AFMProviderID, modelID: AFMModelID)
}

/// Engine-level configuration — set once when the engine is created. Mirrors the
/// `afm`/`afm mlx` server flags that configure the runtime (not per-request sampling).
public struct EngineConfig: Sendable {
    // Foundation Models
    public var instructions: String
    public var adapter: String?
    public var permissiveGuardrails: Bool
    // MLX runtime knobs (ignored by the Foundation Models backend)
    public var kvBits: Int?
    public var enablePrefixCaching: Bool
    public var mlxKernels: String
    public var mtpEnabled: Bool
    public var mtpDepth: Int
    public var mtpModelID: String?
    public var eagle3DrafterPath: String?
    public var enableGrammarConstraints: Bool
    public var toolCallParser: String?
    public var maxConcurrent: Int
    public var prefillStepSize: Int?
    public var kvEvictionPolicy: String
    public var fixToolArguments: Bool
    public var forceVLM: Bool
    public var cacheProfilePath: String?
    public var trace: Bool
    public var gpuCapturePath: String?
    public var gpuTraceDuration: Int?
    public var gpuProfile: Bool
    public var gpuProfileBandwidth: Bool

    public init(
        instructions: String = "You are a helpful assistant",
        adapter: String? = nil,
        permissiveGuardrails: Bool = false,
        kvBits: Int? = nil,
        enablePrefixCaching: Bool = false,
        mlxKernels: String = "native",
        mtpEnabled: Bool = false,
        mtpDepth: Int = 3,
        mtpModelID: String? = nil,
        eagle3DrafterPath: String? = nil,
        enableGrammarConstraints: Bool = false,
        toolCallParser: String? = nil,
        maxConcurrent: Int = 0,
        prefillStepSize: Int? = nil,
        kvEvictionPolicy: String = "none",
        fixToolArguments: Bool = false,
        forceVLM: Bool = false,
        cacheProfilePath: String? = nil,
        trace: Bool = false,
        gpuCapturePath: String? = nil,
        gpuTraceDuration: Int? = nil,
        gpuProfile: Bool = false,
        gpuProfileBandwidth: Bool = false
    ) {
        self.instructions = instructions
        self.adapter = adapter
        self.permissiveGuardrails = permissiveGuardrails
        self.kvBits = kvBits
        self.enablePrefixCaching = enablePrefixCaching
        self.mlxKernels = mlxKernels
        self.mtpEnabled = mtpEnabled
        self.mtpDepth = mtpDepth
        self.mtpModelID = mtpModelID
        self.eagle3DrafterPath = eagle3DrafterPath
        self.enableGrammarConstraints = enableGrammarConstraints
        self.toolCallParser = toolCallParser
        self.maxConcurrent = maxConcurrent
        self.prefillStepSize = prefillStepSize
        self.kvEvictionPolicy = kvEvictionPolicy
        self.fixToolArguments = fixToolArguments
        self.forceVLM = forceVLM
        self.cacheProfilePath = cacheProfilePath
        self.trace = trace
        self.gpuCapturePath = gpuCapturePath
        self.gpuTraceDuration = gpuTraceDuration
        self.gpuProfile = gpuProfile
        self.gpuProfileBandwidth = gpuProfileBandwidth
    }
}

private extension EngineConfig {
    var mlxProviderConfiguration: AFMProviderConfiguration {
        var values: [String: AFMJSONValue] = [
            "enablePrefixCaching": .bool(enablePrefixCaching),
            "mlxKernels": .string(mlxKernels),
            "mtpEnabled": .bool(mtpEnabled),
            "mtpDepth": .integer(mtpDepth),
            "enableGrammarConstraints": .bool(enableGrammarConstraints),
            "maxConcurrent": .integer(maxConcurrent),
            "kvEvictionPolicy": .string(kvEvictionPolicy),
            "fixToolArguments": .bool(fixToolArguments),
            "forceVLM": .bool(forceVLM),
            "trace": .bool(trace),
            "gpuProfile": .bool(gpuProfile),
            "gpuProfileBandwidth": .bool(gpuProfileBandwidth)
        ]
        if let kvBits {
            values["kvBits"] = .integer(kvBits)
        }
        if let eagle3DrafterPath {
            values["eagle3DrafterPath"] = .string(eagle3DrafterPath)
        }
        if let mtpModelID {
            values["mtpModelID"] = .string(mtpModelID)
        }
        if let toolCallParser {
            values["toolCallParser"] = .string(toolCallParser)
        }
        if let prefillStepSize {
            values["prefillStepSize"] = .integer(prefillStepSize)
        }
        if let cacheProfilePath {
            values["cacheProfilePath"] = .string(cacheProfilePath)
        }
        if let gpuCapturePath {
            values["gpuCapturePath"] = .string(gpuCapturePath)
        }
        if let gpuTraceDuration {
            values["gpuTraceDuration"] = .integer(gpuTraceDuration)
        }
        return AFMProviderConfiguration(values: values)
    }
}

/// Per-request generation parameters — the same knobs exposed as CLI flags
/// (`--temperature`, `--top-p`, …) and OpenAI request fields, as a value type.
public struct GenerationConfig: Sendable {
    public var temperature: Double?
    public var maxTokens: Int?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var seed: Int?
    public var logprobs: Bool?
    public var topLogprobs: Int?
    public var stop: [String]?
    public var tools: [RequestTool]?
    public var responseFormat: ResponseFormat?
    public var metadata: [String: AFMJSONValue]

    public init(
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        stop: [String]? = nil,
        tools: [RequestTool]? = nil,
        responseFormat: ResponseFormat? = nil,
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.stop = stop
        self.tools = tools
        self.responseFormat = responseFormat
        self.metadata = metadata
    }
}

/// A completed generation result.
public struct AFMResponse: Sendable {
    public let content: String
    /// Extracted `<think>…</think>` reasoning, when the model produced any.
    public let reasoningContent: String?
    public let toolCalls: [ResponseToolCall]?
    /// Per-token logprobs, when `GenerationConfig.logprobs == true` (MLX backend).
    public let logprobs: [AFMTokenLogProbability]?
    public let promptTokens: Int
    public let cachedPromptTokens: Int
    public let completionTokens: Int
    public let finishReason: AFMFinishReason
    public let metadata: [String: AFMJSONValue]
    public init(
        content: String,
        reasoningContent: String? = nil,
        toolCalls: [ResponseToolCall]? = nil,
        logprobs: [AFMTokenLogProbability]? = nil,
        promptTokens: Int = 0,
        cachedPromptTokens: Int = 0,
        completionTokens: Int = 0,
        finishReason: AFMFinishReason = .stop,
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.logprobs = logprobs
        self.promptTokens = promptTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.completionTokens = completionTokens
        self.finishReason = finishReason
        self.metadata = metadata
    }
}

/// A streaming response event with the usage information needed by framework
/// adapters and API clients.
public enum AFMStreamEvent: Sendable {
    case text(String, tokenCount: Int)
    case reasoning(String, tokenCount: Int)
    case tokenLogprobs([AFMTokenLogProbability])
    case toolCall(AFMToolCall, stage: AFMToolCallStage)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int)
    case metadata([String: AFMJSONValue])
    case custom(type: String, payload: Data)
    case completed(AFMFinishReason)
}

// MARK: - AFMEngine

/// A headless, embeddable entry point to afm's inference backends.
///
/// `AFMEngine` is the programmatic equivalent of the `afm` CLI: construct it with a
/// backend + ``EngineConfig`` (the server-level flags), then call ``respond(to:_:)`` /
/// ``streamRespond(to:_:)`` with a ``GenerationConfig`` (the per-request sampling flags).
///
/// ```swift
/// let engine = try await AFMEngine(backend: .mlx(modelID: "mlx-community/Qwen3-4B-MLX-4bit"))
/// _ = try await engine.load()
/// let reply = try await engine.respond(to: [Message(role: "user", content: "Hello!")])
/// print(reply.content)
/// ```
public actor AFMEngine {
    public let backend: AFMBackend
    private let engineConfig: EngineConfig

    private let registeredModel: AnyAFMModel?

    // Foundation Models backend is created lazily on first use (macOS 26+ only).
    private var foundationService: Any?

    public init(backend: AFMBackend, config: EngineConfig = EngineConfig()) {
        self.backend = backend
        self.engineConfig = config
        switch backend {
        case .mlx(let modelID):
            self.registeredModel = try? AFMMLXProviderFactory().makeModel(
                id: AFMModelID(rawValue: modelID),
                configuration: config.mlxProviderConfiguration
            )
        case .foundationModels:
            self.registeredModel = nil
        case .provider(let providerID, let modelID):
            self.registeredModel = try? AFMProviderRegistry.shared.makeModel(
                providerID: providerID,
                modelID: modelID
            )
        }
    }

    /// Construct an engine from a provider registered by an application or package.
    ///
    /// This is the extensible entry point. Adding a provider requires registering a
    /// factory, not adding a new ``AFMBackend`` case or editing ``AFMEngine``.
    public init(
        providerID: AFMProviderID,
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration = .init(),
        engineConfig: EngineConfig = .init(),
        registry: AFMProviderRegistry = .shared
    ) throws {
        backend = .provider(providerID: providerID, modelID: modelID)
        self.engineConfig = engineConfig
        registeredModel = try registry.makeModel(
            providerID: providerID,
            modelID: modelID,
            configuration: configuration
        )
    }

    /// Load (download if needed) the model and prepare it for inference.
    /// For the MLX backend this resolves + loads the weights; returns the canonical model id.
    @discardableResult
    public func load(progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        switch backend {
        case .mlx, .provider:
            guard let registeredModel else {
                throw AFMEngineError.backendUnavailable
            }
            return try await registeredModel.load(progress: progress).modelID.rawValue
        case .foundationModels:
            try await ensureFoundation()
            return "apple-foundation-model"
        }
    }

    /// Release backend resources held by this engine.
    public func unload() async {
        if let registeredModel {
            await registeredModel.unload()
        }
        foundationService = nil
    }

    /// Generate a single (non-streaming) response for a chat transcript.
    public func respond(to messages: [Message], _ config: GenerationConfig = GenerationConfig()) async throws -> AFMResponse {
        switch backend {
        case .mlx, .provider:
            guard let registeredModel else {
                throw AFMEngineError.backendUnavailable
            }
            let request = try AFMRequest(
                openAIMessages: messages,
                generationConfig: config
            )
            return AFMResponse(
                modelResponse: try await registeredModel.respond(to: request)
            )
        case .foundationModels:
            let text = try await foundationGenerate(messages: messages, config: config)
            return AFMResponse(content: text)
        }
    }

    /// Stream response deltas and final token usage. `nonisolated` so external
    /// callers can start a stream without `await`; the work re-enters the actor.
    public nonisolated func streamEvents(
        to messages: [Message],
        _ config: GenerationConfig = GenerationConfig()
    ) -> AsyncThrowingStream<AFMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    switch backend {
                    case .mlx, .provider:
                        guard let registeredModel else {
                            throw AFMEngineError.backendUnavailable
                        }
                        let request = try AFMRequest(
                            openAIMessages: messages,
                            generationConfig: config
                        )
                        for try await event in registeredModel.streamResponse(to: request) {
                            if Task.isCancelled { break }
                            continuation.yield(Self.streamEvent(from: event))
                        }
                        continuation.finish()
                    case .foundationModels:
                        let text = try await foundationGenerate(messages: messages, config: config)
                        continuation.yield(.text(text, tokenCount: 0))
                        continuation.yield(.completed(.stop))
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Stream response text deltas. This preserves the original source-compatible
    /// API while ``streamEvents(to:_:)`` carries richer metadata.
    public nonisolated func streamRespond(
        to messages: [Message],
        _ config: GenerationConfig = GenerationConfig()
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in streamEvents(to: messages, config) {
                        if case .text(let text, _) = event, !text.isEmpty {
                            continuation.yield(text)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private nonisolated static func streamEvent(
        from event: AFMGenerationEvent
    ) -> AFMStreamEvent {
        switch event {
        case .responseText(_, let text, let tokenCount):
            return .text(text, tokenCount: tokenCount)
        case .reasoningText(_, let text, let tokenCount):
            return .reasoning(text, tokenCount: tokenCount)
        case .tokenLogprobs(let values):
            return .tokenLogprobs(values)
        case .toolCall(let call, let stage):
            return .toolCall(call, stage: stage)
        case .usage(let usage):
            return .usage(
                promptTokens: usage.inputTokens,
                completionTokens: usage.outputTokens,
                cachedTokens: usage.cachedInputTokens
            )
        case .metadata(let metadata):
            return .metadata(metadata)
        case .custom(let type, let payload):
            return .custom(type: type, payload: payload)
        case .completed(let reason):
            return .completed(reason)
        }
    }

    // MARK: - Batch / concurrent generation

    /// Generate responses for several chat transcripts concurrently, bounded by
    /// `EngineConfig.maxConcurrent` (set ≥2 to engage the MLX batch scheduler). Results
    /// are returned in the same order as `transcripts`.
    public nonisolated func respondBatch(_ transcripts: [[Message]], _ config: GenerationConfig = GenerationConfig()) async throws -> [AFMResponse] {
        if transcripts.isEmpty { return [] }
        let cap = max(1, engineConfig.maxConcurrent)
        var results = [AFMResponse?](repeating: nil, count: transcripts.count)
        try await withThrowingTaskGroup(of: (Int, AFMResponse).self) { group in
            var next = 0
            var inFlight = 0
            func submit() {
                let i = next; next += 1; inFlight += 1
                group.addTask { (i, try await self.respond(to: transcripts[i], config)) }
            }
            while next < transcripts.count && inFlight < cap { submit() }
            while inFlight > 0 {
                let (idx, resp) = try await group.next()!
                results[idx] = resp
                inFlight -= 1
                if next < transcripts.count { submit() }
            }
        }
        return results.compactMap { $0 }
    }

    // MARK: - Auxiliary capabilities (Apple-native, backend-independent)

    /// Apple Vision OCR / table / barcode / classification / saliency service.
    public nonisolated func vision() -> VisionService { VisionService() }
    /// Apple Speech transcription service.
    public nonisolated func speech() -> SpeechService { SpeechService() }
    /// Apple text-to-speech synthesis service.
    public nonisolated func speechSynthesis() -> SpeechSynthesisService { SpeechSynthesisService() }
    /// Default Apple NaturalLanguage embeddings resolver (lazy, model loaded on first use).
    public nonisolated func embeddings() -> any EmbeddingBackendResolver { LazyAppleEmbeddingResolver() }

    // MARK: - Foundation Models bridge (macOS 26+)

    private func ensureFoundation() async throws {
        if #available(macOS 26.0, *) {
            if foundationService == nil {
                foundationService = try await FoundationModelService(
                    instructions: engineConfig.instructions,
                    adapter: engineConfig.adapter,
                    temperature: nil,
                    randomness: nil,
                    permissiveGuardrails: engineConfig.permissiveGuardrails
                )
            }
        } else {
            throw AFMEngineError.foundationModelsUnavailable
        }
    }

    private func foundationGenerate(messages: [Message], config: GenerationConfig) async throws -> String {
        try await ensureFoundation()
        if #available(macOS 26.0, *) {
            guard let svc = foundationService as? FoundationModelService else {
                throw AFMEngineError.foundationModelsUnavailable
            }
            return try await svc.generateResponse(
                for: messages,
                temperature: config.temperature,
                randomness: nil,
                maxTokens: config.maxTokens,
                stop: config.stop
            )
        }
        throw AFMEngineError.foundationModelsUnavailable
    }
}

public enum AFMEngineError: Error, LocalizedError {
    case backendUnavailable
    case foundationModelsUnavailable

    public var errorDescription: String? {
        switch self {
        case .backendUnavailable: return "The requested AFM backend is not available."
        case .foundationModelsUnavailable: return "Apple Foundation Models require macOS 26 or later."
        }
    }
}
