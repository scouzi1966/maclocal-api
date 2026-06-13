import Foundation

// MARK: - Public configuration

/// Which model backend an ``AFMEngine`` drives.
public enum AFMBackend: Sendable {
    /// Apple's on-device Foundation Models (macOS 26+).
    case foundationModels
    /// An MLX model from Hugging Face (or local cache), addressed by its id.
    case mlx(modelID: String)
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
    public var mtpEnabled: Bool
    public var mtpDepth: Int
    public var eagle3DrafterPath: String?
    public var enableGrammarConstraints: Bool
    public var toolCallParser: String?
    public var maxConcurrent: Int

    public init(
        instructions: String = "You are a helpful assistant",
        adapter: String? = nil,
        permissiveGuardrails: Bool = false,
        kvBits: Int? = nil,
        enablePrefixCaching: Bool = false,
        mtpEnabled: Bool = false,
        mtpDepth: Int = 3,
        eagle3DrafterPath: String? = nil,
        enableGrammarConstraints: Bool = false,
        toolCallParser: String? = nil,
        maxConcurrent: Int = 0
    ) {
        self.instructions = instructions
        self.adapter = adapter
        self.permissiveGuardrails = permissiveGuardrails
        self.kvBits = kvBits
        self.enablePrefixCaching = enablePrefixCaching
        self.mtpEnabled = mtpEnabled
        self.mtpDepth = mtpDepth
        self.eagle3DrafterPath = eagle3DrafterPath
        self.enableGrammarConstraints = enableGrammarConstraints
        self.toolCallParser = toolCallParser
        self.maxConcurrent = maxConcurrent
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
        responseFormat: ResponseFormat? = nil
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
    }
}

/// A completed generation result.
public struct AFMResponse: Sendable {
    public let content: String
    /// Extracted `<think>…</think>` reasoning, when the model produced any.
    public let reasoningContent: String?
    public let toolCalls: [ResponseToolCall]?
    public let promptTokens: Int
    public let completionTokens: Int
    public init(content: String, reasoningContent: String? = nil, toolCalls: [ResponseToolCall]? = nil, promptTokens: Int = 0, completionTokens: Int = 0) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
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

    private let mlx: MLXModelService?
    private var resolvedModelID: String?

    // Foundation Models backend is created lazily on first use (macOS 26+ only).
    private var foundationService: Any?

    public init(backend: AFMBackend, config: EngineConfig = EngineConfig()) {
        self.backend = backend
        self.engineConfig = config
        switch backend {
        case .mlx:
            let service = MLXModelService(resolver: MLXCacheResolver())
            service.kvBits = config.kvBits
            service.enablePrefixCaching = config.enablePrefixCaching
            service.mtpEnabled = config.mtpEnabled
            service.mtpDepth = config.mtpDepth
            service.eagle3DrafterPath = config.eagle3DrafterPath
            service.enableGrammarConstraints = config.enableGrammarConstraints
            service.toolCallParser = config.toolCallParser
            service.maxConcurrent = config.maxConcurrent
            self.mlx = service
        case .foundationModels:
            self.mlx = nil
        }
    }

    /// Load (download if needed) the model and prepare it for inference.
    /// For the MLX backend this resolves + loads the weights; returns the canonical model id.
    @discardableResult
    public func load(progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        switch backend {
        case .mlx(let modelID):
            guard let mlx else { throw AFMEngineError.backendUnavailable }
            // Point MLX at the bundled default.metallib (the CLI does this at startup;
            // library consumers must too, else MLX reports "Failed to load the default metallib").
            try MLXMetalLibrary.ensureAvailable(verbose: false)
            let resolved = try await mlx.ensureLoaded(model: modelID, progress: { p in
                progress?(p.fractionCompleted)
            })
            resolvedModelID = resolved
            if engineConfig.maxConcurrent >= 2 { try await mlx.initScheduler() }
            return resolved
        case .foundationModels:
            try await ensureFoundation()
            return "apple-foundation-model"
        }
    }

    /// Generate a single (non-streaming) response for a chat transcript.
    public func respond(to messages: [Message], _ config: GenerationConfig = GenerationConfig()) async throws -> AFMResponse {
        switch backend {
        case .mlx(let modelID):
            guard let mlx else { throw AFMEngineError.backendUnavailable }
            let r = try await mlx.generate(
                model: resolvedModelID ?? modelID,
                messages: messages,
                temperature: config.temperature,
                maxTokens: config.maxTokens,
                topP: config.topP,
                repetitionPenalty: config.repetitionPenalty,
                topK: config.topK,
                minP: config.minP,
                presencePenalty: config.presencePenalty,
                seed: config.seed,
                logprobs: config.logprobs,
                topLogprobs: config.topLogprobs,
                tools: config.tools,
                stop: config.stop,
                responseFormat: config.responseFormat,
                chatTemplateKwargs: nil
            )
            return AFMResponse(
                content: r.content,
                toolCalls: r.toolCalls,
                promptTokens: r.promptTokens,
                completionTokens: r.completionTokens
            )
        case .foundationModels:
            let text = try await foundationGenerate(messages: messages, config: config)
            return AFMResponse(content: text)
        }
    }

    /// The canonical model id once loaded, else the requested id (actor-isolated read).
    private func currentModelID(_ fallback: String) -> String { resolvedModelID ?? fallback }

    /// Stream a response token-by-token (text deltas). `nonisolated` so external
    /// callers can start a stream without `await`; the work re-enters the actor.
    public nonisolated func streamRespond(to messages: [Message], _ config: GenerationConfig = GenerationConfig()) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    switch backend {
                    case .mlx(let modelID):
                        guard let mlx else { throw AFMEngineError.backendUnavailable }
                        let resolved = await currentModelID(modelID)
                        let (_, stream, _, _, _, _, _) = try await mlx.generateStreaming(
                            model: resolved,
                            messages: messages,
                            temperature: config.temperature,
                            maxTokens: config.maxTokens,
                            topP: config.topP,
                            repetitionPenalty: config.repetitionPenalty,
                            topK: config.topK,
                            minP: config.minP,
                            presencePenalty: config.presencePenalty,
                            seed: config.seed,
                            logprobs: config.logprobs,
                            topLogprobs: config.topLogprobs,
                            tools: config.tools,
                            stop: config.stop,
                            responseFormat: config.responseFormat,
                            chatTemplateKwargs: nil,
                            requestId: nil
                        )
                        for try await chunk in stream {
                            if Task.isCancelled { break }
                            if !chunk.text.isEmpty { continuation.yield(chunk.text) }
                        }
                        continuation.finish()
                    case .foundationModels:
                        let text = try await foundationGenerate(messages: messages, config: config)
                        continuation.yield(text)
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

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
