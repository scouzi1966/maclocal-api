#if canImport(FoundationModels)
import Foundation
import AFMKit
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

        let messages = try MLXFoundationRequestAdapter.messages(from: request.transcript)
        guard !messages.isEmpty else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: Array(request.transcript),
                    debugDescription: "The MLX provider could not convert the transcript to text messages."
                )
            )
        }

        let engine = try await runtime.preparedEngine()
        let options = try MLXFoundationRequestAdapter.generationConfig(from: request, model: model)

        var channelAdapter = MLXFoundationEventChannelAdapter()
        for try await event in engine.streamEvents(to: messages, options) {
            await channelAdapter.send(event, into: channel)
        }
        await channelAdapter.finish(into: channel)
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
