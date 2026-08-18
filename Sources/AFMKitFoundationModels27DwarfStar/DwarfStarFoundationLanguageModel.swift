#if canImport(FoundationModels)
import AFMKit
import AFMKitDwarfStar
import AFMKitFoundationModels27
import Foundation
import FoundationModels

/// A DwarfStar-backed DeepSeek model that plugs directly into the macOS 27
/// Foundation Models `LanguageModelSession` API.
@available(macOS 27.0, *)
public struct DwarfStarLanguageModel:
    LanguageModel,
    AFMFoundationModelsModelConfiguration,
    Sendable
{
    public typealias Executor = DwarfStarLanguageModelExecutor

    public let executorConfiguration: DwarfStarLanguageModelExecutor.Configuration

    public init(
        modelPath: String,
        contextWindow: Int = 32_768,
        prefillChunk: Int = 0,
        powerPercent: Int = 100,
        dsparkSupportPath: String? = nil,
        dsparkDraftTokens: Int = 5,
        dsparkConfidenceThreshold: Double = 0.7,
        dsparkStrict: Bool = false,
        enablePrefixCaching: Bool = true,
        maxConcurrent: Int = 1,
        defaultMaximumResponseTokens: Int = 2_048
    ) {
        executorConfiguration = .init(
            modelPath: modelPath,
            contextWindow: contextWindow,
            prefillChunk: prefillChunk,
            powerPercent: powerPercent,
            dsparkSupportPath: dsparkSupportPath,
            dsparkDraftTokens: dsparkDraftTokens,
            dsparkConfidenceThreshold: dsparkConfidenceThreshold,
            dsparkStrict: dsparkStrict,
            enablePrefixCaching: enablePrefixCaching,
            maxConcurrent: maxConcurrent,
            defaultMaximumResponseTokens: defaultMaximumResponseTokens
        )
    }

    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([.reasoning, .toolCalling])
    }

    public var defaultMaximumResponseTokens: Int {
        executorConfiguration.defaultMaximumResponseTokens
    }

    public var supportsReasoning: Bool { true }
}

@available(macOS 27.0, *)
public final class DwarfStarLanguageModelExecutor:
    LanguageModelExecutor,
    @unchecked Sendable
{
    public typealias Model = DwarfStarLanguageModel

    public struct Configuration:
        Hashable,
        Sendable,
        AFMFoundationModelsModelConfiguration
    {
        public let modelPath: String
        public let contextWindow: Int
        public let prefillChunk: Int
        public let powerPercent: Int
        public let dsparkSupportPath: String?
        public let dsparkDraftTokens: Int
        public let dsparkConfidenceThreshold: Double
        public let dsparkStrict: Bool
        public let enablePrefixCaching: Bool
        public let maxConcurrent: Int
        public let defaultMaximumResponseTokens: Int
        public let supportsReasoning = true

        public init(
            modelPath: String,
            contextWindow: Int,
            prefillChunk: Int,
            powerPercent: Int,
            dsparkSupportPath: String?,
            dsparkDraftTokens: Int,
            dsparkConfidenceThreshold: Double,
            dsparkStrict: Bool,
            enablePrefixCaching: Bool,
            maxConcurrent: Int,
            defaultMaximumResponseTokens: Int
        ) {
            self.modelPath = modelPath
            self.contextWindow = contextWindow
            self.prefillChunk = prefillChunk
            self.powerPercent = powerPercent
            self.dsparkSupportPath = dsparkSupportPath
            self.dsparkDraftTokens = dsparkDraftTokens
            self.dsparkConfidenceThreshold = dsparkConfidenceThreshold
            self.dsparkStrict = dsparkStrict
            self.enablePrefixCaching = enablePrefixCaching
            self.maxConcurrent = maxConcurrent
            self.defaultMaximumResponseTokens = defaultMaximumResponseTokens
        }
    }

    private let runtime: Runtime

    public init(configuration: Configuration) throws {
        runtime = Runtime(
            model: AnyAFMModel(
                AFMDwarfStarModel(
                    modelID: AFMModelID(
                        rawValue: URL(fileURLWithPath: configuration.modelPath)
                            .deletingPathExtension().lastPathComponent
                    ),
                    modelPath: configuration.modelPath,
                    configuration: AFMDwarfStarRuntimeConfiguration(
                        contextWindow: configuration.contextWindow,
                        prefillChunk: configuration.prefillChunk,
                        powerPercent: configuration.powerPercent,
                        dsparkSupportPath: configuration.dsparkSupportPath,
                        dsparkDraftTokens: configuration.dsparkDraftTokens,
                        dsparkConfidenceThreshold: configuration.dsparkConfidenceThreshold,
                        dsparkStrict: configuration.dsparkStrict,
                        enablePrefixCaching: configuration.enablePrefixCaching,
                        maxConcurrent: configuration.maxConcurrent
                    )
                )
            )
        )
    }

    deinit {
        let runtime = runtime
        Task { await runtime.unload() }
    }

    public func prewarm(model: Model, transcript: Transcript) {
        let runtime = runtime
        Task { _ = try? await runtime.preparedModel() }
    }

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: Model,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "DwarfStar guided generation is not available."
                )
            )
        }

        let messages = try AFMFoundationModelsRequestAdapter.messages(
            from: request.transcript
        )
        guard !messages.isEmpty else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: Array(request.transcript),
                    debugDescription: "DwarfStar could not convert the transcript."
                )
            )
        }
        let options = try AFMFoundationModelsRequestAdapter.generationConfig(
            from: request,
            model: model
        )
        let afmRequest = try AFMRequest(
            openAIMessages: messages,
            generationConfig: options
        )
        let providerModel = try await runtime.preparedModel()
        try await AFMFoundationModelsExecutorBridge.respond(
            events: AFMFoundationModelsExecutorBridge.events(
                from: providerModel,
                request: afmRequest
            ),
            streamingInto: channel
        )
    }
}

@available(macOS 27.0, *)
private actor Runtime {
    let model: AnyAFMModel
    private var loadTask: Task<AFMModelDescriptor, Error>?

    init(model: AnyAFMModel) {
        self.model = model
    }

    func preparedModel() async throws -> AnyAFMModel {
        if let loadTask {
            _ = try await loadTask.value
            return model
        }
        let model = model
        let task = Task { try await model.load() }
        loadTask = task
        do {
            _ = try await task.value
            return model
        } catch {
            loadTask = nil
            throw error
        }
    }

    func unload() async {
        loadTask?.cancel()
        loadTask = nil
        await model.unload()
    }
}
#endif
