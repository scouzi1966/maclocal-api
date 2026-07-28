import Foundation

public struct AFMMLXRuntimeStartupPolicy: Equatable, Sendable {
    public static let isolateLegacyRuntimeArgument = "--afm27-mlx-isolate-legacy-runtime"

    public let shouldInitializeLegacyRuntime: Bool

    public init(shouldInitializeLegacyRuntime: Bool) {
        self.shouldInitializeLegacyRuntime = shouldInitializeLegacyRuntime
    }

    public static func make(
        arguments: [String] = ProcessInfo.processInfo.arguments,
        isolateLegacyRuntimeArgument: String = Self.isolateLegacyRuntimeArgument
    ) -> AFMMLXRuntimeStartupPolicy {
        AFMMLXRuntimeStartupPolicy(
            shouldInitializeLegacyRuntime: !arguments.contains(isolateLegacyRuntimeArgument)
        )
    }
}

public enum AFMMLXLegacyRuntimeReleaseOutcome: Equatable, Sendable {
    case releasedLegacyRuntime
    case skippedMissingLegacyRuntime
    case skippedProviderDoesNotUseMLX
    case skippedLegacyRuntimeNotLoaded
}

public struct AFMMLXLegacyRuntimeReleasePlan: Equatable, Sendable {
    public let outcome: AFMMLXLegacyRuntimeReleaseOutcome

    public init(outcome: AFMMLXLegacyRuntimeReleaseOutcome) {
        self.outcome = outcome
    }

    public var didReleaseLegacyRuntime: Bool {
        outcome == .releasedLegacyRuntime
    }
}

public struct AFMMLXRuntimeCacheKey: Equatable, Sendable {
    public let modelName: String
    public let isVision: Bool
    public let value: String

    public init(modelName: String, isVision: Bool) {
        let trimmedModelName = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        self.modelName = trimmedModelName
        self.isVision = isVision
        self.value = "\(trimmedModelName)#\(isVision ? "vlm" : "llm")"
    }
}

public struct AFMMLXRuntimeUnloadState: Equatable, Sendable {
    public let loadedModelName: String?
    public let isModelLoaded: Bool
    public let isLoadedModelVLM: Bool
    public let loadedModelRepoID: String?
    public let loadedModelType: String?
    public let loadedModelHasImplicitReasoning: Bool
    public let supportsThinkingToggle: Bool
    public let modelContextWindow: Int?
    public let speculativeStatusText: String
    public let speculativeStatusKind: AFMMLXSpeculativeRuntimeKind
    public let speculativeModeAvailability: [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability]
    public let lastSpeculativeGenerationPath: AFMMLXSpeculativeGenerationPath
    public let shouldAskToDownloadEagle3Drafter: Bool

    public init(
        loadedModelName: String?,
        isModelLoaded: Bool,
        isLoadedModelVLM: Bool,
        loadedModelRepoID: String?,
        loadedModelType: String?,
        loadedModelHasImplicitReasoning: Bool,
        supportsThinkingToggle: Bool,
        modelContextWindow: Int?,
        speculativeStatusText: String,
        speculativeStatusKind: AFMMLXSpeculativeRuntimeKind,
        speculativeModeAvailability: [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability],
        lastSpeculativeGenerationPath: AFMMLXSpeculativeGenerationPath,
        shouldAskToDownloadEagle3Drafter: Bool
    ) {
        self.loadedModelName = loadedModelName
        self.isModelLoaded = isModelLoaded
        self.isLoadedModelVLM = isLoadedModelVLM
        self.loadedModelRepoID = loadedModelRepoID
        self.loadedModelType = loadedModelType
        self.loadedModelHasImplicitReasoning = loadedModelHasImplicitReasoning
        self.supportsThinkingToggle = supportsThinkingToggle
        self.modelContextWindow = modelContextWindow
        self.speculativeStatusText = speculativeStatusText
        self.speculativeStatusKind = speculativeStatusKind
        self.speculativeModeAvailability = speculativeModeAvailability
        self.lastSpeculativeGenerationPath = lastSpeculativeGenerationPath
        self.shouldAskToDownloadEagle3Drafter = shouldAskToDownloadEagle3Drafter
    }
}

public struct AFMMLXRuntimeLoadingState: Equatable, Sendable {
    public let isLoadingModel: Bool
    public let loadingModelName: String?
    public let downloadProgress: Double
    public let lastReportedProgress: Double
    public let errorMessage: String?

    public init(
        isLoadingModel: Bool,
        loadingModelName: String?,
        downloadProgress: Double,
        lastReportedProgress: Double,
        errorMessage: String?
    ) {
        self.isLoadingModel = isLoadingModel
        self.loadingModelName = loadingModelName
        self.downloadProgress = downloadProgress
        self.lastReportedProgress = lastReportedProgress
        self.errorMessage = errorMessage
    }
}

public struct AFMMLXRuntimeCancellationState: Equatable, Sendable {
    public let shouldCancelLoading: Bool
    public let loadingState: AFMMLXRuntimeLoadingState

    public init(
        shouldCancelLoading: Bool,
        loadingState: AFMMLXRuntimeLoadingState
    ) {
        self.shouldCancelLoading = shouldCancelLoading
        self.loadingState = loadingState
    }
}

public struct AFMMLXRuntimeLoadedState: Equatable, Sendable {
    public let loadedModelName: String
    public let loadedModelRepoID: String
    public let isModelLoaded: Bool
    public let isLoadedModelVLM: Bool
    public let isLoadingModel: Bool
    public let loadingModelName: String?
    public let isDownloadingPhase: Bool
    public let downloadProgress: Double

    public init(
        loadedModelName: String,
        loadedModelRepoID: String,
        isModelLoaded: Bool,
        isLoadedModelVLM: Bool,
        isLoadingModel: Bool,
        loadingModelName: String?,
        isDownloadingPhase: Bool,
        downloadProgress: Double
    ) {
        self.loadedModelName = loadedModelName
        self.loadedModelRepoID = loadedModelRepoID
        self.isModelLoaded = isModelLoaded
        self.isLoadedModelVLM = isLoadedModelVLM
        self.isLoadingModel = isLoadingModel
        self.loadingModelName = loadingModelName
        self.isDownloadingPhase = isDownloadingPhase
        self.downloadProgress = downloadProgress
    }
}

public struct AFMMLXRuntimeProgressState: Equatable, Sendable {
    public let shouldPublish: Bool
    public let downloadProgress: Double
    public let lastReportedProgress: Double

    public init(
        shouldPublish: Bool,
        downloadProgress: Double,
        lastReportedProgress: Double
    ) {
        self.shouldPublish = shouldPublish
        self.downloadProgress = downloadProgress
        self.lastReportedProgress = lastReportedProgress
    }
}

public enum AFMMLXRuntimeCoordinationPolicy {
    public nonisolated static let defaultProgressPublicationThreshold = 0.01
    public nonisolated static let defaultCompletionProgressThreshold = 0.99
    public nonisolated static let defaultCancellationMessage = "Model download cancelled by user"

    public nonisolated static func cacheKey(
        modelName: String,
        isVision: Bool
    ) -> AFMMLXRuntimeCacheKey {
        AFMMLXRuntimeCacheKey(modelName: modelName, isVision: isVision)
    }

    public nonisolated static func unloadedState() -> AFMMLXRuntimeUnloadState {
        AFMMLXRuntimeUnloadState(
            loadedModelName: nil,
            isModelLoaded: false,
            isLoadedModelVLM: false,
            loadedModelRepoID: nil,
            loadedModelType: nil,
            loadedModelHasImplicitReasoning: false,
            supportsThinkingToggle: false,
            modelContextWindow: nil,
            speculativeStatusText: "Acceleration not loaded",
            speculativeStatusKind: .none,
            speculativeModeAvailability: AFMMLXSpeculativeModeAvailability.unloaded,
            lastSpeculativeGenerationPath: .normal,
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public nonisolated static func loadingState(modelName: String) -> AFMMLXRuntimeLoadingState {
        AFMMLXRuntimeLoadingState(
            isLoadingModel: true,
            loadingModelName: modelName.trimmingCharacters(in: .whitespacesAndNewlines),
            downloadProgress: 0,
            lastReportedProgress: 0,
            errorMessage: nil
        )
    }

    public nonisolated static func failedLoadingState(errorMessage: String) -> AFMMLXRuntimeLoadingState {
        AFMMLXRuntimeLoadingState(
            isLoadingModel: false,
            loadingModelName: nil,
            downloadProgress: 0,
            lastReportedProgress: 0,
            errorMessage: errorMessage
        )
    }

    public nonisolated static func failedLoadingState(
        localizedDescription: String,
        diagnosticDescription: String
    ) -> AFMMLXRuntimeLoadingState {
        let resolvedErrorMessage: String
        if diagnosticDescription.contains("unsupportedModelType") {
            if let match = diagnosticDescription.range(
                of: #"unsupportedModelType\(\"?([^")\]]+)\"?\)"#,
                options: .regularExpression
            ) {
                let typeDescription = String(diagnosticDescription[match])
                resolvedErrorMessage = "Architecture not supported in MLX-Swift: \(typeDescription). This model works in Python mlx-lm but the Swift implementation doesn't support it yet."
            } else {
                resolvedErrorMessage = "Model architecture not supported in MLX-Swift. This model may work in Python mlx-lm."
            }
        } else {
            resolvedErrorMessage = "Failed to load model: \(localizedDescription)"
        }
        return failedLoadingState(errorMessage: resolvedErrorMessage)
    }

    public nonisolated static func cancelledLoadingState(
        errorMessage: String = Self.defaultCancellationMessage
    ) -> AFMMLXRuntimeCancellationState {
        AFMMLXRuntimeCancellationState(
            shouldCancelLoading: true,
            loadingState: AFMMLXRuntimeLoadingState(
                isLoadingModel: false,
                loadingModelName: nil,
                downloadProgress: 0,
                lastReportedProgress: 0,
                errorMessage: errorMessage
            )
        )
    }

    public nonisolated static func loadedState(
        modelName: String,
        repoID: String,
        isVision: Bool
    ) -> AFMMLXRuntimeLoadedState {
        AFMMLXRuntimeLoadedState(
            loadedModelName: modelName,
            loadedModelRepoID: repoID,
            isModelLoaded: true,
            isLoadedModelVLM: isVision,
            isLoadingModel: false,
            loadingModelName: nil,
            isDownloadingPhase: false,
            downloadProgress: 1
        )
    }

    public nonisolated static func progressState(
        newProgress: Double,
        lastReportedProgress: Double,
        publicationThreshold: Double = Self.defaultProgressPublicationThreshold,
        completionThreshold: Double = Self.defaultCompletionProgressThreshold
    ) -> AFMMLXRuntimeProgressState {
        let shouldPublish = newProgress - lastReportedProgress >= publicationThreshold
            || newProgress >= completionThreshold
        return AFMMLXRuntimeProgressState(
            shouldPublish: shouldPublish,
            downloadProgress: shouldPublish ? newProgress : lastReportedProgress,
            lastReportedProgress: shouldPublish ? newProgress : lastReportedProgress
        )
    }

    public nonisolated static func shouldReleaseLegacyRuntime(
        providerUsesMLX: Bool,
        legacyRuntimeIsLoaded: Bool
    ) -> Bool {
        providerUsesMLX && legacyRuntimeIsLoaded
    }

    public nonisolated static func releasePlan(
        providerUsesMLX: Bool,
        legacyRuntimeIsLoaded: Bool
    ) -> AFMMLXLegacyRuntimeReleasePlan {
        guard providerUsesMLX else {
            return AFMMLXLegacyRuntimeReleasePlan(outcome: .skippedProviderDoesNotUseMLX)
        }
        guard legacyRuntimeIsLoaded else {
            return AFMMLXLegacyRuntimeReleasePlan(outcome: .skippedLegacyRuntimeNotLoaded)
        }
        return AFMMLXLegacyRuntimeReleasePlan(outcome: .releasedLegacyRuntime)
    }

    public nonisolated static func missingRuntimePlan() -> AFMMLXLegacyRuntimeReleasePlan {
        AFMMLXLegacyRuntimeReleasePlan(outcome: .skippedMissingLegacyRuntime)
    }
}
