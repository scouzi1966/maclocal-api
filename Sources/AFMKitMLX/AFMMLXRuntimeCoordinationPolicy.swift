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

public enum AFMMLXRuntimeCoordinationPolicy {
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
