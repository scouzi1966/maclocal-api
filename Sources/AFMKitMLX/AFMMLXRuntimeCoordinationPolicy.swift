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

public enum AFMMLXRuntimeCoordinationPolicy {
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
