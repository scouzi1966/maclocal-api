import Foundation

public struct AFMMLXSpeculativeRuntimeSetupState: Equatable, Sendable {
    public let availability: [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability]
    public let statusKind: AFMMLXSpeculativeRuntimeKind
    public let statusText: String
    public let shouldAskToDownloadEagle3Drafter: Bool

    public init(
        availability: [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability],
        statusKind: AFMMLXSpeculativeRuntimeKind,
        statusText: String,
        shouldAskToDownloadEagle3Drafter: Bool
    ) {
        self.availability = availability
        self.statusKind = statusKind
        self.statusText = statusText
        self.shouldAskToDownloadEagle3Drafter = shouldAskToDownloadEagle3Drafter
    }
}

public struct AFMMLXSpeculativeRuntimeSetupPlan: Equatable, Sendable {
    public let initialState: AFMMLXSpeculativeRuntimeSetupState
    public let shouldAttemptMTP: Bool
    public let shouldStopAfterMTPFailure: Bool
    public let shouldAttemptEagle3: Bool
    public let mtpFailureState: AFMMLXSpeculativeRuntimeSetupState?
    public let eagle3FailureState: AFMMLXSpeculativeRuntimeSetupState?

    public init(
        initialState: AFMMLXSpeculativeRuntimeSetupState,
        shouldAttemptMTP: Bool,
        shouldStopAfterMTPFailure: Bool,
        shouldAttemptEagle3: Bool,
        mtpFailureState: AFMMLXSpeculativeRuntimeSetupState?,
        eagle3FailureState: AFMMLXSpeculativeRuntimeSetupState?
    ) {
        self.initialState = initialState
        self.shouldAttemptMTP = shouldAttemptMTP
        self.shouldStopAfterMTPFailure = shouldStopAfterMTPFailure
        self.shouldAttemptEagle3 = shouldAttemptEagle3
        self.mtpFailureState = mtpFailureState
        self.eagle3FailureState = eagle3FailureState
    }
}

public struct AFMMLXEagle3DrafterDownloadState: Equatable, Sendable {
    public let isDownloading: Bool
    public let progress: Double
    public let statusText: String
    public let shouldAskToDownloadEagle3Drafter: Bool

    public init(
        isDownloading: Bool,
        progress: Double,
        statusText: String,
        shouldAskToDownloadEagle3Drafter: Bool
    ) {
        self.isDownloading = isDownloading
        self.progress = progress
        self.statusText = statusText
        self.shouldAskToDownloadEagle3Drafter = shouldAskToDownloadEagle3Drafter
    }
}

public enum AFMMLXEagle3DrafterDownloadPolicy {
    public static func missingLoadedModelState() -> AFMMLXEagle3DrafterDownloadState {
        AFMMLXEagle3DrafterDownloadState(
            isDownloading: false,
            progress: 0,
            statusText: "Load a Gemma4 model before downloading EAGLE3",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func nonDenseVerifierState() -> AFMMLXEagle3DrafterDownloadState {
        AFMMLXEagle3DrafterDownloadState(
            isDownloading: false,
            progress: 0,
            statusText: "Load a dense Gemma4 model before downloading EAGLE3",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func startedState() -> AFMMLXEagle3DrafterDownloadState {
        AFMMLXEagle3DrafterDownloadState(
            isDownloading: true,
            progress: 0,
            statusText: "Downloading EAGLE3 drafter",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func finishedDownloadState() -> AFMMLXEagle3DrafterDownloadState {
        AFMMLXEagle3DrafterDownloadState(
            isDownloading: false,
            progress: 1,
            statusText: "EAGLE3 downloaded",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func currentModelChangedState() -> AFMMLXEagle3DrafterDownloadState {
        AFMMLXEagle3DrafterDownloadState(
            isDownloading: false,
            progress: 1,
            statusText: "EAGLE3 downloaded; current model changed",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func failedState(errorDescription: String) -> AFMMLXEagle3DrafterDownloadState {
        AFMMLXEagle3DrafterDownloadState(
            isDownloading: false,
            progress: 0,
            statusText: "EAGLE3 download failed: \(errorDescription)",
            shouldAskToDownloadEagle3Drafter: true
        )
    }
}

public enum AFMMLXSpeculativeRuntimeSetupPlanner {
    public static func unloadedState(
        selectedMode: AFMMLXSpeculativeDecodingMode
    ) -> AFMMLXSpeculativeRuntimeSetupState {
        AFMMLXSpeculativeRuntimeSetupState(
            availability: AFMMLXSpeculativeModeAvailability.unloaded,
            statusKind: .none,
            statusText: selectedMode == .off ? "Acceleration off" : "Acceleration not loaded",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func make(
        selectedMode: AFMMLXSpeculativeDecodingMode,
        modelDirectoryAvailable: Bool,
        mtpCompatible: Bool,
        denseGemma4Verifier: Bool
    ) -> AFMMLXSpeculativeRuntimeSetupPlan {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: true,
            mtpCompatible: mtpCompatible,
            denseGemma4Verifier: denseGemma4Verifier
        )

        let unavailable = AFMMLXSpeculativeRuntimeSetupState(
            availability: availability,
            statusKind: .none,
            statusText: "Acceleration unavailable",
            shouldAskToDownloadEagle3Drafter: false
        )

        if selectedMode == .off {
            return AFMMLXSpeculativeRuntimeSetupPlan(
                initialState: AFMMLXSpeculativeRuntimeSetupState(
                    availability: availability,
                    statusKind: .none,
                    statusText: "Acceleration off",
                    shouldAskToDownloadEagle3Drafter: false
                ),
                shouldAttemptMTP: false,
                shouldStopAfterMTPFailure: false,
                shouldAttemptEagle3: false,
                mtpFailureState: nil,
                eagle3FailureState: nil
            )
        }

        let shouldAttemptMTP = modelDirectoryAvailable
            && (selectedMode == .auto || selectedMode == .mtp)
        let shouldAttemptEagle3 = denseGemma4Verifier
            && (selectedMode == .auto || selectedMode == .eagle3)

        let mtpFailureState: AFMMLXSpeculativeRuntimeSetupState? = selectedMode == .mtp
            ? AFMMLXSpeculativeRuntimeSetupState(
                availability: availability,
                statusKind: .none,
                statusText: "Unavailable: no compatible MTP sidecar",
                shouldAskToDownloadEagle3Drafter: false
            )
            : nil

        let eagle3FailureState: AFMMLXSpeculativeRuntimeSetupState?
        if selectedMode == .auto || selectedMode == .eagle3 {
            if denseGemma4Verifier {
                eagle3FailureState = AFMMLXSpeculativeRuntimeSetupState(
                    availability: availability,
                    statusKind: .none,
                    statusText: "Unavailable: EAGLE3 drafter missing",
                    shouldAskToDownloadEagle3Drafter: true
                )
            } else if selectedMode == .eagle3 {
                eagle3FailureState = AFMMLXSpeculativeRuntimeSetupState(
                    availability: availability,
                    statusKind: .none,
                    statusText: "Unavailable: verifier is not dense Gemma4",
                    shouldAskToDownloadEagle3Drafter: false
                )
            } else {
                eagle3FailureState = nil
            }
        } else {
            eagle3FailureState = nil
        }

        return AFMMLXSpeculativeRuntimeSetupPlan(
            initialState: unavailable,
            shouldAttemptMTP: shouldAttemptMTP,
            shouldStopAfterMTPFailure: selectedMode == .mtp,
            shouldAttemptEagle3: shouldAttemptEagle3,
            mtpFailureState: mtpFailureState,
            eagle3FailureState: eagle3FailureState
        )
    }

    public static func mtpReadyState(
        availability: [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability]
    ) -> AFMMLXSpeculativeRuntimeSetupState {
        AFMMLXSpeculativeRuntimeSetupState(
            availability: availability,
            statusKind: .mtp,
            statusText: "MTP ready",
            shouldAskToDownloadEagle3Drafter: false
        )
    }

    public static func eagle3ReadyState(
        availability: [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability]
    ) -> AFMMLXSpeculativeRuntimeSetupState {
        AFMMLXSpeculativeRuntimeSetupState(
            availability: availability,
            statusKind: .eagle3,
            statusText: "EAGLE3 ready",
            shouldAskToDownloadEagle3Drafter: false
        )
    }
}
