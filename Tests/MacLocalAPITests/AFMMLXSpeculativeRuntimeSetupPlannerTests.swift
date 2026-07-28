import XCTest
import AFMKitMLX

final class AFMMLXSpeculativeRuntimeSetupPlannerTests: XCTestCase {
    func testOffModeDoesNotAttemptAcceleration() {
        let plan = AFMMLXSpeculativeRuntimeSetupPlanner.make(
            selectedMode: .off,
            modelDirectoryAvailable: true,
            mtpCompatible: true,
            denseGemma4Verifier: true
        )

        XCTAssertEqual(plan.initialState.statusKind, .none)
        XCTAssertEqual(plan.initialState.statusText, "Acceleration off")
        XCTAssertFalse(plan.shouldAttemptMTP)
        XCTAssertFalse(plan.shouldAttemptEagle3)
        XCTAssertFalse(plan.initialState.shouldAskToDownloadEagle3Drafter)
    }

    func testExplicitMTPStopsAfterMissingRuntime() {
        let plan = AFMMLXSpeculativeRuntimeSetupPlanner.make(
            selectedMode: .mtp,
            modelDirectoryAvailable: false,
            mtpCompatible: false,
            denseGemma4Verifier: true
        )

        XCTAssertFalse(plan.shouldAttemptMTP)
        XCTAssertTrue(plan.shouldStopAfterMTPFailure)
        XCTAssertEqual(plan.mtpFailureState?.statusText, "Unavailable: no compatible MTP sidecar")
        XCTAssertEqual(plan.mtpFailureState?.statusKind, AFMMLXSpeculativeRuntimeKind.none)
        XCTAssertFalse(plan.mtpFailureState?.shouldAskToDownloadEagle3Drafter == true)
    }

    func testAutoAttemptsMTPThenEagle3ForDenseGemma4() {
        let plan = AFMMLXSpeculativeRuntimeSetupPlanner.make(
            selectedMode: .auto,
            modelDirectoryAvailable: true,
            mtpCompatible: true,
            denseGemma4Verifier: true
        )

        XCTAssertTrue(plan.shouldAttemptMTP)
        XCTAssertFalse(plan.shouldStopAfterMTPFailure)
        XCTAssertTrue(plan.shouldAttemptEagle3)
        XCTAssertEqual(plan.eagle3FailureState?.statusText, "Unavailable: EAGLE3 drafter missing")
        XCTAssertTrue(plan.eagle3FailureState?.shouldAskToDownloadEagle3Drafter == true)
    }

    func testExplicitEagle3RejectsNonDenseVerifier() {
        let plan = AFMMLXSpeculativeRuntimeSetupPlanner.make(
            selectedMode: .eagle3,
            modelDirectoryAvailable: true,
            mtpCompatible: true,
            denseGemma4Verifier: false
        )

        XCTAssertFalse(plan.shouldAttemptMTP)
        XCTAssertFalse(plan.shouldAttemptEagle3)
        XCTAssertEqual(plan.eagle3FailureState?.statusText, "Unavailable: verifier is not dense Gemma4")
        XCTAssertFalse(plan.eagle3FailureState?.shouldAskToDownloadEagle3Drafter == true)
    }

    func testReadyStatesKeepAvailabilityAndSetRuntimeKind() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: true,
            mtpCompatible: true,
            denseGemma4Verifier: true
        )

        let mtp = AFMMLXSpeculativeRuntimeSetupPlanner.mtpReadyState(availability: availability)
        XCTAssertEqual(mtp.availability, availability)
        XCTAssertEqual(mtp.statusKind, .mtp)
        XCTAssertEqual(mtp.statusText, "MTP ready")

        let eagle3 = AFMMLXSpeculativeRuntimeSetupPlanner.eagle3ReadyState(availability: availability)
        XCTAssertEqual(eagle3.availability, availability)
        XCTAssertEqual(eagle3.statusKind, .eagle3)
        XCTAssertEqual(eagle3.statusText, "EAGLE3 ready")
    }

    func testEagle3DownloadRejectsMissingLoadedModel() {
        let state = AFMMLXEagle3DrafterDownloadPolicy.missingLoadedModelState()

        XCTAssertFalse(state.isDownloading)
        XCTAssertEqual(state.progress, 0)
        XCTAssertEqual(state.statusText, "Load a Gemma4 model before downloading EAGLE3")
        XCTAssertFalse(state.shouldAskToDownloadEagle3Drafter)
    }

    func testEagle3DownloadRejectsNonDenseVerifier() {
        let state = AFMMLXEagle3DrafterDownloadPolicy.nonDenseVerifierState()

        XCTAssertFalse(state.isDownloading)
        XCTAssertEqual(state.progress, 0)
        XCTAssertEqual(state.statusText, "Load a dense Gemma4 model before downloading EAGLE3")
        XCTAssertFalse(state.shouldAskToDownloadEagle3Drafter)
    }

    func testEagle3DownloadStartedAndFinishedStates() {
        let started = AFMMLXEagle3DrafterDownloadPolicy.startedState()
        XCTAssertTrue(started.isDownloading)
        XCTAssertEqual(started.progress, 0)
        XCTAssertEqual(started.statusText, "Downloading EAGLE3 drafter")
        XCTAssertFalse(started.shouldAskToDownloadEagle3Drafter)

        let finished = AFMMLXEagle3DrafterDownloadPolicy.finishedDownloadState()
        XCTAssertFalse(finished.isDownloading)
        XCTAssertEqual(finished.progress, 1)
        XCTAssertEqual(finished.statusText, "EAGLE3 downloaded")
        XCTAssertFalse(finished.shouldAskToDownloadEagle3Drafter)
    }

    func testEagle3DownloadCurrentModelChangedState() {
        let state = AFMMLXEagle3DrafterDownloadPolicy.currentModelChangedState()

        XCTAssertFalse(state.isDownloading)
        XCTAssertEqual(state.progress, 1)
        XCTAssertEqual(state.statusText, "EAGLE3 downloaded; current model changed")
        XCTAssertFalse(state.shouldAskToDownloadEagle3Drafter)
    }

    func testEagle3DownloadFailureKeepsRetryAvailable() {
        let state = AFMMLXEagle3DrafterDownloadPolicy.failedState(
            errorDescription: "network unavailable"
        )

        XCTAssertFalse(state.isDownloading)
        XCTAssertEqual(state.progress, 0)
        XCTAssertEqual(state.statusText, "EAGLE3 download failed: network unavailable")
        XCTAssertTrue(state.shouldAskToDownloadEagle3Drafter)
    }
}
