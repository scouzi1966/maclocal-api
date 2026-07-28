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
}
