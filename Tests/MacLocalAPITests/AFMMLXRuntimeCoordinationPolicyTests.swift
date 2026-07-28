import XCTest
@testable import AFMKitMLX

final class AFMMLXRuntimeCoordinationPolicyTests: XCTestCase {
    func testDefaultArgumentsInitializeLegacyRuntime() {
        let policy = AFMMLXRuntimeStartupPolicy.make(arguments: ["afm"])

        XCTAssertTrue(policy.shouldInitializeLegacyRuntime)
    }

    func testIsolationArgumentSkipsLegacyRuntimeInitialization() {
        let policy = AFMMLXRuntimeStartupPolicy.make(
            arguments: ["afm", AFMMLXRuntimeStartupPolicy.isolateLegacyRuntimeArgument]
        )

        XCTAssertFalse(policy.shouldInitializeLegacyRuntime)
    }

    func testReleasePlanSkipsProvidersThatDoNotUseMLX() {
        let plan = AFMMLXRuntimeCoordinationPolicy.releasePlan(
            providerUsesMLX: false,
            legacyRuntimeIsLoaded: true
        )

        XCTAssertEqual(plan.outcome, .skippedProviderDoesNotUseMLX)
        XCTAssertFalse(plan.didReleaseLegacyRuntime)
    }

    func testReleasePlanSkipsUnloadedLegacyRuntime() {
        let plan = AFMMLXRuntimeCoordinationPolicy.releasePlan(
            providerUsesMLX: true,
            legacyRuntimeIsLoaded: false
        )

        XCTAssertEqual(plan.outcome, .skippedLegacyRuntimeNotLoaded)
        XCTAssertFalse(plan.didReleaseLegacyRuntime)
    }

    func testReleasePlanReleasesLoadedMLXRuntimeForMLXProvider() {
        let plan = AFMMLXRuntimeCoordinationPolicy.releasePlan(
            providerUsesMLX: true,
            legacyRuntimeIsLoaded: true
        )

        XCTAssertEqual(plan.outcome, .releasedLegacyRuntime)
        XCTAssertTrue(plan.didReleaseLegacyRuntime)
    }

    func testMissingRuntimePlanKeepsDistinctOutcome() {
        let plan = AFMMLXRuntimeCoordinationPolicy.missingRuntimePlan()

        XCTAssertEqual(plan.outcome, .skippedMissingLegacyRuntime)
        XCTAssertFalse(plan.didReleaseLegacyRuntime)
    }
}
