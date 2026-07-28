import XCTest
@testable import AFMKitMLX

final class AFMMLXRuntimeCoordinationPolicyTests: XCTestCase {
    func testCacheKeySeparatesLLMAndVLMContainers() {
        let llm = AFMMLXRuntimeCoordinationPolicy.cacheKey(
            modelName: "Qwen3-4bit",
            isVision: false
        )
        let vlm = AFMMLXRuntimeCoordinationPolicy.cacheKey(
            modelName: "Qwen3-4bit",
            isVision: true
        )

        XCTAssertEqual(llm.value, "Qwen3-4bit#llm")
        XCTAssertEqual(vlm.value, "Qwen3-4bit#vlm")
        XCTAssertNotEqual(llm, vlm)
    }

    func testCacheKeyTrimsModelName() {
        let key = AFMMLXRuntimeCoordinationPolicy.cacheKey(
            modelName: "  mlx-community/Gemma-4bit  ",
            isVision: false
        )

        XCTAssertEqual(key.modelName, "mlx-community/Gemma-4bit")
        XCTAssertEqual(key.value, "mlx-community/Gemma-4bit#llm")
    }

    func testUnloadedStateResetsRuntimeAndSpeculativeStatus() {
        let state = AFMMLXRuntimeCoordinationPolicy.unloadedState()

        XCTAssertNil(state.loadedModelName)
        XCTAssertFalse(state.isModelLoaded)
        XCTAssertFalse(state.isLoadedModelVLM)
        XCTAssertNil(state.loadedModelRepoID)
        XCTAssertNil(state.loadedModelType)
        XCTAssertFalse(state.loadedModelHasImplicitReasoning)
        XCTAssertFalse(state.supportsThinkingToggle)
        XCTAssertNil(state.modelContextWindow)
        XCTAssertEqual(state.speculativeStatusText, "Acceleration not loaded")
        XCTAssertEqual(state.speculativeStatusKind, .none)
        XCTAssertEqual(state.speculativeModeAvailability, AFMMLXSpeculativeModeAvailability.unloaded)
        XCTAssertEqual(state.lastSpeculativeGenerationPath, .normal)
        XCTAssertFalse(state.shouldAskToDownloadEagle3Drafter)
    }

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
