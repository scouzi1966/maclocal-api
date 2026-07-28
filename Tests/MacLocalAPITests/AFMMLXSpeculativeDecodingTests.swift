import AFMKitMLX
import XCTest

final class AFMMLXSpeculativeDecodingTests: XCTestCase {
    func testModeLabelsAreStableForUI() {
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.off.displayName, "Off")
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.auto.displayName, "Auto")
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.mtp.displayName, "MTP")
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.eagle3.displayName, "EAGLE3")
    }

    func testAutoModeFallsBackWhenSamplingIsEnabled() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0.7,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .samplingEnabled)
    }

    func testAutoModeUsesMTPWhenGreedyAndRuntimeIsReady() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .mtp)
        XCTAssertNil(decision.reason)
    }

    func testCompletedSpeculativeRuntimeFallsBackWhenNoChunksWereEmitted() {
        let decision = AFMMLXSpeculativeGenerationDecision(path: .mtp, reason: nil)
        let completed = AFMMLXSpeculativeGenerationDecision.completedRuntimeDecision(
            initialDecision: decision,
            emittedChunkCount: 0
        )

        XCTAssertEqual(completed.path, .fallback)
        XCTAssertEqual(completed.reason, .runtimeUnavailable)
    }

    func testCompletedSpeculativeRuntimeKeepsSuccessfulPathWhenChunksWereEmitted() {
        let decision = AFMMLXSpeculativeGenerationDecision(path: .eagle3, reason: nil)
        let completed = AFMMLXSpeculativeGenerationDecision.completedRuntimeDecision(
            initialDecision: decision,
            emittedChunkCount: 1
        )

        XCTAssertEqual(completed, decision)
    }

    func testCompletedNonSpeculativeDecisionIsUnchanged() {
        let decision = AFMMLXSpeculativeGenerationDecision(
            path: .fallback,
            reason: .samplingEnabled
        )
        let completed = AFMMLXSpeculativeGenerationDecision.completedRuntimeDecision(
            initialDecision: decision,
            emittedChunkCount: 0
        )

        XCTAssertEqual(completed, decision)
    }

    func testExplicitEagle3FallsBackWhenVisionInputIsPresent() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .eagle3,
            installedRuntime: .eagle3,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: true,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .visionInput)
    }

    func testExplicitModeReportsUnavailableRuntime() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .mtp,
            installedRuntime: .none,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .runtimeUnavailable)
    }

    func testExplicitEagle3DoesNotUseMTPRuntime() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .eagle3,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .runtimeUnavailable)
    }

    func testGreedyModeFallsBackWhenGenerationModifiersAreEnabled() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: true,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .generationModifiers)
    }

    func testGreedyModeFallsBackWhenReasoningOutputIsEnabled() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: true,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .reasoningOutput)
    }

    func testAvailabilityDisablesAccelerationModesWhenNoModelIsLoaded() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: false,
            mtpCompatible: false,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.off]?.isSelectable == true)
        XCTAssertFalse(availability[.auto]?.isSelectable == true)
        XCTAssertFalse(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testAvailabilityEnablesOnlyMTPWhenCompatibleSidecarExists() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: true,
            mtpCompatible: true,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.auto]?.isSelectable == true)
        XCTAssertTrue(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testAvailabilityEnablesEagle3ForDenseGemma4EvenBeforeDrafterDownload() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: true,
            mtpCompatible: false,
            denseGemma4Verifier: true
        )

        XCTAssertTrue(availability[.auto]?.isSelectable == true)
        XCTAssertFalse(availability[.mtp]?.isSelectable == true)
        XCTAssertTrue(availability[.eagle3]?.isSelectable == true)
    }

    func testPendingSelectionEnablesMTPBeforeModelLoadWhenSidecarIsDetected() {
        let availability = AFMMLXSpeculativeModeAvailability.pendingSelection(
            mtpCompatible: true,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.off]?.isSelectable == true)
        XCTAssertTrue(availability[.auto]?.isSelectable == true)
        XCTAssertTrue(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testPendingSelectionKeepsUnsupportedModesDisabledBeforeModelLoad() {
        let availability = AFMMLXSpeculativeModeAvailability.pendingSelection(
            mtpCompatible: false,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.off]?.isSelectable == true)
        XCTAssertFalse(availability[.auto]?.isSelectable == true)
        XCTAssertFalse(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testSpeculativeModelCompatibilityDetectsMTPFromConfigAndSidecar() {
        let compatibility = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "qwen3.6",
                "architectures": ["Qwen3_6ForCausalLM"],
            ],
            hasMTPSidecar: true
        )

        XCTAssertTrue(compatibility.mtpCompatible)
        XCTAssertFalse(compatibility.denseGemma4Verifier)

        let missingSidecar = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "qwen3.6",
                "architectures": ["Qwen3_6ForCausalLM"],
            ],
            hasMTPSidecar: false
        )

        XCTAssertFalse(missingSidecar.mtpCompatible)
    }

    func testSpeculativeModelCompatibilityDetectsDenseGemma4Verifier() {
        let dense = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "gemma_4",
                "architectures": ["Gemma4ForCausalLM"],
            ],
            hasMTPSidecar: false
        )

        XCTAssertFalse(dense.mtpCompatible)
        XCTAssertTrue(dense.denseGemma4Verifier)

        let moe = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "gemma_4",
                "architectures": ["Gemma4MoeForCausalLM"],
            ],
            hasMTPSidecar: false
        )

        XCTAssertFalse(moe.denseGemma4Verifier)
    }

    func testSpeculativeModelCompatibilityReadsModelDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let config: [String: Any] = [
            "model_type": "qwen3.6",
            "architectures": ["Qwen3_6ForCausalLM"],
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))
        FileManager.default.createFile(
            atPath: directory.appendingPathComponent("mtp.safetensors").path,
            contents: Data()
        )

        let compatibility = AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)

        XCTAssertTrue(compatibility.mtpCompatible)
        XCTAssertFalse(compatibility.denseGemma4Verifier)
    }
}
