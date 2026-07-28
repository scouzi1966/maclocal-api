import XCTest
@testable import AFMKitMLX

final class AFMMLXGenerationPlannerTests: XCTestCase {
    func testPlannerSuppressesImplicitReasoningWhenThinkingToggleIsDisabled() {
        let plan = AFMMLXGenerationPlanner.make(
            maxTokens: 64,
            temperature: 0.4,
            topP: 0.8,
            repetitionPenalty: 1.0,
            fallbackPrefillStepSize: 1024,
            hiddenOverrides: AFMMLXGenerationHiddenOverrides(),
            supportsThinkingToggle: true,
            enableThinking: false,
            modelHasImplicitReasoning: true
        )

        XCTAssertFalse(plan.hasReasoningOutput)
        XCTAssertEqual(plan.thinkingContext, .enableThinking(false))
        XCTAssertEqual(plan.additionalContext?["enable_thinking"] as? Bool, false)
    }

    func testPlannerKeepsImplicitReasoningForModelsWithoutThinkingToggle() {
        let plan = AFMMLXGenerationPlanner.make(
            maxTokens: 64,
            temperature: 0.4,
            topP: 0.8,
            repetitionPenalty: 1.0,
            fallbackPrefillStepSize: 1024,
            hiddenOverrides: AFMMLXGenerationHiddenOverrides(),
            supportsThinkingToggle: false,
            enableThinking: false,
            modelHasImplicitReasoning: true
        )

        XCTAssertTrue(plan.hasReasoningOutput)
        XCTAssertNil(plan.thinkingContext)
        XCTAssertNil(plan.additionalContext)
    }

    func testPlannerMapsHiddenOverridesIntoGenerationRequest() {
        let hiddenOverrides = AFMMLXGenerationHiddenOverrides(
            maxKVSize: 4096,
            kvBits: 4,
            kvGroupSize: 32,
            quantizedKVStart: 24,
            prefillStepSize: 512,
            repetitionContextSize: 96
        )

        let plan = AFMMLXGenerationPlanner.make(
            maxTokens: 256,
            temperature: 0.2,
            topP: 0.7,
            repetitionPenalty: 1.2,
            topK: 40,
            minP: 0.05,
            presencePenalty: 0.1,
            fallbackPrefillStepSize: 2048,
            hiddenOverrides: hiddenOverrides,
            supportsThinkingToggle: true,
            enableThinking: true,
            modelHasImplicitReasoning: true
        )

        XCTAssertEqual(plan.hasReasoningOutput, true)
        XCTAssertEqual(plan.thinkingContext, .enableThinking(true))
        XCTAssertEqual(
            plan.parameters,
            AFMMLXGenerationParameterRequest(
                maxTokens: 256,
                maxKVSize: 4096,
                kvBits: 4,
                kvGroupSize: 32,
                quantizedKVStart: 24,
                temperature: 0.2,
                topP: 0.7,
                repetitionPenalty: 1.2,
                repetitionContextSize: 96,
                topK: 40,
                minP: 0.05,
                presencePenalty: 0.1,
                prefillStepSize: 512
            )
        )
    }

    func testPlannerUsesFallbackPrefillWhenHiddenOverrideIsUnset() {
        let plan = AFMMLXGenerationPlanner.make(
            maxTokens: 32,
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.0,
            fallbackPrefillStepSize: 1536,
            hiddenOverrides: AFMMLXGenerationHiddenOverrides(),
            supportsThinkingToggle: false,
            enableThinking: true,
            modelHasImplicitReasoning: false
        )

        XCTAssertEqual(plan.parameters.prefillStepSize, 1536)
        XCTAssertEqual(plan.parameters.kvGroupSize, 64)
        XCTAssertEqual(plan.parameters.quantizedKVStart, 0)
        XCTAssertEqual(plan.parameters.repetitionContextSize, 64)
    }
}
