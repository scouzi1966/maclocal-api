import XCTest
@testable import AFMKitFoundationModels27

final class FoundationGenerationOptionsPolicyTests: XCTestCase {
    func testProviderDefaultsOnlyCarryToolCallingDecision() {
        let plan = AFMFoundationGenerationOptionsPolicy.plan(
            from: AFMFoundationGenerationParameters(useProviderDefaults: true),
            allowsToolCalling: true,
            toolsEnabled: true,
            requiresToolCalling: true
        )

        XCTAssertEqual(plan.sampling, .providerDefault)
        XCTAssertNil(plan.temperature)
        XCTAssertNil(plan.maximumResponseTokens)
        XCTAssertEqual(plan.toolCalling, .required)
    }

    func testZeroTemperatureUsesGreedySampling() {
        let plan = AFMFoundationGenerationOptionsPolicy.plan(
            from: AFMFoundationGenerationParameters(temperature: 0, topP: 0.5, maxTokens: 256),
            allowsToolCalling: true,
            toolsEnabled: true,
            requiresToolCalling: false
        )

        XCTAssertEqual(plan.sampling, .greedy)
        XCTAssertNil(plan.temperature)
        XCTAssertEqual(plan.maximumResponseTokens, 256)
        XCTAssertEqual(plan.toolCalling, .allowed)
    }

    func testRandomSamplingClampsTopP() {
        let high = AFMFoundationGenerationOptionsPolicy.plan(
            from: AFMFoundationGenerationParameters(temperature: 0.8, topP: 2.0, maxTokens: 128),
            allowsToolCalling: false,
            toolsEnabled: true,
            requiresToolCalling: true
        )
        let low = AFMFoundationGenerationOptionsPolicy.plan(
            from: AFMFoundationGenerationParameters(temperature: 0.8, topP: -1.0, maxTokens: 128),
            allowsToolCalling: false,
            toolsEnabled: true,
            requiresToolCalling: true
        )

        XCTAssertEqual(high.sampling, .random(probabilityThreshold: 1.0))
        XCTAssertEqual(high.temperature, 0.8)
        XCTAssertEqual(high.toolCalling, .disallowed)
        XCTAssertEqual(low.sampling, .random(probabilityThreshold: 0.0))
    }

    func testRequiredToolPolicyNeedsAvailableAndEnabledTools() {
        XCTAssertEqual(
            AFMFoundationGenerationOptionsPolicy.toolCallingDecision(
                allowsTools: false,
                toolsEnabled: true,
                requiresToolCalling: true
            ),
            .disallowed
        )
        XCTAssertEqual(
            AFMFoundationGenerationOptionsPolicy.toolCallingDecision(
                allowsTools: true,
                toolsEnabled: false,
                requiresToolCalling: true
            ),
            .allowed
        )
        XCTAssertEqual(
            AFMFoundationGenerationOptionsPolicy.toolCallingDecision(
                allowsTools: true,
                toolsEnabled: true,
                requiresToolCalling: true
            ),
            .required
        )
    }
}
