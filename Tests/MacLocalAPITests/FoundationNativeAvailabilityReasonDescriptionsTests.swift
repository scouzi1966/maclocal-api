#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationNativeAvailabilityReasonDescriptionsTests: XCTestCase {
    func testSystemLanguageModelReasonDescriptions() {
        XCTAssertEqual(
            AFMFoundationNativeAvailabilityReasonDescriptions.systemLanguageModel(.deviceNotEligible),
            "device does not support Apple Intelligence"
        )
        XCTAssertEqual(
            AFMFoundationNativeAvailabilityReasonDescriptions.systemLanguageModel(.appleIntelligenceNotEnabled),
            "Apple Intelligence is not enabled for this user or locale"
        )
        XCTAssertEqual(
            AFMFoundationNativeAvailabilityReasonDescriptions.systemLanguageModel(.modelNotReady),
            "model assets are not ready yet"
        )
    }

    func testPrivateCloudComputeReasonDescriptions() {
        XCTAssertEqual(
            AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudCompute(.deviceNotEligible),
            "device does not support Apple Intelligence"
        )
        XCTAssertEqual(
            AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudCompute(.systemNotReady),
            "system is not yet ready to serve PCC requests"
        )
    }
}
#endif
