#if canImport(FoundationModels)
import Foundation
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

    func testPrivateCloudQuotaLimitDescriptionOmitsOptionalFields() {
        let detail = AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudComputeQuotaLimit(
            AFMFoundationPrivateCloudComputeQuotaLimitSnapshot(
                resetDate: nil,
                hasLimitIncreaseSuggestion: false
            )
        )

        XCTAssertEqual(detail, "PCC quota limit reached")
    }

    func testPrivateCloudQuotaLimitDescriptionIncludesOptionalFields() {
        let resetDate = Date(timeIntervalSince1970: 1_800_000_000)
        let detail = AFMFoundationNativeAvailabilityReasonDescriptions.privateCloudComputeQuotaLimit(
            AFMFoundationPrivateCloudComputeQuotaLimitSnapshot(
                resetDate: resetDate,
                hasLimitIncreaseSuggestion: true
            )
        )

        XCTAssertTrue(detail.hasPrefix("PCC quota limit reached; resets "))
        XCTAssertTrue(detail.hasSuffix("; limit increase available"))
    }
}
#endif
