#if canImport(FoundationModels)
@testable import AFMKitFoundationModels27
import XCTest

final class FoundationToolAvailabilityPolicyTests: XCTestCase {
    func testVisionToolsRequireImageInputAndAppleVisionSupport() {
        XCTAssertTrue(AFMFoundationToolAvailabilityPolicy.includesVisionTools(
            includesImageInput: true,
            supportsAppleVisionTools: true
        ))
        XCTAssertFalse(AFMFoundationToolAvailabilityPolicy.includesVisionTools(
            includesImageInput: false,
            supportsAppleVisionTools: true
        ))
        XCTAssertFalse(AFMFoundationToolAvailabilityPolicy.includesVisionTools(
            includesImageInput: true,
            supportsAppleVisionTools: false
        ))
    }
}
#endif
