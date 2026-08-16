@testable import AFMKit
import XCTest

final class BuildInfoTests: XCTestCase {
    func testBaseVersionIsNextRelease() {
        XCTAssertEqual(BuildInfo.resolvedVersion(override: nil), "v0.9.17")
    }

    func testBuildVersionOverridePreservesVersionPrefix() {
        XCTAssertEqual(
            BuildInfo.resolvedVersion(override: "v0.9.15-staging.7bb83c9.20260807"),
            "v0.9.15-staging.7bb83c9.20260807"
        )
    }

    func testBuildVersionOverrideAddsVersionPrefix() {
        XCTAssertEqual(
            BuildInfo.resolvedVersion(override: "0.9.15-staging.7bb83c9.20260807"),
            "v0.9.15-staging.7bb83c9.20260807"
        )
    }

    func testBlankBuildVersionOverrideUsesBaseVersion() {
        XCTAssertEqual(BuildInfo.resolvedVersion(override: "  "), "v0.9.17")
    }
}
