@testable import AFMKit
import XCTest

final class BuildInfoTests: XCTestCase {
    func testReleaseCompilerIncludesFoundationModels() {
        XCTAssertTrue(BuildInfo.foundationModelsCompiled)
    }

    func testBaseVersionIsNextRelease() {
        XCTAssertEqual(BuildInfo.version, "v0.9.20")
        let expected = BuildInfo.commit.map { "v0.9.20-\($0)" } ?? "v0.9.20"
        XCTAssertEqual(BuildInfo.resolvedVersion(override: nil), expected)
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
        XCTAssertEqual(BuildInfo.resolvedVersion(override: "  "),
                       BuildInfo.resolvedVersion(override: nil))
    }
}
