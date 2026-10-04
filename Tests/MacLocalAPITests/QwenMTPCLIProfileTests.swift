import AFMKit
import XCTest

final class QwenMTPCLIProfileTests: XCTestCase {
    func testOmittedSelectionPreservesProviderDefaults() throws {
        XCTAssertNil(try QwenMTPCLIProfile.resolve(option: nil, environment: [:]))
        XCTAssertNil(try QwenMTPCLIProfile.resolve(
            option: nil, environment: [QwenMTPCLIProfile.environmentKey: " \n"]))
    }

    func testEnvironmentSelectionAcceptsProviderNormalization() throws {
        XCTAssertEqual(try QwenMTPCLIProfile.resolve(option: nil,
            environment: [QwenMTPCLIProfile.environmentKey: " Throughput-V1\n"]), .throughputV1)
        XCTAssertEqual(try QwenMTPCLIProfile.resolve(option: nil,
            environment: [QwenMTPCLIProfile.environmentKey: "OFF"]), .off)
    }

    func testExplicitCLISelectionOverridesEvenInvalidEnvironment() throws {
        let environment = [QwenMTPCLIProfile.environmentKey: "misspelled",
                           "AFM_QWEN_VERIFY_QMM": "0"]
        for profile in QwenMTPCLIProfile.allCases {
            XCTAssertEqual(try QwenMTPCLIProfile.resolve(
                option: profile, environment: environment), profile)
        }
        XCTAssertEqual(environment["AFM_QWEN_VERIFY_QMM"], "0")
        XCTAssertEqual(environment[QwenMTPCLIProfile.environmentKey], "misspelled")
    }

    func testInvalidEnvironmentReportsSupportedChoicesBeforeLoading() {
        XCTAssertThrowsError(try QwenMTPCLIProfile.resolve(option: nil,
            environment: [QwenMTPCLIProfile.environmentKey: "fastest"])) { error in
            XCTAssertTrue(error.localizedDescription.contains("throughput-v1 or off"))
        }
    }
}
