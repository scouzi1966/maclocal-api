import XCTest
import AFMKitMLX

final class AFMMLXModelPathResolutionPolicyTests: XCTestCase {
    func testCurrentModelPathResolutionRejectsBlankLoadedModel() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.currentModelPathResolution(
                loadedModelName: " \n ",
                resolvedDirectory: URL(fileURLWithPath: "/models/unused")
            ),
            .noLoadedModel
        )
    }

    func testCurrentModelPathResolutionReportsMissingLoadedModel() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.currentModelPathResolution(
                loadedModelName: " Loaded-Model ",
                resolvedDirectory: nil
            ),
            .missing(modelName: "Loaded-Model")
        )
    }

    func testCurrentModelPathResolutionReturnsResolvedDirectoryPath() {
        let directory = URL(fileURLWithPath: "/Volumes/models/Loaded-Model")

        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.currentModelPathResolution(
                loadedModelName: " Loaded-Model ",
                resolvedDirectory: directory
            ),
            .resolved(path: directory.path)
        )
    }

    func testBenchmarkLoadPathRequiresResolvedDirectory() {
        XCTAssertNil(
            AFMMLXModelPathResolutionPolicy.benchmarkLoadPath(
                forSelection: "example-org/Missing-Model",
                resolvedDirectory: nil
            )
        )
    }

    func testBenchmarkLoadPathReturnsResolvedDirectoryPath() {
        let directory = URL(fileURLWithPath: "/models/example-org/Downloaded-Model-4bit")

        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.benchmarkLoadPath(
                forSelection: " example-org/Downloaded-Model-4bit ",
                resolvedDirectory: directory
            ),
            "/models/example-org/Downloaded-Model-4bit"
        )
    }

    func testBenchmarkLoadPathRejectsBlankSelection() {
        XCTAssertNil(
            AFMMLXModelPathResolutionPolicy.benchmarkLoadPath(
                forSelection: " \n ",
                resolvedDirectory: URL(fileURLWithPath: "/models/unused")
            )
        )
    }
}
