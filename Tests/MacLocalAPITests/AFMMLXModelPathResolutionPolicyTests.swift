import XCTest
import AFMKitMLX

final class AFMMLXModelPathResolutionPolicyTests: XCTestCase {
    func testLocalModelDirectoryLookupsRejectBlankSelection() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.localModelDirectoryLookups(
                forSelection: " \n ",
                customModelPath: "custom/repo",
                downloadedCandidates: [
                    .init(id: "downloaded/repo", name: "Downloaded")
                ]
            ),
            []
        )
    }

    func testLocalModelDirectoryLookupsShortCircuitImportedSelection() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.localModelDirectoryLookups(
                forSelection: " imported:/Volumes/models/Local ",
                customModelPath: "custom/repo",
                downloadedCandidates: [
                    .init(id: "downloaded/repo", name: "Downloaded")
                ]
            ),
            [.importedPath("/Volumes/models/Local")]
        )
    }

    func testLocalModelDirectoryLookupsPreferCustomImportedPath() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.localModelDirectoryLookups(
                forSelection: "Qwen",
                customModelPath: " /Volumes/models/Custom ",
                downloadedCandidates: []
            ),
            [
                .importedPath("/Volumes/models/Custom"),
                .modelName("Qwen"),
                .repositoryID("Qwen")
            ]
        )
    }

    func testLocalModelDirectoryLookupsPreferCustomRepositoryBeforeDownloadedAndSelection() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.localModelDirectoryLookups(
                forSelection: " Downloaded Name ",
                customModelPath: " custom/repo ",
                downloadedCandidates: [
                    .init(id: "downloaded/repo", name: "Downloaded Name")
                ]
            ),
            [
                .customRepositoryID("custom/repo"),
                .downloadedModel(repoID: "downloaded/repo"),
                .modelName("Downloaded Name"),
                .repositoryID("Downloaded Name")
            ]
        )
    }

    func testLocalModelDirectoryLookupsDeduplicateEquivalentRepositoryLookups() {
        XCTAssertEqual(
            AFMMLXModelPathResolutionPolicy.localModelDirectoryLookups(
                forSelection: "same/repo",
                customModelPath: "same/repo",
                downloadedCandidates: [
                    .init(id: "same/repo", name: "Same Repo")
                ]
            ),
            [
                .customRepositoryID("same/repo"),
                .downloadedModel(repoID: "same/repo"),
                .modelName("same/repo"),
                .repositoryID("same/repo")
            ]
        )
    }

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

    func testHasLocalModelRequiresResolvedLocalDirectory() {
        XCTAssertFalse(
            AFMMLXModelPathResolutionPolicy.hasLocalModel(
                forSelection: "example-org/Missing-Model",
                resolvedDirectory: nil
            )
        )
    }

    func testHasLocalModelReturnsTrueForResolvedDirectory() {
        XCTAssertTrue(
            AFMMLXModelPathResolutionPolicy.hasLocalModel(
                forSelection: " example-org/Downloaded-Model-4bit ",
                resolvedDirectory: URL(fileURLWithPath: "/models/example-org/Downloaded-Model-4bit")
            )
        )
    }

    func testHasLocalModelRejectsBlankSelection() {
        XCTAssertFalse(
            AFMMLXModelPathResolutionPolicy.hasLocalModel(
                forSelection: " \n ",
                resolvedDirectory: URL(fileURLWithPath: "/models/unused")
            )
        )
    }
}
