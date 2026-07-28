import XCTest
import AFMKitMLX

final class AFMMLXQuickDeletePolicyTests: XCTestCase {
    func testKeepsImportedSelectionAsReferenceOnly() {
        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.make(
                selectionID: " imported:/Volumes/models/local-model ",
                downloadedIDs: []
            ),
            .importedReference(rawPath: "/Volumes/models/local-model")
        )
    }

    func testResolvesUserDownloadedSelection() {
        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.make(
                selectionID: "custom/downloaded-model",
                downloadedIDs: ["custom/downloaded-model"]
            ),
            .userDownloaded(repoID: "custom/downloaded-model")
        )
    }

    func testFallsBackToCachedModelName() {
        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.make(
                selectionID: "mlx-community/Qwen3-4B-Instruct-4bit",
                downloadedIDs: []
            ),
            .cachedModel(name: "Qwen3-4B-Instruct-4bit")
        )

        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.make(
                selectionID: "/Volumes/models/local-model",
                downloadedIDs: []
            ),
            .cachedModel(name: "local-model")
        )
    }

    func testReturnsUnavailableForEmptySelection() {
        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.make(
                selectionID: "  ",
                downloadedIDs: ["custom/downloaded-model"]
            ),
            .unavailable
        )
    }

    func testDownloadedModelDeletionUnloadsMatchingLoadedModel() {
        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.downloadedModelDeletionPlan(
                repoID: "mlx-community/Qwen3-4B-Instruct-4bit",
                isModelLoaded: true,
                loadedModelName: "Qwen3-4B-Instruct-4bit"
            ),
            AFMMLXDownloadedModelDeletionPlan(
                repoID: "mlx-community/Qwen3-4B-Instruct-4bit",
                shouldUnloadCurrentModel: true
            )
        )
    }

    func testDownloadedModelDeletionDoesNotUnloadDifferentOrUnloadedModel() {
        XCTAssertFalse(
            AFMMLXQuickDeletePolicy.downloadedModelDeletionPlan(
                repoID: "mlx-community/Qwen3-4B-Instruct-4bit",
                isModelLoaded: true,
                loadedModelName: "Gemma-4B"
            ).shouldUnloadCurrentModel
        )

        XCTAssertFalse(
            AFMMLXQuickDeletePolicy.downloadedModelDeletionPlan(
                repoID: "mlx-community/Qwen3-4B-Instruct-4bit",
                isModelLoaded: false,
                loadedModelName: "Qwen3-4B-Instruct-4bit"
            ).shouldUnloadCurrentModel
        )
    }

    func testDownloadedModelDeletionNormalizesWhitespace() {
        XCTAssertEqual(
            AFMMLXQuickDeletePolicy.downloadedModelDeletionPlan(
                repoID: " mlx-community/Qwen3-4B-Instruct-4bit ",
                isModelLoaded: true,
                loadedModelName: " Qwen3-4B-Instruct-4bit "
            ),
            AFMMLXDownloadedModelDeletionPlan(
                repoID: "mlx-community/Qwen3-4B-Instruct-4bit",
                shouldUnloadCurrentModel: true
            )
        )
    }
}
