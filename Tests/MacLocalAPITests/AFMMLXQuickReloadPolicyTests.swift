import XCTest
import AFMKitMLX

final class AFMMLXQuickReloadPolicyTests: XCTestCase {
    func testResolvesImportedLoadedPath() {
        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.make(
                loadedModelRepoID: " /Volumes/models/local-model ",
                loadedModelName: "local-model",
                curatedCandidates: [],
                downloadedIDs: []
            ),
            .imported(rawPath: "/Volumes/models/local-model")
        )

        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.make(
                loadedModelRepoID: " imported:/Volumes/models/local-model ",
                loadedModelName: "local-model",
                curatedCandidates: [],
                downloadedIDs: []
            ),
            .imported(rawPath: "/Volumes/models/local-model")
        )
    }

    func testResolvesCuratedByRepoIDOrName() {
        let curated = [
            AFMMLXQuickCuratedLoadCandidate(
                id: "mlx-community/curated-model",
                name: "Curated Model",
                repoID: "mlx-community/curated-model"
            )
        ]

        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.make(
                loadedModelRepoID: "mlx-community/curated-model",
                loadedModelName: nil,
                curatedCandidates: curated,
                downloadedIDs: []
            ),
            .curated(selectionID: "mlx-community/curated-model")
        )

        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.make(
                loadedModelRepoID: nil,
                loadedModelName: " Curated Model ",
                curatedCandidates: curated,
                downloadedIDs: []
            ),
            .curated(selectionID: "mlx-community/curated-model")
        )
    }

    func testResolvesUserDownloadedModel() {
        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.make(
                loadedModelRepoID: "custom/downloaded-model",
                loadedModelName: "downloaded-model",
                curatedCandidates: [],
                downloadedIDs: ["custom/downloaded-model"]
            ),
            .userDownloaded(repoID: "custom/downloaded-model")
        )
    }

    func testReturnsUnavailableForUnknownModel() {
        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.make(
                loadedModelRepoID: "unknown/model",
                loadedModelName: "Unknown Model",
                curatedCandidates: [],
                downloadedIDs: []
            ),
            .unavailable
        )
    }

    func testImportedPathNormalization() {
        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.importedPath(from: " imported:/tmp/model "),
            "/tmp/model"
        )
        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.importedPath(from: " /tmp/model "),
            "/tmp/model"
        )
        XCTAssertNil(AFMMLXQuickReloadPolicy.importedPath(from: "mlx-community/model"))
    }

    func testFallbackImportedAccessUsesNormalizedPathNameAndVisionMode() {
        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.fallbackImportedAccess(
                rawPath: " imported:/Volumes/models/local-model ",
                isVision: true
            ),
            AFMMLXImportedFallbackAccess(
                rawPath: "/Volumes/models/local-model",
                name: "local-model",
                isVision: true
            )
        )

        XCTAssertEqual(
            AFMMLXQuickReloadPolicy.fallbackImportedAccess(
                rawPath: " /Volumes/models/text-model ",
                isVision: false
            ),
            AFMMLXImportedFallbackAccess(
                rawPath: "/Volumes/models/text-model",
                name: "text-model",
                isVision: false
            )
        )
    }

    func testFallbackImportedAccessRejectsRepositoryIdentifier() {
        XCTAssertNil(
            AFMMLXQuickReloadPolicy.fallbackImportedAccess(
                rawPath: "mlx-community/model",
                isVision: true
            )
        )
    }
}
