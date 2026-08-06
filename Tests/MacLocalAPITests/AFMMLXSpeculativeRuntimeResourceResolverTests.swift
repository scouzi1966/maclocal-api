import XCTest
import AFMKitMLX

final class AFMMLXSpeculativeRuntimeResourceResolverTests: XCTestCase {
    func testCurrentLoadedModelDirectoryIsUnavailableForMissingOrBlankID() {
        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
                loadedModelRepoID: nil,
                repositoryDirectory: { _ in XCTFail("Repository resolver should not be called"); return nil }
            )
        )

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
                loadedModelRepoID: "   ",
                repositoryDirectory: { _ in XCTFail("Repository resolver should not be called"); return nil }
            )
        )
    }

    func testCurrentLoadedModelDirectoryUsesImportedPathDirectly() {
        let path = "/Volumes/edata/models/Qwen3.5"

        let directory = AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
            loadedModelRepoID: "  \(path)  ",
            repositoryDirectory: { _ in XCTFail("Repository resolver should not be called"); return nil }
        )

        XCTAssertEqual(directory?.path, path)
    }

    func testCurrentLoadedModelDirectoryResolvesRepositoryID() {
        let expected = URL(fileURLWithPath: "/cache/mlx-community/Qwen3.5")

        let directory = AFMMLXSpeculativeRuntimeResourceResolver.currentLoadedModelDirectory(
            loadedModelRepoID: "  mlx-community/Qwen3.5  ",
            repositoryDirectory: { repoID in
                XCTAssertEqual(repoID, "mlx-community/Qwen3.5")
                return expected
            }
        )

        XCTAssertEqual(directory, expected)
    }

    func testMTPSidecarPathRequiresDirectoryAndExistingSidecar() {
        let directory = URL(fileURLWithPath: "/cache/model", isDirectory: true)
        let expectedPath = "/cache/model/\(AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarFilename)"

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                modelDirectory: nil,
                fileExists: { _ in true }
            )
        )

        XCTAssertNil(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                modelDirectory: directory,
                fileExists: { path in
                    XCTAssertEqual(path, expectedPath)
                    return false
                }
            )
        )

        XCTAssertEqual(
            AFMMLXSpeculativeRuntimeResourceResolver.mtpSidecarPath(
                modelDirectory: directory,
                fileExists: { path in
                    XCTAssertEqual(path, expectedPath)
                    return true
                }
            ),
            expectedPath
        )
    }
}
