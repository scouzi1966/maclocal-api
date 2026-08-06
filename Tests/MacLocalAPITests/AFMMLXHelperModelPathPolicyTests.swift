import XCTest
import AFMKitMLX

final class AFMMLXHelperModelPathPolicyTests: XCTestCase {
    func testRepoIDFallsBackUnderHostCacheRoot() {
        XCTAssertEqual(
            AFMMLXHelperModelPathPolicy.modelPath(
                repoID: "mlx-community/Qwen3-4B-4bit",
                loadedModelName: nil,
                fallbackCacheRoot: URL(fileURLWithPath: "/cache/models")
            ),
            "/cache/models/mlx-community/Qwen3-4B-4bit"
        )
    }

    func testLoadedModelNameFallsBackUnderDefaultOrganization() {
        XCTAssertEqual(
            AFMMLXHelperModelPathPolicy.modelPath(
                repoID: nil,
                loadedModelName: " Qwen3-4B-4bit ",
                fallbackCacheRoot: URL(fileURLWithPath: "/cache/models")
            ),
            "/cache/models/mlx-community/Qwen3-4B-4bit"
        )
    }

    func testResolvedDirectoryWinsForRepoOrLoadedName() {
        let resolvedDirectory = URL(fileURLWithPath: "/Volumes/models/Qwen3-4B-4bit")

        XCTAssertEqual(
            AFMMLXHelperModelPathPolicy.modelPath(
                repoID: "mlx-community/Qwen3-4B-4bit",
                loadedModelName: nil,
                resolvedDirectory: resolvedDirectory,
                fallbackCacheRoot: URL(fileURLWithPath: "/cache/models")
            ),
            resolvedDirectory.path
        )
        XCTAssertEqual(
            AFMMLXHelperModelPathPolicy.modelPath(
                repoID: nil,
                loadedModelName: "Qwen3-4B-4bit",
                resolvedDirectory: resolvedDirectory,
                fallbackCacheRoot: URL(fileURLWithPath: "/cache/models")
            ),
            resolvedDirectory.path
        )
    }

    func testBlankInputsReturnNil() {
        XCTAssertNil(
            AFMMLXHelperModelPathPolicy.modelPath(
                repoID: " \n ",
                loadedModelName: " ",
                fallbackCacheRoot: URL(fileURLWithPath: "/cache/models")
            )
        )
    }
}
