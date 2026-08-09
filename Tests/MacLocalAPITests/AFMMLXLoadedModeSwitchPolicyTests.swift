import XCTest
import AFMKitMLX

final class AFMMLXLoadedModeSwitchPolicyTests: XCTestCase {
    func testRequiresDualModeModelType() {
        XCTAssertNil(
            AFMMLXLoadedModeSwitchPolicy.make(
                loadedModelRepoID: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                loadedModelType: "llama",
                isLoadedModelVLM: false,
                loadedModelDirectoryIsVision: true
            )
        )
    }

    func testRequiresVisionDirectory() {
        XCTAssertNil(
            AFMMLXLoadedModeSwitchPolicy.make(
                loadedModelRepoID: "mlx-community/Qwen3.5-35B-A3B-4bit",
                loadedModelType: "qwen3.5",
                isLoadedModelVLM: false,
                loadedModelDirectoryIsVision: false
            )
        )
    }

    func testTargetsImportedPath() {
        let plan = AFMMLXLoadedModeSwitchPolicy.make(
            loadedModelRepoID: "/Volumes/models/Qwen3.5-35B-A3B-4bit",
            loadedModelType: "qwen3.5",
            isLoadedModelVLM: false,
            loadedModelDirectoryIsVision: true
        )

        XCTAssertEqual(
            plan,
            .imported(rawPath: "/Volumes/models/Qwen3.5-35B-A3B-4bit", targetVLM: true)
        )
        XCTAssertEqual(plan?.targetVLM, true)
    }

    func testTargetsCurrentLoadedModel() {
        let plan = AFMMLXLoadedModeSwitchPolicy.make(
            loadedModelRepoID: "mlx-community/Qwen3.5-35B-A3B-4bit",
            loadedModelType: "qwen3.5",
            isLoadedModelVLM: true,
            loadedModelDirectoryIsVision: true
        )

        XCTAssertEqual(plan, .currentLoadedModel(targetVLM: false))
        XCTAssertEqual(plan?.targetVLM, false)
    }

    func testMultimodalRequestReloadsDualModeTextFactory() {
        XCTAssertTrue(
            AFMMLXLoadedModeSwitchPolicy.shouldReloadVisionFactoryForMultimodalRequest(
                loadedModelType: "gemma4",
                isLoadedModelVLM: false,
                loadedModelDirectoryIsVision: true
            )
        )
    }

    func testMultimodalRequestKeepsAlreadyLoadedVisionFactory() {
        XCTAssertFalse(
            AFMMLXLoadedModeSwitchPolicy.shouldReloadVisionFactoryForMultimodalRequest(
                loadedModelType: "gemma4",
                isLoadedModelVLM: true,
                loadedModelDirectoryIsVision: true
            )
        )
    }

    func testMultimodalRequestDoesNotReloadTextOnlyModel() {
        XCTAssertFalse(
            AFMMLXLoadedModeSwitchPolicy.shouldReloadVisionFactoryForMultimodalRequest(
                loadedModelType: "llama",
                isLoadedModelVLM: false,
                loadedModelDirectoryIsVision: false
            )
        )
    }
}
