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

    func testDualModeVisionConfigurationLoadsLLMInitially() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "gemma4-vision",
            modelType: "gemma4",
            canonicalModelType: "gemma4",
            isVisionConfiguration: true,
            requiresVisionModelFactory: false
        )
        XCTAssertEqual(AFMMLXModelFactoryPolicy.initialFactory(forceVLM: false, architecture: architecture), .llm)
        XCTAssertEqual(AFMMLXModelFactoryPolicy.initialFactory(forceVLM: true, architecture: architecture), .vlm)
    }

    func testTextOnlyDualModeConfigurationLoadsLLM() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "qwen-text",
            modelType: "qwen3_6",
            canonicalModelType: "qwen3_6",
            isVisionConfiguration: false,
            requiresVisionModelFactory: false
        )
        XCTAssertEqual(AFMMLXModelFactoryPolicy.initialFactory(forceVLM: false, architecture: architecture), .llm)
    }

    func testMediaKindsAndCapabilitiesAreExplicit() {
        let gemma = AFMMLXModelArchitecturePreflight(
            modelID: "gemma4-vision",
            modelType: "gemma4",
            canonicalModelType: "gemma4",
            isVisionConfiguration: true,
            requiresVisionModelFactory: false
        )
        let qwen = AFMMLXModelArchitecturePreflight(
            modelID: "qwen-vl",
            modelType: "qwen3_vl",
            canonicalModelType: "qwen3_vl",
            isVisionConfiguration: true,
            requiresVisionModelFactory: true
        )

        XCTAssertEqual(AFMMLXRequestMediaPolicy.kind(contentPartType: "image_url", mediaURL: "data:image/png;base64,AA=="), .image)
        XCTAssertEqual(AFMMLXRequestMediaPolicy.kind(contentPartType: "image_url", mediaURL: "data:video/mp4;base64,AA=="), .video)
        XCTAssertEqual(AFMMLXRequestMediaPolicy.kind(contentPartType: "image_url", mediaURL: "https://example.com/clip.mp4"), .image)
        XCTAssertEqual(AFMMLXRequestMediaPolicy.kind(contentPartType: "input_audio"), .audio)
        XCTAssertTrue(AFMMLXRequestMediaPolicy.supports(.image, architecture: gemma))
        XCTAssertFalse(AFMMLXRequestMediaPolicy.supports(.video, architecture: gemma))
        XCTAssertTrue(AFMMLXRequestMediaPolicy.supports(.video, architecture: qwen))
        XCTAssertFalse(AFMMLXRequestMediaPolicy.supports(.audio, architecture: qwen))
    }
}
