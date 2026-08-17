import AFMKitCore
@testable import AFMKitMLX
import XCTest

final class MLXMediaPreflightTests: XCTestCase {
    func testMediaErrorsDoNotExposeAbsoluteModelPaths() {
        let missingAssets = MLXServiceError.visionAssetsUnavailable(
            model: "/Users/example/private/qwen-snapshot",
            missing: ["processorConfiguration", "visionWeights"]
        )
        let unsupported = MLXServiceError.unsupportedMediaInput(
            model: "/Users/example/private/qwen-snapshot",
            kind: "image"
        )

        XCTAssertEqual(
            missingAssets.localizedDescription,
            "qwen-snapshot: vision assets are unavailable (missing: processorConfiguration, visionWeights)"
        )
        XCTAssertEqual(
            unsupported.localizedDescription,
            "qwen-snapshot: image input is not supported by the loaded MLX model"
        )
        XCTAssertFalse(missingAssets.localizedDescription.contains("/Users/example"))
        XCTAssertFalse(unsupported.localizedDescription.contains("/Users/example"))
    }

    func testCompleteQwenVLMAllowsImagesAndAdvertisesRuntimeVision() {
        let architecture = qwenArchitecture()
        let qualification = qwenQualification()
        let descriptor = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: descriptor(capabilities: [.text, .vision]),
            architecture: architecture,
            qualification: qualification,
            factory: .vlm
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .vlm
            ),
            .allowed
        )
        XCTAssertTrue(descriptor.capabilities.contains(.vision))
    }

    func testIncompleteQwenFallsBackToTypedAssetFailureAndNoRuntimeVision() {
        let architecture = qwenArchitecture()
        let qualification = qwenQualification(
            missing: [.processorConfiguration, .visionWeights]
        )
        let descriptor = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: descriptor(capabilities: [.text, .vision]),
            architecture: architecture,
            qualification: qualification,
            factory: .llm
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .llm
            ),
            .visionAssetsUnavailable(
                missing: ["processorConfiguration", "visionWeights"]
            )
        )
        XCTAssertFalse(descriptor.capabilities.contains(.vision))
    }

    func testCompleteQualificationDoesNotAuthorizeAnLLMContainer() {
        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: qwenArchitecture(),
                qualification: qwenQualification(),
                factory: .llm
            ),
            .unsupported
        )
    }

    func testNonVisionArchitectureRemainsUnsupported() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "org/text",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: false,
            requiresVisionModelFactory: false
        )
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "text",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isConditionalGeneration: false,
            declaresVision: false,
            processorClass: nil,
            visionTensorCount: 0,
            missingAssets: Set(AFMMLXVisionAssetIssue.allCases)
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .llm
            ),
            .unsupported
        )
    }

    func testExistingNonQwenVLMUsesItsArchitectureCapability() {
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: "org/gemma-vision",
            modelType: "gemma3",
            canonicalModelType: "gemma3",
            isVisionConfiguration: true,
            requiresVisionModelFactory: true
        )
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "gemma",
            modelType: "gemma3",
            canonicalModelType: "gemma3",
            isConditionalGeneration: false,
            declaresVision: true,
            processorClass: "Gemma3Processor",
            visionTensorCount: 1,
            missingAssets: [.conditionalGenerationArchitecture]
        )

        XCTAssertEqual(
            AFMMLXRuntimeVisionPolicy.admission(
                for: .image,
                architecture: architecture,
                qualification: qualification,
                factory: .vlm
            ),
            .allowed
        )
    }

    private func qwenArchitecture() -> AFMMLXModelArchitecturePreflight {
        AFMMLXModelArchitecturePreflight(
            modelID: "mlx-community/Qwen3.8-27B-4bit",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: true,
            requiresVisionModelFactory: false
        )
    }

    private func qwenQualification(
        missing: Set<AFMMLXVisionAssetIssue> = []
    ) -> AFMMLXVisionAssetQualification {
        AFMMLXVisionAssetQualification(
            snapshotIdentity: "qwen",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isConditionalGeneration: true,
            declaresVision: true,
            processorClass: missing.contains(.processorConfiguration)
                ? nil
                : "Qwen3VLProcessor",
            visionTensorCount: missing.contains(.visionWeights) ? 0 : 2,
            missingAssets: missing
        )
    }

    private func descriptor(
        capabilities: AFMModelCapabilities
    ) -> AFMModelDescriptor {
        AFMModelDescriptor(
            providerID: "mlx",
            modelID: "mlx-community/Qwen3.8-27B-4bit",
            displayName: "Qwen3.8-27B-4bit",
            capabilities: capabilities,
            privacyBoundary: .device
        )
    }
}
