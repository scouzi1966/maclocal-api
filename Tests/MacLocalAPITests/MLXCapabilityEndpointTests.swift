import AFMKitCore
import AFMKitMLX
@testable import AFMServer
import XCTest

final class MLXCapabilityEndpointTests: XCTestCase {
    func testCompleteLoadedDescriptorEnablesVisionOnBothSurfaces() {
        let descriptor = makeDescriptor(capabilities: [.text, .vision, .streaming])

        XCTAssertTrue(AFMMLXCapabilityPresentation.supportsVision(descriptor: descriptor))
        XCTAssertTrue(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: descriptor)
                .contains("vision")
        )
    }

    func testIncompleteLoadedDescriptorDisablesVisionOnBothSurfaces() {
        let descriptor = makeDescriptor(capabilities: [.text, .streaming])

        XCTAssertFalse(AFMMLXCapabilityPresentation.supportsVision(descriptor: descriptor))
        XCTAssertFalse(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: descriptor)
                .contains("vision")
        )
    }

    func testUnavailableRuntimeDescriptorFailsClosedForVision() {
        XCTAssertFalse(AFMMLXCapabilityPresentation.supportsVision(descriptor: nil))
        XCTAssertFalse(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: nil)
                .contains("vision")
        )
    }

    func testQualifiedVLMWithLoadedMTPIsPublishedOnServerSurfaces() {
        let declared = makeDescriptor(
            capabilities: [.text, .vision, .streaming, .speculativeDecoding]
        )
        let architecture = AFMMLXModelArchitecturePreflight(
            modelID: declared.modelID.rawValue,
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isVisionConfiguration: true,
            requiresVisionModelFactory: true
        )
        let qualification = AFMMLXVisionAssetQualification(
            snapshotIdentity: "complete-vlm",
            modelType: "qwen3_5",
            canonicalModelType: "qwen3_5",
            isConditionalGeneration: true,
            declaresVision: true,
            processorClass: "Qwen3VLProcessor",
            visionTensorCount: 333,
            missingAssets: []
        )
        let active = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: declared,
            architecture: architecture,
            qualification: qualification,
            factory: .vlm,
            mtpEnabled: true,
            mtpBindingModelID: declared.modelID.rawValue
        )
        let inactive = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: declared,
            architecture: architecture,
            qualification: qualification,
            factory: .vlm,
            mtpEnabled: false,
            mtpBindingModelID: declared.modelID.rawValue
        )

        XCTAssertTrue(AFMMLXCapabilityPresentation.supportsVision(descriptor: active))
        XCTAssertEqual(
            Set(AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: active)),
            ["chat", "completion", "vision", "streaming", "speculative_decoding"]
        )
        XCTAssertTrue(AFMMLXCapabilityPresentation.supportsVision(descriptor: inactive))
        XCTAssertFalse(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: inactive)
                .contains("speculative_decoding")
        )

        let concurrent = AFMMLXRuntimeVisionPolicy.runtimeDescriptor(
            declared: declared,
            architecture: architecture,
            qualification: qualification,
            factory: .vlm,
            mtpEnabled: true,
            mtpBindingModelID: declared.modelID.rawValue,
            concurrentServing: true
        )
        XCTAssertFalse(
            AFMMLXCapabilityPresentation.modelCapabilityLabels(descriptor: concurrent)
                .contains("speculative_decoding"),
            "Concurrent serving forces autoregressive decoding and must not advertise speculation."
        )
    }

    func testCancellationGateReflectsTaskCancellation() async {
        XCTAssertTrue(AFMMLXTaskCancellationGate.shouldContinue)

        let observedCancellation = await Task { () -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            return !AFMMLXTaskCancellationGate.shouldContinue
        }.value

        XCTAssertTrue(observedCancellation)
    }

    private func makeDescriptor(
        capabilities: AFMModelCapabilities
    ) -> AFMModelDescriptor {
        AFMModelDescriptor(
            providerID: "mlx",
            modelID: "test-model",
            displayName: "test-model",
            capabilities: capabilities,
            privacyBoundary: .device
        )
    }
}
