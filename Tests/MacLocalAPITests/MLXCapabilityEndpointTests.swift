import AFMKitCore
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
