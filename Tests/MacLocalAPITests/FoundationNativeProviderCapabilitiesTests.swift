#if canImport(FoundationModels)
import AFMKit
@testable import AFMKitFoundationModels27
import XCTest

final class FoundationNativeProviderCapabilitiesTests: XCTestCase {
    func testAppleOnDeviceSnapshotUsesMinimumContextAndDevicePrivacy() {
        let snapshot = AFMFoundationNativeProviderCapabilities.appleOnDevice(
            systemContextWindow: 4_096
        )

        XCTAssertEqual(snapshot.kind, .appleOnDevice)
        XCTAssertEqual(snapshot.modelIdentifier, "apple.system.default")
        XCTAssertEqual(snapshot.contextWindow, 8_192)
        XCTAssertEqual(snapshot.privacyBoundary, .device)
        XCTAssertFalse(snapshot.requiresNetwork)
        XCTAssertNil(snapshot.entitlement)
        XCTAssertTrue(snapshot.capabilities.contains(.text))
        XCTAssertTrue(snapshot.capabilities.contains(.vision))
        XCTAssertTrue(snapshot.capabilities.contains(.toolCalling))
        XCTAssertTrue(snapshot.capabilities.contains(.structuredOutput))
        XCTAssertTrue(snapshot.supportedReasoningLevels.isEmpty)
    }

    func testPrivateCloudComputeSnapshotAdvertisesPCCBoundaryAndReasoning() {
        let snapshot = AFMFoundationNativeProviderCapabilities.privateCloudCompute()

        XCTAssertEqual(snapshot.kind, .privateCloudCompute)
        XCTAssertEqual(snapshot.modelIdentifier, "apple.private-cloud-compute")
        XCTAssertEqual(snapshot.contextWindow, 32_768)
        XCTAssertEqual(snapshot.privacyBoundary, .privateCloud)
        XCTAssertTrue(snapshot.requiresNetwork)
        XCTAssertEqual(snapshot.entitlement, "com.apple.developer.private-cloud-compute")
        XCTAssertEqual(snapshot.supportedReasoningLevels, [.light, .moderate, .deep])
    }
}
#endif
