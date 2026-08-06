import XCTest
import AFMKitCore
import AFMKitMLX

final class AFMMLXClearablePackagePolicyTests: XCTestCase {
    func testDeduplicatesPackageRootsInDiscoveryOrder() {
        let discovered = [
            discoveredModel(
                id: "example-org/first",
                packagePath: "/models/shared-package"
            ),
            discoveredModel(
                id: "example-org/second",
                packagePath: "/models/shared-package"
            ),
            discoveredModel(
                id: "example-org/third",
                packagePath: "/models/other-package"
            )
        ]

        XCTAssertEqual(
            AFMMLXClearablePackagePolicy.packageIdentifiers(from: discovered),
            [
                "/models/shared-package",
                "/models/other-package"
            ]
        )
    }

    private func discoveredModel(
        id: String,
        packagePath: String,
        sizeBytes: Int64 = 1
    ) -> AFMMLXDiscoveredModel {
        let packageDirectory = URL(fileURLWithPath: packagePath)
        let localDirectory = packageDirectory.appendingPathComponent("snapshot")
        let descriptor = AFMModelDescriptor(
            providerID: "mlx",
            modelID: AFMModelID(rawValue: id),
            displayName: id.split(separator: "/").last.map(String.init) ?? id,
            capabilities: [.text],
            privacyBoundary: .device,
            requiresNetwork: false
        )
        return AFMMLXDiscoveredModel(
            id: AFMModelID(rawValue: id),
            loadIdentifier: id,
            localDirectory: localDirectory,
            packageDirectory: packageDirectory,
            sizeBytes: sizeBytes,
            origin: .systemCache,
            descriptor: descriptor
        )
    }
}
