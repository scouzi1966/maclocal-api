import Foundation
import XCTest
import AFMKitCore
@testable import AFMKitMLX

final class AFMMLXOrphanModelPolicyTests: XCTestCase {
    func testCandidateUsesDiscoveredPackageMetadata() throws {
        let discovered = discoveredModel(
            id: "example-org/example-model",
            packagePath: "/models/example-org/example-model",
            sizeBytes: 42
        )

        let candidate = try XCTUnwrap(
            AFMMLXOrphanModelPolicy.candidate(
                from: discovered,
                includeSpecialty: false,
                registeredModelIDs: []
            )
        )

        XCTAssertEqual(candidate.id, "example-org/example-model")
        XCTAssertEqual(candidate.name, "example-model")
        XCTAssertEqual(candidate.author, "example-org")
        XCTAssertEqual(candidate.packageDirectory.path, "/models/example-org/example-model")
        XCTAssertEqual(candidate.sizeBytes, 42)
    }

    func testCandidateExcludesRegisteredAndCuratedOrphans() {
        XCTAssertNil(
            AFMMLXOrphanModelPolicy.candidate(
                from: discoveredModel(id: "example-org/registered"),
                includeSpecialty: false,
                registeredModelIDs: ["example-org/registered"]
            )
        )

        XCTAssertNil(
            AFMMLXOrphanModelPolicy.candidate(
                from: discoveredModel(id: "mlx-community/Qwen3-VL-4B-Instruct-5bit"),
                includeSpecialty: false,
                registeredModelIDs: []
            )
        )
    }

    func testCandidateRoutesSpecialtyModelsOnlyToSpecialtyScan() {
        let specialty = discoveredModel(id: "prince-canuma/Kokoro-82M")

        XCTAssertNil(
            AFMMLXOrphanModelPolicy.candidate(
                from: specialty,
                includeSpecialty: false,
                registeredModelIDs: []
            )
        )
        XCTAssertNotNil(
            AFMMLXOrphanModelPolicy.candidate(
                from: specialty,
                includeSpecialty: true,
                registeredModelIDs: []
            )
        )
    }

    func testCandidatesSortBySizeDescending() {
        let candidates = AFMMLXOrphanModelPolicy.candidates(
            from: [
                discoveredModel(id: "example-org/small", sizeBytes: 1),
                discoveredModel(id: "example-org/large", sizeBytes: 10)
            ],
            includeSpecialty: false,
            registeredModelIDs: []
        )

        XCTAssertEqual(candidates.map(\.id), ["example-org/large", "example-org/small"])
    }

    private func discoveredModel(
        id: String,
        packagePath: String = "/models/package",
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
