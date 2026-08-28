import AFMKit
import AFMKitMLX
import Foundation
import XCTest

final class MLXConversionStoragePreflightTests: XCTestCase {
    func testGLMDestinationRequiresSixHundredGigabytes() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let output = root.appendingPathComponent("converted/model", isDirectory: true)
        let inspection = makeInspection(required: 600_000_000_000)

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: source,
            output: output,
            inspection: inspection,
            capacity: { _ in 599_999_999_999 })) { error in
                guard case MLXConversionStoragePreflight.PreflightError.insufficientCapacity(
                    let required, let available, _) = error
                else { return XCTFail("Unexpected error: \(error)") }
                XCTAssertEqual(required, 600_000_000_000)
                XCTAssertEqual(available, 599_999_999_999)
            }
    }

    func testCapacityProbeUsesNearestExistingDestinationParent() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        let destinationRoot = root.appendingPathComponent("destination", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: destinationRoot, withIntermediateDirectories: true)
        let output = destinationRoot.appendingPathComponent("new/model", isDirectory: true)
        var probed: URL?

        let report = try MLXConversionStoragePreflight.validate(
            source: source,
            output: output,
            inspection: makeInspection(required: 600_000_000_000),
            capacity: { url in
                probed = url
                return 700_000_000_000
            })

        XCTAssertEqual(probed, destinationRoot)
        XCTAssertEqual(report.capacityProbe, destinationRoot)
        XCTAssertEqual(report.availableBytes, 700_000_000_000)
    }

    func testRemoteOrMissingSourceIsRejectedBeforeCapacityProbe() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        var didProbe = false

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: root.appendingPathComponent("missing"),
            output: root.appendingPathComponent("output"),
            inspection: makeInspection(required: 600_000_000_000),
            capacity: { _ in
                didProbe = true
                return Int64.max
            })) { error in
                XCTAssertTrue(error.localizedDescription.contains("existing local"))
            }
        XCTAssertFalse(didProbe)
    }

    func testOutputInsideSourceIsRejected() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: source,
            output: source.appendingPathComponent("converted"),
            inspection: makeInspection(required: nil),
            capacity: { _ in Int64.max })) { error in
                XCTAssertTrue(error.localizedDescription.contains("cannot be inside"))
            }
    }

    func testDeepSeekWithoutPublishedEstimatePreservesExistingBehavior() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        let report = try MLXConversionStoragePreflight.validate(
            source: source,
            output: root.appendingPathComponent("output"),
            inspection: AFMMLXCheckpointConverter.Inspection(
                modelKind: .deepseekV4,
                defaultProfile: "native",
                supportedProfiles: ["native"],
                sourceRevision: nil,
                sourceBytes: 1,
                estimatedOutputBytes: nil,
                requiredDestinationFreeBytes: nil),
            capacity: { _ in 1 })

        XCTAssertNil(report.requiredBytes)
    }

    private func makeInspection(
        required: Int64?
    ) -> AFMMLXCheckpointConverter.Inspection {
        AFMMLXCheckpointConverter.Inspection(
            modelKind: .glm5Next,
            defaultProfile: "mlx-affine-4",
            supportedProfiles: ["mlx-affine-4"],
            sourceRevision: String(repeating: "a", count: 40),
            sourceBytes: 328_326_771_576,
            estimatedOutputBytes: 190_700_000_000,
            requiredDestinationFreeBytes: required)
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
