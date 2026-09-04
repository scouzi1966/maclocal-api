@testable import AFMKit
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
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
    }

    func testOutputAncestorOfSourceIsRejectedBeforeOverwrite() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("models/source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: source,
            output: root,
            inspection: makeInspection(required: nil),
            overwrite: true,
            capacity: { _ in Int64.max })) { error in
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
    }

    func testSymlinkedOutputAncestorOfSourceIsRejected() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let real = root.appendingPathComponent("real", isDirectory: true)
        let source = real.appendingPathComponent("source", isDirectory: true)
        let alias = root.appendingPathComponent("alias", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: source,
            output: alias,
            inspection: makeInspection(required: nil),
            capacity: { _ in Int64.max })) { error in
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
    }

    func testFilesystemRootAndRootSymlinkAreRejected() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        let alias = root.appendingPathComponent("root-alias", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: alias, withDestinationURL: URL(fileURLWithPath: "/", isDirectory: true))

        for output in [URL(fileURLWithPath: "/", isDirectory: true), alias] {
            XCTAssertThrowsError(try MLXConversionStoragePreflight.validateLocalPaths(
                source: source,
                output: output)) { error in
                    XCTAssertTrue(error.localizedDescription.contains("filesystem or volume root"))
                }
        }
    }

    func testInjectedHiddenMountedRootIsRecognized() throws {
        let root = try makeRoot()
        let hiddenMount = root.appendingPathComponent("nobrowse-volume", isDirectory: true)
        try FileManager.default.createDirectory(at: hiddenMount, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(MLXConversionStoragePreflight.isFilesystemOrVolumeRoot(
            hiddenMount,
            mountedVolumes: [hiddenMount]))
    }

    func testResumeCreditsOnlyProviderVerifiedCompletedOutput() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        let output = root.appendingPathComponent("output", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

        let report = try MLXConversionStoragePreflight.validate(
            source: source,
            output: output,
            inspection: makeInspection(required: 600_000_000_000),
            verifiedCompletedOutputBytes: 4,
            capacity: { _ in 599_999_999_996 })
        XCTAssertEqual(report.requiredBytes, 599_999_999_996)

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: source,
            output: output,
            inspection: makeInspection(required: 600_000_000_000),
            overwrite: true,
            verifiedCompletedOutputBytes: 4,
            capacity: { _ in 599_999_999_996 }))
    }

    func testConsumerDoesNotInferCreditFromPrivateManifest() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        let output = root.appendingPathComponent("output", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let state: [String: Any] = [
            "sourceRevision": String(repeating: "a", count: 40),
            "completed": [
                "unit": [
                    "outputFile": "model-00001.safetensors",
                    "outputSize": 3,
                    "outputSHA256": String(repeating: "0", count: 64),
                ],
            ],
        ]
        try JSONSerialization.data(withJSONObject: state).write(
            to: output.appendingPathComponent(".afm-mlx-conversion.json"))

        XCTAssertThrowsError(try MLXConversionStoragePreflight.validate(
            source: source,
            output: output,
            inspection: makeInspection(required: 600_000_000_000),
            capacity: { _ in 599_999_999_999 }))
    }

    func testProviderResumeMayExceedTensorEstimateBySafeTensorOverhead() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        let report = try MLXConversionStoragePreflight.validate(
            source: source,
            output: root.appendingPathComponent("output"),
            inspection: makeInspection(required: 600_000_000_000),
            verifiedCompletedOutputBytes: 190_700_000_001,
            capacity: { _ in 409_299_999_999 })

        XCTAssertEqual(report.requiredBytes, 409_299_999_999)
        XCTAssertEqual(report.availableBytes, 409_299_999_999)
    }

    func testDeepSeekWithoutPublishedEstimatePreservesExistingBehavior() throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

        var didProbe = false
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
            capacity: { _ in
                didProbe = true
                throw CocoaError(.fileReadUnknown)
            })

        XCTAssertNil(report.requiredBytes)
        XCTAssertNil(report.availableBytes)
        XCTAssertFalse(didProbe)
    }

    func testInvalidProfileAndMissingTemplateFailBeforeModelInspection() throws {
        XCTAssertNoThrow(try MLXConversionStoragePreflight.validateProfileName(
            Qwen4ExpCheckpointConverter.Profile.afmMapped4.rawValue))
        XCTAssertThrowsError(try MLXConversionStoragePreflight.validateProfileName(
            "not-a-profile")) { error in
                XCTAssertTrue(error.localizedDescription.contains("Unknown conversion profile"))
            }
        XCTAssertThrowsError(try MLXConversionStoragePreflight.validateTemplateFile(
            "/definitely/missing/template.gguf")) { error in
                XCTAssertTrue(error.localizedDescription.contains("existing local GGUF"))
            }
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
