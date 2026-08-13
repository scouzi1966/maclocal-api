import XCTest
@testable import AFMKitDwarfStar

final class AFMDwarfStarHubResolverTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    func testSelectorExcludesDSparkSupportArtifact() throws {
        let selected = try AFMDwarfStarHubSelector.selectModel(
            from: [
                .init(path: "DeepSeek-DSpark.gguf", size: 6_000),
                .init(path: "DeepSeek-main.gguf", size: 150_000)
            ],
            repositoryID: "owner/model",
            physicalMemory: 1_000_000)

        XCTAssertEqual(selected.path, "DeepSeek-main.gguf")
    }

    func testSelectorChoosesLargestModelThatFitsMemoryBudget() throws {
        let selected = try AFMDwarfStarHubSelector.selectModel(
            from: [
                .init(path: "q2.gguf", size: 20),
                .init(path: "q4.gguf", size: 70),
                .init(path: "q8.gguf", size: 95)
            ],
            repositoryID: "owner/model",
            physicalMemory: 100,
            memoryFraction: 0.8)

        XCTAssertEqual(selected.path, "q4.gguf")
    }

    func testExplicitFileOverridesMemorySelection() throws {
        let selected = try AFMDwarfStarHubSelector.selectModel(
            from: [
                .init(path: "small.gguf", size: 20),
                .init(path: "large.gguf", size: 95)
            ],
            repositoryID: "owner/model",
            requestedPath: "large.gguf",
            physicalMemory: 100)

        XCTAssertEqual(selected.path, "large.gguf")
    }

    func testSelectorRejectsRepositoryWithOnlySupportGGUF() {
        XCTAssertThrowsError(try AFMDwarfStarHubSelector.selectModel(
            from: [.init(path: "dspark-support.gguf", size: 10)],
            repositoryID: "owner/model")) { error in
                XCTAssertEqual(
                    error as? AFMDwarfStarHubSelectionError,
                    .noModelGGUF("owner/model"))
            }
    }

    func testAppendSegmentPreservesAndExtendsPartialDownload() throws {
        let directory = try temporaryDirectory()
        let partial = directory.appendingPathComponent("model.incomplete")
        let segment = directory.appendingPathComponent("model.segment")
        try Data("first".utf8).write(to: partial)
        try Data("second".utf8).write(to: segment)

        try AFMDwarfStarResumableDownload.appendSegment(segment, to: partial, expectedBytes: 11)

        XCTAssertEqual(try Data(contentsOf: partial), Data("firstsecond".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: segment.path))
    }

    func testAdoptSegmentMovesFirstRangeWithoutCopying() throws {
        let directory = try temporaryDirectory()
        let partial = directory.appendingPathComponent("model.incomplete")
        let segment = directory.appendingPathComponent("model.segment")
        try Data("first-range".utf8).write(to: segment)

        try AFMDwarfStarResumableDownload.adoptSegment(segment, as: partial, expectedBytes: 100)

        XCTAssertEqual(try Data(contentsOf: partial), Data("first-range".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: segment.path))
    }

    func testAdoptSegmentRecoversRangeLeftByInterruptedProcess() throws {
        let directory = try temporaryDirectory()
        let partial = directory.appendingPathComponent("model.incomplete")
        let segment = directory.appendingPathComponent("model.segment")
        try Data("first".utf8).write(to: partial)
        try Data("-recovered".utf8).write(to: segment)

        try AFMDwarfStarResumableDownload.adoptSegment(segment, as: partial, expectedBytes: 100)

        XCTAssertEqual(try Data(contentsOf: partial), Data("first-recovered".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: segment.path))
    }

    func testPublishRejectsTruncatedDownloadAndPreservesPartial() throws {
        let directory = try temporaryDirectory()
        let partial = directory.appendingPathComponent("model.incomplete")
        let blob = directory.appendingPathComponent("model.gguf")
        try Data("short".utf8).write(to: partial)

        XCTAssertThrowsError(try AFMDwarfStarResumableDownload.publish(
            partial: partial,
            blob: blob,
            expectedBytes: 100,
            expectedSHA256: nil))
        XCTAssertTrue(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: blob.path))
    }

    func testPublishRejectsChecksumMismatchAndPreservesPartial() throws {
        let directory = try temporaryDirectory()
        let partial = directory.appendingPathComponent("model.incomplete")
        let blob = directory.appendingPathComponent("model.gguf")
        try Data("payload".utf8).write(to: partial)

        XCTAssertThrowsError(try AFMDwarfStarResumableDownload.publish(
            partial: partial,
            blob: blob,
            expectedBytes: 7,
            expectedSHA256: String(repeating: "0", count: 64)))
        XCTAssertTrue(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: blob.path))
    }

    func testPublishAtomicallyMovesVerifiedDownload() throws {
        let directory = try temporaryDirectory()
        let partial = directory.appendingPathComponent("model.incomplete")
        let blob = directory.appendingPathComponent("model.gguf")
        try Data("payload".utf8).write(to: partial)
        let digest = try AFMDwarfStarResumableDownload.sha256(partial)

        try AFMDwarfStarResumableDownload.publish(
            partial: partial,
            blob: blob,
            expectedBytes: 7,
            expectedSHA256: digest)

        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
        XCTAssertEqual(try Data(contentsOf: blob), Data("payload".utf8))
    }

    func testDetailedErrorIncludesUnderlyingFailure() {
        let underlying = NSError(
            domain: "download.transport",
            code: 42,
            userInfo: [NSLocalizedDescriptionKey: "connection reset"])
        let wrapper = NSError(
            domain: "download.wrapper",
            code: 1,
            userInfo: [NSUnderlyingErrorKey: underlying])

        let detail = AFMDwarfStarResumableDownload.detailedError(wrapper)

        XCTAssertTrue(detail.contains("download.wrapper"))
        XCTAssertTrue(detail.contains("download.transport"))
    }

    func testDwarfStarDetectionFollowsHuggingFaceSnapshotSymlink() throws {
        let directory = try temporaryDirectory()
        let blobs = directory.appendingPathComponent("blobs", isDirectory: true)
        let snapshot = directory.appendingPathComponent("snapshots/commit", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        let blob = blobs.appendingPathComponent("model-blob")
        try minimalGGUF(architecture: "deepseek4").write(to: blob)
        let link = snapshot.appendingPathComponent("model.gguf")
        try FileManager.default.createSymbolicLink(
            atPath: link.path,
            withDestinationPath: "../../blobs/model-blob")

        XCTAssertTrue(AFMDwarfStarCheckpointCatalog.isDwarfStarCompatibleGGUF(at: link))
        XCTAssertEqual(AFMDwarfStarCheckpointCatalog.ggufArchitecture(at: link), "deepseek4")
    }

    private func minimalGGUF(architecture: String) -> Data {
        var data = Data("GGUF".utf8)
        append(UInt32(3), to: &data)
        append(UInt64(0), to: &data) // tensor count
        append(UInt64(1), to: &data) // metadata count
        appendString("general.architecture", to: &data)
        append(UInt32(8), to: &data) // GGUF string value
        appendString(architecture, to: &data)
        return data
    }

    private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var value = value.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }

    private func appendString(_ value: String, to data: inout Data) {
        let bytes = Data(value.utf8)
        append(UInt64(bytes.count), to: &data)
        data.append(bytes)
    }
}
