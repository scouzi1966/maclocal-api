import Foundation
import XCTest
@testable import AFMKitMLX

final class AFMDownloadProgressTests: XCTestCase {
    func testEnrichmentReportsActiveAndCompletedFiles() {
        let root = Progress(totalUnitCount: 300)
        let complete = Progress(totalUnitCount: 100)
        let active = Progress(totalUnitCount: 100)
        let pending = Progress(totalUnitCount: 100)
        complete.completedUnitCount = 100
        active.completedUnitCount = 25

        let files = [
            AFMDownloadProgressUserInfo.File(path: "config.json", expectedBytes: 100, destination: nil, progress: complete, transport: "lfs"),
            AFMDownloadProgressUserInfo.File(path: "weights-01.safetensors", expectedBytes: 100, destination: nil, progress: active, transport: "xet"),
            AFMDownloadProgressUserInfo.File(path: "weights-02.safetensors", expectedBytes: 100, destination: nil, progress: pending, transport: "pending"),
        ]
        AFMDownloadProgressUserInfo.enrich(root, files: files)

        XCTAssertEqual(root.userInfo[AFMDownloadProgressUserInfo.completedFiles] as? Int, 1)
        XCTAssertEqual(root.userInfo[AFMDownloadProgressUserInfo.totalFiles] as? Int, 3)
        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentFiles] as? [String],
            ["weights-01.safetensors"])
        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentTransports] as? [String],
            ["xet"])
    }

    func testEnrichmentNamesFirstPendingFileBeforeBytesArrive() {
        let root = Progress(totalUnitCount: 200)
        let first = Progress(totalUnitCount: 100)
        let second = Progress(totalUnitCount: 100)

        AFMDownloadProgressUserInfo.enrich(
            root,
            files: [
                AFMDownloadProgressUserInfo.File(path: "a.bin", expectedBytes: 100, destination: nil, progress: first),
                AFMDownloadProgressUserInfo.File(path: "b.bin", expectedBytes: 100, destination: nil, progress: second),
            ])

        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentFiles] as? [String],
            ["a.bin"])
        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentTransports] as? [String],
            ["pending"])
        XCTAssertEqual(root.userInfo[AFMDownloadProgressUserInfo.completedFiles] as? Int, 0)
    }

    func testEnrichmentPublishesFallbackTransportForPendingFile() {
        let root = Progress(totalUnitCount: 100)
        let child = Progress(totalUnitCount: 100)
        let file = AFMDownloadProgressUserInfo.File(
            path: "weights.safetensors",
            expectedBytes: 100,
            destination: nil,
            progress: child,
            transport: "xet")
        file.setTransport("xet-fallback-lfs")

        AFMDownloadProgressUserInfo.enrich(root, files: [file])

        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentFiles] as? [String],
            ["weights.safetensors"])
        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentTransports] as? [String],
            ["xet-fallback-lfs"])
    }

    func testEnrichmentSamplesGrowingXetDestination() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("weights.safetensors")
        try Data(repeating: 0xA5, count: 64).write(to: destination)

        let root = Progress(totalUnitCount: 100)
        let child = Progress(totalUnitCount: 100)
        AFMDownloadProgressUserInfo.enrich(
            root,
            files: [
                AFMDownloadProgressUserInfo.File(
                    path: "weights.safetensors",
                    expectedBytes: 100,
                    destination: destination,
                    progress: child),
            ])

        XCTAssertEqual(child.completedUnitCount, 64)
        XCTAssertEqual(root.completedUnitCount, 64)
        XCTAssertEqual(
            root.userInfo[AFMDownloadProgressUserInfo.currentFiles] as? [String],
            ["weights.safetensors"])
    }
}
