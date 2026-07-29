import Foundation
import XCTest
@testable import AFMKitMLX

final class AFMMLXMetalLibraryTests: XCTestCase {
    func testFindsRenamedNestedSwiftPMResourceBundle() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let resourceDirectory = root
            .appendingPathComponent("MacLocalAPI_AFMKitMLX.bundle/Contents/Resources")
        try FileManager.default.createDirectory(
            at: resourceDirectory,
            withIntermediateDirectories: true
        )
        let expected = resourceDirectory.appendingPathComponent("default.metallib")
        XCTAssertTrue(FileManager.default.createFile(atPath: expected.path, contents: Data()))
        defer { try? FileManager.default.removeItem(at: root) }

        let resolved = MLXMetalLibrary.metallib(inResourceDirectory: root)

        XCTAssertEqual(resolved?.standardizedFileURL, expected.standardizedFileURL)
    }
}
