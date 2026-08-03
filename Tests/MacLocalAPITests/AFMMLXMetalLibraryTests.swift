import Foundation
import MLX
import XCTest
@testable import AFMKitMLX

final class AFMMLXMetalLibraryTests: XCTestCase {
    func testMLXRuntimeLoadsStagedMetallib() {
        let result = MLXArray([Float(1), Float(2)]) + 1

        MLX.eval(result)

        XCTAssertEqual(result.asArray(Float.self), [2, 3])
    }

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
