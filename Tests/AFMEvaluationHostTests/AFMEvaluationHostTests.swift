import Foundation
import XCTest
@testable import AFMEvaluationHost

final class AFMEvaluationHostTests: XCTestCase {
    func testBundledComprehensiveSuiteStaysInMaclocalHost() throws {
        let url = try XCTUnwrap(
            AFMEvaluationSuiteStore.bundledSuiteURL(named: "comprehensive"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let suite = try AFMEvaluationSuiteStore(
            rootDirectory: temporaryDirectory()).load(named: "comprehensive")
        XCTAssertEqual(suite.cases.count, 91)
    }

    func testCustomDiscoveryUsesSharedStrictValidator() throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("""
        {"schemaVersion":1,"name":"valid","description":"Local suite.",
         "cases":[{"id":"one","prompt":"Say one"}]}
        """.utf8).write(to: root.appendingPathComponent("valid.json"))
        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        XCTAssertEqual(try store.load(named: "valid").name, "valid")
    }

    func testCLIPlanningRemainsHostOwned() throws {
        XCTAssertEqual(
            try AFMEvaluationCLIPlan.resolve(
                evaluate: true, bench: false, suites: [], list: false,
                scaffold: nil, validate: nil, noOpen: false),
            .run(suites: ["comprehensive"], openReport: true))
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
    }
}
