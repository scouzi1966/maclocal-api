import CryptoKit
import Foundation
import XCTest
@testable import AFMEvaluationHost

private actor AsyncStartBarrier {
    private let participantCount: Int
    private var continuations: [CheckedContinuation<Void, Never>] = []

    init(participantCount: Int) {
        self.participantCount = participantCount
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            continuations.append(continuation)
            guard continuations.count == participantCount else { return }
            let waiting = continuations
            continuations.removeAll(keepingCapacity: false)
            for continuation in waiting {
                continuation.resume()
            }
        }
    }
}

final class AFMEvaluationHostTests: XCTestCase {
    func testBundledComprehensiveSuiteStaysInMaclocalHost() throws {
        let url = try XCTUnwrap(
            AFMEvaluationSuiteStore.bundledSuiteURL(named: "comprehensive"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let suite = try AFMEvaluationSuiteStore(
            rootDirectory: temporaryDirectory()).load(named: "comprehensive")
        XCTAssertEqual(suite.cases.count, 91)

        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.count, 63_469)
        XCTAssertEqual(
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            "a77abd1e3b0b32122dafbd89a37f7c5480537c4d779bd0492a51852fa52b6e28")
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
        XCTAssertThrowsError(try AFMEvaluationCLIPlan.resolve(
            evaluate: true, bench: false, suites: [], list: true,
            scaffold: nil, validate: nil, noOpen: false))
        XCTAssertThrowsError(try AFMEvaluationCLIPlan.resolve(
            evaluate: false, bench: false, suites: [], list: false,
            scaffold: "new", validate: nil, noOpen: true))
    }

    func testConcurrentRunDirectoriesAreCreatedAtomically() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let barrier = AsyncStartBarrier(participantCount: 16)
        let urls = try await withThrowingTaskGroup(of: URL.self, returning: [URL].self) { group in
            for _ in 0..<16 {
                group.addTask {
                    await barrier.wait()
                    return try AFMEvaluationSuiteStore(rootDirectory: root).makeRunDirectory(
                        model: "org/model",
                        suites: ["comprehensive"],
                        date: date)
                }
            }

            var results: [URL] = []
            for try await url in group {
                results.append(url)
            }
            return results
        }

        XCTAssertEqual(Set(urls.map(\.path)).count, 16)
        for url in urls {
            var isDirectory: ObjCBool = false
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory))
            XCTAssertTrue(isDirectory.boolValue)
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o700)
        }
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
    }
}
