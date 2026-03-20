import Foundation
import XCTest

@testable import MacLocalAPI

final class CacheProfileExporterTests: XCTestCase {
    func testAppendCreatesJSONLLines() throws {
        let path = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cache-profile-export-\(UUID().uuidString).jsonl")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        CacheProfileExporter.append(record: [
            "phase": "restore",
            "mode": "non-streaming",
            "cached_tokens": 12,
            "prompt_time_s": 0.25,
        ], to: path)
        CacheProfileExporter.append(record: [
            "phase": "save",
            "mode": "streaming",
            "cached_tokens": 20,
            "prompt_time_s": 0.10,
        ], to: path)

        let contents = try String(contentsOfFile: path, encoding: .utf8)
        let lines = contents.split(separator: "\n", omittingEmptySubsequences: true)

        XCTAssertEqual(lines.count, 2)

        let first = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(lines[0].utf8)) as? [String: Any]
        )
        let second = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(lines[1].utf8)) as? [String: Any]
        )

        XCTAssertEqual(first["phase"] as? String, "restore")
        XCTAssertEqual(first["mode"] as? String, "non-streaming")
        XCTAssertEqual(first["cached_tokens"] as? Int, 12)

        XCTAssertEqual(second["phase"] as? String, "save")
        XCTAssertEqual(second["mode"] as? String, "streaming")
        XCTAssertEqual(second["cached_tokens"] as? Int, 20)
    }
}
