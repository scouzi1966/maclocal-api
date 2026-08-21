import XCTest
@testable import AFMKit

final class AFMEvaluationTests: XCTestCase {
    func testBundledComprehensiveSuiteIsPackagedAndDiscoverable() throws {
        let url = try XCTUnwrap(AFMEvaluationSuiteStore.bundledSuiteURL(named: "comprehensive"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        let root = temporaryDirectory()
        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        let suite = try store.load(named: "comprehensive")
        XCTAssertEqual(suite.name, "comprehensive")
        XCTAssertEqual(suite.cases.count, 91)
        XCTAssertEqual(suite.cases.first?.id, "greedy")
        XCTAssertEqual(suite.cases.last?.id, "strict-format")
        XCTAssertTrue(try store.discover().contains { $0.name == "comprehensive" && $0.origin == .bundled })
    }

    func testCustomSuiteParsingAndDiscovery() throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let json = """
        {
          "schemaVersion": 1,
          "name": "custom-check",
          "description": "A test suite.",
          "defaults": { "temperature": 0, "maxTokens": 64 },
          "cases": [{
            "id": "one",
            "prompt": "Say one",
            "expectations": { "contains": ["one"] }
          }]
        }
        """
        try Data(json.utf8).write(to: root.appendingPathComponent("custom-check.json"))
        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        let suite = try store.load(named: "custom-check")
        XCTAssertEqual(suite.cases.first?.id, "one")
        XCTAssertTrue(try store.discover().contains { $0.name == "custom-check" && $0.origin == .custom })
    }

    func testMalformedAndUnknownSuiteFieldsAreRejectedActionably() throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("bad.json")
        try Data("""
        {"schemaVersion":1,"name":"bad","description":"Bad.","surprise":true,
         "cases":[{"id":"x","prompt":"hello"}]}
        """.utf8).write(to: url)
        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        XCTAssertThrowsError(try store.decode(url: url)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Unknown key"))
            XCTAssertTrue(error.localizedDescription.contains("surprise"))
        }
    }

    func testSanitizationAndCollisionSafeRunPath() throws {
        XCTAssertEqual(
            AFMEvaluationSuiteStore.sanitizePathComponent("org/model name<script>"),
            "org-model-name-script")
        let root = temporaryDirectory()
        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let first = try store.makeRunDirectory(model: "org/model", suites: ["suite"], date: date)
        let second = try store.makeRunDirectory(model: "org/model", suites: ["suite"], date: date)
        XCTAssertNotEqual(first, second)
        XCTAssertTrue(second.lastPathComponent.hasSuffix("-2"))
        XCTAssertFalse(first.lastPathComponent.contains("/"))
    }

    func testDeterministicScoringPassMissAndObservation() {
        let pass = AFMEvaluationScorer.score(
            output: "{\"answer\":\"Hello\"}",
            toolCallNames: ["lookup"],
            expectations: .init(
                contains: ["hello"],
                notContains: ["secret"],
                validJSON: true,
                toolCallName: "lookup"))
        XCTAssertEqual(pass.0, .passed)
        XCTAssertTrue(pass.1.allSatisfy(\.passed))

        let miss = AFMEvaluationScorer.score(
            output: "wrong",
            expectations: .init(exact: "right"))
        XCTAssertEqual(miss.0, .missed)

        XCTAssertEqual(AFMEvaluationScorer.score(output: "anything", expectations: nil).0, .observed)
    }

    func testHTMLReportEscapesModelPromptOutputAndErrors() {
        let parameters = AFMEvaluationParameters(temperature: 0, maxTokens: 16)
        let result = AFMEvaluationCaseResult(
            suite: "suite<script>",
            caseID: "case&one",
            prompt: "<img src=x onerror=alert(1)>",
            system: nil,
            output: "<script>alert('x')</script>",
            reasoning: "a > b",
            toolCalls: [],
            outcome: .error,
            checks: [],
            error: "bad & worse",
            startedAt: Date(timeIntervalSince1970: 0),
            durationSeconds: 1,
            timeToFirstTokenSeconds: 0.2,
            promptTimeSeconds: 0.1,
            generationTimeSeconds: 0.8,
            promptTokens: 3,
            cachedPromptTokens: 0,
            completionTokens: 2,
            tokensPerSecond: 2,
            finishReason: "error",
            parameters: parameters)
        let report = AFMEvaluationRunReport(
            afmVersion: "v-test",
            model: "model<script>",
            suites: ["suite<script>"],
            startedAt: Date(timeIntervalSince1970: 0),
            finishedAt: Date(timeIntervalSince1970: 1),
            interrupted: false,
            reproducibilityCommand: "afm <unsafe>",
            system: .init(
                operatingSystem: "macOS & test",
                architecture: "arm64",
                processorCount: 8,
                physicalMemoryBytes: 16_000_000_000),
            results: [result])
        let html = AFMEvaluationReportWriter.html(for: report)
        XCTAssertFalse(html.contains("<script>alert('x')</script>"))
        XCTAssertTrue(html.contains("&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"))
        XCTAssertTrue(html.contains("bad &amp; worse"))
        XCTAssertTrue(html.contains("<!doctype html>"))
    }

    func testCLIPlanCoversDefaultAliasSuitesAndManagement() throws {
        XCTAssertEqual(
            try AFMEvaluationCLIPlan.resolve(
                evaluate: true, bench: false, suites: [], list: false,
                scaffold: nil, validate: nil, noOpen: false),
            .run(suites: ["comprehensive"], openReport: true))
        XCTAssertEqual(
            try AFMEvaluationCLIPlan.resolve(
                evaluate: false, bench: true, suites: ["a", "b"], list: false,
                scaffold: nil, validate: nil, noOpen: true),
            .run(suites: ["a", "b"], openReport: false))
        XCTAssertEqual(
            try AFMEvaluationCLIPlan.resolve(
                evaluate: false, bench: false, suites: [], list: true,
                scaffold: nil, validate: nil, noOpen: false),
            .list)
        XCTAssertThrowsError(
            try AFMEvaluationCLIPlan.resolve(
                evaluate: false, bench: false, suites: [], list: true,
                scaffold: "x", validate: nil, noOpen: false))
    }

    func testScaffoldCreatesValidSuiteAndRefusesOverwrite() throws {
        let root = temporaryDirectory()
        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        let url = try store.scaffold(named: "new-suite")
        XCTAssertEqual(try store.decode(url: url).name, "new-suite")
        XCTAssertThrowsError(try store.scaffold(named: "new-suite"))
        XCTAssertThrowsError(try store.scaffold(named: "../escape"))
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-eval-tests-\(UUID().uuidString)", isDirectory: true)
    }
}
