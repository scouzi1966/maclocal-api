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

    func testMalformedCustomSuiteDoesNotBlockValidDiscoveryOrNamedLoad() throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("{not-json".utf8).write(to: root.appendingPathComponent("broken.json"))
        try Data("""
        {
          "schemaVersion": 1,
          "name": "valid",
          "description": "Still discoverable.",
          "cases": [{"id":"one","prompt":"Say one"}]
        }
        """.utf8).write(to: root.appendingPathComponent("valid.json"))

        let store = AFMEvaluationSuiteStore(rootDirectory: root)
        XCTAssertEqual(try store.load(named: "valid").name, "valid")
        XCTAssertTrue(try store.discover().contains { $0.name == "valid" })
        XCTAssertFalse(try store.discover().contains { $0.name == "broken" })
    }

    func testMatchesCaseIsBundledOnlyAndMustReferenceEarlierCase() throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("matching.json")
        try Data("""
        {
          "schemaVersion": 1,
          "name": "matching",
          "description": "Cross-case matching.",
          "cases": [
            {"id":"first","prompt":"one"},
            {"id":"second","prompt":"two","expectations":{"matchesCase":"first"}}
          ]
        }
        """.utf8).write(to: url)
        let store = AFMEvaluationSuiteStore(rootDirectory: root)

        XCTAssertThrowsError(try store.decode(url: url, origin: .custom)) { error in
            XCTAssertTrue(error.localizedDescription.contains("reserved for bundled suites"))
        }
        XCTAssertNoThrow(try store.decode(url: url, origin: .bundled))

        try Data("""
        {
          "schemaVersion": 1,
          "name": "matching",
          "description": "Invalid forward match.",
          "cases": [
            {"id":"first","prompt":"one","expectations":{"matchesCase":"second"}},
            {"id":"second","prompt":"two"}
          ]
        }
        """.utf8).write(to: url)
        XCTAssertThrowsError(try store.decode(url: url, origin: .bundled)) { error in
            XCTAssertTrue(error.localizedDescription.contains("earlier case in the same suite"))
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

    func testRunPolicyRejectsUnboundedAggregateOutput() throws {
        let base = AFMEvaluationParameters(maxTokens: 256)
        let acceptable = AFMEvaluationSuite(
            name: "acceptable",
            description: "Within the run budget.",
            cases: (0..<10).map { .init(id: "case-\($0)", prompt: "test") })
        XCTAssertNoThrow(try AFMEvaluationRunPolicy.validatePlannedOutput(
            suites: [acceptable],
            baseParameters: base))

        let oversized = AFMEvaluationSuite(
            name: "oversized",
            description: "Exceeds the aggregate run budget.",
            defaults: .init(maxTokens: 32_768),
            cases: (0..<31).map { .init(id: "case-\($0)", prompt: "test") })
        XCTAssertThrowsError(try AFMEvaluationRunPolicy.validatePlannedOutput(
            suites: [oversized],
            baseParameters: base)) { error in
            XCTAssertTrue(error.localizedDescription.contains("1000000 output tokens"))
        }

        let oneCase = AFMEvaluationSuite(
            name: "one-case",
            description: "Valid suite whose CLI default is invalid.",
            cases: [.init(id: "case", prompt: "test")])
        XCTAssertThrowsError(try AFMEvaluationRunPolicy.validatePlannedOutput(
            suites: [oneCase],
            baseParameters: .init(maxTokens: 0))) { error in
            XCTAssertTrue(error.localizedDescription.contains("maxTokens must be 1...32768"))
        }
    }

    func testThroughputFallsBackWhenEngineGenerationTimeIsInvalid() {
        XCTAssertEqual(
            AFMEvaluationRunPolicy.tokensPerSecond(
                completionTokens: 50,
                generationTime: 2,
                duration: 5),
            25)
        XCTAssertEqual(
            AFMEvaluationRunPolicy.tokensPerSecond(
                completionTokens: 50,
                generationTime: 0,
                duration: 5),
            10)
        XCTAssertNil(AFMEvaluationRunPolicy.tokensPerSecond(
            completionTokens: 0,
            generationTime: 1,
            duration: 1))
        XCTAssertNil(AFMEvaluationRunPolicy.tokensPerSecond(
            completionTokens: 1,
            generationTime: .nan,
            duration: 0))
    }

    func testSnapshotPolicyIsBoundedByCaseCountAndElapsedTime() {
        let now = Date(timeIntervalSince1970: 1_000)
        XCTAssertTrue(AFMEvaluationRunPolicy.shouldWriteSnapshot(
            completedCases: 1,
            lastSnapshotAt: nil,
            now: now))
        XCTAssertFalse(AFMEvaluationRunPolicy.shouldWriteSnapshot(
            completedCases: 2,
            lastSnapshotAt: now,
            now: now.addingTimeInterval(29)))
        XCTAssertTrue(AFMEvaluationRunPolicy.shouldWriteSnapshot(
            completedCases: 25,
            lastSnapshotAt: now,
            now: now.addingTimeInterval(1)))
        XCTAssertTrue(AFMEvaluationRunPolicy.shouldWriteSnapshot(
            completedCases: 2,
            lastSnapshotAt: now,
            now: now.addingTimeInterval(30)))
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-eval-tests-\(UUID().uuidString)", isDirectory: true)
    }
}
