import Vapor
import Foundation

// MARK: - SSE Event Types

/// A single event sent to the frontend via SSE.
struct TestEvent: Codable, Sendable {
    var type: String
    var timestamp: String

    var runId: String?
    var binary: String?
    var model: String?
    var suites: [String]?
    var suite: String?
    var line: String?
    var test: String?
    var error: String?
    var passed: Int?
    var failed: Int?
    var durationS: Double?
    var returncode: Int?
    var port: Int?
    var cmd: String?
    var action: String?
    var totalPassed: Int?
    var totalFailed: Int?
    var results: [SuiteResult]?
    var response: String?
    var pid: Int?
    var command: String?
    var status: String?
    var running: Bool?

    enum CodingKeys: String, CodingKey {
        case type, timestamp, suite, line, test, error, passed, failed
        case port, cmd, action, results, binary, model, suites
        case returncode, response, pid, command, status, running
        case runId = "run_id"
        case durationS = "duration_s"
        case totalPassed = "total_passed"
        case totalFailed = "total_failed"
    }

    static func now() -> String {
        ISO8601DateFormatter().string(from: Date())
    }
}

/// Result summary for a single suite within a run.
struct SuiteResult: Codable, Sendable {
    var suite: String
    var passed: Int
    var failed: Int
    var durationS: Double
    var returncode: Int

    enum CodingKeys: String, CodingKey {
        case suite, passed, failed, returncode
        case durationS = "duration_s"
    }
}

// MARK: - Suite Definition

/// Static definition of a test suite, mirroring the Python SUITES dict.
struct SuiteDefinition {
    let id: String
    let label: String
    let description: String
    let estMinutes: Double
    let port: Int?
    let needsServer: Bool
    let cmdTemplate: [String]?
    let cmd: [String]?
    let serverFlags: [String]
    let parsePattern: String
    let envExtras: [String: String]

    /// Public-facing metadata for GET /api/suites
    var metadata: [String: Any] {
        return [
            "id": id,
            "label": label,
            "description": description,
            "est_minutes": estMinutes,
            "port": port as Any,
            "needs_server": needsServer,
        ]
    }
}

// MARK: - Run Request/Response

struct RunRequest: Codable, Content {
    var binary: String
    var model: String
    var suites: [String]
    var options: [String: [String: String]]?
}

struct AFMStartRequest: Codable, Content {
    var binary: String
    var model: String
    var opts: [String: String]?
}

// MARK: - Test Orchestration Service (Actor)

/// Thread-safe orchestrator that manages test runs, SSE subscribers, and
/// AFM server processes. Uses Swift actor isolation for safe concurrency.
actor TestOrchestrationService {
    private let repoRoot: String
    let modelCache: String
    private let healthTimeout: Double = 120.0
    private let healthPollInterval: Double = 1.0

    // SSE subscribers: each gets its own AsyncStream continuation
    private var subscribers: [UUID: AsyncStream<TestEvent>.Continuation] = [:]

    // Active run state
    private var activeRunId: String?
    private var activeProcesses: [Process] = []
    private var stopRequested: Bool = false

    // User-managed AFM server
    private var userAFMProcess: Process?
    private var userAFMLogPath: String?

    // JSONL log handle
    private var logFileHandle: FileHandle?
    private var logDir: String

    init(repoRoot: String) {
        self.repoRoot = repoRoot
        self.logDir = "\(repoRoot)/test-reports/dashboard-logs"
        self.modelCache = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/edata/models/vesta-test-cache"
    }

    // MARK: - SSE Subscription

    func subscribe(clientId: UUID) -> AsyncStream<TestEvent> {
        let (stream, continuation) = AsyncStream<TestEvent>.makeStream(bufferingPolicy: .bufferingNewest(1000))
        subscribers[clientId] = continuation
        return stream
    }

    func unsubscribe(clientId: UUID) {
        subscribers[clientId]?.finish()
        subscribers.removeValue(forKey: clientId)
    }

    /// Broadcast an event to all SSE subscribers and write to JSONL log.
    private func emit(_ event: TestEvent) {
        var ev = event
        if ev.timestamp.isEmpty {
            ev.timestamp = Self.isoNow()
        }
        for (_, continuation) in subscribers {
            continuation.yield(ev)
        }
        writeLog(ev)
    }

    private func writeLog(_ event: TestEvent) {
        guard let fh = logFileHandle else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        if let data = try? encoder.encode(event),
           var line = String(data: data, encoding: .utf8) {
            line += "\n"
            if let lineData = line.data(using: .utf8) {
                fh.write(lineData)
            }
        }
    }

    // MARK: - Logging

    private func openLog(runId: String) {
        let fm = FileManager.default
        try? fm.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        let logPath = "\(logDir)/\(runId).jsonl"
        fm.createFile(atPath: logPath, contents: nil)
        logFileHandle = FileHandle(forWritingAtPath: logPath)
        logFileHandle?.seekToEndOfFile()

        // Update LATEST.jsonl symlink
        let latestPath = "\(logDir)/LATEST.jsonl"
        try? fm.removeItem(atPath: latestPath)
        try? fm.createSymbolicLink(atPath: latestPath, withDestinationPath: "\(runId).jsonl")
    }

    private func closeLog() {
        logFileHandle?.closeFile()
        logFileHandle = nil
    }

    // MARK: - Suite Definitions

    static let suites: [SuiteDefinition] = [
        SuiteDefinition(
            id: "unit",
            label: "Unit Tests",
            description: "Swift package unit tests",
            estMinutes: 0.1,
            port: nil,
            needsServer: false,
            cmdTemplate: nil,
            cmd: ["swift", "test"],
            serverFlags: [],
            parsePattern: #"Test run with (\d+) tests.*passed"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "assertions-smoke",
            label: "Assertions (smoke)",
            description: "Server, completion, stop, logprobs, think, tools",
            estMinutes: 2,
            port: 9998,
            needsServer: true,
            cmdTemplate: [
                "./Scripts/test-assertions.sh",
                "--tier", "smoke",
                "--model", "{model}",
                "--port", "9998",
                "--bin", "{binary}",
            ],
            cmd: nil,
            serverFlags: [
                "--tool-call-parser", "afm_adaptive_xml",
                "--enable-prefix-caching",
                "--enable-grammar-constraints",
            ],
            parsePattern: #"(✅|❌)\s+(.+)"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "assertions-standard",
            label: "Assertions (standard)",
            description: "Extended assertion tests with grammar constraints",
            estMinutes: 5,
            port: 9998,
            needsServer: true,
            cmdTemplate: [
                "./Scripts/test-assertions.sh",
                "--tier", "standard",
                "--model", "{model}",
                "--port", "9998",
                "--bin", "{binary}",
                "--grammar-constraints",
            ],
            cmd: nil,
            serverFlags: [
                "--tool-call-parser", "afm_adaptive_xml",
                "--enable-prefix-caching",
                "--enable-grammar-constraints",
            ],
            parsePattern: #"(✅|❌)\s+(.+)"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "assertions-full",
            label: "Assertions (full)",
            description: "Full assertion test suite with all checks",
            estMinutes: 15,
            port: 9998,
            needsServer: true,
            cmdTemplate: [
                "./Scripts/test-assertions.sh",
                "--tier", "full",
                "--model", "{model}",
                "--port", "9998",
                "--bin", "{binary}",
                "--grammar-constraints",
            ],
            cmd: nil,
            serverFlags: [
                "--tool-call-parser", "afm_adaptive_xml",
                "--enable-prefix-caching",
                "--enable-grammar-constraints",
            ],
            parsePattern: #"(✅|❌)\s+(.+)"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "assertions-grammar",
            label: "Assertions + Grammar + Forced Parser",
            description: "Multi-model assertions with grammar and forced parser",
            estMinutes: 30,
            port: 9998,
            needsServer: false,
            cmdTemplate: [
                "./Scripts/test-assertions-multi.sh",
                "--models", "{model}",
                "--tier", "full",
                "--also-forced-parser", "qwen3_xml",
                "--grammar-constraints",
            ],
            cmd: nil,
            serverFlags: [],
            parsePattern: #"(✅|❌)\s+(.+)"#,
            envExtras: ["AFM_BINARY": "{binary}"]
        ),
        SuiteDefinition(
            id: "smart-analysis",
            label: "Comprehensive Smart Analysis",
            description: "LLM-judged quality analysis with Claude or Codex",
            estMinutes: 60,
            port: 9877,
            needsServer: false,
            cmdTemplate: [
                "./Scripts/mlx-model-test.sh",
                "--model", "{model}",
                "--prompts", "Scripts/test-llm-comprehensive.txt",
                "--smart", "1:claude",
            ],
            cmd: nil,
            serverFlags: [],
            parsePattern: #"\[(\d+)/(\d+)\].*score=(\d)"#,
            envExtras: ["AFM_BIN": "{binary}"]
        ),
        SuiteDefinition(
            id: "promptfoo",
            label: "Promptfoo Agentic Evals",
            description: "Agentic evaluation suite via promptfoo",
            estMinutes: 90,
            port: 9999,
            needsServer: false,
            cmdTemplate: [
                "./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh",
                "all",
            ],
            cmd: nil,
            serverFlags: [],
            parsePattern: #"Results: [✓✗] (\d+) passed, (\d+) failed"#,
            envExtras: ["AFM_MODEL": "{model}", "AFM_BINARY": "{binary}"]
        ),
        SuiteDefinition(
            id: "batch-correctness",
            label: "Batch Correctness B={1,2,4,8}",
            description: "Validates response correctness across batch sizes",
            estMinutes: 12,
            port: 9999,
            needsServer: true,
            cmdTemplate: [
                "python3",
                "Scripts/feature-mlx-concurrent-batch/validate_responses.py",
            ],
            cmd: nil,
            serverFlags: ["--concurrent", "8"],
            parsePattern: #"(PASS|FAIL)"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "batch-mixed",
            label: "Batch Mixed Workload",
            description: "Mixed workload validation with concurrent batching",
            estMinutes: 20,
            port: 9999,
            needsServer: true,
            cmdTemplate: [
                "python3",
                "Scripts/feature-mlx-concurrent-batch/validate_mixed_workload.py",
            ],
            cmd: nil,
            serverFlags: ["--concurrent", "8"],
            parsePattern: #"(PASS|FAIL)"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "batch-multiturn",
            label: "Batch Multiturn Prefix",
            description: "Multiturn prefix caching validation",
            estMinutes: 20,
            port: 9999,
            needsServer: true,
            cmdTemplate: [
                "python3",
                "Scripts/feature-mlx-concurrent-batch/validate_multiturn_prefix.py",
            ],
            cmd: nil,
            serverFlags: ["--concurrent", "8", "--enable-prefix-caching"],
            parsePattern: #"(PASS|FAIL)"#,
            envExtras: [:]
        ),
        SuiteDefinition(
            id: "openai-compat",
            label: "OpenAI Compat Evals",
            description: "OpenAI API compatibility evaluation suite",
            estMinutes: 8,
            port: 9999,
            needsServer: false,
            cmdTemplate: [
                "python3",
                "Scripts/feature-codex-optimize-api/test-openai-compat-evals.py",
                "--start-server",
                "--model", "{model}",
            ],
            cmd: nil,
            serverFlags: [],
            parsePattern: #"(PASS|FAIL|✅|❌)"#,
            envExtras: ["AFM_BINARY": "{binary}"]
        ),
        SuiteDefinition(
            id: "guided-json",
            label: "Guided JSON Evals",
            description: "Structured JSON output evaluation",
            estMinutes: 12,
            port: 9999,
            needsServer: false,
            cmdTemplate: [
                "python3",
                "Scripts/feature-codex-optimize-api/test-guided-json-evals.py",
                "--start-server",
                "--model", "{model}",
            ],
            cmd: nil,
            serverFlags: [],
            parsePattern: #"(PASS|FAIL|✅|❌)"#,
            envExtras: ["AFM_BINARY": "{binary}"]
        ),
        SuiteDefinition(
            id: "gpu-profile",
            label: "GPU Profile",
            description: "GPU performance profiling report",
            estMinutes: 1,
            port: nil,
            needsServer: false,
            cmdTemplate: [
                "python3",
                "Scripts/gpu-profile-report.py",
                "{model}",
            ],
            cmd: nil,
            serverFlags: [],
            parsePattern: #"Report:"#,
            envExtras: ["AFM_BIN": "{binary}"]
        ),
    ]

    static func suiteById(_ id: String) -> SuiteDefinition? {
        return suites.first(where: { $0.id == id })
    }

    // MARK: - Run Orchestration

    var isRunActive: Bool { activeRunId != nil }
    var currentRunId: String? { activeRunId }

    func startRun(binary: String, model: String, suiteNames: [String],
                  options: [String: [String: String]]?) -> String {
        let runId = Self.makeRunId()
        activeRunId = runId
        stopRequested = false

        emit(TestEvent(
            type: "trigger",
            timestamp: Self.isoNow(),
            runId: runId,
            binary: binary,
            model: model,
            suites: suiteNames,
            action: "user_clicked_run"
        ))

        // Launch orchestration in a detached task
        Task.detached { [weak self] in
            await self?.orchestrateRun(
                runId: runId, binary: binary, model: model,
                suiteNames: suiteNames, options: options ?? [:]
            )
        }

        return runId
    }

    func stopRun() {
        stopRequested = true
        emit(TestEvent(type: "trigger", timestamp: Self.isoNow(), action: "user_clicked_stop"))
        for proc in activeProcesses {
            Self.killProcessGroup(proc)
        }
        activeProcesses.removeAll()
    }

    private func orchestrateRun(runId: String, binary: String, model: String,
                                suiteNames: [String],
                                options: [String: [String: String]]) async {
        openLog(runId: runId)

        emit(TestEvent(
            type: "config",
            timestamp: Self.isoNow(),
            runId: runId,
            binary: binary,
            model: model,
            suites: suiteNames
        ))

        var totalPassed = 0
        var totalFailed = 0
        var results: [SuiteResult] = []

        // Partition suites: no-server first, then grouped by port
        var noServer: [String] = []
        var portGroups: [Int: [String]] = [:]

        for name in suiteNames {
            guard let sdef = Self.suiteById(name) else { continue }
            if !sdef.needsServer {
                noServer.append(name)
            } else {
                let p = sdef.port ?? 0
                portGroups[p, default: []].append(name)
            }
        }

        // 1. Run non-server suites
        for name in noServer {
            if stopRequested { break }
            guard let sdef = Self.suiteById(name) else { continue }
            let appliedDef = Self.applySuiteOptions(name: name, sdef: sdef, options: options)
            let r = await runSuite(name: name, sdef: appliedDef, binary: binary, model: model)
            results.append(r)
            totalPassed += r.passed
            totalFailed += r.failed
        }

        // 2. Run port groups with server lifecycle
        for port in portGroups.keys.sorted() {
            if stopRequested { break }
            let names = portGroups[port]!

            // Use first suite's server flags
            guard let firstSdef = Self.suiteById(names[0]) else { continue }
            let appliedFirst = Self.applySuiteOptions(name: names[0], sdef: firstSdef, options: options)
            let serverFlags = appliedFirst.serverFlags

            // Start AFM server
            let afmProc = await startAFMServer(binary: binary, model: model,
                                               port: port, extraFlags: serverFlags)

            if let afmProc = afmProc {
                activeProcesses.append(afmProc)

                defer {
                    Self.killProcess(afmProc)
                    activeProcesses.removeAll(where: { $0 === afmProc })
                }

                // Check server is alive
                guard afmProc.isRunning else {
                    emit(TestEvent(type: "server_error", timestamp: Self.isoNow(),
                                   error: "Server exited prematurely", port: port))
                    continue
                }

                for name in names {
                    if stopRequested { break }
                    guard let sdef = Self.suiteById(name) else { continue }
                    let appliedDef = Self.applySuiteOptions(name: name, sdef: sdef, options: options)
                    let r = await runSuite(name: name, sdef: appliedDef, binary: binary, model: model)
                    results.append(r)
                    totalPassed += r.passed
                    totalFailed += r.failed
                }
            }
        }

        emit(TestEvent(
            type: "done",
            timestamp: Self.isoNow(),
            runId: runId,
            totalPassed: totalPassed,
            totalFailed: totalFailed,
            results: results
        ))

        closeLog()
        activeRunId = nil
    }

    // MARK: - Single Suite Runner

    private func runSuite(name: String, sdef: SuiteDefinition,
                          binary: String, model: String) async -> SuiteResult {
        let startTime = Date()
        var passed = 0
        var failed = 0

        emit(TestEvent(type: "suite_start", timestamp: Self.isoNow(), suite: name))

        // Build command
        let cmd: [String]
        if let staticCmd = sdef.cmd {
            cmd = staticCmd
        } else if let template = sdef.cmdTemplate {
            cmd = Self.expandTemplate(template, binary: binary, model: model)
        } else {
            emit(TestEvent(type: "suite_error", timestamp: Self.isoNow(),
                           suite: name, error: "No command defined"))
            return SuiteResult(suite: name, passed: 0, failed: 0, durationS: 0, returncode: -1)
        }

        // Build environment
        var env = ProcessInfo.processInfo.environment
        env["MACAFM_MLX_MODEL_CACHE"] = modelCache
        for (k, v) in sdef.envExtras {
            let expanded = v.replacingOccurrences(of: "{binary}", with: binary)
                            .replacingOccurrences(of: "{model}", with: model)
            env[k] = expanded
        }

        let returncode: Int32

        do {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = cmd
            process.currentDirectoryURL = URL(fileURLWithPath: repoRoot)
            process.environment = env

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            try process.run()
            activeProcesses.append(process)

            // Stream stdout line-by-line as it arrives (non-blocking via AsyncStream bridge)
            let fileHandle = pipe.fileHandleForReading
            let pattern = sdef.parsePattern

            let (lineStream, lineContinuation) = AsyncStream<String>.makeStream(
                bufferingPolicy: .bufferingNewest(4096)
            )

            // Detached reader: polls availableData on a tight loop until process exits
            let readerTask = Task.detached {
                var buffer = Data()
                let newline = UInt8(ascii: "\n")
                while true {
                    let chunk = fileHandle.availableData
                    if chunk.isEmpty {
                        if !process.isRunning { break }
                        try? await Task.sleep(nanoseconds: 40_000_000) // 40ms
                        continue
                    }
                    buffer.append(chunk)
                    // Flush all complete lines
                    while let idx = buffer.firstIndex(of: newline) {
                        let lineData = buffer[buffer.startIndex..<idx]
                        if let line = String(data: lineData, encoding: .utf8) {
                            lineContinuation.yield(line)
                        }
                        buffer.removeSubrange(buffer.startIndex...idx)
                    }
                }
                // Flush any remaining partial line
                if !buffer.isEmpty, let line = String(data: buffer, encoding: .utf8), !line.isEmpty {
                    lineContinuation.yield(line)
                }
                lineContinuation.finish()
            }

            // Process lines as they arrive, emitting SSE events in real time
            for await rawLine in lineStream {
                if stopRequested {
                    Self.killProcessGroup(process)
                    break
                }
                let trimmedLine = rawLine.trimmingCharacters(in: .init(charactersIn: "\r"))
                guard !trimmedLine.isEmpty else { continue }

                emit(TestEvent(type: "stdout", timestamp: Self.isoNow(),
                               suite: name, line: trimmedLine))

                // Parse for pass/fail indicators
                if let regex = try? NSRegularExpression(pattern: pattern),
                   let match = regex.firstMatch(in: trimmedLine,
                                                range: NSRange(trimmedLine.startIndex..., in: trimmedLine)) {
                    let firstGroup: String
                    if match.numberOfRanges > 1,
                       let range = Range(match.range(at: 1), in: trimmedLine) {
                        firstGroup = String(trimmedLine[range])
                    } else {
                        firstGroup = ""
                    }

                    let testName: String
                    if match.numberOfRanges > 2,
                       let range = Range(match.range(at: 2), in: trimmedLine) {
                        testName = String(trimmedLine[range]).trimmingCharacters(in: .whitespaces)
                    } else {
                        testName = trimmedLine.trimmingCharacters(in: .whitespaces)
                    }

                    switch firstGroup {
                    case "\u{2705}", "PASS":
                        passed += 1
                        emit(TestEvent(type: "test_pass", timestamp: Self.isoNow(),
                                       suite: name, test: testName))
                    case "\u{274c}", "FAIL":
                        failed += 1
                        emit(TestEvent(type: "test_fail", timestamp: Self.isoNow(),
                                       suite: name, test: testName, error: trimmedLine))
                    default:
                        // For unit tests: "Test run with N tests passed"
                        if trimmedLine.lowercased().contains("passed"), let n = Int(firstGroup) {
                            passed += n
                        }
                    }
                }
            }

            await readerTask.value
            process.waitUntilExit()
            returncode = process.terminationStatus
            activeProcesses.removeAll(where: { $0 === process })

        } catch {
            emit(TestEvent(type: "suite_error", timestamp: Self.isoNow(),
                           suite: name, error: error.localizedDescription))
            return SuiteResult(suite: name, passed: passed, failed: failed,
                               durationS: Date().timeIntervalSince(startTime), returncode: -1)
        }

        let durationS = round(Date().timeIntervalSince(startTime) * 10) / 10

        emit(TestEvent(
            type: "suite_end",
            timestamp: Self.isoNow(),
            suite: name,
            passed: passed,
            failed: failed,
            durationS: durationS,
            returncode: Int(returncode)
        ))

        return SuiteResult(suite: name, passed: passed, failed: failed,
                           durationS: durationS, returncode: Int(returncode))
    }

    // MARK: - AFM Server Lifecycle

    private func startAFMServer(binary: String, model: String,
                                port: Int, extraFlags: [String]) async -> Process? {
        var cmd = [binary, "mlx", "-m", model, "--port", String(port)]
        cmd.append(contentsOf: extraFlags)

        var env = ProcessInfo.processInfo.environment
        env["MACAFM_MLX_MODEL_CACHE"] = modelCache

        emit(TestEvent(type: "server_start", timestamp: Self.isoNow(),
                        port: port, cmd: cmd.joined(separator: " ")))

        let process = Process()
        process.executableURL = URL(fileURLWithPath: cmd[0])
        process.arguments = Array(cmd.dropFirst())
        process.currentDirectoryURL = URL(fileURLWithPath: repoRoot)
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
        } catch {
            emit(TestEvent(type: "server_error", timestamp: Self.isoNow(),
                           error: "Failed to start: \(error.localizedDescription)", port: port))
            return nil
        }

        // Stream server stdout in background so SSE subscribers see live logs
        let fileHandle = pipe.fileHandleForReading
        let drainPort = port
        Task.detached { [weak self] in
            var buffer = Data()
            let newline = UInt8(ascii: "\n")
            while true {
                let chunk = fileHandle.availableData
                if chunk.isEmpty {
                    if !process.isRunning { break }
                    try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
                    continue
                }
                buffer.append(chunk)
                while let idx = buffer.firstIndex(of: newline) {
                    let lineData = buffer[buffer.startIndex..<idx]
                    if let line = String(data: lineData, encoding: .utf8), !line.isEmpty {
                        await self?.emit(TestEvent(type: "server_log", timestamp: Self.isoNow(),
                                                   line: line, port: drainPort))
                    }
                    buffer.removeSubrange(buffer.startIndex...idx)
                }
            }
        }

        // Wait for health check
        let healthy = await waitForServer(port: port, timeout: healthTimeout,
                                          poll: healthPollInterval, process: process)

        if healthy {
            emit(TestEvent(type: "server_ready", timestamp: Self.isoNow(), port: port))
        } else {
            emit(TestEvent(type: "server_timeout", timestamp: Self.isoNow(), port: port))
            Self.killProcess(process)
            return nil
        }

        return process
    }

    /// Poll the health endpoint until the server responds or timeout elapses.
    private func waitForServer(port: Int, timeout: Double,
                               poll: Double, process: Process) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if stopRequested { return false }
            if !process.isRunning { return false }

            // Try HTTP health check
            if let url = URL(string: "http://127.0.0.1:\(port)/health") {
                var request = URLRequest(url: url)
                request.timeoutInterval = 2
                do {
                    let (_, response) = try await URLSession.shared.data(for: request)
                    if let httpResponse = response as? HTTPURLResponse,
                       httpResponse.statusCode == 200 {
                        return true
                    }
                } catch {
                    // Server not ready yet
                }
            }

            try? await Task.sleep(nanoseconds: UInt64(poll * 1_000_000_000))
        }
        return false
    }

    // MARK: - User AFM Server Management

    func startUserAFMServer(binary: String, model: String,
                            opts: [String: String]) async -> (pid: Int?, port: Int, command: String, error: String?) {
        // Check if already running
        if let proc = userAFMProcess, proc.isRunning {
            return (pid: Int(proc.processIdentifier), port: 0,
                    command: "", error: "AFM server already running")
        }

        // Build command from options
        var cmd = [binary, "mlx", "-m", model]
        let port = opts.intValue(for: "port") ?? 9998
        cmd.append(contentsOf: ["--port", String(port)])

        if let hostname = opts.stringValue(for: "hostname"), hostname != "127.0.0.1" {
            cmd.append(contentsOf: ["-H", hostname])
        }
        if let maxTokens = opts.intValue(for: "maxTokens"), maxTokens != 8192 {
            cmd.append(contentsOf: ["--max-tokens", String(maxTokens)])
        }
        if let temp = opts.stringValue(for: "temperature"), !temp.isEmpty {
            cmd.append(contentsOf: ["-t", temp])
        }
        if let topP = opts.stringValue(for: "topP"), !topP.isEmpty {
            cmd.append(contentsOf: ["--top-p", topP])
        }
        if let topK = opts.intValue(for: "topK"), topK > 0 {
            cmd.append(contentsOf: ["--top-k", String(topK)])
        }
        if let minP = opts.doubleValue(for: "minP"), minP > 0 {
            cmd.append(contentsOf: ["--min-p", String(minP)])
        }
        if let concurrent = opts.intValue(for: "concurrent"), concurrent > 1 {
            cmd.append(contentsOf: ["--concurrent", String(concurrent)])
        }
        if let seed = opts.stringValue(for: "seed"), !seed.isEmpty {
            cmd.append(contentsOf: ["--seed", seed])
        }
        if let presence = opts.doubleValue(for: "presencePenalty"), presence > 0 {
            cmd.append(contentsOf: ["--presence-penalty", String(presence)])
        }
        if let rep = opts.doubleValue(for: "repetitionPenalty"), rep > 0 {
            cmd.append(contentsOf: ["--repetition-penalty", String(rep)])
        }
        if let maxKv = opts.stringValue(for: "maxKvSize") {
            cmd.append(contentsOf: ["--max-kv-size", maxKv])
        }
        if let kvBits = opts.stringValue(for: "kvBits") {
            cmd.append(contentsOf: ["--kv-bits", kvBits])
        }
        if let prefill = opts.stringValue(for: "prefillStepSize") {
            cmd.append(contentsOf: ["--prefill-step-size", prefill])
        }
        if let parser = opts.stringValue(for: "toolCallParser") {
            cmd.append(contentsOf: ["--tool-call-parser", parser])
        }
        if let stop = opts.stringValue(for: "stop") {
            cmd.append(contentsOf: ["--stop", stop])
        }
        if opts.boolValue(for: "enablePrefixCaching") == true { cmd.append("--enable-prefix-caching") }
        if opts.boolValue(for: "enableGrammarConstraints") == true { cmd.append("--enable-grammar-constraints") }
        if opts.boolValue(for: "noThink") == true { cmd.append("--no-think") }
        if opts.boolValue(for: "noStreaming") == true { cmd.append("--no-streaming") }
        if opts.boolValue(for: "raw") == true { cmd.append("--raw") }
        if opts.boolValue(for: "vlm") == true { cmd.append("--vlm") }
        if opts.boolValue(for: "webui") == true { cmd.append("--webui") }
        if opts.boolValue(for: "verbose") == true { cmd.append("-v") }
        if opts.boolValue(for: "gpuProfile") == true { cmd.append("--gpu-profile") }

        let commandStr = cmd.joined(separator: " ")

        var env = ProcessInfo.processInfo.environment
        env["MACAFM_MLX_MODEL_CACHE"] = modelCache

        let fm = FileManager.default
        try? fm.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        let serverLogPath = "\(logDir)/afm-server.log"
        fm.createFile(atPath: serverLogPath, contents: nil)
        userAFMLogPath = serverLogPath

        let process = Process()
        process.executableURL = URL(fileURLWithPath: cmd[0])
        process.arguments = Array(cmd.dropFirst())
        process.currentDirectoryURL = URL(fileURLWithPath: repoRoot)
        process.environment = env

        if let logHandle = FileHandle(forWritingAtPath: serverLogPath) {
            process.standardOutput = logHandle
            process.standardError = logHandle
        }

        do {
            try process.run()
        } catch {
            return (pid: nil, port: port, command: commandStr, error: error.localizedDescription)
        }

        userAFMProcess = process

        emit(TestEvent(type: "trigger", timestamp: Self.isoNow(),
                       action: "user_started_afm_server",
                       pid: Int(process.processIdentifier), command: commandStr))

        // Wait for readiness
        let healthy = await waitForServer(port: port, timeout: 120,
                                          poll: 1.0, process: process)
        if healthy {
            return (pid: Int(process.processIdentifier), port: port,
                    command: commandStr, error: nil)
        } else {
            Self.killProcess(process)
            userAFMProcess = nil
            return (pid: nil, port: port, command: commandStr,
                    error: "Server failed to start (health check timeout)")
        }
    }

    func stopUserAFMServer() {
        if let proc = userAFMProcess, proc.isRunning {
            Self.killProcess(proc)
            emit(TestEvent(type: "trigger", timestamp: Self.isoNow(),
                           action: "user_stopped_afm_server"))
        }
        userAFMProcess = nil
    }

    func getAFMLogs(maxLines: Int, filter: String?) -> (lines: [String], running: Bool) {
        let running = userAFMProcess?.isRunning ?? false
        guard let logPath = userAFMLogPath,
              let data = FileManager.default.contents(atPath: logPath),
              let content = String(data: data, encoding: .utf8) else {
            return (lines: [], running: running)
        }

        var allLines = content.components(separatedBy: "\n")

        if let filter = filter, !filter.isEmpty {
            allLines = allLines.filter { $0.localizedCaseInsensitiveContains(filter) }
        }

        let start = max(0, allLines.count - maxLines)
        let trimmed = Array(allLines[start...])
        return (lines: trimmed, running: running)
    }

    // MARK: - Helpers

    static func isoNow() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: Date())
    }

    static func makeRunId() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter.string(from: Date())
    }

    static func expandTemplate(_ template: [String], binary: String, model: String) -> [String] {
        return template.map {
            $0.replacingOccurrences(of: "{binary}", with: binary)
              .replacingOccurrences(of: "{model}", with: model)
        }
    }

    static func killProcess(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(5)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if process.isRunning {
            process.interrupt()
        }
    }

    /// Kill the entire process group rooted at `process` (catches child processes
    /// spawned by shell scripts, e.g. the `afm` server started by test scripts).
    static func killProcessGroup(_ process: Process) {
        guard process.isRunning else { return }
        let pid = process.processIdentifier
        // Try SIGTERM on process group; fall back to SIGTERM on process itself
        if kill(-pid, SIGTERM) != 0 {
            process.terminate()
        }
        let deadline = Date().addingTimeInterval(5)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if process.isRunning {
            if kill(-pid, SIGKILL) != 0 {
                process.interrupt()
            }
        }
    }

    /// Apply per-suite user options to a suite definition, returning a modified copy.
    static func applySuiteOptions(name: String, sdef: SuiteDefinition,
                                  options: [String: [String: String]]) -> SuiteDefinition {
        guard let opts = options[name], !opts.isEmpty else { return sdef }

        var cmdTemplate = sdef.cmdTemplate ?? []
        var serverFlags = sdef.serverFlags

        // Assertions: --section, --grammar-constraints toggle
        if name.hasPrefix("assertions-") && name != "assertions-grammar" {
            if let section = opts.stringValue(for: "section"), !section.isEmpty {
                cmdTemplate.append(contentsOf: ["--section", section])
            }
            if opts.boolValue(for: "grammar") == false {
                cmdTemplate.removeAll(where: { $0 == "--grammar-constraints" })
                serverFlags.removeAll(where: { $0 == "--enable-grammar-constraints" })
            }
        }

        // Assertions-grammar: tier, parser
        if name == "assertions-grammar" {
            if let tier = opts.stringValue(for: "tier") {
                cmdTemplate = cmdTemplate.enumerated().map { (i, a) in
                    if a == "full" && i > 0 && cmdTemplate[i-1] == "--tier" { return tier }
                    return a
                }
            }
            if let parser = opts.stringValue(for: "forced_parser") {
                if parser == "none" {
                    var newCmd: [String] = []
                    var skipNext = false
                    for a in cmdTemplate {
                        if skipNext { skipNext = false; continue }
                        if a == "--also-forced-parser" { skipNext = true; continue }
                        newCmd.append(a)
                    }
                    cmdTemplate = newCmd
                } else {
                    cmdTemplate = cmdTemplate.enumerated().map { (i, a) in
                        if a == "qwen3_xml" && i > 0 && cmdTemplate[i-1] == "--also-forced-parser" { return parser }
                        return a
                    }
                }
            }
        }

        // Smart analysis: judge, batch_mode, tests
        if name == "smart-analysis" {
            let judge = opts.stringValue(for: "judge") ?? "claude"
            let batch = opts.stringValue(for: "batch_mode") ?? "1"
            let smartVal = batch != "0" ? "\(batch):\(judge)" : judge
            cmdTemplate = cmdTemplate.enumerated().map { (i, a) in
                if i > 0 && cmdTemplate[i-1] == "--smart" { return smartVal }
                return a
            }
            if let tests = opts.stringValue(for: "tests") {
                cmdTemplate.append(contentsOf: ["--tests", tests])
            }
        }

        // Promptfoo: mode
        if name == "promptfoo" {
            if let mode = opts.stringValue(for: "mode"), mode != "all" {
                cmdTemplate = cmdTemplate.map { $0 == "all" ? mode : $0 }
            }
        }

        // Batch tests: concurrent, batch_sizes, prefix_caching
        if name.hasPrefix("batch-") {
            if let concurrent = opts.intValue(for: "concurrent") {
                var newFlags: [String] = []
                var skipNext = false
                for f in serverFlags {
                    if skipNext { skipNext = false; continue }
                    if f == "--concurrent" {
                        newFlags.append(contentsOf: ["--concurrent", String(concurrent)])
                        skipNext = true
                        continue
                    }
                    newFlags.append(f)
                }
                serverFlags = newFlags
            }
            if let sizes = opts.stringValue(for: "batch_sizes") {
                cmdTemplate.append(contentsOf: sizes.split(separator: ",").map(String.init))
            }
            if name == "batch-multiturn" && opts.boolValue(for: "prefix_caching") == false {
                serverFlags.removeAll(where: { $0 == "--enable-prefix-caching" })
            }
        }

        // GPU profile: max_tokens
        if name == "gpu-profile", let maxTokens = opts.stringValue(for: "max_tokens") {
            cmdTemplate.append(maxTokens)
        }

        return SuiteDefinition(
            id: sdef.id, label: sdef.label, description: sdef.description,
            estMinutes: sdef.estMinutes, port: sdef.port, needsServer: sdef.needsServer,
            cmdTemplate: cmdTemplate, cmd: sdef.cmd, serverFlags: serverFlags,
            parsePattern: sdef.parsePattern, envExtras: sdef.envExtras
        )
    }
}

// MARK: - Option value helpers

/// Convenience accessors for option strings (which may contain ints, bools, etc.)
extension Dictionary where Key == String, Value == String {
    func intValue(for key: String) -> Int? {
        guard let v = self[key] else { return nil }
        return Int(v)
    }

    func doubleValue(for key: String) -> Double? {
        guard let v = self[key] else { return nil }
        return Double(v)
    }

    func boolValue(for key: String) -> Bool? {
        guard let v = self[key] else { return nil }
        return v == "true" || v == "1" || v == "yes"
    }

    func stringValue(for key: String) -> String? {
        guard let v = self[key], !v.isEmpty else { return nil }
        return v
    }
}

// MARK: - Controller

/// Vapor RouteCollection for the test dashboard GUI.
/// Registered when `--test-gui` flag is passed.
/// Serves the SPA at /test-dashboard/ and provides REST/SSE APIs.
final class TestDashboardController: RouteCollection, @unchecked Sendable {

    private let repoRoot: String
    private let orchestrator: TestOrchestrationService
    private let dashboardHTMLPath: String?

    init(repoRoot: String, dashboardHTMLPath: String) {
        self.repoRoot = repoRoot
        self.orchestrator = TestOrchestrationService(repoRoot: repoRoot)
        self.dashboardHTMLPath = dashboardHTMLPath
    }

    /// Find the test dashboard HTML using the same multi-location search as webui.
    /// Works for: dev builds (.build/release), Homebrew (/opt/homebrew/share), pip (share/test-dashboard).
    private static func findDashboardPath() -> String? {
        let fm = FileManager.default
        let cwd = fm.currentDirectoryPath

        let executablePath = CommandLine.arguments[0]
        let executableURL: URL
        if executablePath.hasPrefix("/") {
            executableURL = URL(fileURLWithPath: executablePath)
        } else {
            executableURL = URL(fileURLWithPath: cwd).appendingPathComponent(executablePath)
        }
        let execDir = executableURL.deletingLastPathComponent().standardized.path

        let paths = [
            // SPM resource bundle (release build)
            "\(execDir)/MacLocalAPI_MacLocalAPI.bundle/test-dashboard/index.html",
            // Bundled with executable
            "\(execDir)/Resources/test-dashboard/index.html",
            "\(execDir)/../Resources/test-dashboard/index.html",
            "\(execDir)/../../Resources/test-dashboard/index.html",
            "\(execDir)/../../../Resources/test-dashboard/index.html",
            // pip: share directory
            "\(execDir)/../share/test-dashboard/index.html",
            // Homebrew: share directory
            "\(execDir)/../share/afm/test-dashboard/index.html",
            "/opt/homebrew/share/afm/test-dashboard/index.html",
            "/usr/local/share/afm/test-dashboard/index.html",
            // Development: Resources in CWD
            "\(cwd)/Resources/test-dashboard/index.html",
            // Development: Scripts directory
            "\(cwd)/Scripts/test-dashboard/index.html",
        ]

        for path in paths {
            // Check both raw and standardized (handles /tmp vs /private/tmp)
            if fm.fileExists(atPath: path) {
                return path
            }
            let resolved = URL(fileURLWithPath: path).standardized.path
            if fm.fileExists(atPath: resolved) {
                return resolved
            }
        }

        // Last resort: Bundle.main resource bundle
        if let bundlePath = Bundle.main.path(forResource: "index", ofType: "html", inDirectory: "test-dashboard") {
            return bundlePath
        }

        return nil
    }

    func boot(routes: RoutesBuilder) throws {
        let td = routes.grouped("test-dashboard")

        // SPA + static assets
        td.get(use: serveDashboard)
        td.get("index.html", use: serveDashboard)
        td.get("_app", "**", use: serveAppAsset)

        // OPTIONS handler for CORS preflight
        td.on(.OPTIONS, "**", use: handleCORSPreflight)

        // API endpoints
        let api = td.grouped("api")
        api.get("project-root", use: getProjectRoot)
        api.get("browse-dir", use: browseDir)
        api.get("binary", use: getBinary)
        api.get("models", use: getModels)
        api.get("suites", use: getSuites)
        api.get("preflight", use: getPreflight)
        api.get("events", use: getEvents)
        api.get("results", use: listResults)
        api.get("results", ":runId", use: getResult)
        api.get("reports", "**", use: serveReport)
        api.get("afm", "logs", use: getAFMLogs)

        api.on(.POST, "run", body: .collect(maxSize: "1mb"), use: postRun)
        api.on(.POST, "stop", use: postStop)
        api.on(.POST, "afm", "start", body: .collect(maxSize: "1mb"), use: postAFMStart)
        api.on(.POST, "afm", "stop", use: postAFMStop)
        api.on(.POST, "promptfoo-view", use: postPromptfooView)
    }

    // MARK: - CORS

    private func handleCORSPreflight(req: Request) async throws -> Response {
        let response = Response(status: .noContent)
        addCORSHeaders(response)
        response.headers.add(name: .accessControlAllowMethods, value: "GET, POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type")
        response.headers.add(name: "Access-Control-Max-Age", value: "86400")
        return response
    }

    private func addCORSHeaders(_ response: Response) {
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
    }

    private func jsonResponse<T: Encodable>(_ data: T, status: HTTPStatus = .ok) throws -> Response {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let jsonData = try encoder.encode(data)
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json; charset=utf-8")
        addCORSHeaders(response)
        response.body = .init(data: jsonData)
        return response
    }

    private func jsonDictResponse(_ dict: [String: Any], status: HTTPStatus = .ok) throws -> Response {
        let data = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json; charset=utf-8")
        addCORSHeaders(response)
        response.body = .init(data: data)
        return response
    }

    // MARK: - GET /test-dashboard/ — Serve SPA

    func serveDashboard(req: Request) async throws -> Response {
        guard let htmlPath = dashboardHTMLPath,
              let data = FileManager.default.contents(atPath: htmlPath) else {
            throw Abort(.notFound, reason: "Dashboard HTML not found. Searched: SPM bundle, Resources/, share/, Scripts/test-dashboard/")
        }

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "text/html; charset=utf-8")
        response.headers.add(name: .cacheControl, value: "no-cache")
        addCORSHeaders(response)
        response.body = .init(data: data)
        return response
    }

    // MARK: - GET /test-dashboard/_app/**

    func serveAppAsset(req: Request) async throws -> Response {
        let subpath = req.parameters.getCatchall().joined(separator: "/")
        let bundleBase = Bundle.module.bundlePath + "/test-dashboard/_app/"
        let filePath = bundleBase + subpath

        guard let data = FileManager.default.contents(atPath: filePath) else {
            throw Abort(.notFound)
        }

        let ext = URL(fileURLWithPath: subpath).pathExtension
        let contentType: String
        switch ext {
        case "js":  contentType = "application/javascript; charset=utf-8"
        case "css": contentType = "text/css; charset=utf-8"
        case "json": contentType = "application/json"
        case "map": contentType = "application/json"
        default:    contentType = "application/octet-stream"
        }

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: contentType)
        response.headers.add(name: .cacheControl, value: "public, max-age=31536000, immutable")
        response.body = .init(data: data)
        return response
    }

    // MARK: - GET /test-dashboard/api/project-root

    func getProjectRoot(req: Request) async throws -> Response {
        let overridePath = req.query[String.self, at: "path"]

        if let path = overridePath, !path.isEmpty {
            let scriptsDir = "\(path)/Scripts"
            let valid = FileManager.default.isReadableFile(atPath: scriptsDir)
            return try jsonDictResponse(["path": path, "valid": valid])
        }

        return try jsonDictResponse(["path": repoRoot, "valid": true])
    }

    // MARK: - GET /test-dashboard/api/browse-dir

    func browseDir(req: Request) async throws -> Response {
        let browsePath = req.query[String.self, at: "path"] ?? "/"
        let fm = FileManager.default
        let resolved = URL(fileURLWithPath: browsePath).standardized.path

        let showHidden = (req.query[String.self, at: "show_hidden"] ?? "false") == "true"
        do {
            let contents = try fm.contentsOfDirectory(atPath: resolved)
            let dirs = contents
                .filter { showHidden || !$0.hasPrefix(".") }
                .filter { var isDir: ObjCBool = false; return fm.fileExists(atPath: "\(resolved)/\($0)", isDirectory: &isDir) && isDir.boolValue }
                .sorted()
            let hasScripts = fm.fileExists(atPath: "\(resolved)/Scripts")
            return try jsonDictResponse([
                "path": resolved,
                "entries": dirs,
                "has_scripts": hasScripts
            ])
        } catch {
            return try jsonDictResponse([
                "path": resolved,
                "entries": [] as [String],
                "has_scripts": false,
                "error": error.localizedDescription
            ])
        }
    }

    // MARK: - GET /test-dashboard/api/binary

    func getBinary(req: Request) async throws -> Response {
        let root = req.query[String.self, at: "root"] ?? repoRoot
        let candidates = [
            ".build/arm64-apple-macosx/release/afm",
            ".build/release/afm",
            ".build/arm64-apple-macosx/debug/afm",
        ]

        for candidate in candidates {
            let fullPath = "\(root)/\(candidate)"
            if FileManager.default.isExecutableFile(atPath: fullPath) {
                let version = await getVersion(binary: fullPath)
                return try jsonDictResponse([
                    "path": fullPath,
                    "version": version,
                    "found": true,
                ])
            }
        }

        return try jsonDictResponse(["path": "", "version": "", "found": false])
    }

    private func getVersion(binary: String) async -> String {
        do {
            let (output, _) = try await runCommand([binary, "--version"])
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return "error: \(error.localizedDescription)"
        }
    }

    // MARK: - GET /test-dashboard/api/models

    func getModels(req: Request) async throws -> Response {
        let root = req.query[String.self, at: "root"] ?? repoRoot
        let scriptPath = "\(root)/Scripts/list-models.sh"
        guard FileManager.default.isExecutableFile(atPath: scriptPath) else {
            return try jsonResponse([String: String]())
        }

        do {
            let (output, _) = try await runCommand(
                [scriptPath],
                env: ["MACAFM_MLX_MODEL_CACHE": orchestrator.modelCache]
            )

            var models: [[String: Any]] = []
            for line in output.components(separatedBy: "\n") {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty || trimmed.hasPrefix("MACAFM") ||
                   trimmed.hasPrefix("Error") || trimmed.contains("models found") {
                    continue
                }

                // Format: "org/model                     19.0 GB"
                let pattern = #"^(\S+)\s+([\d.]+)\s+GB"#
                if let regex = try? NSRegularExpression(pattern: pattern),
                   let match = regex.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)) {
                    if let idRange = Range(match.range(at: 1), in: trimmed),
                       let sizeRange = Range(match.range(at: 2), in: trimmed) {
                        models.append([
                            "id": String(trimmed[idRange]),
                            "size": "\(trimmed[sizeRange]) GB",
                        ])
                    }
                }
            }

            return try jsonDictResponse(["models": models])
        } catch {
            return try jsonDictResponse(["models": [String]()])
        }
    }

    // MARK: - GET /test-dashboard/api/suites

    func getSuites(req: Request) async throws -> Response {
        let suitesInfo: [[String: Any]] = TestOrchestrationService.suites.map { sdef in
            [
                "id": sdef.id,
                "label": sdef.label,
                "description": sdef.description,
                "est_minutes": sdef.estMinutes,
                "port": sdef.port as Any,
                "needs_server": sdef.needsServer,
            ]
        }
        return try jsonDictResponse(["suites": suitesInfo])
    }

    // MARK: - GET /test-dashboard/api/preflight

    func getPreflight(req: Request) async throws -> Response {
        guard let binary = req.query[String.self, at: "binary"], !binary.isEmpty else {
            throw Abort(.badRequest, reason: "Missing 'binary' query parameter")
        }

        // 1. Version check
        let versionStr = await getVersion(binary: binary)
        let hasSHA = versionStr.range(of: #"-[0-9a-f]{7,}"#, options: .regularExpression) != nil
        let versionResult: [String: Any] = [
            "status": hasSHA ? "pass" : "warn",
            "value": versionStr,
        ]

        // 2. Metallib check
        let metallibResult = checkMetallib(binary: binary)

        // 3. Relocated check
        let relocatedResult = await checkRelocated(binary: binary)

        // 4. Bundle.module check
        let bundleResult = await checkBundleModule()

        let result: [String: Any] = [
            "version": versionResult,
            "metallib": metallibResult,
            "relocated": relocatedResult,
            "bundle_module": bundleResult,
        ]

        return try jsonDictResponse(result)
    }

    private func checkMetallib(binary: String) -> [String: Any] {
        let binaryDir = (binary as NSString).deletingLastPathComponent
        let fm = FileManager.default

        // Check loose metallib
        let loosePath = "\(binaryDir)/default.metallib"
        if fm.fileExists(atPath: loosePath) {
            return ["status": "pass", "location": loosePath]
        }

        // Check in .bundle directories
        if let entries = try? fm.contentsOfDirectory(atPath: binaryDir) {
            for entry in entries where entry.hasSuffix(".bundle") {
                let bundlePath = "\(binaryDir)/\(entry)"
                // Resources path
                let resourcesMetallib = "\(bundlePath)/Contents/Resources/default.metallib"
                if fm.fileExists(atPath: resourcesMetallib) {
                    return ["status": "pass", "location": resourcesMetallib]
                }
                // Flat bundle
                let flatMetallib = "\(bundlePath)/default.metallib"
                if fm.fileExists(atPath: flatMetallib) {
                    return ["status": "pass", "location": flatMetallib]
                }
            }
        }

        return ["status": "fail", "location": ""]
    }

    private func checkRelocated(binary: String) async -> [String: Any] {
        let fm = FileManager.default
        let tmpDir = NSTemporaryDirectory() + "afm-relocate-\(UUID().uuidString)"
        let testPort = 19876

        do {
            try fm.createDirectory(atPath: tmpDir, withIntermediateDirectories: true)
            let dstBin = "\(tmpDir)/afm"
            try fm.copyItem(atPath: binary, toPath: dstBin)

            // Make executable
            try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dstBin)

            // Copy metallib and bundles from binary directory
            let binaryDir = (binary as NSString).deletingLastPathComponent
            if let entries = try? fm.contentsOfDirectory(atPath: binaryDir) {
                for entry in entries {
                    if entry == "afm" { continue }
                    let src = "\(binaryDir)/\(entry)"
                    let dst = "\(tmpDir)/\(entry)"
                    if entry.hasSuffix(".metallib") {
                        try? fm.copyItem(atPath: src, toPath: dst)
                    } else if entry.hasSuffix(".bundle") {
                        try? fm.copyItem(atPath: src, toPath: dst)
                    }
                }
            }

            // Start relocated binary
            let process = Process()
            process.executableURL = URL(fileURLWithPath: dstBin)
            process.arguments = [
                "mlx", "-m", "mlx-community/SmolLM3-3B-4bit",
                "--port", String(testPort), "--max-tokens", "16",
            ]
            process.currentDirectoryURL = URL(fileURLWithPath: tmpDir)
            var env = ProcessInfo.processInfo.environment
            env["MACAFM_MLX_MODEL_CACHE"] = "/Volumes/edata/models/vesta-test-cache"
            process.environment = env
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice

            try process.run()

            defer {
                TestOrchestrationService.killProcess(process)
                try? fm.removeItem(atPath: tmpDir)
            }

            // Wait for server
            let deadline = Date().addingTimeInterval(60)
            var serverReady = false
            while Date() < deadline {
                if !process.isRunning { break }
                if let url = URL(string: "http://127.0.0.1:\(testPort)/health") {
                    var request = URLRequest(url: url)
                    request.timeoutInterval = 2
                    if let (_, response) = try? await URLSession.shared.data(for: request),
                       let httpResponse = response as? HTTPURLResponse,
                       httpResponse.statusCode == 200 {
                        serverReady = true
                        break
                    }
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }

            guard serverReady else {
                return ["status": "fail", "error": "Relocated server failed to start"]
            }

            // Send a completion request
            guard let chatURL = URL(string: "http://127.0.0.1:\(testPort)/v1/chat/completions") else {
                return ["status": "fail", "error": "Invalid URL"]
            }

            var chatRequest = URLRequest(url: chatURL)
            chatRequest.httpMethod = "POST"
            chatRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            chatRequest.timeoutInterval = 30

            let body: [String: Any] = [
                "model": "mlx-community/SmolLM3-3B-4bit",
                "messages": [["role": "user", "content": "Say hi"]],
                "max_tokens": 5,
            ]
            chatRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

            let (data, chatResponse) = try await URLSession.shared.data(for: chatRequest)
            if let httpResponse = chatResponse as? HTTPURLResponse, httpResponse.statusCode == 200 {
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let first = choices.first,
                   let message = first["message"] as? [String: Any],
                   let content = message["content"] as? String, !content.isEmpty {
                    let preview = String(content.prefix(50))
                    return ["status": "pass", "response": preview]
                }
                return ["status": "pass"]
            }
            return ["status": "fail", "error": "HTTP \((chatResponse as? HTTPURLResponse)?.statusCode ?? 0)"]

        } catch {
            try? fm.removeItem(atPath: tmpDir)
            return ["status": "fail", "error": error.localizedDescription]
        }
    }

    private func checkBundleModule() async -> [String: Any] {
        let sourcesDir = "\(repoRoot)/Sources"
        guard FileManager.default.isReadableFile(atPath: sourcesDir) else {
            return ["status": "pass", "hits": 0]
        }

        do {
            let (output, _) = try await runCommand([
                "grep", "-r", "Bundle.module", sourcesDir, "--include=*.swift",
            ])

            var hits = 0
            for line in output.components(separatedBy: "\n") {
                let stripped: String
                if let colonIndex = line.range(of: ":", range: line.index(after: line.startIndex)..<line.endIndex) {
                    stripped = String(line[colonIndex.upperBound...]).trimmingCharacters(in: .whitespaces)
                } else {
                    stripped = line.trimmingCharacters(in: .whitespaces)
                }
                if stripped.hasPrefix("//") || stripped.hasPrefix("/*") || stripped.hasPrefix("*") {
                    continue
                }
                if !stripped.isEmpty { hits += 1 }
            }

            let status = hits == 0 ? "pass" : "warn"
            return ["status": status, "hits": hits]
        } catch {
            return ["status": "error", "error": error.localizedDescription, "hits": -1]
        }
    }

    // MARK: - GET /test-dashboard/api/events — SSE Streaming

    func getEvents(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "text/event-stream")
        response.headers.add(name: .cacheControl, value: "no-cache")
        response.headers.add(name: .connection, value: "keep-alive")
        response.headers.add(name: "X-Accel-Buffering", value: "no")
        addCORSHeaders(response)

        let clientId = UUID()
        let orchestrator = self.orchestrator

        response.body = .init(asyncStream: { writer in
            let stream = await orchestrator.subscribe(clientId: clientId)

            // Send initial heartbeat
            try await writer.write(.buffer(.init(string: ": heartbeat\n\n")))

            // Set up a keepalive task
            let keepaliveTask = Task {
                while !Task.isCancelled {
                    try await Task.sleep(nanoseconds: 15_000_000_000) // 15 seconds
                    try await writer.write(.buffer(.init(string: ": keepalive\n\n")))
                }
            }

            defer {
                keepaliveTask.cancel()
                Task { await orchestrator.unsubscribe(clientId: clientId) }
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]

            for await event in stream {
                do {
                    let jsonData = try encoder.encode(event)
                    if let jsonString = String(data: jsonData, encoding: .utf8) {
                        try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                    }
                } catch {
                    break
                }
            }

            try await writer.write(.end)
        })

        return response
    }

    // MARK: - POST /test-dashboard/api/run

    func postRun(req: Request) async throws -> Response {
        let runRequest = try req.content.decode(RunRequest.self)

        guard !runRequest.binary.isEmpty,
              !runRequest.model.isEmpty,
              !runRequest.suites.isEmpty else {
            throw Abort(.badRequest, reason: "Required fields: binary, model, suites")
        }

        // Validate suite names
        let validIds = Set(TestOrchestrationService.suites.map(\.id))
        let invalid = runRequest.suites.filter { !validIds.contains($0) }
        if !invalid.isEmpty {
            throw Abort(.badRequest, reason: "Unknown suites: \(invalid)")
        }

        // Check if a run is already active
        let isActive = await orchestrator.isRunActive
        if isActive {
            let currentId = await orchestrator.currentRunId ?? "unknown"
            throw Abort(.conflict, reason: "Run \(currentId) already active")
        }

        let runId = await orchestrator.startRun(
            binary: runRequest.binary,
            model: runRequest.model,
            suiteNames: runRequest.suites,
            options: runRequest.options
        )

        return try jsonDictResponse(["run_id": runId, "suites": runRequest.suites])
    }

    // MARK: - POST /test-dashboard/api/stop

    func postStop(req: Request) async throws -> Response {
        await orchestrator.stopRun()
        return try jsonDictResponse(["status": "stopping"])
    }

    // MARK: - POST /test-dashboard/api/afm/start

    func postAFMStart(req: Request) async throws -> Response {
        let startReq = try req.content.decode(AFMStartRequest.self)

        guard !startReq.binary.isEmpty, !startReq.model.isEmpty else {
            throw Abort(.badRequest, reason: "Required: binary, model")
        }

        let result = await orchestrator.startUserAFMServer(
            binary: startReq.binary,
            model: startReq.model,
            opts: startReq.opts ?? [:]
        )

        if let error = result.error {
            if error.contains("already running") {
                throw Abort(.conflict, reason: error)
            }
            throw Abort(.internalServerError, reason: error)
        }

        return try jsonDictResponse([
            "pid": result.pid ?? 0,
            "port": result.port,
            "command": result.command,
            "status": "running",
        ])
    }

    // MARK: - POST /test-dashboard/api/afm/stop

    func postAFMStop(req: Request) async throws -> Response {
        await orchestrator.stopUserAFMServer()
        return try jsonDictResponse(["status": "stopped"])
    }

    // MARK: - GET /test-dashboard/api/afm/logs

    func getAFMLogs(req: Request) async throws -> Response {
        let maxLines = req.query[Int.self, at: "lines"] ?? 50
        let filter = req.query[String.self, at: "filter"]

        let (lines, running) = await orchestrator.getAFMLogs(maxLines: maxLines, filter: filter)

        return try jsonDictResponse([
            "lines": lines,
            "running": running,
            "total": lines.count,
        ])
    }

    // MARK: - GET /test-dashboard/api/results

    func listResults(req: Request) async throws -> Response {
        let logDir = "\(repoRoot)/test-reports/dashboard-logs"
        let fm = FileManager.default

        guard fm.isReadableFile(atPath: logDir),
              let entries = try? fm.contentsOfDirectory(atPath: logDir) else {
            return try jsonResponse([[String: String]]())
        }

        var runs: [[String: Any]] = []
        let sortedEntries = entries.sorted().reversed()

        for fname in sortedEntries {
            guard fname.hasSuffix(".jsonl"), fname != "LATEST.jsonl" else { continue }
            let runId = String(fname.dropLast(6)) // remove .jsonl
            let fpath = "\(logDir)/\(fname)"

            guard let attrs = try? fm.attributesOfItem(atPath: fpath),
                  let size = attrs[.size] as? Int else { continue }

            // Quick-parse for summary
            var suites: [String] = []
            var totalPassed = 0
            var totalFailed = 0
            var dateStr = ""

            if let data = fm.contents(atPath: fpath),
               let content = String(data: data, encoding: .utf8) {
                for line in content.components(separatedBy: "\n") {
                    let trimmed = line.trimmingCharacters(in: .whitespaces)
                    guard !trimmed.isEmpty,
                          let lineData = trimmed.data(using: .utf8),
                          let ev = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else {
                        continue
                    }

                    let etype = ev["type"] as? String ?? ""
                    if etype == "config" {
                        suites = ev["suites"] as? [String] ?? []
                        dateStr = ev["timestamp"] as? String ?? ""
                    } else if etype == "done" {
                        totalPassed = ev["total_passed"] as? Int ?? 0
                        totalFailed = ev["total_failed"] as? Int ?? 0
                    }
                }
            }

            runs.append([
                "id": runId,
                "date": dateStr,
                "size": size,
                "suites": suites,
                "passed": totalPassed,
                "failed": totalFailed,
            ])
        }

        return try jsonDictResponse(["results": runs])
    }

    // MARK: - GET /test-dashboard/api/results/:runId

    func getResult(req: Request) async throws -> Response {
        guard let runId = req.parameters.get("runId") else {
            throw Abort(.badRequest, reason: "Missing runId")
        }

        // Sanitize runId to prevent traversal
        let safeId = runId.filter { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
        let logPath = "\(repoRoot)/test-reports/dashboard-logs/\(safeId).jsonl"

        guard FileManager.default.fileExists(atPath: logPath),
              let data = FileManager.default.contents(atPath: logPath) else {
            throw Abort(.notFound, reason: "Run \(safeId) not found")
        }

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/jsonl")
        addCORSHeaders(response)
        response.body = .init(data: data)
        return response
    }

    // MARK: - GET /test-dashboard/api/reports/**

    func serveReport(req: Request) async throws -> Response {
        let reportDir = "\(repoRoot)/test-reports"

        // Get the catchall path components
        let pathComponents = req.parameters.getCatchall()
        let relPath = pathComponents.joined(separator: "/")

        // Prevent directory traversal
        let safePath = (reportDir as NSString).appendingPathComponent(relPath)
        let normalizedSafe = (safePath as NSString).standardizingPath
        let normalizedBase = (reportDir as NSString).standardizingPath
        guard normalizedSafe.hasPrefix(normalizedBase) else {
            throw Abort(.forbidden, reason: "Directory traversal not allowed")
        }

        guard FileManager.default.fileExists(atPath: normalizedSafe) else {
            throw Abort(.notFound)
        }

        guard let data = FileManager.default.contents(atPath: normalizedSafe) else {
            throw Abort(.internalServerError, reason: "Failed to read file")
        }

        // Determine content type
        let ext = (normalizedSafe as NSString).pathExtension.lowercased()
        let contentTypeMap: [String: String] = [
            "html": "text/html",
            "json": "application/json",
            "jsonl": "application/jsonl",
            "md": "text/markdown",
            "txt": "text/plain",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "csv": "text/csv",
        ]
        let contentType = contentTypeMap[ext] ?? "application/octet-stream"

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: contentType)
        addCORSHeaders(response)
        response.body = .init(data: data)
        return response
    }

    // MARK: - POST /test-dashboard/api/promptfoo-view

    func postPromptfooView(req: Request) async throws -> Response {
        do {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = ["promptfoo", "view", "-y"]
            process.currentDirectoryURL = URL(fileURLWithPath: repoRoot)
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            try process.run()
            return try jsonDictResponse(["url": "http://localhost:15500", "status": "started"])
        } catch {
            throw Abort(.internalServerError, reason: "promptfoo not found in PATH")
        }
    }

    // MARK: - Process Helpers

    /// Run a command and return stdout + exit code.
    private func runCommand(_ args: [String], env: [String: String]? = nil) async throws -> (String, Int32) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = args
        process.currentDirectoryURL = URL(fileURLWithPath: repoRoot)

        if let env = env {
            var environment = ProcessInfo.processInfo.environment
            for (k, v) in env { environment[k] = v }
            process.environment = environment
        }

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        let output = String(data: data, encoding: .utf8) ?? ""
        return (output, process.terminationStatus)
    }
}
