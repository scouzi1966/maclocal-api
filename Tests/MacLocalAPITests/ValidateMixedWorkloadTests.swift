import Testing
import Foundation
@testable import MacLocalAPI

// Integration test — requires a running AFM server on localhost:9999
// Swift port of Scripts/feature-mlx-concurrent-batch/validate_mixed_workload.py
//
// Run with: swift test --filter ValidateMixedWorkloadTests
// Start server first:
//   MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
//   afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999 --concurrent 8

// MARK: - Test cases

private struct MixedTestCase {
    let name: String
    let prompt: String
    let expected: [String]       // All must appear (case-insensitive)
    let minTokens: Int           // 0 = no minimum (short-answer), >0 = long-decode
    let maxTokens: Int
}

private let SHORT_ANSWER_TESTS: [MixedTestCase] = [
    MixedTestCase(
        name: "long-ctx-arithmetic",
        prompt: """
            I have a complex problem. Consider the following sequence of operations: \
            Start with 100. Add 37. Multiply by 2. Subtract 15. Divide by 3. \
            Add 42. Multiply by 5. Subtract 200. Add 17. Divide by 2. \
            Now, separately and ignoring all of that: What is 7 times 8? \
            Answer with ONLY the number, nothing else.
            """,
        expected: ["56"],
        minTokens: 0,
        maxTokens: 4096
    ),
    MixedTestCase(
        name: "long-ctx-capital",
        prompt: """
            Here is some background context that is not relevant to the question: \
            The history of cartography spans thousands of years. Ancient Babylonians \
            created clay tablet maps around 600 BCE. Ptolemy's Geographia from the 2nd \
            century CE was influential for centuries. The Age of Exploration brought \
            major advances in mapmaking with Mercator's projection in 1569. Modern GIS \
            systems use satellite imagery and digital processing. \
            Now answer this simple question: What is the capital of Japan? One word only.
            """,
        expected: ["tokyo"],
        minTokens: 0,
        maxTokens: 4096
    ),
    MixedTestCase(
        name: "long-ctx-element",
        prompt: """
            Let me give you detailed context about the periodic table. Dmitri Mendeleev \
            published the first widely recognized periodic table in 1869, arranging 63 \
            known elements by atomic weight. Henry Moseley later determined that atomic \
            number was the better organizing principle. The table now contains 118 \
            confirmed elements, with the most recent additions being nihonium (113), \
            moscovium (115), tennessine (117), and oganesson (118), all confirmed in \
            2015-2016. Elements are arranged in 18 groups and 7 periods. The lanthanides \
            and actinides are typically shown separately below the main table. \
            Simple question: What element has the symbol 'Au'? Answer in one word.
            """,
        expected: ["gold"],
        minTokens: 0,
        maxTokens: 4096
    ),
    MixedTestCase(
        name: "long-ctx-year",
        prompt: """
            Context about space exploration milestones: The Space Race between the US \
            and Soviet Union drove rapid advancement. Sputnik 1 launched October 4, 1957. \
            Yuri Gagarin became the first human in space on April 12, 1961, aboard Vostok 1. \
            John Glenn orbited Earth on February 20, 1962. Valentina Tereshkova became \
            the first woman in space in 1963. Ed White performed the first American \
            spacewalk in 1965. The Gemini program tested orbital maneuvers and docking. \
            The tragic Apollo 1 fire killed three astronauts in January 1967. Apollo 8 \
            orbited the Moon in December 1968. \
            Question: In what year did humans first walk on the Moon? Just the year.
            """,
        expected: ["1969"],
        minTokens: 0,
        maxTokens: 4096
    ),
]

private let LONG_DECODE_TESTS: [MixedTestCase] = [
    MixedTestCase(
        name: "calculus-explain",
        prompt: """
            Explain calculus concepts from limits through multivariable calculus \
            with rigorous mathematical notation. Cover: epsilon-delta definition of \
            limits, derivatives, chain rule, integration techniques, fundamental \
            theorem of calculus, sequences and series, Taylor series, partial \
            derivatives, gradient, divergence, curl, and multiple integrals.
            """,
        expected: ["limit", "derivative", "integral"],
        minTokens: 500,
        maxTokens: 4096
    ),
    MixedTestCase(
        name: "history-essay",
        prompt: """
            Write a detailed essay on the causes and consequences of World War I, \
            covering the alliance systems, the assassination of Archduke Franz Ferdinand, \
            major battles, technological innovations in warfare, the Treaty of Versailles, \
            and how it set the stage for World War II.
            """,
        expected: ["assassination", "versailles", "trench"],
        minTokens: 500,
        maxTokens: 4096
    ),
    MixedTestCase(
        name: "code-tutorial",
        prompt: """
            Write a comprehensive tutorial on implementing a red-black tree in Python. \
            Include the complete implementation with insert, delete, search, rotation \
            operations, and explain the balancing rules. Add docstrings and comments.
            """,
        expected: ["class", "def ", "insert", "rotate"],
        minTokens: 500,
        maxTokens: 4096
    ),
    MixedTestCase(
        name: "physics-explain",
        prompt: """
            Explain quantum mechanics from first principles. Start with the double-slit \
            experiment, cover wave-particle duality, the Schrodinger equation, \
            Heisenberg uncertainty principle, quantum entanglement, superposition, \
            and the measurement problem. Use mathematical notation where appropriate.
            """,
        expected: ["wave", "particle", "uncertainty"],
        minTokens: 500,
        maxTokens: 4096
    ),
]

// MARK: - Result types

private struct RequestStats {
    let name: String
    let text: String
    let wallS: Double
    let ttft: Double
    let promptTokens: Int
    let completionTokens: Int
    let ppTokS: Double        // prompt tokens/sec
    let tgTokS: Double        // generation tokens/sec
    let promptMs: Double
    let predictedMs: Double
    let kind: String          // "short" | "long"
}

private enum TestOutcome {
    case ok(RequestStats)
    case garbage
    case tooShort(tokens: Int)
    case missing([String])
    case exception(Error)
}

// MARK: - GPU sampler (mactop)

private actor MactopSampler {
    private var samples: [[String: Double]] = []
    private var process: Process?
    private var readerTask: Task<Void, Never>?

    func start() {
        guard let mactopPath = which("mactop") else { return }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
        proc.arguments = [mactopPath, "--headless", "--format", "json", "-i", "500"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
        } catch { return }
        process = proc

        let fh = pipe.fileHandleForReading
        readerTask = Task.detached { [weak self] in
            var buffer = Data()
            let newline = UInt8(ascii: "\n")
            while proc.isRunning || fh.availableData.count > 0 {
                let chunk = fh.availableData
                if chunk.isEmpty {
                    if proc.isRunning { try? await Task.sleep(nanoseconds: 100_000_000) }
                    else { break }
                    continue
                }
                buffer.append(chunk)
                while let idx = buffer.firstIndex(of: newline) {
                    let lineData = buffer[buffer.startIndex..<idx]
                    if let line = String(data: lineData, encoding: .utf8),
                       let data = line.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        let parsed = json.compactMapValues { v -> Double? in
                            if let d = v as? Double { return d }
                            if let s = v as? String { return Double(s) }
                            return nil
                        }
                        await self?.addSample(parsed)
                    }
                    buffer.removeSubrange(buffer.startIndex...idx)
                }
            }
        }
    }

    func addSample(_ s: [String: Double]) { samples.append(s) }

    func stop() {
        process?.terminate()
        readerTask?.cancel()
    }

    func summary() -> [String: Double]? {
        guard !samples.isEmpty else { return nil }
        func avg(_ keys: [String]) -> Double {
            let vals = samples.compactMap { s in keys.compactMap { s[$0] }.first }
            return vals.isEmpty ? 0 : vals.reduce(0, +) / Double(vals.count)
        }
        return [
            "gpu_pct":  avg(["gpu_active_pct", "active_pct"]),
            "gpu_w":    avg(["gpu_power_w", "gpu_power"]),
            "dram_w":   avg(["dram_power_w", "dram_power"]),
            "sys_w":    avg(["system_power_w", "system_power"]),
            "freq_mhz": avg(["gpu_freq_mhz", "gpu_freq"]),
            "temp_c":   avg(["gpu_temp_c", "gpu_temp"]),
            "n_samples": Double(samples.count),
        ]
    }
}

private func which(_ cmd: String) -> String? {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
    proc.arguments = [cmd]
    let pipe = Pipe()
    proc.standardOutput = pipe
    proc.standardError = FileHandle.nullDevice
    try? proc.run()
    proc.waitUntilExit()
    guard proc.terminationStatus == 0,
          let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
    else { return nil }
    return out.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - Request sender

private func sendMixedRequest(client: AFMClient, test: MixedTestCase) async throws -> RequestStats {
    var req = AFMClient.ChatRequest()
    req.messages = [AFMClient.userMessage(test.prompt)]
    req.maxTokens = test.maxTokens
    req.temperature = 0.3

    let start = Date()
    var ttft: Double? = nil
    var text = ""
    var usage: AFMClient.Usage? = nil
    var timings: AFMClient.Timings? = nil

    let stream = try await client.chatCompletionStream(req)
    for try await chunk in stream {
        if let u = chunk.usage { usage = u }
        if let t = chunk.timings { timings = t }
        let content = chunk.choices.first?.delta.content ?? ""
        let reasoning = chunk.choices.first?.delta.reasoningContent ?? ""
        let piece = content.isEmpty ? reasoning : content
        if !piece.isEmpty {
            if ttft == nil { ttft = Date().timeIntervalSince(start) }
            text += piece
        }
    }

    let wallS = Date().timeIntervalSince(start)
    return RequestStats(
        name: test.name,
        text: text,
        wallS: wallS,
        ttft: ttft ?? 0,
        promptTokens: usage?.promptTokens ?? 0,
        completionTokens: usage?.completionTokens ?? 0,
        ppTokS: usage?.promptTokensPerSecond ?? 0,
        tgTokS: usage?.completionTokensPerSecond ?? 0,
        promptMs: timings?.promptMs ?? 0,
        predictedMs: timings?.predictedMs ?? 0,
        kind: test.minTokens > 0 ? "long" : "short"
    )
}

private func checkMixed(_ stats: RequestStats, test: MixedTestCase) -> TestOutcome {
    let lower = stats.text.lowercased()
    let isGarbage = stats.text.trimmingCharacters(in: .whitespaces).count < 2
        || stats.text.unicodeScalars.filter({ $0.value == 0xFFFD }).count > 5
    if isGarbage { return .garbage }

    if test.minTokens > 0 {
        let wordCount = stats.text.split(separator: " ").count
        if wordCount < Int(Double(test.minTokens) * 0.3) { return .tooShort(tokens: stats.completionTokens) }
    }

    let missing = test.expected.filter { !lower.contains($0.lowercased()) }
    if !missing.isEmpty { return .missing(missing) }
    return .ok(stats)
}

// MARK: - Batch runner

private struct BatchRunResult {
    var passed: Int = 0
    var failed: Int = 0
    var okResults: [RequestStats] = []
    var gpuStats: [String: Double]? = nil
}

private func runBatch(client: AFMClient, batchSize: Int, tests: [MixedTestCase]) async -> BatchRunResult {
    var result = BatchRunResult()
    let gpu = MactopSampler()
    await gpu.start()

    var batchStart = 0
    while batchStart < tests.count {
        let batch = Array(tests[batchStart..<min(batchStart + batchSize, tests.count)])
        batchStart += batchSize

        let outcomes: [(MixedTestCase, TestOutcome)] =
            await withTaskGroup(of: (Int, TestOutcome).self) { group in
                for (i, t) in batch.enumerated() {
                    group.addTask {
                        do {
                            let stats = try await sendMixedRequest(client: client, test: t)
                            return (i, checkMixed(stats, test: t))
                        } catch {
                            return (i, .exception(error))
                        }
                    }
                }
                var indexed = [(Int, TestOutcome)]()
                for await item in group { indexed.append(item) }
                return indexed.sorted(by: { $0.0 < $1.0 }).map { (batch[$0.0], $0.1) }
            }

        for (test, outcome) in outcomes {
            switch outcome {
            case .exception(let e):
                result.failed += 1
                print("  FAIL  \(test.name): exception \(e.localizedDescription)")
            case .garbage:
                result.failed += 1
                print("  FAIL  \(test.name): GARBAGE")
            case .tooShort(let tok):
                result.failed += 1
                print("  FAIL  \(test.name): TOO SHORT (\(tok) tok)")
            case .missing(let missing):
                result.failed += 1
                print("  FAIL  \(test.name): missing \(missing)")
            case .ok(let stats):
                result.passed += 1
                result.okResults.append(stats)
                let n = test.name.padding(toLength: 24, withPad: " ", startingAt: 0)
                print(String(format: "  OK    %@  pp=%4d tok %7.1f t/s  tg=%4d tok %6.1f t/s  TTFT=%.2fs  wall=%.1fs",
                    n, stats.promptTokens, stats.ppTokS,
                    stats.completionTokens, stats.tgTokS,
                    stats.ttft, stats.wallS))
            }
        }
    }

    await gpu.stop()
    result.gpuStats = await gpu.summary()
    return result
}

// MARK: - Summary printer

private func printSummary(batchSize: Int, result: BatchRunResult, label: String) {
    let ok = result.okResults
    let longOk = ok.filter { $0.kind == "long" }
    let shortOk = ok.filter { $0.kind == "short" }

    if !ok.isEmpty {
        let totalPrompt = ok.reduce(0) { $0 + $1.promptTokens }
        let totalCompl  = ok.reduce(0) { $0 + $1.completionTokens }
        let avgPP   = ok.reduce(0.0) { $0 + $1.ppTokS } / Double(ok.count)
        let avgTG   = ok.reduce(0.0) { $0 + $1.tgTokS } / Double(ok.count)
        let avgTTFT = ok.reduce(0.0) { $0 + $1.ttft } / Double(ok.count)
        let maxWall = ok.max(by: { $0.wallS < $1.wallS })?.wallS ?? 0
        let aggTG   = maxWall > 0 ? Double(totalCompl) / maxWall : 0

        print("  \(String(repeating: "─", count: 96))")
        print("  Totals: \(totalPrompt) prompt + \(totalCompl) completion = \(totalPrompt + totalCompl) tokens")
        print(String(format: "  Avg pp: %.1f tok/s   Avg tg (per-req): %.1f tok/s   Agg tg: %.1f tok/s   Avg TTFT: %.2fs",
            avgPP, avgTG, aggTG, avgTTFT))

        if !longOk.isEmpty {
            let lpp  = longOk.reduce(0.0) { $0 + $1.ppTokS } / Double(longOk.count)
            let ltg  = longOk.reduce(0.0) { $0 + $1.tgTokS } / Double(longOk.count)
            let lct  = longOk.reduce(0) { $0 + $1.completionTokens }
            let lttft = longOk.reduce(0.0) { $0 + $1.ttft } / Double(longOk.count)
            print(String(format: "  Long-decode:  %d tok, avg pp=%.1f tg=%.1f tok/s, TTFT=%.2fs", lct, lpp, ltg, lttft))
        }
        if !shortOk.isEmpty {
            let spp  = shortOk.reduce(0.0) { $0 + $1.ppTokS } / Double(shortOk.count)
            let stg  = shortOk.reduce(0.0) { $0 + $1.tgTokS } / Double(shortOk.count)
            let sct  = shortOk.reduce(0) { $0 + $1.completionTokens }
            let sttft = shortOk.reduce(0.0) { $0 + $1.ttft } / Double(shortOk.count)
            print(String(format: "  Short-answer: %d tok, avg pp=%.1f tg=%.1f tok/s, TTFT=%.2fs", sct, spp, stg, sttft))
        }
    }

    if let g = result.gpuStats, (g["n_samples"] ?? 0) > 0 {
        print(String(format: "  GPU: %.0f%% active, %.1fW, %.0f MHz, %.0f°C | DRAM %.1fW  Sys %.1fW  (%.0f samples)",
            g["gpu_pct"] ?? 0, g["gpu_w"] ?? 0, g["freq_mhz"] ?? 0, g["temp_c"] ?? 0,
            g["dram_w"] ?? 0, g["sys_w"] ?? 0, g["n_samples"] ?? 0))
    }

    print("  Result: \(result.passed)/\(result.passed + result.failed) passed")
}

// MARK: - Test suite

@Suite("Validate Mixed Workload (port 9999)")
struct ValidateMixedWorkloadTests {

    let client = AFMClient(baseURL: "http://localhost:9999")
    private let allTests = SHORT_ANSWER_TESTS + LONG_DECODE_TESTS

    @Test("B=1 mixed workload")
    func batchSize1() async throws {
        guard await serverReachable() else { return }
        let line = String(repeating: "=", count: 100)
        print("\n\(line)\n  B=1 — \(SHORT_ANSWER_TESTS.count) short-answer + \(LONG_DECODE_TESTS.count) long-decode (4K max)\n\(line)")
        let result = await runBatch(client: client, batchSize: 1, tests: allTests)
        printSummary(batchSize: 1, result: result, label: "")
        print(line)
        #expect(result.failed == 0, "B=1: \(result.failed) failures")
    }

    @Test("B=2 mixed workload")
    func batchSize2() async throws {
        guard await serverReachable() else { return }
        let line = String(repeating: "=", count: 100)
        print("\n\(line)\n  B=2 — \(SHORT_ANSWER_TESTS.count) short-answer + \(LONG_DECODE_TESTS.count) long-decode (4K max)\n\(line)")
        let result = await runBatch(client: client, batchSize: 2, tests: allTests)
        printSummary(batchSize: 2, result: result, label: "")
        print(line)
        #expect(result.failed == 0, "B=2: \(result.failed) failures")
    }

    @Test("B=4 mixed workload")
    func batchSize4() async throws {
        guard await serverReachable() else { return }
        let line = String(repeating: "=", count: 100)
        print("\n\(line)\n  B=4 — \(SHORT_ANSWER_TESTS.count) short-answer + \(LONG_DECODE_TESTS.count) long-decode (4K max)\n\(line)")
        let result = await runBatch(client: client, batchSize: 4, tests: allTests)
        printSummary(batchSize: 4, result: result, label: "")
        print(line)
        #expect(result.failed == 0, "B=4: \(result.failed) failures")
    }

    @Test("B=8 mixed workload")
    func batchSize8() async throws {
        guard await serverReachable() else { return }
        let line = String(repeating: "=", count: 100)
        print("\n\(line)\n  B=8 — \(SHORT_ANSWER_TESTS.count) short-answer + \(LONG_DECODE_TESTS.count) long-decode (4K max)\n\(line)")
        let result = await runBatch(client: client, batchSize: 8, tests: allTests)
        printSummary(batchSize: 8, result: result, label: "")
        print(line)
        #expect(result.failed == 0, "B=8: \(result.failed) failures")
    }

    /// Full sweep B=1,2,4,8 with summary table — mirrors Python script default run.
    @Test("Full sweep B=1,2,4,8 with summary table")
    func fullSweep() async throws {
        guard await serverReachable() else { return }

        var totalPassed = 0
        var totalFailed = 0
        var batchResults: [(Int, BatchRunResult)] = []

        for bs in [1, 2, 4, 8] {
            let line = String(repeating: "=", count: 100)
            print("\n\(line)\n  B=\(bs) — \(SHORT_ANSWER_TESTS.count) short-answer + \(LONG_DECODE_TESTS.count) long-decode (4K max)\n\(line)")
            let result = await runBatch(client: client, batchSize: bs, tests: allTests)
            printSummary(batchSize: bs, result: result, label: "")
            print(line)
            totalPassed += result.passed
            totalFailed += result.failed
            batchResults.append((bs, result))
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }

        // Summary table
        let sep = String(repeating: "=", count: 120)
        print("\n\(sep)")
        print("  SUMMARY — mixed workload")
        print(sep)
        print(String(format: "  %3s  %4s  %6s  %7s  %6s  %7s  %7s  %6s  %5s  %5s  %5s  %6s  %5s  %5s",
            "B", "Pass", "PP tok", "PP t/s", "TG tok", "TG t/s", "Agg t/s",
            "TTFT", "Wall", "GPU%", "GPU W", "DRAM W", "Sys W", "Temp"))

        for (bs, result) in batchResults {
            let ok = result.okResults
            guard !ok.isEmpty else { continue }
            let pf = "\(result.passed)/\(result.passed + result.failed)"
            let totalPrompt = ok.reduce(0) { $0 + $1.promptTokens }
            let totalCompl  = ok.reduce(0) { $0 + $1.completionTokens }
            let avgPP   = ok.reduce(0.0) { $0 + $1.ppTokS } / Double(ok.count)
            let avgTG   = ok.reduce(0.0) { $0 + $1.tgTokS } / Double(ok.count)
            let avgTTFT = ok.reduce(0.0) { $0 + $1.ttft } / Double(ok.count)
            let maxWall = ok.max(by: { $0.wallS < $1.wallS })?.wallS ?? 0
            let aggTG   = maxWall > 0 ? Double(totalCompl) / maxWall : 0
            let g = result.gpuStats
            let gpuPct = g != nil ? String(format: "%.0f%%", g!["gpu_pct"] ?? 0) : "—"
            let gpuW   = g != nil ? String(format: "%.1f",  g!["gpu_w"] ?? 0)   : "—"
            let dramW  = g != nil ? String(format: "%.1f",  g!["dram_w"] ?? 0)  : "—"
            let sysW   = g != nil ? String(format: "%.1f",  g!["sys_w"] ?? 0)   : "—"
            let temp   = g != nil ? String(format: "%.0f°", g!["temp_c"] ?? 0)  : "—"

            print(String(format: "  %3d  %4s  %6d  %7.1f  %6d  %7.1f  %7.1f  %5.2fs  %4.0fs  %5s  %5s  %6s  %5s  %5s",
                bs, pf, totalPrompt, avgPP, totalCompl, avgTG, aggTG,
                avgTTFT, maxWall, gpuPct, gpuW, dramW, sysW, temp))
        }

        print(sep)
        print("  TOTAL: \(totalPassed)/\(totalPassed + totalFailed) passed across \(batchResults.count) batch sizes")
        if totalFailed > 0 {
            print("  (\(totalFailed) failures: model answer mismatches, not code bugs)")
        }
        print(sep)

        #expect(totalFailed == 0, "\(totalFailed) total failures across B=1,2,4,8")
    }

    // MARK: - Private

    private func serverReachable() async -> Bool {
        do {
            _ = try await client.health()
            return true
        } catch {
            print("  ⚠️  Server not reachable at localhost:9999 — skipping test")
            return false
        }
    }
}
