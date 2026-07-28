import AFMKit
import Foundation

// =============================================================================
// AFMParityCheck — differential parity harness
//
// Runs a fixed battery of requests at temperature 0 TWICE:
//   (A) AFMKit-direct, via `AFMEngine` / the public services, in-process.
//   (B) The `afm` HTTP server, spawned on a local port and hit over /v1/*.
// and asserts the two produce identical output. This is the executable proof
// that "the library == the server".
//
// Run:
//   MACAFM_MLX_MODEL_CACHE=/path/to/cache \
//   AFM_BINARY=../../.build/release/afm \
//   AFM_PARITY_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit \
//   swift run AFMParityCheck
//
// Exit code 0 = full parity; non-zero = at least one mismatch (details printed).
// =============================================================================

// MARK: - Configuration

enum Config {
    static let env = ProcessInfo.processInfo.environment
    static let defaultModel = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    static var models: [String] {
        if let raw = env["AFM_PARITY_MODELS"], !raw.isEmpty {
            let parsed = splitList(raw)
            if !parsed.isEmpty { return parsed }
        }
        return [env["AFM_PARITY_MODEL"] ?? defaultModel]
    }
    static let host = "127.0.0.1"
    static let basePort = Int(env["AFM_PARITY_PORT"] ?? "9998") ?? 9998

    /// Resolve the `afm` binary: AFM_BINARY override, else the repo's release/debug build.
    static var binaryPath: String {
        if let b = env["AFM_BINARY"], !b.isEmpty { return b }
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // .../Sources/AFMParityCheck
            .deletingLastPathComponent()   // .../Sources
            .deletingLastPathComponent()   // .../AFMParityCheck (example)
            .deletingLastPathComponent()   // .../Examples
            .deletingLastPathComponent()   // repo root
        for rel in [".build/release/afm", ".build/arm64-apple-macosx/release/afm", ".build/debug/afm"] {
            let p = here.appendingPathComponent(rel).path
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return here.appendingPathComponent(".build/release/afm").path
    }

    // Numeric tolerances (no magic numbers inline).
    static let logprobTolerance: Float = 1e-3
    static let serverReadyTimeout: TimeInterval = 120
    static let serverPollInterval: TimeInterval = 1.0
    static let maxTokens = 64
    static let allCases: [String] = [
        "greedy-text",
        "streaming-concat",
        "logprobs",
        "structured-json-object",
        "tool-call",
        "stop-sequence",
        "strict-json-schema"
    ]
    static var requiredCases: Set<String> {
        guard let raw = env["AFM_PARITY_REQUIRED_CASES"], !raw.isEmpty else {
            return Set(allCases)
        }
        return Set(splitList(raw))
    }
    static var enabledCases: Set<String> {
        guard let raw = env["AFM_PARITY_CASES"], !raw.isEmpty else {
            return Set(allCases)
        }
        return Set(splitList(raw))
    }
    static var reportPath: String? {
        guard let path = env["AFM_PARITY_REPORT"], !path.isEmpty else { return nil }
        return path
    }
    static func shouldRun(_ name: String) -> Bool {
        enabledCases.contains(name)
    }

    static func splitList(_ raw: String) -> [String] {
        raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

struct ModelRun {
    let model: String
    let port: Int

    var baseURL: String {
        "http://\(Config.host):\(port)"
    }
}

// MARK: - Result tracking

enum ParityCaseStatus: String, Codable, Sendable {
    case passed
    case failed
    case skipped
}

struct ParityCaseRecord: Codable, Sendable {
    let model: String
    let caseID: String
    let status: ParityCaseStatus
    let detail: String?
}

struct ParityReport: Codable, Sendable {
    let generatedAt: Date
    let models: [String]
    let enabledCases: [String]
    let requiredCases: [String]
    let missingRequiredCases: [String]
    let binaryPath: String
    let passed: Int
    let failed: Int
    let skipped: Int
    let records: [ParityCaseRecord]
}

actor Tally {
    private(set) var passed = 0
    private(set) var failed = 0
    private(set) var skipped = 0
    private(set) var records: [ParityCaseRecord] = []

    func pass(_ name: String, model: String) {
        passed += 1
        records.append(ParityCaseRecord(model: model, caseID: name, status: .passed, detail: nil))
        print("  ✅ PASS  \(name)")
    }

    func fail(_ name: String, model: String, _ detail: String) {
        failed += 1
        records.append(ParityCaseRecord(model: model, caseID: name, status: .failed, detail: detail))
        print("  ❌ FAIL  \(name)")
        for line in detail.split(separator: "\n") { print("           \(line)") }
    }

    func skip(_ name: String, model: String) {
        skipped += 1
        records.append(ParityCaseRecord(model: model, caseID: name, status: .skipped, detail: nil))
        print("  ⏭️ SKIP  \(name)")
    }

    func missingRequiredCases(models: [String], requiredCases: Set<String>) -> [String] {
        var missing: [String] = []
        for model in models {
            let passedCases = Set(records
                .filter { $0.model == model && $0.status == .passed }
                .map(\.caseID))
            for requiredCase in requiredCases.sorted() where !passedCases.contains(requiredCase) {
                missing.append("\(model):\(requiredCase)")
            }
        }
        return missing
    }

    func report(
        models: [String],
        enabledCases: Set<String>,
        requiredCases: Set<String>,
        missingRequiredCases: [String]
    ) -> ParityReport {
        ParityReport(
            generatedAt: Date(),
            models: models,
            enabledCases: enabledCases.sorted(),
            requiredCases: requiredCases.sorted(),
            missingRequiredCases: missingRequiredCases,
            binaryPath: Config.binaryPath,
            passed: passed,
            failed: failed,
            skipped: skipped,
            records: records
        )
    }
}
let tally = Tally()

func normalize(_ s: String) -> String {
    s.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - HTTP client (Foundation only — no Vapor)

enum HTTPError: Error, CustomStringConvertible {
    case status(Int, String)
    case decode(String)
    var description: String {
        switch self {
        case .status(let c, let b): return "HTTP \(c): \(b.prefix(200))"
        case .decode(let m): return "decode: \(m)"
        }
    }
}

func chatRequest(_ body: [String: Any], stream: Bool, run: ModelRun) -> URLRequest {
    var req = URLRequest(url: URL(string: "\(run.baseURL)/v1/chat/completions")!)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    var b = body
    b["model"] = run.model
    b["stream"] = stream
    b["temperature"] = 0
    if b["max_tokens"] == nil { b["max_tokens"] = Config.maxTokens }
    req.httpBody = try! JSONSerialization.data(withJSONObject: b)
    return req
}

/// Non-streaming POST → parsed top-level JSON object.
func httpChat(_ body: [String: Any], run: ModelRun) async throws -> [String: Any] {
    let (data, resp) = try await URLSession.shared.data(for: chatRequest(body, stream: false, run: run))
    let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
    guard code == 200 else { throw HTTPError.status(code, String(data: data, encoding: .utf8) ?? "") }
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw HTTPError.decode("not a JSON object")
    }
    return obj
}

/// Streaming POST → concatenated `choices[0].delta.content` across SSE chunks.
func httpChatStreamConcat(_ body: [String: Any], run: ModelRun) async throws -> String {
    let (bytes, resp) = try await URLSession.shared.bytes(for: chatRequest(body, stream: true, run: run))
    let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
    guard code == 200 else { throw HTTPError.status(code, "stream") }
    var out = ""
    for try await line in bytes.lines {
        guard line.hasPrefix("data:") else { continue }
        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" { break }
        guard let d = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let delta = choices.first?["delta"] as? [String: Any],
              let piece = delta["content"] as? String else { continue }
        out += piece
    }
    return out
}

// Extractors over the non-streaming response shape.
func contentOf(_ obj: [String: Any]) -> String {
    guard let choices = obj["choices"] as? [[String: Any]],
          let msg = choices.first?["message"] as? [String: Any],
          let c = msg["content"] as? String else { return "" }
    return c
}

func toolCallsOf(_ obj: [String: Any]) -> [(name: String, args: String)] {
    guard let choices = obj["choices"] as? [[String: Any]],
          let msg = choices.first?["message"] as? [String: Any],
          let tcs = msg["tool_calls"] as? [[String: Any]] else { return [] }
    return tcs.compactMap { tc in
        guard let fn = tc["function"] as? [String: Any],
              let name = fn["name"] as? String else { return nil }
        return (name, (fn["arguments"] as? String) ?? "")
    }
}

func logprobTokensOf(_ obj: [String: Any]) -> [(token: String, logprob: Double)] {
    guard let choices = obj["choices"] as? [[String: Any]],
          let lp = choices.first?["logprobs"] as? [String: Any],
          let content = lp["content"] as? [[String: Any]] else { return [] }
    return content.compactMap { e in
        guard let t = e["token"] as? String, let v = e["logprob"] as? Double else { return nil }
        return (t, v)
    }
}

// Canonicalize a JSON string so semantically-equal objects compare equal.
func canonicalJSON(_ s: String) -> String? {
    guard let d = s.data(using: .utf8),
          let obj = try? JSONSerialization.jsonObject(with: d),
          let out = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else { return nil }
    return String(data: out, encoding: .utf8)
}

// MARK: - Server lifecycle

func spawnServer(run: ModelRun) throws -> Process {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: Config.binaryPath)
    p.arguments = ["mlx", "-m", run.model, "--port", "\(run.port)"]
    // Pass the full environment (incl. MACAFM_MLX_MODEL_CACHE) through so the
    // server reuses the same local weights the direct engine loaded.
    p.environment = ProcessInfo.processInfo.environment
    let devnull = FileHandle.nullDevice
    p.standardOutput = devnull
    p.standardError = devnull
    try p.run()
    return p
}

func waitForServerReady(run: ModelRun) async throws {
    let deadline = Date().addingTimeInterval(Config.serverReadyTimeout)
    let url = URL(string: "\(run.baseURL)/v1/models")!
    while Date() < deadline {
        if let (_, resp) = try? await URLSession.shared.data(from: url),
           (resp as? HTTPURLResponse)?.statusCode == 200 {
            return
        }
        try await Task.sleep(nanoseconds: UInt64(Config.serverPollInterval * 1_000_000_000))
    }
    throw HTTPError.status(-1, "server did not become ready within \(Int(Config.serverReadyTimeout))s")
}

// MARK: - Battery

let systemMsg = Message(role: "system", content: "You are concise. Answer in one short sentence.")
func userMessages(_ text: String) -> [Message] {
    [systemMsg, Message(role: "user", content: text)]
}
func wireMessages(_ text: String) -> [[String: String]] {
    [["role": "system", "content": "You are concise. Answer in one short sentence."],
     ["role": "user", "content": text]]
}

@main
struct AFMParityCheck {
    static func main() async {
        print("AFMParityCheck — models=\(Config.models.joined(separator: ",")) basePort=\(Config.basePort)")
        print("binary=\(Config.binaryPath)")
        print("cases=\(Config.allCases.filter(Config.shouldRun).joined(separator: ","))")

        for (index, model) in Config.models.enumerated() {
            let run = ModelRun(model: model, port: Config.basePort + index)
            await runBattery(run: run)
        }

        // ---- Summary -------------------------------------------------------
        let passed = await tally.passed
        var failed = await tally.failed
        let skipped = await tally.skipped
        let missingRequired = await tally.missingRequiredCases(
            models: Config.models,
            requiredCases: Config.requiredCases
        )
        if !missingRequired.isEmpty {
            failed += missingRequired.count
            print("\nMissing required parity passes:")
            for missing in missingRequired {
                print("  ❌ \(missing)")
            }
        }
        if let reportPath = Config.reportPath {
            do {
                let report = await tally.report(
                    models: Config.models,
                    enabledCases: Config.enabledCases,
                    requiredCases: Config.requiredCases,
                    missingRequiredCases: missingRequired
                )
                let encoder = JSONEncoder()
                encoder.dateEncodingStrategy = .iso8601
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(report)
                try data.write(to: URL(fileURLWithPath: reportPath), options: .atomic)
                print("report=\(reportPath)")
            } catch {
                failed += 1
                print("  ❌ FAIL  report-write")
                print("           \(error)")
            }
        }
        print("\n=====================================================")
        print("PARITY: \(passed) passed, \(failed) failed, \(skipped) skipped")
        print("=====================================================")
        exit(failed == 0 ? 0 : 1)
    }

    static func runBattery(run: ModelRun) async {
        print("\n=====================================================")
        print("Model: \(run.model) port=\(run.port)")
        print("=====================================================")

        // ---- (A) Build the in-process engine -------------------------------
        let engine = AFMEngine(
            backend: .mlx(modelID: run.model),
            config: EngineConfig(enablePrefixCaching: false)
        )
        do {
            print("loading model (direct)…")
            _ = try await engine.load()
        } catch {
            FileHandle.standardError.write(Data("FATAL: could not load model: \(error)\n".utf8))
            exit(2)
        }

        // ---- (B) Spawn the HTTP server -------------------------------------
        var server: Process
        do {
            print("spawning afm server…")
            server = try spawnServer(run: run)
            try await waitForServerReady(run: run)
            print("server ready at \(run.baseURL)")
        } catch {
            FileHandle.standardError.write(Data("FATAL: server did not start: \(error)\n".utf8))
            exit(2)
        }
        defer { server.terminate() }

        // ---- Case 1: greedy chat text --------------------------------------
        print("\n[1] greedy chat text")
        if !Config.shouldRun("greedy-text") {
            await tally.skip("greedy-text", model: run.model)
        } else { do {
            let prompt = "Name three primary colors."
            let cfg = GenerationConfig(temperature: 0, maxTokens: Config.maxTokens)
            let direct = try await engine.respond(to: userMessages(prompt), cfg)
            let http = try await httpChat(["messages": wireMessages(prompt)], run: run)
            let d = normalize(direct.content), h = normalize(contentOf(http))
            if d == h { await tally.pass("greedy-text", model: run.model) }
            else { await tally.fail("greedy-text", model: run.model, "direct: \(d)\nserver: \(h)") }
        } catch { await tally.fail("greedy-text", model: run.model, "\(error)") } }

        // ---- Case 2: streaming concat == non-streaming ---------------------
        print("\n[2] streaming determinism (deltas concat == full)")
        if !Config.shouldRun("streaming-concat") {
            await tally.skip("streaming-concat", model: run.model)
        } else { do {
            let prompt = "List the days of the week."
            let cfg = GenerationConfig(temperature: 0, maxTokens: Config.maxTokens)
            var directStream = ""
            for try await delta in engine.streamRespond(to: userMessages(prompt), cfg) { directStream += delta }
            let directFull = try await engine.respond(to: userMessages(prompt), cfg)
            let httpStream = try await httpChatStreamConcat(["messages": wireMessages(prompt)], run: run)
            let httpFull = contentOf(try await httpChat(["messages": wireMessages(prompt)], run: run))

            let values = [normalize(directStream), normalize(directFull.content),
                          normalize(httpStream), normalize(httpFull)]
            if Set(values).count == 1 { await tally.pass("streaming-concat", model: run.model) }
            else {
                await tally.fail("streaming-concat", model: run.model,
                    "direct.stream: \(values[0])\ndirect.full:   \(values[1])\n"
                    + "server.stream: \(values[2])\nserver.full:   \(values[3])")
            }
        } catch { await tally.fail("streaming-concat", model: run.model, "\(error)") } }

        // ---- Case 3: logprobs (token + value) ------------------------------
        print("\n[3] logprobs parity")
        if !Config.shouldRun("logprobs") {
            await tally.skip("logprobs", model: run.model)
        } else { do {
            let prompt = "Reply with exactly: OK"
            let cfg = GenerationConfig(temperature: 0, maxTokens: Config.maxTokens, logprobs: true, topLogprobs: 0)
            let direct = try await engine.respond(to: userMessages(prompt), cfg)
            let http = try await httpChat(["messages": wireMessages(prompt),
                                           "logprobs": true], run: run)
            let dTokens = (direct.logprobs ?? []).map { (token: $0.token, logprob: Double($0.logprob)) }
            let hTokens = logprobTokensOf(http)
            if dTokens.isEmpty && hTokens.isEmpty {
                await tally.fail("logprobs", model: run.model, "neither side returned logprobs (model/build may not support them)")
            } else if dTokens.count != hTokens.count {
                await tally.fail("logprobs", model: run.model, "count differs: direct=\(dTokens.count) server=\(hTokens.count)")
            } else {
                var mismatch: String?
                for (i, (dt, ht)) in zip(dTokens, hTokens).enumerated() {
                    if dt.token != ht.token {
                        mismatch = "token[\(i)] '\(dt.token)' != '\(ht.token)'"; break
                    }
                    if abs(dt.logprob - ht.logprob) > Double(Config.logprobTolerance) {
                        mismatch = "logprob[\(i)] \(dt.logprob) != \(ht.logprob)"; break
                    }
                }
                if let m = mismatch { await tally.fail("logprobs", model: run.model, m) }
                else { await tally.pass("logprobs", model: run.model) }
            }
        } catch { await tally.fail("logprobs", model: run.model, "\(error)") } }

        // ---- Case 4: structured JSON (response_format) ---------------------
        print("\n[4] structured JSON (response_format=json_object)")
        if !Config.shouldRun("structured-json-object") {
            await tally.skip("structured-json-object", model: run.model)
        } else { do {
            let prompt = "Return a JSON object with keys \"city\" and \"country\" for Paris."
            let fmt = ResponseFormat(type: "json_object")
            let cfg = GenerationConfig(temperature: 0, maxTokens: Config.maxTokens, responseFormat: fmt)
            let direct = try await engine.respond(to: userMessages(prompt), cfg)
            let http = contentOf(try await httpChat(["messages": wireMessages(prompt),
                                                     "response_format": ["type": "json_object"]], run: run))
            let dc = canonicalJSON(direct.content), hc = canonicalJSON(http)
            if let dc, let hc, dc == hc { await tally.pass("structured-json-object", model: run.model) }
            else {
                await tally.fail("structured-json-object", model: run.model,
                    "direct: \(direct.content)\nserver: \(http)\n(parsed-equal: \(dc != nil && dc == hc))")
            }
        } catch { await tally.fail("structured-json-object", model: run.model, "\(error)") } }

        // ---- Case 5: tool call (name + arguments) --------------------------
        print("\n[5] tool call parity")
        if !Config.shouldRun("tool-call") {
            await tally.skip("tool-call", model: run.model)
        } else { do {
            let schema = AnyCodable([
                "type": "object",
                "properties": ["location": ["type": "string"]],
                "required": ["location"]
            ] as [String: Any])
            let tool = RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "get_weather",
                    description: "Get the current weather for a location",
                    parameters: schema,
                    strict: nil))
            let prompt = "What's the weather in Tokyo? Use the tool."
            let cfg = GenerationConfig(temperature: 0, maxTokens: Config.maxTokens, tools: [tool])
            let direct = try await engine.respond(to: userMessages(prompt), cfg)
            let wireTools: [[String: Any]] = [[
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": ["type": "object",
                                   "properties": ["location": ["type": "string"]],
                                   "required": ["location"]]
                ]
            ]]
            let http = try await httpChat(["messages": wireMessages(prompt), "tools": wireTools], run: run)
            let dTC = (direct.toolCalls ?? []).map { (name: $0.function.name, args: $0.function.arguments) }
            let hTC = toolCallsOf(http)
            if dTC.isEmpty && hTC.isEmpty {
                await tally.fail("tool-call", model: run.model, "neither side emitted a tool call (model may have answered directly)")
            } else if dTC.first?.name != hTC.first?.name {
                await tally.fail("tool-call", model: run.model, "name: direct=\(dTC.first?.name ?? "-") server=\(hTC.first?.name ?? "-")")
            } else {
                let da = canonicalJSON(dTC.first?.args ?? "") ?? (dTC.first?.args ?? "")
                let ha = canonicalJSON(hTC.first?.args ?? "") ?? (hTC.first?.args ?? "")
                if da == ha { await tally.pass("tool-call", model: run.model) }
                else { await tally.fail("tool-call", model: run.model, "args: direct=\(da) server=\(ha)") }
            }
        } catch { await tally.fail("tool-call", model: run.model, "\(error)") } }

        // ---- Case 6: stop sequence -----------------------------------------
        print("\n[6] stop sequence parity")
        if !Config.shouldRun("stop-sequence") {
            await tally.skip("stop-sequence", model: run.model)
        } else { do {
            let prompt = "Write exactly: alpha STOP beta"
            let stop = [" STOP"]
            let cfg = GenerationConfig(
                temperature: 0,
                maxTokens: Config.maxTokens,
                stop: stop
            )
            let direct = try await engine.respond(to: userMessages(prompt), cfg)
            let http = try await httpChat([
                "messages": wireMessages(prompt),
                "stop": stop
            ], run: run)
            let d = normalize(direct.content), h = normalize(contentOf(http))
            if d == h { await tally.pass("stop-sequence", model: run.model) }
            else { await tally.fail("stop-sequence", model: run.model, "direct: \(d)\nserver: \(h)") }
        } catch { await tally.fail("stop-sequence", model: run.model, "\(error)") } }

        // ---- Case 7: strict JSON schema ------------------------------------
        print("\n[7] strict JSON schema parity")
        if !Config.shouldRun("strict-json-schema") {
            await tally.skip("strict-json-schema", model: run.model)
        } else { do {
            let schema: [String: Any] = [
                "type": "object",
                "properties": ["answer": ["type": "string"]],
                "required": ["answer"],
                "additionalProperties": false
            ]
            let prompt = "Return JSON with one string field named answer and value yes."
            let fmt = ResponseFormat(
                type: "json_schema",
                jsonSchema: ResponseJsonSchema(
                    name: "answer_schema",
                    description: nil,
                    schema: AnyCodable(schema),
                    strict: true
                )
            )
            let cfg = GenerationConfig(
                temperature: 0,
                maxTokens: Config.maxTokens,
                responseFormat: fmt
            )
            let direct = try await engine.respond(to: userMessages(prompt), cfg)
            let http = contentOf(try await httpChat([
                "messages": wireMessages(prompt),
                "response_format": [
                    "type": "json_schema",
                    "json_schema": [
                        "name": "answer_schema",
                        "strict": true,
                        "schema": schema
                    ]
                ]
            ], run: run))
            let dc = canonicalJSON(direct.content), hc = canonicalJSON(http)
            if let dc, let hc, dc == hc { await tally.pass("strict-json-schema", model: run.model) }
            else {
                await tally.fail(
                    "strict-json-schema",
                    model: run.model,
                    "direct: \(direct.content)\nserver: \(http)\n(parsed-equal: \(dc != nil && dc == hc))"
                )
            }
        } catch { await tally.fail("strict-json-schema", model: run.model, "\(error)") } }

        server.terminate()
    }
}
