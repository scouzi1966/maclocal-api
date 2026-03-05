import Foundation

// MARK: - Error types

enum XGrammarError: Error, LocalizedError {
    case notRunning
    case invalidResponse
    case bridgeError(String)

    var errorDescription: String? {
        switch self {
        case .notRunning:
            return "XGrammar bridge subprocess is not running"
        case .invalidResponse:
            return "XGrammar bridge returned an invalid or unparseable response"
        case .bridgeError(let message):
            return "XGrammar bridge error: \(message)"
        }
    }
}

// MARK: - XGrammarBridge actor

/// Communicates with the `Scripts/xgrammar-bridge.py` subprocess via JSON-lines stdio.
///
/// Spawn once per server lifetime (or lazily on first use). All methods are
/// `async throws` and are protected by actor isolation — no external locking needed.
actor XGrammarBridge {

    // MARK: Private state

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?

    /// True when a XGrammarSession has been handed out and may be using the pipes.
    /// Actor methods that do I/O must check this and refuse to run concurrently.
    private var sessionActive = false

    /// Accumulated bytes from stdout that have not yet been consumed as a complete line.
    private var readBuffer = Data()

    // MARK: Script resolution

    /// Candidate locations for `xgrammar-bridge.py`, checked in order.
    private static func scriptCandidates() -> [String] {
        var candidates: [String] = []

        // 1. Relative to the Swift executable bundle (release layout: bin/../Scripts/)
        let bundleBin = Bundle.main.bundlePath  // e.g. /usr/local/bin or .build/release
        let bundleAdjacentScripts = URL(fileURLWithPath: bundleBin)
            .deletingLastPathComponent()
            .appendingPathComponent("Scripts/xgrammar-bridge.py")
            .path
        candidates.append(bundleAdjacentScripts)

        // 2. Working-directory-relative (development / test usage)
        candidates.append("./Scripts/xgrammar-bridge.py")

        // 3. Absolute path relative to known project root (fallback for xctest etc.)
        if let repoRoot = ProcessInfo.processInfo.environment["MACAFM_REPO_ROOT"] {
            candidates.append("\(repoRoot)/Scripts/xgrammar-bridge.py")
        }

        return candidates
    }

    private static func resolveScriptPath() throws -> String {
        for candidate in scriptCandidates() {
            let expanded = (candidate as NSString).expandingTildeInPath
            if FileManager.default.fileExists(atPath: expanded) {
                return expanded
            }
        }
        throw XGrammarError.bridgeError(
            "xgrammar-bridge.py not found. Searched: \(scriptCandidates().joined(separator: ", "))"
        )
    }

    // MARK: Lifecycle

    /// Start the Python subprocess. Safe to call multiple times — no-ops if already running.
    func start() throws {
        guard process == nil else { return }

        let scriptPath = try XGrammarBridge.resolveScriptPath()

        let proc = Process()
        let inPipe = Pipe()
        let outPipe = Pipe()

        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["python3", scriptPath]
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = FileHandle.standardError  // Let Python errors surface in host stderr

        try proc.run()

        self.process = proc
        self.stdinPipe = inPipe
        self.stdoutPipe = outPipe
        self.readBuffer = Data()
    }

    /// Terminate the subprocess. Safe to call even if not running.
    func stop() {
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        readBuffer = Data()
    }

    // MARK: Private I/O helpers

    private func isRunning() -> Bool {
        guard let proc = process else { return false }
        return proc.isRunning
    }

    /// Write a JSON command object to the subprocess stdin.
    private func sendCommand(_ command: [String: Any]) throws {
        guard isRunning(), let pipe = stdinPipe else {
            throw XGrammarError.notRunning
        }

        let data = try JSONSerialization.data(withJSONObject: command)
        var lineData = data
        lineData.append(0x0A)  // newline

        pipe.fileHandleForWriting.write(lineData)
    }

    /// Read the next complete newline-terminated line from the subprocess stdout.
    ///
    /// Blocks synchronously until a full line is available. This is acceptable because
    /// the actor serialises all calls — no concurrent reads happen.
    private func readLine() throws -> String {
        guard isRunning(), let pipe = stdoutPipe else {
            throw XGrammarError.notRunning
        }

        let handle = pipe.fileHandleForReading
        let newline: UInt8 = 0x0A

        while true {
            // Check if buffer already contains a complete line
            if let nlIndex = readBuffer.firstIndex(of: newline) {
                let lineData = readBuffer[readBuffer.startIndex..<nlIndex]
                readBuffer = readBuffer[(nlIndex + 1)...]
                guard let line = String(data: lineData, encoding: .utf8) else {
                    throw XGrammarError.invalidResponse
                }
                return line
            }

            // Need more data — read a chunk
            let chunk = handle.availableData
            if chunk.isEmpty {
                // EOF — process likely terminated
                throw XGrammarError.notRunning
            }
            readBuffer.append(chunk)
        }
    }

    /// Send a command and return the parsed JSON response dictionary.
    /// Asserts that no XGrammarSession is currently using the pipes.
    private func roundTrip(_ command: [String: Any]) throws -> [String: Any] {
        precondition(!sessionActive, "XGrammarBridge: actor I/O while a session is active — pipe corruption risk")
        try sendCommand(command)

        let line = try readLine()
        guard
            let data = line.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw XGrammarError.invalidResponse
        }

        // Check for bridge-level errors
        if let ok = json["ok"] as? Bool, !ok {
            let msg = json["error"] as? String ?? "unknown bridge error"
            throw XGrammarError.bridgeError(msg)
        }

        return json
    }

    // MARK: Public API

    /// Compile a JSON schema and return a grammar_id handle.
    ///
    /// - Parameters:
    ///   - schema: A JSON-serialisable object (typically `[String: Any]`).
    ///   - vocabSize: The vocabulary size of the target model's tokenizer.
    ///   - tokenizerPath: Filesystem path to the `tokenizer.json` (or directory containing it).
    /// - Returns: An opaque `grammar_id` string to be passed to subsequent calls.
    func compile(schema: Any, vocabSize: Int, tokenizerPath: String) throws -> String {
        let command: [String: Any] = [
            "cmd": "compile",
            "schema": schema,
            "vocab_size": vocabSize,
            "tokenizer_path": tokenizerPath
        ]
        let resp = try roundTrip(command)
        guard let grammarID = resp["grammar_id"] as? String else {
            throw XGrammarError.invalidResponse
        }
        return grammarID
    }

    /// Return the list of token IDs allowed at the current grammar position.
    func getAllowedTokens(grammarID: String) throws -> [Int] {
        let command: [String: Any] = [
            "cmd": "mask",
            "grammar_id": grammarID
        ]
        let resp = try roundTrip(command)
        guard let allowed = resp["allowed"] as? [Int] else {
            throw XGrammarError.invalidResponse
        }
        return allowed
    }

    /// Advance the grammar state by accepting the given token.
    func acceptToken(grammarID: String, tokenID: Int) throws {
        let command: [String: Any] = [
            "cmd": "accept",
            "grammar_id": grammarID,
            "token_id": tokenID
        ]
        _ = try roundTrip(command)
    }

    /// Returns `true` when the grammar has reached a terminal (accepting) state.
    func isTerminated(grammarID: String) throws -> Bool {
        let command: [String: Any] = [
            "cmd": "is_terminated",
            "grammar_id": grammarID
        ]
        let resp = try roundTrip(command)
        guard let terminated = resp["terminated"] as? Bool else {
            throw XGrammarError.invalidResponse
        }
        return terminated
    }

    /// Release the grammar matcher, freeing memory in the subprocess.
    func release(grammarID: String) throws {
        let command: [String: Any] = [
            "cmd": "release",
            "grammar_id": grammarID
        ]
        _ = try roundTrip(command)
    }

    /// Create a synchronous session for per-token grammar operations.
    /// The session uses blocking I/O directly on the subprocess pipes.
    /// While a session is active, the actor's own I/O methods are blocked (precondition).
    /// Call `releaseSession()` when done to re-enable actor I/O.
    func createSession(grammarID: String) -> XGrammarSession? {
        guard !sessionActive, isRunning(), let inPipe = stdinPipe, let outPipe = stdoutPipe else { return nil }
        sessionActive = true
        return XGrammarSession(
            stdinHandle: inPipe.fileHandleForWriting,
            stdoutHandle: outPipe.fileHandleForReading,
            grammarID: grammarID
        )
    }

    /// Mark the session as finished, re-enabling actor I/O.
    func releaseSession() {
        sessionActive = false
    }
}

// MARK: - XGrammarSession (synchronous per-token operations)

/// Synchronous wrapper for per-token grammar operations.
/// Uses blocking pipe I/O — suitable for use within the synchronous TokenIterator loop.
///
/// **Important:** This class shares the same stdin/stdout pipes as the `XGrammarBridge` actor.
/// It must only be used while the actor is NOT concurrently performing its own I/O
/// (i.e., within a `container.perform` block where the actor is idle).
final class XGrammarSession: @unchecked Sendable {
    private let stdinHandle: FileHandle
    private let stdoutHandle: FileHandle
    private let grammarID: String
    private var buffer = Data()

    init(stdinHandle: FileHandle, stdoutHandle: FileHandle, grammarID: String) {
        self.stdinHandle = stdinHandle
        self.stdoutHandle = stdoutHandle
        self.grammarID = grammarID
    }

    /// Get allowed token IDs for the current grammar state (blocking).
    func getAllowedTokens() -> [Int]? {
        let cmd: [String: Any] = ["cmd": "mask", "grammar_id": grammarID]
        guard let resp = roundTrip(cmd) else { return nil }
        return resp["allowed"] as? [Int]
    }

    /// Accept a sampled token, advancing grammar state (blocking).
    func acceptToken(_ tokenID: Int) {
        let cmd: [String: Any] = ["cmd": "accept", "grammar_id": grammarID, "token_id": tokenID]
        _ = roundTrip(cmd)
    }

    /// Check if grammar has reached terminal state (blocking).
    func isTerminated() -> Bool {
        let cmd: [String: Any] = ["cmd": "is_terminated", "grammar_id": grammarID]
        guard let resp = roundTrip(cmd) else { return false }
        return resp["terminated"] as? Bool ?? false
    }

    /// Release the grammar matcher.
    func release() {
        let cmd: [String: Any] = ["cmd": "release", "grammar_id": grammarID]
        _ = roundTrip(cmd)
    }

    private func roundTrip(_ command: [String: Any]) -> [String: Any]? {
        guard let data = try? JSONSerialization.data(withJSONObject: command) else { return nil }
        var line = data
        line.append(0x0A)  // newline
        stdinHandle.write(line)

        guard let responseData = readLine() else { return nil }
        return try? JSONSerialization.jsonObject(with: responseData) as? [String: Any]
    }

    private func readLine() -> Data? {
        while true {
            if let newlineIndex = buffer.firstIndex(of: 0x0A) {
                let line = Data(buffer[buffer.startIndex..<newlineIndex])
                buffer = Data(buffer[buffer.index(after: newlineIndex)...])
                return line
            }
            let chunk = stdoutHandle.availableData
            if chunk.isEmpty { return buffer.isEmpty ? nil : buffer }
            buffer.append(chunk)
        }
    }
}
