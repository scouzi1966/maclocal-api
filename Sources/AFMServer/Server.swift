import Vapor
import AFMKit
import AFMKitMLX
import Foundation
import Compression
import Darwin
#if canImport(AppKit)
import AppKit
#endif
import Logging
// Storage key for the continuation
public struct ContinuationKey: StorageKey {
    public typealias Value = CheckedContinuation<Void, Error>
}

/// Storage key for the per-request correlation ID (T1.1).
/// Set by `RequestIDMiddleware`, read by controllers and error emitters so the
/// same ID appears on the `X-Request-ID` / `OpenAI-Request-ID` response header,
/// inside `error.request_id`, and in server logs.
enum RequestIDKey: StorageKey {
    typealias Value = String
}

extension Request {
    /// The agent-correlatable request ID (T1.1). Always present after the
    /// `RequestIDMiddleware` has run.
    var afmRequestID: String {
        storage[RequestIDKey.self] ?? ""
    }
}

/// Mints or echoes a stable request ID for every HTTP request and copies it
/// to the response headers. Honors inbound `X-Request-ID` (most common) and
/// `OpenAI-Request-ID` (OpenAI SDK convention); otherwise mints `req_<uuid12>`
/// matching the format used by `BatchAPIController`. (T1.1)
struct RequestIDMiddleware: AsyncMiddleware {
    static let inboundHeaders = ["X-Request-ID", "OpenAI-Request-ID"]
    static let outboundHeaders = ["X-Request-ID", "OpenAI-Request-ID"]

    static func mint() -> String {
        "req_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(12)
    }

    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        let inbound = Self.inboundHeaders
            .compactMap { request.headers.first(name: $0)?.trimmingCharacters(in: .whitespaces) }
            .first(where: { !$0.isEmpty })
        let id = inbound ?? Self.mint()
        await request.storage.setWithAsyncShutdown(RequestIDKey.self, to: id)

        let response = try await next.respond(to: request)
        for name in Self.outboundHeaders {
            response.headers.replaceOrAdd(name: name, value: id)
        }
        return response
    }
}

/// Renders our typed errors (`TokenizeUnsupportedError`, `TokenizeBadRequestError`)
/// in OpenAI's `{"error": {"message", "type", "code", "request_id"}}` shape so
/// agents using the OpenAI SDK get a parsable error body. Falls through for
/// everything else so Vapor's default error middleware still handles it. (T1.6)
///
/// Both `type` and `code` are populated with the same machine-readable
/// identifier (`tokenize_unsupported`, `invalid_request_error`, …). OpenAI's
/// SDKs commonly switch on `code`, so leaving it nil makes those clients fall
/// through to a generic-error path even though `type` is meaningful.
struct OpenAIErrorRenderingMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let err as TokenizeUnsupportedError {
            return try Self.render(
                request: request,
                status: err.status,
                message: err.reason,
                type: TokenizeUnsupportedError.errorType,
                code: TokenizeUnsupportedError.errorType,
                requestId: err.requestId
            )
        } catch let err as TokenizeBadRequestError {
            return try Self.render(
                request: request,
                status: err.status,
                message: err.reason,
                type: TokenizeBadRequestError.errorType,
                code: TokenizeBadRequestError.errorType,
                requestId: err.requestId
            )
        }
    }

    private static func render(request: Request, status: HTTPResponseStatus, message: String, type: String, code: String?, requestId: String) throws -> Response {
        let id = requestId.isEmpty ? request.afmRequestID : requestId
        let payload = OpenAIError(
            message: message,
            type: type,
            code: code,
            requestId: id.isEmpty ? nil : id
        )
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(payload)
        return response
    }
}

// Middleware to handle payload too large errors with a user-friendly message
struct PayloadTooLargeMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let abort as Abort where abort.status == .payloadTooLarge {
            // Return a JSON error response compatible with OpenAI format
            let reqId = request.afmRequestID
            let errorResponse = OpenAIError(
                message: "Your conversation is too long. Please start a new conversation.",
                type: "payload_too_large",
                requestId: reqId.isEmpty ? nil : reqId
            )
            let response = Response(status: .payloadTooLarge)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(errorResponse)
            return response
        }
    }
}

/// Counts active HTTP requests so `/metrics` can expose
/// `afm:num_active_connections`. Increments on entry, decrements on
/// exit (via defer-style task-local cleanup so the gauge always
/// returns to zero even on early throws). Filters out the /metrics
/// endpoint itself so a Prometheus scrape doesn't show up as a
/// connection — that would be self-referential noise on every poll.
///
/// Streaming endpoints (chat completions when `stream:true`) account
/// for themselves: their handler returns the Response object
/// immediately while the SSE body keeps writing for the duration of
/// the generation. If we counted them here, the gauge would
/// undercount — defer fires when `next.respond` returns, not when the
/// body finishes. Streaming controllers wrap the asyncStream body in
/// their own connectionStarted/connectionEnded bracket.
struct ActiveConnectionsMiddleware: AsyncMiddleware {
    static let nonStreamingExcluded: Set<String> = ["/metrics", "/health", "/healthz", "/openapi.json", "/docs"]
    static let streamingPaths: Set<String> = [
        "/v1/chat/completions",
        "/v1/batch/completions"
    ]

    static func shouldTrackInMiddleware(path: String) -> Bool {
        if nonStreamingExcluded.contains(path) { return false }
        // Filter the streaming chat path — its controller handles its own counting.
        if streamingPaths.contains(path) { return false }
        return true
    }

    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        let track = Self.shouldTrackInMiddleware(path: request.url.path)
        if track { ActiveConnectionTracker.shared.connectionStarted() }
        defer { if track { ActiveConnectionTracker.shared.connectionEnded() } }
        return try await next.respond(to: request)
    }
}

private struct AFMWebLaunchSnapshot: Sendable {
    let running: Bool
    let pid: Int32?
    let port: Int?
    let backend: String?
    let model: String?
    let logPath: String?
    let command: [String]
}

/// Owns child runtimes started from the local WebUI. Process arguments are
/// passed directly to Foundation.Process (never a shell), and only arguments
/// assembled from the server-side allowlists below reach this actor.
private actor AFMWebRuntimeManager {
    private var process: Process?
    private var logHandle: FileHandle?
    private var port: Int?
    private var backend: String?
    private var model: String?
    private var logPath: String?
    private var command: [String] = []

    func launch(
        executable: URL,
        request: AFMWebLaunchRequest,
        startingAt startPort: Int
    ) throws -> AFMWebLaunchSnapshot {
        // Select the port inside the actor. Concurrent start requests must not
        // both observe the same free port before either child is running.
        guard let port = Server.availableWebRuntimePort(startingAt: startPort) else {
            throw Abort(.serviceUnavailable, reason: "No local runtime port is available")
        }
        let arguments = try Server.webLaunchArguments(for: request, port: port)
        let displayCommand = Server.redactedWebLaunchCommand(
            executable: executable,
            arguments: arguments)

        if let process, process.isRunning {
            process.terminate()
        }
        try? logHandle?.close()

        let fileManager = FileManager.default
        let afmDirectory = fileManager.homeDirectoryForCurrentUser.appendingPathComponent(".afm", isDirectory: true)
        let logsDirectory = afmDirectory.appendingPathComponent("logs", isDirectory: true)
        try fileManager.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        let logURL = logsDirectory.appendingPathComponent("webui-runtime.log")
        if !fileManager.fileExists(atPath: logURL.path) {
            fileManager.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.truncate(atOffset: 0)

        let child = Process()
        child.executableURL = executable
        child.arguments = arguments
        child.currentDirectoryURL = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        var environment = ProcessInfo.processInfo.environment
        environment["AFM_WEBUI_MANAGED_CHILD"] = "1"
        child.environment = environment
        // A WebUI-managed runtime is always a server. Detach it from the
        // manager's stdin so a piped or non-interactive parent cannot make the
        // MLX command block while probing stdin for single-prompt mode.
        child.standardInput = FileHandle.nullDevice
        child.standardOutput = handle
        child.standardError = handle
        do {
            try child.run()
        } catch {
            try? handle.close()
            throw error
        }

        self.process = child
        self.logHandle = handle
        self.port = port
        self.backend = request.backend.lowercased()
        self.model = request.model
        self.logPath = logURL.path
        self.command = displayCommand
        return snapshot()
    }

    func stop() -> AFMWebLaunchSnapshot {
        if let process, process.isRunning { process.terminate() }
        return snapshot()
    }

    func snapshot() -> AFMWebLaunchSnapshot {
        closeLogIfExited()
        return AFMWebLaunchSnapshot(
            running: process?.isRunning ?? false,
            pid: process?.processIdentifier,
            port: port,
            backend: backend,
            model: model,
            logPath: logPath,
            command: command
        )
    }

    func logTail(maxBytes: Int = 24_000) -> String {
        closeLogIfExited()
        guard let logPath, let handle = FileHandle(forReadingAtPath: logPath) else { return "" }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        let start = size > UInt64(maxBytes) ? size - UInt64(maxBytes) : 0
        try? handle.seek(toOffset: start)
        return String(data: (try? handle.readToEnd()) ?? Data(), encoding: .utf8) ?? ""
    }

    private func closeLogIfExited() {
        guard let process, !process.isRunning, let logHandle else { return }
        try? logHandle.close()
        self.logHandle = nil
    }
}

// @unchecked Sendable: the server owns a Vapor Application and assorted service
// references that aren't Sendable-audited. Lifecycle (start/shutdown) is driven
// from a single controlling flow and the closures it spawns only read immutable
// configuration or hop to @MainActor, so cross-task sharing is safe in practice.
public class Server: @unchecked Sendable {
    private let app: Application
    private let port: Int
    private let hostname: String
    private let verbose: Bool
    private let veryVerbose: Bool
    private let trace: Bool
    private let streamingEnabled: Bool
    private let instructions: String
    private let adapter: String?
    private let temperature: Double?
    private let randomness: String?
    private let permissiveGuardrails: Bool
    private let stop: String?
    private let webuiEnabled: Bool
    private let webuiPath: String?
    private let gatewayEnabled: Bool
    private let prewarmEnabled: Bool
    private let telegramConfiguration: TelegramConfiguration?
    private let defaultGuidedJsonSchema: ResponseFormat?
    private let defaultChatTemplateKwargs: [String: AnyCodable]?
    private let forceDisableThinking: Bool
    private let mlxModelID: String?
    private let mlxModel: AFMMLXModel?
    private let afmModel: AnyAFMModel?
    private let mlxRepetitionPenalty: Double?
    private let mlxTopP: Double?
    private let mlxMaxTokens: Int?
    private let mlxRawOutput: Bool
    private let mlxTopK: Int?
    private let mlxMinP: Double?
    private let mlxPresencePenalty: Double?
    private let mlxSeed: Int?
    private let mlxMaxLogprobs: Int
    private let contextWindow: Int?
    private var telegramBridge: TelegramBridge?
    private let webRuntimeManager = AFMWebRuntimeManager()

    private static let audioAvailable: Bool = {
        if #available(macOS 13.0, *) { return true }
        return false
    }()

    public init(port: Int, hostname: String, verbose: Bool, veryVerbose: Bool = false, trace: Bool = false, streamingEnabled: Bool, instructions: String, adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool = false, stop: String? = nil, webuiEnabled: Bool = false, gatewayEnabled: Bool = false, prewarmEnabled: Bool = true, telegramConfiguration: TelegramConfiguration? = nil, defaultGuidedJsonSchema: ResponseFormat? = nil, defaultChatTemplateKwargs: [String: AnyCodable]? = nil, forceDisableThinking: Bool = false, mlxModelID: String? = nil, mlxModel: AFMMLXModel? = nil, afmModel: AnyAFMModel? = nil, mlxRepetitionPenalty: Double? = nil, mlxTopP: Double? = nil, mlxMaxTokens: Int? = nil, mlxRawOutput: Bool = false, mlxTopK: Int? = nil, mlxMinP: Double? = nil, mlxPresencePenalty: Double? = nil, mlxSeed: Int? = nil, mlxMaxLogprobs: Int? = nil, contextWindow: Int? = nil) async throws {
        self.port = port
        self.hostname = hostname
        self.verbose = verbose
        self.veryVerbose = veryVerbose
        self.trace = trace
        self.streamingEnabled = streamingEnabled
        self.instructions = instructions
        self.adapter = adapter
        self.temperature = temperature
        self.randomness = randomness
        self.permissiveGuardrails = permissiveGuardrails
        self.stop = stop
        self.webuiEnabled = webuiEnabled
        self.webuiPath = Server.findWebuiPath()
        self.gatewayEnabled = gatewayEnabled
        self.prewarmEnabled = prewarmEnabled
        self.telegramConfiguration = telegramConfiguration
        self.defaultGuidedJsonSchema = defaultGuidedJsonSchema
        self.defaultChatTemplateKwargs = defaultChatTemplateKwargs
        self.forceDisableThinking = forceDisableThinking
        self.mlxModelID = mlxModelID
        self.mlxModel = mlxModel
        self.afmModel = afmModel
        self.mlxRepetitionPenalty = mlxRepetitionPenalty
        self.mlxTopP = mlxTopP
        self.mlxMaxTokens = mlxMaxTokens
        self.mlxRawOutput = mlxRawOutput
        self.mlxTopK = mlxTopK
        self.mlxMinP = mlxMinP
        self.mlxPresencePenalty = mlxPresencePenalty
        self.mlxSeed = mlxSeed
        self.mlxMaxLogprobs = mlxMaxLogprobs ?? 20
        self.contextWindow = contextWindow

        // Create environment without command line arguments to prevent Vapor from parsing them
        var env = Environment(name: "development", arguments: ["afm"])
        LoggingSystem.bootstrap { label in
            CompactLogHandler(label: label)
        }

        self.app = try await Application.make(env)

        if veryVerbose {
            app.logger.logLevel = .trace
        } else if verbose {
            app.logger.logLevel = .debug
        }

        // Initialize backend discovery and proxy services (only in gateway mode)
        if gatewayEnabled {
            let discovery = BackendDiscoveryService(logger: app.logger, selfPort: port)
            let proxy = BackendProxyService(logger: app.logger)
            app.backendDiscovery = discovery
            app.backendProxy = proxy
        }

        try configure()
    }
    
    private func configure() throws {
        app.http.server.configuration.port = port
        app.http.server.configuration.hostname = hostname

        // Increase max body size for long conversations (default is 16KB)
        // 100MB should handle very long conversation histories
        app.routes.defaultMaxBodySize = "100mb"

        // Mint/echo X-Request-ID for every request — must run before other
        // middleware so ID is visible in error paths too. (T1.1)
        app.middleware.use(RequestIDMiddleware())

        // Render typed errors (TokenizeUnsupportedError, etc.) in OpenAI shape. (T1.6)
        app.middleware.use(OpenAIErrorRenderingMiddleware())

        // Add custom error middleware to handle payload too large errors
        app.middleware.use(PayloadTooLargeMiddleware())
        // Track concurrent client connections for /metrics' afm:num_active_connections gauge.
        app.middleware.use(ActiveConnectionsMiddleware())

        try routes()
    }

    private static let foundationWebValueOptions: Set<String> = [
        "--instructions", "--adapter", "--temperature", "--randomness", "--stop",
        "--prewarm", "--guided-json", "--telegram-bot-token", "--telegram-allow",
        "--telegram-format", "--telegram-require-prefix"
    ]
    private static let foundationWebFlagOptions: Set<String> = [
        "--verbose", "--very-verbose", "--vv", "--no-streaming",
        "--permissive-guardrails", "--gateway"
    ]
    private static let mlxWebValueOptions: Set<String> = [
        "--instructions", "--temperature", "--top-p", "--top-k", "--min-p",
        "--presence-penalty", "--repetition-penalty", "--max-tokens", "--seed",
        "--max-logprobs", "--kv-cache-size", "--kv-bits", "--prefill-step-size",
        "--mlx-runtime", "--gguf-file", "--prewarm", "--stop", "--guided-json", "--tool-call-parser",
        "--kv-eviction", "--default-chat-template-kwargs", "--reasoning-effort",
        "--concurrent", "--mtp-depth", "--mtp-model", "--dspark-support", "--dspark-draft-tokens",
        "--dspark-confidence", "--eagle3", "--cache-profile-path", "--gpu-capture",
        "--gpu-trace", "--chat-template", "--dtype", "--telegram-bot-token", "--telegram-allow", "--telegram-format",
        "--telegram-require-prefix"
    ]
    private static let mlxWebFlagOptions: Set<String> = [
        "--verbose", "--very-verbose", "--vv", "--no-streaming", "--raw", "--vlm",
        "--trust-remote-code", "--fix-tool-args", "--enable-prefix-caching", "--mtp",
        "--dspark-strict", "--enable-grammar-constraints", "--no-think", "--gpu-profile",
        "--gpu-profile-bw"
    ]

    private static let webLaunchSecretOptions: Set<String> = ["--telegram-bot-token"]

    static func isLoopbackWebAddress(_ address: String?) -> Bool {
        guard let address else { return false }
        return address == "127.0.0.1"
            || address == "::1"
            || address.lowercased() == "localhost"
            || address.lowercased().hasPrefix("::ffff:127.")
    }

    static func isTrustedWebOrigin(_ origin: String?) -> Bool {
        guard let origin else { return true } // Non-browser/local API clients.
        guard origin != "null", let url = URL(string: origin),
              url.scheme == "http" || url.scheme == "https" else {
            return false
        }
        return isLoopbackWebAddress(url.host)
    }

    static func isTrustedWebHost(_ host: String?) -> Bool {
        guard let host, let url = URL(string: "http://\(host)") else { return false }
        return isLoopbackWebAddress(url.host)
    }

    private func requireLocalWebControl(_ request: Request) throws {
        guard webuiEnabled else { throw Abort(.notFound) }
        guard Self.isLoopbackWebAddress(request.remoteAddress?.ipAddress),
              Self.isTrustedWebHost(request.headers.first(name: .host)),
              Self.isTrustedWebOrigin(request.headers.first(name: .origin)) else {
            throw Abort(.forbidden, reason: "AFM WebUI runtime controls are available only from this Mac")
        }
    }

    static func redactedWebLaunchCommand(executable: URL, arguments: [String]) -> [String] {
        var result = [executable.path]
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            result.append(argument)
            if webLaunchSecretOptions.contains(argument), index + 1 < arguments.count {
                result.append("<redacted>")
                index += 2
            } else {
                index += 1
            }
        }
        return result
    }

    static func webLaunchArguments(for request: AFMWebLaunchRequest, port: Int) throws -> [String] {
        let backend = request.backend.lowercased()
        guard backend == "foundation" || backend == "mlx" else {
            throw Abort(.badRequest, reason: "backend must be 'foundation' or 'mlx'")
        }
        var arguments: [String] = []
        let valueAllowlist: Set<String>
        let flagAllowlist: Set<String>
        if backend == "mlx" {
            guard let model = request.model?.trimmingCharacters(in: .whitespacesAndNewlines), !model.isEmpty else {
                throw Abort(.badRequest, reason: "An MLX model id or local path is required")
            }
            guard model.count <= 2048, !model.contains("\0") else {
                throw Abort(.badRequest, reason: "Invalid MLX model")
            }
            arguments += ["mlx", "--model", model]
            valueAllowlist = mlxWebValueOptions
            flagAllowlist = mlxWebFlagOptions
        } else {
            valueAllowlist = foundationWebValueOptions
            flagAllowlist = foundationWebFlagOptions
        }

        for key in (request.values ?? [:]).keys.sorted() {
            guard valueAllowlist.contains(key) else {
                throw Abort(.badRequest, reason: "Unsupported \(backend) option: \(key)")
            }
            guard let value = request.values?[key], !value.isEmpty else { continue }
            guard value.count <= 16_384, !value.contains("\0") else {
                throw Abort(.badRequest, reason: "Invalid value for \(key)")
            }
            arguments += [key, value]
        }
        for flag in Set(request.flags ?? []).sorted() {
            guard flagAllowlist.contains(flag) else {
                throw Abort(.badRequest, reason: "Unsupported \(backend) flag: \(flag)")
            }
            arguments.append(flag)
        }

        // The manager owns the bind address and port so a launched runtime can
        // never replace or expose the manager process unexpectedly.
        arguments += ["--hostname", "127.0.0.1", "--port", String(port), "--webui"]
        return arguments
    }

    static func availableWebRuntimePort(startingAt start: Int) -> Int? {
        let lowerBound = max(1024, start)
        guard lowerBound <= 65_535 else { return nil }
        let upperBound = min(65_535, lowerBound + 100)
        for candidate in lowerBound...upperBound {
            let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
            guard descriptor >= 0 else { continue }
            defer { Darwin.close(descriptor) }
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = in_port_t(candidate).bigEndian
            address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
            let result = withUnsafePointer(to: &address) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            if result == 0 { return candidate }
        }
        return nil
    }

    /// Mirrors the chat-controller routing predicate without constructing a
    /// service adapter. A model id plus either concrete model selects the
    /// MLX-compatible path; every other configuration falls back to Apple's
    /// Foundation Models controller.
    static func usesAppleFoundationBackend(
        mlxModelID: String?,
        hasMLXModel: Bool,
        hasProviderModel: Bool
    ) -> Bool {
        mlxModelID == nil || (!hasMLXModel && !hasProviderModel)
    }

    private static func currentExecutableURL() -> URL {
        var size: UInt32 = 0
        _ = _NSGetExecutablePath(nil, &size)
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: Int(size))
            if _NSGetExecutablePath(&buffer, &size) == 0 {
                return URL(fileURLWithPath: String(cString: buffer)).resolvingSymlinksInPath()
            }
        }
        return Bundle.main.executableURL?.resolvingSymlinksInPath()
            ?? URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    }

    private static var webLaunchProfileURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".afm", isDirectory: true)
            .appendingPathComponent("webui-launch-profile.json")
    }

    private static func jsonResponse(_ object: [String: Any], status: HTTPResponseStatus = .ok) throws -> Response {
        let response = Response(status: status)
        response.headers.replaceOrAdd(name: .contentType, value: "application/json; charset=utf-8")
        response.headers.add(name: .cacheControl, value: "no-store")
        response.body = .init(data: try JSONSerialization.data(withJSONObject: object))
        return response
    }
    
    private func routes() throws {
        let mlxServiceAdapter = mlxModel.map {
            AFMKitMLXChatServingAdapter(
                model: $0,
                defaultGuidedJsonSchema: defaultGuidedJsonSchema,
                defaultChatTemplateKwargs: defaultChatTemplateKwargs,
                forceDisableThinking: forceDisableThinking)
        }
        let mlxChatService: (any AFMChatServing)?
        if let afmModel, let mlxModelID {
            mlxChatService = AFMKitMLXChatServingAdapter(
                model: afmModel,
                modelID: mlxModelID,
                defaultGuidedJsonSchema: defaultGuidedJsonSchema,
                defaultChatTemplateKwargs: defaultChatTemplateKwargs,
                forceDisableThinking: forceDisableThinking)
        } else {
            mlxChatService = mlxServiceAdapter
        }

        app.get("health") { req async -> HealthResponse in
            return HealthResponse(
                status: "healthy",
                timestamp: Date().timeIntervalSince1970,
                version: "1.0.0"
            )
        }

        // Small read-only endpoint used by the injected local runtime console.
        // Keep machine-specific paths out of the static web asset and resolve
        // them in the process that actually owns the model/runtime state.
        app.get("afm", "runtime") { req -> Response in
            try self.requireLocalWebControl(req)
            let environment = ProcessInfo.processInfo.environment
            let home = FileManager.default.homeDirectoryForCurrentUser.path
            let modelCache: String
            if let configured = environment["MACAFM_MLX_MODEL_CACHE"], !configured.isEmpty {
                modelCache = configured
            } else if let configured = environment["HUGGINGFACE_HUB_CACHE"] ?? environment["HF_HUB_CACHE"], !configured.isEmpty {
                modelCache = configured
            } else if let configured = environment["HF_HOME"], !configured.isEmpty {
                modelCache = URL(fileURLWithPath: configured).appendingPathComponent("hub").path
            } else {
                modelCache = URL(fileURLWithPath: home).appendingPathComponent("Documents/huggingface/models").path
            }

            let payload: [String: Any] = [
                "version": BuildInfo.fullVersion,
                "backend": self.mlxModelID == nil ? "Apple Foundation Models / router" : "MLX",
                "modelCache": modelCache,
                "persistence": URL(fileURLWithPath: home).appendingPathComponent(".afm").path,
                "streaming": self.streamingEnabled,
                "webui": self.webuiEnabled
            ]
            let response = Response(status: .ok)
            response.headers.replaceOrAdd(name: .contentType, value: "application/json; charset=utf-8")
            response.headers.add(name: .cacheControl, value: "no-store")
            response.body = .init(data: try JSONSerialization.data(withJSONObject: payload))
            return response
        }

        app.get("afm", "launcher", "profile") { req -> Response in
            try self.requireLocalWebControl(req)
            let response = Response(status: .ok)
            response.headers.replaceOrAdd(name: .contentType, value: "application/json; charset=utf-8")
            response.headers.add(name: .cacheControl, value: "no-store")
            if let data = try? Data(contentsOf: Self.webLaunchProfileURL),
               let profile = try? JSONDecoder().decode(AFMWebLaunchRequest.self, from: data),
               let safeData = try? JSONEncoder().encode(profile.persistableProfile()) {
                // Sanitize legacy profiles written by earlier prototypes that
                // persisted the Telegram token, then never return it to JS.
                if profile.values?["--telegram-bot-token"] != nil || profile.dryRun != false {
                    try? safeData.write(to: Self.webLaunchProfileURL, options: .atomic)
                    try? FileManager.default.setAttributes(
                        [.posixPermissions: 0o600],
                        ofItemAtPath: Self.webLaunchProfileURL.path)
                }
                response.body = .init(data: safeData)
            } else {
                response.body = .init(string: "{}")
            }
            return response
        }

        app.on(.POST, "afm", "launcher", "start", body: .collect(maxSize: "128kb")) { req async throws -> Response in
            try self.requireLocalWebControl(req)
            let launch = try req.content.decode(AFMWebLaunchRequest.self)
            let executable = Self.currentExecutableURL()
            let launchPort: Int
            let command: [String]

            if launch.dryRun == true {
                guard let previewPort = Self.availableWebRuntimePort(startingAt: self.port + 1) else {
                    throw Abort(.serviceUnavailable, reason: "No local runtime port is available")
                }
                let arguments = try Self.webLaunchArguments(for: launch, port: previewPort)
                launchPort = previewPort
                command = Self.redactedWebLaunchCommand(executable: executable, arguments: arguments)
            } else {
                let snapshot = try await self.webRuntimeManager.launch(
                    executable: executable,
                    request: launch,
                    startingAt: self.port + 1
                )
                guard let selectedPort = snapshot.port else {
                    throw Abort(.internalServerError, reason: "Managed runtime did not select a port")
                }
                launchPort = selectedPort
                command = snapshot.command
                let profileURL = Self.webLaunchProfileURL
                try FileManager.default.createDirectory(at: profileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try JSONEncoder().encode(launch.persistableProfile()).write(to: profileURL, options: .atomic)
                try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: profileURL.path)
            }
            return try Self.jsonResponse([
                "accepted": true,
                "dryRun": launch.dryRun == true,
                "backend": launch.backend.lowercased(),
                "model": (launch.model as Any?) ?? NSNull(),
                "port": launchPort,
                "url": "http://127.0.0.1:\(launchPort)",
                "command": command
            ])
        }

        app.get("afm", "launcher", "status") { req async throws -> Response in
            try self.requireLocalWebControl(req)
            let snapshot = await self.webRuntimeManager.snapshot()
            var healthy = false
            if snapshot.running, let port = snapshot.port,
               let url = URL(string: "http://127.0.0.1:\(port)/health") {
                var request = URLRequest(url: url)
                request.timeoutInterval = 0.6
                if let (_, response) = try? await URLSession.shared.data(for: request),
                   let http = response as? HTTPURLResponse {
                    healthy = (200..<300).contains(http.statusCode)
                }
            }
            return try Self.jsonResponse([
                "running": snapshot.running,
                "healthy": healthy,
                "pid": (snapshot.pid as Any?) ?? NSNull(),
                "port": (snapshot.port as Any?) ?? NSNull(),
                "backend": (snapshot.backend as Any?) ?? NSNull(),
                "model": (snapshot.model as Any?) ?? NSNull(),
                "url": (snapshot.port.map { "http://127.0.0.1:\($0)" } as Any?) ?? NSNull(),
                "logPath": (snapshot.logPath as Any?) ?? NSNull(),
                "command": snapshot.command
            ])
        }

        app.get("afm", "launcher", "log") { req async throws -> Response in
            try self.requireLocalWebControl(req)
            return try Self.jsonResponse(["log": await self.webRuntimeManager.logTail()])
        }

        app.on(.POST, "afm", "launcher", "stop") { req async throws -> Response in
            try self.requireLocalWebControl(req)
            let snapshot = await self.webRuntimeManager.stop()
            return try Self.jsonResponse(["stopped": true, "pid": (snapshot.pid as Any?) ?? NSNull()])
        }

        app.get("v1", "models") { req async -> ModelsResponse in
            // Apple NL embedding models are served on the unified endpoint (lazily
            // loaded on first /v1/embeddings). Advertise them so clients discover
            // embedding capability on the main server, not just on `afm embed`. (#132/#133)
            let embeddingCatalog = EmbeddingModelRegistry().shippedModels()
            let embeddingModelInfos = embeddingCatalog.map { m in
                ModelInfo(id: m.id, object: "model", created: m.createdEpoch, owned_by: "apple", loaded: false)
            }
            let embeddingDetails = embeddingCatalog.map { m in
                ModelDetails(name: "\(m.id) (Apple NL)", model: m.id, capabilities: ["embeddings"])
            }

            if let mlxModelID = self.mlxModelID {
                let loadedDescriptor = mlxChatService?.loadedModelDescriptor(model: mlxModelID)
                return ModelsResponse(
                    object: "list",
                    data: [
                        ModelInfo(
                            id: mlxModelID,
                            object: "model",
                            created: Int(Date().timeIntervalSince1970),
                            owned_by: "mlx",
                            loaded: true,
                            max_context_length: self.contextWindow
                        )
                    ] + embeddingModelInfos,
                    models: [
                        ModelDetails(
                            name: mlxModelID,
                            model: mlxModelID,
                            capabilities: AFMMLXCapabilityPresentation.modelCapabilityLabels(
                                descriptor: loadedDescriptor
                            )
                        )
                    ] + embeddingDetails
                )
            }

            var models: [ModelInfo] = [
                ModelInfo(
                    id: "foundation",
                    object: "model",
                    created: Int(Date().timeIntervalSince1970),
                    owned_by: "apple",
                    loaded: true
                )
            ]
            var details: [ModelDetails] = [
                ModelDetails(name: "foundation (Apple)", model: "foundation", capabilities: ModelCapabilities.foundation.capabilities)
            ]

            if let discovery = req.application.backendDiscovery {
                // Rescan backends if stale so new models/backends appear quickly
                await discovery.refreshIfStale()
                let discovered = await discovery.allDiscoveredModels()
                for dm in discovered {
                    models.append(ModelInfo(
                        id: dm.id,
                        object: "model",
                        created: dm.created,
                        owned_by: dm.ownedBy,
                        loaded: dm.loaded
                    ))
                    // Use cached capabilities if available, otherwise nil (probed lazily via /props)
                    let caps = await discovery.capabilitiesForModel(dm.id)
                    details.append(ModelDetails(
                        name: "\(dm.id) (\(dm.backendName))",
                        model: dm.id,
                        capabilities: caps.capabilities
                    ))
                }
            }

            models += embeddingModelInfos
            details += embeddingDetails
            return ModelsResponse(object: "list", data: models, models: details)
        }

        // Stub /models/load and /models/unload for router mode compatibility
        // The webui calls these when switching models; we just acknowledge success
        // since our backends manage their own model loading
        app.on(.POST, "models", "load", body: .collect(maxSize: "1mb")) { req -> Response in
            // Parse the requested model from the body
            var modelName = "unknown"
            if let body = req.body.data {
                let bodyData = Data(buffer: body)
                if let json = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any],
                   let model = json["model"] as? String {
                    modelName = model
                }
            }
            req.logger.info("WebUI model load request: '\(modelName)'")

            let response = Response(status: .ok)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            // Echo back the model info so the webui confirms the switch
            let responseBody: [String: Any] = [
                "success": true,
                "model": modelName
            ]
            if let data = try? JSONSerialization.data(withJSONObject: responseBody) {
                response.body = .init(data: data)
            }
            return response
        }
        app.on(.POST, "models", "unload", body: .collect(maxSize: "1mb")) { req -> Response in
            let response = Response(status: .ok)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(["success": true])
            return response
        }

        try app.register(collection: VisionAPIController())
        try app.register(collection: SpeechAPIController())
        // POST /v1/embeddings on the main server (#132). The Apple NL embedding
        // model is loaded lazily on first request (a chat-only server pays
        // nothing until used) and this path never triggers MLX init. It does NOT
        // register /v1/models — the main server owns that route. `afm embed`
        // remains a standalone option.
        try app.register(collection: EmbeddingsController(
            resolver: LazyAppleEmbeddingResolver(),
            registersModelsRoute: false
        ))
        // POST /v1/chat/completions/{id}/cancel — agent cancel endpoint (T1.5).
        try app.register(collection: CancelController())
        // POST /v1/tokenize, /v1/count_tokens — agent token-budgeting endpoints (T1.6).
        try app.register(collection: TokenizeController(
            mlxModelID: mlxModelID,
            tokenizer: mlxServiceAdapter,
            contextWindow: contextWindow
        ))
        // GET /openapi.json + /docs — schema discovery for self-configuring agents (T1.7).
        try app.register(collection: OpenAPIController())

        if let mlxModelID = mlxModelID,
           let mlxChatService {
            let mlxController = MLXChatCompletionsController(
                streamingEnabled: streamingEnabled,
                modelID: mlxModelID,
                service: mlxChatService,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                topK: mlxTopK,
                minP: mlxMinP,
                presencePenalty: mlxPresencePenalty,
                seed: mlxSeed,
                maxLogprobs: mlxMaxLogprobs,
                veryVerbose: veryVerbose,
                trace: trace,
                rawOutput: mlxRawOutput,
                stop: stop
            )
            try app.register(collection: mlxController)

            if mlxModel != nil {
                // Batch endpoints remain MLX-specific. Fixed-schedule providers
                // currently expose one serial generation slot.
                let batchStore = BatchStore()

                let batchAPIController = BatchAPIController(
                    service: mlxChatService,
                    store: batchStore,
                    modelID: mlxModelID,
                    temperature: temperature,
                    topP: mlxTopP,
                    maxTokens: mlxMaxTokens,
                    repetitionPenalty: mlxRepetitionPenalty,
                    topK: mlxTopK,
                    minP: mlxMinP,
                    presencePenalty: mlxPresencePenalty,
                    seed: mlxSeed,
                    maxLogprobs: mlxMaxLogprobs
                )
                try app.register(collection: batchAPIController)

                let batchCompletionsController = BatchCompletionsController(
                    service: mlxChatService,
                    modelID: mlxModelID,
                    temperature: temperature,
                    topP: mlxTopP,
                    maxTokens: mlxMaxTokens,
                    repetitionPenalty: mlxRepetitionPenalty,
                    topK: mlxTopK,
                    minP: mlxMinP,
                    presencePenalty: mlxPresencePenalty,
                    seed: mlxSeed,
                    maxLogprobs: mlxMaxLogprobs
                )
                try app.register(collection: batchCompletionsController)
            }

            // Seed the metrics aggregator with the live model id and the
            // configured concurrency so /metrics labels are correct from
            // the first scrape.
            StatsAggregator.shared.setModel(
                mlxModelID,
                maxConcurrent: mlxChatService.maxConcurrent
            )
        } else {
            let chatController = ChatCompletionsController(
                streamingEnabled: streamingEnabled,
                instructions: instructions,
                adapter: adapter,
                temperature: temperature,
                randomness: randomness,
                permissiveGuardrails: permissiveGuardrails,
                veryVerbose: veryVerbose,
                stop: stop,
                defaultGuidedJsonSchema: defaultGuidedJsonSchema
            )
            try app.register(collection: chatController)
        }

        // Prometheus metrics — always on, regardless of backend.
        // GET /metrics returns afm:* counters/gauges modelled after vLLM.
        try app.register(collection: MetricsController())

        // Props endpoint for llama.cpp webui compatibility (per-model capabilities)
        app.get("props") { [self] req async -> PropsResponse in
            if let mlxModelID = self.mlxModelID {
                let loadedDescriptor = mlxChatService?.loadedModelDescriptor(model: mlxModelID)
                return PropsResponse(
                    default_generation_settings: DefaultGenerationSettings(
                        n_ctx: 8192,
                        params: GenerationParams(
                            n_predict: -1,
                            temperature: self.temperature ?? 0.8,
                            top_k: 40,
                            top_p: 0.95,
                            min_p: 0.05,
                            stream: self.streamingEnabled,
                            max_tokens: MLXChatCompletionsController.defaultMaxCompletionTokens
                        )
                    ),
                    total_slots: 1,
                    model_path: mlxModelID,
                    role: "mlx",
                    modalities: Modalities(
                        vision: AFMMLXCapabilityPresentation.supportsVision(
                            descriptor: loadedDescriptor
                        ),
                        audio: Self.audioAvailable
                    ),
                    chat_template: "",
                    bos_token: "",
                    eos_token: "",
                    build_info: "AFM \(BuildInfo.fullVersion)",
                    default_model: mlxModelID
                )
            }

            let modelParam = req.query[String.self, at: "model"]
            let isFoundation = modelParam == nil || modelParam == "foundation"

            var nCtx = 4096
            var hasVision = isFoundation
            var modelPath = "foundation"

            if !isFoundation, let modelName = modelParam {
                modelPath = modelName
                if let discovery = req.application.backendDiscovery {
                    let caps = await discovery.capabilitiesForModel(modelName)
                    hasVision = caps.vision
                    nCtx = caps.contextLength ?? 4096
                } else {
                    hasVision = false
                }
            }

            return PropsResponse(
                default_generation_settings: DefaultGenerationSettings(
                    n_ctx: nCtx,
                    params: GenerationParams(
                        n_predict: -1,
                        temperature: self.temperature ?? 0.8,
                        top_k: 40,
                        top_p: 0.95,
                        min_p: 0.05,
                        stream: self.streamingEnabled,
                        max_tokens: 2000
                    )
                ),
                total_slots: 1,
                model_path: modelPath,
                role: self.gatewayEnabled ? "router" : "model",
                modalities: Modalities(vision: hasVision, audio: Self.audioAvailable),
                chat_template: "",
                bos_token: "",
                eos_token: "",
                build_info: "AFM \(BuildInfo.fullVersion)",
                default_model: "foundation"
            )
        }

        // WebUI routes (if enabled and webui files exist)
        if webuiEnabled, let webuiFilePath = webuiPath {
            // Serve index.html with injected CSS for root path
            app.get { req -> Response in
                return try await self.serveWebuiWithCustomCSS(webuiFilePath: webuiFilePath, req: req)
            }

            // SPA fallback for non-API routes
            app.get("**") { req -> Response in
                let path = req.url.path

                // Don't intercept API routes
                if path.hasPrefix("/v1/") || path.hasPrefix("/afm/") || path == "/health" || path == "/props" {
                    throw Abort(.notFound)
                }

                return try await self.serveWebuiWithCustomCSS(webuiFilePath: webuiFilePath, req: req)
            }
        }
    }

    /// Custom CSS/JS to inject into webui (branding + auto-select default model + /metrics dashboard)
    private var customCSS: String {
        Self.customCSSTemplate.replacingOccurrences(of: "/*_IS_MLX_PLACEHOLDER*/false", with: mlxModelID != nil ? "true" : "false")
        + Self.controlCenterTemplate
        + Self.dashboardTemplate
    }
    private static let customCSSTemplate = """
    <style>
    /* AFM WebUI prototype: presentation-only skin over the stock llama.cpp app. */
    :root {
      --afm-canvas: #f4f6f8;
      --afm-canvas-accent: rgba(255, 143, 31, 0.10);
      --afm-panel: rgba(255, 255, 255, 0.82);
      --afm-panel-solid: #ffffff;
      --afm-panel-muted: rgba(241, 244, 247, 0.88);
      --afm-line: rgba(16, 24, 40, 0.10);
      --afm-line-strong: rgba(16, 24, 40, 0.16);
      --afm-ink: #121820;
      --afm-muted: #687381;
      --afm-orange: #f47b20;
      --afm-orange-strong: #d95f08;
      --afm-blue: #3276f6;
      --afm-green: #159b6c;
      --afm-radius-panel: 22px;
      --afm-radius-control: 16px;
      --afm-shadow-panel: 0 20px 55px rgba(24, 32, 44, 0.10);
      --afm-shadow-control: 0 12px 30px rgba(24, 32, 44, 0.12);
      --afm-content-width: 54rem;
      --radius: 0.875rem;
      --background: oklch(0.965 0.004 255);
      --sidebar: oklch(0.985 0.003 255);
      --sidebar-border: oklch(0.87 0.006 255);
      --primary: oklch(0.56 0.18 45);
      --ring: oklch(0.68 0.16 48);
    }
    .dark {
      --afm-canvas: #0b0f14;
      --afm-canvas-accent: rgba(244, 123, 32, 0.11);
      --afm-panel: rgba(21, 27, 35, 0.82);
      --afm-panel-solid: #151b23;
      --afm-panel-muted: rgba(29, 36, 46, 0.90);
      --afm-line: rgba(255, 255, 255, 0.09);
      --afm-line-strong: rgba(255, 255, 255, 0.16);
      --afm-ink: #f3f6fa;
      --afm-muted: #98a4b3;
      --afm-orange: #ff923f;
      --afm-orange-strong: #ffad68;
      --afm-blue: #6ea0ff;
      --afm-green: #43c895;
      --afm-shadow-panel: 0 24px 70px rgba(0, 0, 0, 0.34);
      --afm-shadow-control: 0 16px 36px rgba(0, 0, 0, 0.30);
      --background: oklch(0.13 0.008 255);
      --sidebar: oklch(0.16 0.008 255);
      --sidebar-border: oklch(1 0 0 / 10%);
      --primary: oklch(0.72 0.16 48);
      --ring: oklch(0.72 0.16 48);
    }

    /* Hide page until branding + model selection complete */
    body { opacity: 0 !important; }
    body.afm-ready { opacity: 1 !important; transition: opacity 0.15s ease-in; }

    body {
      color: var(--afm-ink);
      background:
        radial-gradient(circle at 76% -12%, var(--afm-canvas-accent), transparent 31rem),
        var(--afm-canvas) !important;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
      letter-spacing: -0.005em;
    }

    /* Turn the stock sidebar/inset into a calm desktop shell without replacing its DOM. */
    [data-slot="sidebar-wrapper"] {
      background: transparent !important;
    }
    [data-slot="sidebar-container"] {
      padding: 10px 0 10px 10px;
    }
    [data-slot="sidebar-inner"] {
      border: 1px solid var(--afm-line);
      border-radius: var(--afm-radius-panel);
      background: var(--afm-panel) !important;
      box-shadow: var(--afm-shadow-panel);
      backdrop-filter: blur(22px) saturate(130%);
      -webkit-backdrop-filter: blur(22px) saturate(130%);
      overflow: hidden;
    }
    [data-slot="sidebar-header"] {
      border-bottom: 1px solid var(--afm-line);
      background: transparent !important;
    }
    [data-slot="sidebar-header"] h1 {
      color: var(--afm-ink);
      font-size: 1.05rem !important;
      font-weight: 760 !important;
      letter-spacing: -0.025em;
    }
    [data-slot="sidebar-header"] h1::before {
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--afm-orange), #ffc36e);
      box-shadow: 0 0 0 4px var(--afm-canvas-accent);
      margin-right: 7px;
    }
    [data-slot="sidebar-content"],
    [data-slot="sidebar-group"] {
      background: transparent;
    }
    [data-slot="sidebar-group-label"] {
      color: var(--afm-muted);
      font-size: 0.67rem;
      font-weight: 720;
      letter-spacing: 0.11em;
      text-transform: uppercase;
    }
    [data-slot="sidebar-menu-button"] {
      border-radius: 12px !important;
    }
    [data-slot="sidebar-menu-button"]:hover,
    [data-slot="sidebar-menu-button"][data-active="true"] {
      background: var(--afm-panel-muted) !important;
    }
    [data-slot="sidebar-inset"] {
      position: relative;
      margin: 10px;
      border: 1px solid var(--afm-line);
      border-radius: var(--afm-radius-panel) !important;
      background: var(--afm-panel) !important;
      box-shadow: var(--afm-shadow-panel);
      overflow: hidden;
      backdrop-filter: blur(18px) saturate(120%);
      -webkit-backdrop-filter: blur(18px) saturate(120%);
    }

    /* Keep the native chat layout, only increasing breathing room and visual hierarchy. */
    main[aria-label*="Chat"],
    main[aria-label*="Welcome"] {
      background:
        linear-gradient(180deg, color-mix(in srgb, var(--afm-panel-solid) 36%, transparent), transparent 10rem);
    }
    main[aria-label*="Welcome"] h1 {
      color: var(--afm-ink);
      font-size: clamp(2rem, 4vw, 3.25rem) !important;
      font-weight: 780 !important;
      letter-spacing: -0.055em !important;
    }
    main[aria-label*="Welcome"] .afm-sub {
      color: var(--afm-muted) !important;
      font-size: 0.72rem !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    main[aria-label*="Welcome"] .afm-ai-badge {
      justify-content: center;
      font-size: 0.78rem !important;
    }
    main[aria-label*="Welcome"] p {
      color: var(--afm-muted);
      font-size: 1rem !important;
    }
    [role="main"] > div > [class*="max-w-[48rem]"],
    [role="main"] [class*="max-w-[48rem]"] {
      max-width: var(--afm-content-width) !important;
    }

    /* Composer: prominent enough to anchor the page, but still the stock working form. */
    .conversation-chat-form {
      padding: 16px 20px 20px !important;
      background: linear-gradient(180deg, transparent, var(--afm-panel-solid) 34%);
    }
    [data-slot="input-area"] {
      border: 1px solid var(--afm-line-strong) !important;
      border-radius: var(--afm-radius-control) !important;
      background: color-mix(in srgb, var(--afm-panel-solid) 91%, transparent) !important;
      box-shadow: var(--afm-shadow-control) !important;
      backdrop-filter: blur(22px) saturate(130%) !important;
      -webkit-backdrop-filter: blur(22px) saturate(130%) !important;
      transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }
    [data-slot="input-area"]:focus-within {
      border-color: color-mix(in srgb, var(--afm-orange) 62%, transparent) !important;
      box-shadow: 0 0 0 4px var(--afm-canvas-accent), var(--afm-shadow-control) !important;
      transform: translateY(-1px);
    }
    [data-slot="input-area"] textarea {
      color: var(--afm-ink) !important;
      font-size: 0.96rem !important;
      line-height: 1.55 !important;
    }
    [data-slot="input-area"] textarea::placeholder {
      color: var(--afm-muted) !important;
    }
    [data-slot="input-area"] [data-slot="button"] {
      border-radius: 11px !important;
    }

    /* Messages and their built-in generation statistics remain fully interactive. */
    [aria-label="User message with actions"] [data-slot="card"] {
      border: 1px solid color-mix(in srgb, var(--afm-orange) 18%, transparent) !important;
      background: color-mix(in srgb, var(--afm-orange) 8%, var(--afm-panel-solid)) !important;
      box-shadow: 0 7px 22px rgba(24, 32, 44, 0.05);
    }
    [aria-label="Assistant message with actions"] {
      padding: 2px 0;
    }
    [aria-label="Assistant message with actions"] .info {
      color: var(--afm-muted);
    }
    [data-slot="card"],
    [data-slot="dialog-content"],
    [data-slot="popover-content"],
    [data-slot="dropdown-menu-content"] {
      border-color: var(--afm-line) !important;
    }
    [data-slot="dialog-content"],
    [data-slot="popover-content"],
    [data-slot="dropdown-menu-content"] {
      background: var(--afm-panel-solid) !important;
      box-shadow: var(--afm-shadow-panel) !important;
    }

    /* Make model labels on response bubbles static (non-clickable) */
    .info [data-slot="popover-trigger"] { pointer-events: none; }
    .info [data-slot="popover-trigger"] svg { display: none; }

    @media (max-width: 767px) {
      [data-slot="sidebar-container"] { padding: 0; }
      [data-slot="sidebar-inner"] { border-radius: 0; border-width: 0 1px 0 0; }
      [data-slot="sidebar-inset"] {
        margin: 0;
        border: 0;
        border-radius: 0 !important;
        box-shadow: none;
      }
      .conversation-chat-form { padding: 10px 10px 14px !important; }
      [data-slot="input-area"] { border-radius: 15px !important; }
    }

    @media (prefers-reduced-motion: reduce) {
      body.afm-ready,
      [data-slot="input-area"] { transition: none !important; }
    }
    </style>
    <script>
    (function(){
        var _isMLX = /*_IS_MLX_PLACEHOLDER*/false;
        console.log('[AFM] _isMLX =', _isMLX);
        var _aiGradient = 'linear-gradient(to right, #3b82f6, #a855f7, #ec4899, #f97316)';

        function rebrand(){
            document.querySelectorAll('h1,h2,h3,p,span').forEach(function(el){
                if(el.textContent==='llama.cpp' || el.textContent==='AFM'){
                    el.textContent='AFM';
                    if(!el.nextElementSibling?.classList?.contains('afm-sub')){
                        var sub = document.createElement('div');
                        sub.className = 'afm-sub';
                        sub.style.cssText = 'font-size:11px;color:#888;font-weight:normal;margin-top:4px;';
                        sub.textContent = 'llama.cpp webui';
                        el.parentElement.insertBefore(sub, el.nextSibling);
                    }
                    var existingBadge = el.parentElement.querySelector('.afm-ai-badge');
                    if(_isMultiModel && existingBadge){ existingBadge.remove(); existingBadge=null; }
                    if(!_isMultiModel && !existingBadge){
                        var badge = document.createElement('div');
                        badge.className = 'afm-ai-badge';
                        badge.style.cssText = 'display:inline-flex;align-items:center;gap:5px;margin-top:8px;font-size:13px;font-weight:600;';
                        if(_isMLX){
                            badge.innerHTML = '<span style="font-size:15px;flex-shrink:0">⚡</span>'
                                + '<span style="background:linear-gradient(to right, #f97316, #eab308);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Swift MLX</span>';
                        } else {
                        badge.innerHTML = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAyCAYAAAATIfj2AAAAAXNSR0IArs4c6QAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAANKADAAQAAAABAAAAMgAAAADxNzqZAAAGSElEQVRoBe2aa4hVVRTHHU0nxTRNoZjRbCSnmDEle1l+iDJCKypIsyYCNchJLCoI6kNEQmhmhFBh5JcKi8KBHiSBNCMVZA/LsqnJGVLJsKxMranxNf3+t7su6+w5595zH8pILvi511577bXOPnefvfc544ABJ+XkHTiud6DqWGbr7e29nPiz4Ww4CO2wrqqqajvliSMMpAY2QJwcwvgUDD4hRsSFToCdUEjexqF/D4oLHApbgpFspL4UVkJX0PZMv/6VuNhl7oKPos/3F0x9CLzkfKRe6X36jc6FTYSDusKsrIi7ONo0qC/MifJrGBjnW4ot9SpH0rEkuBmmwRmgVWs3dMBmeADmgWQH1LOa9WRqwT/EugjTJ2D5m9E74QKogzHQC7vgA3iXWIcoyxeSV8HD0A1p5Vkca+KyYx8E50H4rGFKlB20zIyLF9rsDoX2XJ1Az1NZlDMUp+zD/RfYC1rRRsFZUA3FyhE6zOWXasnXMe+AGMxtdF7rAuxBfw26YCicC9PhfKiEvE+QLbAT9FxdDLfAKSA5AI0MSu3FCYPRVNsGJptQdIcjgu05c6D8Cd6EH50tTtXm2g6vB406VUSE9umw3/mtijikrRCgwQU5gj4x7IttEujiTBabD4aRcAnMhiaYC9fAZBji/N6jbqLnSr9MRLDdbw6U2yONaSt0vMEF2Rr2o02/YKvz0aaZu9DQP6lOnwtBe5aJVsuI0FBnjdmyz6AjHeIqdNSdNYkb0IPWmC3nxMVJY6P/yy7W3+gNvh/1c1z7YfSSBlTvgmhajbAk6DeBApvkXXmsX1JJkDGw24JRapmuNX90TVeTbWYvuiSCT7JAAbDNAt1FEy0A2gjLEmJcB37qfUt9nIJStoDJmpITEUEbpMnnKPPBLwJ7qWt3r4gQ6z7wopulgfrZMKvkZASa6qMH+j7qM0oOntCRmI8GeXxVU3FQQtd0ZgK0+ohZXfvNlHQRivcidjNoqwilzwpYVHSiTQHdFS+bqWTmdlHBinQmh6aaZoEX7Vkjiwz1nzsdrwK/Q1vgsla0tBdDsuHwuyV15VZ0nQfTCx2ugG4XxK8+mgp16aOV5kmOJS5/qGoFHJ0qMo5j4WcX4U/066HD2ZanClaGE7m+cfmeRl/h6lLXpwqP4xrX8S90naa1F9zr7Bpw8bt1qivI5NIZ0CQ3IzA8YsZsmf90gtMo6HGdFto1YBsNfj/QN7djIuR53F1Dm0+CvcW16W02WXDUymLS51egoc0aKZcmRyqvhdibXJ7IUo39UtemTT5yIA6njb4VmPzAi9RRq2TLVlePHCCdvSyVC9RLZ6ML0uZ0qXq5NNGLX+6MKWM4IL2RmugrTrgrd1gjZb3TK6lqjxvmAn7vdKk+rz6c6PU+XhiAnhN/Vot8S6DNP6x+8PEBS7AGOfQ9Iie0VcFbYLIx15ik4PmqeVNqcA9B5o5R6muNSU9SjHLsBJ9pCSj1aSsj6OPgDdcm9XZrTyxxqgW/D6mjTtUvwGJVnIRTMjFu2gZi3+jid6EvAn0H9yusXNZD3o88uZw46r1/OxSShThU5NRAnIGgKb0aCokGODx3wWkUdYAn4A8oJDpFrITJaWKbD/4ahLaKV+BXKCQ68jRZ/5JKAgyDebAW/Bss1Vj5CGver5y0Dwa9zIUneUwR0RlSh1EdfWZAuimWdqQE1IPppd1XAl2rUZ/XDGxXgz8X+m46pXQ6w5dpr60kPxLpY4aX06nUgB7cT31DVtdikvmAT1kNutP+5E41s5Kuo9RioGl+B5h8XNKFpu1EFk0Vf0ETfF/adCz5EEJ5EcNngVGHTi0AkXcb6n4lTXei9hdRrE7CXWBybVx/Gu+E8G3T+qjU34SmJvRd5RxXx/lU1Eayd1zCZUnB8amDuJVrA/ZT8/T7ysW/O8mvYnaSNbuE+9EnhcGx1YIuPE40ZZ+E6ph+2tdM5Dc+9Kl4nSSngb/ze6gvgQa4DJbDAfCiTfofb0DvhHugEfTsaaodBpPj8u0ic4PIOMeypii1f42AafBdCn+56Cb1WfIr/uv4gCRsAk25JNGmeWvQR5u0plv4a/kYGnSl/oDm0xfWSXwm6HVZu3k3aGXTs7MA8j3442l/DLTaaXC/gd6G74L+/Z8xCt+Wkx7/rzvwL0grYyNpW+UdAAAAAElFTkSuQmCC" width="18" height="18" style="flex-shrink:0" />'
                            + '<span style="background:'+_aiGradient+';-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Apple Intelligence</span>';
                        }
                        el.parentElement.insertBefore(badge, el.nextElementSibling?.nextElementSibling || null);
                    }
                }
            });
            document.title=document.title.replace('llama.cpp','AFM');
        }

        var _autoSelectDone = false;
        var _userClickedModel = false;
        var _isMultiModel = false; // detected from /v1/models count

        // Auto-select "foundation" model in router mode if no model is selected
        function autoSelectFoundation(){
            var trigger = getModelTrigger();
            if(!trigger) { _autoSelectDone = true; return; }
            var txt = (trigger.textContent || '').trim();
            // Already selected
            if(txt && !txt.includes('Select model')){ _autoSelectDone = true; return; }
            // Open the dropdown
            _selectingModel = true;
            trigger.click();
            setTimeout(function(){
                // Find the "foundation" option in the listbox
                var options = document.querySelectorAll('[role="option"]');
                var found = false;
                for(var i=0;i<options.length;i++){
                    var label = (options[i].textContent || '').trim().toLowerCase();
                    if(label.indexOf('foundation') !== -1 || label.indexOf('apple') !== -1){
                        options[i].click();
                        found = true;
                        break;
                    }
                }
                // If only one option and not found by name, click the first one
                if(!found && options.length === 1){
                    options[0].click();
                }
                // Close dropdown if still open
                setTimeout(function(){
                    var trigger2 = getModelTrigger();
                    if(trigger2){
                        var listbox = document.querySelector('[role="listbox"]');
                        if(listbox){ trigger2.click(); }
                    }
                    _selectingModel = false;
                    _autoSelectDone = true;
                }, 150);
            }, 300);
        }

        function getModelTrigger(){
            var form = document.querySelector('[data-slot="chat-form"]');
            if(!form) return null;

            // Prefer the actual model picker button (popover/listbox trigger)
            var popoverButtons = form.querySelectorAll('button[aria-haspopup="listbox"],button[data-slot="popover-trigger"]');
            if(popoverButtons.length > 0) return popoverButtons[0];

            // Fallback: find a button that looks like model text, not action buttons
            var buttons = form.querySelectorAll('button');
            for(var i=0;i<buttons.length;i++){
                var txt = (buttons[i].textContent || '').trim();
                if(!txt) continue;
                if(txt.includes('Select model')) return buttons[i];
                if(txt !== 'Send' && txt !== 'Stop' && txt !== '+') return buttons[i];
            }
            return null;
        }

        // Listen for user clicks on model dropdown options to track user intent
        document.addEventListener('click', function(e){
            if(!_isMultiModel) return;
            var el = e.target;
            while(el && el !== document.body){
                if(el.getAttribute && el.getAttribute('role') === 'option'){
                    _userClickedModel = true;
                    setTimeout(function(){
                        var trigger = getModelTrigger();
                        if(trigger){
                            var model = trigger.textContent.trim();
                            if(model && !model.includes('Select model')){
                                localStorage.setItem('afm-preferred-model', model);
                            }
                        }
                        _userClickedModel = false;
                    }, 300);
                    return;
                }
                el = el.parentElement;
            }
        }, true);

        // Model info strip
        var _lastModel = '';
        var _selectingModel = false; // Prevent repeated auto-select during dropdown animation
        var _modelsCache = null;

        function fmtCtx(n){
            if(!n) return '';
            if(n>=1000) return Math.round(n/1024)+'K ctx';
            return n+' ctx';
        }

        function getOrCreateStrip(){
            var el = document.getElementById('afm-model-info');
            if(el) return el;
            var trigger = getModelTrigger();
            if(!trigger) return null;
            var parent = trigger.parentElement;
            if(!parent) return null;
            el = document.createElement('div');
            el.id = 'afm-model-info';
            el.style.cssText = 'font-size:11px;color:#888;padding:2px 8px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:320px;';
            parent.insertBefore(el, trigger.nextSibling);
            return el;
        }

        function updateInfoStrip(){
            if(_selectingModel) return; // Don't interfere while selecting
            var trigger = getModelTrigger();
            if(!trigger) return;
            var model = trigger.textContent.trim();
            if(!model || model.includes('Select model')) {
                var strip = document.getElementById('afm-model-info');
                if(strip) strip.textContent = '';
                _lastModel = '';
                return;
            }
            // Re-render if strip was destroyed by SPA re-render even if model unchanged
            var stripExists = document.getElementById('afm-model-info');
            if(model === _lastModel && stripExists) return;
            _lastModel = model;

            // Fetch model details from /v1/models (cached) and /props
            var p1 = _modelsCache ? Promise.resolve(_modelsCache) : fetch('/v1/models').then(function(r){return r.json()}).then(function(d){_modelsCache=d;return d});
            var p2 = fetch('/props?model='+encodeURIComponent(model)).then(function(r){return r.json()});

            Promise.all([p1,p2]).then(function(res){
                var modelsData = res[0];
                var props = res[1];
                if(_lastModel !== model) return; // stale

                var backend = '';
                var hasTools = false;
                if(modelsData && modelsData.models){
                    for(var i=0;i<modelsData.models.length;i++){
                        var m = modelsData.models[i];
                        if(m.model === model){
                            // Extract backend from name like "model (Backend)"
                            var match = m.name && m.name.match(new RegExp('\\\\(([^)]+)\\\\)$'));
                            if(match) backend = match[1];
                            if(m.capabilities && m.capabilities.indexOf('tools')!==-1) hasTools=true;
                            break;
                        }
                    }
                }

                var hasVision = props.modalities && props.modalities.vision;
                var nCtx = props.default_generation_settings && props.default_generation_settings.n_ctx;

                var parts = [];
                if(backend) parts.push(backend);
                if(hasVision) parts.push('Vision');
                if(hasTools) parts.push('Tools');
                var ctx = fmtCtx(nCtx);
                if(ctx) parts.push(ctx);

                var strip = getOrCreateStrip();
                if(strip) strip.textContent = parts.join(' \\u00b7 ');
            }).catch(function(){});
        }

        function refreshModelList(){
            // Invalidate models cache so info strip picks up new backends
            _modelsCache = null;
        }

        function waitForSpaAndReveal(){
            // Wait for the SPA to render AND model auto-select to finish before revealing
            var attempts = 0;
            var check = setInterval(function(){
                attempts++;
                var h1 = document.querySelector('h1');
                var spaReady = h1 || attempts > 50;
                var selectReady = _autoSelectDone || attempts > 100; // max ~5s wait for select
                if(spaReady && selectReady){
                    clearInterval(check);
                    rebrand();
                    document.body.classList.add('afm-ready');
                }
            }, 50);
        }

        function init(){
            waitForSpaAndReveal();
            // Discover if gateway mode has multiple models, then auto-select foundation.
            fetch('/v1/models').then(function(r){return r.json()}).then(function(d){
                var count = d && d.data ? d.data.length : 0;
                _isMultiModel = count > 1;
                // In router mode, auto-select foundation after SPA renders
                if(_isMultiModel){
                    // Wait for the SPA model list to populate, then auto-select
                    var selectAttempts = 0;
                    var selectInterval = setInterval(function(){
                        selectAttempts++;
                        var trigger = getModelTrigger();
                        if(trigger || selectAttempts > 40){
                            clearInterval(selectInterval);
                            autoSelectFoundation();
                        }
                    }, 100);
                } else {
                    _autoSelectDone = true;
                }
            }).catch(function(){ _autoSelectDone = true; });

            // Update branding/info on real DOM changes rather than polling.
            var refreshTimer = null;
            function scheduleRefresh(){
                if(refreshTimer) clearTimeout(refreshTimer);
                refreshTimer = setTimeout(function(){
                    rebrand();
                    updateInfoStrip();
                }, 120);
            }
            scheduleRefresh();
            var observer = new MutationObserver(scheduleRefresh);
            observer.observe(document.documentElement, { childList: true, subtree: true });

            // Periodically check for new models from background port scanning
            setInterval(refreshModelList, 15000);
            setTimeout(refreshModelList, 5000);
        }

        if(document.readyState==='loading'){
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    })();
    </script>
    """

    /// Always-available local runtime console. This deliberately composes with
    /// the stock llama.cpp chat DOM instead of replacing it, so conversations,
    /// uploads, generation controls, and persistence keep their native behavior.
    private static let controlCenterTemplate = """
    <style>
      .afm-console-toggle {
        position: fixed; left: 14px; top: 14px; z-index: 999997;
        height: 40px; padding: 0 14px; border: 1px solid var(--afm-line-strong);
        border-radius: 12px; background: var(--afm-panel-solid); color: var(--afm-ink);
        box-shadow: var(--afm-shadow-control); cursor: pointer; font: 680 12px/1 ui-sans-serif, sans-serif;
      }
      .afm-console {
        position: fixed; inset: 10px auto 10px 10px; z-index: 999998;
        width: 372px; display: flex; flex-direction: column;
        border: 1px solid var(--afm-line); border-radius: 20px;
        background: color-mix(in srgb, var(--afm-panel-solid) 94%, transparent);
        color: var(--afm-ink); box-shadow: 0 24px 70px rgba(0,0,0,.28);
        backdrop-filter: blur(24px) saturate(130%); -webkit-backdrop-filter: blur(24px) saturate(130%);
        transform: translateX(calc(-100% - 18px)); transition: transform .2s ease;
        overflow: hidden; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
      }
      .afm-console.open { transform: translateX(0); }
      .afm-console-head { padding: 16px; border-bottom: 1px solid var(--afm-line); }
      .afm-console-brand { display:flex; align-items:center; gap:10px; }
      .afm-console-logo {
        width:34px; height:34px; display:grid; place-items:center; border-radius:11px;
        color:white; background:linear-gradient(135deg,var(--afm-orange),#ffb464); font-weight:820; font-size:12px;
        box-shadow:0 7px 18px color-mix(in srgb,var(--afm-orange) 28%,transparent);
      }
      .afm-console-title { min-width:0; flex:1; }
      .afm-console-title strong { display:block; font-size:14px; letter-spacing:-.02em; }
      .afm-console-title span { display:flex; align-items:center; gap:6px; color:var(--afm-muted); font-size:10px; margin-top:3px; }
      .afm-console-title i { width:7px; height:7px; border-radius:50%; background:var(--afm-green); box-shadow:0 0 0 3px color-mix(in srgb,var(--afm-green) 14%,transparent); }
      .afm-console-close { width:30px; height:30px; border:0; border-radius:9px; background:var(--afm-panel-muted); color:var(--afm-muted); cursor:pointer; font-size:17px; }
      .afm-console-scroll { overflow:auto; padding:12px; display:grid; gap:10px; }
      .afm-console-section { border:1px solid var(--afm-line); border-radius:14px; background:var(--afm-panel); padding:12px; }
      .afm-console-label { margin-bottom:9px; color:var(--afm-muted); font-size:9px; font-weight:780; letter-spacing:.11em; text-transform:uppercase; }
      .afm-console-select {
        width:100%; height:36px; padding:0 30px 0 10px; border:1px solid var(--afm-line-strong); border-radius:10px;
        color:var(--afm-ink); background:var(--afm-panel-solid); font-size:11px; outline:none;
      }
      .afm-console-modelmeta { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
      .afm-chip { padding:4px 7px; border-radius:999px; background:var(--afm-panel-muted); color:var(--afm-muted); font-size:9px; font-weight:650; }
      .afm-chip.on { color:var(--afm-green); background:color-mix(in srgb,var(--afm-green) 11%,transparent); }
      .afm-console-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
      .afm-console-stat { min-width:0; padding:9px; border-radius:10px; background:var(--afm-panel-muted); }
      .afm-console-stat span { display:block; color:var(--afm-muted); font-size:9px; }
      .afm-console-stat strong { display:block; margin-top:4px; overflow:hidden; text-overflow:ellipsis; color:var(--afm-ink); font-size:13px; font-variant-numeric:tabular-nums; }
      .afm-console-meter { grid-column:1/-1; height:4px; overflow:hidden; border-radius:4px; background:var(--afm-line); }
      .afm-console-meter i { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--afm-green),#7ed6b1); transition:width .25s; }
      .afm-console-row { display:flex; align-items:center; gap:8px; padding:6px 0; color:var(--afm-muted); font-size:10px; }
      .afm-console-row + .afm-console-row { border-top:1px solid var(--afm-line); }
      .afm-console-row b { margin-left:auto; max-width:178px; color:var(--afm-ink); font-size:10px; font-weight:650; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      .afm-console-actions { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
      .afm-console-btn {
        min-height:34px; padding:7px 9px; border:1px solid var(--afm-line); border-radius:10px;
        background:var(--afm-panel-solid); color:var(--afm-ink); cursor:pointer; font-size:10px; font-weight:670;
      }
      .afm-console-btn:hover { border-color:var(--afm-line-strong); background:var(--afm-panel-muted); }
      .afm-console-btn.primary { color:white; border-color:transparent; background:var(--afm-orange); }
      .afm-console-note { color:var(--afm-muted); font-size:9px; line-height:1.45; margin-top:8px; }
      .afm-launch-choice { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:9px; }
      .afm-launch-choice button { height:36px; border:1px solid var(--afm-line); border-radius:10px; background:var(--afm-panel-muted); color:var(--afm-muted); cursor:pointer; font-size:11px; font-weight:700; }
      .afm-launch-choice button.active { border-color:color-mix(in srgb,var(--afm-orange) 55%,transparent); background:color-mix(in srgb,var(--afm-orange) 13%,var(--afm-panel-solid)); color:var(--afm-orange-strong); }
      .afm-field { display:grid; gap:5px; margin-top:8px; }
      .afm-field > span { color:var(--afm-muted); font-size:9px; font-weight:650; }
      .afm-field input,.afm-field select,.afm-field textarea { width:100%; min-height:34px; padding:7px 9px; border:1px solid var(--afm-line); border-radius:9px; background:var(--afm-panel-solid); color:var(--afm-ink); font:10px/1.35 ui-monospace,SFMono-Regular,monospace; outline:none; }
      .afm-field textarea { min-height:54px; resize:vertical; }
      .afm-option-group { border-top:1px solid var(--afm-line); margin-top:10px; padding-top:9px; }
      .afm-option-group summary { color:var(--afm-ink); cursor:pointer; font-size:10px; font-weight:700; list-style:none; display:flex; justify-content:space-between; }
      .afm-option-group summary::after { content:'+'; color:var(--afm-muted); }
      .afm-option-group[open] summary::after { content:'−'; }
      .afm-option-list { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:8px; }
      .afm-option-list .wide { grid-column:1/-1; }
      .afm-check { display:flex; align-items:center; gap:6px; min-height:30px; padding:5px 7px; border:1px solid var(--afm-line); border-radius:8px; color:var(--afm-muted); font-size:9px; }
      .afm-check input { accent-color:var(--afm-orange); }
      .afm-launch-command { max-height:90px; overflow:auto; margin-top:8px; padding:8px; border-radius:8px; background:#090d12; color:#a7f3d0; font:9px/1.45 ui-monospace,SFMono-Regular,monospace; white-space:pre-wrap; word-break:break-all; display:none; }
      .afm-launch-status { display:none; margin-top:8px; padding:8px; border-radius:9px; background:var(--afm-panel-muted); color:var(--afm-muted); font-size:9px; line-height:1.4; }
      .afm-launch-status.show { display:block; }
      .afm-launch-log { max-height:92px; overflow:auto; margin-top:6px; white-space:pre-wrap; color:var(--afm-muted); font:8px/1.35 ui-monospace,SFMono-Regular,monospace; }
      .afm-console-footer { padding:10px 14px; border-top:1px solid var(--afm-line); color:var(--afm-muted); font-size:9px; display:flex; justify-content:space-between; }
      .afm-dash-toggle { display:none !important; }
      @media (min-width: 900px) {
        body.afm-console-open [data-slot="sidebar-wrapper"] { padding-left:382px; transition:padding-left .2s ease; }
        body.afm-console-open .afm-console-toggle { display:none; }
      }
      @media (max-width: 899px) {
        .afm-console { inset:0 auto 0 0; width:min(372px,94vw); border-radius:0 18px 18px 0; }
        .afm-console-toggle { left:10px; top:10px; }
      }
      @media (prefers-reduced-motion: reduce) { .afm-console { transition:none; } }
    </style>
    <button class="afm-console-toggle" id="afm-console-toggle" aria-label="Open AFM controls">AFM controls</button>
    <aside class="afm-console" id="afm-console" aria-label="AFM local runtime controls">
      <div class="afm-console-head">
        <div class="afm-console-brand">
          <div class="afm-console-logo">AFM</div>
          <div class="afm-console-title"><strong>Local runtime</strong><span><i id="afm-runtime-dot"></i><em id="afm-runtime-status">connecting</em></span></div>
          <button class="afm-console-close" id="afm-console-close" aria-label="Close AFM controls">&times;</button>
        </div>
      </div>
      <div class="afm-console-scroll">
        <section class="afm-console-section" id="afm-launcher">
          <div class="afm-console-label">Launch local runtime</div>
          <div class="afm-launch-choice">
            <button type="button" data-backend="foundation" class="active">Foundation</button>
            <button type="button" data-backend="mlx">MLX / Hugging Face</button>
          </div>
          <div class="afm-field" id="afm-launch-model-row" hidden>
            <span>Model id or local path</span>
            <input id="afm-launch-model" list="afm-launch-models" placeholder="mlx-community/Qwen3.5-4B-4bit" autocomplete="off">
            <datalist id="afm-launch-models"></datalist>
          </div>
          <div id="afm-launch-options"></div>
          <div class="afm-console-actions" style="margin-top:10px">
            <button class="afm-console-btn" id="afm-launch-preview">Preview command</button>
            <button class="afm-console-btn primary" id="afm-launch-start">Launch runtime</button>
          </div>
          <pre class="afm-launch-command" id="afm-launch-command"></pre>
          <div class="afm-launch-status" id="afm-launch-status"><strong id="afm-launch-status-title">Starting…</strong><div id="afm-launch-status-detail"></div><pre class="afm-launch-log" id="afm-launch-log"></pre><div class="afm-console-actions" style="margin-top:7px"><button class="afm-console-btn primary" id="afm-launch-open" style="display:none">Open runtime</button><button class="afm-console-btn" id="afm-launch-stop" style="display:none">Stop runtime</button></div></div>
          <div class="afm-console-note">Startup-only changes create a managed loopback runtime. The current chat remains available until the new runtime is healthy.</div>
        </section>
        <section class="afm-console-section">
          <div class="afm-console-label">Current runtime model</div>
          <select class="afm-console-select" id="afm-console-model" aria-label="Active model"><option>Discovering models…</option></select>
          <div class="afm-console-modelmeta" id="afm-console-caps"></div>
        </section>
        <section class="afm-console-section">
          <div class="afm-console-label">Resources</div>
          <div class="afm-console-grid">
            <div class="afm-console-stat"><span>GPU pressure</span><strong id="afm-console-gpu">—</strong></div>
            <div class="afm-console-stat"><span>Requests</span><strong id="afm-console-active">—</strong></div>
            <div class="afm-console-meter"><i id="afm-console-gpu-bar"></i></div>
            <div class="afm-console-stat"><span>Queue</span><strong id="afm-console-queue">—</strong></div>
            <div class="afm-console-stat"><span>Generated</span><strong id="afm-console-tokens">—</strong></div>
          </div>
        </section>
        <section class="afm-console-section">
          <div class="afm-console-label">Runtime</div>
          <div class="afm-console-row"><span>Backend</span><b id="afm-console-backend">—</b></div>
          <div class="afm-console-row"><span>Context</span><b id="afm-console-context">—</b></div>
          <div class="afm-console-row"><span>Streaming</span><b id="afm-console-streaming">—</b></div>
          <div class="afm-console-row"><span>API</span><b id="afm-console-api">—</b></div>
        </section>
        <section class="afm-console-section">
          <div class="afm-console-label">Local storage</div>
          <div class="afm-console-row"><span>AFM state</span><b id="afm-console-state-path" title="Click to copy">~/.afm</b></div>
          <div class="afm-console-row"><span>Model cache</span><b id="afm-console-cache-path" title="Click to copy">—</b></div>
          <div class="afm-console-note">Chat history remains in the native WebUI browser store. AFM runtime state and the model cache stay on this Mac.</div>
        </section>
        <section class="afm-console-section">
          <div class="afm-console-label">Controls</div>
          <div class="afm-console-actions">
            <button class="afm-console-btn primary" id="afm-console-new">New chat</button>
            <button class="afm-console-btn" id="afm-console-settings">Chat settings</button>
            <button class="afm-console-btn" id="afm-console-stats">Live statistics</button>
            <button class="afm-console-btn" id="afm-console-docs">API docs</button>
          </div>
          <div class="afm-console-note" id="afm-console-tools-note">Tool calling is selected per request and executed by the connected local client.</div>
        </section>
      </div>
      <div class="afm-console-footer"><span id="afm-console-version">AFM</span><span>localhost only</span></div>
    </aside>
    <script>
    (function(){
      var panel=document.getElementById('afm-console');
      var modelSelect=document.getElementById('afm-console-model');
      var modelDetails={};
      var launchBackend='foundation', launchURL='', launchPoll=null, launchDraft={values:{},flags:[]};
      var launchGroups=[
        {name:'Common',open:true,fields:[
          {key:'--instructions',label:'Instructions',type:'textarea',backends:'both',wide:true},
          {key:'--temperature',label:'Temperature',type:'number',backends:'both',placeholder:'server default'},
          {key:'--prewarm',label:'Prewarm',type:'select',backends:'both',choices:['','y','n']},
          {key:'--stop',label:'Stop sequences',type:'text',backends:'both',placeholder:'###,END'},
          {key:'--guided-json',label:'Guided JSON schema',type:'textarea',backends:'both',wide:true},
          {key:'--no-streaming',label:'Disable streaming',type:'flag',backends:'both'},
          {key:'--verbose',label:'Verbose logging',type:'flag',backends:'both'},
          {key:'--very-verbose',label:'Full request logging',type:'flag',backends:'both'},
          {key:'--vv',label:'Boundary trace logging',type:'flag',backends:'both'}
        ]},
        {name:'Foundation options',fields:[
          {key:'--adapter',label:'LoRA adapter (.fmadapter)',type:'text',backends:'foundation',wide:true},
          {key:'--randomness',label:'Sampling mode',type:'text',backends:'foundation',placeholder:'greedy or random:top-p=0.9'},
          {key:'--permissive-guardrails',label:'Permissive guardrails',type:'flag',backends:'foundation'},
          {key:'--gateway',label:'Discover local backends',type:'flag',backends:'foundation'}
        ]},
        {name:'MLX sampling',fields:[
          {key:'--top-p',label:'Top P',type:'number',backends:'mlx'},
          {key:'--top-k',label:'Top K',type:'number',backends:'mlx'},
          {key:'--min-p',label:'Min P',type:'number',backends:'mlx'},
          {key:'--presence-penalty',label:'Presence penalty',type:'number',backends:'mlx'},
          {key:'--repetition-penalty',label:'Repetition penalty',type:'number',backends:'mlx'},
          {key:'--max-tokens',label:'Maximum tokens',type:'number',backends:'mlx'},
          {key:'--seed',label:'Seed',type:'number',backends:'mlx'},
          {key:'--max-logprobs',label:'Maximum logprobs',type:'number',backends:'mlx'}
        ]},
        {name:'MLX runtime + cache',fields:[
          {key:'--mlx-runtime',label:'Runtime',type:'select',backends:'mlx',choices:['','auto','mlx','dwarfstar']},
          {key:'--gguf-file',label:'Exact repository GGUF',type:'text',backends:'mlx',wide:true},
          {key:'--kv-bits',label:'KV bits',type:'select',backends:'mlx',choices:['','4','8']},
          {key:'--kv-cache-size',label:'KV cache size',type:'number',backends:'mlx'},
          {key:'--kv-eviction',label:'KV eviction',type:'select',backends:'mlx',choices:['','none','streaming']},
          {key:'--prefill-step-size',label:'Prefill step size',type:'number',backends:'mlx'},
          {key:'--concurrent',label:'Concurrent requests',type:'number',backends:'mlx'},
          {key:'--chat-template',label:'Chat template',type:'text',backends:'mlx',wide:true},
          {key:'--dtype',label:'Dtype',type:'text',backends:'mlx'},
          {key:'--vlm',label:'Vision model (VLM)',type:'flag',backends:'mlx'},
          {key:'--raw',label:'Raw model output',type:'flag',backends:'mlx'},
          {key:'--trust-remote-code',label:'Trust remote code',type:'flag',backends:'mlx'},
          {key:'--enable-prefix-caching',label:'Prefix caching',type:'flag',backends:'mlx'}
        ]},
        {name:'Tools + reasoning',fields:[
          {key:'--tool-call-parser',label:'Tool-call parser',type:'select',backends:'mlx',wide:true,choices:['','none','afm_adaptive_xml','hermes','llama3_json','gemma','mistral','qwen3_xml']},
          {key:'--reasoning-effort',label:'Reasoning effort',type:'select',backends:'mlx',choices:['','low','high','max']},
          {key:'--default-chat-template-kwargs',label:'Chat-template kwargs (JSON)',type:'textarea',backends:'mlx',wide:true},
          {key:'--fix-tool-args',label:'Repair tool arguments',type:'flag',backends:'mlx'},
          {key:'--enable-grammar-constraints',label:'Grammar constraints',type:'flag',backends:'mlx'},
          {key:'--no-think',label:'Disable thinking',type:'flag',backends:'mlx'}
        ]},
        {name:'Speculative decoding',fields:[
          {key:'--mtp',label:'Enable MTP',type:'flag',backends:'mlx'},
          {key:'--mtp-depth',label:'MTP depth',type:'number',backends:'mlx'},
          {key:'--mtp-model',label:'MTP model override',type:'text',backends:'mlx',wide:true},
          {key:'--dspark-support',label:'DSpark support GGUF',type:'text',backends:'mlx',wide:true},
          {key:'--dspark-draft-tokens',label:'DSpark draft tokens',type:'number',backends:'mlx'},
          {key:'--dspark-confidence',label:'DSpark confidence',type:'number',backends:'mlx'},
          {key:'--dspark-strict',label:'DSpark strict',type:'flag',backends:'mlx'},
          {key:'--eagle3',label:'EAGLE3 drafter directory',type:'text',backends:'mlx',wide:true}
        ]},
        {name:'Profiling + files',fields:[
          {key:'--cache-profile-path',label:'Cache profile JSONL',type:'text',backends:'mlx',wide:true},
          {key:'--gpu-capture',label:'GPU capture path',type:'text',backends:'mlx',wide:true},
          {key:'--gpu-trace',label:'GPU trace seconds',type:'number',backends:'mlx'},
          {key:'--gpu-profile',label:'GPU profiling',type:'flag',backends:'mlx'},
          {key:'--gpu-profile-bw',label:'DRAM bandwidth (mactop)',type:'flag',backends:'mlx'}
        ]},
        {name:'Telegram bridge',fields:[
          {key:'--telegram-bot-token',label:'Bot token',type:'password',backends:'both',wide:true},
          {key:'--telegram-allow',label:'Allowed user ids',type:'text',backends:'both',wide:true},
          {key:'--telegram-format',label:'Reply format',type:'select',backends:'both',choices:['','markdown','plain','html']},
          {key:'--telegram-require-prefix',label:'Required prefix',type:'text',backends:'both'}
        ]}
      ];

      function fieldApplies(field){return field.backends==='both'||field.backends===launchBackend}
      function stashLaunchOptions(){
        document.querySelectorAll('#afm-launch-options [data-option]').forEach(function(el){
          var key=el.getAttribute('data-option');
          if(el.type==='checkbox'){
            launchDraft.flags=launchDraft.flags.filter(function(f){return f!==key}); if(el.checked)launchDraft.flags.push(key);
          }else if(el.value){launchDraft.values[key]=el.value}else{delete launchDraft.values[key]}
        });
      }
      function renderLaunchOptions(){
        var host=document.getElementById('afm-launch-options'); host.innerHTML='';
        launchGroups.forEach(function(group){
          var fields=group.fields.filter(fieldApplies); if(!fields.length)return;
          var details=document.createElement('details'); details.className='afm-option-group'; details.open=!!group.open;
          var summary=document.createElement('summary'); summary.textContent=group.name; details.appendChild(summary);
          var list=document.createElement('div'); list.className='afm-option-list';
          fields.forEach(function(field){
            var wrap=document.createElement('label'); wrap.className=(field.wide?'wide ':'')+(field.type==='flag'?'afm-check':'afm-field');
            var input;
            if(field.type==='flag'){
              input=document.createElement('input'); input.type='checkbox'; input.checked=launchDraft.flags.indexOf(field.key)!==-1;
              wrap.appendChild(input); var label=document.createElement('span'); label.textContent=field.label; wrap.appendChild(label);
            }else{
              var label=document.createElement('span'); label.textContent=field.label; wrap.appendChild(label);
              if(field.type==='select'){
                input=document.createElement('select'); (field.choices||[]).forEach(function(choice){var o=document.createElement('option');o.value=choice;o.textContent=choice||'Default';input.appendChild(o)});
              }else if(field.type==='textarea'){input=document.createElement('textarea')}
              else{input=document.createElement('input');input.type=field.type||'text';if(field.type==='number')input.step='any'}
              input.value=launchDraft.values[field.key]||''; if(field.placeholder)input.placeholder=field.placeholder;
            }
            input.setAttribute('data-option',field.key); input.onchange=stashLaunchOptions; wrap.appendChild(input); list.appendChild(wrap);
          });
          details.appendChild(list); host.appendChild(details);
        });
      }
      function setLaunchBackend(backend){
        stashLaunchOptions(); launchBackend=backend;
        document.querySelectorAll('.afm-launch-choice button').forEach(function(b){b.classList.toggle('active',b.getAttribute('data-backend')===backend)});
        document.getElementById('afm-launch-model-row').hidden=backend!=='mlx'; renderLaunchOptions();
      }
      document.querySelectorAll('.afm-launch-choice button').forEach(function(button){button.onclick=function(){setLaunchBackend(button.getAttribute('data-backend'))}});
      function launchPayload(dryRun){
        stashLaunchOptions();
        return {backend:launchBackend,model:launchBackend==='mlx'?document.getElementById('afm-launch-model').value.trim():null,values:launchDraft.values,flags:Array.from(new Set(launchDraft.flags)),dryRun:!!dryRun};
      }
      function launchRequest(payload){
        return fetch('/afm/launcher/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(function(r){return r.json().then(function(j){if(!r.ok)throw new Error(j.reason||j.error||('HTTP '+r.status));return j})});
      }
      function displayCommand(command){var pre=document.getElementById('afm-launch-command');pre.style.display='block';pre.textContent=(command||[]).map(function(a){return /[\\s"']/.test(a)?JSON.stringify(a):a}).join(' ')}
      document.getElementById('afm-launch-preview').onclick=function(){
        launchRequest(launchPayload(true)).then(function(r){displayCommand(r.command)}).catch(function(e){var s=document.getElementById('afm-launch-status');s.classList.add('show');text('afm-launch-status-title','Cannot build command');text('afm-launch-status-detail',e.message)});
      };
      function pollLaunch(){
        Promise.all([fetch('/afm/launcher/status').then(function(r){return r.json()}),fetch('/afm/launcher/log').then(function(r){return r.json()})]).then(function(all){
          var s=all[0],log=all[1].log||''; text('afm-launch-log',log.slice(-12000));
          if(s.healthy){text('afm-launch-status-title','Runtime ready');text('afm-launch-status-detail',(s.backend||'runtime')+(s.model?' · '+s.model:'')+' · port '+s.port);launchURL=s.url;document.getElementById('afm-launch-open').style.display='';document.getElementById('afm-launch-stop').style.display='';clearInterval(launchPoll);launchPoll=null}
          else if(s.running){text('afm-launch-status-title','Starting local runtime…');text('afm-launch-status-detail','Downloading or loading on port '+s.port);document.getElementById('afm-launch-stop').style.display=''}
          else{text('afm-launch-status-title','Runtime stopped');text('afm-launch-status-detail','Check the log for startup errors.');document.getElementById('afm-launch-stop').style.display='none';clearInterval(launchPoll);launchPoll=null}
        }).catch(function(e){text('afm-launch-status-detail',e.message)});
      }
      document.getElementById('afm-launch-start').onclick=function(){
        var status=document.getElementById('afm-launch-status');status.classList.add('show');text('afm-launch-status-title','Starting local runtime…');text('afm-launch-status-detail','Validating configuration');document.getElementById('afm-launch-open').style.display='none';
        launchRequest(launchPayload(false)).then(function(r){displayCommand(r.command);launchURL=r.url;if(launchPoll)clearInterval(launchPoll);pollLaunch();launchPoll=setInterval(pollLaunch,900)}).catch(function(e){text('afm-launch-status-title','Launch failed');text('afm-launch-status-detail',e.message)});
      };
      document.getElementById('afm-launch-open').onclick=function(){if(launchURL)window.open(launchURL,'_blank','noopener')};
      document.getElementById('afm-launch-stop').onclick=function(){fetch('/afm/launcher/stop',{method:'POST'}).then(function(){pollLaunch()})};
      function loadLaunchProfile(){fetch('/afm/launcher/profile').then(function(r){return r.json()}).then(function(p){if(!p.backend)return;launchDraft={values:p.values||{},flags:p.flags||[]};document.getElementById('afm-launch-model').value=p.model||'';launchBackend=p.backend;document.querySelectorAll('.afm-launch-choice button').forEach(function(b){b.classList.toggle('active',b.getAttribute('data-backend')===launchBackend)});document.getElementById('afm-launch-model-row').hidden=launchBackend!=='mlx';renderLaunchOptions()}).catch(function(){})}
      function setOpen(open){ panel.classList.toggle('open',open); document.body.classList.toggle('afm-console-open',open); }
      document.getElementById('afm-console-toggle').onclick=function(){setOpen(true)};
      document.getElementById('afm-console-close').onclick=function(){setOpen(false)};
      setOpen(window.matchMedia('(min-width: 900px)').matches);

      function text(id,value){ var el=document.getElementById(id); if(el) el.textContent=value; }
      function compact(n){ n=Number(n)||0; return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':String(Math.round(n)); }
      function metric(raw,name){ var m=raw.match(new RegExp('^'+name+'(?:\\\\{[^}]*\\\\})?\\\\s+([^\\\\s]+)','m')); return m?Number(m[1]):null; }
      function nativeModelTrigger(){
        var form=document.querySelector('[data-slot="chat-form"]')||document.querySelector('.conversation-chat-form');
        return form && (form.querySelector('button[aria-haspopup="listbox"]')||form.querySelector('button[data-slot="popover-trigger"]'));
      }
      function activeNativeModel(){ var t=nativeModelTrigger(); return t?(t.textContent||'').trim():''; }
      function selectNativeModel(id){
        var trigger=nativeModelTrigger(); if(!trigger) return;
        trigger.click();
        setTimeout(function(){
          var options=document.querySelectorAll('[role="option"]');
          for(var i=0;i<options.length;i++){
            var label=(options[i].textContent||'').trim();
            if(label===id || label.indexOf(id)!==-1){ options[i].click(); return; }
          }
          trigger.click();
        },180);
      }
      function paintCaps(id){
        var d=modelDetails[id]||{}, caps=d.capabilities||[], host=document.getElementById('afm-console-caps'); host.innerHTML='';
        ['chat','vision','tools','embeddings'].forEach(function(cap){
          var chip=document.createElement('span'), yes=caps.indexOf(cap)!==-1 || (cap==='tools'&&caps.indexOf('tool_calling')!==-1);
          chip.className='afm-chip'+(yes?' on':''); chip.textContent=(yes?'✓ ':'')+cap; host.appendChild(chip);
        });
        text('afm-console-tools-note',caps.indexOf('tools')!==-1?'Tool calling supported. Tools are authorized and executed per request by the connected local client.':'This model does not advertise tool-calling support.');
      }
      function loadModels(){
        fetch('/v1/models').then(function(r){return r.json()}).then(function(data){
          modelDetails={}; (data.models||[]).forEach(function(d){modelDetails[d.model]=d});
          var models=(data.data||[]).filter(function(m){var caps=(modelDetails[m.id]&&modelDetails[m.id].capabilities)||[];return !(caps.length===1&&caps[0]==='embeddings')}); modelSelect.innerHTML='';
          var modelHints=document.getElementById('afm-launch-models'); modelHints.innerHTML='';
          models.forEach(function(m){var hint=document.createElement('option');hint.value=m.id;modelHints.appendChild(hint)});
          models.forEach(function(m){ var o=document.createElement('option'); o.value=m.id; o.textContent=m.id+(m.status&&m.status.value==='unloaded'?' · available':''); modelSelect.appendChild(o); });
          var current=activeNativeModel(); var matched=models.find(function(m){return current.indexOf(m.id)!==-1});
          if(matched) modelSelect.value=matched.id; paintCaps(modelSelect.value);
          if(modelSelect.value) loadProps(modelSelect.value);
        }).catch(function(){ modelSelect.innerHTML='<option>Models unavailable</option>'; });
      }
      function loadProps(model){
        fetch('/props?model='+encodeURIComponent(model)).then(function(r){return r.json()}).then(function(p){
          var g=p.default_generation_settings||{}; text('afm-console-context',compact(g.n_ctx||0)+' tokens');
        }).catch(function(){});
      }
      modelSelect.onchange=function(){ var id=this.value; paintCaps(id); loadProps(id); fetch('/models/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:id})}).catch(function(){}); selectNativeModel(id); };

      function loadRuntime(){
        Promise.all([fetch('/health').then(function(r){return r.json()}),fetch('/afm/runtime').then(function(r){return r.json()})]).then(function(all){
          var h=all[0],r=all[1]; text('afm-runtime-status',h.status||'healthy'); text('afm-console-version','AFM '+(r.version||h.version||''));
          text('afm-console-backend',r.backend||'local'); text('afm-console-streaming',r.streaming?'enabled':'disabled'); text('afm-console-api',location.host);
          text('afm-console-state-path',r.persistence||'~/.afm'); text('afm-console-cache-path',r.modelCache||'default');
        }).catch(function(){ text('afm-runtime-status','offline'); document.getElementById('afm-runtime-dot').style.background='#ef4444'; });
      }
      function pollMetrics(){ fetch('/metrics').then(function(r){return r.text()}).then(function(raw){
        var gpu=metric(raw,'afm:gpu_cache_usage_perc'), active=metric(raw,'afm:num_requests_running'), started=metric(raw,'afm:requests_started_total'), done=metric(raw,'afm:requests_completed_total');
        if((active==null||active===0)&&started!=null&&done!=null) active=Math.max(0,started-done);
        var queue=metric(raw,'afm:num_requests_waiting'), tokens=metric(raw,'afm:generation_tokens_total');
        text('afm-console-gpu',gpu==null?'not reported':(gpu*100).toFixed(1)+'%'); text('afm-console-active',String(active==null?0:active));
        text('afm-console-queue',String(queue==null?0:queue)); text('afm-console-tokens',compact(tokens));
        document.getElementById('afm-console-gpu-bar').style.width=(gpu==null?0:Math.max(0,Math.min(100,gpu*100)))+'%';
      }).catch(function(){}); }

      function clickNative(words){
        var all=document.querySelectorAll('button,a');
        for(var i=0;i<all.length;i++){ if(all[i].closest('.afm-console'))continue; var hay=((all[i].getAttribute('aria-label')||'')+' '+(all[i].getAttribute('title')||'')+' '+(all[i].textContent||'')).toLowerCase(); if(words.some(function(w){return hay.indexOf(w)!==-1})){all[i].click();return true;} }
        return false;
      }
      document.getElementById('afm-console-new').onclick=function(){ if(!clickNative(['new chat','new conversation'])) location.href='/'; };
      document.getElementById('afm-console-settings').onclick=function(){
        if(clickNative(['settings','parameters']))return;
        var gears=document.querySelectorAll('button.rounded-full.backdrop-blur-lg');
        if(gears.length)gears[gears.length-1].click();
      };
      document.getElementById('afm-console-stats').onclick=function(){ var b=document.getElementById('afm-dash-toggle'); if(b)b.click(); };
      document.getElementById('afm-console-docs').onclick=function(){ window.open('/docs','_blank','noopener'); };
      ['afm-console-state-path','afm-console-cache-path','afm-console-api'].forEach(function(id){document.getElementById(id).onclick=function(){navigator.clipboard&&navigator.clipboard.writeText(this.textContent||'')}});
      window.addEventListener('resize',function(){ if(window.innerWidth<900&&panel.classList.contains('open')) document.body.classList.remove('afm-console-open'); });
      renderLaunchOptions(); loadLaunchProfile(); loadRuntime(); loadModels(); pollMetrics(); setInterval(pollMetrics,1000); setInterval(loadRuntime,15000);
    })();
    </script>
    """

    /// Live `/metrics` dashboard injected alongside the webui customCSS.
    /// Renders a slide-out panel from the right edge of the page that polls
    /// `GET /metrics` every 1s, parses the Prometheus text exposition output,
    /// and renders every `afm:*` series with p50/p95/p99 derived from
    /// cumulative bucket counts. Toggle button is fixed to the right edge.
    private static let dashboardTemplate = """
    <style>
      .afm-dash-toggle {
        position: fixed; right: 4.25rem; top: 1rem;
        z-index: 999998;
        min-width: 88px; height: 40px; padding: 0 14px;
        border: 1px solid var(--afm-line); background: var(--afm-panel); color: var(--afm-ink);
        border-radius: 999px;
        cursor: pointer; font-size: 12px; font-weight: 680; letter-spacing: -0.01em;
        display: flex; align-items: center; justify-content: center;
        gap: 7px;
        box-shadow: 0 8px 24px rgba(24, 32, 44, 0.10);
        backdrop-filter: blur(18px) saturate(130%);
        -webkit-backdrop-filter: blur(18px) saturate(130%);
        transition: background 0.15s, border-color 0.15s, transform 0.15s;
      }
      .afm-dash-toggle::before {
        content: ""; width: 7px; height: 7px; border-radius: 999px;
        background: var(--afm-green); box-shadow: 0 0 0 4px color-mix(in srgb, var(--afm-green) 12%, transparent);
      }
      .afm-dash-toggle:hover {
        background: var(--afm-panel-solid); border-color: var(--afm-line-strong); transform: translateY(-1px);
      }
      /* Non-modal: no backdrop so the rest of the page stays interactive
         (chat input, model picker, etc.) while the dashboard is open. */
      .afm-dash {
        position: fixed; top: 0; right: 0; bottom: 0;
        width: min(560px, 94vw);
        z-index: 999999;
        background: #0f172a; color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 13px; line-height: 1.5;
        transform: translateX(100%); transition: transform 0.25s ease-out;
        overflow-y: auto;
        box-shadow: -8px 0 32px rgba(0,0,0,0.4);
      }
      .afm-dash.open { transform: translateX(0); }
      .afm-dash header {
        position: sticky; top: 0; z-index: 1;
        background: linear-gradient(to right, #1e293b, #0f172a);
        padding: 14px 20px; border-bottom: 1px solid #334155;
        display: flex; align-items: baseline; gap: 14px;
      }
      .afm-dash header h1 {
        margin: 0; font-size: 16px; font-weight: 600;
        background: linear-gradient(to right, #3b82f6, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
      }
      .afm-dash header .afm-dash-meta { color: #94a3b8; font-size: 11px; }
      .afm-dash header .afm-dash-meta b { color: #cbd5e1; font-weight: 600; }
      .afm-dash header .afm-dash-close {
        margin-left: auto; background: none; border: 1px solid #334155;
        color: #cbd5e1; padding: 4px 10px; border-radius: 4px; cursor: pointer;
        font-size: 12px;
      }
      .afm-dash header .afm-dash-close:hover { background: #1e293b; }
      .afm-dash section { padding: 16px 20px; border-bottom: 1px solid #1e293b; }
      .afm-dash section h2 {
        margin: 0 0 10px; font-size: 11px; font-weight: 700;
        color: #64748b; text-transform: uppercase; letter-spacing: 1.2px;
      }
      .afm-dash .grid {
        display: grid; gap: 8px;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      }
      .afm-tile {
        background: #1e293b; border: 1px solid #334155; border-radius: 6px;
        padding: 10px 12px;
      }
      .afm-tile .lbl { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
      .afm-tile .val { color: #f1f5f9; font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 4px; }
      .afm-tile .sub { color: #64748b; font-size: 10px; margin-top: 2px; font-variant-numeric: tabular-nums; }
      .afm-tile.accent .val { color: #60a5fa; }
      .afm-tile.live .val { color: #34d399; }
      .afm-bar {
        height: 4px; background: #334155; border-radius: 2px;
        margin-top: 6px; overflow: hidden;
      }
      .afm-bar > div { height: 100%; background: linear-gradient(to right, #3b82f6, #a855f7); transition: width 0.3s; }
      .afm-hist {
        background: #1e293b; border: 1px solid #334155; border-radius: 6px;
        padding: 10px 12px;
      }
      .afm-hist .row1 {
        display: flex; align-items: baseline; gap: 12px;
        font-variant-numeric: tabular-nums;
      }
      .afm-hist .name { color: #cbd5e1; font-size: 12px; font-weight: 600; flex: 1; }
      .afm-hist .pcts { color: #94a3b8; font-size: 11px; }
      .afm-hist .pcts b { color: #f1f5f9; }
      .afm-hist .summary { color: #64748b; font-size: 10px; margin-top: 2px; font-variant-numeric: tabular-nums; }
      .afm-hist .bars {
        display: flex; align-items: flex-end; gap: 1px; height: 36px;
        margin-top: 6px;
      }
      .afm-hist .bars > span {
        flex: 1; background: #475569; border-radius: 1px 1px 0 0; min-height: 1px;
      }
      .afm-hist .bars > span.hot { background: linear-gradient(to top, #3b82f6, #60a5fa); }
      .afm-spark {
        height: 30px; display: flex; align-items: flex-end; gap: 1px;
        margin-top: 8px;
      }
      .afm-spark > span {
        flex: 1; background: #34d399; border-radius: 1px 1px 0 0; min-height: 1px;
        opacity: 0.5;
      }
      .afm-spark > span:last-child { opacity: 1; }
      .afm-reasons { display: grid; gap: 4px; }
      .afm-reasons > div {
        display: grid; grid-template-columns: 110px 1fr 60px;
        align-items: center; gap: 8px; font-size: 12px;
        font-variant-numeric: tabular-nums;
      }
      .afm-reasons .name { color: #cbd5e1; }
      .afm-reasons .bar { height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
      .afm-reasons .bar > div { height: 100%; background: #34d399; transition: width 0.3s; }
      .afm-reasons .v { color: #f1f5f9; text-align: right; }
      .afm-status { font-size: 11px; color: #64748b; }
      .afm-status.err { color: #f87171; }
      .afm-empty { color: #64748b; font-size: 11px; font-style: italic; }
      @media (max-width: 767px) {
        .afm-dash-toggle { top: 0.65rem; right: 3.5rem; min-width: 72px; height: 36px; padding: 0 11px; }
        .afm-dash { width: 100vw; }
      }
    </style>
    <button class="afm-dash-toggle" id="afm-dash-toggle" title="Open AFM /metrics dashboard (Esc to close)">Live stats</button>
    <aside class="afm-dash" id="afm-dash" aria-hidden="true">
      <header>
        <h1>AFM /metrics</h1>
        <div class="afm-dash-meta">
          <span id="afm-dash-model">—</span> ·
          <span id="afm-dash-status" class="afm-status">connecting…</span>
        </div>
        <button class="afm-dash-close" id="afm-dash-close">Close</button>
      </header>

      <section>
        <h2>Live</h2>
        <div class="grid" id="afm-live-grid"></div>
        <div style="margin-top:10px; font-size:11px; color:#64748b;">Decode rate (last 60s):</div>
        <div class="afm-spark" id="afm-spark"></div>
      </section>

      <section>
        <h2>Counters</h2>
        <div class="grid" id="afm-counter-grid"></div>
      </section>

      <section>
        <h2>Finished reasons</h2>
        <div class="afm-reasons" id="afm-reasons"></div>
      </section>

      <section>
        <h2>Latency histograms</h2>
        <div class="grid" style="grid-template-columns: 1fr;" id="afm-latency-grid"></div>
      </section>

      <section>
        <h2>Size + sampling histograms</h2>
        <div class="grid" style="grid-template-columns: 1fr;" id="afm-size-grid"></div>
      </section>

      <section style="font-size:10px; color:#475569;">
        Last scrape: <span id="afm-dash-ts">—</span>. Polling interval: 1s.
      </section>
    </aside>
    <script>
    (function(){
      // ── Toggle plumbing ────────────────────────────────────────────
      var dash = document.getElementById('afm-dash');
      var toggle = document.getElementById('afm-dash-toggle');
      var closeBtn = document.getElementById('afm-dash-close');
      function openDash() {
        dash.classList.add('open');
        dash.setAttribute('aria-hidden', 'false');
        startPolling();
      }
      function closeDash() {
        dash.classList.remove('open');
        dash.setAttribute('aria-hidden', 'true');
      }
      toggle.addEventListener('click', function(){
        dash.classList.contains('open') ? closeDash() : openDash();
      });
      closeBtn.addEventListener('click', closeDash);
      // Esc closes only when focus is NOT inside the chat input
      // (so the user's escape-to-cancel-typing doesn't accidentally close the panel).
      document.addEventListener('keydown', function(e){
        if (e.key !== 'Escape' || !dash.classList.contains('open')) return;
        var t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        closeDash();
      });

      // ── Prometheus text parser ─────────────────────────────────────
      // Returns: { name: { type, help, samples: [{labels:{...}, value, le?}] } }
      function parsePrometheus(txt) {
        var out = {}, name;
        var lines = txt.split(/\\r?\\n/);
        for (var i = 0; i < lines.length; i++) {
          var l = lines[i];
          if (!l) continue;
          if (l.charCodeAt(0) === 35) {  // '#'
            var m = l.match(/^# (HELP|TYPE) (\\S+) (.*)$/);
            if (!m) continue;
            name = m[2];
            out[name] = out[name] || { type: '', help: '', samples: [] };
            if (m[1] === 'HELP') out[name].help = m[3];
            else                 out[name].type = m[3];
            continue;
          }
          // sample line: NAME{labels} VALUE   or   NAME VALUE
          var sp = l.indexOf(' ');
          if (sp < 0) continue;
          var head = l.slice(0, sp);
          var val = parseFloat(l.slice(sp + 1));
          var lb = head.indexOf('{');
          var key, labels = {};
          if (lb < 0) {
            key = head;
          } else {
            key = head.slice(0, lb);
            var rb = head.lastIndexOf('}');
            var inner = head.slice(lb + 1, rb);
            // crude label parser — handles escaped \\" inside values
            var re = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\\\]|\\\\.)*)"/g;
            var lm;
            while ((lm = re.exec(inner)) !== null) {
              labels[lm[1]] = lm[2].replace(/\\\\(.)/g, '$1');
            }
          }
          // For histogram bucket lines, key ends with "_bucket"
          var rec = out[key] || (out[key] = { type: 'untyped', help: '', samples: [] });
          rec.samples.push({ labels: labels, value: val });
        }
        return out;
      }

      // ── Histogram percentile from cumulative buckets ───────────────
      // Buckets are emitted in ascending order; values are cumulative.
      // Returns an interpolated quantile value, or null if no data.
      function percentile(buckets, total, p) {
        if (!buckets || !buckets.length || !total) return null;
        var target = total * p;
        var prevCount = 0, prevLe = 0;
        for (var i = 0; i < buckets.length; i++) {
          var b = buckets[i];
          if (b.count >= target) {
            if (b.le === Infinity) return prevLe;  // cap at last finite bucket
            if (b.count === prevCount) return b.le;
            // linear interp inside (prevLe, le]
            var frac = (target - prevCount) / (b.count - prevCount);
            return prevLe + (b.le - prevLe) * frac;
          }
          prevCount = b.count; prevLe = (b.le === Infinity ? prevLe : b.le);
        }
        return prevLe;
      }
      function histBuckets(rec) {
        // Convert bucket samples into [{le, count}], sorted by le ascending,
        // with +Inf last.
        if (!rec || !rec.samples) return [];
        return rec.samples.map(function(s){
          var leStr = s.labels.le;
          var le = leStr === '+Inf' ? Infinity : parseFloat(leStr);
          return { le: le, count: s.value };
        }).filter(function(b){ return !isNaN(b.le); })
          .sort(function(a,b){ return a.le - b.le; });
      }

      // ── Format helpers ─────────────────────────────────────────────
      function fmtN(v) {
        if (v == null || isNaN(v)) return '—';
        if (v >= 1e9) return (v/1e9).toFixed(2)+'B';
        if (v >= 1e6) return (v/1e6).toFixed(2)+'M';
        if (v >= 1e3) return (v/1e3).toFixed(1)+'k';
        return Math.round(v).toString();
      }
      function fmtSec(v) {
        if (v == null || isNaN(v) || !isFinite(v)) return '—';
        if (v < 0.001) return (v*1e6).toFixed(0)+'µs';
        if (v < 1) return (v*1000).toFixed(0)+'ms';
        if (v < 60) return v.toFixed(2)+'s';
        return (v/60).toFixed(1)+'m';
      }
      function fmtPct(v) {
        if (v == null || isNaN(v)) return '—';
        return (v*100).toFixed(1)+'%';
      }

      // ── Tile + histogram + reason renderers ────────────────────────
      function renderTiles(host, tiles) {
        // tiles: [{key, lbl, val, sub?, cls?, barPct?}]
        var existing = {};
        host.querySelectorAll('[data-key]').forEach(function(el){
          existing[el.getAttribute('data-key')] = el;
        });
        tiles.forEach(function(t){
          var el = existing[t.key];
          if (!el) {
            el = document.createElement('div');
            el.className = 'afm-tile' + (t.cls ? ' '+t.cls : '');
            el.setAttribute('data-key', t.key);
            el.innerHTML = '<div class="lbl"></div><div class="val"></div><div class="sub"></div><div class="afm-bar" style="display:none"><div></div></div>';
            host.appendChild(el);
          }
          el.className = 'afm-tile' + (t.cls ? ' '+t.cls : '');
          el.querySelector('.lbl').textContent = t.lbl;
          el.querySelector('.val').textContent = t.val;
          el.querySelector('.sub').textContent = t.sub || '';
          var bar = el.querySelector('.afm-bar');
          if (t.barPct != null) {
            bar.style.display = '';
            bar.firstChild.style.width = Math.max(0, Math.min(100, t.barPct*100))+'%';
          } else {
            bar.style.display = 'none';
          }
          delete existing[t.key];
        });
        Object.values(existing).forEach(function(el){ el.remove(); });
      }

      function renderHistogram(host, key, label, rec, fmt) {
        var bs = histBuckets(rec);
        var sumRec = (state.metrics[key.replace(/_bucket$/,'')+'_sum']) || null;
        var countRec = (state.metrics[key.replace(/_bucket$/,'')+'_count']) || null;
        var sum = sumRec ? sumRec.samples.reduce(function(a,s){return a+s.value;},0) : null;
        var total = countRec ? countRec.samples.reduce(function(a,s){return a+s.value;},0) : null;
        var p50 = percentile(bs, total, 0.5);
        var p95 = percentile(bs, total, 0.95);
        var p99 = percentile(bs, total, 0.99);

        var el = host.querySelector('[data-key="'+key+'"]');
        if (!el) {
          el = document.createElement('div');
          el.className = 'afm-hist';
          el.setAttribute('data-key', key);
          el.innerHTML = '<div class="row1"><div class="name"></div><div class="pcts"></div></div><div class="summary"></div><div class="bars"></div>';
          host.appendChild(el);
        }
        el.querySelector('.name').textContent = label;
        el.querySelector('.pcts').innerHTML =
          'p50 <b>'+fmt(p50)+'</b> · p95 <b>'+fmt(p95)+'</b> · p99 <b>'+fmt(p99)+'</b>';
        el.querySelector('.summary').textContent =
          'count '+fmtN(total)+' · sum '+(sum!=null?fmt(sum):'—')+
          ' · mean '+(total>0?fmt(sum/total):'—');

        // Per-bucket bar widths (relative to most populated finite bucket)
        var bars = el.querySelector('.bars');
        bars.innerHTML = '';
        if (!total) {
          bars.style.display = 'none';
        } else {
          bars.style.display = '';
          // Compute non-cumulative deltas for visual clarity
          var prevCount = 0;
          var deltas = [];
          for (var i = 0; i < bs.length; i++) {
            deltas.push(bs[i].count - prevCount);
            prevCount = bs[i].count;
          }
          var max = Math.max.apply(Math, deltas);
          var hotIdx = deltas.indexOf(max);
          deltas.forEach(function(d, i){
            var s = document.createElement('span');
            s.style.height = max > 0 ? Math.max(1, (d/max)*100)+'%' : '1%';
            s.title = 'le=' + (bs[i].le === Infinity ? '+Inf' : bs[i].le) + ': ' + d + ' obs';
            if (i === hotIdx) s.classList.add('hot');
            bars.appendChild(s);
          });
        }
      }

      function renderReasons(host, sumByReason) {
        host.innerHTML = '';
        var keys = Object.keys(sumByReason).sort();
        if (!keys.length || keys.every(function(k){ return sumByReason[k] === 0; })) {
          host.innerHTML = '<div class="afm-empty">No completed requests yet.</div>';
          return;
        }
        var max = Math.max.apply(Math, keys.map(function(k){ return sumByReason[k]; }));
        keys.forEach(function(k){
          var v = sumByReason[k];
          var row = document.createElement('div');
          row.innerHTML = '<div class="name"></div><div class="bar"><div></div></div><div class="v"></div>';
          row.querySelector('.name').textContent = k;
          row.querySelector('.bar > div').style.width = max > 0 ? (v/max)*100+'%' : '0';
          row.querySelector('.v').textContent = fmtN(v);
          host.appendChild(row);
        });
      }

      // ── Polling state ─────────────────────────────────────────────
      var state = {
        polling: false,
        timer: null,
        prev: null,        // previous gen-token snapshot for spark rate
        prevTs: null,
        spark: [],         // last 60 tok/s samples (instantaneous, for sparkline only)
        peakClient: 0,     // client-side high-water for inflight (serial mode fallback)
        // ── Per-completed-request tracking ─────────────────────────────
        // We detect a request finishing by `requests_completed_total`
        // ticking up between polls; at that moment the dashboard's
        // shown rate becomes the CURRENT request's actual decode rate
        // (Δgen / Δdecode_time_sum), matching what the chat UI shows
        // per message (model-reported decode tok/s, not wall clock).
        // The value sticks until another request completes.
        prevSnap: null,    // { compTot, genTot, decodeSum }
        lastReqTps: null,  // tok/s of the most recent completed request
        lastReqGen: null,  // tokens of the most recent completed request
        lastReqAt: null,   // wall-clock instant we observed the completion
        // ── Sticky live values ─────────────────────────────────────────
        // Whatever Active was last seen non-zero, keep it visible so the
        // panel reflects "what just happened" instead of snapping to 0.
        lastActive: 0,
        lastActiveAt: null,
        metrics: {}
      };

      function tick() {
        fetch('/metrics', { headers: { Accept: 'text/plain' } })
          .then(function(r){
            if (!r.ok) throw new Error('HTTP '+r.status);
            return r.text();
          })
          .then(function(txt){
            state.metrics = parsePrometheus(txt);
            paint();
            var statusEl = document.getElementById('afm-dash-status');
            statusEl.textContent = 'live'; statusEl.classList.remove('err');
            document.getElementById('afm-dash-ts').textContent = new Date().toLocaleTimeString();
          })
          .catch(function(err){
            var statusEl = document.getElementById('afm-dash-status');
            statusEl.textContent = 'error: '+err.message; statusEl.classList.add('err');
          });
      }
      function startPolling() {
        if (state.polling) return;
        state.polling = true;
        tick();
        state.timer = setInterval(tick, 1000);
      }
      // Note: we keep polling even when closed so the spark stays continuous.

      // ── Paint loop ────────────────────────────────────────────────
      function paint() {
        var m = state.metrics;
        function single(name) {
          var rec = m[name];
          if (!rec || !rec.samples || !rec.samples.length) return null;
          // For our schema each gauge/counter has a single sample.
          return rec.samples[0].value;
        }
        function modelName() {
          var rec = m['afm:max_concurrent_slots'];
          return rec && rec.samples[0] ? rec.samples[0].labels.model_name : '—';
        }

        var modelHost = document.getElementById('afm-dash-model');
        modelHost.textContent = '';
        var modelStrong = document.createElement('b');
        modelStrong.textContent = modelName();
        modelHost.appendChild(modelStrong);

        var runningRaw = single('afm:num_requests_running') || 0;
        var waiting = single('afm:num_requests_waiting') || 0;
        var peakRaw = single('afm:batch_size_peak') || 0;
        var slots   = single('afm:max_concurrent_slots') || 0;
        var gpu     = single('afm:gpu_cache_usage_perc');
        var genTot  = single('afm:generation_tokens_total') || 0;
        var promTot = single('afm:prompt_tokens_total') || 0;
        var startTot = single('afm:requests_started_total') || 0;
        var compTot = single('afm:requests_completed_total') || 0;
        var hits    = single('afm:radix_cache_hits_total') || 0;
        var misses  = single('afm:radix_cache_misses_total') || 0;
        // Decode time accumulator from histogram _sum line — drives the
        // accurate per-request tok/s computation below.
        var decodeSum = (function(){
          var rec = state.metrics['afm:request_decode_time_seconds_sum'];
          if (!rec || !rec.samples || !rec.samples.length) return 0;
          return rec.samples.reduce(function(a,s){return a + s.value;}, 0);
        })();

        // In serial mode (no BatchScheduler) the running/peak gauge readers
        // are never registered and the server-side values stay at 0. Fall
        // back to (started - completed) so the live panel still tells the
        // user whether anything is actively generating.
        var inflightDerived = Math.max(0, startTot - compTot);
        var running = runningRaw > 0 ? runningRaw : inflightDerived;
        if (running > state.peakClient) state.peakClient = running;
        var peak = Math.max(peakRaw, state.peakClient);
        var serialMode = (slots === 0);

        var now = Date.now();
        if (running > 0) { state.lastActive = running; state.lastActiveAt = now; }

        // Detect a request completing on this poll: requests_completed_total
        // ticked up. When it does, compute that request's TRUE decode tok/s
        // from the deltas of cumulative counters — this matches the chat
        // UI's per-message "X.XX t/s" (which is gen_tokens / decode_time
        // straight from the model). The value sticks in state.lastReqTps
        // until another request completes.
        if (state.prevSnap == null) {
          state.prevSnap = { compTot: compTot, genTot: genTot, decodeSum: decodeSum };
        } else if (compTot > state.prevSnap.compTot) {
          var dGen = genTot - state.prevSnap.genTot;
          var dDec = decodeSum - state.prevSnap.decodeSum;
          if (dDec > 0 && dGen > 0) {
            state.lastReqTps = dGen / dDec;
            state.lastReqGen = dGen;
            state.lastReqAt = now;
          }
          state.prevSnap = { compTot: compTot, genTot: genTot, decodeSum: decodeSum };
        }

        // Spark (visual only): instantaneous wall-clock delta of genTot.
        var tpsInst = null;
        if (state.prev != null && state.prevTs != null) {
          var dt = (now - state.prevTs) / 1000;
          if (dt > 0) tpsInst = (genTot - state.prev) / dt;
        }
        state.prev = genTot; state.prevTs = now;
        if (tpsInst != null) {
          state.spark.push(Math.max(0, tpsInst));
          if (state.spark.length > 60) state.spark.shift();
        }

        // Display priority for the Decode rate tile:
        //   1. Actively generating now → use decodeSum-based live rate
        //      (Δgen since the active request started / Δdecode if we
        //       can compute it; fall back to the spark window).
        //   2. Last completed request's rate (sticky) — matches the chat
        //      UI's per-message t/s number.
        //   3. Lifetime cumulative gen / decode_sum.
        //   4. 0.
        var displayTps = null, displaySub = '', displayLabel = 'Decode rate';
        if (running > 0) {
          // Live: prefer the spark's recent peak so a fast in-progress
          // burst shows realistic tok/s instead of being averaged down
          // against the 1s polling window.
          var window = state.spark.slice(-5);
          var nz = window.filter(function(v){ return v > 0; });
          if (nz.length) {
            displayTps = Math.max.apply(Math, nz);
            displaySub = 'generating';
          } else if (state.lastReqTps != null) {
            displayTps = state.lastReqTps;
            displaySub = 'generating (last: ' + state.lastReqTps.toFixed(1) + ' tok/s)';
          } else {
            displayTps = 0; displaySub = 'generating';
          }
        } else if (state.lastReqTps != null) {
          displayTps = state.lastReqTps;
          var ageS = state.lastReqAt ? Math.max(0, Math.round((now - state.lastReqAt) / 1000)) : 0;
          displaySub = 'last request · ' + (state.lastReqGen ? state.lastReqGen + ' tok' : '') +
                      (ageS > 0 ? ' · ' + ageS + 's ago' : '');
          displayLabel = 'Decode rate (last)';
        } else if (decodeSum > 0) {
          displayTps = genTot / decodeSum;
          displaySub = 'lifetime avg';
        } else {
          displayTps = 0;
          displaySub = 'idle';
        }

        // Active tile: sticky high-water + sticky last-non-zero count.
        var activeVal = String(running);
        var activeSub;
        if (running > 0) {
          activeSub = 'peak ' + peak;
        } else if (state.lastActiveAt != null) {
          var aS = Math.max(0, Math.round((now - state.lastActiveAt) / 1000));
          activeSub = 'last ' + state.lastActive + ' · ' + aS + 's ago · peak ' + peak;
        } else {
          activeSub = 'peak ' + peak;
        }

        // Sustained (wall-clock) throughput — mean of the spark's recent
        // non-empty window. This is what Grafana's Token Throughput panel
        // computes via rate(generation_tokens_total[5m]). Always lower
        // than `displayTps` because it includes idle gaps between
        // requests; both numbers are useful, they answer different
        // questions ("how fast does the model decode?" vs "how many
        // tokens is the system sustaining?").
        var sustained = null;
        if (state.spark.length > 0) {
          var sum = state.spark.reduce(function(a, b) { return a + b; }, 0);
          sustained = sum / state.spark.length;
        }

        // Live tiles
        var liveTiles = [
          { key: 'tps', lbl: displayLabel,
            val: displayTps == null ? '—' : displayTps.toFixed(1) + ' tok/s',
            sub: displaySub, cls: 'live' },
          { key: 'sustained', lbl: 'Sustained throughput',
            val: sustained == null ? '—' : sustained.toFixed(1) + ' tok/s',
            sub: 'wall-clock · last ' + state.spark.length + 's',
            cls: 'live' },
          { key: 'inflight', lbl: serialMode ? 'Active' : 'In-flight',
            val: activeVal, sub: activeSub, cls: 'accent',
            barPct: slots > 0 ? running/slots : (running > 0 ? 1 : 0) },
        ];
        var conn = single('afm:num_active_connections') || 0;
        var connPeak = single('afm:active_connections_peak') || 0;
        liveTiles.push({ key: 'conn', lbl: 'Connections',
          val: String(conn),
          sub: 'peak ' + connPeak,
          cls: 'accent' });
        if (!serialMode) {
          liveTiles.push({ key: 'queue', lbl: 'Queue depth',
            val: String(waiting), sub: 'cap ' + slots });
        }
        liveTiles.push({ key: 'gpu', lbl: 'GPU memory',
          val: gpu == null ? '—' : fmtPct(gpu),
          sub: gpu == null ? 'not exported' : 'of recommended VRAM',
          barPct: gpu });
        var radixFill = single('afm:radix_cache_fill_perc');
        liveTiles.push({ key: 'radix', lbl: 'Prefix cache fill',
          val: radixFill == null ? '—' : fmtPct(radixFill),
          sub: radixFill == null ? '--enable-prefix-caching off' : 'radix tree',
          barPct: radixFill });
        renderTiles(document.getElementById('afm-live-grid'), liveTiles);
        var sparkHost = document.getElementById('afm-spark');
        sparkHost.innerHTML = '';
        var max = Math.max.apply(Math, state.spark.length ? state.spark : [1]);
        state.spark.forEach(function(v){
          var s = document.createElement('span');
          s.style.height = max > 0 ? Math.max(1, (v/max)*100)+'%' : '1%';
          s.title = v.toFixed(1)+' tok/s';
          sparkHost.appendChild(s);
        });

        // Counter tiles
        var hitRate = (hits + misses) > 0 ? hits / (hits + misses) : null;
        renderTiles(document.getElementById('afm-counter-grid'), [
          { key: 'gen', lbl: 'Generation tokens', val: fmtN(genTot) },
          { key: 'prompt', lbl: 'Prompt tokens', val: fmtN(promTot) },
          { key: 'started', lbl: 'Requests started', val: fmtN(startTot) },
          { key: 'completed', lbl: 'Requests completed', val: fmtN(compTot), sub: startTot ? fmtPct(compTot/startTot) + ' of started' : '' },
          { key: 'hits', lbl: 'Radix cache hits', val: fmtN(hits) },
          { key: 'misses', lbl: 'Radix cache misses', val: fmtN(misses), sub: hitRate == null ? '' : 'hit rate ' + fmtPct(hitRate), barPct: hitRate },
        ]);

        // Reasons
        var reasonRec = m['afm:request_success_total'];
        var reasonMap = {};
        if (reasonRec && reasonRec.samples) {
          reasonRec.samples.forEach(function(s){
            var k = s.labels.finished_reason || 'unknown';
            reasonMap[k] = (reasonMap[k] || 0) + s.value;
          });
        }
        renderReasons(document.getElementById('afm-reasons'), reasonMap);

        // Latency histograms
        var latencyHost = document.getElementById('afm-latency-grid');
        var latencyHists = [
          ['afm:e2e_request_latency_seconds_bucket',     'End-to-end latency', fmtSec],
          ['afm:request_queue_time_seconds_bucket',      'Queue time',          fmtSec],
          ['afm:request_inference_time_seconds_bucket',  'Inference time',      fmtSec],
          ['afm:request_prefill_time_seconds_bucket',    'Prefill time',        fmtSec],
          ['afm:request_decode_time_seconds_bucket',     'Decode time',         fmtSec],
          ['afm:time_to_first_token_seconds_bucket',     'Time to first token', fmtSec],
          ['afm:time_per_output_token_seconds_bucket',   'Time per output token', fmtSec],
        ];
        latencyHists.forEach(function(h){
          renderHistogram(latencyHost, h[0], h[1], m[h[0]], h[2]);
        });

        // Size + sampling
        var sizeHost = document.getElementById('afm-size-grid');
        var sizeHists = [
          ['afm:request_prompt_tokens_bucket',     'Prompt tokens / request',     fmtN],
          ['afm:request_generation_tokens_bucket', 'Generation tokens / request', fmtN],
          ['afm:request_params_n_bucket',          'Sampling param n',            fmtN],
          ['afm:request_params_best_of_bucket',    'Sampling param best_of',      fmtN],
        ];
        sizeHists.forEach(function(h){
          renderHistogram(sizeHost, h[0], h[1], m[h[0]], h[2]);
        });
      }

      // Don't auto-poll — only when opened.
      // (User can still click the toggle to peek.)
    })();
    </script>
    """

    /// Serve the webui with custom CSS injected
    private func serveWebuiWithCustomCSS(webuiFilePath: String, req: Request) async throws -> Response {
        let fileURL = URL(fileURLWithPath: webuiFilePath)
        let compressedData = try Data(contentsOf: fileURL)

        // Decompress gzip data
        guard let decompressedData = try? Self.gunzip(compressedData),
              var htmlString = String(data: decompressedData, encoding: .utf8) else {
            // Fallback: serve compressed if decompression fails
            var headers = HTTPHeaders()
            headers.add(name: .contentType, value: "text/html; charset=utf-8")
            headers.add(name: .contentEncoding, value: "gzip")
            headers.add(name: "Cache-Control", value: "no-cache")
            return Response(status: .ok, headers: headers, body: .init(data: compressedData))
        }

        // Inject custom CSS before </head>
        if let headEndRange = htmlString.range(of: "</head>") {
            htmlString.insert(contentsOf: customCSS, at: headEndRange.lowerBound)
        }

        var headers = HTTPHeaders()
        headers.add(name: .contentType, value: "text/html; charset=utf-8")
        headers.add(name: "Cache-Control", value: "no-cache")

        return Response(status: .ok, headers: headers, body: .init(string: htmlString))
    }
    
    public func start() async throws {
        // Print ASCII art splash screen
        let version = BuildInfo.fullVersion

        // ANSI color codes - Apple Intelligence inspired gradient
        let cyan = "\u{001B}[36m"
        let blue = "\u{001B}[34m"
        let magenta = "\u{001B}[35m"
        let brightCyan = "\u{001B}[96m"
        let brightBlue = "\u{001B}[94m"
        let brightMagenta = "\u{001B}[95m"
        let white = "\u{001B}[97m"
        let gray = "\u{001B}[90m"
        let reset = "\u{001B}[0m"
        let bold = "\u{001B}[1m"

        // Center the version string properly (box content width is 68 chars)
        let boxContentWidth = 68
        let versionTextPadding = (boxContentWidth - version.count) / 2
        let versionLeftPad = String(repeating: " ", count: versionTextPadding)
        let versionRightPad = String(repeating: " ", count: boxContentWidth - version.count - versionTextPadding)

        print("")
        print("  \(brightCyan)╔════════════════════════════════════════════════════════════════════╗\(reset)")
        print("  \(brightCyan)║\(reset)                                                                    \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                    \(brightMagenta)█████╗\(reset) \(brightBlue)███████╗\(reset)\(brightCyan)███╗   ███╗\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)██╔══██╗\(reset)\(brightBlue)██╔════╝\(reset)\(brightCyan)████╗ ████║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)███████║\(reset)\(brightBlue)█████╗\(reset)  \(brightCyan)██╔████╔██║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)██╔══██║\(reset)\(brightBlue)██╔══╝\(reset)  \(brightCyan)██║╚██╔╝██║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(brightMagenta)██║  ██║\(reset)\(brightBlue)██║\(reset)     \(brightCyan)██║ ╚═╝ ██║\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                   \(gray)╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝\(reset)                      \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                                                                    \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)           \(white)Apple Foundation Models - OpenAI Compatible API\(reset)          \(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)\(versionLeftPad)\(bold)\(brightBlue)\(version)\(reset)\(versionRightPad)\(brightCyan)║\(reset)")
        print("  \(brightCyan)║\(reset)                                                                    \(brightCyan)║\(reset)")
        print("  \(brightCyan)╚════════════════════════════════════════════════════════════════════╝\(reset)")
        print("")

        // Initialize Foundation Models only for the Foundation route. MLX and
        // provider-backed servers must remain usable with the Xcode 26 compiler.
        if Self.usesAppleFoundationBackend(
            mlxModelID: mlxModelID,
            hasMLXModel: mlxModel != nil,
            hasProviderModel: afmModel != nil
        ) {
#if compiler(>=6.4)
            if #available(macOS 26.0, *) {
                try await FoundationModelService.initialize(instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails, prewarm: prewarmEnabled)
            }
#else
            throw Abort(
                .serviceUnavailable,
                reason: "Apple Foundation Models require the Swift 6.4 toolchain or newer.")
#endif
        }

        let repoURL = "https://github.com/scouzi1966/maclocal-api"
        let link = "\u{001B}]8;;\(repoURL)\u{001B}\\\(repoURL)\u{001B}]8;;\u{001B}\\"
        print("  \(gray)🚀 Server: http://\(hostname):\(port)\(reset)")
        print("  \(gray)📦 \(link)\(reset)")
        print("")
        print("  📡 Endpoints:")
        print("     • POST   /v1/chat/completions    - Chat completion (streaming supported)")
        print("     • GET    /v1/models              - List available models")
        print("     • GET    /health                 - Health check")
        print("")
        print("  ⚙️  Configuration:")
        print("     • Streaming:          \(streamingEnabled ? "✓ enabled" : "✗ disabled")")
        print("     • Prewarm:            \(prewarmEnabled ? "✓ enabled" : "✗ disabled")")
        if webuiEnabled {
            if webuiPath != nil {
                print("     • WebUI:              ✓ enabled (with image/PDF upload)")
            } else {
                print("     • WebUI:              ⚠️  enabled but not found (run 'make webui')")
            }
        }
        if let temp = temperature {
            print("     • Temperature:        \(String(format: "%.1f", temp))")
        }
        if let rand = randomness {
            print("     • Randomness:         \(rand)")
        }
        if permissiveGuardrails {
            print("     • Guardrails:         ⚠️  permissive mode")
        }
        if let adapterPath = adapter {
            print("     • Adapter:            \(adapterPath)")
        }
        if gatewayEnabled {
            print("     • Gateway:            ✓ enabled (multi-backend proxy)")
        }
        if veryVerbose {
            let red = "\u{001B}[38;5;196m"
            let pink = "\u{001B}[38;5;213m"
            let purple = "\u{001B}[38;5;135m"
            let teal = "\u{001B}[38;5;43m"
            let orange = "\u{001B}[38;5;208m"
            print("  🎨 Log colors (-V):")
            print("     • \(red)Red\(reset)      User prompt")
            print("     • \(pink)Pink\(reset)     Full request JSON")
            print("     • \(purple)Purple\(reset)   Reasoning")
            print("     • \(teal)Teal\(reset)     Content / answer / usage")
            print("     • \(orange)Orange\(reset)   Start / done bookends")
        }
        print("")
        print("  ℹ️  Requires macOS 26+ with Apple Intelligence")
        print("  💡 Press Ctrl+C to stop the server")
        if let mlxModel = mlxModelID {
            print("  💡 OpenClaw:  afm mlx -m \(mlxModel) --openclaw-config")
        } else {
            print("  💡 OpenClaw:  afm mlx -m <model> --openclaw-config")
        }
        if gatewayEnabled {
            print("")
            let yellow = "\u{001B}[33m"
            print("  ⚠️  API Key for detected backends: \(yellow)\(afmAPIKey)\(reset)")
            print("     This is NOT a security measure and is considered unsafe and insecure.")
            print("     It is a shared passphrase for backends absolutely requiring API keys")
            print("     (e.g. Jan). Set this key in your backend's API")
            print("     key settings if it rejects requests.")
        }
        print("")
        print("  ─────────────────────────────────────────────────────────────────────────")
        print("")

        // Start backend discovery scanning (gateway mode only)
        if gatewayEnabled, let discovery = app.backendDiscovery {
            await discovery.startPeriodicScanning()

            let discovered = await discovery.allDiscoveredModels()
            if !discovered.isEmpty {
                print("  🔍 Discovered LLM Backends:")
                // Group by backend name
                var byBackend: [String: [String]] = [:]
                for model in discovered {
                    byBackend[model.backendName, default: []].append(model.id)
                }
                for (backend, modelIds) in byBackend.sorted(by: { $0.key < $1.key }) {
                    print("     • \(backend): \(modelIds.count) model(s)")
                    for id in modelIds.prefix(5) {
                        print("       - \(id)")
                    }
                    if modelIds.count > 5 {
                        print("       ... and \(modelIds.count - 5) more")
                    }
                }
                print("")
            }
        }

        // Start the server
        try await app.server.start(address: .hostname(hostname, port: port))

        if let telegramConfiguration {
            let bridge = try TelegramBridge(config: telegramConfiguration)
            try await bridge.start()
            self.telegramBridge = bridge
        }

        // Open browser if webui is enabled
        if webuiEnabled && webuiPath != nil && ProcessInfo.processInfo.environment["AFM_WEBUI_MANAGED_CHILD"] != "1" {
            let url = "http://\(browserLaunchHost(for: hostname)):\(port)"
            print("  🌐 Opening WebUI in browser: \(url)")
            print("")
            Task { @MainActor in
                self.openBrowser(url: url)
            }
        }

        // Wait indefinitely (until shutdown is called)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Store continuation for later use in shutdown
            app.storage[ContinuationKey.self] = continuation
        }
    }
    
    public func shutdown() {
        print("🛑 Shutting down server...")
        telegramBridge?.stop()
        telegramBridge = nil

        // Shutdown the server first
        Task {
            _ = await webRuntimeManager.stop()
            // Stop backend discovery
            if let discovery = app.backendDiscovery {
                await discovery.stopScanning()
            }

            await app.server.shutdown()

            if let mlxModel {
                await mlxModel.unload()
            }
            if let afmModel {
                await afmModel.unload()
            }

            print("Server shutdown complete")

            // Resume the continuation to exit the wait
            if let continuation = app.storage[ContinuationKey.self] {
                continuation.resume()
                app.storage[ContinuationKey.self] = nil
            }
        }
    }

    /// Find the webui index.html.gz file
    private static func findWebuiPath() -> String? {
        let fileManager = FileManager.default
        let cwd = fileManager.currentDirectoryPath

        // Get the executable's absolute directory. argv[0] is unreliable when invoked
        // via PATH (it is just "afm"), so go through the Mach-O loader.
        var size: UInt32 = 0
        _ = _NSGetExecutablePath(nil, &size)
        let executableURL: URL
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: Int(size))
            if _NSGetExecutablePath(&buffer, &size) == 0 {
                executableURL = URL(fileURLWithPath: String(cString: buffer)).resolvingSymlinksInPath()
            } else if let bundleExec = Bundle.main.executableURL {
                executableURL = bundleExec.resolvingSymlinksInPath()
            } else {
                executableURL = URL(fileURLWithPath: cwd).appendingPathComponent(CommandLine.arguments[0])
            }
        } else if let bundleExec = Bundle.main.executableURL {
            executableURL = bundleExec.resolvingSymlinksInPath()
        } else {
            executableURL = URL(fileURLWithPath: cwd).appendingPathComponent(CommandLine.arguments[0])
        }
        let executableDir = executableURL.deletingLastPathComponent().standardized.path

        // Paths to check (in order of priority)
        let pathsToCheck = [
            // Bundled with executable (portable distribution)
            "\(executableDir)/Resources/webui/index.html.gz",
            // One level up from executable
            "\(executableDir)/../Resources/webui/index.html.gz",
            // Two levels up (e.g., .build/release -> .build -> project root)
            "\(executableDir)/../../Resources/webui/index.html.gz",
            // Three levels up for deeper nesting
            "\(executableDir)/../../../Resources/webui/index.html.gz",
            // pip: webui bundled in macafm package (sibling share directory)
            "\(executableDir)/../share/webui/index.html.gz",
            // Homebrew: share directory relative to bin (Apple Silicon)
            "\(executableDir)/../share/afm/webui/index.html.gz",
            // Homebrew: share directory relative to bin (Intel)
            "/usr/local/share/afm/webui/index.html.gz",
            // Homebrew: Apple Silicon path
            "/opt/homebrew/share/afm/webui/index.html.gz",
            // Development: Resources folder in current working directory
            "\(cwd)/Resources/webui/index.html.gz",
            // Development: vendored llama.cpp webui relative to executable
            "\(executableDir)/../../../vendor/llama.cpp/tools/server/public/index.html.gz",
            // Development: llama.cpp submodule public folder
            "\(cwd)/vendor/llama.cpp/tools/server/public/index.html.gz"
        ]

        for path in pathsToCheck {
            let standardizedPath = URL(fileURLWithPath: path).standardized.path
            if fileManager.fileExists(atPath: standardizedPath) {
                return standardizedPath
            }
        }

        return nil
    }

    private func browserLaunchHost(for hostname: String) -> String {
        switch hostname {
        case "0.0.0.0", "::", "[::]":
            return "127.0.0.1"
        default:
            return hostname
        }
    }

    /// Open URL in default browser
    @MainActor
    private func openBrowser(url: String) {
        guard let targetURL = URL(string: url) else { return }

        #if canImport(AppKit)
        if NSWorkspace.shared.open(targetURL) {
            return
        }
        #endif

        if runBrowserOpenProcess(executable: "/usr/bin/open", arguments: [targetURL.absoluteString]) {
            return
        }

        if runBrowserOpenProcess(executable: "/usr/bin/osascript", arguments: ["-e", "open location \"\(targetURL.absoluteString)\""]) {
            return
        }

        print("  ⚠️  Failed to open WebUI automatically. Open this URL manually: \(targetURL.absoluteString)")
    }

    @MainActor
    private func runBrowserOpenProcess(executable: String, arguments: [String]) -> Bool {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: executable)
        task.arguments = arguments
        task.standardOutput = nil
        task.standardError = nil
        do {
            try task.run()
            task.waitUntilExit()
            return task.terminationReason == .exit && task.terminationStatus == 0
        } catch {
            return false
        }
    }

    /// Decompress gzip data
    private static func gunzip(_ data: Data) throws -> Data {
        // Gzip has a header we need to skip (minimum 10 bytes)
        guard data.count > 10 else { throw GzipError.invalidData }

        // Check gzip magic number
        guard data[0] == 0x1f && data[1] == 0x8b else { throw GzipError.invalidData }

        // Skip gzip header (10 bytes minimum, more if there are extra fields)
        var headerSize = 10
        let flags = data[3]

        // Check for extra field (FEXTRA)
        if flags & 0x04 != 0 {
            guard data.count > headerSize + 2 else { throw GzipError.invalidData }
            let extraLen = Int(data[headerSize]) | (Int(data[headerSize + 1]) << 8)
            headerSize += 2 + extraLen
        }

        // Check for original filename (FNAME)
        if flags & 0x08 != 0 {
            while headerSize < data.count && data[headerSize] != 0 {
                headerSize += 1
            }
            headerSize += 1 // skip null terminator
        }

        // Check for comment (FCOMMENT)
        if flags & 0x10 != 0 {
            while headerSize < data.count && data[headerSize] != 0 {
                headerSize += 1
            }
            headerSize += 1 // skip null terminator
        }

        // Check for header CRC (FHCRC)
        if flags & 0x02 != 0 {
            headerSize += 2
        }

        guard headerSize < data.count - 8 else { throw GzipError.invalidData }

        // Extract compressed data (excluding 8-byte trailer: CRC32 + original size)
        let compressedData = data.subdata(in: headerSize..<(data.count - 8))

        // Decompress using zlib raw deflate
        let destinationBufferSize = 10 * 1024 * 1024 // 10MB max
        let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: destinationBufferSize)
        defer { destinationBuffer.deallocate() }

        let decompressedSize = compressedData.withUnsafeBytes { sourcePtr -> Int in
            guard let baseAddress = sourcePtr.baseAddress else { return 0 }
            return compression_decode_buffer(
                destinationBuffer,
                destinationBufferSize,
                baseAddress.assumingMemoryBound(to: UInt8.self),
                compressedData.count,
                nil,
                COMPRESSION_ZLIB
            )
        }

        guard decompressedSize > 0 else { throw GzipError.decompressionFailed }

        return Data(bytes: destinationBuffer, count: decompressedSize)
    }
}

enum GzipError: Error {
    case invalidData
    case decompressionFailed
}

struct ModelsResponse: Content {
    let object: String
    let data: [ModelInfo]
    let models: [ModelDetails]?
}

struct ModelDetails: Content {
    let name: String
    let model: String
    let capabilities: [String]?
}

struct ModelInfo: Content {
    let id: String
    let object: String
    let created: Int
    let owned_by: String
    let status: ModelStatus
    let max_context_length: Int?
    init(id: String, object: String, created: Int, owned_by: String, loaded: Bool = true, max_context_length: Int? = nil) {
        self.id = id
        self.object = object
        self.created = created
        self.owned_by = owned_by
        self.status = ModelStatus(value: loaded ? "loaded" : "unloaded")
        self.max_context_length = max_context_length
    }
}

struct ModelStatus: Content {
    let value: String
}

public struct HealthResponse: Content {
    public let status: String
    public let timestamp: Double
    public let version: String
    public init(status: String, timestamp: Double, version: String) {
        self.status = status; self.timestamp = timestamp; self.version = version
    }
}

struct AFMWebLaunchRequest: Content, Sendable {
    let backend: String
    let model: String?
    let values: [String: String]?
    let flags: [String]?
    let dryRun: Bool?

    func persistableProfile() -> AFMWebLaunchRequest {
        var persistedValues = values ?? [:]
        persistedValues.removeValue(forKey: "--telegram-bot-token")
        return AFMWebLaunchRequest(
            backend: backend,
            model: model,
            values: persistedValues,
            flags: flags,
            dryRun: false
        )
    }
}

// MARK: - Props Response (llama.cpp webui compatibility)

struct PropsResponse: Content {
    let default_generation_settings: DefaultGenerationSettings
    let total_slots: Int
    let model_path: String
    let role: String
    let modalities: Modalities
    let chat_template: String
    let bos_token: String
    let eos_token: String
    let build_info: String
    let default_model: String
}

struct DefaultGenerationSettings: Content {
    let n_ctx: Int
    let params: GenerationParams
}

struct GenerationParams: Content {
    let n_predict: Int
    let temperature: Double
    let top_k: Int
    let top_p: Double
    let min_p: Double
    let stream: Bool
    let max_tokens: Int
}

struct Modalities: Content {
    let vision: Bool
    let audio: Bool
}

// Compact log handler that prints "[INFO]" instead of Vapor's padded "[ INFO ]"
public struct CompactLogHandler: LogHandler {
    public var metadata: Logger.Metadata = [:]
    public var logLevel: Logger.Level = .info
    let label: String

    public init(label: String) {
        self.label = label
    }

    public subscript(metadataKey key: String) -> Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    private static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    public func log(level: Logger.Level, message: Logger.Message, metadata: Logger.Metadata?,
             source: String, file: String, function: String, line: UInt) {
        let ts = Self.timestampFormatter.string(from: Date())
        let levelStr = level.rawValue.uppercased()
        let metaStr = Self.formatMetadata(self.metadata, metadata)
        if metaStr.isEmpty {
            print("[\(ts)] [\(levelStr)] \(message)")
        } else {
            print("[\(ts)] [\(levelStr)] \(message) \(metaStr)")
        }
    }

    private static func formatMetadata(_ base: Logger.Metadata, _ extra: Logger.Metadata?) -> String {
        var merged = base
        if let extra { merged.merge(extra) { _, new in new } }
        guard !merged.isEmpty else { return "" }
        return merged.sorted(by: { $0.key < $1.key })
            .map { "[\($0.key): \($0.value)]" }
            .joined(separator: " ")
    }
}
