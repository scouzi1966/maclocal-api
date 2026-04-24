import ArgumentParser
import Foundation
import Logging
import Vapor

struct EmbeddingsPayloadTooLargeMiddleware: AsyncMiddleware {
    func respond(to request: Request, chainingTo next: any AsyncResponder) async throws -> Response {
        do {
            return try await next.respond(to: request)
        } catch let abort as Abort where abort.status == .payloadTooLarge {
            let errorResponse = OpenAIError(
                message: "Embeddings request body exceeds the configured size limit.",
                type: "payload_too_large"
            )
            let response = Response(status: .payloadTooLarge)
            response.headers.add(name: .contentType, value: "application/json")
            response.headers.add(name: .accessControlAllowOrigin, value: "*")
            try response.content.encode(errorResponse)
            return response
        }
    }
}

private struct EmbeddingShutdownRequestedKey: StorageKey {
    typealias Value = Bool
}

/// Installs async-signal-safe SIGINT/SIGTERM observers that drive `server.shutdown()`
/// from a normal dispatch queue. The kernel signal delivery is captured by
/// `DispatchSourceSignal`; the shutdown work runs on a Swift queue, so `print()`
/// and reference-counted Swift state are safe to touch.
///
/// Returns an opaque handle that keeps the sources alive for the lifetime of
/// the command run.
@discardableResult
private func installEmbeddingShutdownHandlers(for server: EmbeddingHTTPServer) -> [DispatchSourceSignal] {
    let queue = DispatchQueue(label: "afm.embeddings.shutdown")
    let sources = [SIGINT, SIGTERM].map { signo -> DispatchSourceSignal in
        signal(signo, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: signo, queue: queue)
        source.setEventHandler { [weak server] in
            print("\n🛑 Received shutdown signal, shutting down embeddings server...")
            server?.shutdown()
        }
        source.resume()
        return source
    }
    return sources
}

struct EmbeddingsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Serve OpenAI-compatible embeddings using Apple NaturalLanguage contextual embeddings"
    )

    private static let defaultPort = 9998
    private static let defaultHostname = "127.0.0.1"

    @ArgumentParser.Option(name: [.customShort("m"), .long], help: "Embedding model id")
    var model: String = EmbeddingModelRegistry.defaultModelID

    @ArgumentParser.Option(name: .shortAndLong, help: "Port to run the embeddings server on")
    var port: Int = EmbeddingsCommand.defaultPort

    @ArgumentParser.Option(name: [.customShort("H"), .long], help: "Hostname to bind server to")
    var hostname: String = EmbeddingsCommand.defaultHostname

    @ArgumentParser.Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false

    @ArgumentParser.Flag(name: [.customShort("V"), .long], help: "Enable very verbose logging")
    var veryVerbose: Bool = false

    @ArgumentParser.Flag(name: .long, help: "List available embedding models and exit")
    var listModels: Bool = false

    func run() async throws {
        let registry = EmbeddingModelRegistry()

        if listModels {
            for modelID in registry.listModelIDs() {
                print(modelID)
            }
            return
        }

        guard let requestedEntry = registry.resolve(modelID: model) else {
            throw ValidationError("Unknown embedding model: \(model)")
        }

        let nlBackend = try NLContextualEmbeddingBackend(modelID: requestedEntry.id)
        try await nlBackend.prepare()
        guard let resolvedEntry = registry.makeResolvedAppleEntry(
            modelID: requestedEntry.id,
            nativeDimension: nlBackend.nativeDimension,
            maxInputTokens: nlBackend.maxInputTokens
        ) else {
            throw ValidationError("Failed to resolve Apple embedding metadata for \(requestedEntry.id)")
        }

        let server = try await EmbeddingHTTPServer(
            port: port,
            hostname: hostname,
            verbose: verbose,
            veryVerbose: veryVerbose,
            modelEntry: resolvedEntry,
            backend: nlBackend
        )

        let shutdownSources = installEmbeddingShutdownHandlers(for: server)
        defer { shutdownSources.forEach { $0.cancel() } }

        try await server.start()
    }
}

final class EmbeddingHTTPServer {
    private let app: Application
    private let port: Int
    private let hostname: String
    private let modelEntry: EmbeddingModelEntry
    private let backend: any EmbeddingBackend
    private let shutdownLock = NSLock()

    init(
        port: Int,
        hostname: String,
        verbose: Bool,
        veryVerbose: Bool,
        modelEntry: EmbeddingModelEntry,
        backend: any EmbeddingBackend
    ) async throws {
        self.port = port
        self.hostname = hostname
        self.modelEntry = modelEntry
        self.backend = backend

        let env = Environment(name: "development", arguments: ["afm", "embed"])
        LoggingSystem.bootstrap { label in
            CompactLogHandler(label: label)
        }

        self.app = try await Application.make(env)
        if veryVerbose {
            app.logger.logLevel = .trace
        } else if verbose {
            app.logger.logLevel = .debug
        }

        app.http.server.configuration.port = port
        app.http.server.configuration.hostname = hostname
        app.middleware.use(EmbeddingsPayloadTooLargeMiddleware())

        try routes()
    }

    func start() async throws {
        // A signal can arrive between installing the signal handlers and the
        // server finishing bind. If shutdown() ran against an un-started
        // server, its app.server.shutdown() was a no-op, so we need to either
        // skip the bind entirely or tear it down right after it completes.
        shutdownLock.lock()
        let shutdownRequestedBeforeStart = app.storage[EmbeddingShutdownRequestedKey.self] == true
        shutdownLock.unlock()

        if shutdownRequestedBeforeStart {
            print("Embeddings server shutdown requested before bind; skipping start.")
            return
        }

        print("Starting embeddings server for \(modelEntry.id) on http://\(hostname):\(port)")
        try await app.server.start(address: .hostname(hostname, port: port))

        // Race window: a shutdown signal delivered between handler install and
        // this point already marked the flag. The prior shutdown() call ran
        // app.server.shutdown() on an un-started server (no-op), so tear down
        // the freshly-bound listener here.
        shutdownLock.lock()
        let shutdownRequestedDuringStart = app.storage[EmbeddingShutdownRequestedKey.self] == true
        shutdownLock.unlock()
        if shutdownRequestedDuringStart {
            print("Shutdown requested during start; shutting the just-bound server down.")
            await app.server.shutdown()
            return
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            shutdownLock.lock()
            let alreadyRequested = app.storage[EmbeddingShutdownRequestedKey.self] == true
            if !alreadyRequested {
                app.storage[ContinuationKey.self] = continuation
            }
            shutdownLock.unlock()

            if alreadyRequested {
                continuation.resume()
            }
        }
    }

    func shutdown() {
        shutdownLock.lock()
        if app.storage[EmbeddingShutdownRequestedKey.self] == true {
            shutdownLock.unlock()
            return
        }
        app.storage[EmbeddingShutdownRequestedKey.self] = true
        let continuation = app.storage[ContinuationKey.self]
        app.storage[ContinuationKey.self] = nil
        shutdownLock.unlock()

        Task {
            await app.server.shutdown()
            // If start() hadn't registered its continuation yet, its re-entry
            // path will resume itself after seeing the flag was set.
            continuation?.resume()
        }
    }

    private func routes() throws {
        app.get("health") { _ async -> HealthResponse in
            HealthResponse(
                status: "healthy",
                timestamp: Date().timeIntervalSince1970,
                version: BuildInfo.fullVersion
            )
        }

        try app.register(collection: EmbeddingsController(modelEntry: modelEntry, backend: backend))
    }
}
