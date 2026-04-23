import ArgumentParser
import Foundation
import Logging
import Vapor

private var globalEmbeddingServer: EmbeddingHTTPServer?

private func handleEmbeddingShutdown(_ signal: Int32) {
    print("\n🛑 Received shutdown signal, shutting down embeddings server...")
    globalEmbeddingServer?.shutdown()
}

struct EmbeddingsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Serve OpenAI-compatible embeddings using Apple NaturalLanguage contextual embeddings"
    )

    @ArgumentParser.Option(name: [.customShort("m"), .long], help: "Embedding model id")
    var model: String = EmbeddingModelRegistry.defaultModelID

    @ArgumentParser.Option(name: .shortAndLong, help: "Port to run the embeddings server on")
    var port: Int = 9998

    @ArgumentParser.Option(name: [.customShort("H"), .long], help: "Hostname to bind server to")
    var hostname: String = "127.0.0.1"

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

        globalEmbeddingServer = server
        signal(SIGINT, handleEmbeddingShutdown)
        signal(SIGTERM, handleEmbeddingShutdown)

        try await server.start()
    }
}

final class EmbeddingHTTPServer {
    private let app: Application
    private let port: Int
    private let hostname: String
    private let modelEntry: EmbeddingModelEntry
    private let backend: any EmbeddingBackend

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
        app.middleware.use(PayloadTooLargeMiddleware())

        try routes()
    }

    func start() async throws {
        print("Starting embeddings server for \(modelEntry.id) on http://\(hostname):\(port)")
        try await app.server.start(address: .hostname(hostname, port: port))

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            app.storage[ContinuationKey.self] = continuation
        }
    }

    func shutdown() {
        Task {
            await app.server.shutdown()
            if let continuation = app.storage[ContinuationKey.self] {
                continuation.resume()
                app.storage[ContinuationKey.self] = nil
            }
        }
    }

    private func routes() throws {
        app.get("health") { _ async -> HealthResponse in
            HealthResponse(
                status: "healthy",
                timestamp: Date().timeIntervalSince1970,
                version: "1.0.0"
            )
        }

        try app.register(collection: EmbeddingsController(modelEntry: modelEntry, backend: backend))
    }
}
