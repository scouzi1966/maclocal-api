import Vapor
import Foundation

// Storage key for the continuation
struct ContinuationKey: StorageKey {
    typealias Value = CheckedContinuation<Void, Error>
}

class Server {
    private let app: Application
    private let port: Int
    private let verbose: Bool
    private let streamingEnabled: Bool
    private let instructions: String
    private let adapter: String?
    
    init(port: Int, verbose: Bool, streamingEnabled: Bool, instructions: String, adapter: String? = nil) async throws {
        self.port = port
        self.verbose = verbose
        self.streamingEnabled = streamingEnabled
        self.instructions = instructions
        self.adapter = adapter
        
        // Create environment without command line arguments to prevent Vapor from parsing them
        var env = Environment(name: "development", arguments: ["afm"])
        try LoggingSystem.bootstrap(from: &env)
        
        self.app = try await Application.make(env)
        
        if verbose {
            app.logger.logLevel = .debug
        }
        
        try configure()
    }
    
    private func configure() throws {
        app.http.server.configuration.port = port
        app.http.server.configuration.hostname = "127.0.0.1"
        
        try routes()
    }
    
    private func routes() throws {
        app.get("health") { req async -> HealthResponse in
            return HealthResponse(
                status: "healthy",
                timestamp: Date().timeIntervalSince1970,
                version: "1.0.0"
            )
        }
        
        app.get("v1", "models") { req in
            return ModelsResponse(
                object: "list",
                data: [
                    ModelInfo(
                        id: "foundation",
                        object: "model",
                        created: Int(Date().timeIntervalSince1970),
                        ownedBy: "apple"
                    )
                ]
            )
        }
        
        let chatController = ChatCompletionsController(streamingEnabled: streamingEnabled, instructions: instructions, adapter: adapter)
        try app.register(collection: chatController)
    }
    
    func start() async throws {
        print("🚀 afm server starting on http://localhost:\(port)")
        print("📱 Using Apple Foundation Models (requires macOS 26+ with Apple Intelligence)")
        
        // Initialize the Foundation Model Service once at startup
        if #available(macOS 26.0, *) {
            try await FoundationModelService.initialize(instructions: instructions, adapter: adapter)
        }
        
        print("🔗 OpenAI API compatible endpoints:")
        print("   POST http://localhost:\(port)/v1/chat/completions")
        print("   GET  http://localhost:\(port)/v1/models")
        print("   GET  http://localhost:\(port)/health")
        print("Press Ctrl+C to stop the server")
        
        // Start the server
        try await app.server.start(address: .hostname("127.0.0.1", port: port))
        
        // Wait indefinitely (until shutdown is called)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Store continuation for later use in shutdown
            app.storage[ContinuationKey.self] = continuation
        }
    }
    
    func shutdown() {
        print("🛑 Shutting down server...")
        
        // Shutdown the server first
        Task {
            await app.server.shutdown()
            print("Server shutdown complete")
            
            // Resume the continuation to exit the wait
            if let continuation = app.storage[ContinuationKey.self] {
                continuation.resume()
                app.storage[ContinuationKey.self] = nil
            }
        }
    }
}

struct ModelsResponse: Content {
    let object: String
    let data: [ModelInfo]
}

struct ModelInfo: Content {
    let id: String
    let object: String
    let created: Int
    let ownedBy: String
}

struct HealthResponse: Content {
    let status: String
    let timestamp: Double
    let version: String
}