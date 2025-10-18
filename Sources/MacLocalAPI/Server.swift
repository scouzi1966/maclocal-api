import Vapor
import Foundation

// Storage key for the continuation
struct ContinuationKey: StorageKey {
    typealias Value = CheckedContinuation<Void, Error>
}

class Server {
    private let app: Application
    private let port: Int
    private let hostname: String
    private let verbose: Bool
    private let streamingEnabled: Bool
    private let instructions: String
    private let adapter: String?
    private let temperature: Double?
    private let randomness: String?
    private let permissiveGuardrails: Bool
    
    init(port: Int, hostname: String, verbose: Bool, streamingEnabled: Bool, instructions: String, adapter: String? = nil, temperature: Double? = nil, randomness: String? = nil, permissiveGuardrails: Bool = false) async throws {
        self.port = port
        self.hostname = hostname
        self.verbose = verbose
        self.streamingEnabled = streamingEnabled
        self.instructions = instructions
        self.adapter = adapter
        self.temperature = temperature
        self.randomness = randomness
        self.permissiveGuardrails = permissiveGuardrails
        
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
        app.http.server.configuration.hostname = hostname
        
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
        
        let chatController = ChatCompletionsController(streamingEnabled: streamingEnabled, instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
        try app.register(collection: chatController)
    }
    
    func start() async throws {
        // Print ASCII art splash screen
        let version = BuildInfo.version ?? "dev-build"

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

        // Initialize the Foundation Model Service once at startup
        if #available(macOS 26.0, *) {
            try await FoundationModelService.initialize(instructions: instructions, adapter: adapter, temperature: temperature, randomness: randomness, permissiveGuardrails: permissiveGuardrails)
        }

        print("  🚀 Server: http://\(hostname):\(port)")
        print("")
        print("  📡 Endpoints:")
        print("     • POST   /v1/chat/completions    - Chat completion (streaming supported)")
        print("     • GET    /v1/models              - List available models")
        print("     • GET    /health                 - Health check")
        print("")
        print("  ⚙️  Configuration:")
        print("     • Streaming:          \(streamingEnabled ? "✓ enabled" : "✗ disabled")")
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
        print("")
        print("  ℹ️  Requires macOS 26+ with Apple Intelligence")
        print("  💡 Press Ctrl+C to stop the server")
        print("")
        print("  ─────────────────────────────────────────────────────────────────────────")
        print("")
        
        // Start the server
        try await app.server.start(address: .hostname(hostname, port: port))
        
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
