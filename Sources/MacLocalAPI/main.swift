import ArgumentParser
import Foundation

// Global references for signal handling
private var globalServer: Server?
private var shouldKeepRunning = true

// Signal handler function
func handleShutdown(_ signal: Int32) {
    print("\n🛑 Received shutdown signal, shutting down...")
    globalServer?.shutdown()
    shouldKeepRunning = false
}

struct MacLocalAPI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "MacLocalAPI",
        abstract: "macOS server that exposes Apple's Foundation Models through OpenAI-compatible API",
        version: "1.0.0"
    )
    
    @Option(name: .shortAndLong, help: "Port to run the server on")
    var port: Int = 9999
    
    @Flag(name: .shortAndLong, help: "Enable verbose logging")
    var verbose: Bool = false
    
    @Flag(name: .long, help: "Disable streaming responses (streaming is enabled by default)")
    var noStreaming: Bool = false
    
    @Option(name: [.short, .long], help: "Custom instructions for the AI assistant")
    var instructions: String = "You are a helpful assistant"
    
    func run() throws {
        if verbose {
            print("Starting MacLocalAPI server with verbose logging enabled...")
        }
        
        // Use RunLoop to handle the server lifecycle properly
        let runLoop = RunLoop.current
        
        // Set up signal handling for graceful shutdown
        signal(SIGINT, handleShutdown)
        signal(SIGTERM, handleShutdown)
        
        // Start server in async context
        _ = Task {
            do {
                let server = try await Server(port: port, verbose: verbose, streamingEnabled: !noStreaming, instructions: instructions)
                globalServer = server
                try await server.start()
            } catch {
                print("Error starting server: \(error)")
                shouldKeepRunning = false
            }
        }
        
        // Keep the main thread alive until shutdown
        while shouldKeepRunning && runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1)) {
            // Keep running until shutdown signal
        }
        
        print("Server shutdown complete.")
    }
}

MacLocalAPI.main()