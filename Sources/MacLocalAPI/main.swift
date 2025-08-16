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
        commandName: "afm",
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
    
    @Option(name: [.customShort("s"), .long], help: "Run a single prompt without starting the server")
    var singlePrompt: String?
    
    func run() throws {
        // Handle single-prompt mode
        if let prompt = singlePrompt {
            return try runSinglePrompt(prompt)
        }
        
        if verbose {
            print("Starting afm server with verbose logging enabled...")
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
    
    private func runSinglePrompt(_ prompt: String) throws {
        let group = DispatchGroup()
        var result: Result<String, Error>?
        
        group.enter()
        Task {
            do {
                if #available(macOS 26.0, *) {
                    let foundationService = try await FoundationModelService(instructions: instructions)
                    let message = Message(role: "user", content: prompt)
                    let response = try await foundationService.generateResponse(for: [message])
                    result = .success(response)
                } else {
                    result = .failure(FoundationModelError.notAvailable)
                }
            } catch {
                result = .failure(error)
            }
            group.leave()
        }
        
        group.wait()
        
        switch result {
        case .success(let response):
            print(response)
        case .failure(let error):
            if let foundationError = error as? FoundationModelError {
                print("Error: \(foundationError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
            throw ExitCode.failure
        case .none:
            print("Error: Unexpected error occurred")
            throw ExitCode.failure
        }
    }
}

MacLocalAPI.main()