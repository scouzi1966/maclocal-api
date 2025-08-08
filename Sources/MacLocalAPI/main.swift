import ArgumentParser
import Foundation

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
    
    func run() throws {
        if verbose {
            print("Starting MacLocalAPI server with verbose logging enabled...")
        }
        
        let server = try Server(port: port, verbose: verbose)
        
        if #available(macOS 10.15, *) {
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                do {
                    try await server.start()
                } catch {
                    print("Error starting server: \(error)")
                }
                semaphore.signal()
            }
            semaphore.wait()
        } else {
            print("This application requires macOS 10.15 or later")
            throw ExitCode.failure
        }
    }
}

MacLocalAPI.main()