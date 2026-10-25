import ArgumentParser

struct SplashAPICommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "splash-api",
        abstract: "Serve a local Splash package through AFM's API using the native AFMKit provider",
        discussion: "Requires macOS 26.4+, M3 or newer, and an existing Splash model package. No automatic model downloads. Text chat and streaming are supported; use afm splash for Splash's full native CLI feature set.")

    @Option(name: .shortAndLong, help: "Existing Splash package directory containing target/, draft/, and tokenizer/")
    var model: String

    @Option(name: .long, help: "Model name advertised by AFM; defaults to the package directory name")
    var modelID: String?

    @Option(name: .shortAndLong, help: "AFM API port")
    var port: Int = 9999

    @Option(name: .long, help: "AFM API bind address")
    var hostname: String = "127.0.0.1"

    @Option(name: .long, help: "Context token limit; omitted uses Splash's automatic memory budget")
    var maxContext: Int?

    @Option(name: .long, help: "Native engine memory budget in bytes; omitted uses Splash's automatic budget")
    var maxMemoryBytes: UInt64?

    @Flag(name: .shortAndLong)
    var verbose = false

    mutating func validate() throws {
        guard (1...65535).contains(port), maxContext == nil || maxContext! > 0,
              maxMemoryBytes == nil || maxMemoryBytes! > 0 else {
            throw ValidationError("Port must be 1...65535; context and memory limits must be positive")
        }
    }

    func run() throws { try runSplashAPI(self) }
}
