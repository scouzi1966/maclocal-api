import AFMKit
import AFMKitCore
import AFMServer
import AFMTerminalUI
import ArgumentParser
import Darwin
import Foundation
import Synchronization

extension PCCConfiguration.Reasoning: ExpressibleByArgument {}

struct PCCCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pcc",
        abstract: "Use Apple Private Cloud Compute (macOS 27, signed PCC-enabled app required)",
        discussion: "Requests run on Apple's Private Cloud Compute using this Mac's availability and quota. No automatic on-device fallback. See docs/private-cloud-compute.md for development signing.",
        subcommands: [PCCStatusCommand.self, PCCRespondCommand.self, PCCChatCommand.self, PCCServeCommand.self],
        defaultSubcommand: PCCStatusCommand.self)
}

private struct PCCOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "Instructions for the assistant")
    var instructions: String = "You are a helpful assistant"

    @Option(name: .long, help: "Reasoning level: automatic, light, moderate, or deep")
    var reasoning: PCCConfiguration.Reasoning = .automatic

    var configuration: PCCConfiguration {
        .init(instructions: instructions, reasoning: reasoning)
    }
}

private enum PCCCLI {
    static let defaultPort = 9999
    static let runLoopInterval: TimeInterval = 0.1

    static func requireRuntime() throws {
        do { try PCCConfiguration.requireSupportedRuntime() }
        catch { throw ValidationError(error.localizedDescription) }
    }

    @available(macOS 27.0, *)
    static func requireAvailable() throws {
        let status = PCCStatus.current()
        guard status.available else {
            throw ValidationError("PCC unavailable (\(status.reason ?? "unknown")): \(status.detail ?? "Check afm pcc status.")\nUse the executable inside your PCC-signed AFM.app. See docs/private-cloud-compute.md.")
        }
    }

    // Keep the root ParsableCommand's synchronous lifecycle while async work runs
    // off the main thread, as in the existing CLI's single-prompt/TUI commands.
    static func wait(_ action: @escaping @Sendable () async throws -> Void) throws {
        let result = Mutex<(any Error)?>(nil)
        let group = DispatchGroup()
        group.enter()
        Task.detached {
            defer { group.leave() }
            do { try await action() } catch { result.withLock { $0 = error } }
        }
        group.wait()
        if let error = result.withLock({ $0 }) { throw error }
    }
}

private struct PCCStatusCommand: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "status", abstract: "Check PCC entitlement, availability, and quota without generating text")

    @Flag(name: .long, help: "Print machine-readable diagnostics")
    var json = false

    func run() throws {
        try PCCCLI.requireRuntime()
        if #available(macOS 27.0, *) {
            let status = PCCStatus.current()
            if json {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                print(String(decoding: try encoder.encode(status), as: UTF8.self))
            } else {
                print("Model: \(status.model)")
                print("PCC entitlement in signature: \(status.hasEntitlement ? "present" : "missing")")
                print("Available: \(status.available ? "yes" : "no")")
                print("Quota: \(status.quotaStatus)")
                if let detail = status.detail { print(detail) }
                if !status.hasEntitlement { print("Build/sign with Scripts/package-pcc-app.py. See docs/private-cloud-compute.md.") }
            }
            if !status.available { throw ExitCode.failure }
        }
    }
}

private struct PCCRespondCommand: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "respond", abstract: "Send a prompt to PCC; write the answer to stdout")
    @OptionGroup var options: PCCOptions
    @Argument(help: "Prompt, or '-' to read UTF-8 text from stdin") var prompt: String
    @Flag(name: .long, help: "Wait for the complete answer instead of streaming") var noStreaming = false

    func run() throws {
        try PCCCLI.requireRuntime()
        let text: String
        if prompt == "-" {
            guard let decoded = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) else {
                throw ValidationError("stdin must contain UTF-8 text")
            }
            text = decoded
        } else { text = prompt }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("Prompt must not be empty")
        }
        if #available(macOS 27.0, *) {
            try PCCCLI.requireAvailable()
            let engine = try options.configuration.makeEngine()
            let streaming = !noStreaming
            try PCCCLI.wait {
                do {
                    _ = try await engine.load()
                    let messages = [Message(role: "user", content: .text(text))]
                    if streaming {
                        for try await delta in engine.streamRespond(to: messages) {
                            FileHandle.standardOutput.write(Data(delta.utf8))
                        }
                        print("")
                    } else {
                        print(try await engine.respond(to: messages).content)
                    }
                    await engine.unload()
                } catch {
                    await engine.unload()
                    throw error
                }
            }
        }
    }
}

private struct PCCChatCommand: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "chat", abstract: "Open terminal chat using PCC")
    @OptionGroup var options: PCCOptions
    @Flag(name: .long, help: "Wait for complete responses") var noStreaming = false

    func run() throws {
        try PCCCLI.requireRuntime()
        try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false,
            inputIsTTY: isatty(STDIN_FILENO) == 1, outputIsTTY: isatty(STDOUT_FILENO) == 1)
        if #available(macOS 27.0, *) {
            try PCCCLI.requireAvailable()
            let engine = try options.configuration.makeEngine()
            try runTerminalChat(.init(
                backend: engine.backend,
                backendName: "Private Cloud Compute",
                modelName: "apple.private-cloud-compute",
                engine: .init(instructions: ""),
                streaming: !noStreaming), engine: engine)
        }
    }
}

private struct PCCServeCommand: ParsableCommand {
    static let configuration = CommandConfiguration(commandName: "serve", abstract: "Serve PCC through the OpenAI-compatible API")
    @OptionGroup var options: PCCOptions
    @Option(name: .shortAndLong, help: "Port to listen on") var port = PCCCLI.defaultPort
    @Option(name: [.customShort("H"), .long], help: "Address to bind") var hostname = "127.0.0.1"
    @Flag(name: .shortAndLong, help: "Enable verbose logging") var verbose = false
    @Flag(name: .long, help: "Disable streaming responses") var noStreaming = false

    func run() throws {
        try PCCCLI.requireRuntime()
        guard (1...Int(UInt16.max)).contains(port) else { throw ValidationError("Port must be between 1 and 65535") }
        if #available(macOS 27.0, *) {
            try PCCCLI.requireAvailable()
            let model = try options.configuration.makeModel()
            let failure = Mutex<(any Error)?>(nil)
            let port = port, hostname = hostname, verbose = verbose, streaming = !noStreaming
            signal(SIGINT, handleShutdown)
            signal(SIGTERM, handleShutdown)
            Task {
                do {
                    let descriptor = try await model.load(progress: nil)
                    let server = try await Server(
                        port: port, hostname: hostname, verbose: verbose,
                        streamingEnabled: streaming, instructions: "", prewarmEnabled: false,
                        mlxModelID: descriptor.modelID.rawValue, afmModel: model,
                        contextWindow: descriptor.contextWindow)
                    globalServer = server
                    try await server.start()
                } catch {
                    failure.withLock { $0 = error }
                    shouldKeepRunning = false
                }
            }
            let runLoop = RunLoop.current
            while shouldKeepRunning && runLoop.run(mode: .default, before: Date(timeIntervalSinceNow: PCCCLI.runLoopInterval)) {}
            try PCCCLI.wait { await model.unload() }
            if let error = failure.withLock({ $0 }) { throw error }
        }
    }
}
