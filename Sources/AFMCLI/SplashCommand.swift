import AFMExternalCLI
import AFMKitSplash
import ArgumentParser
import Foundation

/// Splash retains its own parser, server, and model lifecycle. This adapter only
/// connects its existing CLI to AFM; it is not an AFMKit model provider.
struct SplashCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "splash",
        abstract: "Run the bundled, release-pinned Splash CLI",
        discussion: """
        Run: afm splash serve --model incoai/Qwen3.8-27B-Splash

        All arguments after 'afm splash' go unchanged to Splash, including --help
        and --version. Set AFM_SPLASH_EXECUTABLE to explicitly use an unpinned
        custom installation. Use afm splash-api for AFM's HTTP API with the
        native AFMKit provider.
        Splash's launcher/server uses Python and its inference engine uses native
        C++/Metal. AFM does not start its own HTTP server for this command.
        """)

    @Argument(parsing: .unconditionalRemaining)
    var arguments: [String] = []

    static func launch(_ arguments: [String]) throws -> Never {
        try AFMSplashRuntime.checkPlatform()
        let cli = ExternalCLI(
            name: "splash",
            executableOverride: "AFM_SPLASH_EXECUTABLE",
            installationHint: "Rebuild or reinstall AFM with its bundled Splash runtime.")
        if ProcessInfo.processInfo.environment["AFM_SPLASH_EXECUTABLE"] != nil {
            try cli.invocation(arguments: arguments).execute()
        }
        let runtime = try AFMSplashRuntime.bundled()
        try runtime.validate()
        var environment = ProcessInfo.processInfo.environment
        environment["AFM_SPLASH_EXECUTABLE"] = runtime.pythonExecutable.path
        try cli.invocation(arguments: ["-B", runtime.launcher.path] + arguments, environment: environment).execute()
    }

    func run() throws { try Self.launch(arguments) }
}
