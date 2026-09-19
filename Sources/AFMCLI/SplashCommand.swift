import AFMExternalCLI
import ArgumentParser

/// Splash retains its own parser, server, and model lifecycle. This adapter only
/// connects its existing CLI to AFM; it is not an AFMKit model provider.
struct SplashCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "splash",
        abstract: "Run Splash's own CLI (requires a separate Splash installation)",
        discussion: """
        Install Splash with: brew install incoai/tap/splash
        Then run: afm splash serve --model incoai/Qwen3.8-27B-Splash

        All arguments after 'afm splash' go unchanged to Splash, including --help
        and --version. Set AFM_SPLASH_EXECUTABLE to select a custom installation.
        Splash's launcher/server uses Python and its inference engine uses native
        C++/Metal. AFM does not start its own HTTP server for this command.
        """)

    @Argument(parsing: .unconditionalRemaining)
    var arguments: [String] = []

    static func launch(_ arguments: [String]) throws -> Never {
        try ExternalCLI(
            name: "splash",
            executableOverride: "AFM_SPLASH_EXECUTABLE",
            installationHint: "Install it with: brew install incoai/tap/splash.")
            .invocation(arguments: arguments).execute()
    }

    func run() throws { try Self.launch(arguments) }
}
