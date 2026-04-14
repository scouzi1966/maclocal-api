import Foundation
import ArgumentParser

enum MLXMetalLibrary {
    private static let lock = NSLock()
    private static var initialized = false

    /// Find the metallib without using Bundle.module (which fatalError's when relocated).
    ///
    /// Search order:
    /// 1. `MACAFM_MLX_METALLIB` env var — explicit override
    /// 2. `default.metallib` next to the executable (pip wheel layout: bin/default.metallib)
    /// 3. `MacLocalAPI_MacLocalAPI.bundle/default.metallib` next to the executable (SPM build layout)
    /// 4. SPM Bundle.module (only if the bundle actually exists — never fatalError)
    private static func resolveMetallib() -> URL? {
        let fileManager = FileManager.default
        let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let executableDir = executableURL.deletingLastPathComponent()

        // 1. Explicit env var override
        let env = ProcessInfo.processInfo.environment
        if let raw = env["MACAFM_MLX_METALLIB"] {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let url = URL(fileURLWithPath: trimmed)
                if fileManager.fileExists(atPath: url.path) { return url }
            }
        }

        // 2. Loose metallib next to the binary (pip wheel: macafm_next/bin/default.metallib)
        let loose = executableDir.appendingPathComponent("default.metallib")
        if fileManager.fileExists(atPath: loose.path) { return loose }

        // 3. SPM bundle next to the binary (direct build: .build/release/MacLocalAPI_MacLocalAPI.bundle/)
        let bundled = executableDir
            .appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle")
            .appendingPathComponent("default.metallib")
        if fileManager.fileExists(atPath: bundled.path) { return bundled }

        // 3a. Walk up a few parent directories from the test runner/executable.
        //     SwiftPM test layouts often place the test binary deeper than the app bundle.
        var searchDir = executableDir
        for _ in 0..<5 {
            let candidate = searchDir
                .appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle")
                .appendingPathComponent("default.metallib")
            if fileManager.fileExists(atPath: candidate.path) { return candidate }
            searchDir.deleteLastPathComponent()
        }

        // 3aa. Current working directory and common SwiftPM build layouts.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let cwdCandidates = [
            cwd.appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle/default.metallib"),
            cwd.appendingPathComponent(".build/debug/MacLocalAPI_MacLocalAPI.bundle/default.metallib"),
            cwd.appendingPathComponent(".build/arm64-apple-macosx/debug/MacLocalAPI_MacLocalAPI.bundle/default.metallib"),
        ]
        for candidate in cwdCandidates where fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }

        // 3b. Homebrew layout: binary in bin/, bundle in ../libexec/
        let homebrew = executableDir
            .deletingLastPathComponent()
            .appendingPathComponent("libexec")
            .appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle")
            .appendingPathComponent("default.metallib")
        if fileManager.fileExists(atPath: homebrew.path) { return homebrew }

        // 4. SPM Bundle.module — only if the bundle file physically exists.
        //    We probe the path before calling Bundle(path:) to avoid the auto-generated
        //    fatalError when the bundle can't be found (happens on any relocated binary).
        let mainBundlePath = Bundle.main.bundleURL
            .appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle").path
        if let b = Bundle(path: mainBundlePath),
           let url = b.url(forResource: "default", withExtension: "metallib"),
           fileManager.fileExists(atPath: url.path) {
            return url
        }

        return nil
    }

    static func ensureAvailable(verbose: Bool) throws {
        try lock.withLock {
            if initialized {
                return
            }

            guard let source = resolveMetallib() else {
                throw ValidationError(
                    "MLX metallib not found. Searched next to binary and in MacLocalAPI_MacLocalAPI.bundle. "
                    + "Set MACAFM_MLX_METALLIB=/path/to/default.metallib to override."
                )
            }

            let metalDir = source.deletingLastPathComponent().path
            // MLX resolves the default metallib relative to the process CWD, so this is
            // intentionally a one-time process-global change during startup/test bootstrap.
            guard FileManager.default.changeCurrentDirectoryPath(metalDir) else {
                throw ValidationError("Failed to switch to metallib directory: \(metalDir)")
            }

            if verbose {
                print("Using MLX metallib: \(source.path)")
            }

            initialized = true
        }
    }
}
