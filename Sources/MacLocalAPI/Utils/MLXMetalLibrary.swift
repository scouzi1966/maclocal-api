import Foundation
import ArgumentParser

enum MLXMetalLibrary {
    private static let lock = NSLock()
    private static var initialized = false

    static func ensureAvailable(verbose: Bool) throws {
        try lock.withLock {
            if initialized {
                return
            }

            let fileManager = FileManager.default
            let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            let executableDir = executableURL.deletingLastPathComponent()

            let env = ProcessInfo.processInfo.environment
            let explicit = env["MACAFM_MLX_METALLIB"].flatMap { raw -> URL? in
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : URL(fileURLWithPath: trimmed)
            }
            let packaged = Bundle.module.url(forResource: "default", withExtension: "metallib")
            let bundleRelative: URL? = {
                let candidate = executableDir
                    .appendingPathComponent("MacLocalAPI_MacLocalAPI.bundle")
                    .appendingPathComponent("default.metallib")
                return fileManager.fileExists(atPath: candidate.path) ? candidate : nil
            }()

            guard let source = explicit ?? packaged ?? bundleRelative,
                  fileManager.fileExists(atPath: source.path) else {
                throw ValidationError(
                    "MLX metallib not found. Ensure MacLocalAPI_MacLocalAPI.bundle is next to the afm binary."
                )
            }

            let metalDir = source.deletingLastPathComponent().path
            // MLX resolves the default metallib relative to the process CWD, so this is
            // intentionally a one-time process-global change during startup/test bootstrap.
            guard fileManager.changeCurrentDirectoryPath(metalDir) else {
                throw ValidationError("Failed to switch to metallib directory: \(metalDir)")
            }

            if verbose {
                print("Using MLX metallib: \(source.path)")
            }

            initialized = true
        }
    }
}
