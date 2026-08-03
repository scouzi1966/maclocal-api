import Foundation
import Darwin

/// Error thrown while resolving/installing the MLX Metal shader library.
/// (Plain Swift error so AFMKit stays free of the ArgumentParser dependency.)
public struct MetalLibraryError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
    public var errorDescription: String? { message }
}

public enum MLXMetalLibrary {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var initialized = false
    private static let resourceBundleNames = [
        "MacLocalAPI_AFMKitMLX.bundle",
        "MacLocalAPI_AFMKit.bundle",
    ]

    /// Resolve the absolute path to this binary.
    ///
    /// `CommandLine.arguments[0]` is unreliable: when invoked via PATH it is just the
    /// basename ("afm"), so `URL(fileURLWithPath:)` resolves it relative to the current
    /// working directory instead of the actual binary location. Use `_NSGetExecutablePath`
    /// (which goes through the Mach-O loader) and fall back to `Bundle.main.executableURL`
    /// or argv[0] only if that fails.
    private static func resolveExecutableURL() -> URL {
        var size: UInt32 = 0
        _ = _NSGetExecutablePath(nil, &size)
        if size > 0 {
            var buffer = [CChar](repeating: 0, count: Int(size))
            if _NSGetExecutablePath(&buffer, &size) == 0 {
                let path = String(cString: buffer)
                if !path.isEmpty {
                    return URL(fileURLWithPath: path).resolvingSymlinksInPath()
                }
            }
        }
        if let bundleExec = Bundle.main.executableURL {
            return bundleExec.resolvingSymlinksInPath()
        }
        return URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    }

    private static func metallib(inBundleAt bundleURL: URL, fileManager: FileManager) -> URL? {
        let candidates = [
            bundleURL.appendingPathComponent("default.metallib"),
            bundleURL.appendingPathComponent("Contents/Resources/default.metallib"),
        ]
        for candidate in candidates where fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }

        if let bundle = Bundle(url: bundleURL),
           let resource = bundle.url(forResource: "default", withExtension: "metallib"),
           fileManager.fileExists(atPath: resource.path) {
            return resource
        }

        return nil
    }

    static func metallib(
        inResourceDirectory directory: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        for bundleName in resourceBundleNames {
            let bundleURL = directory.appendingPathComponent(bundleName)
            if let resource = metallib(inBundleAt: bundleURL, fileManager: fileManager) {
                return resource
            }
        }
        return nil
    }

    /// Find the metallib without using Bundle.module (which fatalError's when relocated).
    ///
    /// Search order:
    /// 1. `MACAFM_MLX_METALLIB` env var — explicit override
    /// 2. `default.metallib` next to the executable (pip wheel layout: bin/default.metallib)
    /// 3. AFMKitMLX SwiftPM resource bundle next to the executable
    /// 4. SPM Bundle.module (only if the bundle actually exists — never fatalError)
    private static func resolveMetallib() -> URL? {
        let fileManager = FileManager.default
        let executableURL = resolveExecutableURL()
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

        // 3. SPM bundle next to the binary.
        if let resource = metallib(
            inResourceDirectory: executableDir,
            fileManager: fileManager
        ) {
            return resource
        }

        // macOS app and XCTest bundles place resources beside Contents/MacOS.
        let executableResources = executableDir
            .deletingLastPathComponent()
            .appendingPathComponent("Resources")
        if let resource = metallib(
            inResourceDirectory: executableResources,
            fileManager: fileManager
        ) {
            return resource
        }

        // An app host keeps SwiftPM resource bundles in Contents/Resources, not
        // beside the executable in Contents/MacOS.
        if let appResources = Bundle.main.resourceURL {
            if let resource = metallib(
                inResourceDirectory: appResources,
                fileManager: fileManager
            ) {
                return resource
            }
        }

        // 3a. Walk up a few parent directories from the test runner/executable.
        //     SwiftPM test layouts often place the test binary deeper than the app bundle.
        var searchDir = executableDir
        for _ in 0..<5 {
            if let resource = metallib(
                inResourceDirectory: searchDir,
                fileManager: fileManager
            ) {
                return resource
            }
            searchDir.deleteLastPathComponent()
        }

        // 3aa. Current working directory and common SwiftPM build layouts.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let configurations = ["debug", "release"]
        let cwdCandidates = resourceBundleNames.flatMap { bundleName in
            configurations.flatMap { configuration in
                [
                    cwd.appendingPathComponent(".build/\(configuration)/\(bundleName)/default.metallib"),
                    cwd.appendingPathComponent(".build/arm64-apple-macosx/\(configuration)/\(bundleName)/default.metallib"),
                    cwd.appendingPathComponent(".build/out/Products/\(configuration.capitalized)/\(bundleName)/Contents/Resources/default.metallib"),
                ]
            } + [cwd.appendingPathComponent("\(bundleName)/default.metallib")]
        } + [cwd.appendingPathComponent("Sources/AFMKitMLX/Resources/default.metallib")]
        for candidate in cwdCandidates where fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }

        // 3b. Homebrew layout: binary in bin/, bundle in ../libexec/
        for bundleName in resourceBundleNames {
            let homebrew = executableDir
                .deletingLastPathComponent()
                .appendingPathComponent("libexec")
                .appendingPathComponent(bundleName)
                .appendingPathComponent("default.metallib")
            if fileManager.fileExists(atPath: homebrew.path) { return homebrew }
        }

        // 4. SPM Bundle.module — only if the bundle file physically exists.
        //    We probe the path before calling Bundle(path:) to avoid the auto-generated
        //    fatalError when the bundle can't be found (happens on any relocated binary).
        for bundleName in resourceBundleNames {
            let mainBundleURL = Bundle.main.bundleURL.appendingPathComponent(bundleName)
            if let resource = metallib(inBundleAt: mainBundleURL, fileManager: fileManager) {
                return resource
            }
        }

        return nil
    }

    public static func ensureAvailable(verbose: Bool) throws {
        try lock.withLock {
            if initialized {
                return
            }

            guard let source = resolveMetallib() else {
                throw MetalLibraryError(
                    "MLX metallib not found. Searched next to binary and in AFMKitMLX resource bundles. "
                    + "Set MACAFM_MLX_METALLIB=/path/to/default.metallib to override."
                )
            }

            let metalDir = source.deletingLastPathComponent().path
            // MLX resolves the default metallib relative to the process CWD, so this is
            // intentionally a one-time process-global change during startup/test bootstrap.
            guard FileManager.default.changeCurrentDirectoryPath(metalDir) else {
                throw MetalLibraryError("Failed to switch to metallib directory: \(metalDir)")
            }

            if verbose {
                print("Using MLX metallib: \(source.path)")
            }

            initialized = true
        }
    }
}
