import Darwin
import Foundation

/// Application-owned process integration. This module has no inference-provider,
/// HTTP, ArgumentParser, or Python dependencies.
public struct ExternalCLI: Sendable {
    public let name: String
    public let executableOverride: String
    public let installationHint: String

    public init(name: String, executableOverride: String, installationHint: String) {
        self.name = name
        self.executableOverride = executableOverride
        self.installationHint = installationHint
    }

    public struct Invocation: Equatable, Sendable {
        public let executable: String
        public let arguments: [String]

        /// Replace AFM so the original process's terminal, working directory,
        /// environment, exit status, and signal handling belong to the external CLI.
        public func execute() throws -> Never {
            let strings = [executable] + arguments
            guard strings.allSatisfy({ !$0.utf8.contains(0) }) else {
                throw LaunchError.invalidArgument
            }
            var argv = [UnsafeMutablePointer<CChar>?]()
            defer { argv.forEach { free($0) } }
            for string in strings {
                guard let copy = strdup(string) else { throw LaunchError.allocationFailed }
                argv.append(copy)
            }
            argv.append(nil)
            let result = argv.withUnsafeMutableBufferPointer { pointers in
                execv(executable, pointers.baseAddress!)
            }
            let code = errno
            throw LaunchError.executionFailed(path: executable, code: result == -1 ? code : EIO)
        }
    }

    public enum LaunchError: Error, LocalizedError {
        case missing(name: String, hint: String, override: String)
        case invalidExecutable(String)
        case recursion(String)
        case invalidArgument
        case allocationFailed
        case executionFailed(path: String, code: Int32)

        public var errorDescription: String? {
            switch self {
            case .missing(let name, let hint, let override):
                return "\(name) is not installed or is not on PATH. \(hint) Set \(override) to an explicit executable path for a custom installation."
            case .invalidExecutable(let path): return "External CLI is not an executable file: \(path)"
            case .recursion(let path): return "External CLI resolves to AFM itself: \(path)"
            case .invalidArgument: return "External CLI arguments cannot contain NUL bytes."
            case .allocationFailed: return "Unable to allocate external CLI arguments."
            case .executionFailed(let path, let code): return "Could not execute \(path): \(String(cString: strerror(code)))"
            }
        }
    }

    public func invocation(
        arguments: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment,
        workingDirectory: String = FileManager.default.currentDirectoryPath,
        currentExecutable: String = Bundle.main.executablePath ?? CommandLine.arguments[0]
    ) throws -> Invocation {
        guard arguments.allSatisfy({ !$0.utf8.contains(0) }) else { throw LaunchError.invalidArgument }
        let fileManager = FileManager.default
        func absolute(_ path: String) -> String {
            URL(fileURLWithPath: path, relativeTo: URL(fileURLWithPath: workingDirectory, isDirectory: true))
                .standardizedFileURL.path
        }
        func executable(_ path: String) -> Bool {
            var directory: ObjCBool = false
            return fileManager.fileExists(atPath: path, isDirectory: &directory)
                && !directory.boolValue && fileManager.isExecutableFile(atPath: path)
        }
        let path: String
        if let override = environment[executableOverride] {
            guard !override.isEmpty, !override.utf8.contains(0) else { throw LaunchError.invalidExecutable(override) }
            path = absolute(override)
            guard executable(path) else { throw LaunchError.invalidExecutable(path) }
        } else {
            let candidates = (environment["PATH"] ?? "").split(separator: ":", omittingEmptySubsequences: false)
                .map { absolute($0.isEmpty ? name : String($0) + "/" + name) }
            guard let found = candidates.first(where: executable) else {
                throw LaunchError.missing(name: name, hint: installationHint, override: executableOverride)
            }
            path = found
        }
        let canonical = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        let ownCanonical = URL(fileURLWithPath: absolute(currentExecutable)).resolvingSymlinksInPath().path
        let attributes = try fileManager.attributesOfItem(atPath: canonical)
        let ownAttributes = try? fileManager.attributesOfItem(atPath: ownCanonical)
        let sameFile = attributes[.systemNumber] as? NSNumber == ownAttributes?[.systemNumber] as? NSNumber
            && attributes[.systemFileNumber] as? NSNumber == ownAttributes?[.systemFileNumber] as? NSNumber
        guard canonical != ownCanonical && !sameFile else { throw LaunchError.recursion(path) }
        return Invocation(executable: path, arguments: arguments)
    }
}
