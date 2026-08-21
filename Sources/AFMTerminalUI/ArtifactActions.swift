import Darwin
import Foundation

public enum TUIArtifactError: Error, LocalizedError, Equatable {
    case exists(String), invalidPath(String), unsupportedBlock(String), commandFailed(String)
    case unsafeReadableFile(String)
    case incompatibleSession(savedBackend: String, savedModel: String, currentBackend: String, currentModel: String)

    public var errorDescription: String? {
        switch self {
        case .exists(let path): return "Refusing to overwrite existing file: \(path). Add ! to the command to confirm."
        case .invalidPath(let path): return "Invalid output path: \(path)"
        case .unsupportedBlock(let language): return "Only HTML and JavaScript blocks can be opened in a browser (got \(language.isEmpty ? "untyped code" : language))."
        case .commandFailed(let reason): return reason
        case .unsafeReadableFile(let path): return "Refusing to read a non-regular, linked, unreadable, or oversized local artifact: \(path)"
        case .incompatibleSession(let savedBackend, let savedModel, let currentBackend, let currentModel):
            return "Saved session uses \(savedBackend) · \(savedModel), but this TUI uses \(currentBackend) · \(currentModel)."
        }
    }
}

public enum TUIArtifactActions {
    public static func resolvedURL(_ rawPath: String, cwd: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)) throws -> URL {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.contains("\0") else { throw TUIArtifactError.invalidPath(rawPath) }
        let expanded = NSString(string: trimmed).expandingTildeInPath
        let url = expanded.hasPrefix("/") ? URL(fileURLWithPath: expanded) : cwd.appendingPathComponent(expanded)
        return url.standardizedFileURL
    }

    public static func save(_ data: Data, to url: URL, overwrite: Bool = false) throws {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        var info = stat()
        if lstat(url.path, &info) == 0 {
            if !overwrite { throw TUIArtifactError.exists(url.path) }
            guard (info.st_mode & S_IFMT) == S_IFREG else { throw TUIArtifactError.invalidPath(url.path) }
        }
        let flags = O_WRONLY | O_CREAT | (overwrite ? O_TRUNC : O_EXCL) | O_NOFOLLOW
        let descriptor = open(url.path, flags, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            if errno == EEXIST { throw TUIArtifactError.exists(url.path) }
            throw TUIArtifactError.invalidPath(url.path)
        }
        defer { close(descriptor) }
        try data.withUnsafeBytes { bytes in
            guard let start = bytes.baseAddress else { return }
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(descriptor, start.advanced(by: offset), bytes.count - offset)
                guard count > 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                offset += count
            }
        }
    }

    public static func copyToClipboard(_ text: String) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pbcopy")
        let pipe = Pipe()
        process.standardInput = pipe
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        pipe.fileHandleForWriting.write(Data(text.utf8))
        try pipe.fileHandleForWriting.close()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw TUIArtifactError.commandFailed("pbcopy failed") }
    }

    public static func prepareBrowserArtifact(_ block: TUICodeBlock, temporaryRoot: URL = FileManager.default.temporaryDirectory) throws -> URL {
        let language = block.language.lowercased()
        guard ["html", "htm", "javascript", "js"].contains(language) else {
            throw TUIArtifactError.unsupportedBlock(block.language)
        }
        let directory = temporaryRoot.appendingPathComponent("afm-tui-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        let url = directory.appendingPathComponent("artifact.html")
        let contentSecurityPolicy = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; media-src 'none'; object-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'"
        let containerSecurityPolicy = contentSecurityPolicy.replacingOccurrences(
            of: "frame-src 'none'",
            with: "frame-src 'self'"
        )
        let html: String
        if ["html", "htm"].contains(language) {
            let document = """
            <!doctype html><meta charset="utf-8">
            <meta http-equiv="Content-Security-Policy" content="\(contentSecurityPolicy)">
            \(block.content)
            """
            let escaped = document
                .replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "\"", with: "&quot;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
            html = """
            <!doctype html><meta charset="utf-8">
            <meta http-equiv="Content-Security-Policy" content="\(containerSecurityPolicy)">
            <title>AFM TUI HTML Preview</title>
            <iframe sandbox="allow-scripts" srcdoc="\(escaped)" style="position:fixed;inset:0;width:100%;height:100%;border:0"></iframe>
            """
        } else {
            let document = """
            <!doctype html><meta charset="utf-8">
            <meta http-equiv="Content-Security-Policy" content="\(contentSecurityPolicy)">
            <style>body{font:16px -apple-system;padding:2rem;color:#eee;background:#111}pre{white-space:pre-wrap}</style>
            <div id="app"></div><script>\(block.content)</script>
            """
            let escaped = document
                .replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "\"", with: "&quot;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
            html = """
            <!doctype html><meta charset="utf-8">
            <meta http-equiv="Content-Security-Policy" content="\(containerSecurityPolicy)">
            <title>AFM TUI JavaScript Preview</title>
            <iframe sandbox="allow-scripts" srcdoc="\(escaped)" style="position:fixed;inset:0;width:100%;height:100%;border:0"></iframe>
            """
        }
        try save(Data(html.utf8), to: url)
        return url
    }

    public static func openInBrowser(_ block: TUICodeBlock) throws -> URL {
        let url = try prepareBrowserArtifact(block)
        guard let executableURL = Bundle.main.executableURL else {
            throw TUIArtifactError.commandFailed("Unable to locate the afm executable for browser preview")
        }
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ["__tui-preview", url.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        return url
    }

    public static func quickLook(_ rawPath: String, maximumBytes: Int = 20_000_000) throws {
        let url = try resolvedURL(rawPath)
        try preflightRegularFile(at: url, maximumBytes: maximumBytes)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/qlmanage")
        process.arguments = ["-p", url.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
    }

    public static func inlineImageSequence(path rawPath: String, capabilities: TerminalCapabilities) throws -> String? {
        guard capabilities.inlineImages != .none else { return nil }
        let url = try resolvedURL(rawPath)
        if capabilities.inlineImages == .kitty, url.pathExtension.lowercased() != "png" { return nil }
        let data = try readRegularFile(at: url, maximumBytes: 20_000_000)
        let encoded = data.base64EncodedString()
        switch capabilities.inlineImages {
        case .iTerm2:
            return "\u{001B}]1337;File=inline=1;preserveAspectRatio=1:\(encoded)\u{0007}"
        case .kitty:
            let chunkSize = 4096
            var chunks: [String] = []
            var offset = encoded.startIndex
            var first = true
            while offset < encoded.endIndex {
                let end = encoded.index(offset, offsetBy: min(chunkSize, encoded.distance(from: offset, to: encoded.endIndex)))
                let more = end < encoded.endIndex ? 1 : 0
                let control = first ? "a=T,f=100,m=\(more)" : "m=\(more)"
                chunks.append("\u{001B}_G\(control);\(encoded[offset..<end])\u{001B}\\")
                first = false
                offset = end
            }
            return chunks.joined()
        case .none:
            return nil
        }
    }

    public static func preflightRegularFile(at url: URL, maximumBytes: Int) throws {
        var info = stat()
        guard maximumBytes >= 0,
              lstat(url.path, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size >= 0,
              info.st_size <= maximumBytes else {
            throw TUIArtifactError.unsafeReadableFile(url.path)
        }
    }

    public static func readRegularFile(at url: URL, maximumBytes: Int) throws -> Data {
        try preflightRegularFile(at: url, maximumBytes: maximumBytes)
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else { throw TUIArtifactError.unsafeReadableFile(url.path) }
        defer { close(descriptor) }

        var info = stat()
        guard fstat(descriptor, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size >= 0,
              info.st_size <= maximumBytes else {
            throw TUIArtifactError.unsafeReadableFile(url.path)
        }
        var data = Data()
        data.reserveCapacity(Int(info.st_size))
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let count = buffer.withUnsafeMutableBytes { bytes in
                Darwin.read(descriptor, bytes.baseAddress, bytes.count)
            }
            guard count >= 0 else { throw TUIArtifactError.unsafeReadableFile(url.path) }
            if count == 0 { break }
            guard data.count <= maximumBytes - count else {
                throw TUIArtifactError.unsafeReadableFile(url.path)
            }
            data.append(contentsOf: buffer.prefix(count))
        }
        return data
    }
}
