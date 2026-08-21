import Darwin
import Foundation

public struct TerminalCapabilities: Equatable, Sendable {
    public enum InlineImageProtocol: Equatable, Sendable { case iTerm2, kitty, none }

    public let isInteractive: Bool
    public let color: Bool
    public let hyperlinks: Bool
    public let inlineImages: InlineImageProtocol
    public let terminalProgram: String

    public static func detect(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        inputFD: Int32 = STDIN_FILENO,
        outputFD: Int32 = STDOUT_FILENO
    ) -> Self {
        detect(
            environment: environment,
            inputIsTTY: isatty(inputFD) != 0,
            outputIsTTY: isatty(outputFD) != 0
        )
    }

    public static func detect(
        environment: [String: String],
        inputIsTTY: Bool,
        outputIsTTY: Bool
    ) -> Self {
        let isInteractive = inputIsTTY && outputIsTTY
        let program = environment["TERM_PROGRAM"] ?? environment["TERM"] ?? "terminal"
        let term = environment["TERM"] ?? ""
        let color = isInteractive && environment["NO_COLOR"] == nil && term != "dumb"
        let imageProtocol: InlineImageProtocol
        if !isInteractive { imageProtocol = .none }
        else if program == "iTerm.app" { imageProtocol = .iTerm2 }
        else if term.contains("kitty") || environment["KITTY_WINDOW_ID"] != nil { imageProtocol = .kitty }
        else { imageProtocol = .none }
        return Self(
            isInteractive: isInteractive,
            color: color,
            hyperlinks: isInteractive && term != "dumb",
            inlineImages: imageProtocol,
            terminalProgram: program
        )
    }
}

public enum TerminalKey: Equatable, Sendable {
    case text(String), enter, newline, backspace, delete, left, right, up, down
    case home, end, escape, interrupt, eof, clear, unknown
}

public enum TerminalOutputSanitizer {
    public static func sanitize(_ value: String) -> String {
        var result = String.UnicodeScalarView()
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x0A:
                result.append(scalar)
            case 0x09:
                result.append(contentsOf: "    ".unicodeScalars)
            case 0x00...0x1F, 0x7F...0x9F, 0x202A...0x202E, 0x2066...0x2069:
                result.append("�")
            default:
                result.append(scalar)
            }
        }
        return String(result)
    }
}

/// Keeps process-level backend diagnostics away from the interactive terminal.
///
/// MLX and some of its dependencies write directly to stdout/stderr instead of
/// going through AFM's structured response stream. Those writes invalidate the
/// TUI's cursor accounting during a redraw. This scope preserves a duplicate of
/// the original terminal for `TerminalIO`, redirects the process streams to a
/// private log, and restores both streams on every exit path.
public final class TerminalOutputIsolation: @unchecked Sendable {
    public static var defaultLogURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".afm/logs/tui.log", isDirectory: false)
    }

    public let terminalOutputFD: Int32
    public let logURL: URL

    private let outputFD: Int32
    private let errorFD: Int32
    private var originalErrorFD: Int32
    private var logFD: Int32
    private let lock = NSLock()
    private var active = true

    public init(
        logURL: URL = TerminalOutputIsolation.defaultLogURL,
        outputFD: Int32 = STDOUT_FILENO,
        errorFD: Int32 = STDERR_FILENO
    ) throws {
        self.logURL = logURL
        self.outputFD = outputFD
        self.errorFD = errorFD

        let directory = logURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: directory.path
        )

        Self.flushIfStandard(outputFD, errorFD)

        let savedOutput = dup(outputFD)
        guard savedOutput >= 0 else { throw Self.posixError() }
        terminalOutputFD = savedOutput

        let savedError = dup(errorFD)
        guard savedError >= 0 else {
            close(savedOutput)
            throw Self.posixError()
        }
        originalErrorFD = savedError

        let openedLog = open(
            logURL.path,
            O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC,
            S_IRUSR | S_IWUSR
        )
        guard openedLog >= 0 else {
            close(savedOutput)
            close(savedError)
            throw Self.posixError()
        }
        logFD = openedLog
        _ = fchmod(openedLog, S_IRUSR | S_IWUSR)

        guard dup2(openedLog, outputFD) >= 0 else {
            close(savedOutput)
            close(savedError)
            close(openedLog)
            throw Self.posixError()
        }
        guard dup2(openedLog, errorFD) >= 0 else {
            _ = dup2(savedOutput, outputFD)
            close(savedOutput)
            close(savedError)
            close(openedLog)
            throw Self.posixError()
        }
    }

    public func restore() {
        lock.lock()
        guard active else { lock.unlock(); return }
        active = false
        let savedOutput = terminalOutputFD
        let savedError = originalErrorFD
        let sink = logFD
        originalErrorFD = -1
        logFD = -1
        lock.unlock()

        Self.flushIfStandard(outputFD, errorFD)
        _ = dup2(savedOutput, outputFD)
        _ = dup2(savedError, errorFD)
        close(savedOutput)
        close(savedError)
        close(sink)
    }

    deinit { restore() }

    private static func flushIfStandard(_ outputFD: Int32, _ errorFD: Int32) {
        if outputFD == STDOUT_FILENO { fflush(stdout) }
        if errorFD == STDERR_FILENO { fflush(stderr) }
    }

    private static func posixError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}

/// Owns terminal mode transitions. Restoration is idempotent and occurs on every exit path.
public final class TerminalModeController: @unchecked Sendable {
    private let enterAction: () throws -> Void
    private let restoreAction: () -> Void
    private let lock = NSLock()
    private var active = false

    public init(enter: @escaping () throws -> Void, restore: @escaping () -> Void) {
        enterAction = enter
        restoreAction = restore
    }

    public func enter() throws {
        lock.lock(); defer { lock.unlock() }
        guard !active else { return }
        try enterAction()
        active = true
    }

    public func restore() {
        lock.lock()
        let shouldRestore = active
        active = false
        lock.unlock()
        if shouldRestore { restoreAction() }
    }

    deinit { restore() }
}

public final class TerminalIO: @unchecked Sendable {
    private let inputFD: Int32
    private let outputFD: Int32
    private var original = termios()
    private lazy var mode = TerminalModeController(
        enter: { [weak self] in
            guard let self else { throw POSIXError(.ENOTTY) }
            try self.enterRawMode()
        },
        restore: { [weak self] in self?.restoreTermios() }
    )

    public init(inputFD: Int32 = STDIN_FILENO, outputFD: Int32 = STDOUT_FILENO) {
        self.inputFD = inputFD
        self.outputFD = outputFD
    }

    public func enter() throws { try mode.enter() }

    public func restore() {
        mode.restore()
        write("\u{001B}[?25h\u{001B}[0m\u{001B}[?1049l")
    }

    public func enterAlternateScreen() { write("\u{001B}[?1049h\u{001B}[2J\u{001B}[H") }
    public func clearScreen() { write("\u{001B}[2J\u{001B}[H") }
    public func clearLine() { write("\r\u{001B}[2K") }
    public func hideCursor() { write("\u{001B}[?25l") }
    public func showCursor() { write("\u{001B}[?25h") }

    public func write(_ string: String) {
        let data = Data(string.utf8)
        data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(outputFD, base.advanced(by: offset), bytes.count - offset)
                if count <= 0 { break }
                offset += count
            }
        }
    }

    public func width() -> Int {
        var size = winsize()
        return ioctl(outputFD, TIOCGWINSZ, &size) == 0 && size.ws_col > 0 ? Int(size.ws_col) : 80
    }

    public func readKey(timeoutMilliseconds: Int32 = 100) -> TerminalKey? {
        var readSet = fd_set()
        FD_ZERO(&readSet)
        FD_SET(inputFD, &readSet)
        var timeout = timeval(tv_sec: 0, tv_usec: timeoutMilliseconds * 1_000)
        let ready = select(inputFD + 1, &readSet, nil, nil, &timeout)
        guard ready > 0 else { return nil }
        var byte: UInt8 = 0
        guard Darwin.read(inputFD, &byte, 1) == 1 else { return .eof }
        switch byte {
        case 3: return .interrupt
        case 4: return .eof
        case 8, 127: return .backspace
        case 10: return .newline
        case 12: return .clear
        case 13: return .enter
        case 27: return readEscapeSequence()
        default:
            if byte < 32 { return .unknown }
            return readUTF8(first: byte)
        }
    }

    private func readEscapeSequence() -> TerminalKey {
        guard let next = readByte(20) else { return .escape }
        if next == 13 || next == 10 { return .newline } // Option/Escape + Enter
        guard next == 91 else { return .escape }
        guard let third = readByte(20) else { return .escape }
        switch third {
        case 65: return .up
        case 66: return .down
        case 67: return .right
        case 68: return .left
        case 72: return .home
        case 70: return .end
        case 51:
            _ = readByte(20)
            return .delete
        default: return .unknown
        }
    }

    private func readByte(_ timeoutMS: Int32) -> UInt8? {
        var readSet = fd_set(); FD_ZERO(&readSet); FD_SET(inputFD, &readSet)
        var timeout = timeval(tv_sec: 0, tv_usec: timeoutMS * 1_000)
        guard select(inputFD + 1, &readSet, nil, nil, &timeout) > 0 else { return nil }
        var byte: UInt8 = 0
        return Darwin.read(inputFD, &byte, 1) == 1 ? byte : nil
    }

    private func readUTF8(first: UInt8) -> TerminalKey {
        let length: Int
        if first < 0x80 { length = 1 }
        else if first & 0xE0 == 0xC0 { length = 2 }
        else if first & 0xF0 == 0xE0 { length = 3 }
        else if first & 0xF8 == 0xF0 { length = 4 }
        else { return .unknown }
        var bytes = [first]
        while bytes.count < length {
            guard let byte = readByte(20) else { return .unknown }
            bytes.append(byte)
        }
        return String(bytes: bytes, encoding: .utf8).map(TerminalKey.text) ?? .unknown
    }

    private func enterRawMode() throws {
        guard tcgetattr(inputFD, &original) == 0 else {
            throw POSIXError(.ENOTTY)
        }
        var raw = original
        raw.c_lflag &= ~tcflag_t(ECHO | ICANON | IEXTEN | ISIG)
        raw.c_iflag &= ~tcflag_t(IXON | ICRNL | BRKINT | INPCK | ISTRIP)
        raw.c_cflag |= tcflag_t(CS8)
        withUnsafeMutablePointer(to: &raw.c_cc) { pointer in
            pointer.withMemoryRebound(to: cc_t.self, capacity: Int(NCCS)) { control in
                control[Int(VMIN)] = 0
                control[Int(VTIME)] = 1
            }
        }
        guard tcsetattr(inputFD, TCSAFLUSH, &raw) == 0 else { throw POSIXError(.EIO) }
    }

    private func restoreTermios() { _ = tcsetattr(inputFD, TCSAFLUSH, &original) }
}

private func FD_ZERO(_ set: inout fd_set) { set = fd_set() }

private func FD_SET(_ fd: Int32, _ set: inout fd_set) {
    let intOffset = Int(fd) / 32
    let bitOffset = Int(fd) % 32
    withUnsafeMutablePointer(to: &set.fds_bits) { pointer in
        pointer.withMemoryRebound(to: Int32.self, capacity: 32) { bits in
            bits[intOffset] |= 1 << bitOffset
        }
    }
}
