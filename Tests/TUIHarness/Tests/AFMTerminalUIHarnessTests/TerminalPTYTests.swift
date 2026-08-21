import Darwin
import Foundation
import XCTest
import AFMTerminalUI

final class TerminalPTYTests: XCTestCase {
    func testMacKeyboardSequencesThroughARealPTY() throws {
        let pty = try PTYFixture()
        defer { pty.close() }
        let terminal = TerminalIO(inputFD: pty.slave, outputFD: pty.slave)
        try terminal.enter()
        defer { terminal.restore() }

        try pty.send([13])
        XCTAssertEqual(terminal.readKey(), .enter)
        try pty.send([27, 13])
        XCTAssertEqual(terminal.readKey(), .newline)
        try pty.send([20])
        XCTAssertEqual(terminal.readKey(), .openTranscript)
        try pty.send([21])
        XCTAssertEqual(terminal.readKey(), .halfPageUp)
        try pty.send([9])
        XCTAssertEqual(terminal.readKey(), .tab)
        try pty.send([27, 91, 65])
        XCTAssertEqual(terminal.readKey(), .up)
        try pty.send([27, 91, 66])
        XCTAssertEqual(terminal.readKey(), .down)
        try pty.send([27, 91, 67])
        XCTAssertEqual(terminal.readKey(), .right)
        try pty.send([27, 91, 68])
        XCTAssertEqual(terminal.readKey(), .left)
        try pty.send([27, 91, 72])
        XCTAssertEqual(terminal.readKey(), .home)
        try pty.send([27, 91, 70])
        XCTAssertEqual(terminal.readKey(), .end)
        try pty.send([27, 91, 53, 126])
        XCTAssertEqual(terminal.readKey(), .pageUp)
        try pty.send([27, 91, 54, 126])
        XCTAssertEqual(terminal.readKey(), .pageDown)
        try pty.send(Array("🧪".utf8))
        XCTAssertEqual(terminal.readKey(), .text("🧪"))
    }

    func testRawModeAndAlternateScreenAreAlwaysRestored() throws {
        let pty = try PTYFixture()
        defer { pty.close() }
        let before = try pty.attributes()
        let terminal = TerminalIO(inputFD: pty.slave, outputFD: pty.slave)

        try terminal.enter()
        let raw = try pty.attributes()
        XCTAssertEqual(raw.c_lflag & tcflag_t(ICANON), 0)
        XCTAssertEqual(raw.c_lflag & tcflag_t(ECHO), 0)
        terminal.enterAlternateScreen()
        terminal.hideCursor()
        // Drain the PTY before tcsetattr(TCSAFLUSH): a real terminal emulator is
        // concurrently consuming these bytes, while an unattended PTY is not.
        let enteredOutput = try pty.receiveUntilQuiet()
        terminal.restore()
        terminal.restore()

        let restored = try pty.attributes()
        XCTAssertEqual(restored.c_lflag & tcflag_t(ICANON), before.c_lflag & tcflag_t(ICANON))
        XCTAssertEqual(restored.c_lflag & tcflag_t(ECHO), before.c_lflag & tcflag_t(ECHO))

        let output = enteredOutput + (try pty.receiveUntilQuiet())
        XCTAssertTrue(output.contains("\u{001B}[?1049h"), "alternate screen was never entered")
        XCTAssertTrue(output.contains("\u{001B}[?1007h"), "alternate scroll was never enabled")
        XCTAssertTrue(output.contains("\u{001B}[?1007l"), "alternate scroll was not disabled")
        XCTAssertTrue(output.contains("\u{001B}[?25h"), "cursor was not restored")
        XCTAssertTrue(output.contains("\u{001B}[?1049l"), "alternate screen was not left")
    }

    func testWindowDimensionsComeFromPTY() throws {
        let pty = try PTYFixture(rows: 41, columns: 117)
        defer { pty.close() }
        let terminal = TerminalIO(inputFD: pty.slave, outputFD: pty.slave)
        XCTAssertEqual(terminal.width(), 117)
        XCTAssertEqual(terminal.height(), 41)
    }

    func testNonInteractiveInvocationFailsBeforeModelLoading() {
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(
            tui: true,
            webUI: false,
            singlePrompt: false,
            inputIsTTY: false,
            outputIsTTY: true
        )) { error in
            XCTAssertEqual(
                error as? TUIInvocationError,
                .conflict("--tui requires interactive terminal input")
            )
        }
        XCTAssertNoThrow(try TUIInvocationPolicy.validate(
            tui: true,
            webUI: false,
            singlePrompt: false,
            inputIsTTY: true,
            outputIsTTY: true
        ))
    }

    func testModelOutputCannotInjectTerminalControlSequences() {
        XCTAssertEqual(
            TerminalOutputSanitizer.sanitize("safe\u{001B}[2J\u{0007}text\tend"),
            "safe�[2J�text    end"
        )
    }
}

private final class PTYFixture {
    let master: Int32
    let slave: Int32
    private var isClosed = false

    init(rows: UInt16 = 30, columns: UInt16 = 100) throws {
        var master: Int32 = -1
        var slave: Int32 = -1
        var size = winsize(ws_row: rows, ws_col: columns, ws_xpixel: 0, ws_ypixel: 0)
        guard openpty(&master, &slave, nil, nil, &size) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        self.master = master
        self.slave = slave
    }

    func send(_ bytes: [UInt8]) throws {
        let count = bytes.withUnsafeBytes { buffer in
            Darwin.write(master, buffer.baseAddress, buffer.count)
        }
        guard count == bytes.count else { throw POSIXError(.EIO) }
    }

    func attributes() throws -> termios {
        var value = termios()
        guard tcgetattr(slave, &value) == 0 else { throw POSIXError(.EIO) }
        return value
    }

    func receiveUntilQuiet() throws -> String {
        var data = Data()
        while true {
            var readSet = fd_set()
            readSet.zero()
            readSet.set(master)
            var timeout = timeval(tv_sec: 0, tv_usec: 50_000)
            let ready = select(master + 1, &readSet, nil, nil, &timeout)
            if ready == 0 { break }
            guard ready > 0 else { throw POSIXError(.EIO) }
            var buffer = [UInt8](repeating: 0, count: 4096)
            let count = Darwin.read(master, &buffer, buffer.count)
            guard count > 0 else { break }
            data.append(contentsOf: buffer.prefix(count))
        }
        return String(decoding: data, as: UTF8.self)
    }

    func close() {
        guard !isClosed else { return }
        isClosed = true
        Darwin.close(master)
        Darwin.close(slave)
    }

    deinit { close() }
}

private extension fd_set {
    mutating func zero() { self = fd_set() }

    mutating func set(_ descriptor: Int32) {
        let offset = Int(descriptor) / 32
        let bit = Int(descriptor) % 32
        withUnsafeMutablePointer(to: &fds_bits) { pointer in
            pointer.withMemoryRebound(to: Int32.self, capacity: 32) { words in
                words[offset] |= 1 << bit
            }
        }
    }
}
