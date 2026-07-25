import AFMKit
import Foundation
import Darwin

struct CachedModelEntry {
    let id: String
    let loadIdentifier: String
    let source: String
}

// MARK: - Model Discovery

func discoverAllModels(resolver: MLXCacheResolver) -> [CachedModelEntry] {
    AFMMLXModelStore(resolver: resolver).discoverLocalModels().map {
        CachedModelEntry(
            id: $0.id.rawValue,
            loadIdentifier: $0.loadIdentifier,
            source: "[\($0.origin.displayLabel)]"
        )
    }
}

// MARK: - Interactive Picker

nonisolated(unsafe) private var savedTermios = termios()
nonisolated(unsafe) private var terminalModified = false

func runInteractiveModelPicker(models: [CachedModelEntry]) -> String? {
    guard !models.isEmpty else { return nil }

    // Save terminal state and switch to raw mode
    tcgetattr(STDIN_FILENO, &savedTermios)
    var raw = savedTermios
    raw.c_lflag &= ~UInt(ICANON | ECHO)
    raw.c_cc.16 = 1  // VMIN
    raw.c_cc.17 = 0  // VTIME
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
    terminalModified = true

    // Install signal handler to restore terminal on Ctrl-C
    signal(SIGINT) { _ in
        restoreTerminal()
        print("\n")
        _exit(1)
    }

    // Hide cursor
    print("\u{1B}[?25l", terminator: "")

    // Get terminal height for viewport
    var ws = winsize()
    let termHeight: Int
    if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &ws) == 0, ws.ws_row > 0 {
        termHeight = Int(ws.ws_row)
    } else {
        termHeight = 24  // fallback
    }

    // Print banner
    let s = models.count == 1 ? "" : "s"
    print("\u{1B}[1;33mFound \(models.count) model\(s) cached locally (no download needed).\u{1B}[0m")
    print("\u{1B}[1;33mSelect a model (\u{2191}\u{2193} navigate, Enter select, q quit):\u{1B}[0m\n")

    let headerLines = 3  // banner + instruction + blank
    let maxVisible = max(termHeight - headerLines - 1, 5)  // reserve 1 line for safety
    let visibleCount = min(models.count, maxVisible)

    var selected = 0
    var scrollOffset = 0
    var firstDraw = true

    func draw() {
        // Move cursor up to overwrite previous draw (except first time)
        if !firstDraw {
            print("\u{1B}[\(visibleCount)A", terminator: "")
        }
        firstDraw = false

        // Adjust scroll offset to keep selection visible
        if selected < scrollOffset {
            scrollOffset = selected
        } else if selected >= scrollOffset + visibleCount {
            scrollOffset = selected - visibleCount + 1
        }

        for row in 0..<visibleCount {
            let i = scrollOffset + row
            if i < models.count {
                let m = models[i]
                let suffix = m.source.isEmpty ? "" : "  \(m.source)"
                if i == selected {
                    print("\u{1B}[2K \u{1B}[7m> \(m.id)\(suffix)\u{1B}[0m")
                } else {
                    print("\u{1B}[2K   \(m.id)\(suffix)")
                }
            } else {
                print("\u{1B}[2K")
            }
        }
        fflush(stdout)
    }

    draw()

    while true {
        var c: UInt8 = 0
        let n = read(STDIN_FILENO, &c, 1)
        guard n == 1 else { continue }

        switch c {
        case 0x1B: // Escape sequence
            var seq: [UInt8] = [0, 0]
            let n1 = read(STDIN_FILENO, &seq, 2)
            if n1 == 2, seq[0] == 0x5B { // CSI sequence
                switch seq[1] {
                case 0x41: // Up arrow
                    if selected > 0 { selected -= 1 }
                    draw()
                case 0x42: // Down arrow
                    if selected < models.count - 1 { selected += 1 }
                    draw()
                default:
                    break
                }
            } else if n1 == 0 || (n1 == 2 && seq[0] != 0x5B) {
                // Bare Escape - quit
                restoreTerminal()
                print("\u{1B}[?25h", terminator: "")
                fflush(stdout)
                return nil
            }

        case 0x0A, 0x0D: // Enter
            restoreTerminal()
            print("\u{1B}[?25h", terminator: "")
            fflush(stdout)
            return models[selected].loadIdentifier

        case 0x71, 0x51: // q or Q
            restoreTerminal()
            print("\u{1B}[?25h", terminator: "")
            fflush(stdout)
            return nil

        case 0x03: // Ctrl-C
            restoreTerminal()
            print("\u{1B}[?25h\n", terminator: "")
            fflush(stdout)
            return nil

        default:
            break
        }
    }
}

private func restoreTerminal() {
    if terminalModified {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &savedTermios)
        terminalModified = false
        // Show cursor
        print("\u{1B}[?25h", terminator: "")
        fflush(stdout)
    }
    // Restore default signal handler
    signal(SIGINT, SIG_DFL)
}
