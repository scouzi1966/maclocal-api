import Foundation
import Darwin

struct CachedModelEntry {
    let id: String        // e.g. "mlx-community/Qwen3-VL-4B-Instruct-8bit"
    let source: String    // e.g. "" or "[LM Studio]"
}

// MARK: - Model Discovery

func discoverAllModels(resolver: MLXCacheResolver) -> [CachedModelEntry] {
    let fm = FileManager.default
    let env = ProcessInfo.processInfo.environment
    var seen = Set<String>()
    var results: [CachedModelEntry] = []

    func addModel(_ id: String, source: String = "") {
        guard !seen.contains(id) else { return }
        seen.insert(id)
        results.append(CachedModelEntry(id: id, source: source))
    }

    // Helper: scan HF-style hub directory (models--org--name)
    func scanHFHub(_ hubDir: URL) {
        guard let entries = try? fm.contentsOfDirectory(atPath: hubDir.path) else { return }
        for entry in entries where entry.hasPrefix("models--") {
            // HF naming: models--org--model where org/model parts use -- as separator
            let components = entry.dropFirst("models--".count).components(separatedBy: "--")
            guard components.count >= 2 else { continue }
            let org = components[0]
            let model = components.dropFirst().joined(separator: "--")
            let modelID = "\(org)/\(model)"
            let dir = hubDir.appendingPathComponent(entry)
            if hasValidModel(dir) {
                addModel(modelID)
            }
        }
    }

    // Helper: scan flat-style directory (<root>/<org>/<model>)
    func scanFlat(_ baseDir: URL, source: String = "") {
        guard let orgs = try? fm.contentsOfDirectory(atPath: baseDir.path) else { return }
        for org in orgs {
            let orgDir = baseDir.appendingPathComponent(org)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: orgDir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            guard let models = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for model in models {
                let modelDir = orgDir.appendingPathComponent(model)
                guard fm.fileExists(atPath: modelDir.path, isDirectory: &isDir), isDir.boolValue else { continue }
                if hasValidModel(modelDir) {
                    addModel("\(org)/\(model)", source: source)
                }
            }
        }
    }

    // 1. MACAFM_MLX_MODEL_CACHE
    if let root = resolver.cacheRoot {
        scanFlat(root)
        scanFlat(root.appendingPathComponent("models"))
        scanHFHub(root.appendingPathComponent("huggingface/hub"))
    }

    // 2. Swift Hub default: ~/Documents/huggingface/models/
    if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
        scanFlat(docs.appendingPathComponent("huggingface/models"))
    }

    // 3. HF env vars
    for key in ["HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"] {
        if let val = env[key]?.trimmingCharacters(in: .whitespacesAndNewlines), !val.isEmpty {
            let base = URL(fileURLWithPath: NSString(string: val).expandingTildeInPath)
            scanHFHub(base)
        }
    }

    // 4. HF_HOME
    if let val = env["HF_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines), !val.isEmpty {
        let base = URL(fileURLWithPath: NSString(string: val).expandingTildeInPath)
        scanHFHub(base.appendingPathComponent("hub"))
    }

    // 5. XDG_CACHE_HOME
    if let val = env["XDG_CACHE_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines), !val.isEmpty {
        let base = URL(fileURLWithPath: NSString(string: val).expandingTildeInPath)
        scanHFHub(base.appendingPathComponent("huggingface/hub"))
    }

    // 6. Default Python HF cache
    let defaultHFCache = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
    scanHFHub(defaultHFCache)

    // 7. macOS Library/Caches
    if let library = fm.urls(for: .libraryDirectory, in: .userDomainMask).first {
        scanFlat(library.appendingPathComponent("Caches/models"))
        scanHFHub(library.appendingPathComponent("Caches/huggingface/hub"))
    }

    // 8. LM Studio cache: ~/.cache/lm-studio/models/<publisher>/<model>/
    let lmStudioDir = fm.homeDirectoryForCurrentUser.appendingPathComponent(".cache/lm-studio/models")
    scanFlat(lmStudioDir, source: "[LM Studio]")

    return results.sorted { $0.id.lowercased() < $1.id.lowercased() }
}

private func hasValidModel(_ dir: URL) -> Bool {
    let fm = FileManager.default

    // Check snapshots subdir (HF-style)
    let snapshots = dir.appendingPathComponent("snapshots")
    if fm.fileExists(atPath: snapshots.path),
       let names = try? fm.contentsOfDirectory(atPath: snapshots.path),
       let first = names.first {
        let snapshotDir = snapshots.appendingPathComponent(first)
        if hasConfigAndWeights(snapshotDir) { return true }
    }

    return hasConfigAndWeights(dir)
}

private func hasConfigAndWeights(_ dir: URL) -> Bool {
    guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return false }
    let hasConfig = files.contains("config.json")
    let hasWeights = files.contains(where: { $0.hasSuffix(".safetensors") || $0 == "model.safetensors.index.json" })
    return hasConfig && hasWeights
}

// MARK: - Interactive Picker

private var savedTermios = termios()
private var terminalModified = false

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

    // Print banner before entering raw mode
    let s = models.count == 1 ? "" : "s"
    print("Found \(models.count) model\(s) cached locally (no download needed).")
    print("Select a model (\u{2191}\u{2193} navigate, Enter select, q quit):\n")

    var selected = 0
    var firstDraw = true
    let headerLines = 3  // banner + instruction + blank

    func draw() {
        // Move cursor up to overwrite previous draw (except first time)
        if !firstDraw {
            let totalLines = models.count
            print("\u{1B}[\(totalLines)A", terminator: "")
        }
        firstDraw = false

        for (i, m) in models.enumerated() {
            let suffix = m.source.isEmpty ? "" : "  \(m.source)"
            if i == selected {
                // Reverse video for selected line
                print("\u{1B}[2K \u{1B}[7m> \(m.id)\(suffix)\u{1B}[0m")
            } else {
                print("\u{1B}[2K   \(m.id)\(suffix)")
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
            return models[selected].id

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
