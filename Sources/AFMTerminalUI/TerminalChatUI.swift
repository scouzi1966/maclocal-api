import Darwin
import Dispatch
import Foundation
import AFMKit
import AFMOpenAICompat

public struct TerminalChatConfiguration: Sendable {
    public let backend: AFMBackend
    public let backendName: String
    public let modelName: String
    public let engine: EngineConfig
    public let generation: GenerationConfig
    public let theme: TerminalMarkdownRenderer.Theme
    public let showReasoning: Bool
    public let streaming: Bool
    public let initialAttachments: [URL]

    public init(
        backend: AFMBackend,
        backendName: String,
        modelName: String,
        engine: EngineConfig = .init(),
        generation: GenerationConfig = .init(),
        theme: TerminalMarkdownRenderer.Theme = .auto,
        showReasoning: Bool = false,
        streaming: Bool = true,
        initialAttachments: [URL] = []
    ) {
        self.backend = backend
        self.backendName = backendName
        self.modelName = modelName
        self.engine = engine
        self.generation = generation
        self.theme = theme
        self.showReasoning = showReasoning
        self.streaming = streaming
        self.initialAttachments = initialAttachments
    }
}

public enum TUIInvocationPolicy {
    public static func validate(tui: Bool, webUI: Bool, singlePrompt: Bool, pipedInput: Bool) throws {
        guard tui else { return }
        if webUI { throw TUIInvocationError.conflict("--tui cannot be combined with --webui") }
        if singlePrompt { throw TUIInvocationError.conflict("--tui cannot be combined with --single-prompt") }
        if pipedInput { throw TUIInvocationError.conflict("--tui requires an interactive terminal and cannot read piped input") }
    }
}

public enum TUIInvocationError: Error, LocalizedError, Equatable {
    case conflict(String)
    public var errorDescription: String? { if case .conflict(let value) = self { return value }; return nil }
}

struct GenerationSnapshot: Sendable {
    let revision: UInt64
    let text: String
    let reasoning: String
    let tools: [AFMToolCall]
    let toolStages: [AFMToolCallStage]
    let promptTokens: Int
    let completionTokens: Int
    let cachedTokens: Int
    let completed: Bool
    let cancelled: Bool
    let error: String?
}

actor GenerationBuffer {
    private var text = ""
    private var reasoning = ""
    private var tools: [AFMToolCall] = []
    private var toolStages: [AFMToolCallStage] = []
    private var promptTokens = 0
    private var completionTokens = 0
    private var cachedTokens = 0
    private var completed = false
    private var cancelled = false
    private var error: String?
    private var revision: UInt64 = 0

    func accept(_ event: AFMStreamEvent) {
        guard !completed else { return }
        switch event {
        case .text(let value, let tokens): text += value; completionTokens = max(completionTokens, tokens)
        case .reasoning(let value, _): reasoning += value
        case .toolCall(let call, let stage):
            if let index = tools.firstIndex(where: { $0.id == call.id }) {
                tools[index] = call
                toolStages[index] = stage
            } else {
                tools.append(call)
                toolStages.append(stage)
            }
        case .usage(let prompt, let completion, let cached):
            promptTokens = prompt; completionTokens = completion; cachedTokens = cached
        case .completed: completed = true
        case .tokenLogprobs, .metadata, .custom: break
        }
        revision &+= 1
    }

    func accept(_ response: AFMResponse) {
        guard !completed else { return }
        text = response.content
        reasoning = response.reasoningContent ?? ""
        promptTokens = response.promptTokens
        completionTokens = response.completionTokens
        cachedTokens = response.cachedPromptTokens
        for call in response.toolCalls ?? [] {
            tools.append(AFMToolCall(id: call.id, name: call.function.name, arguments: call.function.arguments))
            toolStages.append(.completed)
        }
        completed = true
        revision &+= 1
    }

    func fail(_ value: Error) {
        guard !completed else { return }
        error = value.localizedDescription; completed = true; revision &+= 1
    }
    func cancel() {
        guard !completed else { return }
        cancelled = true; completed = true; revision &+= 1
    }
    func finish() {
        guard !completed else { return }
        completed = true; revision &+= 1
    }
    func snapshot() -> GenerationSnapshot {
        GenerationSnapshot(
            revision: revision,
            text: text, reasoning: reasoning, tools: tools, toolStages: toolStages,
            promptTokens: promptTokens, completionTokens: completionTokens,
            cachedTokens: cachedTokens, completed: completed,
            cancelled: cancelled, error: error
        )
    }

    func snapshot(ifChangedSince priorRevision: UInt64) -> GenerationSnapshot? {
        guard revision != priorRevision else { return nil }
        return snapshot()
    }
}

private actor GenerationTaskState {
    private var finished = false
    func markFinished() { finished = true }
    func isFinished() -> Bool { finished }
}

final class TUISignalMonitor: @unchecked Sendable {
    private typealias SignalHandler = @convention(c) (Int32) -> Void
    private let lock = NSLock()
    private var terminated = false
    private var stopped = false
    private var sources: [DispatchSourceSignal] = []
    private var previousHandlers: [(Int32, SignalHandler?)] = []
    private let monitoredSignals = [SIGTERM, SIGHUP]
    private let queue = DispatchQueue(label: "ai.maclocal.afm.tui-signals")

    init() {
        for signalNumber in monitoredSignals {
            previousHandlers.append((signalNumber, signal(signalNumber, SIG_IGN)))
            let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: queue)
            source.setEventHandler { [weak self] in
                self?.lock.lock(); self?.terminated = true; self?.lock.unlock()
            }
            source.resume()
            sources.append(source)
        }
    }

    var shouldTerminate: Bool { lock.lock(); defer { lock.unlock() }; return terminated }
    func stop() {
        lock.lock()
        guard !stopped else { lock.unlock(); return }
        stopped = true
        let sources = sources
        self.sources.removeAll()
        let handlers = previousHandlers
        previousHandlers.removeAll()
        lock.unlock()

        for source in sources { source.cancel() }
        queue.sync {}
        for (signalNumber, handler) in handlers { signal(signalNumber, handler) }
    }

    deinit { stop() }
}

public final class AFMTerminalChat: @unchecked Sendable {
    private let configuration: TerminalChatConfiguration
    private let terminal: TerminalIO
    private let capabilities: TerminalCapabilities
    private let store: TUISessionStore
    private let engine: AFMEngine
    private var session: TUISession
    private var renderer: TerminalMarkdownRenderer
    private var showReasoning: Bool
    private var promptHistory: [String] = []
    private var codeBlocks: [TUICodeBlock] = []
    private var images: [TUIImageReference] = []
    private var attachments: [URL] = []
    private var lastStatistics = "No generation yet"
    private var lastInput: String?
    private var lastReasoning = ""

    public init(
        configuration: TerminalChatConfiguration,
        terminal: TerminalIO = TerminalIO(),
        capabilities: TerminalCapabilities = .detect(),
        sessionStore: TUISessionStore = TUISessionStore()
    ) {
        self.configuration = configuration
        self.terminal = terminal
        self.capabilities = capabilities
        self.store = sessionStore
        self.engine = AFMEngine(backend: configuration.backend, config: configuration.engine)
        self.session = TUISession(
            backend: configuration.backendName,
            model: configuration.modelName,
            messages: Self.initialMessages(for: configuration)
        )
        self.renderer = TerminalMarkdownRenderer(color: capabilities.color, theme: configuration.theme)
        self.showReasoning = configuration.showReasoning
        self.attachments = configuration.initialAttachments
    }

    public func run() async throws {
        guard capabilities.isInteractive else {
            throw TUIInvocationError.conflict("--tui requires an interactive terminal")
        }
        try terminal.enter()
        terminal.enterAlternateScreen()
        let signalMonitor = TUISignalMonitor()
        defer {
            signalMonitor.stop()
            Task { await engine.unload() }
            terminal.restore()
        }

        drawWelcome()
        terminal.write("Loading \(configuration.modelName)…\n")
        let modelID = try await engine.load { [terminal] progress in
            terminal.clearLine()
            terminal.write(String(format: "Loading model %3.0f%%", progress * 100))
        }
        terminal.clearLine()
        terminal.write("Ready: \(modelID)\n\n")

        while !signalMonitor.shouldTerminate {
            guard let input = readInput(initial: nil, signalMonitor: signalMonitor) else { break }
            let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed.hasPrefix("/") {
                do {
                    if try await handleCommand(trimmed, signalMonitor: signalMonitor) { break }
                } catch {
                    terminal.write("\(style("error", "1;31")): \(error.localizedDescription)\n")
                }
                continue
            }
            try await send(input, signalMonitor: signalMonitor)
        }
    }

    private func send(_ input: String, signalMonitor: TUISignalMonitor) async throws {
        promptHistory.append(input)
        lastInput = input
        let userMessage = try makeUserMessage(input)
        attachments.removeAll()
        if !session.messages.contains(where: { $0.role == "user" }) {
            session.title = String(input.replacingOccurrences(of: "\n", with: " ").prefix(72))
        }
        session.messages.append(userMessage)
        session.updatedAt = Date()
        terminal.write("\n\(style("you", "1;34")) › \(input)\n\n")

        let buffer = GenerationBuffer()
        let taskState = GenerationTaskState()
        let messages = session.messages
        let generation = configuration.generation
        let task = Task {
            do {
                if configuration.streaming {
                    for try await event in engine.streamEvents(to: messages, generation) {
                        try Task.checkCancellation()
                        await buffer.accept(event)
                    }
                    await buffer.finish()
                } else {
                    await buffer.accept(try await engine.respond(to: messages, generation))
                }
            } catch is CancellationError {
                await buffer.cancel()
            } catch {
                await buffer.fail(error)
            }
            await taskState.markFinished()
        }

        var previousRows = 0
        var lastSnapshot = await buffer.snapshot()
        let start = Date()
        while !lastSnapshot.completed && !signalMonitor.shouldTerminate {
            if let key = terminal.readKey(timeoutMilliseconds: 40), key == .interrupt {
                task.cancel()
                await buffer.cancel()
            }
            if let changed = await buffer.snapshot(ifChangedSince: lastSnapshot.revision) {
                lastSnapshot = changed
                previousRows = redrawGeneration(lastSnapshot, previousRows: previousRows, final: false)
            }
        }
        if signalMonitor.shouldTerminate { task.cancel() }
        let clock = ContinuousClock()
        let cancellationRequested = lastSnapshot.cancelled || signalMonitor.shouldTerminate
        if cancellationRequested {
            task.cancel()
        }
        let deadline = clock.now.advanced(by: cancellationRequested ? .milliseconds(500) : .seconds(1))
        while !(await taskState.isFinished()), clock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        if !(await taskState.isFinished()) { task.cancel() }
        lastSnapshot = await buffer.snapshot()
        lastReasoning = lastSnapshot.reasoning
        _ = redrawGeneration(lastSnapshot, previousRows: previousRows, final: true)
        terminal.write("\n")

        if let error = lastSnapshot.error {
            terminal.write("\(style("error", "1;31")): \(error)\n\n")
            return
        }
        if lastSnapshot.cancelled || (lastSnapshot.text.isEmpty && lastSnapshot.tools.isEmpty) {
            terminal.write("\(style("cancelled", "33")) — the partial response was not added.\n\n")
            return
        }

        let messageToolCalls = lastSnapshot.tools.map {
            MessageToolCall(
                id: $0.id,
                type: "function",
                function: MessageToolCallFunction(name: $0.name, arguments: $0.arguments)
            )
        }
        let assistantIndex = session.messages.count
        session.messages.append(Message(
            role: "assistant",
            content: lastSnapshot.text.isEmpty ? nil : .text(lastSnapshot.text),
            toolCalls: messageToolCalls.isEmpty ? nil : messageToolCalls
        ))
        if !lastSnapshot.reasoning.isEmpty {
            session.reasoningByMessage[String(assistantIndex)] = lastSnapshot.reasoning
        }
        session.updatedAt = Date()
        _ = try? store.save(session)
        let rendered = renderMarkdown(lastSnapshot.text)
        codeBlocks = rendered.codeBlocks
        images = rendered.images
        let elapsed = max(0.001, Date().timeIntervalSince(start))
        let rate = Double(lastSnapshot.completionTokens) / elapsed
        lastStatistics = String(
            format: "%d prompt · %d cached · %d generated · %.2fs · %.1f tok/s",
            lastSnapshot.promptTokens, lastSnapshot.cachedTokens,
            lastSnapshot.completionTokens, elapsed, rate
        )
        terminal.write("\(style("↳ \(lastStatistics)", "2"))\n")
        if !codeBlocks.isEmpty { terminal.write("\(style("\(codeBlocks.count) code block(s): /blocks, /copy, /save, /open", "2;36"))\n") }
        if !images.isEmpty { presentImages(images) }
        terminal.write("\n")
    }

    private func redrawGeneration(_ snapshot: GenerationSnapshot, previousRows: Int, final: Bool) -> Int {
        if previousRows > 0 {
            for index in 0..<previousRows {
                terminal.clearLine()
                if index < previousRows - 1 { terminal.write("\u{001B}[1A") }
            }
            terminal.write("\r")
        }
        var display = ""
        if !snapshot.reasoning.isEmpty {
            if showReasoning {
                display += style("reasoning", "2;35") + " ›\n" + renderGenerationMarkdown(snapshot.reasoning, final: final) + "\n\n"
            } else {
                display += style("… reasoning hidden (\(snapshot.reasoning.count) chars; /reasoning show)", "2") + "\n"
            }
        }
        display += style("assistant", "1;32") + " › " + renderGenerationMarkdown(snapshot.text, final: final)
        if snapshot.text.isEmpty && snapshot.reasoning.isEmpty { display += style("thinking…", "2") }
        if !snapshot.tools.isEmpty {
            let lines = zip(snapshot.tools, snapshot.toolStages).map { call, stage in
                "\(stage): \(call.name)(\(call.arguments))"
            }
            display += "\n" + style(lines.joined(separator: "\n"), "36")
        }
        terminal.write(display)
        return displayRows(display, width: terminal.width())
    }

    private func renderGenerationMarkdown(_ source: String, final: Bool) -> String {
        let limit = 12_000
        guard !final, source.count > limit else { return renderMarkdown(source).text }
        let suffix = String(source.suffix(limit))
        return style("… earlier streaming output omitted from redraw …", "2") + "\n" + renderMarkdown(suffix).text
    }

    private func readInput(initial: String?, signalMonitor: TUISignalMonitor) -> String? {
        var characters = Array(initial ?? "")
        var cursor = characters.count
        var historyIndex = promptHistory.count
        var priorRows = 0
        while !signalMonitor.shouldTerminate {
            priorRows = redrawInput(characters, cursor: cursor, priorRows: priorRows)
            guard let key = terminal.readKey() else { continue }
            switch key {
            case .text(let value): characters.insert(contentsOf: value, at: cursor); cursor += value.count
            case .enter:
                eraseRows(priorRows)
                return String(characters)
            case .newline: characters.insert("\n", at: cursor); cursor += 1
            case .backspace where cursor > 0: cursor -= 1; characters.remove(at: cursor)
            case .delete where cursor < characters.count: characters.remove(at: cursor)
            case .left where cursor > 0: cursor -= 1
            case .right where cursor < characters.count: cursor += 1
            case .home: cursor = 0
            case .end: cursor = characters.count
            case .up where !promptHistory.isEmpty:
                historyIndex = max(0, historyIndex - 1); characters = Array(promptHistory[historyIndex]); cursor = characters.count
            case .down where !promptHistory.isEmpty:
                historyIndex = min(promptHistory.count, historyIndex + 1)
                characters = historyIndex == promptHistory.count ? [] : Array(promptHistory[historyIndex]); cursor = characters.count
            case .clear: terminal.clearScreen(); drawWelcome(); priorRows = 0
            case .interrupt:
                if characters.isEmpty { eraseRows(priorRows); return nil }
                characters.removeAll(); cursor = 0
            case .eof: eraseRows(priorRows); return nil
            default: break
            }
        }
        return nil
    }

    private func redrawInput(_ characters: [Character], cursor: Int, priorRows: Int) -> Int {
        eraseRows(priorRows)
        let attachmentBadge = attachments.isEmpty ? "" : " [\(attachments.count) attachment(s)]"
        let content = String(characters).replacingOccurrences(of: "\n", with: "\n… ")
        let display = style("you\(attachmentBadge)", "1;34") + " › " + content
        terminal.write(display)
        // Cursor positioning across wide Unicode is terminal-dependent. Keep navigation exact
        // in the buffer and show a visual cursor when editing away from the end.
        if cursor < characters.count { terminal.write(style("  [cursor \(cursor)/\(characters.count)]", "2")) }
        return displayRows(display, width: terminal.width())
    }

    private func eraseRows(_ count: Int) {
        guard count > 0 else { return }
        for index in 0..<count {
            terminal.clearLine()
            if index < count - 1 { terminal.write("\u{001B}[1A") }
        }
        terminal.write("\r")
    }

    /// Returns true when the UI should exit.
    private func handleCommand(_ raw: String, signalMonitor: TUISignalMonitor) async throws -> Bool {
        let pieces = splitCommand(raw)
        let command = pieces.first?.lowercased() ?? ""
        switch command {
        case "/quit", "/exit", "/q": return true
        case "/help", "/?": drawHelp()
        case "/new":
            session = TUISession(
                backend: configuration.backendName,
                model: configuration.modelName,
                messages: Self.initialMessages(for: configuration)
            )
            codeBlocks = []; images = []; terminal.clearScreen(); drawWelcome()
        case "/clear": terminal.clearScreen(); drawWelcome()
        case "/status": terminal.write("\(configuration.backendName) · \(configuration.modelName)\n\(lastStatistics)\n")
        case "/reasoning":
            let value = pieces.dropFirst().first?.lowercased()
            showReasoning = value == "show" || value == "on" ? true : value == "hide" || value == "off" ? false : !showReasoning
            terminal.write("Reasoning is now \(showReasoning ? "visible" : "hidden").\n")
            if showReasoning, !lastReasoning.isEmpty {
                terminal.write("\(style("latest reasoning", "2;35")) ›\n\(renderMarkdown(lastReasoning).text)\n")
            }
        case "/blocks": listBlocks()
        case "/copy": try copyBlock(pieces)
        case "/save", "/save!": try saveBlock(pieces, overwrite: command == "/save!")
        case "/open": try openBlock(pieces)
        case "/images": listImages()
        case "/image": try showImage(pieces)
        case "/attach": try attach(pieces)
        case "/export", "/export!": try exportSession(pieces, overwrite: command == "/export!")
        case "/history": try listHistory()
        case "/search": try searchSessions(pieces)
        case "/resume": try resumeSession(pieces)
        case "/retry":
            guard let lastInput else { terminal.write("Nothing to retry.\n"); return false }
            if session.messages.last?.role == "assistant" { session.messages.removeLast() }
            if session.messages.last?.role == "user" { session.messages.removeLast() }
            try await send(lastInput, signalMonitor: signalMonitor)
        case "/edit":
            guard let lastInput else { terminal.write("Nothing to edit.\n"); return false }
            if let revised = readInput(initial: lastInput, signalMonitor: signalMonitor), !revised.isEmpty {
                if session.messages.last?.role == "assistant" { session.messages.removeLast() }
                if session.messages.last?.role == "user" { session.messages.removeLast() }
                try await send(revised, signalMonitor: signalMonitor)
            }
        case "/theme":
            guard let value = pieces.dropFirst().first, let theme = TerminalMarkdownRenderer.Theme(rawValue: value) else {
                terminal.write("Themes: \(TerminalMarkdownRenderer.Theme.allCases.map(\.rawValue).joined(separator: ", "))\n"); return false
            }
            renderer = TerminalMarkdownRenderer(color: capabilities.color, theme: theme)
            terminal.write("Theme changed to \(theme.rawValue).\n")
        default: terminal.write("Unknown command. Use /help.\n")
        }
        return false
    }

    private func makeUserMessage(_ input: String) throws -> Message {
        guard !attachments.isEmpty else { return Message(role: "user", content: input) }
        var parts = [ContentPart(type: "text", text: input)]
        for url in attachments {
            let ext = url.pathExtension.lowercased()
            if ["png", "jpg", "jpeg", "gif", "webp", "heic"].contains(ext) {
                let data = try Data(contentsOf: url)
                let mime = ext == "jpg" || ext == "jpeg" ? "image/jpeg" : "image/\(ext)"
                parts.append(ContentPart(type: "image_url", image_url: ImageURL(url: "data:\(mime);base64,\(data.base64EncodedString())", detail: "auto")))
            } else {
                let data = try Data(contentsOf: url)
                guard data.count <= 2_000_000, let text = String(data: data, encoding: .utf8) else {
                    throw TUIArtifactError.invalidPath("Attachment must be an image or UTF-8 text file under 2 MB: \(url.path)")
                }
                parts.append(ContentPart(type: "text", text: "\n<attachment path=\"\(url.lastPathComponent)\">\n\(text)\n</attachment>"))
            }
        }
        return Message(role: "user", content: .parts(parts))
    }

    private func drawWelcome() {
        let colorNote = capabilities.color ? "color" : "no-color"
        terminal.write("\(style("AFM Terminal Chat", "1;36"))  \(configuration.backendName) · \(configuration.modelName)\n")
        terminal.write(style("Terminal: \(capabilities.terminalProgram) · \(colorNote) · Enter sends · Esc+Enter adds a line · Ctrl-C cancels", "2") + "\n")
        terminal.write("Type \(style("/help", "36")) for commands. Model output is never executed automatically.\n\n")
    }

    private func drawHelp() {
        terminal.write("""

        /new /clear /quit                 session controls
        /retry /edit                      regenerate or edit the previous prompt
        /reasoning [show|hide]            toggle reasoning panels
        /status /history /search <text>   runtime and persisted-session search
        /resume <session-id>              resume a saved session
        /blocks /copy <n>                 list/copy response code blocks
        /save[!] <n> <path>               save block; ! explicitly permits overwrite
        /open <n>                         explicitly preview HTML/JS in the browser
        /attach <path>                    attach an image or UTF-8 text file
        /images /image <n>                list/display images (inline, or Quick Look fallback)
        /export[!] <path>                 export this transcript as Markdown
        /theme <auto|dark|light|mono>     change terminal rendering

        Arrow keys edit/navigate prompt history. Esc+Enter inserts a newline.
        During generation Ctrl-C cancels only that response; Ctrl-C at an empty prompt exits.

        """)
    }

    private func listBlocks() {
        guard !codeBlocks.isEmpty else { terminal.write("No code blocks in the latest response.\n"); return }
        for (index, block) in codeBlocks.enumerated() {
            terminal.write("[\(index + 1)] \(block.language.isEmpty ? "code" : block.language) · \(block.content.split(separator: "\n", omittingEmptySubsequences: false).count) lines\n")
        }
    }

    private func copyBlock(_ pieces: [String]) throws {
        let block = try selectedBlock(pieces)
        try TUIArtifactActions.copyToClipboard(block.content)
        terminal.write("Copied code block to the clipboard.\n")
    }

    private func saveBlock(_ pieces: [String], overwrite: Bool) throws {
        guard pieces.count >= 3 else { throw TUIArtifactError.invalidPath("usage: /save[!] <block> <path>") }
        let block = try selectedBlock(pieces)
        let url = try TUIArtifactActions.resolvedURL(pieces.dropFirst(2).joined(separator: " "))
        try TUIArtifactActions.save(Data(block.content.utf8), to: url, overwrite: overwrite)
        terminal.write("Saved \(url.path)\n")
    }

    private func openBlock(_ pieces: [String]) throws {
        let url = try TUIArtifactActions.openInBrowser(try selectedBlock(pieces))
        terminal.write("Opened explicit browser preview: \(url.path)\n")
    }

    private func selectedBlock(_ pieces: [String]) throws -> TUICodeBlock {
        guard pieces.count >= 2, let number = Int(pieces[1]), codeBlocks.indices.contains(number - 1) else {
            throw TUIArtifactError.invalidPath("Choose a block number from /blocks")
        }
        return codeBlocks[number - 1]
    }

    private func listImages() {
        guard !images.isEmpty else { terminal.write("No local image paths in the latest response.\n"); return }
        for (index, image) in images.enumerated() { terminal.write("[\(index + 1)] \(image.alt): \(image.path)\n") }
    }

    private func showImage(_ pieces: [String]) throws {
        guard pieces.count >= 2, let number = Int(pieces[1]), images.indices.contains(number - 1) else {
            throw TUIArtifactError.invalidPath("Choose an image number from /images")
        }
        let image = images[number - 1]
        if let sequence = try TUIArtifactActions.inlineImageSequence(path: image.path, capabilities: capabilities) {
            terminal.write(sequence + "\n")
        } else {
            try TUIArtifactActions.quickLook(image.path)
            terminal.write("Opened image with Quick Look (inline images are unavailable in \(capabilities.terminalProgram)).\n")
        }
    }

    private func presentImages(_ values: [TUIImageReference]) {
        for (index, image) in values.enumerated() {
            if let sequence = try? TUIArtifactActions.inlineImageSequence(path: image.path, capabilities: capabilities) {
                terminal.write("\n" + sequence + "\n")
            } else {
                terminal.write(style("Image \(index + 1): \(image.path) — use /image \(index + 1) for Quick Look", "2;36") + "\n")
            }
        }
    }

    private func attach(_ pieces: [String]) throws {
        guard pieces.count >= 2 else { throw TUIArtifactError.invalidPath("usage: /attach <path>") }
        let url = try TUIArtifactActions.resolvedURL(pieces.dropFirst().joined(separator: " "))
        guard FileManager.default.fileExists(atPath: url.path) else { throw TUIArtifactError.invalidPath(url.path) }
        attachments.append(url); terminal.write("Attached \(url.lastPathComponent). It will be sent with the next prompt.\n")
    }

    private func exportSession(_ pieces: [String], overwrite: Bool) throws {
        guard pieces.count >= 2 else { throw TUIArtifactError.invalidPath("usage: /export[!] <path>") }
        let url = try TUIArtifactActions.resolvedURL(pieces.dropFirst().joined(separator: " "))
        try store.exportMarkdown(session, to: url, overwrite: overwrite)
        terminal.write("Exported \(url.path)\n")
    }

    private func listHistory() throws {
        for value in try store.recent() { terminal.write("\(value.id.uuidString)  \(value.title)\n") }
    }

    private func searchSessions(_ pieces: [String]) throws {
        guard pieces.count >= 2 else { terminal.write("usage: /search <text>\n"); return }
        for value in try store.search(pieces.dropFirst().joined(separator: " ")) {
            terminal.write("\(value.id.uuidString)  \(value.title)\n")
        }
    }

    private func resumeSession(_ pieces: [String]) throws {
        guard pieces.count == 2, let id = UUID(uuidString: pieces[1]) else { throw TUIArtifactError.invalidPath("usage: /resume <session-id>") }
        session = try store.load(id: id)
        promptHistory = session.messages.filter { $0.role == "user" }.map(\.textContent)
        lastInput = promptHistory.last
        let lastAssistantIndex = session.messages.indices.reversed().first { session.messages[$0].role == "assistant" }
        lastReasoning = lastAssistantIndex.flatMap { session.reasoning(atMessageIndex: $0) } ?? ""
        codeBlocks = []
        images = []
        terminal.clearScreen(); drawWelcome()
        for message in session.messages where message.role != "system" {
            terminal.write("\(style(message.role, message.role == "user" ? "1;34" : "1;32")) › \(renderMarkdown(message.textContent).text)\n\n")
        }
    }

    private func splitCommand(_ input: String) -> [String] {
        var result: [String] = [], current = "", quote: Character?
        for character in input {
            if character == "\"" || character == "'" {
                if quote == character { quote = nil } else if quote == nil { quote = character } else { current.append(character) }
            } else if character.isWhitespace && quote == nil {
                if !current.isEmpty { result.append(current); current = "" }
            } else { current.append(character) }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private func displayRows(_ input: String, width: Int) -> Int {
        let withoutLinks = input.replacingOccurrences(
            of: #"\u001B\][^\u0007]*(?:\u0007|\u001B\\)"#,
            with: "",
            options: .regularExpression
        )
        let plain = withoutLinks.replacingOccurrences(
            of: #"\u001B\[[0-?]*[ -/]*[@-~]"#,
            with: "",
            options: .regularExpression
        )
        let columns = max(1, width)
        return max(1, plain.split(separator: "\n", omittingEmptySubsequences: false).reduce(0) {
            let cells = String($1).unicodeScalars.reduce(0) { total, scalar in
                total + max(0, Int(wcwidth(wchar_t(scalar.value))))
            }
            return $0 + max(1, (cells + columns - 1) / columns)
        })
    }

    private func renderMarkdown(_ source: String) -> MarkdownRenderResult {
        renderer.render(
            source,
            width: max(24, terminal.width() - 4),
            hyperlinks: capabilities.hyperlinks
        )
    }

    private static func initialMessages(for configuration: TerminalChatConfiguration) -> [Message] {
        if case .mlx = configuration.backend {
            return [Message(role: "system", content: configuration.engine.instructions)]
        }
        return []
    }

    private func style(_ value: String, _ code: String) -> String {
        capabilities.color ? "\u{001B}[\(code)m\(value)\u{001B}[0m" : value
    }
}
