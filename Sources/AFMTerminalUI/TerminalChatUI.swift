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

struct TUIUserTurn: Sendable {
    let input: String
    let message: Message

    init(input: String, message: Message) {
        self.input = input
        self.message = message
    }

    init?(message: Message) {
        guard message.role == "user", let content = message.content else { return nil }
        self.message = message
        switch content {
        case .text(let text):
            input = text
        case .parts(let parts):
            input = parts.first { $0.type == "text" }?.text ?? ""
        }
    }

    func replacingInput(with revisedInput: String) -> TUIUserTurn {
        let revisedContent: MessageContent?
        switch message.content {
        case .some(.text):
            revisedContent = .text(revisedInput)
        case .some(.parts(let parts)):
            var replacedPrimaryText = false
            var revisedParts = parts.map { part in
                guard !replacedPrimaryText, part.type == "text" else { return part }
                replacedPrimaryText = true
                return ContentPart(
                    type: part.type,
                    text: revisedInput,
                    image_url: part.image_url,
                    input_audio: part.input_audio
                )
            }
            if !replacedPrimaryText {
                revisedParts.insert(ContentPart(type: "text", text: revisedInput), at: 0)
            }
            revisedContent = .parts(revisedParts)
        case .none:
            revisedContent = .text(revisedInput)
        }
        return TUIUserTurn(
            input: revisedInput,
            message: Message(
                role: message.role,
                content: revisedContent,
                toolCalls: message.toolCalls,
                toolCallId: message.toolCallId,
                name: message.name
            )
        )
    }
}

public enum TUIInvocationPolicy {
    public static func validate(
        tui: Bool,
        webUI: Bool,
        singlePrompt: Bool,
        telegramOptions: Bool = false,
        inputIsTTY: Bool,
        outputIsTTY: Bool
    ) throws {
        guard tui else { return }
        if webUI { throw TUIInvocationError.conflict("--tui cannot be combined with --webui") }
        if singlePrompt { throw TUIInvocationError.conflict("--tui cannot be combined with --single-prompt") }
        if telegramOptions { throw TUIInvocationError.conflict("--tui cannot be combined with Telegram server options") }
        if !inputIsTTY { throw TUIInvocationError.conflict("--tui requires interactive terminal input") }
        if !outputIsTTY { throw TUIInvocationError.conflict("--tui requires interactive terminal output") }
    }

    public static func hasTelegramOptions(
        botToken: String?,
        allowlist: String?,
        replyFormat: String?,
        requirePrefix: String?
    ) -> Bool {
        botToken != nil || allowlist != nil || replyFormat != nil || requirePrefix != nil
    }
}

public enum TUIInvocationError: Error, LocalizedError, Equatable {
    case conflict(String)
    public var errorDescription: String? { if case .conflict(let value) = self { return value }; return nil }
}

public struct TUILogprobConfiguration: Equatable, Sendable {
    public let enabled: Bool
    public let maximum: Int?

    public init(maximum: Int?) {
        self.enabled = maximum != nil
        self.maximum = maximum
    }
}

public enum TUIReasoningDisplayMode: String, Equatable, Sendable {
    case collapsed
    case expanded
    case hidden

    mutating func togglePanel() {
        self = self == .expanded ? .collapsed : .expanded
    }
}

enum TUIGenerationPhase: Equatable, Sendable {
    case preparing
    case reasoning
    case answering
    case usingTools
    case completed
}

enum TUIGenerationDisposition: Equatable, Sendable {
    case accepted
    case cancelled
    case incomplete(AFMFinishReason?)
}

enum TUIActivityIndicator {
    private static let unicodeFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    private static let asciiFrames = ["|", "/", "-", "\\"]

    static func symbol(frame: Int, unicode: Bool) -> String {
        let frames = unicode ? unicodeFrames : asciiFrames
        return frames[abs(frame) % frames.count]
    }
}

struct GenerationSnapshot: Sendable {
    let revision: UInt64
    let text: String
    let reasoning: String
    let textTail: String
    let reasoningTail: String
    let textCharacterCount: Int
    let reasoningCharacterCount: Int
    let tools: [AFMToolCall]
    let toolDisplayLines: [String]
    let promptTokens: Int
    let completionTokens: Int
    let cachedTokens: Int
    let phase: TUIGenerationPhase
    let reasoningDuration: TimeInterval?
    let finishReason: AFMFinishReason?
    let completed: Bool
    let cancelled: Bool
    let error: String?
}

actor GenerationBuffer {
    static let renderTailLimit = 12_000
    private var text = ""
    private var reasoning = ""
    private var textTail = ""
    private var reasoningTail = ""
    private var textCharacterCount = 0
    private var reasoningCharacterCount = 0
    private var tools: [AFMToolCall] = []
    private var toolStages: [AFMToolCallStage] = []
    private var toolDisplayLines: [String] = []
    private var promptTokens = 0
    private var completionTokens = 0
    private var cachedTokens = 0
    private var currentPhase: TUIGenerationPhase = .preparing
    private var reasoningStartedAt: Date?
    private var reasoningFinishedAt: Date?
    private var finishReason: AFMFinishReason?
    private var completed = false
    private var cancelled = false
    private var error: String?
    private var revision: UInt64 = 0

    func accept(_ event: AFMStreamEvent) {
        guard !completed else { return }
        switch event {
        case .text(let value, let tokens):
            finishReasoningIfNeeded()
            currentPhase = .answering
            let safe = TerminalOutputSanitizer.sanitize(value)
            text += safe
            textTail = Self.appendingToRenderTail(textTail, safe)
            textCharacterCount += safe.count
            completionTokens = max(completionTokens, tokens)
        case .reasoning(let value, _):
            if reasoningStartedAt == nil { reasoningStartedAt = Date() }
            currentPhase = .reasoning
            let safe = TerminalOutputSanitizer.sanitize(value)
            reasoning += safe
            reasoningTail = Self.appendingToRenderTail(reasoningTail, safe)
            reasoningCharacterCount += safe.count
        case .toolCall(let call, let stage):
            finishReasoningIfNeeded()
            currentPhase = .usingTools
            let safeCall = AFMToolCall(
                id: TerminalOutputSanitizer.sanitize(call.id),
                name: TerminalOutputSanitizer.sanitize(call.name),
                arguments: TerminalOutputSanitizer.sanitize(call.arguments)
            )
            if let index = tools.firstIndex(where: { $0.id == safeCall.id }) {
                tools[index] = safeCall
                toolStages[index] = stage
                toolDisplayLines[index] = Self.toolDisplayLine(call: safeCall, stage: stage)
            } else {
                tools.append(safeCall)
                toolStages.append(stage)
                toolDisplayLines.append(Self.toolDisplayLine(call: safeCall, stage: stage))
            }
        case .usage(let prompt, let completion, let cached):
            promptTokens = prompt; completionTokens = completion; cachedTokens = cached
        case .completed(let reason):
            finishReasoningIfNeeded()
            finishReason = reason
            completed = true
        case .tokenLogprobs, .metadata, .custom: break
        }
        revision &+= 1
    }

    func accept(_ response: AFMResponse) {
        guard !completed else { return }
        text = TerminalOutputSanitizer.sanitize(response.content)
        reasoning = TerminalOutputSanitizer.sanitize(response.reasoningContent ?? "")
        if !reasoning.isEmpty {
            reasoningStartedAt = Date()
            reasoningFinishedAt = reasoningStartedAt
        }
        textCharacterCount = text.count
        reasoningCharacterCount = reasoning.count
        textTail = String(text.suffix(Self.renderTailLimit))
        reasoningTail = String(reasoning.suffix(Self.renderTailLimit))
        promptTokens = response.promptTokens
        completionTokens = response.completionTokens
        cachedTokens = response.cachedPromptTokens
        finishReason = response.finishReason
        for call in response.toolCalls ?? [] {
            let safeCall = AFMToolCall(
                id: TerminalOutputSanitizer.sanitize(call.id),
                name: TerminalOutputSanitizer.sanitize(call.function.name),
                arguments: TerminalOutputSanitizer.sanitize(call.function.arguments)
            )
            tools.append(safeCall)
            toolStages.append(.completed)
            toolDisplayLines.append(Self.toolDisplayLine(call: safeCall, stage: .completed))
        }
        completed = true
        revision &+= 1
    }

    func fail(_ value: Error) {
        guard !completed else { return }
        finishReasoningIfNeeded()
        error = TerminalOutputSanitizer.sanitize(value.localizedDescription)
        finishReason = .error
        completed = true; revision &+= 1
    }
    func cancel() {
        guard !completed else { return }
        finishReasoningIfNeeded()
        cancelled = true; finishReason = .cancelled; completed = true; revision &+= 1
    }
    func finish() {
        guard !completed else { return }
        finishReasoningIfNeeded()
        finishReason = .unknown; completed = true; revision &+= 1
    }
    func snapshot() -> GenerationSnapshot {
        makeSnapshot(includeFullContent: true)
    }

    func renderSnapshot() -> GenerationSnapshot {
        makeSnapshot(includeFullContent: false)
    }

    private func makeSnapshot(includeFullContent: Bool) -> GenerationSnapshot {
        let phase: TUIGenerationPhase = completed ? .completed : currentPhase
        let reasoningDuration = reasoningStartedAt.map {
            (reasoningFinishedAt ?? Date()).timeIntervalSince($0)
        }
        return GenerationSnapshot(
            revision: revision,
            text: includeFullContent ? text : "",
            reasoning: includeFullContent ? reasoning : "",
            textTail: textTail, reasoningTail: reasoningTail,
            textCharacterCount: textCharacterCount,
            reasoningCharacterCount: reasoningCharacterCount,
            tools: includeFullContent ? tools : [],
            toolDisplayLines: toolDisplayLines,
            promptTokens: promptTokens, completionTokens: completionTokens,
            cachedTokens: cachedTokens, phase: phase,
            reasoningDuration: reasoningDuration, finishReason: finishReason,
            completed: completed,
            cancelled: cancelled, error: error
        )
    }

    private func finishReasoningIfNeeded() {
        if reasoningStartedAt != nil, reasoningFinishedAt == nil {
            reasoningFinishedAt = Date()
        }
    }

    func snapshot(ifChangedSince priorRevision: UInt64) -> GenerationSnapshot? {
        guard revision != priorRevision else { return nil }
        return renderSnapshot()
    }

    private static func appendingToRenderTail(_ tail: String, _ value: String) -> String {
        var updated = tail
        updated += value
        guard updated.count > renderTailLimit else { return updated }
        return String(updated.suffix(renderTailLimit))
    }

    private static func toolDisplayLine(call: AFMToolCall, stage: AFMToolCallStage) -> String {
        let name = bounded(call.name, limit: 256)
        let arguments = bounded(call.arguments, limit: 2_048)
        let stageLabel: String
        switch stage {
        case .started: stageLabel = "started"
        case .argumentsDelta: stageLabel = "arguments"
        case .completed: stageLabel = "completed"
        case .retracted: stageLabel = "retracted"
        }
        return "\(stageLabel): \(name)(\(arguments))"
    }

    private static func bounded(_ value: String, limit: Int) -> String {
        let prefix = String(value.prefix(limit + 1))
        return prefix.count > limit ? String(prefix.prefix(limit)) + "…" : prefix
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
    private let monitoredSignals = [SIGINT, SIGTERM, SIGHUP]
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
    private var reasoningDisplayMode: TUIReasoningDisplayMode
    private var promptHistory: [String] = []
    private var codeBlocks: [TUICodeBlock] = []
    private var images: [TUIImageReference] = []
    private var attachments: [URL] = []
    private var lastStatistics = "No generation yet"
    private var lastUserTurn: TUIUserTurn?
    private var lastReasoning = ""
    private var lastInputWasRolledBack = false

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
        self.reasoningDisplayMode = configuration.showReasoning ? .expanded : .collapsed
        self.attachments = configuration.initialAttachments
    }

    public func run() async throws {
        guard capabilities.isInteractive else {
            throw TUIInvocationError.conflict("--tui requires an interactive terminal")
        }
        for attachment in configuration.initialAttachments {
            try TUIMediaAttachmentPolicy.validate(attachment)
        }
        try terminal.enter()
        terminal.enterAlternateScreen()
        let signalMonitor = TUISignalMonitor()
        defer {
            signalMonitor.stop()
            terminal.restore()
        }

        do {
            drawWelcome()
            terminal.write("Loading \(configuration.modelName)…\n")
            if let modelID = try await loadModel(signalMonitor: signalMonitor) {
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
                    try await send(try makeUserTurn(input), signalMonitor: signalMonitor)
                }
            }
        } catch {
            signalMonitor.stop()
            terminal.restore()
            await engine.unload()
            throw error
        }
        signalMonitor.stop()
        terminal.restore()
        await engine.unload()
    }

    private func loadModel(signalMonitor: TUISignalMonitor) async throws -> String? {
        let state = GenerationTaskState()
        let task = Task<String, Error> {
            do {
                let modelID = try await engine.load { [terminal] progress in
                    terminal.clearLine()
                    terminal.write(String(format: "Loading model %3.0f%%", progress * 100))
                }
                await state.markFinished()
                return modelID
            } catch {
                await state.markFinished()
                throw error
            }
        }
        var interrupted = false
        while !(await state.isFinished()) {
            if signalMonitor.shouldTerminate {
                interrupted = true
                task.cancel()
                break
            }
            if let key = terminal.readKey(timeoutMilliseconds: 40), key == .interrupt || key == .eof {
                interrupted = true
                task.cancel()
                break
            }
        }
        if interrupted {
            signalMonitor.stop()
            terminal.restore()
            _ = try? await task.value
            if !signalMonitor.shouldTerminate {
                terminal.clearLine()
                terminal.write("Loading cancelled.\n")
            }
            return nil
        }
        return try await task.value
    }

    private func send(_ userTurn: TUIUserTurn, signalMonitor: TUISignalMonitor) async throws {
        let input = userTurn.input
        lastInputWasRolledBack = false
        promptHistory.append(input)
        lastUserTurn = userTurn
        attachments.removeAll()
        if !session.messages.contains(where: { $0.role == "user" }) {
            session.title = String(input.replacingOccurrences(of: "\n", with: " ").prefix(72))
        }
        session.messages.append(userTurn.message)
        let pendingUserIndex = session.messages.index(before: session.messages.endIndex)
        session.updatedAt = Date()
        terminal.write("\n\(style("you", "1;34")) › \(input)\n\n")

        let buffer = GenerationBuffer()
        let messages = Self.requestMessages(for: configuration.backend, transcript: session.messages)
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
        }

        var previousRows = 0
        var lastSnapshot = await buffer.renderSnapshot()
        let start = Date()
        var activityFrame = 0
        var nextActivityRedraw = Date.distantPast
        while !lastSnapshot.completed && !signalMonitor.shouldTerminate {
            var forceRedraw = false
            if let key = terminal.readKey(timeoutMilliseconds: 40) {
                switch key {
                case .interrupt:
                    task.cancel()
                    await buffer.cancel()
                case .tab:
                    reasoningDisplayMode.togglePanel()
                    forceRedraw = true
                default:
                    break
                }
            }
            if let changed = await buffer.snapshot(ifChangedSince: lastSnapshot.revision) {
                lastSnapshot = changed
                forceRedraw = true
            }
            if lastSnapshot.completed { break }
            let now = Date()
            if forceRedraw || now >= nextActivityRedraw {
                if !forceRedraw { lastSnapshot = await buffer.renderSnapshot() }
                previousRows = redrawGeneration(
                    lastSnapshot,
                    previousRows: previousRows,
                    final: false,
                    activityFrame: activityFrame,
                    elapsed: now.timeIntervalSince(start)
                )
                activityFrame &+= 1
                nextActivityRedraw = now.addingTimeInterval(0.12)
            }
        }
        if signalMonitor.shouldTerminate {
            signalMonitor.stop()
            terminal.restore()
            task.cancel()
            await buffer.cancel()
        }
        await task.value
        lastSnapshot = await buffer.snapshot()
        if signalMonitor.shouldTerminate {
            session.removeMessage(at: pendingUserIndex)
            lastInputWasRolledBack = true
            try? await engine.resetConversation(with: session.messages)
            return
        }
        _ = redrawGeneration(
            lastSnapshot,
            previousRows: previousRows,
            final: true,
            activityFrame: activityFrame,
            elapsed: Date().timeIntervalSince(start)
        )
        terminal.write("\n")

        if let error = lastSnapshot.error {
            session.removeMessage(at: pendingUserIndex)
            lastInputWasRolledBack = true
            try? await engine.resetConversation(with: session.messages)
            terminal.write("\(style("error", "1;31")): \(error)\n\n")
            return
        }
        lastReasoning = lastSnapshot.reasoning
        switch Self.generationDisposition(for: lastSnapshot) {
        case .cancelled:
            session.removeMessage(at: pendingUserIndex)
            lastInputWasRolledBack = true
            try? await engine.resetConversation(with: session.messages)
            terminal.write("\(style("cancelled", "33")) — the partial response was not added.\n\n")
            return
        case .incomplete(let reason):
            session.removeMessage(at: pendingUserIndex)
            lastInputWasRolledBack = true
            try? await engine.resetConversation(with: session.messages)
            let notice: String
            if reason == .length {
                notice = "token limit reached before the model produced an answer"
            } else if reason == .contentFilter {
                notice = "generation stopped by the content filter before producing an answer"
            } else {
                notice = "model stopped before producing an answer"
            }
            terminal.write("\(style("incomplete", "1;33")) — \(notice). The turn was not added; use /reasoning last, then /edit the prompt or restart with a higher --max-tokens value.\n\n")
            return
        case .accepted:
            break
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
        reportPersistenceResult(store.persistRecoveringSession(session))
        let rendered = renderMarkdown(lastSnapshot.text)
        codeBlocks = rendered.codeBlocks
        images = rendered.images
        let elapsed = max(0.001, Date().timeIntervalSince(start))
        lastStatistics = Self.turnStatistics(
            backend: configuration.backend,
            requestMessages: messages,
            responseText: lastSnapshot.text,
            promptTokens: lastSnapshot.promptTokens,
            completionTokens: lastSnapshot.completionTokens,
            cachedTokens: lastSnapshot.cachedTokens,
            elapsed: elapsed
        )
        terminal.write("\(style("↳ \(lastStatistics)", "2"))\n")
        if !codeBlocks.isEmpty { terminal.write("\(style("\(codeBlocks.count) code block(s): /blocks, /copy, /save, /open", "2;36"))\n") }
        if !images.isEmpty { presentImages(images) }
        terminal.write("\n")
    }

    static func generationDisposition(for snapshot: GenerationSnapshot) -> TUIGenerationDisposition {
        if snapshot.cancelled || snapshot.finishReason == .cancelled {
            return .cancelled
        }
        if snapshot.text.isEmpty && snapshot.tools.isEmpty {
            return .incomplete(snapshot.finishReason)
        }
        return .accepted
    }

    static func turnStatistics(
        backend: AFMBackend,
        requestMessages: [Message],
        responseText: String,
        promptTokens: Int,
        completionTokens: Int,
        cachedTokens: Int,
        elapsed: TimeInterval
    ) -> String {
        let duration = max(0.001, elapsed)
        if case .foundationModels = backend {
            let estimatedInput = estimatedTokenCount(
                requestMessages.map(\.textContent).joined(separator: " ")
            )
            let estimatedOutput = estimatedTokenCount(responseText)
            let rate = Double(estimatedOutput) / duration
            return String(
                format: "~%d input · ~%d generated · %.2fs · ~%.1f tok/s · estimated",
                estimatedInput, estimatedOutput, duration, rate
            )
        }
        let rate = Double(completionTokens) / duration
        return String(
            format: "%d prompt · %d cached · %d generated · %.2fs · %.1f tok/s",
            promptTokens, cachedTokens, completionTokens, duration, rate
        )
    }

    static func estimatedTokenCount(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        let words = text.split(whereSeparator: \.isWhitespace).count
        let characterEstimate = Double(text.count) / 4.0
        let wordEstimate = Double(words) / 0.75
        return max(1, Int(ceil(max(characterEstimate, wordEstimate))))
    }

    private func reportPersistenceResult(_ result: TUISessionPersistenceResult) {
        guard let notice = Self.persistenceNotice(for: result) else { return }
        terminal.write(TerminalOutputSanitizer.sanitize(notice))
    }

    static func persistenceNotice(for result: TUISessionPersistenceResult) -> String? {
        switch result {
        case .saved:
            return nil
        case .recovered(let saveError, let recoveryURL):
            return """
            warning: Session save failed: \(saveError)
            Full recovery session: \(recoveryURL.path)
            This bounded JSON recovery preserves multimodal content and tool calls. Use /export <path> for a separate text copy.

            """
        case .failed(let saveError, let recoveryError):
            return """
            error: Session save failed: \(saveError)
            Automatic recovery export also failed: \(recoveryError)
            This chat remains in memory. Use /export <path> before quitting.

            """
        }
    }

    private func redrawGeneration(
        _ snapshot: GenerationSnapshot,
        previousRows: Int,
        final: Bool,
        activityFrame: Int,
        elapsed: TimeInterval
    ) -> Int {
        if previousRows > 0 {
            for index in 0..<previousRows {
                terminal.clearLine()
                if index < previousRows - 1 { terminal.write("\u{001B}[1A") }
            }
            terminal.write("\r")
        }
        var display = ""
        if snapshot.reasoningCharacterCount > 0 {
            display += reasoningPanel(
                snapshot,
                final: final,
                activityFrame: activityFrame
            ) + "\n"
        } else if !final && snapshot.phase == .preparing {
            display += activityLine(
                label: "Preparing context",
                frame: activityFrame,
                elapsed: elapsed
            ) + "\n"
        }
        if !final && snapshot.phase == .answering {
            display += activityLine(label: "Writing answer", frame: activityFrame, elapsed: elapsed) + "\n"
        } else if !final && snapshot.phase == .usingTools {
            display += activityLine(label: "Using tools", frame: activityFrame, elapsed: elapsed) + "\n"
        }
        if !final || snapshot.textCharacterCount > 0 || !snapshot.toolDisplayLines.isEmpty {
            display += style("assistant", "1;32") + " › " + renderGenerationMarkdown(
                full: snapshot.text,
                tail: snapshot.textTail,
                characterCount: snapshot.textCharacterCount,
                final: final
            )
        }
        if !snapshot.toolDisplayLines.isEmpty {
            display += "\n" + style(snapshot.toolDisplayLines.joined(separator: "\n"), "36")
        }
        terminal.write(display)
        return displayRows(display, width: terminal.width())
    }

    private func reasoningPanel(
        _ snapshot: GenerationSnapshot,
        final: Bool,
        activityFrame: Int
    ) -> String {
        let isActive = !final && snapshot.phase == .reasoning
        let duration = Self.formattedDuration(snapshot.reasoningDuration ?? 0)
        let size = Self.compactCount(snapshot.reasoningCharacterCount)
        let completedStatus: String
        if snapshot.finishReason == .length && snapshot.textCharacterCount == 0 {
            completedStatus = "◇ Reasoning — token limit"
        } else if snapshot.cancelled || snapshot.finishReason == .cancelled {
            completedStatus = "◇ Reasoning — interrupted"
        } else {
            completedStatus = "◇ Reasoning ✓"
        }
        let status = isActive
            ? "◆ Reasoning \(activitySymbol(activityFrame))"
            : completedStatus
        let hint = isActive
            ? (reasoningDisplayMode == .expanded ? "Tab collapse" : "Tab expand")
            : "/reasoning last"
        let summary = "\(status)  \(duration) · \(size) chars  [\(hint)]"

        switch reasoningDisplayMode {
        case .hidden:
            let hiddenStatus = isActive
                ? "◆ Reasoning \(activitySymbol(activityFrame))"
                : completedStatus
            return style("\(hiddenStatus) (hidden)", "2;35")
        case .collapsed:
            return style(summary, isActive ? "35" : "2;35")
        case .expanded:
            let rendered = renderGenerationMarkdown(
                full: snapshot.reasoning,
                tail: snapshot.reasoningTail,
                characterCount: snapshot.reasoningCharacterCount,
                final: final
            )
            let body = rendered.split(separator: "\n", omittingEmptySubsequences: false)
                .map { style("│", "2;35") + " " + String($0) }
                .joined(separator: "\n")
            return style("╭─ \(summary)", isActive ? "35" : "2;35")
                + "\n" + body + "\n" + style("╰─", "2;35")
        }
    }

    private func activityLine(label: String, frame: Int, elapsed: TimeInterval) -> String {
        style("◆ \(label)  \(activitySymbol(frame))  \(Self.formattedDuration(elapsed))  [Ctrl-C cancel]", "2;36")
    }

    private func activitySymbol(_ frame: Int) -> String {
        TUIActivityIndicator.symbol(
            frame: frame,
            unicode: capabilities.terminalProgram != "dumb"
        )
    }

    private static func formattedDuration(_ value: TimeInterval) -> String {
        String(format: "%.1fs", max(0, value))
    }

    private static func compactCount(_ value: Int) -> String {
        value >= 1_000 ? String(format: "%.1fk", Double(value) / 1_000) : "\(value)"
    }

    private func renderGenerationMarkdown(
        full: String,
        tail: String,
        characterCount: Int,
        final: Bool
    ) -> String {
        if final { return renderMarkdown(full).text }
        let renderedTail = renderMarkdown(tail).text
        guard characterCount > GenerationBuffer.renderTailLimit else { return renderedTail }
        return style("… earlier streaming output omitted from redraw …", "2") + "\n" + renderedTail
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
            try await engine.resetConversation()
            session = TUISession(
                backend: configuration.backendName,
                model: configuration.modelName,
                messages: Self.initialMessages(for: configuration)
            )
            resetTransientSessionState()
            terminal.clearScreen(); drawWelcome()
        case "/clear": terminal.clearScreen(); drawWelcome()
        case "/status": terminal.write("\(configuration.backendName) · \(configuration.modelName)\n\(lastStatistics)\n")
        case "/reasoning":
            let value = pieces.dropFirst().first?.lowercased()
            if value == "last" {
                if lastReasoning.isEmpty {
                    terminal.write("No reasoning is available for the latest response.\n")
                } else {
                    terminal.write("\(style("latest reasoning", "2;35")) ›\n\(renderMarkdown(lastReasoning).text)\n")
                }
                return false
            }
            switch value {
            case "show", "on", "expanded", "expand": reasoningDisplayMode = .expanded
            case "collapsed", "collapse": reasoningDisplayMode = .collapsed
            case "hide", "hidden", "off": reasoningDisplayMode = .hidden
            case nil: reasoningDisplayMode.togglePanel()
            default:
                terminal.write("Usage: /reasoning [collapsed|expanded|off|last]\n")
                return false
            }
            terminal.write("Reasoning display: \(reasoningDisplayMode.rawValue).\n")
            if reasoningDisplayMode == .expanded, !lastReasoning.isEmpty {
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
        case "/resume": try await resumeSession(pieces)
        case "/retry":
            guard let lastUserTurn else { terminal.write("Nothing to retry.\n"); return false }
            if !lastInputWasRolledBack { session.removeLastExchange() }
            try await engine.resetConversation(with: session.messages)
            try await send(lastUserTurn, signalMonitor: signalMonitor)
        case "/edit":
            guard let lastUserTurn else { terminal.write("Nothing to edit.\n"); return false }
            if let revised = readInput(initial: lastUserTurn.input, signalMonitor: signalMonitor), !revised.isEmpty {
                if !lastInputWasRolledBack { session.removeLastExchange() }
                try await engine.resetConversation(with: session.messages)
                try await send(lastUserTurn.replacingInput(with: revised), signalMonitor: signalMonitor)
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

    private func makeUserTurn(_ input: String) throws -> TUIUserTurn {
        try Self.makeUserTurn(input, attachments: attachments)
    }

    static func makeUserTurn(_ input: String, attachments: [URL]) throws -> TUIUserTurn {
        guard !attachments.isEmpty else {
            return TUIUserTurn(input: input, message: Message(role: "user", content: input))
        }
        var parts = [ContentPart(type: "text", text: input)]
        for url in attachments {
            let ext = url.pathExtension.lowercased()
            if TUIMediaAttachmentPolicy.kind(for: url) == .image {
                let data = try TUIArtifactActions.readRegularFile(at: url, maximumBytes: 20_000_000)
                let mime = ext == "jpg" || ext == "jpeg" ? "image/jpeg" : "image/\(ext)"
                parts.append(ContentPart(type: "image_url", image_url: ImageURL(url: "data:\(mime);base64,\(data.base64EncodedString())", detail: "auto")))
            } else if TUIMediaAttachmentPolicy.kind(for: url) == .video {
                try TUIMediaAttachmentPolicy.validate(url)
                parts.append(ContentPart(
                    type: "image_url",
                    image_url: ImageURL(url: url.absoluteString, detail: nil)
                ))
            } else {
                let data = try TUIArtifactActions.readRegularFile(at: url, maximumBytes: 2_000_000)
                guard let text = String(data: data, encoding: .utf8) else {
                    throw TUIArtifactError.invalidPath("Attachment must be an image or UTF-8 text file under 2 MB: \(url.path)")
                }
                parts.append(ContentPart(type: "text", text: "\n<attachment path=\"\(url.lastPathComponent)\">\n\(text)\n</attachment>"))
            }
        }
        return TUIUserTurn(
            input: input,
            message: Message(role: "user", content: .parts(parts))
        )
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
        /reasoning [collapsed|expanded|off|last]
                                        control reasoning panels
        /status /history /search <text>   runtime and persisted-session search
        /resume <session-id>              resume a saved session
        /blocks /copy <n>                 list/copy response code blocks
        /save[!] <n> <path>               save block; ! explicitly permits overwrite
        /open <n>                         explicitly preview HTML/JS in the browser
        /attach <path>                    attach an image, video, or UTF-8 text file
        /images /image <n>                list/display images (inline, or Quick Look fallback)
        /export[!] <path>                 export this transcript as Markdown
        /theme <auto|dark|light|mono>     change terminal rendering

        Arrow keys edit/navigate prompt history. Esc+Enter inserts a newline.
        During generation Tab expands/collapses reasoning and Ctrl-C cancels only that response.
        Ctrl-C at an empty prompt exits.

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
            terminal.write(style("Image \(index + 1): \(image.path) — use /image \(index + 1) to read and display", "2;36") + "\n")
        }
    }

    private func attach(_ pieces: [String]) throws {
        guard pieces.count >= 2 else { throw TUIArtifactError.invalidPath("usage: /attach <path>") }
        let url = try TUIArtifactActions.resolvedURL(pieces.dropFirst().joined(separator: " "))
        if TUIMediaAttachmentPolicy.kind(for: url) != nil {
            try TUIMediaAttachmentPolicy.validate(url)
        } else {
            try TUIArtifactActions.preflightRegularFile(at: url, maximumBytes: 2_000_000)
        }
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

    private func resumeSession(_ pieces: [String]) async throws {
        guard pieces.count == 2, let id = UUID(uuidString: pieces[1]) else { throw TUIArtifactError.invalidPath("usage: /resume <session-id>") }
        var restored = try store.loadBestAvailable(id: id)
        try Self.validateRestoredSession(
            restored,
            backendName: configuration.backendName,
            modelName: configuration.modelName
        )
        restored.pruneReasoningMetadata()
        try await engine.resetConversation(with: restored.messages)
        session = restored
        let restoredUserTurns = session.messages.compactMap(TUIUserTurn.init(message:))
        promptHistory = restoredUserTurns.map(\.input)
        lastUserTurn = restoredUserTurns.last
        lastInputWasRolledBack = false
        let lastAssistantIndex = session.messages.indices.reversed().first { session.messages[$0].role == "assistant" }
        lastReasoning = lastAssistantIndex.flatMap { session.reasoning(atMessageIndex: $0) } ?? ""
        codeBlocks = []
        images = []
        attachments = []
        lastStatistics = "No generation yet"
        terminal.clearScreen(); drawWelcome()
        for message in session.messages where message.role != "system" {
            terminal.write("\(style(message.role, message.role == "user" ? "1;34" : "1;32")) › \(renderMarkdown(message.textContent).text)\n\n")
        }
    }

    private func resetTransientSessionState() {
        promptHistory = []
        codeBlocks = []
        images = []
        attachments = []
        lastStatistics = "No generation yet"
        lastUserTurn = nil
        lastReasoning = ""
        lastInputWasRolledBack = false
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
            TerminalOutputSanitizer.sanitize(source),
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

    static func requestMessages(for backend: AFMBackend, transcript: [Message]) -> [Message] {
        if case .foundationModels = backend {
            return transcript.last.map { [$0] } ?? []
        }
        return transcript
    }

    static func validateRestoredSession(
        _ restored: TUISession,
        backendName: String,
        modelName: String
    ) throws {
        guard restored.backend == backendName, restored.model == modelName else {
            throw TUIArtifactError.incompatibleSession(
                savedBackend: restored.backend,
                savedModel: restored.model,
                currentBackend: backendName,
                currentModel: modelName
            )
        }
    }

    private func style(_ value: String, _ code: String) -> String {
        capabilities.color ? "\u{001B}[\(code)m\(value)\u{001B}[0m" : value
    }
}
