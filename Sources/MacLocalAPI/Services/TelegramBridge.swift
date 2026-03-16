import ArgumentParser
import CryptoKit
import Foundation

enum TelegramReplyFormat: String, Codable, CaseIterable, ExpressibleByArgument, Sendable {
    case markdown
    case plain
    case html

    var apiParseMode: String? {
        switch self {
        case .markdown:
            return "HTML"
        case .plain:
            return nil
        case .html:
            return "HTML"
        }
    }

    func render(_ text: String) -> String {
        switch self {
        case .markdown:
            return TelegramFormatter.markdownToHTML(text)
        case .plain:
            return text
        case .html:
            return TelegramFormatter.escapeHTML(text)
        }
    }
}

struct TelegramConfiguration: Sendable {
    let botToken: String
    let allowedUserIDs: Set<Int64>
    let localBaseURL: String
    let modelID: String
    let instructions: String
    let verbose: Bool
    let pollIntervalSeconds: TimeInterval
    let replyFormat: TelegramReplyFormat
    let requiredPrefix: String?

    static func parseAllowedUserIDs(_ raw: String) throws -> Set<Int64> {
        let values = raw
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !values.isEmpty else {
            throw ValidationError("--telegram-allow requires at least one comma-separated Telegram user ID")
        }

        var ids = Set<Int64>()
        for value in values {
            guard let id = Int64(value), id > 0 else {
                throw ValidationError("Invalid Telegram user ID: \(value)")
            }
            ids.insert(id)
        }
        return ids
    }
}

enum TelegramFormatter {
    static func escapeHTML(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    static func markdownToHTML(_ text: String) -> String {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n")
        let protected = protectCodeFences(in: normalized)
        let escaped = escapeHTML(protected.text)
        let linked = replace(pattern: #"\[([^\]]+)\]\((https?://[^)\s]+)\)"#, in: escaped) { match, source in
            let label = htmlUnescape(extract(match, in: source, at: 1))
            let url = htmlUnescape(extract(match, in: source, at: 2))
            return #"<a href="\#(escapeAttribute(url))">\#(escapeHTML(label))</a>"#
        }
        let bolded = replace(pattern: #"(?s)\*\*(.+?)\*\*"#, in: linked) { match, source in
            "<b>\(htmlUnescape(extract(match, in: source, at: 1)))</b>"
        }
        let underBolded = replace(pattern: #"(?s)__(.+?)__"#, in: bolded) { match, source in
            "<b>\(htmlUnescape(extract(match, in: source, at: 1)))</b>"
        }
        let italicized = replace(pattern: #"(?s)(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"#, in: underBolded) { match, source in
            "<i>\(htmlUnescape(extract(match, in: source, at: 1)))</i>"
        }
        let underItalicized = replace(pattern: #"(?s)(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"#, in: italicized) { match, source in
            "<i>\(htmlUnescape(extract(match, in: source, at: 1)))</i>"
        }
        let codeInlined = replace(pattern: #"(?s)`([^`\n]+)`"#, in: underItalicized) { match, source in
            "<code>\(htmlUnescape(extract(match, in: source, at: 1)))</code>"
        }
        let headings = codeInlined
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> String in
                let text = String(line)
                guard let range = text.range(of: #"^#{1,6}\s+"#, options: .regularExpression) else {
                    return text
                }
                return "<b>\(String(text[range.upperBound...]))</b>"
            }
            .joined(separator: "\n")
        return restoreCodeFences(in: headings, placeholders: protected.placeholders)
    }

    private static func protectCodeFences(in text: String) -> (text: String, placeholders: [String: String]) {
        let pattern = #"(?s)```([^\n`]*)\n(.*?)```"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return (text, [:])
        }
        let source = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: source.length)).reversed()
        var updated = text
        var placeholders: [String: String] = [:]
        for (index, match) in matches.enumerated() {
            let lang = extract(match, in: text, at: 1).trimmingCharacters(in: .whitespacesAndNewlines)
            let code = extract(match, in: text, at: 2)
            let html = lang.isEmpty
                ? "<pre><code>\(escapeHTML(code))</code></pre>"
                : #"<pre><code class="language-\#(escapeAttribute(lang))">\#(escapeHTML(code))</code></pre>"#
            let token = "AFM_TELEGRAM_CODE_BLOCK_\(index)_TOKEN"
            placeholders[token] = html
            if let range = Range(match.range, in: updated) {
                updated.replaceSubrange(range, with: token)
            }
        }
        return (updated, placeholders)
    }

    private static func restoreCodeFences(in text: String, placeholders: [String: String]) -> String {
        var result = text
        for (token, html) in placeholders {
            result = result.replacingOccurrences(of: token, with: html)
        }
        return result
    }

    private static func replace(pattern: String, in text: String, transform: (NSTextCheckingResult, String) -> String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let source = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: source.length)).reversed()
        var updated = text
        for match in matches {
            let replacement = transform(match, updated)
            if let range = Range(match.range, in: updated) {
                updated.replaceSubrange(range, with: replacement)
            }
        }
        return updated
    }

    private static func extract(_ match: NSTextCheckingResult, in text: String, at index: Int) -> String {
        guard let range = Range(match.range(at: index), in: text) else { return "" }
        return String(text[range])
    }

    private static func escapeAttribute(_ text: String) -> String {
        escapeHTML(text).replacingOccurrences(of: "\"", with: "&quot;")
    }

    private static func htmlUnescape(_ text: String) -> String {
        text
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&amp;", with: "&")
    }
}

enum TelegramInboundCommand: Equatable {
    case none
    case newConversation
    case prompt(String)
}

struct TelegramPolicy {
    static let commandPrefix = "/afm"
    static let newConversationCommand = "new"
    static let maxImageCount = 4
    static let maxImageBytes = 20 * 1024 * 1024
    static let maxConversationTurns = 8

    static func parseCommand(_ text: String?, requiredPrefix: String?) -> TelegramInboundCommand {
        let trimmed = (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .none }
        if let requiredPrefix, !requiredPrefix.isEmpty {
            let loweredPrefix = requiredPrefix.lowercased()
            guard trimmed.lowercased().hasPrefix(loweredPrefix) else { return .none }
            let remainder = String(trimmed.dropFirst(requiredPrefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
            if remainder.lowercased() == newConversationCommand {
                return .newConversation
            }
            return .prompt(remainder)
        }
        if trimmed.lowercased().hasPrefix(commandPrefix) {
            let remainder = String(trimmed.dropFirst(commandPrefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
            if remainder.lowercased() == newConversationCommand {
                return .newConversation
            }
            return .prompt(remainder)
        }
        return .prompt(trimmed)
    }

    static func isPrivateChat(_ type: String) -> Bool {
        type.caseInsensitiveCompare("private") == .orderedSame
    }

    static func isAllowedUser(_ id: Int64, allowedUserIDs: Set<Int64>) -> Bool {
        allowedUserIDs.contains(id)
    }

    static func defaultPrompt(for attachmentCount: Int) -> String {
        attachmentCount <= 1 ? "Please analyze the attached image." : "Please analyze the attached images."
    }

    static func chunkResponse(_ text: String, limit: Int = 3200) -> [String] {
        guard text.count > limit else { return [text] }
        var chunks: [String] = []
        var start = text.startIndex
        while start < text.endIndex {
            let hardEnd = text.index(start, offsetBy: limit, limitedBy: text.endIndex) ?? text.endIndex
            var split = hardEnd

            if hardEnd < text.endIndex {
                if let paragraphBreak = text[start..<hardEnd].range(of: "\n\n", options: .backwards) {
                    split = paragraphBreak.upperBound
                } else if let lineBreak = text[start..<hardEnd].range(of: "\n", options: .backwards) {
                    split = lineBreak.upperBound
                } else if let wordBreak = text[start..<hardEnd].range(of: " ", options: .backwards) {
                    split = wordBreak.upperBound
                }
            }

            if split == start {
                split = hardEnd
            }

            let chunk = text[start..<split].trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunk.isEmpty {
                chunks.append(String(chunk))
            }
            start = split
            while start < text.endIndex, text[start].isWhitespace {
                start = text.index(after: start)
            }
        }
        return chunks
    }
}

struct TelegramUpdateEnvelope: Decodable {
    let ok: Bool
    let result: [TelegramUpdate]
}

struct TelegramUpdate: Decodable {
    let update_id: Int64
    let message: TelegramMessage?
}

struct TelegramMessage: Decodable {
    let message_id: Int64
    let date: Int64?
    let text: String?
    let from: TelegramUser?
    let chat: TelegramChat
    let photo: [TelegramPhotoSize]?
}

struct TelegramUser: Decodable {
    let id: Int64
    let is_bot: Bool?
    let first_name: String?
    let username: String?
}

struct TelegramChat: Decodable {
    let id: Int64
    let type: String
}

struct TelegramPhotoSize: Decodable {
    let file_id: String
    let file_unique_id: String?
    let width: Int
    let height: Int
    let file_size: Int?
}

struct TelegramGetFileEnvelope: Decodable {
    let ok: Bool
    let result: TelegramFile?
}

struct TelegramFile: Decodable {
    let file_id: String
    let file_unique_id: String?
    let file_size: Int?
    let file_path: String?
}

private struct TelegramState: Codable {
    var lastUpdateID: Int64 = 0
}

actor TelegramStateStore {
    private let stateURL: URL
    private var state: TelegramState

    init(token: String) throws {
        let base = try Self.stateDirectory()
        self.stateURL = base.appendingPathComponent(Self.stateFileName(for: token))
        self.state = try Self.loadState(from: stateURL)
    }

    func lastUpdateID() -> Int64 {
        state.lastUpdateID
    }

    func advance(to updateID: Int64) throws {
        if updateID > state.lastUpdateID {
            state.lastUpdateID = updateID
            try persist()
        }
    }

    private func persist() throws {
        let data = try JSONEncoder().encode(state)
        try data.write(to: stateURL, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: stateURL.path)
    }

    private static func stateDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("afm/telegram", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private static func loadState(from url: URL) throws -> TelegramState {
        guard FileManager.default.fileExists(atPath: url.path) else { return TelegramState() }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TelegramState.self, from: data)
    }

    static func stateFileName(for token: String) -> String {
        let digest = SHA256.hash(data: Data(token.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "state-\(hex).json"
    }
}

enum TelegramBridgeError: LocalizedError {
    case invalidResponse(String)
    case downloadFailed(String)
    case unsupportedAttachmentType
    case attachmentTooLarge
    case tooManyAttachments
    case localClientError(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse(let message):
            return message
        case .downloadFailed(let message):
            return message
        case .unsupportedAttachmentType:
            return "Only Telegram photo attachments are supported"
        case .attachmentTooLarge:
            return "Attachment exceeds size limit"
        case .tooManyAttachments:
            return "Too many attachments"
        case .localClientError(let message):
            return message
        }
    }
}

private struct TelegramConversationTurn: Codable, Equatable, Sendable {
    let userPrompt: String
    let assistantReply: String
    let createdAt: Date
}

actor TelegramConversationStore {
    private var turnsByChatID: [Int64: [TelegramConversationTurn]] = [:]

    func historyMessages(chatID: Int64) -> [Message] {
        let turns = turnsByChatID[chatID] ?? []
        return turns.flatMap { turn in
            [
                Message(role: "user", content: turn.userPrompt),
                Message(role: "assistant", content: turn.assistantReply)
            ]
        }
    }

    func append(chatID: Int64, userPrompt: String, assistantReply: String) {
        let turn = TelegramConversationTurn(
            userPrompt: userPrompt,
            assistantReply: assistantReply,
            createdAt: Date()
        )
        var turns = turnsByChatID[chatID] ?? []
        turns.append(turn)
        if turns.count > TelegramPolicy.maxConversationTurns {
            turns.removeFirst(turns.count - TelegramPolicy.maxConversationTurns)
        }
        turnsByChatID[chatID] = turns
    }

    func reset(chatID: Int64) {
        turnsByChatID.removeValue(forKey: chatID)
    }
}

final class TelegramBridge {
    private let config: TelegramConfiguration
    private let client: AFMLocalClient
    private let bot: TelegramBotClient
    private let stateStore: TelegramStateStore
    private let conversationStore = TelegramConversationStore()
    private var watcherTask: Task<Void, Never>?

    init(config: TelegramConfiguration) throws {
        self.config = config
        self.client = AFMLocalClient(baseURL: config.localBaseURL, modelID: config.modelID, instructions: config.instructions)
        self.bot = TelegramBotClient(token: config.botToken)
        self.stateStore = try TelegramStateStore(token: config.botToken)
    }

    func start() async throws {
        let startupBaseline = try await discardPendingUpdates()
        if let startupBaseline {
            info("startup baseline set to update \(startupBaseline)")
        }
        info("enabled for \(config.allowedUserIDs.count) allowlisted Telegram user(s)")
        if let requiredPrefix = config.requiredPrefix {
            info("security: private chats only, prefix required (\(requiredPrefix)), images only, no remote tools")
        } else {
            info("security: private chats only, prefix optional, images only, no remote tools")
        }
        watcherTask = Task { [weak self] in
            guard let self else { return }
            await self.watchLoop()
        }
    }

    func stop() {
        watcherTask?.cancel()
        watcherTask = nil
    }

    private func discardPendingUpdates(maxBatches: Int = 50) async throws -> Int64? {
        var baseline = await stateStore.lastUpdateID()
        var sawAny = false

        for _ in 0..<maxBatches {
            debug("startup drain polling with offset \(baseline + 1)")
            let updates = try await bot.getUpdates(offset: baseline + 1, timeoutSeconds: 1)
            guard !updates.isEmpty else {
                return sawAny ? baseline : nil
            }
            sawAny = true
            debug("startup drain fetched updates \(updates.map(\.update_id))")
            if let highest = updates.map(\.update_id).max() {
                baseline = highest
                try await stateStore.advance(to: highest)
                debug("startup drain advanced baseline to \(highest)")
            }
            if updates.count < 100 {
                return baseline
            }
        }

        return sawAny ? baseline : nil
    }

    private func watchLoop() async {
        while !Task.isCancelled {
            do {
                let offset = await stateStore.lastUpdateID() + 1
                debug("polling getUpdates with offset \(offset)")
                let updates = try await bot.getUpdates(offset: offset, timeoutSeconds: 25)
                if updates.isEmpty {
                    try await Task.sleep(nanoseconds: UInt64(config.pollIntervalSeconds * 1_000_000_000))
                    continue
                }
                debug("fetched updates \(updates.map(\.update_id))")

                for update in updates {
                    let lastSeen = await stateStore.lastUpdateID()
                    if update.update_id <= lastSeen {
                        info("ignored stale Telegram update \(update.update_id) (lastUpdateID=\(lastSeen))")
                        continue
                    }
                    await process(update: update)
                    try await stateStore.advance(to: update.update_id)
                    debug("advanced lastUpdateID to \(update.update_id)")
                }
            } catch {
                info("poll error: \(error.localizedDescription)")
                do {
                    try await Task.sleep(nanoseconds: UInt64(config.pollIntervalSeconds * 1_000_000_000))
                } catch {
                    return
                }
            }
        }
    }

    private func process(update: TelegramUpdate) async {
        guard let message = update.message, let from = message.from else { return }
        if from.is_bot == true {
            debug("ignored bot-authored update \(update.update_id)")
            return
        }

        info(
            "detected update \(update.update_id) from user \(from.id) chat=\(message.chat.id) type=\(message.chat.type) " +
            "photos=\(message.photo?.count ?? 0)"
        )

        guard TelegramPolicy.isAllowedUser(from.id, allowedUserIDs: config.allowedUserIDs) else {
            info("ignored message from non-allowlisted Telegram user \(from.id)")
            return
        }

        guard TelegramPolicy.isPrivateChat(message.chat.type) else {
            info("ignored non-private Telegram chat from user \(from.id)")
            return
        }

        switch TelegramPolicy.parseCommand(message.text, requiredPrefix: config.requiredPrefix) {
        case .none:
            debug("ignored non-command Telegram message from user \(from.id)")
            return
        case .newConversation:
            await handleNewConversation(message: message, userID: from.id)
        case .prompt(let prompt):
            await handlePrompt(prompt: prompt, message: message, userID: from.id)
        }
    }

    private func handleNewConversation(message: TelegramMessage, userID: Int64) async {
        await conversationStore.reset(chatID: message.chat.id)
        do {
            try await bot.sendMessage(
                text: "Started a new AFM conversation for this chat. Previous context was cleared.",
                format: config.replyFormat,
                chatID: message.chat.id
            )
            info("reset conversation for Telegram user \(userID)")
            info("sent reply to Telegram user \(userID)")
        } catch {
            info("failed to send conversation reset reply to Telegram user \(userID): \(error.localizedDescription)")
        }
    }

    private func handlePrompt(prompt: String, message: TelegramMessage, userID: Int64) async {
        do {
            let imageURLs = try await prepareImages(from: message.photo ?? [])
            defer { cleanupTempFiles(imageURLs) }

            let effectivePrompt: String
            if prompt.isEmpty {
                if imageURLs.isEmpty {
                    try await bot.sendMessage(
                        text: "Usage: send /afm followed by your message. Photos may be attached in the same private chat.",
                        format: config.replyFormat,
                        chatID: message.chat.id
                    )
                    info("sent usage reply to Telegram user \(userID)")
                    return
                }
                effectivePrompt = TelegramPolicy.defaultPrompt(for: imageURLs.count)
            } else {
                effectivePrompt = prompt
            }

            info("accepted Telegram AFM request from user \(userID) with \(imageURLs.count) image attachment(s)")
            debug("effective prompt for Telegram user \(userID): \(effectivePrompt)")
            var messages = [Message(role: "system", content: config.instructions)]
            messages.append(contentsOf: await conversationStore.historyMessages(chatID: message.chat.id))

            let currentUserMessage: Message
            if imageURLs.isEmpty {
                currentUserMessage = Message(role: "user", content: effectivePrompt)
            } else {
                var parts = [ContentPart(type: "text", text: effectivePrompt, image_url: nil)]
                for imageURL in imageURLs {
                    parts.append(ContentPart(type: "image_url", text: nil, image_url: ImageURL(url: imageURL.absoluteString, detail: nil)))
                }
                currentUserMessage = Message(role: "user", content: .parts(parts))
            }
            messages.append(currentUserMessage)

            let response = try await client.sendMessages(messages, userTag: "telegram:\(userID)")
            let reply = response.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !reply.isEmpty else {
                try await bot.sendMessage(text: "AFM did not return any content for that request.", format: config.replyFormat, chatID: message.chat.id)
                info("sent empty-response notice to Telegram user \(userID)")
                return
            }

            await conversationStore.append(chatID: message.chat.id, userPrompt: effectivePrompt, assistantReply: reply)

            for chunk in TelegramPolicy.chunkResponse(reply) {
                try await bot.sendMessage(text: chunk, format: config.replyFormat, chatID: message.chat.id)
            }
            info("sent reply to Telegram user \(userID)")
        } catch {
            info("request failed for Telegram user \(userID): \(error.localizedDescription)")
            do {
                try await bot.sendMessage(
                    text: "AFM could not process that request securely. Try again with /afm in a private chat.",
                    format: config.replyFormat,
                    chatID: message.chat.id
                )
                info("sent failure reply to Telegram user \(userID)")
            } catch {
                info("failed to send error reply to Telegram user \(userID): \(error.localizedDescription)")
            }
        }
    }

    private func prepareImages(from photoSizes: [TelegramPhotoSize]) async throws -> [URL] {
        guard !photoSizes.isEmpty else { return [] }
        guard photoSizes.count <= TelegramPolicy.maxImageCount else {
            throw TelegramBridgeError.tooManyAttachments
        }

        // Telegram stores one photo as multiple sizes. Download the largest variant only.
        guard let largest = photoSizes.max(by: { ($0.file_size ?? 0) < ($1.file_size ?? 0) }) else {
            return []
        }

        if let size = largest.file_size, size > TelegramPolicy.maxImageBytes {
            throw TelegramBridgeError.attachmentTooLarge
        }

        let file = try await bot.getFile(fileID: largest.file_id)
        guard let remotePath = file.file_path, !remotePath.isEmpty else {
            throw TelegramBridgeError.invalidResponse("Telegram did not return a file path")
        }

        let downloaded = try await bot.downloadFile(path: remotePath, sizeLimit: TelegramPolicy.maxImageBytes)
        return [downloaded]
    }

    private func cleanupTempFiles(_ urls: [URL]) {
        let fm = FileManager.default
        for url in urls {
            try? fm.removeItem(at: url)
        }
    }

    private func info(_ message: String) {
        print("[Telegram] \(message)")
    }

    private func debug(_ message: String) {
        if config.verbose {
            print("[Telegram] \(message)")
        }
    }
}

final class TelegramBotClient {
    private let token: String
    private let session: URLSession
    private let apiBaseURL: URL
    private let fileBaseURL: URL

    init(token: String) {
        self.token = token
        self.session = URLSession(configuration: .ephemeral)
        self.apiBaseURL = URL(string: "https://api.telegram.org/bot\(token)/")!
        self.fileBaseURL = URL(string: "https://api.telegram.org/file/bot\(token)/")!
    }

    func getUpdates(offset: Int64, timeoutSeconds: Int) async throws -> [TelegramUpdate] {
        let endpoint = apiBaseURL.appendingPathComponent("getUpdates")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("application/x-www-form-urlencoded; charset=utf-8", forHTTPHeaderField: "Content-Type")
        let body = [
            "offset=\(offset)",
            "timeout=\(timeoutSeconds)",
            "allowed_updates=%5B%22message%22%5D"
        ].joined(separator: "&")
        request.httpBody = body.data(using: .utf8)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        let envelope = try JSONDecoder().decode(TelegramUpdateEnvelope.self, from: data)
        guard envelope.ok else {
            throw TelegramBridgeError.invalidResponse("Telegram getUpdates returned ok=false")
        }
        return envelope.result
    }

    func sendMessage(text: String, format: TelegramReplyFormat, chatID: Int64) async throws {
        let endpoint = apiBaseURL.appendingPathComponent("sendMessage")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var payload: [String: Any] = [
            "chat_id": chatID,
            "text": format.render(text)
        ]
        if let parseMode = format.apiParseMode {
            payload["parse_mode"] = parseMode
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
    }

    func getFile(fileID: String) async throws -> TelegramFile {
        let endpoint = apiBaseURL.appendingPathComponent("getFile")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["file_id": fileID])
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        let envelope = try JSONDecoder().decode(TelegramGetFileEnvelope.self, from: data)
        guard envelope.ok, let file = envelope.result else {
            throw TelegramBridgeError.invalidResponse("Telegram getFile returned ok=false")
        }
        return file
    }

    func downloadFile(path: String, sizeLimit: Int) async throws -> URL {
        let remoteURL = fileBaseURL.appendingPathComponent(path)
        let (tempURL, response) = try await session.download(from: remoteURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw TelegramBridgeError.downloadFailed("Telegram file download failed")
        }

        let attrs = try FileManager.default.attributesOfItem(atPath: tempURL.path)
        let size = attrs[.size] as? Int ?? 0
        guard size > 0, size <= sizeLimit else {
            throw TelegramBridgeError.attachmentTooLarge
        }

        let targetDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("afm-telegram", isDirectory: true)
        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let targetURL = targetDir.appendingPathComponent(UUID().uuidString + "-" + remoteURL.lastPathComponent)
        try? FileManager.default.removeItem(at: targetURL)
        try FileManager.default.moveItem(at: tempURL, to: targetURL)
        return targetURL
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw TelegramBridgeError.invalidResponse("Telegram request failed without HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw TelegramBridgeError.invalidResponse("Telegram API error \(http.statusCode): \(body)")
        }
    }
}
