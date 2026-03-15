import ArgumentParser
import Foundation
import Security
import SQLite3

struct IMessageConfiguration: Sendable {
    let allowedSenders: Set<String>
    let localBaseURL: String
    let modelID: String
    let instructions: String
    let verbose: Bool

    static func parseAllowedSenders(_ raw: String) throws -> Set<String> {
        let senders = raw
            .split(separator: ",")
            .map { Self.normalizeSender(String($0)) }
            .filter { !$0.isEmpty }
        guard !senders.isEmpty else {
            throw ValidationError("--imessage requires at least one comma-separated Apple ID")
        }
        return Set(senders)
    }

    static func normalizeSender(_ raw: String) -> String {
        raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "mailto:", with: "", options: [.caseInsensitive])
            .replacingOccurrences(of: "tel:", with: "", options: [.caseInsensitive])
            .lowercased()
    }
}

enum IMessageInboundCommand: Equatable {
    case none
    case pair(String)
    case prompt(String)
}

struct IMessageAttachment: Equatable, Sendable {
    let path: String
    let mimeType: String?
    let transferName: String?
}

struct IMessageEvent: Sendable {
    let rowID: Int64
    let messageGUID: String
    let sender: String
    let service: String
    let chatGUID: String
    let participantCount: Int
    let text: String
    let attachments: [IMessageAttachment]
}

struct IMessagePolicy {
    static let commandPrefix = "/afm"
    static let pairPrefix = "/afm pair"
    static let maxImageCount = 4
    static let maxImageBytes = 20 * 1024 * 1024

    static func parseCommand(_ text: String) -> IMessageInboundCommand {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .none }
        let lower = trimmed.lowercased()
        guard lower.hasPrefix(commandPrefix) else { return .none }

        let remainder = String(trimmed.dropFirst(commandPrefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        if remainder.lowercased().hasPrefix("pair ") {
            let code = String(remainder.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
            return code.isEmpty ? .none : .pair(code)
        }

        return .prompt(remainder)
    }

    static func isDirectChat(participantCount: Int) -> Bool {
        participantCount == 1
    }

    static func isAllowedSender(_ sender: String, allowedSenders: Set<String>) -> Bool {
        allowedSenders.contains(IMessageConfiguration.normalizeSender(sender))
    }

    static func defaultPrompt(for attachments: [URL]) -> String {
        attachments.count <= 1 ? "Please analyze the attached image." : "Please analyze the attached images."
    }

    static func chunkResponse(_ text: String, limit: Int = 3200) -> [String] {
        guard text.count > limit else { return [text] }
        var chunks: [String] = []
        var start = text.startIndex
        while start < text.endIndex {
            let end = text.index(start, offsetBy: limit, limitedBy: text.endIndex) ?? text.endIndex
            chunks.append(String(text[start..<end]))
            start = end
        }
        return chunks
    }
}

final class IMessageBridge {
    private let config: IMessageConfiguration
    private let pairingStore: IMessagePairingStore
    private let database: IMessageDatabase
    private let sender: IMessageSender
    private let client: AFMLocalClient
    private var watcherTask: Task<Void, Never>?
    private var lastSeenRowID: Int64 = 0

    init(config: IMessageConfiguration) throws {
        self.config = config
        self.pairingStore = try IMessagePairingStore()
        self.database = try IMessageDatabase()
        self.sender = IMessageSender()
        self.client = AFMLocalClient(baseURL: config.localBaseURL, modelID: config.modelID, instructions: config.instructions)
    }

    func start() async throws {
        try database.assertReadable()
        lastSeenRowID = try database.currentMaxRowID()
        info("enabled for \(config.allowedSenders.count) allowlisted sender(s)")
        info("security: direct chats only, pairing required, /afm prefix required, images only, no remote tools")
        info("security: recommended deployment is a dedicated macOS user + dedicated Apple ID with Full Disk Access")
        watcherTask = Task { [weak self] in
            guard let self else { return }
            await self.watchLoop()
        }
    }

    func stop() {
        watcherTask?.cancel()
        watcherTask = nil
    }

    private func watchLoop() async {
        while !Task.isCancelled {
            do {
                let events = try database.fetchEvents(after: lastSeenRowID)
                for event in events {
                    lastSeenRowID = max(lastSeenRowID, event.rowID)
                    await process(event: event)
                }
            } catch {
                info("poll error: \(error.localizedDescription)")
            }

            do {
                try await Task.sleep(nanoseconds: 2_000_000_000)
            } catch {
                return
            }
        }
    }

    private func process(event: IMessageEvent) async {
        let senderID = IMessageConfiguration.normalizeSender(event.sender)
        info(
            "detected event from \(redacted(senderID)) service=\(event.service) " +
            "participants=\(event.participantCount) attachments=\(event.attachments.count)"
        )
        info("received message from \(redacted(senderID)) via \(event.service)")
        guard event.service.caseInsensitiveCompare("iMessage") == .orderedSame else { return }
        guard IMessagePolicy.isAllowedSender(senderID, allowedSenders: config.allowedSenders) else {
            info("ignored message from non-allowlisted sender \(redacted(senderID))")
            return
        }
        guard IMessagePolicy.isDirectChat(participantCount: event.participantCount) else {
            info("ignored group chat message from \(redacted(senderID))")
            return
        }

        switch IMessagePolicy.parseCommand(event.text) {
        case .none:
            debug("ignored non-command message from \(redacted(senderID))")
            return
        case .pair(let code):
            await handlePairingReply(code: code, event: event, senderID: senderID)
        case .prompt(let prompt):
            await handlePrompt(prompt: prompt, event: event, senderID: senderID)
        }
    }

    private func handlePairingReply(code: String, event: IMessageEvent, senderID: String) async {
        do {
            let paired = try await pairingStore.completePairing(sender: senderID, chatGUID: event.chatGUID, providedCode: code)
            if paired {
                info("pairing complete for \(redacted(senderID))")
                try sender.send("AFM pairing complete. Future /afm messages from this direct chat will be processed.", to: senderID)
                info("sent reply to \(redacted(senderID))")
            } else {
                info("pairing failed for \(redacted(senderID))")
                try sender.send("AFM pairing failed. Request a new challenge by sending /afm.", to: senderID)
                info("sent reply to \(redacted(senderID))")
            }
        } catch {
            info("pairing reply failed for \(redacted(senderID)): \(error.localizedDescription)")
        }
    }

    private func handlePrompt(prompt: String, event: IMessageEvent, senderID: String) async {
        do {
            let isPaired = try await pairingStore.isPaired(sender: senderID, chatGUID: event.chatGUID)
            guard isPaired else {
                let challenge = try await pairingStore.issueChallenge(sender: senderID, chatGUID: event.chatGUID)
                info("issued pairing challenge to \(redacted(senderID))")
                let reply = """
                AFM pairing required for this direct chat.
                Reply with: /afm pair \(challenge)
                Challenge expires in 10 minutes.
                """
                try sender.send(reply, to: senderID)
                info("sent reply to \(redacted(senderID))")
                return
            }

            let imageURLs = try prepareImages(from: event.attachments)
            defer { cleanupTempFiles(imageURLs) }

            if !event.attachments.isEmpty && imageURLs.isEmpty {
                try sender.send("AFM rejected the message because only image attachments are allowed in iMessage mode.", to: senderID)
                info("rejected unsupported attachments from \(redacted(senderID))")
                info("sent reply to \(redacted(senderID))")
                return
            }

            let effectivePrompt: String
            if prompt.isEmpty {
                if imageURLs.isEmpty {
                    try sender.send("Usage: send /afm followed by your message. Images may be attached in the same direct chat.", to: senderID)
                    info("sent usage reply to \(redacted(senderID))")
                    return
                }
                effectivePrompt = IMessagePolicy.defaultPrompt(for: imageURLs)
            } else {
                effectivePrompt = prompt
            }

            info("accepted AFM request from \(redacted(senderID)) with \(imageURLs.count) image attachment(s)")
            let response = try await client.sendMessage(prompt: effectivePrompt, imageURLs: imageURLs, userTag: "imessage:\(senderID)")
            let reply = response.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !reply.isEmpty else {
                try sender.send("AFM did not return any content for that request.", to: senderID)
                info("sent empty-response notice to \(redacted(senderID))")
                return
            }

            for chunk in IMessagePolicy.chunkResponse(reply) {
                try sender.send(chunk, to: senderID)
            }
            info("sent reply to \(redacted(senderID))")
        } catch {
            info("request failed for \(redacted(senderID)): \(error.localizedDescription)")
            do {
                try sender.send("AFM could not process that request securely. Try again or resend the pairing command if needed.", to: senderID)
                info("sent failure reply to \(redacted(senderID))")
            } catch {
                info("failed to send error reply to \(redacted(senderID)): \(error.localizedDescription)")
            }
        }
    }

    private func prepareImages(from attachments: [IMessageAttachment]) throws -> [URL] {
        guard attachments.count <= IMessagePolicy.maxImageCount else {
            throw IMessageBridgeError.tooManyAttachments
        }

        var prepared: [URL] = []
        for attachment in attachments {
            let copied = try secureCopyImage(at: attachment)
            prepared.append(copied)
        }
        return prepared
    }

    private func secureCopyImage(at attachment: IMessageAttachment) throws -> URL {
        let fm = FileManager.default
        let expandedPath = NSString(string: attachment.path).expandingTildeInPath
        let originalURL = URL(fileURLWithPath: expandedPath).standardizedFileURL.resolvingSymlinksInPath()
        let allowedRoot = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Messages/Attachments").standardizedFileURL.resolvingSymlinksInPath()
        guard originalURL.path.hasPrefix(allowedRoot.path + "/") || originalURL.path == allowedRoot.path else {
            throw IMessageBridgeError.attachmentOutsideMessagesDirectory
        }

        let resourceValues = try originalURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey, .contentTypeKey])
        guard resourceValues.isRegularFile == true else {
            throw IMessageBridgeError.invalidAttachment
        }
        let size = resourceValues.fileSize ?? 0
        guard size > 0, size <= IMessagePolicy.maxImageBytes else {
            throw IMessageBridgeError.attachmentTooLarge
        }

        let mime = attachment.mimeType?.lowercased() ?? resourceValues.contentType?.preferredMIMEType?.lowercased()
        guard mime?.hasPrefix("image/") == true else {
            throw IMessageBridgeError.unsupportedAttachmentType
        }

        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("afm-imessage", isDirectory: true)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let targetURL = tempDir.appendingPathComponent(UUID().uuidString + "-" + originalURL.lastPathComponent)
        try fm.copyItem(at: originalURL, to: targetURL)
        return targetURL
    }

    private func cleanupTempFiles(_ urls: [URL]) {
        let fm = FileManager.default
        for url in urls {
            try? fm.removeItem(at: url)
        }
    }

    private func info(_ message: String) {
        print("[iMessage] \(message)")
    }

    private func debug(_ message: String) {
        if config.verbose {
            print("[iMessage] \(message)")
        }
    }

    private func redacted(_ sender: String) -> String {
        if sender.contains("@") {
            let parts = sender.split(separator: "@", maxSplits: 1, omittingEmptySubsequences: false)
            let local = String(parts.first ?? "")
            let domain = String(parts.count > 1 ? parts[1] : "")
            let localSuffix = String(local.suffix(min(2, local.count)))
            return "***\(localSuffix)@\(domain)"
        }
        let digits = sender.filter(\.isNumber)
        if !digits.isEmpty {
            return "***\(digits.suffix(min(4, digits.count)))"
        }
        return "***"
    }
}

enum IMessageBridgeError: LocalizedError {
    case tooManyAttachments
    case unsupportedAttachmentType
    case attachmentTooLarge
    case invalidAttachment
    case attachmentOutsideMessagesDirectory
    case automationFailed(String)
    case localClientError(String)

    var errorDescription: String? {
        switch self {
        case .tooManyAttachments:
            return "Too many attachments"
        case .unsupportedAttachmentType:
            return "Only image attachments are supported"
        case .attachmentTooLarge:
            return "Attachment exceeds size limit"
        case .invalidAttachment:
            return "Attachment is invalid"
        case .attachmentOutsideMessagesDirectory:
            return "Attachment path is outside the Messages attachment directory"
        case .automationFailed(let message):
            return message
        case .localClientError(let message):
            return message
        }
    }
}

private struct IMessagePairingMetadata: Codable {
    var pairings: [IMessagePairingRecord] = []
    var pendingChallenges: [IMessagePendingChallenge] = []
}

private struct IMessagePairingRecord: Codable, Equatable {
    let sender: String
    let chatGUID: String
    let pairedAt: Date
}

private struct IMessagePendingChallenge: Codable, Equatable {
    let sender: String
    let chatGUID: String
    let challengeKey: String
    let issuedAt: Date
}

actor IMessagePairingStore {
    private static let service = "com.scouzi1966.afm.imessage"
    private static let challengeTTL: TimeInterval = 600

    private let metadataURL: URL
    private var metadata: IMessagePairingMetadata

    init() throws {
        let base = try Self.stateDirectory()
        self.metadataURL = base.appendingPathComponent("pairings.json")
        self.metadata = try Self.loadMetadata(from: metadataURL)
    }

    func isPaired(sender: String, chatGUID: String) throws -> Bool {
        metadata.pairings.contains { $0.sender == sender && $0.chatGUID == chatGUID }
    }

    func issueChallenge(sender: String, chatGUID: String) throws -> String {
        pruneExpiredChallenges()
        if let pending = metadata.pendingChallenges.first(where: { $0.sender == sender && $0.chatGUID == chatGUID }),
           let existing = try Self.keychainValue(for: pending.challengeKey) {
            return existing
        }

        let challenge = Self.randomChallengeCode()
        let key = "challenge:\(sender):\(chatGUID)"
        try Self.setKeychainValue(challenge, for: key)
        metadata.pendingChallenges.removeAll { $0.sender == sender && $0.chatGUID == chatGUID }
        metadata.pendingChallenges.append(
            IMessagePendingChallenge(sender: sender, chatGUID: chatGUID, challengeKey: key, issuedAt: Date())
        )
        try persist()
        return challenge
    }

    func completePairing(sender: String, chatGUID: String, providedCode: String) throws -> Bool {
        pruneExpiredChallenges()
        guard let pending = metadata.pendingChallenges.first(where: { $0.sender == sender && $0.chatGUID == chatGUID }),
              let expected = try Self.keychainValue(for: pending.challengeKey) else {
            return false
        }
        guard expected == providedCode else { return false }

        try Self.deleteKeychainValue(for: pending.challengeKey)
        let pairingSecretKey = "pairing:\(sender):\(chatGUID)"
        try Self.setKeychainValue(UUID().uuidString, for: pairingSecretKey)
        metadata.pendingChallenges.removeAll { $0.sender == sender && $0.chatGUID == chatGUID }
        metadata.pairings.removeAll { $0.sender == sender && $0.chatGUID == chatGUID }
        metadata.pairings.append(IMessagePairingRecord(sender: sender, chatGUID: chatGUID, pairedAt: Date()))
        try persist()
        return true
    }

    private func pruneExpiredChallenges() {
        let now = Date()
        let expired = metadata.pendingChallenges.filter { now.timeIntervalSince($0.issuedAt) > Self.challengeTTL }
        for item in expired {
            try? Self.deleteKeychainValue(for: item.challengeKey)
        }
        metadata.pendingChallenges.removeAll { now.timeIntervalSince($0.issuedAt) > Self.challengeTTL }
    }

    private func persist() throws {
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: metadataURL, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: metadataURL.path)
    }

    private static func stateDirectory() throws -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("afm/imessage", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private static func loadMetadata(from url: URL) throws -> IMessagePairingMetadata {
        guard FileManager.default.fileExists(atPath: url.path) else { return IMessagePairingMetadata() }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(IMessagePairingMetadata.self, from: data)
    }

    private static func randomChallengeCode() -> String {
        var bytes = [UInt8](repeating: 0, count: 4)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        let value = bytes.reduce(0) { ($0 << 8) | UInt32($1) } % 1_000_000
        return String(format: "%06u", value)
    }

    private static func setKeychainValue(_ value: String, for account: String) throws {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(query as CFDictionary)
        let attrs: [String: Any] = query.merging([kSecValueData as String: data]) { _, new in new }
        let status = SecItemAdd(attrs as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw IMessageBridgeError.automationFailed("Failed to persist iMessage pairing secret (status \(status))")
        }
    }

    private static func keychainValue(for account: String) throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data, let string = String(data: data, encoding: .utf8) else {
            throw IMessageBridgeError.automationFailed("Failed to read iMessage pairing secret (status \(status))")
        }
        return string
    }

    private static func deleteKeychainValue(for account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw IMessageBridgeError.automationFailed("Failed to delete iMessage pairing secret (status \(status))")
        }
    }
}

final class IMessageDatabase {
    private let databaseURL: URL

    init(databaseURL: URL? = nil) throws {
        self.databaseURL = databaseURL ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Messages/chat.db")
    }

    func assertReadable() throws {
        guard FileManager.default.isReadableFile(atPath: databaseURL.path) else {
            throw ValidationError("iMessage mode requires read access to \(databaseURL.path). Grant Full Disk Access to AFM.")
        }
    }

    func currentMaxRowID() throws -> Int64 {
        try withConnection { db in
            let sql = "SELECT IFNULL(MAX(ROWID), 0) FROM message;"
            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw sqliteError(db)
            }
            guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return sqlite3_column_int64(stmt, 0)
        }
    }

    func fetchEvents(after rowID: Int64) throws -> [IMessageEvent] {
        try withConnection { db in
            let sql = """
            SELECT
              m.ROWID,
              IFNULL(m.guid, ''),
              IFNULL(h.id, ''),
              IFNULL(h.service, ''),
              IFNULL(c.guid, ''),
              IFNULL(m.text, ''),
              IFNULL((SELECT COUNT(*) FROM chat_handle_join chj WHERE chj.chat_id = c.ROWID), 0)
            FROM message m
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            JOIN chat c ON c.ROWID = cmj.chat_id
            LEFT JOIN handle h ON h.ROWID = m.handle_id
            WHERE m.ROWID > ? AND m.is_from_me = 0
            ORDER BY m.ROWID ASC;
            """

            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw sqliteError(db)
            }
            sqlite3_bind_int64(stmt, 1, rowID)

            var events: [IMessageEvent] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                let currentRowID = sqlite3_column_int64(stmt, 0)
                let guid = Self.string(stmt, index: 1)
                let sender = Self.string(stmt, index: 2)
                let service = Self.string(stmt, index: 3)
                let chatGUID = Self.string(stmt, index: 4)
                let text = Self.string(stmt, index: 5)
                let participantCount = Int(sqlite3_column_int(stmt, 6))
                let attachments = try fetchAttachments(messageRowID: currentRowID, db: db)
                events.append(
                    IMessageEvent(
                        rowID: currentRowID,
                        messageGUID: guid,
                        sender: sender,
                        service: service,
                        chatGUID: chatGUID,
                        participantCount: participantCount,
                        text: text,
                        attachments: attachments
                    )
                )
            }
            return events
        }
    }

    private func fetchAttachments(messageRowID: Int64, db: OpaquePointer?) throws -> [IMessageAttachment] {
        let sql = """
        SELECT IFNULL(a.filename, ''), a.mime_type, a.transfer_name
        FROM attachment a
        JOIN message_attachment_join maj ON maj.attachment_id = a.ROWID
        WHERE maj.message_id = ?;
        """
        var stmt: OpaquePointer?
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw sqliteError(db)
        }
        sqlite3_bind_int64(stmt, 1, messageRowID)

        var attachments: [IMessageAttachment] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let path = Self.string(stmt, index: 0)
            guard !path.isEmpty else { continue }
            let mime = sqlite3_column_text(stmt, 1).map { String(cString: $0) }
            let transferName = sqlite3_column_text(stmt, 2).map { String(cString: $0) }
            attachments.append(IMessageAttachment(path: path, mimeType: mime, transferName: transferName))
        }
        return attachments
    }

    private func withConnection<T>(_ body: (OpaquePointer?) throws -> T) throws -> T {
        var db: OpaquePointer?
        let flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(databaseURL.path, &db, flags, nil) == SQLITE_OK else {
            defer { if db != nil { sqlite3_close(db) } }
            throw sqliteError(db)
        }
        defer { sqlite3_close(db) }
        sqlite3_busy_timeout(db, 1_000)
        return try body(db)
    }

    private func sqliteError(_ db: OpaquePointer?) -> Error {
        let message = db.flatMap { sqlite3_errmsg($0) }.map { String(cString: $0) } ?? "Unknown SQLite error"
        return ValidationError("Failed to read Messages database: \(message)")
    }

    private static func string(_ stmt: OpaquePointer?, index: Int32) -> String {
        guard let cString = sqlite3_column_text(stmt, index) else { return "" }
        return String(cString: cString)
    }
}

final class IMessageSender {
    func send(_ text: String, to recipient: String) throws {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        task.arguments = [
            "-e",
            """
            on run argv
              set targetHandle to item 1 of argv
              set messageBody to item 2 of argv
              tell application "Messages"
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy targetHandle of targetService
                send messageBody to targetBuddy
              end tell
            end run
            """,
            recipient,
            text
        ]
        do {
            try task.run()
            task.waitUntilExit()
        } catch {
            throw IMessageBridgeError.automationFailed("Failed to invoke Messages automation: \(error.localizedDescription)")
        }
        guard task.terminationReason == .exit, task.terminationStatus == 0 else {
            throw IMessageBridgeError.automationFailed("Messages automation exited with status \(task.terminationStatus)")
        }
    }
}

final class AFMLocalClient {
    private let baseURL: URL
    private let modelID: String
    private let instructions: String
    private let session: URLSession

    init(baseURL: String, modelID: String, instructions: String) {
        self.baseURL = URL(string: baseURL)!
        self.modelID = modelID
        self.instructions = instructions
        self.session = URLSession(configuration: .ephemeral)
    }

    func sendMessage(prompt: String, imageURLs: [URL], userTag: String) async throws -> String {
        let endpoint = baseURL.appendingPathComponent("v1/chat/completions")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let userMessage: Message
        if imageURLs.isEmpty {
            userMessage = Message(role: "user", content: prompt)
        } else {
            var parts = [ContentPart(type: "text", text: prompt, image_url: nil)]
            for imageURL in imageURLs {
                parts.append(ContentPart(type: "image_url", text: nil, image_url: ImageURL(url: imageURL.absoluteString, detail: nil)))
            }
            userMessage = Message(role: "user", content: .parts(parts))
        }

        let payload = ChatCompletionRequest(
            model: modelID,
            messages: [
                Message(role: "system", content: instructions),
                userMessage
            ],
            temperature: nil,
            maxTokens: nil,
            maxCompletionTokens: nil,
            topP: nil,
            repetitionPenalty: nil,
            repeatPenalty: nil,
            frequencyPenalty: nil,
            presencePenalty: nil,
            topK: nil,
            minP: nil,
            seed: nil,
            logprobs: nil,
            topLogprobs: nil,
            stop: nil,
            stream: false,
            user: userTag,
            tools: nil,
            toolChoice: nil,
            responseFormat: nil,
            chatTemplateKwargs: nil
        )

        request.httpBody = try JSONEncoder().encode(payload)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw IMessageBridgeError.localClientError("AFM iMessage request failed without HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw IMessageBridgeError.localClientError("AFM API error \(http.statusCode): \(body)")
        }

        let completion = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
        return completion.choices.first?.message.content ?? ""
    }
}
