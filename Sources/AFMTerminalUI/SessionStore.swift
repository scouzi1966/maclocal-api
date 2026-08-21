import Darwin
import Foundation
import AFMOpenAICompat

public struct TUISession: Codable, Sendable {
    public var id: UUID
    public var title: String
    public var backend: String
    public var model: String
    public var createdAt: Date
    public var updatedAt: Date
    public var messages: [Message]
    /// Reasoning is UI metadata rather than part of the OpenAI request message. Keys are
    /// message indexes encoded as strings so older session files remain decodable.
    public var reasoningByMessage: [String: String]

    public init(
        id: UUID = UUID(),
        title: String = "New chat",
        backend: String,
        model: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        messages: [Message] = [],
        reasoningByMessage: [String: String] = [:]
    ) {
        self.id = id
        self.title = title
        self.backend = backend
        self.model = model
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messages = messages
        self.reasoningByMessage = reasoningByMessage
    }

    private enum CodingKeys: String, CodingKey {
        case id, title, backend, model, createdAt, updatedAt, messages, reasoningByMessage
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        id = try values.decode(UUID.self, forKey: .id)
        title = try values.decode(String.self, forKey: .title)
        backend = try values.decode(String.self, forKey: .backend)
        model = try values.decode(String.self, forKey: .model)
        createdAt = try values.decode(Date.self, forKey: .createdAt)
        updatedAt = try values.decode(Date.self, forKey: .updatedAt)
        messages = try values.decode([Message].self, forKey: .messages)
        reasoningByMessage = try values.decodeIfPresent([String: String].self, forKey: .reasoningByMessage) ?? [:]
        pruneReasoningMetadata()
    }

    public func reasoning(atMessageIndex index: Int) -> String? {
        reasoningByMessage[String(index)]
    }

    public mutating func removeLastExchange() {
        if messages.last?.role == "assistant" { messages.removeLast() }
        if messages.last?.role == "user" { messages.removeLast() }
        pruneReasoningMetadata()
    }

    public mutating func removeMessage(at index: Int) {
        guard messages.indices.contains(index) else { return }
        messages.remove(at: index)
        var remapped: [String: String] = [:]
        for (key, value) in reasoningByMessage {
            guard let oldIndex = Int(key), oldIndex != index else { continue }
            let newIndex = oldIndex > index ? oldIndex - 1 : oldIndex
            if messages.indices.contains(newIndex), messages[newIndex].role == "assistant" {
                remapped[String(newIndex)] = value
            }
        }
        reasoningByMessage = remapped
    }

    public mutating func pruneReasoningMetadata() {
        reasoningByMessage = reasoningByMessage.filter { key, _ in
            guard let index = Int(key), messages.indices.contains(index) else { return false }
            return messages[index].role == "assistant"
        }
    }
}

public struct TUISessionSummary: Equatable, Sendable {
    public let id: UUID
    public let title: String
    public let updatedAt: Date
    public let matchingSnippet: String?
}

public enum TUISessionStoreError: Error, LocalizedError, Equatable {
    case sessionTooLarge(maximumBytes: Int)

    public var errorDescription: String? {
        switch self {
        case .sessionTooLarge(let maximumBytes):
            return "Session exceeds the \(maximumBytes)-byte save/load limit."
        }
    }
}

public final class TUISessionStore: @unchecked Sendable {
    public static let defaultMaximumSessionBytes = 32_000_000
    public let directory: URL
    private let fileManager: FileManager
    private let maximumSessionBytes: Int
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder
    private let lock = NSLock()

    public init(
        directory: URL? = nil,
        fileManager: FileManager = .default,
        maximumSessionBytes: Int = TUISessionStore.defaultMaximumSessionBytes
    ) {
        self.fileManager = fileManager
        self.maximumSessionBytes = max(0, maximumSessionBytes)
        if let directory {
            self.directory = directory
        } else {
            self.directory = fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent(".afm/sessions", isDirectory: true)
        }
        encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
    }

    @discardableResult
    public func save(_ session: TUISession) throws -> URL {
        lock.lock()
        defer { lock.unlock() }
        try ensureDirectory()
        let target = url(for: session.id)
        let temporary = directory.appendingPathComponent(
            ".\(session.id.uuidString).\(UUID().uuidString).tmp"
        )
        defer { try? fileManager.removeItem(at: temporary) }
        let data = try encoder.encode(session)
        guard data.count <= maximumSessionBytes else {
            throw TUISessionStoreError.sessionTooLarge(maximumBytes: maximumSessionBytes)
        }
        try TUIArtifactActions.save(data, to: temporary)
        try fileManager.setAttributes(
            [.posixPermissions: 0o600, .modificationDate: session.updatedAt],
            ofItemAtPath: temporary.path
        )
        guard Darwin.rename(temporary.path, target.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return target
    }

    public func load(id: UUID) throws -> TUISession {
        lock.lock()
        defer { lock.unlock() }
        return try decoder.decode(
            TUISession.self,
            from: TUIArtifactActions.readRegularFile(at: url(for: id), maximumBytes: maximumSessionBytes)
        )
    }

    public func recent(limit: Int = 20) throws -> [TUISessionSummary] {
        lock.lock()
        defer { lock.unlock() }
        let limit = max(0, limit)
        guard limit > 0 else { return [] }

        var candidates: [SessionFile] = []
        try forEachSessionURL { url in
            let values = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            candidates.append(SessionFile(
                url: url,
                modificationDate: values?.contentModificationDate ?? .distantPast
            ))
            candidates.sort { $0.modificationDate > $1.modificationDate }
            if candidates.count > limit { candidates.removeLast(candidates.count - limit) }
        }

        return candidates.compactMap { candidate in
            guard let metadata = try? decoder.decode(
                SessionMetadata.self,
                from: TUIArtifactActions.readRegularFile(
                    at: candidate.url,
                    maximumBytes: maximumSessionBytes
                )
            ) else { return nil }
            return TUISessionSummary(
                id: metadata.id,
                title: metadata.title,
                updatedAt: metadata.updatedAt,
                matchingSnippet: nil
            )
        }
    }

    public func search(_ query: String, limit: Int = 20) throws -> [TUISessionSummary] {
        lock.lock()
        defer { lock.unlock() }
        let limit = max(0, limit)
        let lowered = query.lowercased()
        guard limit > 0, !lowered.isEmpty else { return [] }

        var result: [TUISessionSummary] = []
        try forEachSessionURL { url in
            guard let session = try? decoder.decode(
                TUISession.self,
                from: TUIArtifactActions.readRegularFile(at: url, maximumBytes: maximumSessionBytes)
            ) else { return }
            let matching = (session.messages.map(\.textContent) + Array(session.reasoningByMessage.values))
                .first { $0.lowercased().contains(lowered) }
            guard session.title.lowercased().contains(lowered) || matching != nil else { return }
            result.append(TUISessionSummary(
                id: session.id,
                title: session.title,
                updatedAt: session.updatedAt,
                matchingSnippet: matching
            ))
            result.sort { $0.updatedAt > $1.updatedAt }
            if result.count > limit { result.removeLast(result.count - limit) }
        }
        return result
    }

    public func exportMarkdown(_ session: TUISession, to url: URL, overwrite: Bool = false) throws {
        var markdown = "# \(session.title)\n\n"
        markdown += "- Backend: \(session.backend)\n- Model: \(session.model)\n- Updated: \(ISO8601DateFormatter().string(from: session.updatedAt))\n\n"
        for (index, message) in session.messages.enumerated() {
            markdown += "## \(message.role.capitalized)\n\n\(message.textContent)\n\n"
            if let reasoning = session.reasoning(atMessageIndex: index), !reasoning.isEmpty {
                markdown += "<details><summary>Reasoning</summary>\n\n\(reasoning)\n\n</details>\n\n"
            }
        }
        try TUIArtifactActions.save(Data(markdown.utf8), to: url, overwrite: overwrite)
    }

    private struct SessionMetadata: Decodable {
        let id: UUID
        let title: String
        let updatedAt: Date
    }

    private struct SessionFile {
        let url: URL
        let modificationDate: Date
    }

    private func forEachSessionURL(_ body: (URL) -> Void) throws {
        guard fileManager.fileExists(atPath: directory.path) else { return }
        var enumerationError: Error?
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants],
            errorHandler: { _, error in
                enumerationError = error
                return false
            }
        ) else { return }
        for case let url as URL in enumerator where url.pathExtension == "json" {
            body(url)
        }
        if let enumerationError { throw enumerationError }
    }

    private func ensureDirectory() throws {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        try? fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
    }

    private func url(for id: UUID) -> URL { directory.appendingPathComponent("\(id.uuidString).json") }
}
