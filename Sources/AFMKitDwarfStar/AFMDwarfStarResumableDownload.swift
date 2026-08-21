import Foundation
import CryptoKit
import HuggingFace
import Xet

struct AFMDwarfStarXetMetadata: Sendable {
    let fileID: String?
    let expectedBytes: Int64
    let expectedSHA256: String?
}

enum AFMDwarfStarDownloadError: LocalizedError {
    case missingXetMetadata(String)
    case invalidResponse(String)
    case rangeNotHonored(expectedOffset: Int64, statusCode: Int)
    case incompleteDownload(expected: Int64, actual: Int64)
    case checksumMismatch(expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case .missingXetMetadata(let path):
            return "Hugging Face did not return Xet metadata for \(path)."
        case .invalidResponse(let detail):
            return "Invalid Hugging Face response: \(detail)"
        case .rangeNotHonored(let offset, let status):
            return "LFS fallback did not honor byte offset \(offset) (HTTP \(status)); the partial Xet download was preserved."
        case .incompleteDownload(let expected, let actual):
            return "Downloaded file size is \(actual) bytes; expected \(expected) bytes. The partial download was preserved."
        case .checksumMismatch(let expected, let actual):
            return "Downloaded file SHA-256 is \(actual); expected \(expected). The partial download was preserved."
        }
    }
}

enum AFMDwarfStarResumableDownload {
    private static let ggufHeaderProbeBytes = 1_048_576

    static func fetchGGUFArchitecture(
        repositoryID: String,
        revision: String,
        path: String,
        token: String?
    ) async throws -> String? {
        let url = try resolveURL(repositoryID: repositoryID, revision: revision, path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("bytes=0-\(ggufHeaderProbeBytes - 1)", forHTTPHeaderField: "Range")
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let delegate = AFMDwarfStarBoundedRangeDataDelegate(maximumBytes: ggufHeaderProbeBytes)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let data = try await delegate.run(request: request, session: session)
        return AFMDwarfStarCheckpointCatalog.ggufArchitecture(in: data)
    }

    static func fetchXetMetadata(
        repositoryID: String,
        revision: String,
        path: String,
        expectedBytes: Int64,
        token: String?
    ) async throws -> AFMDwarfStarXetMetadata {
        let url = try resolveURL(repositoryID: repositoryID, revision: revision, path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let delegate = NoRedirectDelegate()
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let (_, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AFMDwarfStarDownloadError.invalidResponse("HEAD did not return HTTP metadata")
        }
        guard (200 ... 399).contains(http.statusCode) else {
            throw AFMDwarfStarDownloadError.invalidResponse("HEAD returned HTTP \(http.statusCode)")
        }
        let xetHash = http.value(forHTTPHeaderField: "X-Xet-Hash")?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let fileID = xetHash.flatMap { value -> String? in
            value.count == 64 && value.allSatisfy(\.isHexDigit) ? value : nil
        }
        let linkedSize = http.value(forHTTPHeaderField: "X-Linked-Size").flatMap(Int64.init)
        let linkedEtag = normalizeSHA256(http.value(forHTTPHeaderField: "X-Linked-Etag"))
        return AFMDwarfStarXetMetadata(
            fileID: fileID,
            expectedBytes: linkedSize ?? expectedBytes,
            expectedSHA256: linkedEtag)
    }

    static func downloadXetRange(
        metadata: AFMDwarfStarXetMetadata,
        repositoryID: String,
        revision: String,
        offset: Int64,
        segmentURL: URL,
        token: String?
    ) async throws {
        guard offset < metadata.expectedBytes else { return }
        guard let fileID = metadata.fileID else {
            throw AFMDwarfStarDownloadError.missingXetMetadata(repositoryID)
        }
        guard let refreshURL = URL(
            string: "https://huggingface.co/api/models/\(repositoryID)/xet-read-token/\(revision)"
        ) else {
            throw AFMDwarfStarDownloadError.invalidResponse("invalid Xet refresh URL")
        }
        var configuration = XetDownloader.Configuration.default
        configuration.readTimeout = 600
        configuration.idleTimeout = 300
        configuration.enableMultipath = false
        _ = try await Xet.withDownloader(
            refreshURL: refreshURL,
            hubToken: token,
            configuration: configuration
        ) { downloader in
            try await downloader.download(
                fileID,
                byteRange: UInt64(offset)..<UInt64(metadata.expectedBytes),
                to: segmentURL)
        }
    }

    static func downloadLFSRange(
        repositoryID: String,
        revision: String,
        path: String,
        destination: URL,
        offset: Int64,
        expectedBytes: Int64,
        token: String?,
        progress: @escaping @Sendable (Int64) -> Void
    ) async throws {
        let url = try resolveURL(repositoryID: repositoryID, revision: revision, path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let delegate = CacheLocalDataDelegate(
            destination: destination,
            offset: offset,
            expectedBytes: expectedBytes,
            progress: progress)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await delegate.run(request: request, session: session)
    }

    static func appendSegment(_ segment: URL, to partial: URL, expectedBytes: Int64) throws {
        guard FileManager.default.fileExists(atPath: segment.path) else { return }
        let partialSize = fileSize(partial)
        let remaining = max(expectedBytes - partialSize, 0)
        let segmentSize = fileSize(segment)
        guard segmentSize <= remaining else {
            throw AFMDwarfStarDownloadError.incompleteDownload(
                expected: expectedBytes,
                actual: partialSize + segmentSize)
        }
        if !FileManager.default.fileExists(atPath: partial.path) {
            FileManager.default.createFile(atPath: partial.path, contents: nil)
        }
        let input = try FileHandle(forReadingFrom: segment)
        let output = try FileHandle(forWritingTo: partial)
        defer {
            try? input.close()
            try? output.close()
        }
        try output.seekToEnd()
        while let data = try input.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
            try output.write(contentsOf: data)
        }
        try FileManager.default.removeItem(at: segment)
    }

    static func adoptSegment(_ segment: URL, as partial: URL, expectedBytes: Int64) throws {
        guard FileManager.default.fileExists(atPath: segment.path) else { return }
        if fileSize(partial) == 0 {
            if FileManager.default.fileExists(atPath: partial.path) {
                try FileManager.default.removeItem(at: partial)
            }
            guard fileSize(segment) <= expectedBytes else {
                throw AFMDwarfStarDownloadError.incompleteDownload(
                    expected: expectedBytes,
                    actual: fileSize(segment))
            }
            try FileManager.default.moveItem(at: segment, to: partial)
        } else {
            try appendSegment(segment, to: partial, expectedBytes: expectedBytes)
        }
    }

    static func publish(
        partial: URL,
        blob: URL,
        expectedBytes: Int64,
        expectedSHA256: String?
    ) throws {
        let actual = fileSize(partial)
        guard actual == expectedBytes else {
            throw AFMDwarfStarDownloadError.incompleteDownload(expected: expectedBytes, actual: actual)
        }
        if let expectedSHA256 {
            let actualSHA256 = try sha256(partial)
            guard actualSHA256 == expectedSHA256 else {
                throw AFMDwarfStarDownloadError.checksumMismatch(
                    expected: expectedSHA256,
                    actual: actualSHA256)
            }
        }
        if FileManager.default.fileExists(atPath: blob.path) {
            _ = try FileManager.default.replaceItemAt(blob, withItemAt: partial)
        } else {
            try FileManager.default.moveItem(at: partial, to: blob)
        }
    }

    static func fileSize(_ url: URL) -> Int64 {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else { return 0 }
        return size.int64Value
    }

    static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 8 * 1024 * 1024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func detailedError(_ error: Error) -> String {
        var parts = [String(reflecting: error)]
        var current = error as NSError
        var seen = Set<ObjectIdentifier>()
        while let underlying = current.userInfo[NSUnderlyingErrorKey] as? NSError,
              seen.insert(ObjectIdentifier(underlying)).inserted {
            parts.append(String(reflecting: underlying))
            current = underlying
        }
        return parts.joined(separator: " <- ")
    }

    private static func resolveURL(repositoryID: String, revision: String, path: String) throws -> URL {
        guard let base = URL(string: "https://huggingface.co") else {
            throw AFMDwarfStarDownloadError.invalidResponse("invalid Hugging Face base URL")
        }
        return path.split(separator: "/").reduce(
            base.appendingPathComponent(repositoryID)
                .appendingPathComponent("resolve")
                .appendingPathComponent(revision)
        ) { $0.appendingPathComponent(String($1)) }
    }

    private static func normalizeSHA256(_ value: String?) -> String? {
        guard var value else { return nil }
        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.hasPrefix("W/") { value.removeFirst(2) }
        value = value.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        guard value.count == 64, value.allSatisfy(\.isHexDigit) else { return nil }
        return value.lowercased()
    }
}

private final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

final class AFMDwarfStarBoundedRangeDataDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let maximumBytes: Int
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Data, Error>?
    private var received = Data()
    private var responseError: Error?

    init(maximumBytes: Int) {
        self.maximumBytes = maximumBytes
    }

    var acceptedByteCount: Int { received.count }

    func run(request: URLRequest, session: URLSession) async throws -> Data {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.withLock { self.continuation = continuation }
                session.dataTask(with: request).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            responseError = AFMDwarfStarDownloadError.invalidResponse(
                "GGUF header probe did not return HTTP metadata")
            completionHandler(.cancel)
            return
        }
        guard http.statusCode == 206 else {
            responseError = AFMDwarfStarDownloadError.invalidResponse(
                "GGUF header probe did not honor its byte range (HTTP \(http.statusCode))")
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard data.count <= maximumBytes,
              received.count <= maximumBytes - data.count else {
            responseError = AFMDwarfStarDownloadError.invalidResponse(
                "GGUF header probe exceeded its byte limit")
            dataTask.cancel()
            return
        }
        received.append(data)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let result = responseError ?? error
        let data = received
        let continuation = lock.withLock { () -> CheckedContinuation<Data, Error>? in
            defer { self.continuation = nil }
            return self.continuation
        }
        if let result { continuation?.resume(throwing: result) }
        else { continuation?.resume(returning: data) }
    }
}

private final class CacheLocalDataDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let destination: URL
    private let offset: Int64
    private let expectedBytes: Int64
    private let progress: @Sendable (Int64) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Error>?
    private var handle: FileHandle?
    private var received: Int64 = 0
    private var responseError: Error?

    init(
        destination: URL,
        offset: Int64,
        expectedBytes: Int64,
        progress: @escaping @Sendable (Int64) -> Void
    ) {
        self.destination = destination
        self.offset = offset
        self.expectedBytes = expectedBytes
        self.progress = progress
    }

    func run(request: URLRequest, session: URLSession) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.withLock { self.continuation = continuation }
                session.dataTask(with: request).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            responseError = AFMDwarfStarDownloadError.invalidResponse("GET did not return HTTP metadata")
            completionHandler(.cancel)
            return
        }
        guard (200 ... 299).contains(http.statusCode) else {
            responseError = AFMDwarfStarDownloadError.invalidResponse("GET returned HTTP \(http.statusCode)")
            completionHandler(.cancel)
            return
        }
        guard offset == 0 || http.statusCode == 206 else {
            responseError = AFMDwarfStarDownloadError.rangeNotHonored(
                expectedOffset: offset,
                statusCode: http.statusCode)
            completionHandler(.cancel)
            return
        }
        do {
            if !FileManager.default.fileExists(atPath: destination.path) {
                FileManager.default.createFile(atPath: destination.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: destination)
            try handle.seekToEnd()
            self.handle = handle
            completionHandler(.allow)
        } catch {
            responseError = error
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle?.write(contentsOf: data)
            received += Int64(data.count)
            progress(min(offset + received, expectedBytes))
        } catch {
            responseError = error
            dataTask.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        try? handle?.close()
        handle = nil
        let result = responseError ?? error
        let continuation = lock.withLock { () -> CheckedContinuation<Void, Error>? in
            defer { self.continuation = nil }
            return self.continuation
        }
        if let result { continuation?.resume(throwing: result) }
        else { continuation?.resume() }
    }
}
