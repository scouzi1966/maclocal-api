import Darwin
import Foundation
import AFMOpenAICompat

public enum AFMMLXMediaInputError: Error, Equatable, LocalizedError, Sendable {
    case invalidReference
    case unsupportedScheme(String)
    case remoteHostNotAllowed
    case remotePortNotAllowed
    case remoteAddressNotAllowed
    case redirectNotAllowed
    case tooManyRedirects
    case invalidDataURL
    case unsupportedMIMEType(String)
    case responseTooLarge
    case downloadFailed

    public var errorDescription: String? {
        switch self {
        case .invalidReference:
            return "image_url is not a valid media reference"
        case .unsupportedScheme(let scheme):
            return "image_url scheme '\(scheme)' is not allowed"
        case .remoteHostNotAllowed:
            return "image_url host is not allowed"
        case .remotePortNotAllowed:
            return "image_url must use the default HTTPS port"
        case .remoteAddressNotAllowed:
            return "image_url resolves to a non-public network address"
        case .redirectNotAllowed:
            return "image_url redirect target is not allowed"
        case .tooManyRedirects:
            return "image_url exceeded the redirect limit"
        case .invalidDataURL:
            return "image_url data URL must contain strict base64 media data"
        case .unsupportedMIMEType(let mimeType):
            return "image_url MIME type '\(mimeType)' is not supported"
        case .responseTooLarge:
            return "image_url exceeds the maximum allowed media size"
        case .downloadFailed:
            return "image_url download failed"
        }
    }
}

public enum AFMMLXMediaPayloadKind: Equatable, Sendable {
    case image
    case video
}

public struct AFMMLXMediaPayload: Sendable {
    public let data: Data
    public let mimeType: String
    public let kind: AFMMLXMediaPayloadKind
    public let sourceURL: URL?
}

struct AFMMLXRemoteMediaResponse: Sendable {
    let statusCode: Int
    let mimeType: String?
    let contentLength: Int64?
    let data: Data
    let redirectLocation: String?
}

public enum AFMMLXMediaSecurityPolicy {
    public static let maximumMediaBytes = 20 * 1_024 * 1_024
    static let maximumRedirects = 3

    typealias HostResolver = @Sendable (String) throws -> [String]
    typealias RemoteTransport = @Sendable (URL, Int) throws -> AFMMLXRemoteMediaResponse

    private static let imageMIMETypes: Set<String> = [
        "image/gif", "image/heic", "image/heif", "image/jpeg", "image/png", "image/webp",
    ]
    private static let videoMIMETypes: Set<String> = [
        "video/mp4", "video/quicktime", "video/webm",
    ]
    private static let blockedHostSuffixes = [
        ".internal", ".lan", ".local", ".localhost", ".home",
    ]

    public static func validateReferences(
        in messages: [AFMOpenAICompat.Message]
    ) throws {
        for message in messages {
            guard let content = message.content, case .parts(let parts) = content else {
                continue
            }
            for part in parts where part.type == "image_url" {
                guard let raw = part.image_url?.url else {
                    throw AFMMLXMediaInputError.invalidReference
                }
                if raw.lowercased().hasPrefix("data:") {
                    _ = try decodeDataURL(raw)
                } else {
                    guard let url = URL(string: raw) else {
                        throw AFMMLXMediaInputError.invalidReference
                    }
                    try validateRemoteURL(url, resolver: resolveHost)
                }
            }
        }
    }

    public static func load(_ raw: String) throws -> AFMMLXMediaPayload {
        if raw.lowercased().hasPrefix("data:") {
            return try decodeDataURL(raw)
        }
        guard let url = URL(string: raw) else {
            throw AFMMLXMediaInputError.invalidReference
        }
        return try loadRemote(
            url,
            resolver: resolveHost,
            transport: boundedHTTPSRequest
        )
    }

    /// Converts a host-selected local attachment into the same bounded data URL
    /// accepted from public requests. Callers must not expose this to raw API URLs.
    public static func trustedLocalMediaDataURL(_ url: URL) throws -> String {
        guard url.isFileURL else {
            throw AFMMLXMediaInputError.invalidReference
        }
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true,
              let fileSize = values.fileSize,
              fileSize > 0
        else {
            throw AFMMLXMediaInputError.invalidReference
        }
        guard fileSize <= maximumMediaBytes else {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        let mimeType = try localMIMEType(forExtension: url.pathExtension)
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard !data.isEmpty else {
            throw AFMMLXMediaInputError.invalidReference
        }
        guard data.count <= maximumMediaBytes else {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        return "data:\(mimeType);base64,\(data.base64EncodedString())"
    }

    static func decodeDataURL(_ raw: String) throws -> AFMMLXMediaPayload {
        guard let comma = raw.firstIndex(of: ",") else {
            throw AFMMLXMediaInputError.invalidDataURL
        }
        let header = raw[raw.startIndex..<comma]
        let components = header.dropFirst("data:".count).split(separator: ";")
        guard components.count == 2,
              components[1].lowercased() == "base64"
        else {
            throw AFMMLXMediaInputError.invalidDataURL
        }
        let mimeType = normalizedMIMEType(String(components[0]))
        let kind = try payloadKind(for: mimeType)
        let encoded = String(raw[raw.index(after: comma)...])
        let maximumEncodedBytes = ((maximumMediaBytes + 2) / 3) * 4
        guard !encoded.isEmpty,
              encoded.utf8.count <= maximumEncodedBytes,
              let data = Data(base64Encoded: encoded),
              !data.isEmpty,
              data.count <= maximumMediaBytes
        else {
            throw encoded.utf8.count > maximumEncodedBytes
                ? AFMMLXMediaInputError.responseTooLarge
                : AFMMLXMediaInputError.invalidDataURL
        }
        return AFMMLXMediaPayload(
            data: data,
            mimeType: mimeType,
            kind: kind,
            sourceURL: nil
        )
    }

    static func loadRemote(
        _ initialURL: URL,
        resolver: HostResolver,
        transport: RemoteTransport
    ) throws -> AFMMLXMediaPayload {
        var url = initialURL
        for redirectCount in 0...maximumRedirects {
            try validateRemoteURL(url, resolver: resolver)
            let response = try transport(url, maximumMediaBytes)
            if (300..<400).contains(response.statusCode) {
                guard redirectCount < maximumRedirects else {
                    throw AFMMLXMediaInputError.tooManyRedirects
                }
                guard let location = response.redirectLocation,
                      let redirected = URL(string: location, relativeTo: url)?.absoluteURL
                else {
                    throw AFMMLXMediaInputError.redirectNotAllowed
                }
                do {
                    try validateRemoteURL(redirected, resolver: resolver)
                } catch {
                    throw AFMMLXMediaInputError.redirectNotAllowed
                }
                url = redirected
                continue
            }
            guard (200..<300).contains(response.statusCode) else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            if let contentLength = response.contentLength,
               contentLength < 0 || contentLength > Int64(maximumMediaBytes) {
                throw AFMMLXMediaInputError.responseTooLarge
            }
            guard response.data.count <= maximumMediaBytes else {
                throw AFMMLXMediaInputError.responseTooLarge
            }
            let mimeType = normalizedMIMEType(response.mimeType ?? "")
            let kind = try payloadKind(for: mimeType)
            return AFMMLXMediaPayload(
                data: response.data,
                mimeType: mimeType,
                kind: kind,
                sourceURL: url
            )
        }
        throw AFMMLXMediaInputError.tooManyRedirects
    }

    static func validateRemoteURL(
        _ url: URL,
        resolver: HostResolver
    ) throws {
        let scheme = url.scheme?.lowercased() ?? ""
        guard scheme == "https" else {
            throw AFMMLXMediaInputError.unsupportedScheme(scheme.isEmpty ? "none" : scheme)
        }
        guard url.user == nil, url.password == nil,
              let host = url.host?.lowercased(), !host.isEmpty,
              host != "localhost",
              !blockedHostSuffixes.contains(where: host.hasSuffix)
        else {
            throw AFMMLXMediaInputError.remoteHostNotAllowed
        }
        guard url.port == nil || url.port == 443 else {
            throw AFMMLXMediaInputError.remotePortNotAllowed
        }
        let addresses: [String]
        do {
            addresses = try resolver(host)
        } catch {
            throw AFMMLXMediaInputError.downloadFailed
        }
        guard !addresses.isEmpty,
              addresses.allSatisfy(isPublicIPAddress)
        else {
            throw AFMMLXMediaInputError.remoteAddressNotAllowed
        }
    }

    private static func payloadKind(
        for mimeType: String
    ) throws -> AFMMLXMediaPayloadKind {
        if imageMIMETypes.contains(mimeType) { return .image }
        if videoMIMETypes.contains(mimeType) { return .video }
        throw AFMMLXMediaInputError.unsupportedMIMEType(
            mimeType.isEmpty ? "missing" : mimeType
        )
    }

    private static func localMIMEType(forExtension rawExtension: String) throws -> String {
        let mimeType: String
        switch rawExtension.lowercased() {
        case "gif": mimeType = "image/gif"
        case "heic": mimeType = "image/heic"
        case "heif": mimeType = "image/heif"
        case "jpg", "jpeg": mimeType = "image/jpeg"
        case "png": mimeType = "image/png"
        case "webp": mimeType = "image/webp"
        case "mov": mimeType = "video/quicktime"
        case "mp4", "m4v": mimeType = "video/mp4"
        case "webm": mimeType = "video/webm"
        default:
            throw AFMMLXMediaInputError.unsupportedMIMEType(
                rawExtension.isEmpty ? "missing" : rawExtension.lowercased()
            )
        }
        _ = try payloadKind(for: mimeType)
        return mimeType
    }

    private static func normalizedMIMEType(_ value: String) -> String {
        value.split(separator: ";", maxSplits: 1).first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() ?? ""
    }

    private static func resolveHost(_ host: String) throws -> [String] {
        var hints = addrinfo()
        hints.ai_flags = AI_ADDRCONFIG
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        var result: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, nil, &hints, &result) == 0, let result else {
            throw AFMMLXMediaInputError.downloadFailed
        }
        defer { freeaddrinfo(result) }

        var addresses: [String] = []
        var current: UnsafeMutablePointer<addrinfo>? = result
        while let info = current?.pointee {
            var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(
                info.ai_addr,
                info.ai_addrlen,
                &buffer,
                socklen_t(buffer.count),
                nil,
                0,
                NI_NUMERICHOST
            ) == 0 {
                let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
                addresses.append(String(decoding: bytes, as: UTF8.self))
            }
            current = info.ai_next
        }
        return Array(Set(addresses))
    }

    private static func isPublicIPAddress(_ address: String) -> Bool {
        var ipv4 = in_addr()
        if inet_pton(AF_INET, address, &ipv4) == 1 {
            let value = UInt32(bigEndian: ipv4.s_addr)
            let first = UInt8((value >> 24) & 0xff)
            let second = UInt8((value >> 16) & 0xff)
            let third = UInt8((value >> 8) & 0xff)
            if first == 0 || first == 10 || first == 127 || first >= 224 { return false }
            if first == 100 && (64...127).contains(second) { return false }
            if first == 169 && second == 254 { return false }
            if first == 172 && (16...31).contains(second) { return false }
            if first == 192 && second == 168 { return false }
            if first == 198 && (second == 18 || second == 19) { return false }
            if first == 192 && second == 0 && (third == 0 || third == 2) { return false }
            if first == 192 && second == 88 && third == 99 { return false }
            if first == 198 && second == 51 && third == 100 { return false }
            if first == 203 && second == 0 && third == 113 { return false }
            return true
        }

        var ipv6 = in6_addr()
        guard inet_pton(AF_INET6, address, &ipv6) == 1 else { return false }
        let bytes = withUnsafeBytes(of: &ipv6) { Array($0) }
        if bytes.prefix(10).allSatisfy({ $0 == 0 }), bytes[10] == 0xff, bytes[11] == 0xff {
            return isPublicIPAddress("\(bytes[12]).\(bytes[13]).\(bytes[14]).\(bytes[15])")
        }
        guard (0x20...0x3f).contains(bytes[0]) else { return false }
        if bytes[0] == 0x20, bytes[1] == 0x01, bytes[2] == 0x0d, bytes[3] == 0xb8 {
            return false
        }
        return true
    }

    private static func boundedHTTPSRequest(
        _ url: URL,
        maximumBytes: Int
    ) throws -> AFMMLXRemoteMediaResponse {
        let delegate = AFMMLXBoundedMediaDelegate(maximumBytes: maximumBytes)
        return try delegate.run(url: url)
    }
}

private final class AFMMLXBoundedMediaDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let maximumBytes: Int
    private let lock = NSLock()
    private let semaphore = DispatchSemaphore(value: 0)
    private var data = Data()
    private var response: HTTPURLResponse?
    private var redirectResponse: HTTPURLResponse?
    private var failure: Error?
    private var didFinish = false

    init(maximumBytes: Int) {
        self.maximumBytes = maximumBytes
    }

    func run(url: URL) throws -> AFMMLXRemoteMediaResponse {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        configuration.httpCookieAcceptPolicy = .never
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("image/*, video/mp4, video/quicktime, video/webm", forHTTPHeaderField: "Accept")
        let task = session.dataTask(with: request)
        task.resume()
        semaphore.wait()
        session.finishTasksAndInvalidate()

        if let failure { throw failure }
        guard let response = redirectResponse ?? response else {
            throw AFMMLXMediaInputError.downloadFailed
        }
        return AFMMLXRemoteMediaResponse(
            statusCode: response.statusCode,
            mimeType: response.mimeType,
            contentLength: response.expectedContentLength >= 0
                ? response.expectedContentLength : nil,
            data: data,
            redirectLocation: response.value(forHTTPHeaderField: "Location")
        )
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        lock.withLock { redirectResponse = response }
        completionHandler(nil)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            lock.withLock { failure = AFMMLXMediaInputError.downloadFailed }
            completionHandler(.cancel)
            return
        }
        if http.expectedContentLength > Int64(maximumBytes) {
            lock.withLock { failure = AFMMLXMediaInputError.responseTooLarge }
            completionHandler(.cancel)
            return
        }
        lock.withLock { self.response = http }
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive chunk: Data
    ) {
        let shouldCancel = lock.withLock { () -> Bool in
            guard data.count <= maximumBytes - chunk.count else {
                failure = AFMMLXMediaInputError.responseTooLarge
                return true
            }
            data.append(chunk)
            return false
        }
        if shouldCancel { dataTask.cancel() }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        lock.withLock {
            if failure == nil, let error { failure = error }
            guard !didFinish else { return }
            didFinish = true
            semaphore.signal()
        }
    }
}
