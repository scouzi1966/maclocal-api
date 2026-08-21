import Darwin
import Foundation
import Network
import Security

struct AFMMLXPinnedHTTPSConnectionPlan: Equatable, Sendable {
    let validatedAddress: String
    let tlsServerName: String
    let hostHeader: String
    let requestTarget: String
}

protocol AFMMLXPinnedHTTPSConnectionDriver: AnyObject, Sendable {
    func run(request: Data, maximumBytes: Int) async throws -> AFMMLXRemoteMediaResponse
    func cancel()
}

enum AFMMLXPinnedHTTPSClient {
    typealias DriverFactory = @Sendable (
        AFMMLXPinnedHTTPSConnectionPlan
    ) -> any AFMMLXPinnedHTTPSConnectionDriver

    static func fetch(
        url: URL,
        validatedAddress: String,
        maximumBytes: Int,
        driverFactory: DriverFactory = { AFMMLXNWHTTPSConnectionDriver(plan: $0) }
    ) async throws -> AFMMLXRemoteMediaResponse {
        let plan = try connectionPlan(url: url, validatedAddress: validatedAddress)
        let driver = driverFactory(plan)
        let request = requestData(for: plan)

        return try await withThrowingTaskGroup(of: AFMMLXRemoteMediaResponse.self) { group in
            group.addTask {
                try await withTaskCancellationHandler {
                    try await driver.run(request: request, maximumBytes: maximumBytes)
                } onCancel: {
                    driver.cancel()
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: 30_000_000_000)
                throw AFMMLXMediaInputError.downloadFailed
            }
            defer {
                group.cancelAll()
                driver.cancel()
            }
            guard let response = try await group.next() else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            return response
        }
    }

    static func connectionPlan(
        url: URL,
        validatedAddress: String
    ) throws -> AFMMLXPinnedHTTPSConnectionPlan {
        guard let host = url.host, !host.isEmpty,
              isNumericIPAddress(validatedAddress)
        else {
            throw AFMMLXMediaInputError.invalidReference
        }
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        var target = components?.percentEncodedPath ?? url.path
        if target.isEmpty { target = "/" }
        if let query = components?.percentEncodedQuery, !query.isEmpty {
            target += "?\(query)"
        }
        guard !target.contains("\r"), !target.contains("\n"),
              !host.contains("\r"), !host.contains("\n")
        else {
            throw AFMMLXMediaInputError.invalidReference
        }
        let hostHeader = host.contains(":") ? "[\(host)]" : host
        return AFMMLXPinnedHTTPSConnectionPlan(
            validatedAddress: validatedAddress,
            tlsServerName: host,
            hostHeader: hostHeader,
            requestTarget: target
        )
    }

    private static func isNumericIPAddress(_ address: String) -> Bool {
        var ipv4 = in_addr()
        if inet_pton(AF_INET, address, &ipv4) == 1 { return true }
        var ipv6 = in6_addr()
        return inet_pton(AF_INET6, address, &ipv6) == 1
    }

    private static func requestData(for plan: AFMMLXPinnedHTTPSConnectionPlan) -> Data {
        Data(
            """
            GET \(plan.requestTarget) HTTP/1.1\r
            Host: \(plan.hostHeader)\r
            Accept: image/*, video/mp4, video/quicktime, video/webm\r
            Accept-Encoding: identity\r
            User-Agent: maclocal-api-media/1\r
            Connection: close\r
            \r

            """.utf8
        )
    }
}

private final class AFMMLXNWHTTPSConnectionDriver:
    AFMMLXPinnedHTTPSConnectionDriver,
    @unchecked Sendable
{
    private let connection: NWConnection
    private let queue = DispatchQueue(label: "afm.mlx.pinned-media")
    private let lock = NSLock()
    private var continuation: CheckedContinuation<AFMMLXRemoteMediaResponse, Error>?
    private var completed = false
    private var cancelled = false
    private var parser: AFMMLXHTTPResponseParser?

    init(plan: AFMMLXPinnedHTTPSConnectionPlan) {
        let tls = NWProtocolTLS.Options()
        sec_protocol_options_set_tls_server_name(
            tls.securityProtocolOptions,
            plan.tlsServerName
        )
        sec_protocol_options_add_tls_application_protocol(
            tls.securityProtocolOptions,
            "http/1.1"
        )
        let parameters = NWParameters(tls: tls, tcp: NWProtocolTCP.Options())
        parameters.preferNoProxies = true
        connection = NWConnection(
            host: NWEndpoint.Host(plan.validatedAddress),
            port: .https,
            using: parameters
        )
    }

    func run(
        request: Data,
        maximumBytes: Int
    ) async throws -> AFMMLXRemoteMediaResponse {
        try await withCheckedThrowingContinuation { continuation in
            let shouldCancel = lock.withLock { () -> Bool in
                guard !cancelled else { return true }
                self.continuation = continuation
                parser = AFMMLXHTTPResponseParser(maximumBytes: maximumBytes)
                return false
            }
            guard !shouldCancel else {
                continuation.resume(throwing: CancellationError())
                return
            }

            connection.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    self.send(request)
                case .failed:
                    self.finish(throwing: AFMMLXMediaInputError.downloadFailed)
                case .cancelled:
                    if self.lock.withLock({ self.cancelled }) {
                        self.finish(throwing: CancellationError())
                    }
                default:
                    break
                }
            }
            connection.start(queue: queue)
        }
    }

    func cancel() {
        lock.withLock { cancelled = true }
        connection.cancel()
        finish(throwing: CancellationError())
    }

    private func send(_ request: Data) {
        connection.send(content: request, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            if error != nil {
                self.finish(throwing: AFMMLXMediaInputError.downloadFailed)
            } else {
                self.receive()
            }
        })
    }

    private func receive() {
        connection.receive(
            minimumIncompleteLength: 1,
            maximumLength: 64 * 1_024
        ) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if error != nil {
                self.finish(throwing: AFMMLXMediaInputError.downloadFailed)
                return
            }
            do {
                guard var parser = self.parser else {
                    throw AFMMLXMediaInputError.downloadFailed
                }
                let response = try parser.consume(data ?? Data(), isComplete: isComplete)
                self.parser = parser
                if let response {
                    self.finish(returning: response)
                } else if isComplete {
                    self.finish(throwing: AFMMLXMediaInputError.downloadFailed)
                } else {
                    self.receive()
                }
            } catch {
                self.finish(throwing: error)
            }
        }
    }

    private func finish(returning response: AFMMLXRemoteMediaResponse) {
        finish(.success(response))
    }

    private func finish(throwing error: Error) {
        finish(.failure(error))
    }

    private func finish(_ result: Result<AFMMLXRemoteMediaResponse, Error>) {
        let continuation = lock.withLock { () -> CheckedContinuation<
            AFMMLXRemoteMediaResponse,
            Error
        >? in
            guard !completed else { return nil }
            completed = true
            let continuation = self.continuation
            self.continuation = nil
            return continuation
        }
        guard let continuation else { return }
        connection.cancel()
        continuation.resume(with: result)
    }
}

private struct AFMMLXHTTPResponseParser {
    private enum BodyMode {
        case undecided
        case fixed(Int)
        case chunked
        case untilClose
    }

    private let maximumBytes: Int
    private var buffer = Data()
    private var body = Data()
    private var statusCode: Int?
    private var mimeType: String?
    private var contentLength: Int64?
    private var redirectLocation: String?
    private var bodyMode: BodyMode = .undecided
    private var chunkBytesRemaining: Int?

    init(maximumBytes: Int) {
        self.maximumBytes = maximumBytes
    }

    mutating func consume(
        _ data: Data,
        isComplete: Bool
    ) throws -> AFMMLXRemoteMediaResponse? {
        guard buffer.count <= maximumBytes + 64 * 1_024 - data.count else {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        buffer.append(data)
        if statusCode == nil {
            try parseHeadersIfAvailable()
        }
        guard let statusCode else {
            if isComplete { throw AFMMLXMediaInputError.downloadFailed }
            return nil
        }
        if !(200..<300).contains(statusCode) {
            return response(statusCode: statusCode)
        }

        switch bodyMode {
        case .undecided:
            throw AFMMLXMediaInputError.downloadFailed
        case .fixed(let expected):
            guard buffer.count <= expected else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            if buffer.count == expected {
                body = buffer
                buffer.removeAll(keepingCapacity: false)
                return response(statusCode: statusCode)
            }
            if isComplete { throw AFMMLXMediaInputError.downloadFailed }
        case .chunked:
            if try consumeChunks() {
                return response(statusCode: statusCode)
            }
            if isComplete { throw AFMMLXMediaInputError.downloadFailed }
        case .untilClose:
            try appendBody(buffer)
            buffer.removeAll(keepingCapacity: true)
            if isComplete {
                return response(statusCode: statusCode)
            }
        }
        return nil
    }

    private mutating func parseHeadersIfAvailable() throws {
        let delimiter = Data([13, 10, 13, 10])
        guard let range = buffer.range(of: delimiter) else {
            guard buffer.count <= 64 * 1_024 else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            return
        }
        let headerData = buffer[..<range.lowerBound]
        buffer.removeSubrange(..<range.upperBound)
        guard let raw = String(data: headerData, encoding: .isoLatin1) else {
            throw AFMMLXMediaInputError.downloadFailed
        }
        let lines = raw.components(separatedBy: "\r\n")
        guard let statusLine = lines.first else {
            throw AFMMLXMediaInputError.downloadFailed
        }
        let statusParts = statusLine.split(separator: " ", maxSplits: 2)
        guard statusParts.count >= 2,
              statusParts[0].hasPrefix("HTTP/1."),
              let status = Int(statusParts[1])
        else {
            throw AFMMLXMediaInputError.downloadFailed
        }

        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...]
                .trimmingCharacters(in: .whitespaces)
            guard headers[name] == nil else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            headers[name] = value
        }
        if let encoding = headers["content-encoding"]?.lowercased(),
           encoding != "identity" {
            throw AFMMLXMediaInputError.downloadFailed
        }
        let length: Int64?
        if let rawLength = headers["content-length"] {
            guard let parsed = Int64(rawLength), parsed >= 0 else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            length = parsed
        } else {
            length = nil
        }
        if let length, length > Int64(maximumBytes) {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        let transferEncoding = headers["transfer-encoding"]?.lowercased()
        if transferEncoding != nil, length != nil {
            throw AFMMLXMediaInputError.downloadFailed
        }
        if let transferEncoding, transferEncoding != "chunked" {
            throw AFMMLXMediaInputError.downloadFailed
        }

        statusCode = status
        mimeType = headers["content-type"]
        contentLength = length
        redirectLocation = headers["location"]
        if (200..<300).contains(status) {
            if transferEncoding == "chunked" {
                bodyMode = .chunked
            } else if let length {
                bodyMode = .fixed(Int(length))
            } else {
                bodyMode = .untilClose
            }
        }
    }

    private mutating func consumeChunks() throws -> Bool {
        let crlf = Data([13, 10])
        while true {
            if chunkBytesRemaining == nil {
                guard let lineRange = buffer.range(of: crlf) else {
                    guard buffer.count <= 128 else {
                        throw AFMMLXMediaInputError.downloadFailed
                    }
                    return false
                }
                let rawLine = buffer[..<lineRange.lowerBound]
                buffer.removeSubrange(..<lineRange.upperBound)
                guard let line = String(data: rawLine, encoding: .ascii),
                      let sizeToken = line.split(separator: ";", maxSplits: 1).first,
                      let size = Int(sizeToken.trimmingCharacters(in: .whitespaces), radix: 16),
                      size >= 0
                else {
                    throw AFMMLXMediaInputError.downloadFailed
                }
                if size == 0 { return true }
                guard size <= maximumBytes - body.count else {
                    throw AFMMLXMediaInputError.responseTooLarge
                }
                chunkBytesRemaining = size
            }

            guard let remaining = chunkBytesRemaining,
                  buffer.count >= remaining + 2
            else { return false }
            let payload = buffer.prefix(remaining)
            let suffix = buffer.dropFirst(remaining).prefix(2)
            guard suffix.elementsEqual(crlf) else {
                throw AFMMLXMediaInputError.downloadFailed
            }
            try appendBody(payload)
            buffer.removeFirst(remaining + 2)
            chunkBytesRemaining = nil
        }
    }

    private mutating func appendBody<T: DataProtocol>(_ data: T) throws {
        guard data.count <= maximumBytes - body.count else {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        body.append(contentsOf: data)
    }

    private func response(statusCode: Int) -> AFMMLXRemoteMediaResponse {
        AFMMLXRemoteMediaResponse(
            statusCode: statusCode,
            mimeType: mimeType,
            contentLength: contentLength,
            data: body,
            redirectLocation: redirectLocation
        )
    }
}
