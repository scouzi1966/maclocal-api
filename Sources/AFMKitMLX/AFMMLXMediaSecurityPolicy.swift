@preconcurrency import AVFoundation
import Darwin
import CoreImage
import Foundation
import ImageIO
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
    case tooManyMediaItems
    case aggregateMediaTooLarge
    case imagePixelLimitExceeded
    case videoDurationLimitExceeded
    case videoFrameLimitExceeded
    case mediaInspectionFailed
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
        case .tooManyMediaItems:
            return "request exceeds the maximum allowed media item count"
        case .aggregateMediaTooLarge:
            return "request exceeds the aggregate media byte limit"
        case .imagePixelLimitExceeded:
            return "request exceeds the aggregate decoded image pixel limit"
        case .videoDurationLimitExceeded:
            return "request exceeds the aggregate video duration limit"
        case .videoFrameLimitExceeded:
            return "request exceeds the aggregate video frame limit"
        case .mediaInspectionFailed:
            return "image_url media payload could not be safely inspected"
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

public struct AFMMLXResolvedMediaRequest: Sendable {
    public let messages: [AFMOpenAICompat.Message]
    public let mediaKinds: [AFMMLXRequestMediaKind]
}

struct AFMMLXMediaRequestLimits: Sendable {
    let maximumItems: Int
    let maximumItemBytes: Int
    let maximumAggregateBytes: Int
    let maximumImagePixels: Int64
    let maximumVideoDuration: Double
    let maximumVideoFrames: Int

    static let production = AFMMLXMediaRequestLimits(
        maximumItems: 8,
        maximumItemBytes: 20 * 1_024 * 1_024,
        maximumAggregateBytes: 40 * 1_024 * 1_024,
        maximumImagePixels: 64 * 1_024 * 1_024,
        maximumVideoDuration: 120,
        maximumVideoFrames: 3_600
    )
}

struct AFMMLXMediaInspection: Sendable {
    let imagePixels: Int64
    let videoDuration: Double
    let videoFrames: Int
}

public enum AFMMLXMediaSecurityPolicy {
    public static let maximumMediaBytes = 20 * 1_024 * 1_024
    public static let maximumMediaItems = 8
    public static let maximumAggregateMediaBytes = 40 * 1_024 * 1_024
    public static let maximumAggregateImagePixels: Int64 = 64 * 1_024 * 1_024
    public static let maximumAggregateVideoDuration: Double = 120
    public static let maximumAggregateVideoFrames = 3_600
    static let maximumRedirects = 3

    typealias HostResolver = @Sendable (String) throws -> [String]
    typealias RemoteTransport = @Sendable (
        URL,
        String,
        Int
    ) async throws -> AFMMLXRemoteMediaResponse
    typealias PayloadInspector = @Sendable (
        AFMMLXMediaPayload,
        AFMMLXMediaRequestLimits
    ) async throws -> AFMMLXMediaInspection

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
        var mediaCount = 0
        for message in messages {
            guard let content = message.content, case .parts(let parts) = content else {
                continue
            }
            for part in parts where part.type == "image_url" || part.type == "input_audio" {
                mediaCount += 1
                guard mediaCount <= maximumMediaItems else {
                    throw AFMMLXMediaInputError.tooManyMediaItems
                }
                guard part.type == "image_url" else { continue }
                guard let raw = part.image_url?.url else {
                    throw AFMMLXMediaInputError.invalidReference
                }
                if raw.lowercased().hasPrefix("data:") {
                    _ = try decodeDataURL(raw)
                } else {
                    guard let url = URL(string: raw) else {
                        throw AFMMLXMediaInputError.invalidReference
                    }
                    try validateRemoteURLShape(url)
                }
            }
        }
    }

    public static func resolveRequest(
        in messages: [AFMOpenAICompat.Message]
    ) async throws -> AFMMLXResolvedMediaRequest {
        try await resolveRequest(
            in: messages,
            limits: .production,
            resolver: resolveHost,
            transport: boundedHTTPSRequest,
            inspector: inspectPayload
        )
    }

    static func resolveRequest(
        in messages: [AFMOpenAICompat.Message],
        limits: AFMMLXMediaRequestLimits,
        resolver: HostResolver,
        transport: RemoteTransport,
        inspector: PayloadInspector
    ) async throws -> AFMMLXResolvedMediaRequest {
        var mediaCount = 0
        var aggregateBytes = 0
        var aggregatePixels: Int64 = 0
        var aggregateVideoDuration: Double = 0
        var aggregateVideoFrames = 0
        var mediaKinds: [AFMMLXRequestMediaKind] = []
        var resolvedMessages: [AFMOpenAICompat.Message] = []

        for message in messages {
            try Task.checkCancellation()
            guard let content = message.content, case .parts(let parts) = content else {
                resolvedMessages.append(message)
                continue
            }
            var resolvedParts: [ContentPart] = []
            resolvedParts.reserveCapacity(parts.count)
            for part in parts {
                try Task.checkCancellation()
                if part.type == "input_audio" {
                    mediaCount += 1
                    guard mediaCount <= limits.maximumItems else {
                        throw AFMMLXMediaInputError.tooManyMediaItems
                    }
                    guard let encoded = part.input_audio?.data else {
                        throw AFMMLXMediaInputError.invalidDataURL
                    }
                    let audioData = try decodeBoundedBase64(
                        encoded,
                        maximumBytes: limits.maximumItemBytes
                    )
                    guard audioData.count <= limits.maximumAggregateBytes - aggregateBytes else {
                        throw AFMMLXMediaInputError.aggregateMediaTooLarge
                    }
                    aggregateBytes += audioData.count
                    mediaKinds.append(.audio)
                    resolvedParts.append(part)
                    continue
                }
                guard part.type == "image_url" else {
                    resolvedParts.append(part)
                    continue
                }
                mediaCount += 1
                guard mediaCount <= limits.maximumItems else {
                    throw AFMMLXMediaInputError.tooManyMediaItems
                }
                guard let raw = part.image_url?.url else {
                    throw AFMMLXMediaInputError.invalidReference
                }
                let payload = try await load(
                    raw,
                    maximumBytes: limits.maximumItemBytes,
                    resolver: resolver,
                    transport: transport
                )
                guard payload.data.count <= limits.maximumAggregateBytes - aggregateBytes else {
                    throw AFMMLXMediaInputError.aggregateMediaTooLarge
                }
                aggregateBytes += payload.data.count

                let inspection = try await inspector(payload, limits)
                guard inspection.imagePixels <= limits.maximumImagePixels - aggregatePixels else {
                    throw AFMMLXMediaInputError.imagePixelLimitExceeded
                }
                aggregatePixels += inspection.imagePixels
                guard inspection.videoDuration
                    <= limits.maximumVideoDuration - aggregateVideoDuration
                else {
                    throw AFMMLXMediaInputError.videoDurationLimitExceeded
                }
                aggregateVideoDuration += inspection.videoDuration
                guard inspection.videoFrames
                    <= limits.maximumVideoFrames - aggregateVideoFrames
                else {
                    throw AFMMLXMediaInputError.videoFrameLimitExceeded
                }
                aggregateVideoFrames += inspection.videoFrames

                let kind: AFMMLXRequestMediaKind = payload.kind == .image ? .image : .video
                mediaKinds.append(kind)
                let canonicalURL = "data:\(payload.mimeType);base64,"
                    + payload.data.base64EncodedString()
                resolvedParts.append(
                    ContentPart(
                        type: "image_url",
                        image_url: ImageURL(
                            url: canonicalURL,
                            detail: part.image_url?.detail
                        )
                    )
                )
            }
            resolvedMessages.append(
                AFMOpenAICompat.Message(
                    role: message.role,
                    content: .parts(resolvedParts),
                    toolCalls: message.toolCalls,
                    toolCallId: message.toolCallId,
                    name: message.name
                )
            )
        }
        return AFMMLXResolvedMediaRequest(
            messages: resolvedMessages,
            mediaKinds: mediaKinds
        )
    }

    public static func load(_ raw: String) async throws -> AFMMLXMediaPayload {
        try await load(
            raw,
            maximumBytes: maximumMediaBytes,
            resolver: resolveHost,
            transport: boundedHTTPSRequest
        )
    }

    private static func load(
        _ raw: String,
        maximumBytes: Int,
        resolver: HostResolver,
        transport: RemoteTransport
    ) async throws -> AFMMLXMediaPayload {
        if raw.lowercased().hasPrefix("data:") {
            return try decodeDataURL(raw, maximumBytes: maximumBytes)
        }
        guard let url = URL(string: raw) else {
            throw AFMMLXMediaInputError.invalidReference
        }
        return try await loadRemote(
            url,
            maximumBytes: maximumBytes,
            resolver: resolver,
            transport: transport
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

    static func decodeDataURL(
        _ raw: String,
        maximumBytes: Int = maximumMediaBytes
    ) throws -> AFMMLXMediaPayload {
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
        let data = try decodeBoundedBase64(encoded, maximumBytes: maximumBytes)
        return AFMMLXMediaPayload(
            data: data,
            mimeType: mimeType,
            kind: kind,
            sourceURL: nil
        )
    }

    private static func decodeBoundedBase64(
        _ encoded: String,
        maximumBytes: Int
    ) throws -> Data {
        let maximumEncodedBytes = ((maximumBytes + 2) / 3) * 4
        guard !encoded.isEmpty else { throw AFMMLXMediaInputError.invalidDataURL }
        guard encoded.utf8.count <= maximumEncodedBytes else {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        guard let data = Data(base64Encoded: encoded), !data.isEmpty else {
            throw AFMMLXMediaInputError.invalidDataURL
        }
        guard data.count <= maximumBytes else {
            throw AFMMLXMediaInputError.responseTooLarge
        }
        return data
    }

    static func loadRemote(
        _ initialURL: URL,
        maximumBytes: Int = maximumMediaBytes,
        resolver: HostResolver,
        transport: RemoteTransport
    ) async throws -> AFMMLXMediaPayload {
        var url = initialURL
        for redirectCount in 0...maximumRedirects {
            try Task.checkCancellation()
            let addresses = try validatedRemoteAddresses(url, resolver: resolver)
            var response: AFMMLXRemoteMediaResponse?
            var lastFailure: Error?
            for address in addresses.prefix(4) {
                do {
                    response = try await transport(url, address, maximumBytes)
                    break
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    lastFailure = error
                }
            }
            guard let response else {
                throw lastFailure ?? AFMMLXMediaInputError.downloadFailed
            }
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
                    _ = try validatedRemoteAddresses(redirected, resolver: resolver)
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
               contentLength < 0 || contentLength > Int64(maximumBytes) {
                throw AFMMLXMediaInputError.responseTooLarge
            }
            guard response.data.count <= maximumBytes else {
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
        _ = try validatedRemoteAddresses(url, resolver: resolver)
    }

    private static func validateRemoteURLShape(_ url: URL) throws {
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
    }

    private static func validatedRemoteAddresses(
        _ url: URL,
        resolver: HostResolver
    ) throws -> [String] {
        try validateRemoteURLShape(url)
        guard let host = url.host?.lowercased() else {
            throw AFMMLXMediaInputError.remoteHostNotAllowed
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
        return Array(Set(addresses)).sorted()
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

    private static func inspectPayload(
        _ payload: AFMMLXMediaPayload,
        limits: AFMMLXMediaRequestLimits
    ) async throws -> AFMMLXMediaInspection {
        switch payload.kind {
        case .image:
            guard let source = CGImageSourceCreateWithData(payload.data as CFData, nil) else {
                throw AFMMLXMediaInputError.mediaInspectionFailed
            }
            let frameCount = CGImageSourceGetCount(source)
            guard frameCount > 0, frameCount <= 256,
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                    as? [CFString: Any],
                  let width = integerProperty(properties[kCGImagePropertyPixelWidth]),
                  let height = integerProperty(properties[kCGImagePropertyPixelHeight]),
                  width > 0, height > 0
            else {
                throw AFMMLXMediaInputError.mediaInspectionFailed
            }
            let (pixelsPerFrame, overflow) = Int64(width).multipliedReportingOverflow(
                by: Int64(height)
            )
            guard !overflow else {
                throw AFMMLXMediaInputError.imagePixelLimitExceeded
            }
            let (pixels, frameOverflow) = pixelsPerFrame.multipliedReportingOverflow(
                by: Int64(frameCount)
            )
            guard !frameOverflow, pixels <= limits.maximumImagePixels else {
                throw AFMMLXMediaInputError.imagePixelLimitExceeded
            }
            guard CIImage(data: payload.data) != nil else {
                throw AFMMLXMediaInputError.mediaInspectionFailed
            }
            return AFMMLXMediaInspection(
                imagePixels: pixels,
                videoDuration: 0,
                videoFrames: 0
            )
        case .video:
            let fileExtension = mediaFileExtension(for: payload.mimeType)
            let temp = FileManager.default.temporaryDirectory.appendingPathComponent(
                "afm_media_inspect_\(UUID().uuidString).\(fileExtension)"
            )
            defer { try? FileManager.default.removeItem(at: temp) }
            do {
                try payload.data.write(to: temp, options: .atomic)
                let asset = AVURLAsset(url: temp)
                let duration = try await asset.load(.duration).seconds
                guard duration.isFinite, duration > 0,
                      duration <= limits.maximumVideoDuration
                else {
                    throw AFMMLXMediaInputError.videoDurationLimitExceeded
                }
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard !tracks.isEmpty else {
                    throw AFMMLXMediaInputError.mediaInspectionFailed
                }
                var frameCount = 0
                let reader = try AVAssetReader(asset: asset)
                for track in tracks {
                    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
                    output.alwaysCopiesSampleData = false
                    guard reader.canAdd(output) else {
                        throw AFMMLXMediaInputError.mediaInspectionFailed
                    }
                    reader.add(output)
                }
                guard reader.startReading() else {
                    throw AFMMLXMediaInputError.mediaInspectionFailed
                }
                for output in reader.outputs {
                    while output.copyNextSampleBuffer() != nil {
                        try Task.checkCancellation()
                        frameCount += 1
                        if frameCount > limits.maximumVideoFrames {
                            reader.cancelReading()
                            throw AFMMLXMediaInputError.videoFrameLimitExceeded
                        }
                    }
                }
                guard reader.status == .completed else {
                    throw AFMMLXMediaInputError.mediaInspectionFailed
                }
                return AFMMLXMediaInspection(
                    imagePixels: 0,
                    videoDuration: duration,
                    videoFrames: frameCount
                )
            } catch let error as AFMMLXMediaInputError {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw AFMMLXMediaInputError.mediaInspectionFailed
            }
        }
    }

    private static func integerProperty(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        return (value as? NSNumber)?.intValue
    }

    static func mediaFileExtension(for mimeType: String) -> String {
        switch mimeType {
        case "video/quicktime": "mov"
        case "video/webm": "webm"
        default: "mp4"
        }
    }

    private static func boundedHTTPSRequest(
        _ url: URL,
        validatedAddress: String,
        maximumBytes: Int
    ) async throws -> AFMMLXRemoteMediaResponse {
        try await AFMMLXPinnedHTTPSClient.fetch(
            url: url,
            validatedAddress: validatedAddress,
            maximumBytes: maximumBytes
        )
    }
}
