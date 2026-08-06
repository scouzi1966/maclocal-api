import Foundation

public struct AFMMLXRemoteImageCachePlan: Equatable, Sendable {
    public let sourceURL: URL
    public let cacheDirectory: URL
    public let destinationURL: URL

    public init(
        sourceURL: URL,
        cacheDirectory: URL,
        destinationURL: URL
    ) {
        self.sourceURL = sourceURL
        self.cacheDirectory = cacheDirectory
        self.destinationURL = destinationURL
    }
}

public enum AFMMLXImageInputPlan: Equatable, Sendable {
    case none
    case localFile(URL)
    case remoteImage(AFMMLXRemoteImageCachePlan)

    public var hasImages: Bool {
        switch self {
        case .none:
            false
        case .localFile, .remoteImage:
            true
        }
    }
}

public enum AFMMLXImageInputPolicy {
    public static func plan(
        imageURL: URL?,
        cacheDirectory: URL,
        uniqueSuffix: String
    ) -> AFMMLXImageInputPlan {
        guard let imageURL else { return .none }
        guard isRemoteImageURL(imageURL) else { return .localFile(imageURL) }
        return .remoteImage(
            AFMMLXRemoteImageCachePlan(
                sourceURL: imageURL,
                cacheDirectory: cacheDirectory,
                destinationURL: cacheDirectory.appendingPathComponent(
                    cacheFileName(for: imageURL, uniqueSuffix: uniqueSuffix)
                )
            )
        )
    }

    public static func isRemoteImageURL(_ url: URL) -> Bool {
        switch url.scheme?.lowercased() {
        case "http", "https":
            true
        default:
            false
        }
    }

    public static func cacheFileName(
        for url: URL,
        uniqueSuffix: String
    ) -> String {
        let sanitizedSuffix = uniqueSuffix
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "/", with: "-")
        let suffix = sanitizedSuffix.isEmpty ? "image" : sanitizedSuffix
        return "WebImage_\(suffix).\(fileExtension(for: url))"
    }

    public static func fileExtension(for url: URL) -> String {
        let pathExtension = url.pathExtension
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return pathExtension.isEmpty ? "jpg" : pathExtension
    }
}
