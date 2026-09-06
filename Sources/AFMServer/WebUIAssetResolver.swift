import Foundation

/// Resolves WebUI request paths within one immutable static build directory.
struct WebUIAssetResolver: Equatable {
    enum Resolution: Equatable {
        case asset(url: URL, mimeType: String, cacheControl: String)
        case index
    }

    private let rootURL: URL

    init(rootURL: URL) {
        self.rootURL = rootURL.resolvingSymlinksInPath()
    }

    func resolve(path: String) -> Resolution? {
        guard let decodedPath = path.removingPercentEncoding else {
            return nil
        }

        let components = decodedPath.split(separator: "/").map(String.init)
        guard !components.contains(where: { $0 == ".." || $0 == "." || $0.contains("\\") }) else {
            return nil
        }

        guard !components.isEmpty,
              !(components.count == 1 && components[0].lowercased() == "index.html") else {
            return .index
        }

        let unresolvedAssetURL = components.reduce(rootURL) { $0.appendingPathComponent($1) }
        let assetURL = unresolvedAssetURL.resolvingSymlinksInPath()
        let rootPath = rootURL.path
        guard assetURL.path.hasPrefix(rootPath + "/") else {
            return nil
        }

        let resourceValues = try? assetURL.resourceValues(forKeys: [.isRegularFileKey])
        guard resourceValues?.isRegularFile == true else {
            return nil
        }

        return .asset(
            url: assetURL,
            mimeType: Self.mimeType(forExtension: assetURL.pathExtension.lowercased()),
            cacheControl: Self.cacheControl(forComponents: components)
        )
    }

    static func mimeType(forExtension extension: String) -> String {
        switch `extension` {
        case "html": return "text/html; charset=utf-8"
        case "css": return "text/css; charset=utf-8"
        case "js", "mjs": return "application/javascript; charset=utf-8"
        case "json", "map": return "application/json; charset=utf-8"
        case "webmanifest": return "application/manifest+json"
        case "svg": return "image/svg+xml"
        case "png": return "image/png"
        case "jpg", "jpeg": return "image/jpeg"
        case "ico": return "image/x-icon"
        case "txt": return "text/plain; charset=utf-8"
        case "woff": return "font/woff"
        case "woff2": return "font/woff2"
        default: return "application/octet-stream"
        }
    }

    static func cacheControl(forComponents components: [String]) -> String {
        components.count > 2 && components[0] == "_app" && components[1] == "immutable"
            ? "public, max-age=31536000, immutable"
            : "no-cache"
    }
}
