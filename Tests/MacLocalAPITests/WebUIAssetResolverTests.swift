@testable import AFMServer
import Foundation
import XCTest

final class WebUIAssetResolverTests: XCTestCase {
    private var workDirectory: URL!

    override func setUpWithError() throws {
        workDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-webui-resolver-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: workDirectory)
    }

    func testResolvesAssetsWithMIMETypeAndExactImmutableCacheScope() throws {
        let root = try makeFixture()
        let resolver = WebUIAssetResolver(rootURL: root)

        let bundle = try XCTUnwrap(
            resolver.resolve(path: "/_app/immutable/bundle.abc123.js")
        )
        guard case let .asset(bundleURL, bundleMIME, bundleCache) = bundle else {
            return XCTFail("Expected immutable bundle asset")
        }
        XCTAssertEqual(bundleURL.lastPathComponent, "bundle.abc123.js")
        XCTAssertEqual(bundleMIME, "application/javascript; charset=utf-8")
        XCTAssertEqual(bundleCache, "public, max-age=31536000, immutable")

        let version = try XCTUnwrap(resolver.resolve(path: "/_app/version.json"))
        guard case let .asset(_, versionMIME, versionCache) = version else {
            return XCTFail("Expected version marker asset")
        }
        XCTAssertEqual(versionMIME, "application/json; charset=utf-8")
        XCTAssertEqual(versionCache, "no-cache")
    }

    func testRootAndDirectIndexBothUseInjectedIndexPath() throws {
        let resolver = WebUIAssetResolver(rootURL: try makeFixture())

        XCTAssertEqual(resolver.resolve(path: "/"), .index)
        XCTAssertEqual(resolver.resolve(path: "/index.html"), .index)
        XCTAssertEqual(resolver.resolve(path: "/INDEX.HTML"), .index)
    }

    func testRejectsTraversalDirectoriesAndSymlinkEscapes() throws {
        let root = try makeFixture()
        let secret = workDirectory.appendingPathComponent("secret.txt")
        try "private".write(to: secret, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            atPath: root.appendingPathComponent("escape.js").path,
            withDestinationPath: secret.path
        )

        let resolver = WebUIAssetResolver(rootURL: root)
        XCTAssertNil(resolver.resolve(path: "/../secret.txt"))
        XCTAssertNil(resolver.resolve(path: "/..%2Fsecret.txt"))
        XCTAssertNil(resolver.resolve(path: "/escape.js"))
        XCTAssertNil(resolver.resolve(path: "/_app"))
        XCTAssertNil(resolver.resolve(path: "/missing.js"))
    }

    func testDecodesPercentEncodedAssetPath() throws {
        let root = try makeFixture()
        let resolver = WebUIAssetResolver(rootURL: root)

        let resolution = try XCTUnwrap(resolver.resolve(path: "/%5Fapp/immutable/bundle.abc123.js"))
        guard case let .asset(url, mimeType, cacheControl) = resolution else {
            return XCTFail("Expected percent-decoded asset")
        }
        XCTAssertEqual(url.lastPathComponent, "bundle.abc123.js")
        XCTAssertEqual(mimeType, "application/javascript; charset=utf-8")
        XCTAssertEqual(cacheControl, "public, max-age=31536000, immutable")
    }

    private func makeFixture() throws -> URL {
        let root = workDirectory.appendingPathComponent("webui", isDirectory: true)
        let immutable = root
            .appendingPathComponent("_app", isDirectory: true)
            .appendingPathComponent("immutable", isDirectory: true)
        try FileManager.default.createDirectory(at: immutable, withIntermediateDirectories: true)
        try "console.log('ok');".write(
            to: immutable.appendingPathComponent("bundle.abc123.js"),
            atomically: true,
            encoding: .utf8
        )
        try #"{"version":"fixture"}"#.write(
            to: root.appendingPathComponent("_app/version.json"),
            atomically: true,
            encoding: .utf8
        )
        try "<html><head></head><body></body></html>".write(
            to: root.appendingPathComponent("index.html"),
            atomically: true,
            encoding: .utf8
        )
        return root
    }
}
