import Foundation
@testable import AFMKitMLX
import XCTest

final class AFMMLXMediaSecurityPolicyTests: XCTestCase {
    private let publicResolver: AFMMLXMediaSecurityPolicy.HostResolver = { _ in
        ["93.184.216.34"]
    }

    func testRejectsUnsafeAndUnencryptedSchemes() {
        for raw in [
            "file:///Users/example/private.png",
            "http://example.com/image.png",
            "ftp://example.com/image.png",
        ] {
            XCTAssertThrowsError(
                try AFMMLXMediaSecurityPolicy.validateRemoteURL(
                    XCTUnwrap(URL(string: raw)),
                    resolver: publicResolver
                ),
                raw
            ) { error in
                guard case .unsupportedScheme = error as? AFMMLXMediaInputError else {
                    return XCTFail("unexpected error: \(error)")
                }
            }
        }
    }

    func testRejectsLocalNamesCredentialsAndNonDefaultPorts() throws {
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.validateRemoteURL(
                XCTUnwrap(URL(string: "https://localhost/image.png")),
                resolver: publicResolver
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .remoteHostNotAllowed)
        }
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.validateRemoteURL(
                XCTUnwrap(URL(string: "https://user:pass@example.com/image.png")),
                resolver: publicResolver
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .remoteHostNotAllowed)
        }
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.validateRemoteURL(
                XCTUnwrap(URL(string: "https://example.com:8443/image.png")),
                resolver: publicResolver
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .remotePortNotAllowed)
        }
    }

    func testRejectsPrivateReservedAndMixedDNSAnswers() throws {
        let blockedAddresses = [
            "127.0.0.1", "10.0.0.1", "100.64.0.1", "169.254.169.254",
            "172.16.0.1", "192.168.0.1", "198.18.0.1", "::1", "fd00::1",
            "fe80::1", "2001:db8::1", "::ffff:127.0.0.1",
        ]
        let url = try XCTUnwrap(URL(string: "https://example.com/image.png"))
        for address in blockedAddresses {
            XCTAssertThrowsError(
                try AFMMLXMediaSecurityPolicy.validateRemoteURL(
                    url,
                    resolver: { _ in [address] }
                ),
                address
            ) { error in
                XCTAssertEqual(error as? AFMMLXMediaInputError, .remoteAddressNotAllowed)
            }
        }
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.validateRemoteURL(
                url,
                resolver: { _ in ["93.184.216.34", "10.0.0.1"] }
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .remoteAddressNotAllowed)
        }
    }

    func testRedirectTargetIsRevalidatedAndPrivateRedirectFails() throws {
        let url = try XCTUnwrap(URL(string: "https://public.example/image"))
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: { host in
                    host == "public.example" ? ["93.184.216.34"] : ["127.0.0.1"]
                },
                transport: { _, _ in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 302,
                        mimeType: nil,
                        contentLength: 0,
                        data: Data(),
                        redirectLocation: "https://private.example/secret"
                    )
                }
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .redirectNotAllowed)
        }
    }

    func testRedirectLimitIsEnforced() throws {
        let url = try XCTUnwrap(URL(string: "https://public.example/image"))
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { current, _ in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 302,
                        mimeType: nil,
                        contentLength: 0,
                        data: Data(),
                        redirectLocation: current.absoluteString
                    )
                }
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .tooManyRedirects)
        }
    }

    func testRemoteMIMETypeAndSizeAreEnforced() throws {
        let url = try XCTUnwrap(URL(string: "https://public.example/image"))
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { _, _ in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 200,
                        mimeType: "text/html",
                        contentLength: 4,
                        data: Data("nope".utf8),
                        redirectLocation: nil
                    )
                }
            )
        ) { error in
            XCTAssertEqual(
                error as? AFMMLXMediaInputError,
                .unsupportedMIMEType("text/html")
            )
        }

        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { _, limit in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 200,
                        mimeType: "image/png",
                        contentLength: Int64(limit + 1),
                        data: Data(),
                        redirectLocation: nil
                    )
                }
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .responseTooLarge)
        }

        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { _, limit in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 200,
                        mimeType: "image/png",
                        contentLength: nil,
                        data: Data(repeating: 0, count: limit + 1),
                        redirectLocation: nil
                    )
                }
            )
        ) { error in
            XCTAssertEqual(error as? AFMMLXMediaInputError, .responseTooLarge)
        }
    }

    func testDataURLRequiresAllowedMIMEAndStrictBoundedBase64() throws {
        let payload = try AFMMLXMediaSecurityPolicy.decodeDataURL(
            "data:image/png;base64,iVBORw0KGgo="
        )
        XCTAssertEqual(payload.kind, .image)
        XCTAssertEqual(payload.mimeType, "image/png")

        for raw in [
            "data:text/plain;base64,aGVsbG8=",
            "data:image/png,not-base64",
            "data:image/png;base64,aGVs bG8=",
        ] {
            XCTAssertThrowsError(try AFMMLXMediaSecurityPolicy.decodeDataURL(raw), raw)
        }
    }

    func testTrustedLocalFileUsesExplicitBoundedDataURLPath() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "afm-trusted-media-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = directory.appendingPathComponent("upload.png")
        try Data([0x89, 0x50, 0x4e, 0x47]).write(to: image)

        let dataURL = try AFMMLXMediaSecurityPolicy.trustedLocalMediaDataURL(image)

        XCTAssertTrue(dataURL.hasPrefix("data:image/png;base64,"))
        XCTAssertEqual(try AFMMLXMediaSecurityPolicy.decodeDataURL(dataURL).data.count, 4)
        XCTAssertThrowsError(
            try AFMMLXMediaSecurityPolicy.trustedLocalMediaDataURL(
                directory.appendingPathComponent("upload.txt")
            )
        )
    }
}
