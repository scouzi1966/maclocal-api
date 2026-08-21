import Foundation
@testable import AFMKitMLX
import AFMOpenAICompat
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

    func testRedirectTargetIsRevalidatedAndPrivateRedirectFails() async throws {
        let url = try XCTUnwrap(URL(string: "https://public.example/image"))
        await assertMediaError(.redirectNotAllowed) {
            try await AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: { host in
                    host == "public.example" ? ["93.184.216.34"] : ["127.0.0.1"]
                },
                transport: { _, _, _ in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 302,
                        mimeType: nil,
                        contentLength: 0,
                        data: Data(),
                        redirectLocation: "https://private.example/secret"
                    )
                }
            )
        }
    }

    func testRedirectLimitIsEnforced() async throws {
        let url = try XCTUnwrap(URL(string: "https://public.example/image"))
        await assertMediaError(.tooManyRedirects) {
            try await AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { current, _, _ in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 302,
                        mimeType: nil,
                        contentLength: 0,
                        data: Data(),
                        redirectLocation: current.absoluteString
                    )
                }
            )
        }
    }

    func testRemoteMIMETypeAndSizeAreEnforced() async throws {
        let url = try XCTUnwrap(URL(string: "https://public.example/image"))
        await assertMediaError(.unsupportedMIMEType("text/html")) {
            try await AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { _, _, _ in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 200,
                        mimeType: "text/html",
                        contentLength: 4,
                        data: Data("nope".utf8),
                        redirectLocation: nil
                    )
                }
            )
        }

        await assertMediaError(.responseTooLarge) {
            try await AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { _, _, limit in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 200,
                        mimeType: "image/png",
                        contentLength: Int64(limit + 1),
                        data: Data(),
                        redirectLocation: nil
                    )
                }
            )
        }

        await assertMediaError(.responseTooLarge) {
            try await AFMMLXMediaSecurityPolicy.loadRemote(
                url,
                resolver: publicResolver,
                transport: { _, _, limit in
                    AFMMLXRemoteMediaResponse(
                        statusCode: 200,
                        mimeType: "image/png",
                        contentLength: nil,
                        data: Data(repeating: 0, count: limit + 1),
                        redirectLocation: nil
                    )
                }
            )
        }
    }

    func testValidatedAddressIsPinnedForInitialAndRedirectedHosts() async throws {
        let initial = try XCTUnwrap(URL(string: "https://first.example/misleading.png"))
        let observed = ObservedPinnedAddresses()
        let payload = try await AFMMLXMediaSecurityPolicy.loadRemote(
            initial,
            resolver: { host in
                host == "first.example" ? ["93.184.216.34"] : ["142.250.72.14"]
            },
            transport: { url, validatedAddress, _ in
                observed.append(host: url.host ?? "", address: validatedAddress)
                if url.host == "first.example" {
                    return AFMMLXRemoteMediaResponse(
                        statusCode: 302,
                        mimeType: nil,
                        contentLength: 0,
                        data: Data(),
                        redirectLocation: "https://second.example/media"
                    )
                }
                return AFMMLXRemoteMediaResponse(
                    statusCode: 200,
                    mimeType: "video/mp4",
                    contentLength: 1,
                    data: Data([1]),
                    redirectLocation: nil
                )
            }
        )

        XCTAssertEqual(payload.kind, .video)
        XCTAssertEqual(observed.values.map(\.0), ["first.example", "second.example"])
        XCTAssertEqual(observed.values.map(\.1), ["93.184.216.34", "142.250.72.14"])
    }

    func testProductionPinnedTransportPlanAndCancellation() async throws {
        let started = expectation(description: "pinned driver started")
        let driver = CancellingPinnedDriver { started.fulfill() }
        let url = try XCTUnwrap(URL(string: "https://media.example/path?q=1"))
        let task = Task {
            try await AFMMLXPinnedHTTPSClient.fetch(
                url: url,
                validatedAddress: "93.184.216.34",
                maximumBytes: 32,
                driverFactory: { plan in
                    driver.plan = plan
                    return driver
                }
            )
        }
        await fulfillment(of: [started], timeout: 1)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        XCTAssertEqual(driver.plan?.validatedAddress, "93.184.216.34")
        XCTAssertEqual(driver.plan?.tlsServerName, "media.example")
        XCTAssertEqual(driver.plan?.hostHeader, "media.example")
        XCTAssertEqual(driver.plan?.requestTarget, "/path?q=1")
        XCTAssertTrue(driver.wasCancelled)
        XCTAssertThrowsError(
            try AFMMLXPinnedHTTPSClient.connectionPlan(
                url: url,
                validatedAddress: "localhost"
            )
        )
    }

    func testResolvedMIMEKindOverridesURLFilename() async throws {
        let messages = [mediaMessage(["https://public.example/not-a-video.png"])]
        let resolved = try await AFMMLXMediaSecurityPolicy.resolveRequest(
            in: messages,
            limits: testLimits(),
            resolver: publicResolver,
            transport: { _, _, _ in
                AFMMLXRemoteMediaResponse(
                    statusCode: 200,
                    mimeType: "video/mp4",
                    contentLength: 1,
                    data: Data([1]),
                    redirectLocation: nil
                )
            },
            inspector: { _, _ in
                AFMMLXMediaInspection(imagePixels: 0, videoDuration: 1, videoFrames: 1)
            }
        )

        XCTAssertEqual(resolved.mediaKinds, [.video])
        guard case .parts(let parts)? = resolved.messages.first?.content else {
            return XCTFail("Expected resolved multipart message")
        }
        XCTAssertTrue(parts[0].image_url?.url.hasPrefix("data:video/mp4;base64,") == true)
    }

    func testRequestWideMediaCountAndAggregateBudgets() async throws {
        let inspector: AFMMLXMediaSecurityPolicy.PayloadInspector = { payload, _ in
            switch payload.data.first {
            case 2:
                AFMMLXMediaInspection(imagePixels: 7, videoDuration: 0, videoFrames: 0)
            case 3:
                AFMMLXMediaInspection(imagePixels: 0, videoDuration: 7, videoFrames: 0)
            case 4:
                AFMMLXMediaInspection(imagePixels: 0, videoDuration: 0, videoFrames: 7)
            default:
                AFMMLXMediaInspection(imagePixels: 1, videoDuration: 0, videoFrames: 0)
            }
        }
        let transport: AFMMLXMediaSecurityPolicy.RemoteTransport = { url, _, _ in
            let marker = UInt8(url.lastPathComponent) ?? 1
            let mime = marker >= 3 ? "video/mp4" : "image/png"
            return AFMMLXRemoteMediaResponse(
                statusCode: 200,
                mimeType: mime,
                contentLength: marker == 1 ? 3 : 1,
                data: Data(repeating: marker, count: marker == 1 ? 3 : 1),
                redirectLocation: nil
            )
        }

        await assertResolvedError(.tooManyMediaItems) {
            try await self.resolve(
                ["1", "1", "1"],
                limits: self.testLimits(maximumItems: 2),
                transport: transport,
                inspector: inspector
            )
        }
        await assertResolvedError(.aggregateMediaTooLarge) {
            try await self.resolve(
                ["1", "1"],
                limits: self.testLimits(maximumAggregateBytes: 5),
                transport: transport,
                inspector: inspector
            )
        }
        await assertResolvedError(.imagePixelLimitExceeded) {
            try await self.resolve(
                ["2", "2"],
                limits: self.testLimits(maximumImagePixels: 10),
                transport: transport,
                inspector: inspector
            )
        }
        await assertResolvedError(.videoDurationLimitExceeded) {
            try await self.resolve(
                ["3", "3"],
                limits: self.testLimits(maximumVideoDuration: 10),
                transport: transport,
                inspector: inspector
            )
        }
        await assertResolvedError(.videoFrameLimitExceeded) {
            try await self.resolve(
                ["4", "4"],
                limits: self.testLimits(maximumVideoFrames: 10),
                transport: transport,
                inspector: inspector
            )
        }
    }

    func testInlineAudioConsumesPerItemAndAggregateByteBudgets() async throws {
        let audioPart = ContentPart(
            type: "input_audio",
            input_audio: InputAudio(
                data: Data(repeating: 7, count: 4).base64EncodedString(),
                format: "wav",
                language: nil
            )
        )
        let message = Message(role: "user", content: .parts([audioPart, audioPart]))
        let unusedTransport: AFMMLXMediaSecurityPolicy.RemoteTransport = { _, _, _ in
            XCTFail("Inline audio must not invoke remote transport")
            throw AFMMLXMediaInputError.downloadFailed
        }
        let unusedInspector: AFMMLXMediaSecurityPolicy.PayloadInspector = { _, _ in
            XCTFail("Inline audio must not invoke image/video inspection")
            throw AFMMLXMediaInputError.mediaInspectionFailed
        }

        await assertResolvedError(.aggregateMediaTooLarge) {
            try await AFMMLXMediaSecurityPolicy.resolveRequest(
                in: [message],
                limits: self.testLimits(maximumAggregateBytes: 7),
                resolver: self.publicResolver,
                transport: unusedTransport,
                inspector: unusedInspector
            )
        }

        let oversizedPart = ContentPart(
            type: "input_audio",
            input_audio: InputAudio(
                data: Data(repeating: 7, count: 33).base64EncodedString(),
                format: "wav",
                language: nil
            )
        )
        await assertResolvedError(.responseTooLarge) {
            try await AFMMLXMediaSecurityPolicy.resolveRequest(
                in: [Message(role: "user", content: .parts([oversizedPart]))],
                limits: self.testLimits(),
                resolver: self.publicResolver,
                transport: unusedTransport,
                inspector: unusedInspector
            )
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

    private func resolve(
        _ paths: [String],
        limits: AFMMLXMediaRequestLimits,
        transport: @escaping AFMMLXMediaSecurityPolicy.RemoteTransport,
        inspector: @escaping AFMMLXMediaSecurityPolicy.PayloadInspector
    ) async throws -> AFMMLXResolvedMediaRequest {
        try await AFMMLXMediaSecurityPolicy.resolveRequest(
            in: [mediaMessage(paths.map { "https://public.example/\($0)" })],
            limits: limits,
            resolver: publicResolver,
            transport: transport,
            inspector: inspector
        )
    }

    private func mediaMessage(_ urls: [String]) -> Message {
        Message(
            role: "user",
            content: .parts(urls.map {
                ContentPart(type: "image_url", image_url: ImageURL(url: $0, detail: nil))
            })
        )
    }

    private func testLimits(
        maximumItems: Int = 8,
        maximumAggregateBytes: Int = 64,
        maximumImagePixels: Int64 = 64,
        maximumVideoDuration: Double = 64,
        maximumVideoFrames: Int = 64
    ) -> AFMMLXMediaRequestLimits {
        AFMMLXMediaRequestLimits(
            maximumItems: maximumItems,
            maximumItemBytes: 32,
            maximumAggregateBytes: maximumAggregateBytes,
            maximumImagePixels: maximumImagePixels,
            maximumVideoDuration: maximumVideoDuration,
            maximumVideoFrames: maximumVideoFrames
        )
    }

    private func assertMediaError(
        _ expected: AFMMLXMediaInputError,
        operation: () async throws -> AFMMLXMediaPayload
    ) async {
        do {
            _ = try await operation()
            XCTFail("Expected \(expected)")
        } catch {
            XCTAssertEqual(error as? AFMMLXMediaInputError, expected)
        }
    }

    private func assertResolvedError(
        _ expected: AFMMLXMediaInputError,
        operation: () async throws -> AFMMLXResolvedMediaRequest
    ) async {
        do {
            _ = try await operation()
            XCTFail("Expected \(expected)")
        } catch {
            XCTAssertEqual(error as? AFMMLXMediaInputError, expected)
        }
    }
}

private final class CancellingPinnedDriver:
    AFMMLXPinnedHTTPSConnectionDriver,
    @unchecked Sendable
{
    private let lock = NSLock()
    private let started: () -> Void
    private var continuation: CheckedContinuation<AFMMLXRemoteMediaResponse, Error>?
    var plan: AFMMLXPinnedHTTPSConnectionPlan?
    private(set) var wasCancelled = false

    init(started: @escaping () -> Void) {
        self.started = started
    }

    func run(request: Data, maximumBytes: Int) async throws -> AFMMLXRemoteMediaResponse {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock { self.continuation = continuation }
            started()
        }
    }

    func cancel() {
        let continuation = lock.withLock { () -> CheckedContinuation<
            AFMMLXRemoteMediaResponse,
            Error
        >? in
            wasCancelled = true
            let continuation = self.continuation
            self.continuation = nil
            return continuation
        }
        continuation?.resume(throwing: CancellationError())
    }
}

private final class ObservedPinnedAddresses: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [(String, String)] = []

    var values: [(String, String)] { lock.withLock { storage } }

    func append(host: String, address: String) {
        lock.withLock { storage.append((host, address)) }
    }
}
