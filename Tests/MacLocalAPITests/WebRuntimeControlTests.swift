import XCTest
@testable import AFMServer

final class WebRuntimeControlTests: XCTestCase {
    func testExplicitModelRoutingSkipsAppleFoundationStartup() {
        XCTAssertTrue(Server.usesAppleFoundationBackend(
            mlxModelID: nil,
            hasMLXModel: false,
            hasProviderModel: false
        ))
        XCTAssertFalse(Server.usesAppleFoundationBackend(
            mlxModelID: "mlx-community/model",
            hasMLXModel: true,
            hasProviderModel: false
        ))
        XCTAssertFalse(Server.usesAppleFoundationBackend(
            mlxModelID: "provider/model",
            hasMLXModel: false,
            hasProviderModel: true
        ))
        XCTAssertTrue(Server.usesAppleFoundationBackend(
            mlxModelID: "missing/model",
            hasMLXModel: false,
            hasProviderModel: false
        ))
    }

    func testWebControlAcceptsOnlyLoopbackAddressesAndOrigins() {
        XCTAssertTrue(Server.isLoopbackWebAddress("127.0.0.1"))
        XCTAssertTrue(Server.isLoopbackWebAddress("::1"))
        XCTAssertTrue(Server.isLoopbackWebAddress("::ffff:127.0.0.1"))
        XCTAssertFalse(Server.isLoopbackWebAddress("192.168.1.20"))
        XCTAssertFalse(Server.isLoopbackWebAddress(nil))

        XCTAssertTrue(Server.isTrustedWebOrigin(nil))
        XCTAssertTrue(Server.isTrustedWebOrigin("http://127.0.0.1:9999"))
        XCTAssertTrue(Server.isTrustedWebOrigin("http://localhost:9999"))
        XCTAssertFalse(Server.isTrustedWebOrigin("null"))
        XCTAssertFalse(Server.isTrustedWebOrigin("https://example.com"))

        XCTAssertTrue(Server.isTrustedWebHost("127.0.0.1:9999"))
        XCTAssertTrue(Server.isTrustedWebHost("localhost:9999"))
        XCTAssertTrue(Server.isTrustedWebHost("[::1]:9999"))
        XCTAssertFalse(Server.isTrustedWebHost("afm.example.com:9999"))
        XCTAssertFalse(Server.isTrustedWebHost(nil))
    }

    func testWebLaunchArgumentsAreAllowlistedAndForceLoopbackWebUI() throws {
        let request = AFMWebLaunchRequest(
            backend: "mlx",
            model: "mlx-community/example",
            values: [
                "--gguf-file": "model-q4.gguf",
                "--mtp-model": "mlx-community/example-mtp"
            ],
            flags: ["--mtp", "--no-think"],
            dryRun: false
        )

        let arguments = try Server.webLaunchArguments(for: request, port: 10_001)
        XCTAssertEqual(Array(arguments.prefix(3)), ["mlx", "--model", "mlx-community/example"])
        XCTAssertTrue(arguments.contains("--gguf-file"))
        XCTAssertTrue(arguments.contains("--mtp-model"))
        XCTAssertTrue(arguments.contains("--mtp"))
        XCTAssertEqual(Array(arguments.suffix(5)), ["--hostname", "127.0.0.1", "--port", "10001", "--webui"])
    }

    func testWebLaunchArgumentsRejectUnknownOptions() {
        let request = AFMWebLaunchRequest(
            backend: "mlx",
            model: "mlx-community/example",
            values: ["--single-prompt": "unsafe"],
            flags: nil,
            dryRun: false
        )

        XCTAssertThrowsError(try Server.webLaunchArguments(for: request, port: 10_001))
    }

    func testWebRuntimePortSearchHandlesPortRangeBoundary() {
        XCTAssertNil(Server.availableWebRuntimePort(startingAt: 65_536))
    }

    func testSecretsAreExcludedFromProfilesAndDisplayedCommands() throws {
        let request = AFMWebLaunchRequest(
            backend: "foundation",
            model: nil,
            values: [
                "--telegram-bot-token": "secret-token",
                "--instructions": "Be concise"
            ],
            flags: ["--verbose"],
            dryRun: true
        )

        let profile = request.persistableProfile()
        XCTAssertNil(profile.values?["--telegram-bot-token"])
        XCTAssertEqual(profile.values?["--instructions"], "Be concise")
        XCTAssertEqual(profile.dryRun, false)

        let displayed = Server.redactedWebLaunchCommand(
            executable: URL(fileURLWithPath: "/usr/local/bin/afm"),
            arguments: ["--telegram-bot-token", "secret-token", "--verbose"]
        )
        XCTAssertEqual(displayed, ["/usr/local/bin/afm", "--telegram-bot-token", "<redacted>", "--verbose"])
        XCTAssertFalse(displayed.contains("secret-token"))
    }
}
