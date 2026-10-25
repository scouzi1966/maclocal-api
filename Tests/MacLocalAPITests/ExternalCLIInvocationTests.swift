import AFMExternalCLI
import Foundation
import XCTest

final class ExternalCLIInvocationTests: XCTestCase {
    private var directory: URL!
    private let cli = ExternalCLI(name: "splash", executableOverride: "AFM_SPLASH_EXECUTABLE", installationHint: "Install Splash")

    override func setUpWithError() throws {
        // Deliberate test fixtures live in the checkout, not a RAM disk or /tmp.
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        directory = root.appendingPathComponent(".build-external-cli-tests/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let directory { try FileManager.default.removeItem(at: directory) }
    }

    private func fixture(_ name: String, executable: Bool = true) throws -> String {
        let url = directory.appendingPathComponent(name)
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: executable ? 0o755 : 0o644], ofItemAtPath: url.path)
        return url.path
    }

    func testForwardsAllArgumentsWithoutShellInterpretation() throws {
        let path = try fixture("splash")
        let arguments = ["serve", "--model", "owner/model with spaces", "--unknown-native-flag", "--", "$(echo nope)", "", "--help"]
        let invocation = try cli.invocation(arguments: arguments, environment: ["PATH": directory.path])
        XCTAssertEqual(invocation.executable, path)
        XCTAssertEqual(invocation.arguments, arguments)
    }

    func testExplicitExecutableOverridesPATHAndSupportsSpaces() throws {
        let path = try fixture("custom splash")
        _ = try fixture("splash")
        let invocation = try cli.invocation(arguments: ["--version"], environment: [
            "PATH": directory.path, "AFM_SPLASH_EXECUTABLE": path])
        XCTAssertEqual(invocation.executable, path)
    }

    func testInvalidOverrideDoesNotSilentlyUsePATH() throws {
        _ = try fixture("splash")
        XCTAssertThrowsError(try cli.invocation(arguments: [], environment: [
            "PATH": directory.path, "AFM_SPLASH_EXECUTABLE": directory.appendingPathComponent("missing").path]))
    }

    func testSkipsNonExecutablePATHCandidate() throws {
        _ = try fixture("splash", executable: false)
        XCTAssertThrowsError(try cli.invocation(arguments: [], environment: ["PATH": directory.path])) { error in
            XCTAssertTrue(error.localizedDescription.contains("Install Splash"))
        }
    }

    func testRejectsDirectoryEvenWhenSearchable() {
        XCTAssertThrowsError(try cli.invocation(arguments: [], environment: ["AFM_SPLASH_EXECUTABLE": directory.path]))
    }

    func testRejectsSelfSymlinkAndHardLinkToPreventRecursion() throws {
        let own = try fixture("afm")
        let symlink = directory.appendingPathComponent("splash-link").path
        let hardlink = directory.appendingPathComponent("splash-hardlink").path
        try FileManager.default.createSymbolicLink(atPath: symlink, withDestinationPath: own)
        try FileManager.default.linkItem(atPath: own, toPath: hardlink)
        for path in [own, symlink, hardlink] {
            XCTAssertThrowsError(try cli.invocation(arguments: [], environment: ["AFM_SPLASH_EXECUTABLE": path], currentExecutable: own)) { error in
                XCTAssertTrue(error.localizedDescription.contains("AFM itself"))
            }
        }
    }

    func testRelativeOverrideUsesCallerWorkingDirectory() throws {
        let path = try fixture("splash")
        let invocation = try cli.invocation(arguments: [], environment: ["AFM_SPLASH_EXECUTABLE": "./splash"], workingDirectory: directory.path)
        XCTAssertEqual(invocation.executable, path)
    }

    func testRejectsNULArgumentsInsteadOfTruncatingThem() {
        XCTAssertThrowsError(try cli.invocation(arguments: ["bad\0argument"], environment: [:])) { error in
            XCTAssertTrue(error.localizedDescription.contains("NUL"))
        }
    }
}
