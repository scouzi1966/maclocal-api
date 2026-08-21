import Darwin
import XCTest
@testable import AFMTerminalUI

final class TerminalMarkdownRendererTests: XCTestCase {
    func testParsesCodeDiffLatexAndLocalImagesWithoutColor() {
        let source = #"""
        # Result
        $\frac{\alpha}{\beta} \leq \infty$
        ```swift
        let answer = 42
        ```
        ```diff
        -old
        +new
        ```
        ![plot](/tmp/chart.png)
        """#
        let result = TerminalMarkdownRenderer(color: false).render(source)

        XCTAssertTrue(result.text.contains("(α)/(β) ≤ ∞"))
        XCTAssertTrue(result.text.contains("┌─ swift"))
        XCTAssertEqual(result.codeBlocks, [
            TUICodeBlock(language: "swift", content: "let answer = 42"),
            TUICodeBlock(language: "diff", content: "-old\n+new")
        ])
        XCTAssertEqual(result.images, [TUIImageReference(alt: "plot", path: "/tmp/chart.png")])
        XCTAssertFalse(result.text.contains("\u{001B}"))
    }

    func testTreatsUnclosedFenceAsCodeInsteadOfDroppingContent() {
        let result = TerminalMarkdownRenderer(color: false).render("```python\nprint('safe')")
        XCTAssertEqual(result.codeBlocks.first?.content, "print('safe')")
        XCTAssertTrue(result.text.contains("print('safe')"))
    }

    func testInfersLanguageForUntypedFences() {
        let result = TerminalMarkdownRenderer(color: false).render("```\nconst answer = () => 42\n```")
        XCTAssertEqual(result.codeBlocks.first?.language, "javascript")
    }

    func testRecognizesPlainLocalImagePath() {
        XCTAssertEqual(
            TerminalMarkdownRenderer.imageReferences(in: "Created /tmp/afm-output/chart.webp"),
            [TUIImageReference(alt: "chart.webp", path: "/tmp/afm-output/chart.webp")]
        )
    }
}

final class TUIArtifactActionsTests: XCTestCase {
    func testExclusiveSaveRefusesOverwriteAndExplicitOverwriteSucceeds() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("nested/code.swift")

        try TUIArtifactActions.save(Data("one".utf8), to: target)
        XCTAssertThrowsError(try TUIArtifactActions.save(Data("two".utf8), to: target)) { error in
            XCTAssertEqual(error as? TUIArtifactError, .exists(target.path))
        }
        try TUIArtifactActions.save(Data("two".utf8), to: target, overwrite: true)
        XCTAssertEqual(try String(contentsOf: target, encoding: .utf8), "two")
    }

    func testExplicitOverwriteStillRefusesSymlinkTargets() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let victim = root.appendingPathComponent("victim")
        let link = root.appendingPathComponent("output")
        try Data("safe".utf8).write(to: victim)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: victim)

        XCTAssertThrowsError(try TUIArtifactActions.save(Data("changed".utf8), to: link, overwrite: true))
        XCTAssertEqual(try String(contentsOf: victim, encoding: .utf8), "safe")
    }

    func testJavaScriptBrowserArtifactHasLocalCSPAndDoesNotRunDuringPreparation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = try TUIArtifactActions.prepareBrowserArtifact(
            TUICodeBlock(language: "js", content: "document.querySelector('#app').textContent='ok'"),
            temporaryRoot: root
        )
        let html = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(html.contains("Content-Security-Policy"))
        XCTAssertTrue(html.contains("default-src 'none'"))
        XCTAssertTrue(html.contains("querySelector"))
    }

    func testHTMLBrowserArtifactIsSandboxed() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = try TUIArtifactActions.prepareBrowserArtifact(
            TUICodeBlock(language: "html", content: "<script>document.body.textContent='ok'</script>"),
            temporaryRoot: root
        )
        let html = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(html.contains("sandbox=\"allow-scripts\""))
        XCTAssertTrue(html.contains("default-src 'none'"))
        XCTAssertFalse(html.contains("<script>document.body"))
    }

    func testBrowserPreparationRejectsExecutableLanguages() {
        XCTAssertThrowsError(
            try TUIArtifactActions.prepareBrowserArtifact(TUICodeBlock(language: "bash", content: "rm something"))
        )
    }
}

final class TUISessionStoreTests: XCTestCase {
    func testPersistsSearchesAndExportsSession() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(directory: root)
        var session = TUISession(title: "Swift help", backend: "MLX", model: "test/model")
        session.messages = [
            .init(role: "user", content: "Explain actors"),
            .init(role: "assistant", content: "Actors isolate mutable state.")
        ]

        let saved = try store.save(session)
        XCTAssertEqual(try store.load(id: session.id).messages.count, 2)
        XCTAssertEqual(try store.search("mutable").first?.id, session.id)
        let permissions = try FileManager.default.attributesOfItem(atPath: saved.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o600)

        let export = root.appendingPathComponent("transcript.md")
        try store.exportMarkdown(session, to: export)
        XCTAssertTrue(try String(contentsOf: export, encoding: .utf8).contains("## Assistant"))
    }
}

final class TerminalLifecycleAndInvocationTests: XCTestCase {
    func testTerminalRestorationIsIdempotent() throws {
        var enters = 0
        var restores = 0
        let controller = TerminalModeController(
            enter: { enters += 1 },
            restore: { restores += 1 }
        )
        try controller.enter()
        try controller.enter()
        controller.restore()
        controller.restore()
        XCTAssertEqual(enters, 1)
        XCTAssertEqual(restores, 1)
    }

    func testNoColorAndDumbTerminalFallback() {
        let capabilities = TerminalCapabilities.detect(
            environment: ["TERM": "dumb", "NO_COLOR": "1", "TERM_PROGRAM": "Apple_Terminal"],
            isTTY: true
        )
        XCTAssertFalse(capabilities.color)
        XCTAssertFalse(capabilities.hyperlinks)
        XCTAssertEqual(capabilities.inlineImages, .none)
    }

    func testTUIArgumentConflictsAreRejected() {
        XCTAssertNoThrow(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false, pipedInput: false))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: true, singlePrompt: false, pipedInput: false))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: true, pipedInput: false))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false, pipedInput: true))
    }


    func testTerminalKeyParsingHandlesMacReturnMultilineAndUnicode() throws {
        var descriptors: [Int32] = [0, 0]
        XCTAssertEqual(pipe(&descriptors), 0)
        defer { close(descriptors[0]); close(descriptors[1]) }
        let terminal = TerminalIO(inputFD: descriptors[0], outputFD: descriptors[1])
        let bytes = Array("é".utf8) + [13, 10, 27, 91, 68]
        _ = bytes.withUnsafeBytes { write(descriptors[1], $0.baseAddress, bytes.count) }
        XCTAssertEqual(terminal.readKey(), .text("é"))
        XCTAssertEqual(terminal.readKey(), .enter)
        XCTAssertEqual(terminal.readKey(), .newline)
        XCTAssertEqual(terminal.readKey(), .left)
    }
}
