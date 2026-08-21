import Darwin
import XCTest
@testable import AFMKit
import AFMOpenAICompat
#if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
import FoundationModels
#endif
@testable import AFMKitFoundationModels
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

    func testRendersCommonMarkAndGFMStructures() {
        let source = #"""
        # Heading

        **bold**, *emphasis*, ~~removed~~, `inline`, and [AFM](https://example.com/afm).

        > Quoted **answer**

        - [x] parsed
        - [ ] rendered
          1. nested one
          2. nested two

        | Feature | State |
        |:--------|------:|
        | Tables  | ready |
        """#
        let text = TerminalMarkdownRenderer(color: false).render(source, width: 60).text

        XCTAssertTrue(text.contains("▌ Heading"))
        XCTAssertTrue(text.contains("bold, emphasis, ~~removed~~,  inline "))
        XCTAssertTrue(text.contains("AFM (https://example.com/afm)"))
        XCTAssertTrue(text.contains("│ Quoted answer"))
        XCTAssertTrue(text.contains("☑ parsed"))
        XCTAssertTrue(text.contains("☐ rendered"))
        XCTAssertTrue(text.contains("1. nested one"))
        XCTAssertTrue(text.contains("┌"))
        XCTAssertTrue(text.contains("Feature"))
        XCTAssertTrue(text.contains("ready"))
    }

    func testRendersInlineAndDisplayMathWithoutCorruptingCurrency() {
        let source = #"""
        It costs $5, while $x^2 + y_1 \leq \sqrt{9}$ is math.

        $$
        \begin{bmatrix} 1 & 2 \\ 3 & \frac{4}{5} \end{bmatrix}
        $$
        """#
        let text = TerminalMarkdownRenderer(color: false).render(source).text

        XCTAssertTrue(text.contains("costs $5"))
        XCTAssertTrue(text.contains("x² + y₁ ≤ √(9)"))
        XCTAssertTrue(text.contains("┌─ math"))
        XCTAssertTrue(text.contains("[ 1  2"))
        XCTAssertTrue(text.contains("(4)/(5)"))
    }

    func testLanguageHighlightingDiffsAndThemesProduceDistinctANSI() {
        let source = #"""
        ```swift
        let value = Widget(name: "AFM", count: 42) // comment
        ```
        ```diff
        @@ -1 +1 @@
        -old
        +new
        ```
        """#
        let dark = TerminalMarkdownRenderer(color: true, theme: .dark).render(source).text
        let light = TerminalMarkdownRenderer(color: true, theme: .light).render(source).text

        XCTAssertNotEqual(dark, light)
        XCTAssertTrue(dark.contains("\u{001B}[95mlet"))
        XCTAssertTrue(dark.contains("\u{001B}[94mWidget"))
        XCTAssertTrue(dark.contains("\u{001B}[92m\"AFM\""))
        XCTAssertTrue(dark.contains("\u{001B}[91m42"))
        XCTAssertTrue(dark.contains("\u{001B}[2;37m// comment"))
        XCTAssertTrue(dark.contains("\u{001B}[92m+new"))
        XCTAssertTrue(dark.contains("\u{001B}[91m-old"))
    }

    func testHyperlinksAreCapabilityGatedAndControlCharactersAreInert() {
        let source = "[safe](https://example.com)\u{001B}]2;owned\u{0007}"
        let plain = TerminalMarkdownRenderer(color: true).render(source, hyperlinks: false).text
        let linked = TerminalMarkdownRenderer(color: true).render(source, hyperlinks: true).text

        XCTAssertFalse(plain.contains("\u{001B}]8;;"))
        XCTAssertTrue(linked.contains("\u{001B}]8;;https://example.com"))
        XCTAssertFalse(plain.contains("\u{001B}]2;owned"))
        XCTAssertTrue(plain.contains("�]2;owned�"))
    }

    func testOnlyLocalImagesBecomeArtifactActions() {
        let source = "![local](./plot.png) ![remote](https://example.com/plot.png)"
        XCTAssertEqual(
            TerminalMarkdownRenderer(color: false).render(source).images,
            [TUIImageReference(alt: "local", path: "./plot.png")]
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

    func testOverwriteRestrictsExistingAndReplacementFilesToOwnerOnly() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let target = root.appendingPathComponent("output")
        try Data("old".utf8).write(to: target)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: target.path)

        try TUIArtifactActions.save(Data("new".utf8), to: target, overwrite: true)

        let permissions = try FileManager.default.attributesOfItem(atPath: target.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o600)
        XCTAssertEqual(try String(contentsOf: target, encoding: .utf8), "new")
    }

    func testAtomicOverwriteDoesNotFollowTargetSwappedToSymlinkAfterValidation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let target = root.appendingPathComponent("output")
        let victim = root.appendingPathComponent("victim")
        try Data("old".utf8).write(to: target)
        try Data("safe".utf8).write(to: victim)

        try TUIArtifactActions.save(
            Data("replacement".utf8),
            to: target,
            overwrite: true,
            beforeAtomicInstall: {
                try FileManager.default.removeItem(at: target)
                try FileManager.default.createSymbolicLink(at: target, withDestinationURL: victim)
            }
        )

        XCTAssertEqual(try String(contentsOf: victim, encoding: .utf8), "safe")
        XCTAssertEqual(try String(contentsOf: target, encoding: .utf8), "replacement")
        var info = stat()
        XCTAssertEqual(lstat(target.path, &info), 0)
        XCTAssertEqual(info.st_mode & S_IFMT, S_IFREG)
        XCTAssertEqual(info.st_mode & 0o777, 0o600)
    }

    func testOverwriteRejectsNonRegularTargetsWithoutBlocking() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let fifo = root.appendingPathComponent("pipe")
        XCTAssertEqual(mkfifo(fifo.path, S_IRUSR | S_IWUSR), 0)

        XCTAssertThrowsError(
            try TUIArtifactActions.save(Data("unsafe".utf8), to: fifo, overwrite: true)
        )
        var info = stat()
        XCTAssertEqual(lstat(fifo.path, &info), 0)
        XCTAssertEqual(info.st_mode & S_IFMT, S_IFIFO)
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
        XCTAssertTrue(html.contains("form-action 'none'"))
        XCTAssertTrue(html.contains("connect-src 'none'"))
        XCTAssertTrue(html.contains("querySelector"))
        XCTAssertTrue(html.contains("sandbox=\"allow-scripts\""))
        XCTAssertFalse(html.contains("<script>document.querySelector"))
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
        XCTAssertTrue(html.contains("form-action 'none'"))
        XCTAssertFalse(html.contains("<script>document.body"))
    }

    func testBrowserNavigationBoundaryRejectsOutboundAndSiblingNavigation() {
        let artifact = URL(fileURLWithPath: "/Volumes/edata2/afm-preview/artifact.html")
        let boundary = TUIBrowserNavigationBoundary(artifactURL: artifact)

        XCTAssertTrue(boundary.allows(artifact))
        XCTAssertTrue(boundary.allows(URL(string: "about:blank")))
        XCTAssertTrue(boundary.allows(URL(string: "about:srcdoc")))
        XCTAssertFalse(boundary.allows(URL(string: "https://example.com/exfiltrate")))
        XCTAssertFalse(boundary.allows(URL(string: "data:text/html,escaped")))
        XCTAssertFalse(boundary.allows(artifact.deletingLastPathComponent().appendingPathComponent("sibling.html")))
        XCTAssertFalse(boundary.allows(URL(string: "custom-scheme://escape")))
    }

    func testBrowserPreparationRejectsExecutableLanguages() {
        XCTAssertThrowsError(
            try TUIArtifactActions.prepareBrowserArtifact(TUICodeBlock(language: "bash", content: "rm something"))
        )
    }

    func testBoundedRegularFileReadRejectsLinksDirectoriesAndOversizedFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let regular = root.appendingPathComponent("regular.txt")
        let link = root.appendingPathComponent("linked.txt")
        try Data("1234".utf8).write(to: regular)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: regular)

        XCTAssertEqual(try TUIArtifactActions.readRegularFile(at: regular, maximumBytes: 4), Data("1234".utf8))
        XCTAssertThrowsError(try TUIArtifactActions.readRegularFile(at: regular, maximumBytes: 3))
        XCTAssertThrowsError(try TUIArtifactActions.readRegularFile(at: link, maximumBytes: 100))
        XCTAssertThrowsError(try TUIArtifactActions.readRegularFile(at: root, maximumBytes: 100))
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

    func testPersistsReasoningAndDecodesLegacySessions() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(directory: root)
        var session = TUISession(backend: "MLX", model: "test/model")
        session.messages = [.init(role: "assistant", content: "Answer")]
        session.reasoningByMessage["0"] = "Private reasoning"

        try store.save(session)
        XCTAssertEqual(try store.load(id: session.id).reasoning(atMessageIndex: 0), "Private reasoning")
        XCTAssertEqual(try store.search("private reasoning").first?.id, session.id)

        let export = root.appendingPathComponent("reasoning.md")
        try store.exportMarkdown(session, to: export)
        let markdown = try String(contentsOf: export, encoding: .utf8)
        XCTAssertTrue(markdown.contains("<summary>Reasoning</summary>"))
        XCTAssertTrue(markdown.contains("Private reasoning"))

        let data = try JSONSerialization.data(withJSONObject: [
            "id": session.id.uuidString,
            "title": "Legacy",
            "backend": "MLX",
            "model": "test/model",
            "createdAt": ISO8601DateFormatter().string(from: session.createdAt),
            "updatedAt": ISO8601DateFormatter().string(from: session.updatedAt),
            "messages": [["role": "assistant", "content": "Answer"]]
        ])
        try data.write(to: root.appendingPathComponent("\(session.id.uuidString).json"))
        XCTAssertTrue(try store.load(id: session.id).reasoningByMessage.isEmpty)
    }

    func testReasoningMetadataIsPrunedAndRemappedWhenMessagesAreRemoved() {
        var session = TUISession(backend: "MLX", model: "test/model", messages: [
            .init(role: "user", content: "one"),
            .init(role: "assistant", content: "first"),
            .init(role: "user", content: "two"),
            .init(role: "assistant", content: "second")
        ], reasoningByMessage: ["1": "r1", "3": "r2", "99": "stale"])

        session.removeMessage(at: 0)
        XCTAssertEqual(session.reasoningByMessage, ["0": "r1", "2": "r2"])
        session.removeLastExchange()
        XCTAssertEqual(session.messages.count, 1)
        XCTAssertEqual(session.reasoningByMessage, ["0": "r1"])
    }

    func testTwentyMegabyteImageSessionRoundTripsWithinSharedSaveLoadLimit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(directory: root)
        let imageData = Data(repeating: 0xA5, count: 20_000_000)
        let imageURL = "data:image/png;base64,\(imageData.base64EncodedString())"
        let session = TUISession(backend: "MLX", model: "test/model", messages: [
            Message(role: "user", content: .parts([
                ContentPart(type: "text", text: "inspect"),
                ContentPart(type: "image_url", image_url: ImageURL(url: imageURL, detail: "auto"))
            ]))
        ])

        let url = try store.save(session)
        let fileSize = try XCTUnwrap(
            (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue
        )
        XCTAssertGreaterThan(fileSize, 10_000_000)
        XCTAssertLessThanOrEqual(fileSize, TUISessionStore.defaultMaximumSessionBytes)
        XCTAssertEqual(try store.load(id: session.id).messages.count, 1)
    }

    func testOversizedReplacementIsRejectedWithoutDestroyingLoadableSession() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(directory: root, maximumSessionBytes: 1_024)
        var session = TUISession(title: "original", backend: "MLX", model: "test/model")
        try store.save(session)

        session.title = String(repeating: "x", count: 2_000)
        XCTAssertThrowsError(try store.save(session)) { error in
            XCTAssertEqual(
                error as? TUISessionStoreError,
                .sessionTooLarge(maximumBytes: 1_024)
            )
        }
        XCTAssertEqual(try store.load(id: session.id).title, "original")
    }

    func testOversizedPersistenceWritesFullFidelityJSONAndRetainsLastSavedSession() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(
            directory: root,
            maximumSessionBytes: 1_024,
            maximumRecoveryBytes: 16_000
        )
        var session = TUISession(title: "saved version", backend: "MLX", model: "test/model")
        try store.save(session)

        session.title = "unsaved version"
        session.messages = [
            Message(role: "user", content: .parts([
                ContentPart(type: "text", text: "latest turn " + String(repeating: "x", count: 1_100)),
                ContentPart(
                    type: "image_url",
                    image_url: ImageURL(url: "data:image/png;base64,AQIDBA==", detail: "high")
                ),
                ContentPart(
                    type: "input_audio",
                    input_audio: InputAudio(data: "BQYHCA==", format: "wav", language: "en-CA")
                )
            ])),
            Message(
                role: "assistant",
                content: nil,
                toolCalls: [MessageToolCall(
                    id: "call-1",
                    type: "function",
                    function: MessageToolCallFunction(
                        name: "lookup",
                        arguments: #"{"query":"swift"}"#
                    )
                )]
            ),
            Message(
                role: "tool",
                content: .text("result"),
                toolCallId: "call-1",
                name: "lookup"
            )
        ]
        let result = store.persistRecoveringSession(session)
        guard case .recovered(let saveError, let recoveryURL) = result else {
            return XCTFail("Expected an oversized save to produce a recovery session")
        }

        XCTAssertTrue(saveError.contains("save/load limit"))
        XCTAssertEqual(try store.load(id: session.id).title, "saved version")
        XCTAssertEqual(recoveryURL.pathExtension, "json")
        XCTAssertEqual(
            try canonicalSessionData(store.loadRecovery(id: session.id)),
            try canonicalSessionData(session)
        )
        XCTAssertEqual(
            try canonicalSessionData(store.loadBestAvailable(id: session.id)),
            try canonicalSessionData(session)
        )
        let permissions = try FileManager.default.attributesOfItem(atPath: recoveryURL.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o600)

        try Data("corrupt recovery".utf8).write(to: recoveryURL)
        XCTAssertEqual(try store.loadBestAvailable(id: session.id).title, "saved version")
    }

    func testRepeatedRecoveryAtomicallyReplacesThePreviousVersion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(
            directory: root,
            maximumSessionBytes: 128,
            maximumRecoveryBytes: 8_192
        )
        var session = TUISession(
            title: "first " + String(repeating: "a", count: 256),
            backend: "MLX",
            model: "test/model"
        )
        guard case .recovered = store.persistRecoveringSession(session) else {
            return XCTFail("Expected first recovery")
        }

        session.title = "second " + String(repeating: "b", count: 256)
        session.messages = [.init(role: "assistant", content: "newest complete state")]
        guard case .recovered = store.persistRecoveringSession(session) else {
            return XCTFail("Expected replacement recovery")
        }

        XCTAssertEqual(try store.loadRecovery(id: session.id).title, session.title)
        let recoveryFiles = try FileManager.default.contentsOfDirectory(
            at: root.appendingPathComponent("recovery"),
            includingPropertiesForKeys: nil
        )
        XCTAssertEqual(recoveryFiles.filter { $0.pathExtension == "json" }.count, 1)
        XCTAssertFalse(recoveryFiles.contains { $0.pathExtension == "tmp" })
    }

    func testFailedRecoveryReplacementPreservesThePriorGoodRecovery() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(
            directory: root,
            maximumSessionBytes: 64,
            maximumRecoveryBytes: 2_048
        )
        var session = TUISession(
            title: "recoverable " + String(repeating: "a", count: 256),
            backend: "MLX",
            model: "test/model"
        )
        guard case .recovered = store.persistRecoveringSession(session) else {
            return XCTFail("Expected initial recovery")
        }
        let prior = try canonicalSessionData(store.loadRecovery(id: session.id))

        session.title = "too large " + String(repeating: "z", count: 4_096)
        guard case .failed(_, let recoveryError) = store.persistRecoveringSession(session) else {
            return XCTFail("Expected bounded recovery failure")
        }

        XCTAssertTrue(recoveryError.contains("recovery limit"))
        XCTAssertEqual(try canonicalSessionData(store.loadRecovery(id: session.id)), prior)
    }

    func testInterruptedTemporaryRecoveryDoesNotReplaceLastGoodJSON() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(
            directory: root,
            maximumSessionBytes: 64,
            maximumRecoveryBytes: 4_096
        )
        let session = TUISession(
            title: "recoverable " + String(repeating: "a", count: 256),
            backend: "MLX",
            model: "test/model"
        )
        guard case .recovered(_, let recoveryURL) = store.persistRecoveringSession(session) else {
            return XCTFail("Expected initial recovery")
        }
        let interrupted = recoveryURL.deletingLastPathComponent()
            .appendingPathComponent(".interrupted.tmp")
        try Data(#"{"title":"partial""#.utf8).write(to: interrupted)

        XCTAssertEqual(
            try canonicalSessionData(store.loadRecovery(id: session.id)),
            try canonicalSessionData(session)
        )
    }

    func testRecentHistoryUsesBoundedMetadataDecoding() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let ids = (0..<4).map { _ in UUID() }
        for (index, id) in ids.enumerated() {
            let updatedAt = Date(timeIntervalSince1970: TimeInterval(index + 1))
            let malformedFullSession = try JSONSerialization.data(withJSONObject: [
                "id": id.uuidString,
                "title": "Metadata \(index)",
                "updatedAt": ISO8601DateFormatter().string(from: updatedAt),
                "messages": "intentionally not decodable as a session"
            ])
            let url = root.appendingPathComponent("\(id.uuidString).json")
            try malformedFullSession.write(to: url)
            try FileManager.default.setAttributes([.modificationDate: updatedAt], ofItemAtPath: url.path)
        }
        let store = TUISessionStore(directory: root)

        let recent = try store.recent(limit: 2)
        XCTAssertEqual(recent.map(\.id), [ids[3], ids[2]])
        XCTAssertEqual(recent.map(\.title), ["Metadata 3", "Metadata 2"])
        XCTAssertThrowsError(try store.load(id: ids[3]))
        XCTAssertTrue(try store.recent(limit: 0).isEmpty)
    }

    func testRecentHistoryBackfillsPastNewerCorruptAndOversizedFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = TUISessionStore(directory: root, maximumSessionBytes: 1_024)
        let validSessions = (1...3).map { index in
            TUISession(
                title: "Valid \(index)",
                backend: "MLX",
                model: "test/model",
                createdAt: Date(timeIntervalSince1970: TimeInterval(index)),
                updatedAt: Date(timeIntervalSince1970: TimeInterval(index))
            )
        }
        for session in validSessions { try store.save(session) }

        let corruptURL = root.appendingPathComponent("\(UUID().uuidString).json")
        try Data("not json".utf8).write(to: corruptURL)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 5)],
            ofItemAtPath: corruptURL.path
        )
        let oversizedURL = root.appendingPathComponent("\(UUID().uuidString).json")
        try Data(repeating: 0x41, count: 2_000).write(to: oversizedURL)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 4)],
            ofItemAtPath: oversizedURL.path
        )

        let recent = try store.recent(limit: 2)
        XCTAssertEqual(recent.map(\.id), [validSessions[2].id, validSessions[1].id])
        XCTAssertEqual(recent.map(\.title), ["Valid 3", "Valid 2"])
    }

    func testConcurrentStoresAtomicallyReplaceTheSameSession() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let stores = (0..<6).map { _ in TUISessionStore(directory: root) }
        let validTitles = Set((0..<6).map { "writer-\($0)" })

        try await withThrowingTaskGroup(of: Void.self) { group in
            for (index, store) in stores.enumerated() {
                group.addTask {
                    for iteration in 0..<20 {
                        let session = TUISession(
                            id: id,
                            title: "writer-\(index)",
                            backend: "MLX",
                            model: "test/model",
                            messages: [.init(role: "assistant", content: String(repeating: "\(iteration)", count: 200))]
                        )
                        try store.save(session)
                        let loaded = try store.load(id: id)
                        guard validTitles.contains(loaded.title), loaded.messages.count == 1 else {
                            throw CocoaError(.fileReadCorruptFile)
                        }
                    }
                }
            }
            try await group.waitForAll()
        }

        let loaded = try stores[0].load(id: id)
        XCTAssertTrue(validTitles.contains(loaded.title))
        XCTAssertEqual(loaded.messages.count, 1)
    }

    private func canonicalSessionData(_ session: TUISession) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(session)
    }
}

final class GenerationBufferTests: XCTestCase {
    func testCompletedBufferRejectsLateEventsAndSkipsUnchangedSnapshots() async {
        let buffer = GenerationBuffer()
        await buffer.accept(.text("one", tokenCount: 1))
        await buffer.finish()
        let completed = await buffer.snapshot()

        await buffer.accept(.text("late", tokenCount: 2))
        let unchanged = await buffer.snapshot(ifChangedSince: completed.revision)
        let final = await buffer.snapshot()
        XCTAssertNil(unchanged)
        XCTAssertEqual(final.text, "one")
    }

    func testStreamingSnapshotUsesBoundedSanitizedRenderTails() async {
        let buffer = GenerationBuffer()
        let unsafePrefix = "safe\u{001B}]2;owned\u{0007}"
        let longText = unsafePrefix + String(repeating: "x", count: GenerationBuffer.renderTailLimit + 500)
        await buffer.accept(.text(longText, tokenCount: 1))
        await buffer.accept(.reasoning("think\u{000D}again", tokenCount: 1))
        await buffer.accept(.toolCall(
            AFMToolCall(id: "id", name: "bad\u{001B}name", arguments: "{}\u{0007}"),
            stage: .argumentsDelta("delta\u{001B}]2;owned")
        ))
        let renderSnapshot = await buffer.renderSnapshot()
        let snapshot = await buffer.snapshot()

        XCTAssertTrue(renderSnapshot.text.isEmpty)
        XCTAssertTrue(renderSnapshot.reasoning.isEmpty)
        XCTAssertTrue(renderSnapshot.tools.isEmpty)
        XCTAssertEqual(snapshot.textTail.count, GenerationBuffer.renderTailLimit)
        XCTAssertEqual(snapshot.textCharacterCount, snapshot.text.count)
        XCTAssertFalse(snapshot.text.contains("\u{001B}"))
        XCTAssertFalse(snapshot.reasoning.contains("\u{000D}"))
        XCTAssertFalse(snapshot.toolDisplayLines.joined().contains("\u{0007}"))
        XCTAssertFalse(snapshot.toolDisplayLines.joined().contains("owned"))
    }
}

final class FoundationStopSequenceFilterTests: XCTestCase {
    func testWithholdsSplitStopPrefixesAndDropsTheStopAndSuffix() {
        var filter = FoundationStopSequenceFilter(stopSequences: ["<END>", "STOP"])
        var output = ""
        output += filter.consume("hello<")
        output += filter.consume("EN")
        output += filter.consume("D>ignored")
        output += filter.finish()

        XCTAssertEqual(output, "hello")
        XCTAssertTrue(filter.stopped)
    }

    func testFlushesAnUnmatchedPrefixAtEndOfStream() {
        var filter = FoundationStopSequenceFilter(stopSequences: ["<END>"])
        XCTAssertEqual(filter.consume("hello<EN"), "hello")
        XCTAssertEqual(filter.finish(), "<EN")
    }

    #if canImport(FoundationModels) && !DISABLE_FOUNDATION_MODELS
    func testAcceptedTranscriptReplacesHiddenPostStopOutput() throws {
        let original = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Be concise"))],
                toolDefinitions: []
            )),
            .prompt(.init(segments: [.text(.init(content: "Earlier question"))])),
            .response(.init(assetIDs: [], segments: [.text(.init(content: "Earlier answer"))]))
        ])

        let accepted = FoundationModelService.acceptedTranscript(
            from: original,
            prompt: "User: continue\n\nAssistant: ",
            response: "visible prefix",
            options: GenerationOptions()
        )

        XCTAssertEqual(accepted.count, original.count + 2)
        guard case .response(let response) = accepted[accepted.index(before: accepted.endIndex)] else {
            return XCTFail("Expected the replacement transcript to end with a response")
        }
        let text = response.segments.compactMap { segment -> String? in
            guard case .text(let value) = segment else { return nil }
            return value.content
        }.joined()
        XCTAssertEqual(text, "visible prefix")
        XCTAssertFalse(text.contains("hidden suffix"))
    }
    #endif
}

private actor FoundationOperationProbe {
    private var active = 0
    private var maximumActive = 0
    private var events: [String] = []

    func start(_ name: String) {
        active += 1
        maximumActive = max(maximumActive, active)
        events.append("start:\(name)")
    }

    func finish(_ name: String) {
        events.append("finish:\(name)")
        active -= 1
    }

    func snapshot() -> (events: [String], maximumActive: Int) {
        (events, maximumActive)
    }
}

private actor AFMEngineOrderingProbe {
    private var events: [String] = []
    private var streamTaskArrived = false
    private var streamTaskWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func holdStreamTask() async {
        streamTaskArrived = true
        events.append("stream-task-arrived")
        let waiters = streamTaskWaiters
        streamTaskWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }

    func waitForStreamTask() async {
        guard !streamTaskArrived else { return }
        await withCheckedContinuation { streamTaskWaiters.append($0) }
    }

    func releaseStreamTask() {
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func record(_ event: String) {
        events.append(event)
    }

    func snapshot() -> [String] { events }
}

final class AFMEngineFoundationOrderingTests: XCTestCase {
    func testStreamReservesOrderBeforeReturningAndResetCannotOvertakeIt() async throws {
        let probe = AFMEngineOrderingProbe()
        let engine = AFMEngine(foundationDriver: AFMEngineFoundationDriver(
            beforeStreamTask: { await probe.holdStreamTask() },
            resetConversation: { _ in await probe.record("reset") },
            respond: { _, _ in "unused" },
            stream: { _, _ in
                await probe.record("stream")
                return AsyncThrowingStream { continuation in
                    continuation.yield("answer")
                    continuation.finish()
                }
            }
        ))

        let stream = engine.streamEvents(to: [Message(role: "user", content: "question")])
        let resetTask = Task { try await engine.resetConversation() }

        await probe.waitForStreamTask()
        for _ in 0..<20 { await Task.yield() }
        let blockedEvents = await probe.snapshot()
        XCTAssertEqual(blockedEvents, ["stream-task-arrived"])

        await probe.releaseStreamTask()
        var receivedText = ""
        for try await event in stream {
            if case .text(let text, _) = event { receivedText += text }
        }
        try await resetTask.value

        XCTAssertEqual(receivedText, "answer")
        let completedEvents = await probe.snapshot()
        XCTAssertEqual(completedEvents, ["stream-task-arrived", "stream", "reset"])
    }
}

final class FoundationSessionOperationGateTests: XCTestCase {
    func testEarlierStreamReservationBlocksLaterResetAndExcludesConcurrentUse() async throws {
        let gate = FoundationSessionOperationGate()
        let streamReservation = gate.reserve()
        let resetReservation = gate.reserve()
        let probe = FoundationOperationProbe()

        let resetTask = Task {
            try await resetReservation.perform {
                await probe.start("reset")
                await probe.finish("reset")
            }
        }
        for _ in 0..<10 { await Task.yield() }
        let beforeStreamStarts = await probe.snapshot()
        XCTAssertTrue(beforeStreamStarts.events.isEmpty)

        let streamTask = Task {
            try await streamReservation.perform {
                await probe.start("stream")
                try await Task.sleep(for: .milliseconds(20))
                await probe.finish("stream")
            }
        }

        try await streamTask.value
        try await resetTask.value
        let snapshot = await probe.snapshot()
        XCTAssertEqual(snapshot.maximumActive, 1)
        XCTAssertEqual(snapshot.events, [
            "start:stream", "finish:stream", "start:reset", "finish:reset"
        ])
    }
}

final class TUIConversationPolicyTests: XCTestCase {
    func testFoundationTurnsAreIncrementalWhileStatelessTurnsRetainHistory() {
        let transcript = [
            Message(role: "user", content: "first"),
            Message(role: "assistant", content: "answer"),
            Message(role: "user", content: "second")
        ]

        XCTAssertEqual(
            AFMTerminalChat.requestMessages(for: .foundationModels, transcript: transcript).map(\.textContent),
            ["second"]
        )
        XCTAssertEqual(
            AFMTerminalChat.requestMessages(for: .mlx(modelID: "test"), transcript: transcript).map(\.textContent),
            ["first", "answer", "second"]
        )
    }

    func testResumeRejectsBackendOrModelMismatch() {
        let session = TUISession(backend: "MLX", model: "one", messages: [])
        XCTAssertNoThrow(try AFMTerminalChat.validateRestoredSession(session, backendName: "MLX", modelName: "one"))
        XCTAssertThrowsError(try AFMTerminalChat.validateRestoredSession(session, backendName: "Foundation", modelName: "one"))
        XCTAssertThrowsError(try AFMTerminalChat.validateRestoredSession(session, backendName: "MLX", modelName: "two"))
    }

    func testRetryPreservesExactMultipartTurnAndEditPreservesAttachments() throws {
        let original = Message(role: "user", content: .parts([
            ContentPart(type: "text", text: "original prompt"),
            ContentPart(
                type: "image_url",
                image_url: ImageURL(url: "data:image/png;base64,AQID", detail: "high")
            ),
            ContentPart(type: "text", text: "<attachment>reference text</attachment>"),
            ContentPart(
                type: "input_audio",
                input_audio: InputAudio(data: "BAUG", format: "wav", language: "en-US")
            )
        ]))
        let turn = try XCTUnwrap(TUIUserTurn(message: original))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        XCTAssertEqual(turn.input, "original prompt")
        XCTAssertEqual(try encoder.encode(turn.message), try encoder.encode(original))

        let edited = turn.replacingInput(with: "revised prompt")
        guard case .some(.parts(let originalParts)) = original.content,
              case .some(.parts(let editedParts)) = edited.message.content else {
            return XCTFail("Expected multipart user messages")
        }
        XCTAssertEqual(edited.input, "revised prompt")
        XCTAssertEqual(editedParts.first?.text, "revised prompt")
        XCTAssertEqual(
            try encoder.encode(Array(editedParts.dropFirst())),
            try encoder.encode(Array(originalParts.dropFirst()))
        )
    }

    func testVideoAttachmentRemainsAFileURLMediaPart() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let video = root.appendingPathComponent("clip.mp4")
        try Data("video fixture".utf8).write(to: video)

        let turn = try AFMTerminalChat.makeUserTurn("describe", attachments: [video])
        guard case .some(.parts(let parts)) = turn.message.content else {
            return XCTFail("Expected multipart video request")
        }

        XCTAssertEqual(parts.count, 2)
        XCTAssertEqual(parts[1].type, "image_url")
        XCTAssertEqual(parts[1].image_url?.url, video.absoluteString)
        XCTAssertNil(parts[1].image_url?.detail)
    }

    func testPersistenceFailureNoticeIncludesRecoveryAndManualExportPaths() throws {
        let recoveryURL = URL(fileURLWithPath: "/safe/recovery/session.json")
        let notice = try XCTUnwrap(AFMTerminalChat.persistenceNotice(for: .recovered(
            saveError: "Session exceeds limit",
            recoveryURL: recoveryURL
        )))

        XCTAssertTrue(notice.contains("Session save failed: Session exceeds limit"))
        XCTAssertTrue(notice.contains(recoveryURL.path))
        XCTAssertTrue(notice.contains("/export <path>"))
        XCTAssertNil(AFMTerminalChat.persistenceNotice(for: .saved(recoveryURL)))
    }

    func testMLXMaximumLogprobsEnablesLogprobCollection() {
        let enabled = TUILogprobConfiguration(maximum: 7)
        XCTAssertTrue(enabled.enabled)
        XCTAssertEqual(enabled.maximum, 7)

        let disabled = TUILogprobConfiguration(maximum: nil)
        XCTAssertFalse(disabled.enabled)
        XCTAssertNil(disabled.maximum)
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
            inputIsTTY: true,
            outputIsTTY: true
        )
        XCTAssertFalse(capabilities.color)
        XCTAssertFalse(capabilities.hyperlinks)
        XCTAssertEqual(capabilities.inlineImages, .none)
    }

    func testTUIArgumentConflictsAreRejected() {
        XCTAssertNoThrow(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false, inputIsTTY: true, outputIsTTY: true))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: true, singlePrompt: false, inputIsTTY: true, outputIsTTY: true))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: true, inputIsTTY: true, outputIsTTY: true))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false, telegramOptions: true, inputIsTTY: true, outputIsTTY: true))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false, inputIsTTY: false, outputIsTTY: true))
        XCTAssertThrowsError(try TUIInvocationPolicy.validate(tui: true, webUI: false, singlePrompt: false, inputIsTTY: true, outputIsTTY: false))
    }

    func testExplicitDefaultTelegramFormatStillCountsAsATUIConflict() {
        XCTAssertTrue(TUIInvocationPolicy.hasTelegramOptions(
            botToken: nil,
            allowlist: nil,
            replyFormat: "markdown",
            requirePrefix: nil
        ))
        XCTAssertTrue(TUIInvocationPolicy.hasTelegramOptions(
            botToken: nil,
            allowlist: nil,
            replyFormat: nil,
            requirePrefix: "/afm"
        ))
        XCTAssertFalse(TUIInvocationPolicy.hasTelegramOptions(
            botToken: nil,
            allowlist: nil,
            replyFormat: nil,
            requirePrefix: nil
        ))
    }

    func testMediaValidationAcceptsVideosAndRejectsMissingOrUnsupportedFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let image = root.appendingPathComponent("image.png")
        let video = root.appendingPathComponent("video.mp4")
        let unsupported = root.appendingPathComponent("notes.txt")
        try Data("image".utf8).write(to: image)
        try Data("video".utf8).write(to: video)
        try Data("text".utf8).write(to: unsupported)

        let resolved = try TUIMediaAttachmentPolicy.resolveAndValidate(
            ["image.png", "video.mp4"],
            cwd: root
        )
        XCTAssertEqual(resolved, [image, video])
        XCTAssertEqual(TUIMediaAttachmentPolicy.kind(for: image), .image)
        XCTAssertEqual(TUIMediaAttachmentPolicy.kind(for: video), .video)
        XCTAssertThrowsError(try TUIMediaAttachmentPolicy.resolveAndValidate(
            ["missing.png"],
            cwd: root
        ))
        XCTAssertThrowsError(try TUIMediaAttachmentPolicy.resolveAndValidate(
            ["notes.txt"],
            cwd: root
        ))
    }

    func testTerminalCapabilitiesRequirePTYInputAndOutput() throws {
        var master: Int32 = 0
        var slave: Int32 = 0
        XCTAssertEqual(openpty(&master, &slave, nil, nil, nil), 0)
        defer { close(master); close(slave) }
        var descriptors: [Int32] = [0, 0]
        XCTAssertEqual(pipe(&descriptors), 0)
        defer { close(descriptors[0]); close(descriptors[1]) }

        let environment = ["TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"]
        XCTAssertTrue(TerminalCapabilities.detect(
            environment: environment,
            inputFD: slave,
            outputFD: slave
        ).isInteractive)
        XCTAssertFalse(TerminalCapabilities.detect(
            environment: environment,
            inputFD: slave,
            outputFD: descriptors[1]
        ).isInteractive)
        XCTAssertFalse(TerminalCapabilities.detect(
            environment: environment,
            inputFD: descriptors[0],
            outputFD: slave
        ).isInteractive)
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
