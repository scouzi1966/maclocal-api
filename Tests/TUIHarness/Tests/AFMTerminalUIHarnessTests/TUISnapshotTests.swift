import Foundation
import XCTest
import AFMTerminalUI

final class TUISnapshotTests: XCTestCase {
    func testFeatureRichAssistantTurnAtMacTerminalWidth() throws {
        let markdown = #"""
        # Calculus workspace

        For $f: D \subseteq \mathbb{R} \to \mathbb{R}$,

        $$
        f'(a) = \lim_{h \to 0} \frac{f(a+h)-f(a)}{h}
        $$

        > The same response can contain prose, math, source, and a patch.

        ```swift
        struct Gradient {
            let values: [Double]
            func magnitude() -> Double { values.reduce(0) { $0 + $1 * $1 }.squareRoot() }
        }
        ```

        ```diff
        -let method = "finite differences"
        +let method = "automatic differentiation"
        ```

        | Surface | Shortcut |
        |---|---|
        | Transcript | Ctrl-T |
        | Blocks | `/blocks` |
        """#

        let rendered = TerminalMarkdownRenderer(color: true, theme: .dark)
            .render(markdown, width: 88, hyperlinks: false)

        try assertSnapshot(
            named: "feature-rich-turn",
            value: visibleEscapes(rendered.text)
        )
        XCTAssertEqual(rendered.codeBlocks.map(\.language), ["swift", "diff"])
    }

    func testCalculusRegressionPromptHasNoRawTeXLeakage() throws {
        let response = #"""
        ## Limits and continuity
        A limit $\lim_{x \to c} f(x)=L$ means
        $$\forall \epsilon > 0, \exists \delta > 0 \text{ such that } |x-c|<\delta \implies |f(x)-L|<\epsilon.$$

        ## Single-variable calculus
        $$f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}, \qquad
        \int_a^b f(x)\,dx=\lim_{n\to\infty}\sum_{i=1}^n f(t_i)\Delta x_i.$$

        ## Multivariable calculus
        For $f:\mathbb{R}^n\to\mathbb{R}$, the gradient is
        $$\nabla f(\mathbf a)=\left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right).$$
        """#

        let text = TerminalMarkdownRenderer(color: false).render(response, width: 88).text
        try assertSnapshot(named: "calculus-regression", value: text)

        for token in ["$$", "\\mathbb", "\\lim", "\\frac", "\\partial", "\\mathbf", "\\text"] {
            XCTAssertFalse(text.contains(token), "raw TeX leaked into terminal output: \(token)")
        }
    }

    func testPopularFenceFormatsProduceStableHighlightedOutput() throws {
        let fixtures: [(String, String)] = [
            ("swift", "let answer: Int = 42"),
            ("python", "def greet(name):\n    return f\"Hello {name}\""),
            ("javascript", "const greet = (name) => `Hello ${name}`;"),
            ("typescript", "interface User { id: number }"),
            ("rust", "fn main() { println!(\"hello\"); }"),
            ("go", "package main\nfunc main() { println(\"hello\") }"),
            ("cpp", "std::vector<int> values{1, 2, 3};"),
            ("java", "record User(long id, String name) {}"),
            ("sql", "SELECT id FROM users WHERE active = true;"),
            ("json", "{\"name\": \"afm\", \"ready\": true}"),
            ("yaml", "name: afm\nready: true"),
            ("html", "<main class=\"app\">Hello</main>"),
            ("css", ".app { color: rebeccapurple; }"),
            ("bash", "if true; then echo \"ready\"; fi")
        ]
        let markdown = fixtures.map { "```\($0.0)\n\($0.1)\n```" }.joined(separator: "\n\n")
        let rendered = TerminalMarkdownRenderer(color: true, theme: .dark).render(markdown, width: 100)

        try assertSnapshot(named: "popular-formats", value: visibleEscapes(rendered.text))
        XCTAssertEqual(rendered.codeBlocks.map(\.language), fixtures.map(\.0))
        XCTAssertEqual(rendered.codeBlocks.map(\.content), fixtures.map(\.1))
    }

    private func visibleEscapes(_ value: String) -> String {
        value.replacingOccurrences(of: "\u{001B}", with: "<ESC>")
    }

    private func assertSnapshot(named name: String, value: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let normalizedValue = value
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map {
                String($0).replacingOccurrences(
                    of: #"\s+$"#,
                    with: "",
                    options: .regularExpression
                )
            }
            .joined(separator: "\n")
        let snapshotURL = Bundle.module.url(forResource: name, withExtension: "snap", subdirectory: "Snapshots")
        let record = ProcessInfo.processInfo.environment["AFM_RECORD_TUI_SNAPSHOTS"] == "1"
        if record {
            let sourceDirectory = URL(fileURLWithPath: String(describing: file))
                .deletingLastPathComponent()
                .appendingPathComponent("Snapshots", isDirectory: true)
            try FileManager.default.createDirectory(at: sourceDirectory, withIntermediateDirectories: true)
            try (normalizedValue + "\n").write(to: sourceDirectory.appendingPathComponent("\(name).snap"), atomically: true, encoding: .utf8)
            return
        }
        guard let snapshotURL else {
            XCTFail("Missing snapshot \(name). Run Scripts/test-tui.sh --record.", file: file, line: line)
            return
        }
        let expected = try String(contentsOf: snapshotURL, encoding: .utf8)
        XCTAssertEqual(normalizedValue + "\n", expected, file: file, line: line)
    }
}
