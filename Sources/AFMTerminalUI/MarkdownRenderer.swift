import Foundation

public struct TUICodeBlock: Equatable, Sendable {
    public let language: String
    public let content: String

    public init(language: String, content: String) {
        self.language = language
        self.content = content
    }
}

public struct TUIImageReference: Equatable, Sendable {
    public let alt: String
    public let path: String

    public init(alt: String, path: String) {
        self.alt = alt
        self.path = path
    }
}

public struct MarkdownRenderResult: Equatable, Sendable {
    public let text: String
    public let codeBlocks: [TUICodeBlock]
    public let images: [TUIImageReference]
}

/// A deliberately small terminal renderer. It never evaluates Markdown/HTML or model output.
/// Its output is ANSI text only when color is enabled, which keeps redirected transcripts clean.
public struct TerminalMarkdownRenderer: Sendable {
    public enum Theme: String, CaseIterable, Sendable {
        case auto, dark, light, mono
    }

    private let color: Bool
    private let theme: Theme

    public init(color: Bool, theme: Theme = .auto) {
        self.color = color && theme != .mono
        self.theme = theme
    }

    public func render(_ markdown: String) -> MarkdownRenderResult {
        let normalized = markdown.replacingOccurrences(of: "\r\n", with: "\n")
        var rendered: [String] = []
        var blocks: [TUICodeBlock] = []
        var images: [TUIImageReference] = []
        var language: String?
        var code: [String] = []

        for rawLine in normalized.components(separatedBy: "\n") {
            if rawLine.hasPrefix("```") {
                if let activeLanguage = language {
                    let content = code.joined(separator: "\n")
                    let effectiveLanguage = activeLanguage.isEmpty ? Self.inferredLanguage(for: content) : activeLanguage
                    blocks.append(TUICodeBlock(language: effectiveLanguage, content: content))
                    rendered.append(renderCode(content, language: effectiveLanguage))
                    language = nil
                    code = []
                } else {
                    language = String(rawLine.dropFirst(3))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        .lowercased()
                }
                continue
            }
            if language != nil {
                code.append(rawLine)
                continue
            }

            images.append(contentsOf: Self.imageReferences(in: rawLine))
            rendered.append(renderLine(rawLine))
        }

        // Be forgiving of a model that forgot the closing fence.
        if let activeLanguage = language {
            let content = code.joined(separator: "\n")
            let effectiveLanguage = activeLanguage.isEmpty ? Self.inferredLanguage(for: content) : activeLanguage
            blocks.append(TUICodeBlock(language: effectiveLanguage, content: content))
            rendered.append(renderCode(content, language: effectiveLanguage))
        }
        return MarkdownRenderResult(text: rendered.joined(separator: "\n"), codeBlocks: blocks, images: images)
    }

    public static func imageReferences(in text: String) -> [TUIImageReference] {
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let markdownRegex = try? NSRegularExpression(pattern: #"!\[([^\]]*)\]\(([^)]+)\)"#) else { return [] }
        var results = markdownRegex.matches(in: text, range: nsRange).compactMap { match -> TUIImageReference? in
            guard let altRange = Range(match.range(at: 1), in: text),
                  let pathRange = Range(match.range(at: 2), in: text) else { return nil }
            let path = String(text[pathRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard path.hasPrefix("/") || path.hasPrefix("~") || path.hasPrefix("./") else { return nil }
            return TUIImageReference(alt: String(text[altRange]), path: path)
        }
        if let plainRegex = try? NSRegularExpression(
            pattern: #"(?:^|\s)((?:~|\.\.?/|/)[^\s<>\"']+\.(?:png|jpe?g|gif|webp|heic))\b"#,
            options: [.caseInsensitive]
        ) {
            for match in plainRegex.matches(in: text, range: nsRange) {
                guard let pathRange = Range(match.range(at: 1), in: text) else { continue }
                let path = String(text[pathRange])
                if !results.contains(where: { $0.path == path }) {
                    results.append(TUIImageReference(alt: URL(fileURLWithPath: path).lastPathComponent, path: path))
                }
            }
        }
        return results
    }

    public static func latexFallback(_ input: String) -> String {
        var output = input
        let replacements = [
            #"\alpha"#: "α", #"\beta"#: "β", #"\gamma"#: "γ", #"\delta"#: "δ",
            #"\theta"#: "θ", #"\lambda"#: "λ", #"\mu"#: "μ", #"\pi"#: "π",
            #"\sigma"#: "σ", #"\phi"#: "φ", #"\omega"#: "ω", #"\Delta"#: "Δ",
            #"\Sigma"#: "Σ", #"\Omega"#: "Ω", #"\infty"#: "∞", #"\sum"#: "∑",
            #"\prod"#: "∏", #"\int"#: "∫", #"\sqrt"#: "√", #"\neq"#: "≠",
            #"\leq"#: "≤", #"\geq"#: "≥", #"\rightarrow"#: "→", #"\leftarrow"#: "←",
            #"\times"#: "×", #"\cdot"#: "·", #"\pm"#: "±"
        ]
        for (source, target) in replacements { output = output.replacingOccurrences(of: source, with: target) }
        output = output.replacingOccurrences(of: #"\\frac\{([^{}]+)\}\{([^{}]+)\}"#, with: "($1)/($2)", options: .regularExpression)
        output = output.replacingOccurrences(of: #"\\text\{([^{}]+)\}"#, with: "$1", options: .regularExpression)
        output = output.replacingOccurrences(of: "$", with: "")
        return output
    }

    /// Labels common untyped fences so highlighting and artifact actions remain useful
    /// when a model omits the Markdown language hint.
    public static func inferredLanguage(for content: String) -> String {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("diff --git ") || trimmed.contains("\n@@ ") { return "diff" }
        if trimmed.hasPrefix("<!doctype html") || trimmed.hasPrefix("<html") { return "html" }
        if (trimmed.hasPrefix("{") || trimmed.hasPrefix("[")),
           (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) != nil { return "json" }
        if trimmed.hasPrefix("#!/") || trimmed.contains("\nset -e") { return "bash" }
        if trimmed.contains("func ") && (trimmed.contains("let ") || trimmed.contains("var ")) { return "swift" }
        if trimmed.contains("def ") && trimmed.contains(":") { return "python" }
        if trimmed.contains("const ") || trimmed.contains("function ") || trimmed.contains("=>") { return "javascript" }
        return ""
    }

    private func renderLine(_ raw: String) -> String {
        let line = Self.latexFallback(raw)
        if line.hasPrefix("### ") { return style(String(line.dropFirst(4)), "36") }
        if line.hasPrefix("## ") { return style(String(line.dropFirst(3)), "1;36") }
        if line.hasPrefix("# ") { return style(String(line.dropFirst(2)), "1;35") }
        if line.hasPrefix("> ") { return style("│ " + String(line.dropFirst(2)), "2;36") }
        if line.hasPrefix("@@") { return style(line, "36") }
        if line.hasPrefix("+") && !line.hasPrefix("+++") { return style(line, "32") }
        if line.hasPrefix("-") && !line.hasPrefix("---") { return style(line, "31") }
        return inlineStyle(line)
    }

    private func renderCode(_ content: String, language: String) -> String {
        let label = language.isEmpty ? "code" : language
        let header = style("┌─ \(label)", "2;36")
        let body = content.components(separatedBy: "\n").map { line in
            "│ " + highlight(line, language: language)
        }.joined(separator: "\n")
        return "\(header)\n\(body)\n\(style("└─", "2;36"))"
    }

    private func highlight(_ line: String, language: String) -> String {
        guard color else { return line }
        let normalized = language.lowercased()
        if ["diff", "patch"].contains(normalized) {
            if line.hasPrefix("+") { return style(line, "32") }
            if line.hasPrefix("-") { return style(line, "31") }
            if line.hasPrefix("@@") { return style(line, "36") }
        }
        if ["swift", "js", "javascript", "ts", "typescript", "python", "py", "rust", "go", "c", "cpp", "java", "kotlin", "bash", "sh", "zsh"].contains(normalized) {
            let keywordPattern = #"\b(func|let|var|struct|class|enum|actor|protocol|extension|import|return|if|else|for|while|switch|case|guard|try|await|async|throw|throws|public|private|internal|static|const|function|def|fn|use|package|interface|new|true|false|nil|null)\b"#
            return replace(line, pattern: keywordPattern, code: "35")
        }
        if ["json", "yaml", "yml", "toml"].contains(normalized) {
            return replace(line, pattern: #"("[^"\\]*(?:\\.[^"\\]*)*")\s*:"#, code: "36")
        }
        return line
    }

    private func inlineStyle(_ value: String) -> String {
        guard color else { return value }
        var result = replace(value, pattern: #"`([^`]+)`"#, code: "33")
        result = replace(result, pattern: #"\*\*([^*]+)\*\*"#, code: "1")
        return result
    }

    private func replace(_ input: String, pattern: String, code: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return input }
        let range = NSRange(input.startIndex..<input.endIndex, in: input)
        return regex.stringByReplacingMatches(in: input, range: range, withTemplate: "\u{001B}[\(code)m$1\u{001B}[0m")
    }

    private func style(_ value: String, _ code: String) -> String {
        color ? "\u{001B}[\(code)m\(value)\u{001B}[0m" : value
    }
}
