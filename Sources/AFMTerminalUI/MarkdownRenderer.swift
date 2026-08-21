import Foundation
import Markdown

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

/// CommonMark/GFM renderer for interactive terminal output.
///
/// Parsing is delegated to swift-markdown's cmark-gfm syntax tree. Rendering stays
/// local and inert: HTML, links, code, math, and model-provided control characters
/// are displayed but never evaluated. Browser previews remain an explicit `/open` action.
public struct TerminalMarkdownRenderer: Sendable {
    public enum Theme: String, CaseIterable, Sendable {
        case auto, dark, light, mono
    }

    private struct Palette: Sendable {
        let heading1: String
        let heading2: String
        let heading3: String
        let accent: String
        let secondary: String
        let code: String
        let keyword: String
        let type: String
        let string: String
        let number: String
        let comment: String
        let addition: String
        let deletion: String
        let warning: String

        static func resolve(_ theme: Theme) -> Palette {
            switch theme {
            case .light:
                return Palette(
                    heading1: "1;35", heading2: "1;34", heading3: "1;36",
                    accent: "34", secondary: "2;30", code: "33", keyword: "35",
                    type: "34", string: "32", number: "31", comment: "2;30",
                    addition: "32", deletion: "31", warning: "33"
                )
            case .auto, .dark:
                return Palette(
                    heading1: "1;95", heading2: "1;94", heading3: "1;96",
                    accent: "96", secondary: "2;37", code: "93", keyword: "95",
                    type: "94", string: "92", number: "91", comment: "2;37",
                    addition: "92", deletion: "91", warning: "93"
                )
            case .mono:
                return Palette(
                    heading1: "1", heading2: "1", heading3: "1",
                    accent: "4", secondary: "2", code: "7", keyword: "1",
                    type: "4", string: "0", number: "0", comment: "2",
                    addition: "0", deletion: "0", warning: "1"
                )
            }
        }
    }

    private let ansi: Bool
    private let palette: Palette

    public init(color: Bool, theme: Theme = .auto) {
        self.ansi = color && theme != .mono
        self.palette = Palette.resolve(theme)
    }

    /// Renders Markdown at the current terminal width. OSC-8 links are opt-in and
    /// should only be enabled after terminal capability detection.
    public func render(
        _ markdown: String,
        width: Int = 100,
        hyperlinks: Bool = false
    ) -> MarkdownRenderResult {
        let safeSource = Self.sanitized(markdown).replacingOccurrences(of: "\r\n", with: "\n")
        let document = Document(parsing: safeSource)
        var context = RenderContext(owner: self, width: max(24, width), hyperlinks: hyperlinks)
        let text = context.renderBlock(document).trimmingCharacters(in: .newlines)
        return MarkdownRenderResult(
            text: text,
            codeBlocks: context.codeBlocks,
            images: Self.imageReferences(in: safeSource)
        )
    }

    public static func imageReferences(in text: String) -> [TUIImageReference] {
        let safeSource = sanitized(text)
        let document = Document(parsing: safeSource)
        var results: [TUIImageReference] = []

        func visit(_ markup: Markup) {
            if let image = markup as? Image, let source = image.source {
                let path = source.trimmingCharacters(in: .whitespacesAndNewlines)
                if isLocalPath(path) {
                    let alt = plainText(image).trimmingCharacters(in: .whitespacesAndNewlines)
                    let value = TUIImageReference(
                        alt: alt.isEmpty ? URL(fileURLWithPath: path).lastPathComponent : alt,
                        path: path
                    )
                    if !results.contains(where: { $0.path == value.path }) { results.append(value) }
                }
            }
            for child in markup.children { visit(child) }
        }
        visit(document)

        let nsRange = NSRange(safeSource.startIndex..<safeSource.endIndex, in: safeSource)
        if let plainRegex = try? NSRegularExpression(
            pattern: #"(?:^|\s)((?:~|\.\.?/|/)[^\s<>\"']+\.(?:png|jpe?g|gif|webp|heic))\b"#,
            options: [.caseInsensitive]
        ) {
            for match in plainRegex.matches(in: safeSource, range: nsRange) {
                guard let pathRange = Range(match.range(at: 1), in: safeSource) else { continue }
                let path = String(safeSource[pathRange])
                if !results.contains(where: { $0.path == path }) {
                    results.append(TUIImageReference(alt: URL(fileURLWithPath: path).lastPathComponent, path: path))
                }
            }
        }
        return results
    }

    /// Compatibility entry point used by tests and downstream callers. Unlike the
    /// old fallback, it only consumes balanced math delimiters and preserves currency.
    public static func latexFallback(_ input: String) -> String {
        TerminalMath.renderInlineText(input)
    }

    /// Labels common untyped fences so highlighting and artifact actions remain useful
    /// when a model omits the Markdown language hint.
    public static func inferredLanguage(for content: String) -> String {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowercase = trimmed.lowercased()
        if trimmed.hasPrefix("diff --git ") || trimmed.contains("\n@@ ") { return "diff" }
        if lowercase.hasPrefix("<!doctype html") || lowercase.hasPrefix("<html") { return "html" }
        if (trimmed.hasPrefix("{") || trimmed.hasPrefix("[")),
           (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) != nil { return "json" }
        if trimmed.hasPrefix("#!/") || trimmed.contains("\nset -e") { return "bash" }
        if trimmed.contains("func ") && (trimmed.contains("let ") || trimmed.contains("var ")) { return "swift" }
        if trimmed.contains("def ") && trimmed.contains(":") { return "python" }
        if trimmed.contains("const ") || trimmed.contains("function ") || trimmed.contains("=>") { return "javascript" }
        if trimmed.contains("SELECT ") && trimmed.uppercased().contains(" FROM ") { return "sql" }
        if trimmed.contains("apiVersion:") && trimmed.contains("kind:") { return "yaml" }
        return ""
    }

    private static func isLocalPath(_ value: String) -> Bool {
        value.hasPrefix("/") || value.hasPrefix("~") || value.hasPrefix("./") || value.hasPrefix("../")
    }

    private static func sanitized(_ value: String) -> String {
        var result = ""
        result.reserveCapacity(value.count)
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x09, 0x0A, 0x0D:
                result.unicodeScalars.append(scalar)
            case 0x00...0x08, 0x0B...0x0C, 0x0E...0x1F, 0x7F:
                result.append("�")
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }

    private static func plainText(_ markup: Markup) -> String {
        if let text = markup as? Text { return text.string }
        if let code = markup as? InlineCode { return code.code }
        if let html = markup as? InlineHTML { return html.rawHTML }
        if markup is SoftBreak { return " " }
        if markup is LineBreak { return "\n" }
        return markup.children.map(plainText).joined()
    }

    private func style(_ value: String, _ code: String) -> String {
        ansi ? "\u{001B}[\(code)m\(value)\u{001B}[0m" : value
    }

    private struct RenderContext {
        let owner: TerminalMarkdownRenderer
        let width: Int
        let hyperlinks: Bool
        var codeBlocks: [TUICodeBlock] = []

        mutating func renderBlock(_ markup: Markup) -> String {
            if markup is Document { return renderBlocks(markup.children) }
            if let paragraph = markup as? Paragraph {
                let raw = TerminalMarkdownRenderer.plainText(paragraph)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if raw.hasPrefix("$$"), raw.hasSuffix("$$"), raw.count >= 4 {
                    let body = String(raw.dropFirst(2).dropLast(2))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    return renderMathBlock(body)
                }
                return renderInlineChildren(paragraph)
            }
            if let heading = markup as? Heading {
                let value = renderInlineChildren(heading)
                switch heading.level {
                case 1: return owner.style("▌ \(value)", owner.palette.heading1)
                case 2: return owner.style("▸ \(value)", owner.palette.heading2)
                default: return owner.style("• \(value)", owner.palette.heading3)
                }
            }
            if let code = markup as? CodeBlock { return renderCodeBlock(code) }
            if let quote = markup as? BlockQuote {
                return renderBlocks(quote.children).split(separator: "\n", omittingEmptySubsequences: false)
                    .map { owner.style("│", owner.palette.accent) + " " + $0 }.joined(separator: "\n")
            }
            if let list = markup as? OrderedList {
                return renderList(list.children, start: Int(list.startIndex), ordered: true)
            }
            if let list = markup as? UnorderedList {
                return renderList(list.children, start: 1, ordered: false)
            }
            if let table = markup as? Table { return renderTable(table) }
            if markup is ThematicBreak {
                return owner.style(String(repeating: "─", count: min(width, 72)), owner.palette.secondary)
            }
            if let html = markup as? HTMLBlock {
                return renderPanel(html.rawHTML.trimmingCharacters(in: .newlines), label: "html")
            }
            return renderBlocks(markup.children)
        }

        private mutating func renderBlocks(_ children: MarkupChildren) -> String {
            children.map { renderBlock($0) }.filter { !$0.isEmpty }.joined(separator: "\n\n")
        }

        private mutating func renderInlineChildren(_ markup: Markup) -> String {
            markup.children.map { renderInline($0) }.joined()
        }

        private mutating func renderInline(_ markup: Markup) -> String {
            if let text = markup as? Text { return TerminalMath.renderInlineText(text.string, style: mathStyle) }
            if let code = markup as? InlineCode { return owner.style(" \(code.code) ", owner.palette.code) }
            if markup is Strong { return owner.style(renderInlineChildren(markup), "1") }
            if markup is Emphasis { return owner.style(renderInlineChildren(markup), "3") }
            if markup is Strikethrough {
                let value = renderInlineChildren(markup)
                return owner.ansi ? owner.style(value, "9") : "~~\(value)~~"
            }
            if let link = markup as? Link {
                let label = renderInlineChildren(link)
                guard let destination = link.destination, !destination.isEmpty else { return label }
                let safeDestination = destination.unicodeScalars.filter { $0.value >= 0x20 && $0.value != 0x7F }
                    .map(String.init).joined()
                let visible = label == safeDestination ? label : "\(label) \(owner.style("(\(safeDestination))", owner.palette.secondary))"
                guard hyperlinks else { return visible }
                return "\u{001B}]8;;\(safeDestination)\u{0007}\(visible)\u{001B}]8;;\u{0007}"
            }
            if let image = markup as? Image {
                let alt = renderInlineChildren(image)
                let source = image.source ?? ""
                let label = alt.isEmpty ? "image" : alt
                return owner.style("🖼 \(label)", owner.palette.accent) + (source.isEmpty ? "" : " (\(source))")
            }
            if let html = markup as? InlineHTML { return owner.style(html.rawHTML, owner.palette.warning) }
            if markup is LineBreak { return "\n" }
            if markup is SoftBreak { return " " }
            return renderInlineChildren(markup)
        }

        private var mathStyle: (String) -> String { { owner.style($0, owner.palette.accent) } }

        private mutating func renderMathBlock(_ source: String) -> String {
            let math = TerminalMath.renderExpression(source)
            let label = owner.style(" math ", owner.palette.accent)
            let lines = math.split(separator: "\n", omittingEmptySubsequences: false).map {
                owner.style("│", owner.palette.accent) + " " + owner.style(String($0), owner.palette.accent)
            }.joined(separator: "\n")
            return owner.style("┌─", owner.palette.accent) + label + "\n" + lines + "\n" + owner.style("└─", owner.palette.accent)
        }

        private mutating func renderList(_ children: MarkupChildren, start: Int, ordered: Bool) -> String {
            var lines: [String] = []
            for (offset, child) in children.enumerated() {
                guard let item = child as? ListItem else { continue }
                let checkbox: String
                switch item.checkbox {
                case .checked?: checkbox = owner.style("☑ ", owner.palette.addition)
                case .unchecked?: checkbox = owner.style("☐ ", owner.palette.warning)
                case nil: checkbox = ""
                }
                let marker = ordered ? "\(start + offset)." : "•"
                let prefix = owner.style(marker, owner.palette.accent) + " " + checkbox
                let continuation = String(repeating: " ", count: marker.count + 1)
                let itemLines = renderBlocks(item.children).split(separator: "\n", omittingEmptySubsequences: false)
                for (index, line) in itemLines.enumerated() {
                    lines.append((index == 0 ? prefix : continuation) + String(line))
                }
            }
            return lines.joined(separator: "\n")
        }

        private mutating func renderTable(_ table: Table) -> String {
            let header = Array(table.head.cells.map { Self.normalizedCell($0) })
            let body = Array(table.body.rows.map { row in
                Array(row.cells.map { Self.normalizedCell($0) })
            })
            let columnCount = max(header.count, body.map(\.count).max() ?? 0)
            guard columnCount > 0 else { return "" }
            var widths = (0..<columnCount).map { column in
                max(3, ([header] + body).compactMap { row in row.indices.contains(column) ? row[column].count : nil }.max() ?? 3)
            }
            let budget = max(columnCount * 3, width - (columnCount * 3 + 1))
            while widths.reduce(0, +) > budget,
                  let index = widths.indices.max(by: { widths[$0] < widths[$1] }), widths[index] > 3 {
                widths[index] -= 1
            }
            let alignments = table.columnAlignments
            var rows = [owner.style(tableRule("┌", "┬", "┐", widths), owner.palette.secondary)]
            rows.append(tableRow(header, widths: widths, alignments: alignments, header: true))
            rows.append(owner.style(tableRule("├", "┼", "┤", widths), owner.palette.secondary))
            rows.append(contentsOf: body.map { tableRow($0, widths: widths, alignments: alignments, header: false) })
            rows.append(owner.style(tableRule("└", "┴", "┘", widths), owner.palette.secondary))
            return rows.joined(separator: "\n")
        }

        private static func normalizedCell(_ cell: Table.Cell) -> String {
            TerminalMarkdownRenderer.latexFallback(TerminalMarkdownRenderer.plainText(cell))
                .replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
        }

        private func tableRule(_ left: String, _ middle: String, _ right: String, _ widths: [Int]) -> String {
            left + widths.map { String(repeating: "─", count: $0 + 2) }.joined(separator: middle) + right
        }

        private func tableRow(_ cells: [String], widths: [Int], alignments: [Table.ColumnAlignment?], header: Bool) -> String {
            let values = widths.indices.map { column -> String in
                let raw = cells.indices.contains(column) ? cells[column] : ""
                let clipped = raw.count > widths[column] ? String(raw.prefix(max(1, widths[column] - 1))) + "…" : raw
                let missing = max(0, widths[column] - clipped.count)
                let aligned: String
                switch alignments.indices.contains(column) ? alignments[column] : nil {
                case .right?: aligned = String(repeating: " ", count: missing) + clipped
                case .center?:
                    let left = missing / 2
                    aligned = String(repeating: " ", count: left) + clipped + String(repeating: " ", count: missing - left)
                default: aligned = clipped + String(repeating: " ", count: missing)
                }
                return header ? owner.style(aligned, "1") : aligned
            }
            return owner.style("│", owner.palette.secondary) + " " + values.joined(
                separator: " " + owner.style("│", owner.palette.secondary) + " "
            ) + " " + owner.style("│", owner.palette.secondary)
        }

        private mutating func renderCodeBlock(_ block: CodeBlock) -> String {
            let content = block.code.hasSuffix("\n") ? String(block.code.dropLast()) : block.code
            let declared = block.language?.split(whereSeparator: \.isWhitespace).first.map(String.init)?.lowercased() ?? ""
            let language = declared.isEmpty ? TerminalMarkdownRenderer.inferredLanguage(for: content) : declared
            codeBlocks.append(TUICodeBlock(language: language, content: content))
            return renderPanel(content, label: language.isEmpty ? "code" : language)
        }

        private func renderPanel(_ content: String, label: String) -> String {
            let sourceLines = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
            let digits = max(1, String(sourceLines.count).count)
            let header = owner.style("┌─", owner.palette.secondary) + owner.style(" \(label) ", owner.palette.accent)
            let lines = sourceLines.enumerated().map { index, line in
                let number = String(format: "%\(digits)d", index + 1)
                return owner.style("│ \(number) │", owner.palette.secondary) + " " + highlight(line, language: label)
            }
            let footer = owner.style("└" + String(repeating: "─", count: min(max(8, label.count + 4), width - 1)), owner.palette.secondary)
            return ([header] + lines + [footer]).joined(separator: "\n")
        }

        private func highlight(_ line: String, language: String) -> String {
            guard owner.ansi else { return line }
            let language = language.lowercased()
            if ["diff", "patch"].contains(language) {
                if line.hasPrefix("diff --git") || line.hasPrefix("---") || line.hasPrefix("+++") {
                    return owner.style(line, "1;" + owner.palette.accent)
                }
                if line.hasPrefix("@@") { return owner.style(line, owner.palette.accent) }
                if line.hasPrefix("+") { return owner.style(line, owner.palette.addition) }
                if line.hasPrefix("-") { return owner.style(line, owner.palette.deletion) }
                return owner.style(line, owner.palette.secondary)
            }
            if ["html", "xml", "svg"].contains(language) { return highlightMarkup(line) }

            let keywords: Set<String> = [
                "actor", "as", "async", "await", "break", "case", "catch", "class", "const", "continue",
                "def", "defer", "do", "else", "enum", "export", "extension", "false", "final", "fn", "for",
                "from", "func", "function", "guard", "if", "import", "in", "interface", "internal", "let", "mut",
                "new", "nil", "null", "package", "private", "protocol", "public", "repeat", "return", "self",
                "static", "struct", "switch", "throw", "throws", "true", "try", "typealias", "use", "var", "while", "yield"
            ]
            let hashComments = ["bash", "sh", "zsh", "python", "py", "ruby", "yaml", "yml", "toml"].contains(language)
            let dashComments = ["sql", "lua", "haskell"].contains(language)
            let characters = Array(line)
            var output = ""
            var index = 0
            while index < characters.count {
                if index + 1 < characters.count,
                   (String(characters[index...index + 1]) == "//" || (dashComments && String(characters[index...index + 1]) == "--")) {
                    output += owner.style(String(characters[index...]), owner.palette.comment); break
                }
                if hashComments && characters[index] == "#" {
                    output += owner.style(String(characters[index...]), owner.palette.comment); break
                }
                if characters[index] == "\"" || characters[index] == "'" || characters[index] == "`" {
                    let quote = characters[index]
                    var end = index + 1
                    var escaped = false
                    while end < characters.count {
                        let value = characters[end]
                        if value == quote && !escaped { end += 1; break }
                        escaped = value == "\\" && !escaped
                        if value != "\\" { escaped = false }
                        end += 1
                    }
                    output += owner.style(String(characters[index..<min(end, characters.count)]), owner.palette.string)
                    index = end; continue
                }
                if characters[index].isNumber {
                    var end = index + 1
                    while end < characters.count && (characters[end].isNumber || ".xabcdefABCDEF_".contains(characters[end])) { end += 1 }
                    output += owner.style(String(characters[index..<end]), owner.palette.number)
                    index = end; continue
                }
                if characters[index].isLetter || characters[index] == "_" {
                    var end = index + 1
                    while end < characters.count && (characters[end].isLetter || characters[end].isNumber || characters[end] == "_") { end += 1 }
                    let word = String(characters[index..<end])
                    let next = characters.dropFirst(end).first { !$0.isWhitespace }
                    if keywords.contains(word) { output += owner.style(word, owner.palette.keyword) }
                    else if word.first?.isUppercase == true { output += owner.style(word, owner.palette.type) }
                    else if next == "(" { output += owner.style(word, owner.palette.accent) }
                    else { output += word }
                    index = end; continue
                }
                output.append(characters[index]); index += 1
            }
            return output
        }

        private func highlightMarkup(_ line: String) -> String {
            let characters = Array(line)
            var output = ""
            var index = 0
            while index < characters.count {
                if index + 3 < characters.count, String(characters[index...index + 3]) == "<!--" {
                    output += owner.style(String(characters[index...]), owner.palette.comment); break
                }
                if characters[index] == "<", let end = characters[index...].firstIndex(of: ">") {
                    output += owner.style(String(characters[index...end]), owner.palette.keyword)
                    index = end + 1
                } else {
                    output.append(characters[index]); index += 1
                }
            }
            return output
        }
    }
}

private enum TerminalMath {
    private static let commandSymbols: [String: String] = [
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "theta": "θ",
        "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "phi": "φ", "psi": "ψ",
        "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Pi": "Π",
        "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "infty": "∞", "sum": "∑",
        "prod": "∏", "int": "∫", "oint": "∮", "partial": "∂", "nabla": "∇", "neq": "≠",
        "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "approx": "≈", "equiv": "≡",
        "propto": "∝", "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒",
        "Leftarrow": "⇐", "leftrightarrow": "↔", "times": "×", "cdot": "·", "pm": "±", "mp": "∓",
        "div": "÷", "forall": "∀", "exists": "∃", "in": "∈", "notin": "∉", "subset": "⊂",
        "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇", "cup": "∪", "cap": "∩", "land": "∧", "lor": "∨"
    ]
    private static let superscripts: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
        "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ"
    ]
    private static let subscripts: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
        "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎", "a": "ₐ",
        "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ",
        "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "x": "ₓ"
    ]

    static func renderInlineText(_ input: String, style: (String) -> String = { $0 }) -> String {
        let characters = Array(input)
        var result = ""
        var index = 0
        while index < characters.count {
            if characters[index] == "\\", index + 1 < characters.count, characters[index + 1] == "$" {
                result.append("$"); index += 2; continue
            }
            guard characters[index] == "$", index + 1 < characters.count,
                  characters[index + 1] != "$", !characters[index + 1].isWhitespace else {
                result.append(characters[index]); index += 1; continue
            }
            var closing = index + 1
            var found: Int?
            while closing < characters.count {
                if characters[closing] == "$", characters[closing - 1] != "\\" {
                    let previousIsContent = !characters[closing - 1].isWhitespace
                    let nextIsBoundary = closing + 1 == characters.count || characters[closing + 1].isWhitespace || ".,;:!?)]}".contains(characters[closing + 1])
                    if previousIsContent && nextIsBoundary { found = closing }
                    break
                }
                closing += 1
            }
            guard let end = found else { result.append("$"); index += 1; continue }
            result += style(renderExpression(String(characters[(index + 1)..<end])))
            index = end + 1
        }
        return result
    }

    static func renderExpression(_ source: String) -> String {
        if let matrix = renderMatrix(source) { return matrix }
        var parser = Parser(Array(source))
        return parser.parse(until: nil).replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func renderMatrix(_ source: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: #"\\begin\{(bmatrix|pmatrix|matrix|cases)\}([\s\S]*?)\\end\{\1\}"#) else { return nil }
        let range = NSRange(source.startIndex..<source.endIndex, in: source)
        guard let match = regex.firstMatch(in: source, range: range),
              let environmentRange = Range(match.range(at: 1), in: source),
              let bodyRange = Range(match.range(at: 2), in: source) else { return nil }
        let environment = String(source[environmentRange])
        let cells = String(source[bodyRange]).components(separatedBy: "\\\\").map { row in
            row.split(separator: "&", omittingEmptySubsequences: false).map { renderExpression(String($0)) }
        }
        let columns = cells.map(\.count).max() ?? 0
        let widths = (0..<columns).map { column in cells.compactMap { $0.indices.contains(column) ? $0[column].count : nil }.max() ?? 0 }
        return cells.enumerated().map { rowIndex, row in
            let body = widths.indices.map { column -> String in
                let value = row.indices.contains(column) ? row[column] : ""
                return value + String(repeating: " ", count: max(0, widths[column] - value.count))
            }.joined(separator: "  ")
            switch environment {
            case "bmatrix": return "[ \(body) ]"
            case "pmatrix": return "( \(body) )"
            case "cases": return (rowIndex == 0 ? "⎧ " : rowIndex == cells.count - 1 ? "⎩ " : "⎨ ") + body
            default: return "│ \(body) │"
            }
        }.joined(separator: "\n")
    }

    private struct Parser {
        let characters: [Character]
        var index = 0

        init(_ characters: [Character]) { self.characters = characters }

        mutating func parse(until terminator: Character?) -> String {
            var output = ""
            while index < characters.count {
                let character = characters[index]
                if let terminator, character == terminator { index += 1; break }
                if character == "\\" { output += parseCommand(); continue }
                if character == "^" || character == "_" {
                    index += 1
                    let argument = parseArgument()
                    let map = character == "^" ? TerminalMath.superscripts : TerminalMath.subscripts
                    output += argument.allSatisfy({ map[$0] != nil })
                        ? String(argument.compactMap { map[$0] })
                        : (character == "^" ? "^(\(argument))" : "_(\(argument))")
                    continue
                }
                if character == "{" { index += 1; output += parse(until: "}"); continue }
                if character == "}" && terminator == nil { index += 1; continue }
                if character == "~" { output.append(" "); index += 1; continue }
                output.append(character); index += 1
            }
            return output
        }

        private mutating func parseCommand() -> String {
            index += 1
            guard index < characters.count else { return "\\" }
            if characters[index] == "\\" { index += 1; return "\n" }
            let start = index
            while index < characters.count && characters[index].isLetter { index += 1 }
            if start == index { let value = characters[index]; index += 1; return String(value) }
            let name = String(characters[start..<index])
            if let symbol = TerminalMath.commandSymbols[name] { return symbol }
            if name == "frac" { return "(\(parseArgument()))/(\(parseArgument()))" }
            if name == "sqrt" {
                if index < characters.count, characters[index] == "[" {
                    while index < characters.count && characters[index] != "]" { index += 1 }
                    if index < characters.count { index += 1 }
                }
                return "√(\(parseArgument()))"
            }
            if ["text", "textrm", "textbf", "mathrm", "mathbf", "operatorname", "overline", "underline"].contains(name) {
                return parseArgument()
            }
            if ["left", "right"].contains(name) { return "" }
            if [",", ";", "quad", "qquad"].contains(name) { return " " }
            return "\\\(name)"
        }

        private mutating func parseArgument() -> String {
            while index < characters.count && characters[index].isWhitespace { index += 1 }
            guard index < characters.count else { return "" }
            if characters[index] == "{" { index += 1; return parse(until: "}") }
            let value = characters[index]; index += 1; return String(value)
        }
    }
}
