import Foundation
import SwiftTreeSitter
import TreeSitterBash
import TreeSitterC
import TreeSitterCPP
import TreeSitterCSharp
import TreeSitterCSS
import TreeSitterDiff
import TreeSitterGo
import TreeSitterHTML
import TreeSitterJava
import TreeSitterJavaScript
import TreeSitterJSON
import TreeSitterKotlin
import TreeSitterMarkdown
import TreeSitterPHP
import TreeSitterPython
import TreeSitterRuby
import TreeSitterRust
import TreeSitterSql
import TreeSitterSwift
import TreeSitterTOML
import TreeSitterTSX
import TreeSitterTypeScript
import TreeSitterYAML

enum TUISyntaxKind: Int, CaseIterable, Sendable {
    case comment
    case string
    case number
    case keyword
    case type
    case function
    case attribute
    case `operator`
    case addition
    case deletion
    case metadata
}

struct TUISyntaxToken: Equatable, Sendable {
    let range: NSRange
    let kind: TUISyntaxKind
}

/// Source-compiled Tree-sitter registry used by the terminal renderer.
///
/// There are deliberately no optional imports or runtime grammar downloads here.
/// Removing any parser dependency makes AFMTerminalUI fail to compile, and the
/// build-integrity tests parse a sentinel with every registered grammar.
enum TreeSitterSyntaxHighlighter {
    static let maximumSourceBytes = 1_000_000

    static let supportedLanguages = [
        "bash", "c", "cpp", "csharp", "css", "diff", "go", "html", "java",
        "javascript", "json", "kotlin", "markdown", "php", "python", "ruby",
        "rust", "sql", "swift", "toml", "tsx", "typescript", "yaml"
    ]

    private static let aliases: [String: String] = [
        "shell": "bash", "sh": "bash", "zsh": "bash", "console": "bash",
        "h": "c",
        "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp",
        "c#": "csharp", "cs": "csharp", "dotnet": "csharp",
        "patch": "diff",
        "golang": "go",
        "htm": "html", "xml": "html", "svg": "html",
        "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
        "kt": "kotlin", "kts": "kotlin",
        "md": "markdown", "mdown": "markdown",
        "py": "python", "python3": "python",
        "rb": "ruby",
        "rs": "rust",
        "postgres": "sql", "postgresql": "sql", "mysql": "sql", "sqlite": "sql",
        "ts": "typescript",
        "yml": "yaml"
    ]

    static func canonicalLanguage(_ rawValue: String) -> String? {
        let value = rawValue.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if supportedLanguages.contains(value) { return value }
        return aliases[value]
    }

    static func tokens(in source: String, language rawLanguage: String) -> [TUISyntaxToken]? {
        guard source.utf8.count <= maximumSourceBytes,
              let languageName = canonicalLanguage(rawLanguage),
              let language = language(named: languageName) else {
            return nil
        }

        // The PHP grammar models a complete document. Fenced snippets normally
        // omit `<?php`; add it only for parsing, then translate ranges back to
        // the user's untouched source.
        let parsePrefix = languageName == "php" && !source.contains("<?") ? "<?php " : ""
        let parsedSource = parsePrefix + source
        let prefixLength = (parsePrefix as NSString).length
        let parser = Parser()
        do {
            try parser.setLanguage(language)
        } catch {
            return nil
        }
        guard let root = parser.parse(parsedSource)?.rootNode else { return nil }

        let sourceNSString = parsedSource as NSString
        var candidates: [TUISyntaxToken] = []

        func add(_ node: Node, kind: TUISyntaxKind) {
            let parsedRange = node.range
            guard parsedRange.location >= prefixLength, parsedRange.length > 0,
                  NSMaxRange(parsedRange) <= sourceNSString.length else { return }
            candidates.append(TUISyntaxToken(
                range: NSRange(location: parsedRange.location - prefixLength, length: parsedRange.length),
                kind: kind
            ))
        }

        func visit(_ node: Node) {
            let nodeType = node.nodeType ?? ""
            if let kind = wholeNodeKind(nodeType) {
                add(node, kind: kind)
                return
            }

            if node.childCount == 0 {
                let value = sourceNSString.substring(with: node.range)
                if let kind = leafKind(node: node, type: nodeType, value: value) {
                    add(node, kind: kind)
                }
                return
            }

            for index in 0..<node.childCount {
                if let child = node.child(at: index) { visit(child) }
            }
        }

        visit(root)
        return nonOverlappingTokens(candidates)
    }

    /// Exercises every compiled parser and verifies its ABI against the linked runtime.
    /// Release/build tests call this so a missing or incompatible grammar is a hard failure.
    static func validateCompiledGrammars() -> [String] {
        supportedLanguages.compactMap { name in
            guard let language = language(named: name) else { return name }
            guard language.ABIVersion >= Language.minimumCompatibleVersion,
                  language.ABIVersion <= Language.version else { return name }
            let parser = Parser()
            do {
                try parser.setLanguage(language)
            } catch {
                return name
            }
            return parser.parse(sentinel(for: name))?.rootNode == nil ? name : nil
        }
    }

    private static func language(named name: String) -> Language? {
        switch name {
        case "bash": Language(tree_sitter_bash())
        case "c": Language(tree_sitter_c())
        case "cpp": Language(tree_sitter_cpp())
        case "csharp": Language(tree_sitter_c_sharp())
        case "css": Language(tree_sitter_css())
        case "diff": Language(tree_sitter_diff())
        case "go": Language(tree_sitter_go())
        case "html": Language(tree_sitter_html())
        case "java": Language(tree_sitter_java())
        case "javascript": Language(tree_sitter_javascript())
        case "json": Language(tree_sitter_json())
        case "kotlin": Language(tree_sitter_kotlin())
        case "markdown": Language(tree_sitter_markdown())
        case "php": Language(tree_sitter_php())
        case "python": Language(tree_sitter_python())
        case "ruby": Language(tree_sitter_ruby())
        case "rust": Language(tree_sitter_rust())
        case "sql": Language(tree_sitter_sql())
        case "swift": Language(tree_sitter_swift())
        case "toml": Language(tree_sitter_toml())
        case "tsx": Language(tree_sitter_tsx())
        case "typescript": Language(tree_sitter_typescript())
        case "yaml": Language(tree_sitter_yaml())
        default: nil
        }
    }

    private static func wholeNodeKind(_ type: String) -> TUISyntaxKind? {
        let value = type.lowercased()
        if value == "addition" { return .addition }
        if value == "deletion" { return .deletion }
        if value == "hunk" || value == "location" || value == "old_file" || value == "new_file" ||
            value == "file_change" || value == "commit" || value == "index" {
            return .metadata
        }
        if value.contains("comment") { return .comment }
        if value.contains("string") || value.contains("heredoc") || value.contains("regex_pattern") ||
            value == "regex" || value == "character_literal" || value == "char_literal" {
            return .string
        }
        if value == "integer" || value == "float" || value == "number" ||
            value.contains("integer_literal") || value.contains("float_literal") ||
            value.contains("number_literal") || value.contains("numeric_literal") ||
            value == "integer_value" || value == "float_value" ||
            value == "integer_scalar" || value == "float_scalar" {
            return .number
        }
        if value == "boolean_scalar" || value == "null_scalar" { return .keyword }
        if value == "string_scalar" || value == "plain_scalar" || value == "double_quote_scalar" ||
            value == "single_quote_scalar" || value == "block_scalar" {
            return .string
        }
        if value == "type_identifier" || value == "primitive_type" || value == "predefined_type" ||
            value == "builtin_type" || value == "scoped_type_identifier" {
            return .type
        }
        if value == "tag_name" { return .keyword }
        if value == "attribute_name" || value == "property_name" || value == "property_identifier" {
            return .attribute
        }
        if value.contains("attribute") || value.contains("annotation") || value == "decorator" {
            return .attribute
        }
        if value.contains("heading") { return .type }
        if value == "code_span" || value == "fenced_code_block" || value == "inline_link" { return .string }
        if value == "emphasis" || value == "strong_emphasis" || value == "link_destination" { return .attribute }
        return nil
    }

    private static func leafKind(node: Node, type: String, value: String) -> TUISyntaxKind? {
        let lowercaseType = type.lowercased()
        if ["true", "false", "null", "nil", "none", "self", "super", "this"].contains(value.lowercased()) {
            return .keyword
        }
        if !node.isNamed, value.unicodeScalars.contains(where: CharacterSet.letters.contains) {
            return .keyword
        }
        if lowercaseType.contains("keyword") { return .keyword }
        if lowercaseType.contains("operator") || operatorTokens.contains(value) { return .operator }
        if lowercaseType.contains("function") || lowercaseType.contains("method") || isCallableIdentifier(node) {
            return .function
        }
        if lowercaseType == "type_identifier" || lowercaseType.contains("primitive_type") {
            return .type
        }
        return nil
    }

    private static func isCallableIdentifier(_ node: Node) -> Bool {
        guard node.nodeType?.lowercased().contains("identifier") == true,
              let parentType = node.parent?.nodeType?.lowercased() else { return false }
        return parentType.contains("call") || parentType.contains("invocation") ||
            parentType.contains("function_declarator") || parentType.contains("method_declaration")
    }

    private static let operatorTokens: Set<String> = [
        "=", "==", "===", "!=", "!==", "<", ">", "<=", ">=", "+", "-", "*", "/", "%",
        "&&", "||", "!", "??", "?.", "=>", "->", "::", "..", "...", "|>", "&", "|", "^", "~"
    ]

    private static func nonOverlappingTokens(_ values: [TUISyntaxToken]) -> [TUISyntaxToken] {
        let sorted = values.sorted {
            if $0.range.location != $1.range.location { return $0.range.location < $1.range.location }
            if $0.range.length != $1.range.length { return $0.range.length > $1.range.length }
            return $0.kind.rawValue < $1.kind.rawValue
        }
        var result: [TUISyntaxToken] = []
        for token in sorted {
            guard result.last.map({ NSMaxRange($0.range) <= token.range.location }) ?? true else { continue }
            result.append(token)
        }
        return result
    }

    private static func sentinel(for language: String) -> String {
        switch language {
        case "bash": "echo test"
        case "c", "cpp": "int main(void) { return 0; }"
        case "csharp", "java", "kotlin": "class Test {}"
        case "css": ".test { color: red; }"
        case "diff": "--- a/a\n+++ b/a\n-old\n+new"
        case "go": "package main\nfunc main() {}"
        case "html": "<main>test</main>"
        case "javascript", "typescript", "tsx": "const test = 1;"
        case "json": "{\"test\": true}"
        case "markdown": "# Test"
        case "php": "<?php echo 'test';"
        case "python": "def test():\n    return True"
        case "ruby": "def test\nend"
        case "rust": "fn main() {}"
        case "sql": "SELECT 1;"
        case "swift": "func test() {}"
        case "toml": "test = true"
        case "yaml": "test: true"
        default: "test"
        }
    }
}
