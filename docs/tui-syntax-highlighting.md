# TUI syntax highlighting

AFM's terminal UI uses Tree-sitter for grammar-backed code highlighting. The
runtime, Swift binding, and each grammar are pinned in `Package.swift`; the
parsers are compiled and linked into `afm`. There is no runtime download,
external executable, JavaScript engine, optional import, or grammar lookup.

Supported fence labels are Bash, C, C++, C#, CSS, diff, Go, HTML, Java,
JavaScript/JSX, JSON, Kotlin, Markdown, PHP, Python, Ruby, Rust, SQL, Swift,
TOML, TSX, TypeScript, and YAML. Common aliases such as `sh`, `zsh`, `c++`,
`c#`, `jsx`, `py`, `rb`, `rs`, `ts`, and `yml` resolve to those parsers.
Unknown labels retain the small safe fallback lexer.

The breadth is not free: generated parser tables add tens of megabytes to the
universal CLI compared with the former handwritten lexer. This is an explicit
tradeoff for deterministic offline coverage; Homebrew users still receive a
precompiled binary and do not compile or download grammars at runtime.

## Why Tree-sitter

- Tree-sitter is native, incremental, error-tolerant, and gives AFM syntax
  structure without embedding a browser or JavaScript VM.
- Shiki is an excellent modern TextMate-compatible highlighter, but its normal
  execution model is JavaScript/Wasm plus Oniguruma and its output is aimed at
  HTML rather than ANSI terminal spans.
- highlight.js, used by llama.cpp's Web UI, is a strong browser choice but is a
  lexical JavaScript highlighter. Embedding it in the native CLI would require
  JavaScriptCore and HTML-to-ANSI translation.
- Neon is a promising Swift Tree-sitter editor layer, but it is a broader text
  system and its project still describes the main branch as not ready for a
  release. AFM only needs deterministic, read-only code-block rendering.

## Build integrity

Direct imports make every promised grammar a compile-time dependency. Tests
parse a sentinel with all 23 grammars and verify each grammar ABI against the
pinned Tree-sitter runtime. Popular-language fixtures also cover malformed,
multiline, and Unicode input.

Four official grammar manifests use a working-directory-sensitive check that
can omit their external C scanners when consumed transitively. AFM therefore
pins licensed copies of those scanners in `AFMTreeSitterScanners`; missing
scanner code is a hard final-link failure. See that target's README for source
revisions and update requirements.

The highlighter refuses blocks larger than 1 MB. This keeps adversarial or
accidentally enormous generated blocks from monopolizing the interactive TUI;
the original code remains displayed unmodified through the fallback path.
