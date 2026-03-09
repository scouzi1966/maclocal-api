# XGrammar C++ Interop Design

**Date:** 2026-03-05
**Branch:** `feature/inference-optimizations`
**Status:** Approved

## Goal

Replace the XGrammar Python subprocess bridge with native C++ interop, eliminating the Python runtime dependency while providing true constrained decoding for `response_format: json_schema` and `--guided-json`.

## Background

The current implementation uses two layers for structured output:

1. **Prompt injection** (always active) — injects "Respond with valid JSON matching this schema..." into the prompt. Works but not guaranteed (model can still produce invalid JSON).
2. **XGrammar Python bridge** (current branch) — `XGrammarBridge.swift` spawns `Scripts/xgrammar-bridge.py` as a subprocess, communicating via JSON-lines over stdin/stdout pipes. Provides hard constrained decoding by masking logits. Requires Python 3 + `xgrammar` pip package.

The Python dependency is unacceptable for a native macOS binary. This design replaces layer 2 with compiled-in C++ code.

## Architecture

### Vendor Layer

**xgrammar** is added as a git submodule at `vendor/xgrammar` (following the existing `vendor/llama.cpp` and `vendor/mlx-swift-lm` pattern).

A thin C wrapper (adapted from [mlx-swift-structured](https://github.com/petrukha-ivan/mlx-swift-structured), Apache 2.0) provides `extern "C"` functions callable from Swift:

```
Sources/CXGrammar/
├── include/
│   ├── cxgrammar.h              # Umbrella header (includes all below)
│   ├── module.modulemap          # SPM module map for CXGrammar
│   └── cxgrammar/
│       ├── error_handler.h       # Error callback registration
│       ├── tokenizer_info.h      # Vocabulary + tokenizer metadata
│       ├── grammar_compiler.h    # JSON schema → CompiledGrammar
│       └── grammar_matcher.h     # Per-token bitmask + state advancement
├── error_handler.cpp
├── tokenizer_info.cpp
├── grammar_compiler.cpp
├── grammar_matcher.cpp
└── xgrammar -> ../../vendor/xgrammar  (symlink for SPM source discovery)
```

### SPM Target: CXGrammar

New `Package.swift` target that compiles:
- The 4 C++ wrapper files (`*.cpp`)
- The full xgrammar C++ source tree (via symlink to `vendor/xgrammar`)
- Excludes: `tests/`, `web/`, `nanobind/`, `cpptrace/`, `googletest/`

Settings:
- C++ language standard: `gnucxx17`
- Header search paths: `xgrammar/include`, `xgrammar/3rdparty/dlpack/include`, `xgrammar/3rdparty/picojson`

### C API Surface (12 functions)

```c
// Error handling
void set_error_handler(error_handler_closure handler);
void catch_error(const char* message);

// Tokenizer info
void* tokenizer_info_new(const char* const* vocab, size_t vocab_size,
                         int vocab_type, const int32_t* eos_tokens, size_t eos_count);
void  tokenizer_info_free(void* info);

// Grammar compilation
void* compile_json_schema_grammar(void* info, const char* schema, size_t len, int indent);
void* compile_ebnf_grammar(void* info, const char* grammar, size_t len);
void* compile_regex_grammar(void* info, const char* regex, size_t len);
void  compiled_grammar_free(void* grammar);

// Grammar matching (per-token)
void* grammar_matcher_new(void* compiled_grammar);
void  grammar_matcher_fill_next_token_bitmask(void* matcher, void* bitmask);
bool  grammar_matcher_accept_token(void* matcher, int32_t token_id);
bool  grammar_matcher_is_terminated(void* matcher);
void  grammar_matcher_reset(void* matcher);
void  grammar_matcher_free(void* matcher);
```

All functions use opaque `void*` pointers, cast to C++ types internally. Errors are caught via try-catch and forwarded through the error handler callback.

### Swift Integration: XGrammarService

New file `Sources/MacLocalAPI/Models/XGrammarService.swift` replaces `XGrammarBridge.swift`:

- Imports `CXGrammar`
- Wraps C functions in a Swift class
- Manages `TokenizerInfo` lifecycle (created once per model load from tokenizer vocabulary)
- `compileSchema(_ schema: [String: Any]) -> OpaquePointer?` — compiles JSON schema
- `createMatcher(_ grammar: OpaquePointer) -> OpaquePointer?` — creates stateful matcher
- `fillNextTokenBitmask(_ matcher, _ bitmask)` — fills bitmask for allowed tokens
- `acceptToken(_ matcher, _ tokenID)` — advances grammar state
- `isTerminated(_ matcher) -> Bool` — checks completion
- Converts bitmask → MLXArray mask (0.0 for allowed, -1e9 for disallowed)
- **Synchronous** — no subprocess, no async, no pipes. Called directly from TokenIterator loop.

### GrammarLogitProcessor Update

The existing `GrammarLogitProcessor` in `Scripts/patches/Evaluate.swift` changes minimally:

**Before (Python bridge):**
```swift
// Calls XGrammarSession.getAllowedTokens() via pipe I/O
// Returns [Int] array of allowed token IDs
// Sets disallowed logits to -1e9
```

**After (C++ native):**
```swift
// Calls grammar_matcher_fill_next_token_bitmask() via CXGrammar
// Returns bitmask, converted to MLXArray mask
// Adds mask to logits (0 for allowed, -inf for disallowed)
```

The `LogitProcessor` protocol interface is unchanged. The processor still lives in the vendor patch.

### MLXModelService Wiring

`setupGrammarConstraint()` changes from:

```
Python bridge: start subprocess → send compile command → create session → wrap in processor
```

To:

```
C++ native: extract vocab → create TokenizerInfo → compile schema → create matcher → wrap in processor
```

Key detail: vocabulary extraction happens once per model load. The tokenizer's vocabulary (array of token strings) is extracted from `MLXLMCommon`'s tokenizer and cached.

### Cleanup

Delete:
- `Sources/MacLocalAPI/Models/XGrammarBridge.swift` (Python bridge actor + XGrammarSession)
- `Scripts/xgrammar-bridge.py` (Python subprocess script)

### Impact on Existing Features

| Feature | Impact |
|---------|--------|
| `--guided-json` CLI flag | No change — still produces `ResponseFormat(type: "json_schema")`, which flows to the same `setupGrammarConstraint()` |
| `response_format: json_schema` API | No change — same entry point, now backed by native C++ |
| `response_format: json_object` API | No change — uses prompt injection only (no schema to compile) |
| Prompt injection fallback | Stays as baseline — runs regardless of grammar engine |
| `--kv-eviction` | No interaction |
| Radix tree prefix cache | No interaction |

### Error Handling

- **Schema compilation failure** (unsupported schema features): Log warning, fall back to prompt injection only. Generation still works, just without hard constraints.
- **Matcher runtime failure**: Log warning, release matcher, continue unconstrained.
- **Error callback**: `set_error_handler` registers a Swift closure that captures errors from C++ try-catch blocks. Thread-safe via NSLock.

### Build Impact

- xgrammar C++ source: ~15K lines
- Clean build overhead: ~20-30s on Apple Silicon
- Incremental builds: unaffected (C++ sources don't change)
- Binary size increase: ~1-2MB (static linking)
- No runtime dependencies added

### Build Script Update

`Scripts/build-from-scratch.sh` needs to initialize the `vendor/xgrammar` submodule (add to existing `git submodule update --init --recursive`).

## What NOT to Build

- **EBNF/regex grammar support** — vendored but not exposed in our Swift layer (only `compile_json_schema_grammar` used)
- **Structural tags** — not needed for OpenAI-compatible API
- **Batched grammar matching** — single-sequence only (matches our generation model)
- **Grammar caching across requests** — premature optimization; compile per-request for now
