# XGrammar C++ Interop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the XGrammar Python subprocess bridge with native C++ interop, compiled directly into the `afm` binary.

**Architecture:** Add xgrammar as a git submodule at `vendor/xgrammar`. Create a `CXGrammar` SPM target with a thin C wrapper (vendored from mlx-swift-structured, Apache 2.0) around xgrammar's C++ API. Replace `XGrammarBridge.swift` (Python pipes) with `XGrammarService.swift` (direct C calls). Update `GrammarLogitProcessor` to use bitmask-based masking instead of allowed-token-list.

**Tech Stack:** Swift 5.9+, C++17, SPM, xgrammar (C++ library), MLX Swift

---

### Task 1: Add xgrammar git submodule

**Files:**
- Create: `vendor/xgrammar` (git submodule)
- Modify: `.gitmodules`

**Step 1: Add submodule**

```bash
cd /Volumes/edata/dev/git/NIGHTLY/mar3/maclocal-api
git submodule add https://github.com/mlc-ai/xgrammar.git vendor/xgrammar
cd vendor/xgrammar && git checkout v0.1.17 && cd ../..
```

Use a tagged release for reproducibility. Check `https://github.com/mlc-ai/xgrammar/tags` for latest stable — v0.1.17 is current as of March 2026. If unavailable, use the latest tag or pin to a specific commit.

**Step 2: Verify submodule**

```bash
ls vendor/xgrammar/include/xgrammar/xgrammar.h
ls vendor/xgrammar/cpp/
ls vendor/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h
```

Expected: All three paths exist.

**Step 3: Commit**

```bash
git add vendor/xgrammar .gitmodules
git commit -m "vendor: add xgrammar C++ library as submodule"
```

---

### Task 2: Create CXGrammar C wrapper — header files

**Files:**
- Create: `Sources/CXGrammar/include/cxgrammar.h`
- Create: `Sources/CXGrammar/include/module.modulemap`
- Create: `Sources/CXGrammar/include/cxgrammar/error_handler.h`
- Create: `Sources/CXGrammar/include/cxgrammar/tokenizer_info.h`
- Create: `Sources/CXGrammar/include/cxgrammar/grammar_compiler.h`
- Create: `Sources/CXGrammar/include/cxgrammar/grammar_matcher.h`

These are adapted from mlx-swift-structured's `CMLXStructured` (Apache 2.0). The wrapper uses opaque `void*` pointers and `extern "C"` so Swift can call C++ xgrammar code.

**Step 1: Create directory structure**

```bash
mkdir -p Sources/CXGrammar/include/cxgrammar
```

**Step 2: Write umbrella header**

Create `Sources/CXGrammar/include/cxgrammar.h`:

```c
#include "cxgrammar/error_handler.h"
#include "cxgrammar/tokenizer_info.h"
#include "cxgrammar/grammar_compiler.h"
#include "cxgrammar/grammar_matcher.h"
```

**Step 3: Write module map**

Create `Sources/CXGrammar/include/module.modulemap`:

```
module CXGrammar {
    header "cxgrammar.h"
    export *
}
```

**Step 4: Write error_handler.h**

Create `Sources/CXGrammar/include/cxgrammar/error_handler.h`:

```c
#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*error_handler_closure)(const char*);
void set_error_handler(error_handler_closure handler);
void catch_error(const char* error_message);

#ifdef __cplusplus
}
#endif
```

**Step 5: Write tokenizer_info.h**

Create `Sources/CXGrammar/include/cxgrammar/tokenizer_info.h`:

```c
#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void* tokenizer_info_new(const char* const* vocab, size_t vocab_size,
                         const int vocab_type,
                         const int32_t* eos_tokens, size_t eos_tokens_size);
void tokenizer_info_free(void* tokenizer_info);

#ifdef __cplusplus
}
#endif
```

**Step 6: Write grammar_compiler.h**

Create `Sources/CXGrammar/include/cxgrammar/grammar_compiler.h`:

```c
#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void* compile_json_schema_grammar(void* tokenizer_info,
                                  const char* schema, size_t length,
                                  int indent);
void* compile_ebnf_grammar(void* tokenizer_info,
                            const char* grammar, size_t length);
void* compile_regex_grammar(void* tokenizer_info,
                             const char* regex, size_t length);
void compiled_grammar_free(void* compiled_grammar);

#ifdef __cplusplus
}
#endif
```

**Step 7: Write grammar_matcher.h**

Create `Sources/CXGrammar/include/cxgrammar/grammar_matcher.h`:

```c
#pragma once
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* grammar_matcher_new(void* compiled_grammar);
void  grammar_matcher_fill_next_token_bitmask(void* matcher, void* bitmask);
bool  grammar_matcher_accept_token(void* matcher, int32_t token_id);
bool  grammar_matcher_is_terminated(void* matcher);
void  grammar_matcher_reset(void* matcher);
void  grammar_matcher_free(void* matcher);

#ifdef __cplusplus
}
#endif
```

**Step 8: Commit**

```bash
git add Sources/CXGrammar/include/
git commit -m "feat: add CXGrammar C wrapper header files"
```

---

### Task 3: Create CXGrammar C wrapper — implementation files

**Files:**
- Create: `Sources/CXGrammar/error_handler.cpp`
- Create: `Sources/CXGrammar/tokenizer_info.cpp`
- Create: `Sources/CXGrammar/grammar_compiler.cpp`
- Create: `Sources/CXGrammar/grammar_matcher.cpp`

**Step 1: Write error_handler.cpp**

Create `Sources/CXGrammar/error_handler.cpp`:

```cpp
#include "include/cxgrammar/error_handler.h"

static error_handler_closure _error_handler = nullptr;

extern "C" {

void set_error_handler(error_handler_closure handler) {
    _error_handler = handler;
}

void catch_error(const char* error_message) {
    if (_error_handler) {
        _error_handler(error_message);
    }
}

}
```

**Step 2: Write tokenizer_info.cpp**

Create `Sources/CXGrammar/tokenizer_info.cpp`:

```cpp
#include "include/cxgrammar/error_handler.h"
#include "include/cxgrammar/tokenizer_info.h"
#include <xgrammar/xgrammar.h>
#include <vector>
#include <string>

using namespace xgrammar;

extern "C" {

void* tokenizer_info_new(const char* const* vocab, size_t vocab_size,
                         const int vocab_type,
                         const int32_t* eos_tokens, size_t eos_tokens_size) {
    try {
        std::vector<std::string> vocab_vec(vocab, vocab + vocab_size);
        std::vector<int32_t> eos_vec(eos_tokens, eos_tokens + eos_tokens_size);
        auto* info = new TokenizerInfo(
            vocab_vec,
            static_cast<VocabType>(vocab_type),
            static_cast<int>(vocab_size),
            eos_vec
        );
        return static_cast<void*>(info);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void tokenizer_info_free(void* tokenizer_info) {
    delete static_cast<TokenizerInfo*>(tokenizer_info);
}

}
```

**Step 3: Write grammar_compiler.cpp**

Create `Sources/CXGrammar/grammar_compiler.cpp`:

```cpp
#include "include/cxgrammar/error_handler.h"
#include "include/cxgrammar/grammar_compiler.h"
#include <xgrammar/xgrammar.h>
#include <string>
#include <optional>

using namespace xgrammar;

extern "C" {

void* compile_json_schema_grammar(void* tokenizer_info,
                                  const char* schema, size_t length,
                                  int indent) {
    try {
        auto* info = static_cast<TokenizerInfo*>(tokenizer_info);
        GrammarCompiler compiler(*info);
        std::string schema_str(schema, length);
        std::optional<int> indent_opt = indent >= 0 ? std::optional<int>(indent) : std::nullopt;
        auto grammar = Grammar::FromJSONSchema(schema_str);
        auto compiled = compiler.CompileGrammar(grammar);
        return new CompiledGrammar(compiled);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void* compile_ebnf_grammar(void* tokenizer_info,
                            const char* grammar_str, size_t length) {
    try {
        auto* info = static_cast<TokenizerInfo*>(tokenizer_info);
        GrammarCompiler compiler(*info);
        std::string ebnf(grammar_str, length);
        auto grammar = Grammar::FromEBNF(ebnf);
        auto compiled = compiler.CompileGrammar(grammar);
        return new CompiledGrammar(compiled);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void* compile_regex_grammar(void* tokenizer_info,
                             const char* regex, size_t length) {
    try {
        auto* info = static_cast<TokenizerInfo*>(tokenizer_info);
        GrammarCompiler compiler(*info);
        std::string regex_str(regex, length);
        auto grammar = Grammar::FromRegex(regex_str);
        auto compiled = compiler.CompileGrammar(grammar);
        return new CompiledGrammar(compiled);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void compiled_grammar_free(void* compiled_grammar) {
    delete static_cast<CompiledGrammar*>(compiled_grammar);
}

}
```

**Step 4: Write grammar_matcher.cpp**

Create `Sources/CXGrammar/grammar_matcher.cpp`:

```cpp
#include "include/cxgrammar/error_handler.h"
#include "include/cxgrammar/grammar_matcher.h"
#include <xgrammar/xgrammar.h>
#include <dlpack/dlpack.h>

using namespace xgrammar;

extern "C" {

void* grammar_matcher_new(void* compiled_grammar) {
    try {
        auto* cg = static_cast<CompiledGrammar*>(compiled_grammar);
        return new GrammarMatcher(*cg);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return nullptr;
    }
}

void grammar_matcher_fill_next_token_bitmask(void* matcher, void* bitmask) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        auto* tensor = static_cast<DLTensor*>(bitmask);
        m->FillNextTokenBitmask(tensor);
    } catch (const std::exception& e) {
        catch_error(e.what());
    }
}

bool grammar_matcher_accept_token(void* matcher, int32_t token_id) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        return m->AcceptToken(token_id);
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

bool grammar_matcher_is_terminated(void* matcher) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        return m->IsTerminated();
    } catch (const std::exception& e) {
        catch_error(e.what());
        return false;
    }
}

void grammar_matcher_reset(void* matcher) {
    try {
        auto* m = static_cast<GrammarMatcher*>(matcher);
        m->Reset();
    } catch (const std::exception& e) {
        catch_error(e.what());
    }
}

void grammar_matcher_free(void* matcher) {
    delete static_cast<GrammarMatcher*>(matcher);
}

}
```

**Step 5: Commit**

```bash
git add Sources/CXGrammar/*.cpp
git commit -m "feat: add CXGrammar C wrapper implementation files"
```

---

### Task 4: Create xgrammar symlink and add CXGrammar SPM target

**Files:**
- Create: `Sources/CXGrammar/xgrammar` (symlink to `../../vendor/xgrammar`)
- Modify: `Package.swift`

SPM requires all sources to be under the target directory. A symlink bridges `vendor/xgrammar` into `Sources/CXGrammar/`.

**Step 1: Create symlink**

```bash
cd Sources/CXGrammar
ln -s ../../vendor/xgrammar xgrammar
cd ../..
ls -la Sources/CXGrammar/xgrammar/include/xgrammar/xgrammar.h
```

Expected: File exists (symlink resolves correctly).

**Step 2: Add CXGrammar target to Package.swift**

Add to the `targets` array in `Package.swift`, before the existing `executableTarget`:

```swift
.target(
    name: "CXGrammar",
    exclude: [
        "xgrammar/web",
        "xgrammar/tests",
        "xgrammar/python",
        "xgrammar/3rdparty/cpptrace",
        "xgrammar/3rdparty/googletest",
        "xgrammar/3rdparty/dlpack/contrib",
        "xgrammar/3rdparty/picojson",
        "xgrammar/cpp/nanobind",
    ],
    cSettings: [
        .headerSearchPath("xgrammar/include"),
        .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
        .headerSearchPath("xgrammar/3rdparty/picojson"),
    ],
    cxxSettings: [
        .headerSearchPath("xgrammar/include"),
        .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
        .headerSearchPath("xgrammar/3rdparty/picojson"),
    ]
),
```

Add `"CXGrammar"` to the `MacLocalAPI` target's `dependencies` array:

```swift
.executableTarget(
    name: "MacLocalAPI",
    dependencies: [
        "CXGrammar",    // <-- add this
        .product(name: "Vapor", package: "vapor"),
        // ... rest unchanged
    ],
```

Add C++ language standard to `Package.swift` (after `platforms`):

```swift
cxxLanguageStandard: .gnucxx17
```

**Step 3: Verify build compiles the CXGrammar target**

```bash
swift build 2>&1 | head -30
```

Expected: CXGrammar target compiles (may take 20-30s for xgrammar C++ sources). If there are exclude path errors, adjust the exclude list based on the actual xgrammar directory structure.

**Step 4: Commit**

```bash
git add Sources/CXGrammar/xgrammar Package.swift
git commit -m "feat: add CXGrammar SPM target with xgrammar C++ compilation"
```

---

### Task 5: Create XGrammarService.swift

**Files:**
- Create: `Sources/MacLocalAPI/Models/XGrammarService.swift`

This replaces `XGrammarBridge.swift`. It wraps the C functions from `CXGrammar` in a Swift class.

**Step 1: Write XGrammarService**

Create `Sources/MacLocalAPI/Models/XGrammarService.swift`:

```swift
import Foundation
import CXGrammar
import MLX
import Tokenizers

/// Native C++ XGrammar integration for constrained decoding.
/// Replaces the Python subprocess bridge (XGrammarBridge).
/// NOT thread-safe — callers must ensure serial access (e.g., within container.perform {}).
final class XGrammarService: @unchecked Sendable {

    /// Opaque pointer to xgrammar::TokenizerInfo
    private var tokenizerInfo: UnsafeMutableRawPointer?
    private let vocabSize: Int
    private let debugLogging: Bool

    /// Error message captured from C++ exceptions via the error handler callback.
    private static var lastError: String?
    private static let errorLock = NSLock()

    init(vocabSize: Int, debugLogging: Bool = false) {
        self.vocabSize = vocabSize
        self.debugLogging = debugLogging

        // Register error handler once
        set_error_handler { message in
            guard let message else { return }
            let str = String(cString: message)
            XGrammarService.errorLock.lock()
            XGrammarService.lastError = str
            XGrammarService.errorLock.unlock()
        }
    }

    deinit {
        if let info = tokenizerInfo {
            tokenizer_info_free(info)
        }
    }

    // MARK: - Error handling

    private static func consumeError() -> String? {
        errorLock.lock()
        let err = lastError
        lastError = nil
        errorLock.unlock()
        return err
    }

    // MARK: - Tokenizer setup

    /// Build TokenizerInfo from the model's tokenizer vocabulary.
    /// Must be called once before compileSchema.
    func setupTokenizer(tokenizer: any Tokenizer, eosTokenId: Int?) throws {
        guard tokenizerInfo == nil else { return } // already set up

        // Build vocab array: [String] indexed by token ID
        var vocabStrings: [String] = []
        vocabStrings.reserveCapacity(vocabSize)
        for id in 0..<vocabSize {
            vocabStrings.append(tokenizer.convertIdToToken(id) ?? "")
        }

        // EOS tokens
        var eosTokens: [Int32] = []
        if let eos = eosTokenId { eosTokens.append(Int32(eos)) }
        if let eos = tokenizer.eosTokenId, !eosTokens.contains(Int32(eos)) {
            eosTokens.append(Int32(eos))
        }

        // Create TokenizerInfo via C API
        // vocab_type: 0 = RAW (byte-level BPE, covers most models)
        let info = vocabStrings.withCStringArray { cStrings in
            eosTokens.withUnsafeBufferPointer { eosBuffer in
                tokenizer_info_new(
                    cStrings,
                    vocabStrings.count,
                    0, // VocabType::RAW
                    eosBuffer.baseAddress,
                    eosBuffer.count
                )
            }
        }

        guard let info else {
            let err = Self.consumeError() ?? "unknown error"
            throw XGrammarServiceError.initFailed(err)
        }

        self.tokenizerInfo = info
        if debugLogging {
            print("[XGrammar] TokenizerInfo created (vocab_size=\(vocabSize), eos=\(eosTokens))")
        }
    }

    // MARK: - Grammar compilation

    /// Compile a JSON schema into a grammar matcher.
    /// Returns an opaque matcher handle, or throws on failure.
    func compileAndCreateMatcher(schemaJSON: String) throws -> GrammarMatcherHandle {
        guard let info = tokenizerInfo else {
            throw XGrammarServiceError.notInitialized
        }

        let compiled = schemaJSON.withCString { cStr in
            compile_json_schema_grammar(info, cStr, schemaJSON.utf8.count, -1)
        }
        guard let compiled else {
            let err = Self.consumeError() ?? "unknown compilation error"
            throw XGrammarServiceError.compilationFailed(err)
        }

        let matcher = grammar_matcher_new(compiled)
        compiled_grammar_free(compiled)

        guard let matcher else {
            let err = Self.consumeError() ?? "unknown matcher error"
            throw XGrammarServiceError.matcherFailed(err)
        }

        return GrammarMatcherHandle(
            pointer: matcher,
            vocabSize: vocabSize,
            debugLogging: debugLogging
        )
    }
}

// MARK: - GrammarMatcherHandle

/// Wraps an xgrammar GrammarMatcher for per-token operations.
/// Synchronous — safe to call from the TokenIterator loop.
final class GrammarMatcherHandle: @unchecked Sendable {
    private var pointer: UnsafeMutableRawPointer?
    private let vocabSize: Int
    private let debugLogging: Bool

    /// Reusable bitmask buffer (Int32 array, sized for vocab)
    private var bitmaskBuffer: [Int32]
    private let bitmaskSize: Int

    init(pointer: UnsafeMutableRawPointer, vocabSize: Int, debugLogging: Bool) {
        self.pointer = pointer
        self.vocabSize = vocabSize
        self.debugLogging = debugLogging
        // Bitmask is a bitfield: ceil(vocabSize / 32) Int32 values
        self.bitmaskSize = (vocabSize + 31) / 32
        self.bitmaskBuffer = [Int32](repeating: 0, count: bitmaskSize)
    }

    deinit {
        if let p = pointer {
            grammar_matcher_free(p)
        }
    }

    /// Fill the bitmask with allowed tokens for the current grammar state.
    /// Returns an MLXArray mask: 0.0 for allowed tokens, -1e9 for disallowed.
    func nextTokenMask() -> MLXArray? {
        guard let p = pointer else { return nil }

        // Create a DLTensor pointing to our bitmask buffer
        bitmaskBuffer.withUnsafeMutableBufferPointer { buffer in
            var tensor = DLTensor()
            tensor.data = UnsafeMutableRawPointer(buffer.baseAddress!)
            tensor.ndim = 1
            tensor.dtype = DLDataType(code: UInt8(kDLInt), bits: 32, lanes: 1)
            var shape: Int64 = Int64(bitmaskSize)
            withUnsafeMutablePointer(to: &shape) { shapePtr in
                tensor.shape = shapePtr
                withUnsafeMutablePointer(to: &tensor) { tensorPtr in
                    grammar_matcher_fill_next_token_bitmask(p, tensorPtr)
                }
            }
        }

        // Convert bitmask to float mask array
        var maskFloats = [Float](repeating: -1e9, count: vocabSize)
        for i in 0..<vocabSize {
            let word = i / 32
            let bit = i % 32
            if (bitmaskBuffer[word] >> bit) & 1 == 1 {
                maskFloats[i] = 0.0
            }
        }

        return MLXArray(maskFloats)
    }

    /// Accept a sampled token, advancing grammar state.
    func acceptToken(_ tokenID: Int) {
        guard let p = pointer else { return }
        _ = grammar_matcher_accept_token(p, Int32(tokenID))
    }

    /// Check if the grammar has reached a terminal state.
    func isTerminated() -> Bool {
        guard let p = pointer else { return true }
        return grammar_matcher_is_terminated(p)
    }

    /// Release the matcher. Safe to call multiple times.
    func release() {
        if let p = pointer {
            grammar_matcher_free(p)
            pointer = nil
        }
    }
}

// MARK: - Errors

enum XGrammarServiceError: Error, LocalizedError {
    case notInitialized
    case initFailed(String)
    case compilationFailed(String)
    case matcherFailed(String)

    var errorDescription: String? {
        switch self {
        case .notInitialized: return "XGrammarService: tokenizer not initialized"
        case .initFailed(let msg): return "XGrammarService init failed: \(msg)"
        case .compilationFailed(let msg): return "XGrammar schema compilation failed: \(msg)"
        case .matcherFailed(let msg): return "XGrammar matcher creation failed: \(msg)"
        }
    }
}

// MARK: - Helper: [String] → [UnsafePointer<CChar>]

extension Array where Element == String {
    /// Temporarily expose array of Swift strings as C string array.
    func withCStringArray<R>(_ body: (UnsafePointer<UnsafePointer<CChar>?>) -> R) -> R {
        let cStrings = self.map { strdup($0) }
        defer { cStrings.forEach { free($0) } }
        return cStrings.withUnsafeBufferPointer { buffer in
            // Cast UnsafeMutablePointer<CChar>? to UnsafePointer<CChar>?
            buffer.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self, capacity: buffer.count) { ptr in
                body(ptr)
            }
        }
    }
}
```

**Step 2: Verify it compiles**

```bash
swift build 2>&1 | tail -5
```

Expected: Build Succeeded. If there are type mismatches with the DLTensor struct, check the actual struct layout in `vendor/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h` and adjust.

**Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Models/XGrammarService.swift
git commit -m "feat: add XGrammarService with native C++ grammar matching"
```

---

### Task 6: Update GrammarLogitProcessor to use bitmask

**Files:**
- Modify: `Scripts/patches/Evaluate.swift:377-411`

The processor currently stores `allowedTokens: [Int]?` (a list of allowed token IDs from the Python bridge). Change it to use `GrammarMatcherHandle` directly for bitmask-based masking.

**Step 1: Replace GrammarLogitProcessor**

In `Scripts/patches/Evaluate.swift`, replace lines 377-411 (the entire `GrammarLogitProcessor` class):

```swift
/// Logit processor that masks tokens based on grammar constraints.
/// Uses reference semantics (class) so the matcher handle can be shared.
public final class GrammarLogitProcessor: LogitProcessor, @unchecked Sendable {
    /// The grammar matcher handle. nil = no constraint (passthrough).
    public var matcherHandle: AnyObject?

    /// Callback invoked after each token is sampled, with the token ID.
    public var onTokenSampled: ((Int) -> Void)?

    /// Pre-computed mask from the matcher (set before each token generation).
    public var tokenMask: MLXArray?

    public init() {}

    public func prompt(_ prompt: MLXArray) {
        // No-op — grammar state managed externally
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard let mask = tokenMask else {
            return logits  // No constraint — passthrough
        }
        return logits + mask
    }

    public func didSample(token: MLXArray) {
        let tokenID = token.item(Int.self)
        onTokenSampled?(tokenID)
    }
}
```

**Step 2: Apply the patch**

```bash
./Scripts/apply-mlx-patches.sh
```

**Step 3: Verify build**

```bash
swift build 2>&1 | tail -5
```

Expected: Build Succeeded.

**Step 4: Commit**

```bash
git add Scripts/patches/Evaluate.swift
git commit -m "refactor: update GrammarLogitProcessor for bitmask-based masking"
```

---

### Task 7: Wire XGrammarService into MLXModelService

**Files:**
- Modify: `Sources/MacLocalAPI/Models/MLXModelService.swift`

Replace the Python bridge wiring with native C++ calls.

**Step 1: Replace property declarations**

Find and remove the `xgrammarBridge` property (around line 95):

```swift
// REMOVE:
private var xgrammarBridge: XGrammarBridge?
```

Replace with:

```swift
private var xgrammarService: XGrammarService?
```

**Step 2: Replace setupGrammarConstraint**

Replace the entire `setupGrammarConstraint` method (lines 1209-1270) with:

```swift
    /// Set up grammar-constrained decoding for a json_schema response format.
    /// Returns a GrammarLogitProcessor on success, nil on failure (falls back to prompt injection).
    private func setupGrammarConstraint(
        modelID: String,
        responseFormat: ResponseFormat?,
        tokenizer: any Tokenizer
    ) -> GrammarLogitProcessor? {
        guard let responseFormat, responseFormat.type == "json_schema",
              let schema = responseFormat.jsonSchema?.schema else {
            return nil
        }

        do {
            // Initialize service if needed
            if xgrammarService == nil {
                guard let directory = resolver.localModelDirectory(repoId: modelID) else {
                    if debugLogging { print("[XGrammar] Model directory not found") }
                    return nil
                }
                let vocabSize = readVocabSize(directory: directory) ?? 151936
                let service = XGrammarService(vocabSize: vocabSize, debugLogging: debugLogging)
                let eosId = tokenizer.eosTokenId
                try service.setupTokenizer(tokenizer: tokenizer, eosTokenId: eosId)
                self.xgrammarService = service
            }

            guard let service = xgrammarService else { return nil }

            // Convert schema to JSON string
            let schemaValue = schema.toSendable()
            let schemaData = try JSONSerialization.data(withJSONObject: schemaValue)
            let schemaJSON = String(data: schemaData, encoding: .utf8) ?? "{}"

            // Compile and create matcher
            let matcher = try service.compileAndCreateMatcher(schemaJSON: schemaJSON)

            // Create processor
            let proc = GrammarLogitProcessor()
            proc.matcherHandle = matcher
            // Pre-compute initial mask
            proc.tokenMask = matcher.nextTokenMask()
            // On each sampled token: accept it and update mask for next token
            proc.onTokenSampled = { [weak matcher] tokenID in
                matcher?.acceptToken(tokenID)
                proc.tokenMask = matcher?.nextTokenMask()
            }

            if debugLogging {
                print("[XGrammar] Grammar constraint active for json_schema (native C++, vocab_size=\(service.vocabSize))")
            }

            return proc
        } catch {
            if debugLogging {
                print("[XGrammar] Failed to set up grammar: \(error). Falling back to prompt injection.")
            }
            return nil
        }
    }
```

Note: The `vocabSize` property on `XGrammarService` needs to be made accessible. Add `let` access to the property (it's already `let` in the class definition from Task 5).

**Step 3: Update call sites in generate() and generateStreaming()**

The call sites currently look like:

```swift
let grammarResult = await setupGrammarConstraint(modelID: modelID, responseFormat: responseFormat)
let grammarSession = grammarResult?.1
defer {
    if let session = grammarSession {
        session.release()
        Task { await xgrammarBridge?.releaseSession() }
    }
}
// ... later ...
parameters.extraProcessor = grammarResult?.0
```

Replace with (in both `generate()` and `generateStreaming()`):

```swift
let grammarProcessor = setupGrammarConstraint(
    modelID: modelID,
    responseFormat: responseFormat,
    tokenizer: context.tokenizer
)
defer {
    if let handle = grammarProcessor?.matcherHandle as? GrammarMatcherHandle {
        handle.release()
    }
}
// ... later ...
parameters.extraProcessor = grammarProcessor
```

Note: The method is no longer `async` — it calls C functions directly, no subprocess startup.

**Step 4: Remove XGrammarBridge import/usage**

Search for any remaining references to `XGrammarBridge` or `XGrammarSession` in `MLXModelService.swift` and remove them.

**Step 5: Verify build**

```bash
swift build 2>&1 | tail -5
```

Expected: Build Succeeded.

**Step 6: Commit**

```bash
git add Sources/MacLocalAPI/Models/MLXModelService.swift
git commit -m "feat: wire XGrammarService into generation pipeline (replaces Python bridge)"
```

---

### Task 8: Delete Python bridge code

**Files:**
- Delete: `Sources/MacLocalAPI/Models/XGrammarBridge.swift`
- Delete: `Scripts/xgrammar-bridge.py`

**Step 1: Remove files**

```bash
git rm Sources/MacLocalAPI/Models/XGrammarBridge.swift
git rm Scripts/xgrammar-bridge.py
```

**Step 2: Verify no remaining references**

```bash
grep -r "XGrammarBridge\|xgrammar-bridge\|XGrammarSession" Sources/ Scripts/ --include="*.swift" --include="*.py" --include="*.sh"
```

Expected: No matches (or only in design docs/plans).

**Step 3: Verify build**

```bash
swift build 2>&1 | tail -5
```

Expected: Build Succeeded.

**Step 4: Commit**

```bash
git commit -m "cleanup: remove XGrammar Python subprocess bridge"
```

---

### Task 9: Update build-from-scratch.sh

**Files:**
- Modify: `Scripts/build-from-scratch.sh`

Ensure the xgrammar submodule is initialized during full builds.

**Step 1: Check current submodule init**

```bash
grep -n "submodule" Scripts/build-from-scratch.sh
```

The script likely already has `git submodule update --init --recursive`. If so, `vendor/xgrammar` will be initialized automatically. If not, add it.

**Step 2: Verify**

```bash
bash -n Scripts/build-from-scratch.sh
```

Expected: No syntax errors.

**Step 3: Commit (if changes needed)**

```bash
git add Scripts/build-from-scratch.sh
git commit -m "build: ensure xgrammar submodule initialized in build-from-scratch"
```

---

### Task 10: End-to-end test

**Files:** None (verification only)

**Step 1: Clean build**

```bash
swift build -c release 2>&1 | tail -5
```

Expected: Build Succeeded.

**Step 2: Start server**

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx -m mlx-community/Qwen3-0.6B-4bit --port 9998 &
until curl -sf http://127.0.0.1:9998/v1/models >/dev/null 2>&1; do sleep 1; done
```

**Step 3: Test structured output with json_schema**

```bash
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "Give me a person named Alice who is 30 years old"}],
    "max_tokens": 100,
    "temperature": 0,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }' | python3 -c "
import sys, json
resp = json.load(sys.stdin)
content = resp['choices'][0]['message']['content']
parsed = json.loads(content)
assert 'name' in parsed and isinstance(parsed['name'], str), f'bad name: {parsed}'
assert 'age' in parsed and isinstance(parsed['age'], int), f'bad age: {parsed}'
print(f'PASS: {parsed}')
"
```

Expected: `PASS: {'name': 'Alice', 'age': 30}` (or similar valid JSON).

**Step 4: Test --guided-json CLI flag**

```bash
kill %1
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx -m mlx-community/Qwen3-0.6B-4bit --port 9998 \
  --guided-json '{"type":"object","properties":{"color":{"type":"string"}},"required":["color"]}' &
until curl -sf http://127.0.0.1:9998/v1/models >/dev/null 2>&1; do sleep 1; done
curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-0.6B-4bit","messages":[{"role":"user","content":"What is your favorite color?"}],"max_tokens":50,"temperature":0}' \
  | python3 -c "
import sys, json
resp = json.load(sys.stdin)
content = resp['choices'][0]['message']['content']
parsed = json.loads(content)
assert 'color' in parsed, f'missing color: {parsed}'
print(f'PASS: {parsed}')
"
```

Expected: `PASS: {'color': '...'}` (valid JSON with color field).

**Step 5: Run existing test suite**

```bash
kill %1
# Restart without --guided-json
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  .build/release/afm mlx -m mlx-community/Qwen3-0.6B-4bit --port 9998 &
until curl -sf http://127.0.0.1:9998/v1/models >/dev/null 2>&1; do sleep 1; done
./Scripts/test-assertions.sh --tier standard --model mlx-community/Qwen3-0.6B-4bit --port 9998
```

Expected: All tests pass, including Section 8c (Structured Output).

**Step 6: Stop server**

```bash
kill %1
```

**Step 7: Commit any fixes needed**

Only if adjustments were required during testing.

---

## Build Troubleshooting

### Common issues and fixes:

1. **SPM can't find xgrammar sources**: Check the symlink `Sources/CXGrammar/xgrammar` resolves correctly. Run `ls Sources/CXGrammar/xgrammar/cpp/` — should list `.cc` files.

2. **Exclude path doesn't exist**: SPM will error if an exclude path doesn't exist in the source tree. Check the actual xgrammar directory structure and adjust excludes in `Package.swift`.

3. **DLTensor struct mismatch**: The `DLTensor` struct layout may differ between xgrammar versions. Check `vendor/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h` for the actual struct definition and adjust `XGrammarService.swift`'s `nextTokenMask()` accordingly.

4. **Vocab type mismatch**: If the model's tokenizer doesn't use byte-level BPE (vocab_type=0), try `vocab_type=2` (BYTE_FALLBACK) or check xgrammar's `VocabType` enum.

5. **Linker errors**: If xgrammar uses C++ standard library features, you may need to add `linkerSettings: [.linkedLibrary("c++")]` to the CXGrammar target.
