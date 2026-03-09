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
    let vocabSize: Int
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

    /// Public accessor for consuming the last error from C++ exceptions.
    /// Used by diagnostic logging in the grammar callback.
    static func consumeLastError() -> String? {
        return consumeError()
    }

    private static func consumeError() -> String? {
        errorLock.lock()
        let err = lastError
        lastError = nil
        errorLock.unlock()
        return err
    }

    // MARK: - Tokenizer setup

    /// Build TokenizerInfo from the model's tokenizer vocabulary.
    /// Must be called once before compileAndCreateMatcher.
    func setupTokenizer(tokenizer: any Tokenizer, eosTokenId: Int?) throws {
        guard tokenizerInfo == nil else { return }

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
        // vocab_type: 2 = BYTE_LEVEL (byte-level BPE, covers most HF models)
        let info = vocabStrings.withCStringArray { cStrings in
            eosTokens.withUnsafeBufferPointer { eosBuffer in
                tokenizer_info_new(
                    cStrings,
                    vocabStrings.count,
                    2, // VocabType::BYTE_LEVEL
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

    /// Compile a JSON schema and create a grammar matcher.
    /// Returns a GrammarMatcherHandle for per-token operations.
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
        // Keep compiled grammar alive — matcher references it internally.
        // We free it after the matcher is created since xgrammar copies what it needs.
        compiled_grammar_free(compiled)

        guard let matcher else {
            let err = Self.consumeError() ?? "unknown matcher error"
            throw XGrammarServiceError.matcherFailed(err)
        }

        let bitmaskSize = Int(grammar_bitmask_size(Int32(vocabSize)))

        return GrammarMatcherHandle(
            pointer: matcher,
            vocabSize: vocabSize,
            bitmaskSize: bitmaskSize,
            debugLogging: debugLogging
        )
    }

    /// Compile an EBNF grammar and create a matcher.
    func compileAndCreateMatcherFromEBNF(grammar: String) throws -> GrammarMatcherHandle {
        guard let info = tokenizerInfo else {
            throw XGrammarServiceError.notInitialized
        }

        let compiled = grammar.withCString { cStr in
            compile_ebnf_grammar(info, cStr, grammar.utf8.count)
        }
        guard let compiled else {
            let err = Self.consumeError() ?? "unknown EBNF compilation error"
            throw XGrammarServiceError.compilationFailed(err)
        }

        let matcher = grammar_matcher_new(compiled)
        compiled_grammar_free(compiled)

        guard let matcher else {
            let err = Self.consumeError() ?? "unknown matcher error"
            throw XGrammarServiceError.matcherFailed(err)
        }

        let bitmaskSize = Int(grammar_bitmask_size(Int32(vocabSize)))
        return GrammarMatcherHandle(
            pointer: matcher,
            vocabSize: vocabSize,
            bitmaskSize: bitmaskSize,
            debugLogging: debugLogging
        )
    }

    /// Compile a structural tag JSON spec and create a matcher.
    /// Uses xgrammar's TagDispatch which allows all tokens until a trigger
    /// string is detected, then constrains output to the tag schema.
    func compileAndCreateMatcherFromStructuralTag(json: String) throws -> GrammarMatcherHandle {
        guard let info = tokenizerInfo else {
            throw XGrammarServiceError.notInitialized
        }

        let compiled = json.withCString { cStr in
            compile_structural_tag(info, cStr, json.utf8.count)
        }
        guard let compiled else {
            let err = Self.consumeError() ?? "unknown structural tag compilation error"
            throw XGrammarServiceError.compilationFailed(err)
        }

        let matcher = grammar_matcher_new(compiled)
        compiled_grammar_free(compiled)

        guard let matcher else {
            let err = Self.consumeError() ?? "unknown matcher error"
            throw XGrammarServiceError.matcherFailed(err)
        }

        let bitmaskSize = Int(grammar_bitmask_size(Int32(vocabSize)))
        return GrammarMatcherHandle(
            pointer: matcher,
            vocabSize: vocabSize,
            bitmaskSize: bitmaskSize,
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

    init(pointer: UnsafeMutableRawPointer, vocabSize: Int, bitmaskSize: Int, debugLogging: Bool) {
        self.pointer = pointer
        self.vocabSize = vocabSize
        self.debugLogging = debugLogging
        self.bitmaskSize = bitmaskSize
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

        // Zero the buffer and fill via C API
        bitmaskBuffer.withUnsafeMutableBufferPointer { buffer in
            buffer.baseAddress!.update(repeating: 0, count: bitmaskSize)
        }

        let needsMask = bitmaskBuffer.withUnsafeMutableBufferPointer { buffer in
            grammar_matcher_get_bitmask(p, buffer.baseAddress!, Int32(bitmaskSize))
        }

        // If bitmask is all-true, no masking needed
        if !needsMask { return nil }

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
    /// Returns true if the token was accepted, false if rejected or on error.
    @discardableResult
    func acceptToken(_ tokenID: Int) -> Bool {
        guard let p = pointer else { return false }
        return grammar_matcher_accept_token(p, Int32(tokenID))
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

// MARK: - Helper: [String] → C string array

extension Array where Element == String {
    /// Temporarily expose array of Swift strings as a C string array (const char* const*).
    func withCStringArray<R>(_ body: (UnsafePointer<UnsafePointer<CChar>?>) -> R) -> R {
        let cStrings = self.map { strdup($0) }
        defer { cStrings.forEach { free($0) } }
        return cStrings.withUnsafeBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self, capacity: buffer.count) { ptr in
                body(ptr)
            }
        }
    }
}
