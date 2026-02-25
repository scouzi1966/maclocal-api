// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

struct TokenizerError: Error {
    let message: String
}

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
        configuration: configuration, hub: hub)

    return try PreTrainedTokenizer(
        tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

public func loadTokenizerConfig(configuration: ModelConfiguration, hub: HubApi) async throws -> (
    Config, Config
) {
    // Convert tiktoken.model → tokenizer.json if needed BEFORE creating
    // LanguageModelConfigurationFromHub, which eagerly checks for tokenizer.json
    let modelDir: URL
    switch configuration.id {
    case .id:
        modelDir = configuration.modelDirectory(hub: hub)
    case .directory(let directory):
        modelDir = directory
    }
    convertTiktokenIfNeeded(modelDirectory: modelDir)

    // from AutoTokenizer.from() -- this lets us override parts of the configuration
    let config: LanguageModelConfigurationFromHub

    switch configuration.id {
    case .id(let id, let revision):
        do {
            // the load can fail (async when we try to use it)
            let loaded = LanguageModelConfigurationFromHub(
                modelName: configuration.tokenizerId ?? id, revision: revision, hubApi: hub)
            _ = try await loaded.tokenizerConfig
            config = loaded
        } catch {
            let nserror = error as NSError
            if nserror.domain == NSURLErrorDomain
                && nserror.code == NSURLErrorNotConnectedToInternet
            {
                // Internet connection appears to be offline -- fall back to loading from
                // the local directory
                config = LanguageModelConfigurationFromHub(
                    modelFolder: configuration.modelDirectory(hub: hub), hubApi: hub)
            } else {
                throw error
            }
        }
    case .directory(let directory):
        config = LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
    }

    guard var tokenizerConfig = try await config.tokenizerConfig else {
        throw TokenizerError(message: "missing config")
    }
    let tokenizerData = try await config.tokenizerData

    tokenizerConfig = updateTokenizerConfig(tokenizerConfig)

    // If chat_template is missing from tokenizer_config.json, try loading from
    // chat_template.jinja file (newer HF convention used by Qwen3.5, etc.)
    // The file may already be in the model directory, or we may need to download it.
    if tokenizerConfig.chatTemplate == nil || tokenizerConfig.chatTemplate?.string() == nil {
        let jinjaURL = modelDir.appending(path: "chat_template.jinja")
        var jinjaContent: String? = nil

        // Try local file first
        if FileManager.default.fileExists(atPath: jinjaURL.path) {
            jinjaContent = try? String(contentsOf: jinjaURL, encoding: .utf8)
        }

        // If not found locally, try downloading from HF
        if jinjaContent == nil {
            var repoId: String? = nil
            switch configuration.id {
            case .id(let id, _):
                repoId = configuration.tokenizerId ?? id
            case .directory(let directory):
                // Infer repo ID from directory path (e.g. .../mlx-community/ModelName → mlx-community/ModelName)
                let parent = directory.deletingLastPathComponent().lastPathComponent
                let name = directory.lastPathComponent
                if !parent.isEmpty && parent != "/" {
                    repoId = "\(parent)/\(name)"
                }
            }
            if let repoId {
                let repo = Hub.Repo(id: repoId)
                if let downloaded = try? await hub.snapshot(
                    from: repo, matching: "chat_template.jinja")
                {
                    let downloadedJinja = downloaded.appending(path: "chat_template.jinja")
                    jinjaContent = try? String(contentsOf: downloadedJinja, encoding: .utf8)
                    // Also copy to model directory for future loads
                    if let content = jinjaContent {
                        try? content.write(to: jinjaURL, atomically: true, encoding: .utf8)
                    }
                }
            }
        }

        if let jinjaContent, !jinjaContent.isEmpty,
           var dictionary = tokenizerConfig.dictionary()
        {
            dictionary["chat_template"] = .init(jinjaContent)
            tokenizerConfig = Config(dictionary)
        }
    }

    return (tokenizerConfig, tokenizerData)
}

private func updateTokenizerConfig(_ tokenizerConfig: Config) -> Config {
    // Workaround: replacement tokenizers for unhandled values in swift-transformers
    if let tokenizerClass = tokenizerConfig.tokenizerClass?.string(),
        let replacement = replacementTokenizers[tokenizerClass]
    {
        if var dictionary = tokenizerConfig.dictionary() {
            dictionary["tokenizer_class"] = .init(replacement)
            return Config(dictionary)
        }
    }
    return tokenizerConfig
}

public class TokenizerReplacementRegistry: @unchecked Sendable {

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention. this allows the methods
    // to remain synchronous.
    private let lock = NSLock()

    /// overrides for TokenizerModel/knownTokenizers
    private var replacementTokenizers = [
        "InternLM2Tokenizer": "PreTrainedTokenizer",
        "Qwen2Tokenizer": "PreTrainedTokenizer",
        "Qwen3Tokenizer": "PreTrainedTokenizer",
        "CohereTokenizer": "PreTrainedTokenizer",
        "GPTNeoXTokenizer": "PreTrainedTokenizer",
        "TokenizersBackend": "PreTrainedTokenizer",
        "TikTokenTokenizer": "PreTrainedTokenizer",
    ]

    public subscript(key: String) -> String? {
        get {
            lock.withLock {
                replacementTokenizers[key]
            }
        }
        set {
            lock.withLock {
                replacementTokenizers[key] = newValue
            }
        }
    }
}

public let replacementTokenizers = TokenizerReplacementRegistry()

public protocol StreamingDetokenizer: IteratorProtocol<String> {

    mutating func append(token: Int)

}

public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating public func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokens: segmentTokens)
        } else {
            segment = ""
        }
    }

    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokens: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }

}

// MARK: - TikToken → tokenizer.json conversion

/// Check if the model directory has tiktoken.model but no tokenizer.json,
/// and auto-convert if needed. This supports models like mlx-community/Kimi-K2.5-3bit
/// that ship with a TikTokenTokenizer instead of a standard HuggingFace tokenizer.
private func convertTiktokenIfNeeded(modelDirectory: URL) {
    let tokenizerJsonURL = modelDirectory.appending(path: "tokenizer.json")
    if FileManager.default.fileExists(atPath: tokenizerJsonURL.path) {
        return  // Already exists
    }

    let tiktokenModelURL = modelDirectory.appending(path: "tiktoken.model")
    guard FileManager.default.fileExists(atPath: tiktokenModelURL.path) else {
        return  // No tiktoken.model either
    }

    // Read tokenizer_config.json for special tokens
    let configURL = modelDirectory.appending(path: "tokenizer_config.json")

    do {
        try tiktokenToTokenizerJson(
            tiktokenURL: tiktokenModelURL, configURL: configURL,
            outputURL: tokenizerJsonURL)
    } catch {
        // Non-fatal: if conversion fails, the normal error path will handle it
        print("[MLXLMCommon] tiktoken conversion failed: \(error)")
    }
}

/// Convert a tiktoken.model file to HuggingFace tokenizer.json format.
///
/// tiktoken.model format: each line is "base64_encoded_token rank\n"
/// The output is a standard HuggingFace BPE tokenizer.json with byte-level encoding.
private func tiktokenToTokenizerJson(
    tiktokenURL: URL, configURL: URL, outputURL: URL
) throws {
    // 1. Read tiktoken.model: base64_token rank
    let content = try String(contentsOf: tiktokenURL, encoding: .utf8)
    let lines = content.split(separator: "\n", omittingEmptySubsequences: true)

    struct TokenEntry {
        let bytes: [UInt8]
        let rank: Int
    }

    var entries: [TokenEntry] = []
    entries.reserveCapacity(lines.count)

    for line in lines {
        let parts = line.split(separator: " ")
        guard parts.count == 2,
            let data = Data(base64Encoded: String(parts[0])),
            let rank = Int(parts[1])
        else { continue }
        entries.append(TokenEntry(bytes: Array(data), rank: rank))
    }

    // Sort by rank (should already be sorted, but ensure)
    entries.sort { $0.rank < $1.rank }

    // 2. Build byte-to-unicode mapping (HuggingFace byte-level BPE standard)
    let byteToUnicode = buildByteToUnicode()

    func bytesToUnicodeStr(_ bytes: [UInt8]) -> String {
        String(bytes.map { byteToUnicode[Int($0)]! })
    }

    // 3. Build vocab dict: unicode_str → rank
    var vocabPairs: [(String, Int)] = []
    vocabPairs.reserveCapacity(entries.count)
    for entry in entries {
        vocabPairs.append((bytesToUnicodeStr(entry.bytes), entry.rank))
    }

    // 4. Reconstruct BPE merges from rank ordering
    // Single-byte tokens are the base vocabulary.
    // Multi-byte tokens are formed by merging two existing tokens.
    var existingTokens = Set<[UInt8]>()
    var merges: [(String, String)] = []
    merges.reserveCapacity(entries.count)

    for entry in entries {
        if entry.bytes.count == 1 {
            existingTokens.insert(entry.bytes)
            continue
        }

        // Find valid split: both halves must already exist
        var found = false
        for i in 1 ..< entry.bytes.count {
            let left = Array(entry.bytes[..<i])
            let right = Array(entry.bytes[i...])
            if existingTokens.contains(left) && existingTokens.contains(right) {
                merges.append((bytesToUnicodeStr(left), bytesToUnicodeStr(right)))
                found = true
                break
            }
        }
        if !found {
            // Try from right (longer left part)
            for i in stride(from: entry.bytes.count - 1, through: 1, by: -1) {
                let left = Array(entry.bytes[..<i])
                let right = Array(entry.bytes[i...])
                if existingTokens.contains(left) && existingTokens.contains(right) {
                    merges.append((bytesToUnicodeStr(left), bytesToUnicodeStr(right)))
                    found = true
                    break
                }
            }
        }
        existingTokens.insert(entry.bytes)
    }

    // 5. Read special tokens from tokenizer_config.json
    var addedTokensList: [[String: Any]] = []
    if FileManager.default.fileExists(atPath: configURL.path),
        let configData = try? Data(contentsOf: configURL),
        let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
        let addedTokens = config["added_tokens_decoder"] as? [String: [String: Any]]
    {
        let sorted = addedTokens.sorted { Int($0.key)! < Int($1.key)! }
        for (idStr, info) in sorted {
            guard let id = Int(idStr), let content = info["content"] as? String else { continue }
            addedTokensList.append([
                "id": id,
                "content": content,
                "single_word": info["single_word"] as? Bool ?? false,
                "lstrip": info["lstrip"] as? Bool ?? false,
                "rstrip": info["rstrip"] as? Bool ?? false,
                "normalized": info["normalized"] as? Bool ?? false,
                "special": info["special"] as? Bool ?? true,
            ])
        }
    }

    // 6. Build tokenizer.json
    let vocabDict = Dictionary(uniqueKeysWithValues: vocabPairs)
    let mergeStrings = merges.map { "\($0.0) \($0.1)" }

    let tokenizerJson: [String: Any] = [
        "version": "1.0",
        "added_tokens": addedTokensList,
        "pre_tokenizer": [
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true,
        ] as [String: Any],
        "decoder": [
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        ] as [String: Any],
        "model": [
            "type": "BPE",
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocabDict,
            "merges": mergeStrings,
        ] as [String: Any],
    ]

    let jsonData = try JSONSerialization.data(
        withJSONObject: tokenizerJson, options: [.sortedKeys])
    try jsonData.write(to: outputURL)
}

/// Build the byte-to-unicode mapping used by HuggingFace byte-level BPE.
/// Maps each byte (0-255) to a unicode character, keeping printable ASCII as-is
/// and mapping control/special bytes to characters above U+0100.
private func buildByteToUnicode() -> [Int: Character] {
    // Printable byte ranges that map to themselves
    var bs: [Int] = []
    bs += Array(Int(UInt8(ascii: "!"))...Int(UInt8(ascii: "~")))  // 33-126
    bs += Array(0xA1...0xAC)  // ¡-¬
    bs += Array(0xAE...0xFF)  // ®-ÿ

    var cs = bs
    var n = 0
    for b in 0..<256 {
        if !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
    }

    var result: [Int: Character] = [:]
    for (b, c) in zip(bs, cs) {
        result[b] = Character(UnicodeScalar(c)!)
    }
    return result
}
