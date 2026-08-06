import Foundation

/// Rewrites safetensor shards so every tensor payload begins on a 32-byte
/// boundary. Fixed-schedule Metal executors consume GGUF-aligned tensor ABIs;
/// ordinary safetensor JSON headers do not guarantee that alignment.
public enum AlignedSafetensorRewriter {
    public typealias ProgressHandler = (String) -> Void

    private static let tensorAlignment = 32
    private static let headerAlignment = 4_096
    private static let formatVersion = 11
    private static let executorLayoutVersion = 3

    public static func rewriteCheckpoint(
        at checkpointURL: URL,
        progress: ProgressHandler? = nil
    ) throws {
        let root = checkpointURL.standardizedFileURL
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        var index = try jsonObject(at: indexURL)
        guard let weightMap = index["weight_map"] as? [String: Any] else {
            throw RewriteError.invalidFile("Safetensor index has no weight_map.")
        }
        let shardNames = Set(try weightMap.values.map { value -> String in
            guard let name = value as? String,
                  URL(fileURLWithPath: name).lastPathComponent == name else {
                throw RewriteError.invalidFile("Safetensor index contains an invalid shard name.")
            }
            return name
        }).sorted()

        for (index, shardName) in shardNames.enumerated() {
            let shardURL = root.appendingPathComponent(shardName)
            if try isExecutorAligned(shardURL) {
                progress?("[\(index + 1)/\(shardNames.count)] \(shardName): already aligned")
            } else {
                progress?("[\(index + 1)/\(shardNames.count)] \(shardName): aligning")
                try rewriteFileInPlace(shardURL)
            }
        }

        var totalSize: Int64 = 0
        for shardName in shardNames {
            totalSize += Int64(try fileSize(root.appendingPathComponent(shardName)))
        }
        var metadata = index["metadata"] as? [String: Any] ?? [:]
        metadata["total_size"] = totalSize
        index["metadata"] = metadata
        try writeJSON(index, to: indexURL)

        let configURL = root.appendingPathComponent("config.json")
        var config = try jsonObject(at: configURL)
        config["afm_dwarfstar_executor_layout_version"] = executorLayoutVersion
        config["afm_dwarfstar_tensor_alignment"] = tensorAlignment
        try writeJSON(config, to: configURL)

        let stateURL = root.appendingPathComponent(".afm-mlx-conversion.json")
        if FileManager.default.fileExists(atPath: stateURL.path) {
            var state = try jsonObject(at: stateURL)
            state["formatVersion"] = formatVersion
            if var completed = state["completed"] as? [String: Any] {
                for shardName in shardNames {
                    guard var item = completed[shardName] as? [String: Any] else { continue }
                    item["outputSize"] = Int64(try fileSize(root.appendingPathComponent(shardName)))
                    completed[shardName] = item
                }
                state["completed"] = completed
            }
            try writeJSON(state, to: stateURL)
        }
        progress?("Executor checkpoint alignment complete: \(root.path)")
    }

    public static func rewriteFileInPlace(_ sourceURL: URL) throws {
        let parsed = try ParsedSafetensor(url: sourceURL)
        let temporaryURL = sourceURL.deletingLastPathComponent()
            .appendingPathComponent(".\(sourceURL.lastPathComponent).aligned-partial")
        try? FileManager.default.removeItem(at: temporaryURL)

        var header: [String: Any] = [:]
        if let metadata = parsed.metadata { header["__metadata__"] = metadata }
        var cursor = 0
        var paddingIndex = 0
        for tensor in parsed.tensors {
            let aligned = align(cursor, to: tensorAlignment)
            if aligned > cursor {
                let count = aligned - cursor
                header[String(format: "__afm_padding_%06d", paddingIndex)] = [
                    "dtype": "U8",
                    "shape": [count],
                    "data_offsets": [cursor, aligned],
                ]
                paddingIndex += 1
                cursor = aligned
            }
            var entry = tensor.metadata
            entry["data_offsets"] = [cursor, cursor + tensor.byteCount]
            header[tensor.name] = entry
            cursor += tensor.byteCount
        }

        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let paddedHeaderSize = align(8 + json.count, to: headerAlignment) - 8
        guard paddedHeaderSize >= json.count else {
            throw RewriteError.invalidFile("Cannot align safetensor header.")
        }

        FileManager.default.createFile(atPath: temporaryURL.path, contents: nil)
        let input = try FileHandle(forReadingFrom: sourceURL)
        let output = try FileHandle(forWritingTo: temporaryURL)
        do {
            var littleEndianSize = UInt64(paddedHeaderSize).littleEndian
            try withUnsafeBytes(of: &littleEndianSize) { bytes in
                try output.write(contentsOf: Data(bytes))
            }
            try output.write(contentsOf: json)
            try output.write(contentsOf: Data(repeating: 0x20, count: paddedHeaderSize - json.count))

            var outputCursor = 0
            for tensor in parsed.tensors {
                let aligned = align(outputCursor, to: tensorAlignment)
                if aligned > outputCursor {
                    try output.write(contentsOf: Data(repeating: 0, count: aligned - outputCursor))
                    outputCursor = aligned
                }
                try input.seek(toOffset: UInt64(parsed.payloadStart + tensor.sourceOffset))
                try copy(byteCount: tensor.byteCount, from: input, to: output)
                outputCursor += tensor.byteCount
            }
            try output.synchronize()
            try input.close()
            try output.close()
        } catch {
            try? input.close()
            try? output.close()
            try? FileManager.default.removeItem(at: temporaryURL)
            throw error
        }

        _ = try FileManager.default.replaceItemAt(sourceURL, withItemAt: temporaryURL)
        guard try isExecutorAligned(sourceURL) else {
            throw RewriteError.invalidFile("Aligned safetensor verification failed for \(sourceURL.path).")
        }
    }

    public static func isExecutorAligned(_ url: URL) throws -> Bool {
        let parsed = try ParsedSafetensor(url: url)
        guard parsed.payloadStart.isMultiple(of: headerAlignment) else { return false }
        return parsed.tensors.allSatisfy {
            (parsed.payloadStart + $0.sourceOffset).isMultiple(of: tensorAlignment)
        }
    }

    private static func copy(
        byteCount: Int,
        from input: FileHandle,
        to output: FileHandle
    ) throws {
        var remaining = byteCount
        let chunkSize = 16 * 1_024 * 1_024
        while remaining > 0 {
            let requested = min(remaining, chunkSize)
            guard let data = try input.read(upToCount: requested), data.count == requested else {
                throw RewriteError.invalidFile("Unexpected end of safetensor payload.")
            }
            try output.write(contentsOf: data)
            remaining -= requested
        }
    }

    private static func align(_ value: Int, to alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + alignment - remainder
    }

    private static func jsonObject(at url: URL) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any] else {
            throw RewriteError.invalidFile("Expected a JSON object at \(url.path).")
        }
        return object
    }

    private static func writeJSON(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).partial")
        try data.write(to: temporary, options: .atomic)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
    }

    private static func fileSize(_ url: URL) throws -> Int {
        guard let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize else {
            throw RewriteError.invalidFile("Cannot stat \(url.path).")
        }
        return size
    }
}

private extension AlignedSafetensorRewriter {
    struct Tensor {
        let name: String
        let metadata: [String: Any]
        let sourceOffset: Int
        let byteCount: Int
    }

    struct ParsedSafetensor {
        let payloadStart: Int
        let metadata: Any?
        let tensors: [Tensor]

        init(url: URL) throws {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                throw RewriteError.invalidFile("Truncated safetensor header in \(url.path).")
            }
            let headerSize = prefix.enumerated().reduce(UInt64(0)) { result, item in
                result | (UInt64(item.element) << UInt64(item.offset * 8))
            }
            guard headerSize <= UInt64(Int.max),
                  let data = try handle.read(upToCount: Int(headerSize)),
                  data.count == Int(headerSize),
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                throw RewriteError.invalidFile("Invalid safetensor header in \(url.path).")
            }
            var parsed: [Tensor] = []
            for (name, value) in object where name != "__metadata__" {
                guard let entry = value as? [String: Any],
                      let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                      let start = Self.integer(offsets[0]),
                      let end = Self.integer(offsets[1]),
                      start >= 0, end >= start else {
                    throw RewriteError.invalidFile("Invalid tensor entry \(name) in \(url.path).")
                }
                if name.hasPrefix("__afm_padding_") { continue }
                parsed.append(Tensor(
                    name: name,
                    metadata: entry,
                    sourceOffset: start,
                    byteCount: end - start))
            }
            parsed.sort { $0.sourceOffset < $1.sourceOffset }
            payloadStart = 8 + Int(headerSize)
            metadata = object["__metadata__"]
            tensors = parsed
        }

        private static func integer(_ value: Any) -> Int? {
            if let value = value as? Int { return value }
            if let value = value as? NSNumber { return value.intValue }
            return nil
        }
    }

    enum RewriteError: LocalizedError {
        case invalidFile(String)
        var errorDescription: String? {
            guard case .invalidFile(let message) = self else { return nil }
            return message
        }
    }
}
