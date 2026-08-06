import Foundation

/// Header-only view of an AFM DeepSeek V4 safetensor checkpoint.
///
/// The catalog retains file-backed tensor locations so a fixed-schedule runtime
/// can map shard payloads directly without first materializing a second model.
public struct AFMDwarfStarCheckpointCatalog: Sendable {
    public static let bundledTemplateFilename = "dwarfstar-template.gguf"

    private struct SafetensorHeader {
        let tensors: [String: Any]
        let payloadStart: UInt64
        let fileSize: UInt64
    }

    public struct Layout: Sendable, Equatable {
        public let isAFMNative: Bool
        public let usesQ80DenseWeights: Bool
        public let usesDwarfStarMXFP4Experts: Bool
        public let usesPackedDwarfStarMXFP4Experts: Bool
        public let executorLayoutVersion: Int?

        public var isExecutorReady: Bool {
            isAFMNative
                && usesQ80DenseWeights
                && usesDwarfStarMXFP4Experts
                && usesPackedDwarfStarMXFP4Experts
                && (executorLayoutVersion ?? 0) >= 3
        }
    }

    public struct TensorLocation: Sendable, Equatable {
        public let name: String
        public let shardPath: String
        public let fileOffset: UInt64
        public let byteCount: UInt64
        public let dtype: String
        public let shape: [Int]
    }

    public enum CatalogError: LocalizedError, Equatable {
        case invalidCheckpoint(String)
        case unsupportedLayout(String)

        public var errorDescription: String? {
            switch self {
            case .invalidCheckpoint(let message), .unsupportedLayout(let message):
                return message
            }
        }
    }

    public let checkpointPath: String
    public let layout: Layout
    public let tensors: [String: TensorLocation]
    public let shardPaths: [String]

    public var bundledTemplateURL: URL {
        URL(fileURLWithPath: checkpointPath, isDirectory: true)
            .appendingPathComponent(Self.bundledTemplateFilename, isDirectory: false)
    }

    public var isSelfContainedExecutorReady: Bool {
        layout.isExecutorReady
            && FileManager.default.fileExists(atPath: bundledTemplateURL.path)
    }

    public var totalTensorBytes: UInt64 {
        tensors.values.reduce(0) { $0 + $1.byteCount }
    }

    public init(checkpointURL: URL) throws {
        let root = checkpointURL.standardizedFileURL
        let config = try Self.jsonObject(
            at: root.appendingPathComponent("config.json"),
            description: "config.json"
        )
        guard config["model_type"] as? String == "deepseek_v4" else {
            throw CatalogError.invalidCheckpoint(
                "DwarfStar checkpoint loading requires model_type deepseek_v4."
            )
        }

        layout = Layout(
            isAFMNative: Self.bool(config["afm_native_checkpoint"]),
            usesQ80DenseWeights: Self.bool(config["afm_q8_0"]),
            usesDwarfStarMXFP4Experts: Self.bool(config["afm_dwarfstar_mxfp4_layout"]),
            usesPackedDwarfStarMXFP4Experts: Self.bool(
                config["afm_dwarfstar_mxfp4_packed"]),
            executorLayoutVersion: Self.integer(config["afm_dwarfstar_executor_layout_version"])
        )

        let index = try Self.jsonObject(
            at: root.appendingPathComponent("model.safetensors.index.json"),
            description: "model.safetensors.index.json"
        )
        guard let rawWeightMap = index["weight_map"] as? [String: Any],
              !rawWeightMap.isEmpty
        else {
            throw CatalogError.invalidCheckpoint("Safetensor index has no weight_map entries.")
        }

        var weightMap: [String: String] = [:]
        for (name, value) in rawWeightMap {
            guard let shard = value as? String, !shard.isEmpty else {
                throw CatalogError.invalidCheckpoint(
                    "Safetensor index has an invalid shard for tensor \(name)."
                )
            }
            weightMap[name] = shard
        }

        var discovered: [String: TensorLocation] = [:]
        let shardNames = Set(weightMap.values).sorted()
        for shardName in shardNames {
            let shardURL = try Self.checkedShardURL(named: shardName, under: root)
            let header = try Self.readSafetensorHeader(at: shardURL)
            for (name, metadata) in header.tensors {
                guard name != "__metadata__",
                      !name.hasPrefix("__afm_padding_") else { continue }
                guard discovered[name] == nil else {
                    throw CatalogError.invalidCheckpoint(
                        "Tensor \(name) appears in more than one safetensor shard."
                    )
                }
                discovered[name] = try Self.tensorLocation(
                    name: name,
                    metadata: metadata,
                    shardURL: shardURL,
                    payloadStart: header.payloadStart,
                    fileSize: header.fileSize
                )
            }
        }

        var indexed: [String: TensorLocation] = [:]
        for (name, shardName) in weightMap {
            guard let location = discovered[name] else {
                throw CatalogError.invalidCheckpoint(
                    "Tensor \(name) is indexed in \(shardName) but absent from its safetensor header."
                )
            }
            guard URL(fileURLWithPath: location.shardPath).lastPathComponent == shardName else {
                throw CatalogError.invalidCheckpoint(
                    "Tensor \(name) is stored in \(URL(fileURLWithPath: location.shardPath).lastPathComponent), not indexed shard \(shardName)."
                )
            }
            indexed[name] = location
        }

        checkpointPath = root.path
        tensors = indexed
        shardPaths = shardNames.map {
            root.appendingPathComponent($0, isDirectory: false).standardizedFileURL.path
        }
        if layout.isExecutorReady {
            guard indexed.values.allSatisfy({ $0.fileOffset.isMultiple(of: 32) }) else {
                throw CatalogError.unsupportedLayout(
                    "DwarfStar executor tensors must be aligned to 32-byte file offsets.")
            }
        }
    }

    public func tensor(named name: String) -> TensorLocation? {
        tensors[name]
    }

    public func requireExecutorReady() throws {
        guard layout.isExecutorReady else {
            throw CatalogError.unsupportedLayout(
                "DwarfStar execution requires an AFM native checkpoint with "
                    + "afm_q8_0=true, afm_dwarfstar_mxfp4_layout=true, and "
                    + "afm_dwarfstar_mxfp4_packed=true with "
                    + "afm_dwarfstar_executor_layout_version>=3 with 32-byte-aligned tensors."
            )
        }
    }

    private static func jsonObject(at url: URL, description: String) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CatalogError.invalidCheckpoint("Missing \(description) at \(url.path).")
        }
        do {
            let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            guard let object = value as? [String: Any] else {
                throw CatalogError.invalidCheckpoint("\(description) is not a JSON object.")
            }
            return object
        } catch let error as CatalogError {
            throw error
        } catch {
            throw CatalogError.invalidCheckpoint("Cannot parse \(description): \(error.localizedDescription)")
        }
    }

    private static func checkedShardURL(named name: String, under root: URL) throws -> URL {
        guard URL(fileURLWithPath: name).lastPathComponent == name else {
            throw CatalogError.invalidCheckpoint("Unsafe safetensor shard path \(name).")
        }
        let url = root.appendingPathComponent(name, isDirectory: false).standardizedFileURL
        guard url.deletingLastPathComponent() == root else {
            throw CatalogError.invalidCheckpoint("Safetensor shard escapes checkpoint directory: \(name).")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CatalogError.invalidCheckpoint("Missing safetensor shard at \(url.path).")
        }
        return url
    }

    private static func readSafetensorHeader(at url: URL) throws -> SafetensorHeader {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw CatalogError.invalidCheckpoint(
                "Cannot open safetensor shard \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
        defer { try? handle.close() }

        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw CatalogError.invalidCheckpoint(
                "Safetensor shard \(url.lastPathComponent) has a truncated header length."
            )
        }
        let headerLength = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        let fileSize = try fileByteCount(url)
        guard fileSize >= 8,
              headerLength > 0,
              headerLength <= fileSize - 8,
              headerLength <= UInt64(Int.max)
        else {
            throw CatalogError.invalidCheckpoint(
                "Safetensor shard \(url.lastPathComponent) has invalid header length \(headerLength)."
            )
        }

        let data = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard data.count == Int(headerLength) else {
            throw CatalogError.invalidCheckpoint(
                "Safetensor shard \(url.lastPathComponent) has a truncated JSON header."
            )
        }
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw CatalogError.invalidCheckpoint(
                    "Safetensor shard \(url.lastPathComponent) header is not a JSON object."
                )
            }
            return SafetensorHeader(
                tensors: object,
                payloadStart: 8 + headerLength,
                fileSize: fileSize
            )
        } catch let error as CatalogError {
            throw error
        } catch {
            throw CatalogError.invalidCheckpoint(
                "Cannot parse safetensor header \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    private static func tensorLocation(
        name: String,
        metadata: Any,
        shardURL: URL,
        payloadStart: UInt64,
        fileSize: UInt64
    ) throws -> TensorLocation {
        guard let object = metadata as? [String: Any],
              let dtype = object["dtype"] as? String,
              let rawShape = object["shape"] as? [Any],
              let rawOffsets = object["data_offsets"] as? [Any],
              rawOffsets.count == 2,
              let start = unsignedInteger(rawOffsets[0]),
              let end = unsignedInteger(rawOffsets[1]),
              end >= start
        else {
            throw CatalogError.invalidCheckpoint(
                "Tensor \(name) has invalid safetensor metadata."
            )
        }
        let shape = try rawShape.map { value -> Int in
            guard let dimension = integer(value), dimension >= 0 else {
                throw CatalogError.invalidCheckpoint("Tensor \(name) has an invalid shape.")
            }
            return dimension
        }

        let absoluteStart = payloadStart.addingReportingOverflow(start)
        let absoluteEnd = payloadStart.addingReportingOverflow(end)
        guard !absoluteStart.overflow,
              !absoluteEnd.overflow,
              absoluteEnd.partialValue <= fileSize
        else {
            throw CatalogError.invalidCheckpoint(
                "Tensor \(name) points outside safetensor shard \(shardURL.lastPathComponent)."
            )
        }
        return TensorLocation(
            name: name,
            shardPath: shardURL.path,
            fileOffset: absoluteStart.partialValue,
            byteCount: end - start,
            dtype: dtype,
            shape: shape
        )
    }

    private static func fileByteCount(_ url: URL) throws -> UInt64 {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        guard let size = values.fileSize, size >= 0 else {
            throw CatalogError.invalidCheckpoint("Cannot determine size of \(url.path).")
        }
        return UInt64(size)
    }

    private static func bool(_ value: Any?) -> Bool {
        (value as? Bool) ?? false
    }

    private static func integer(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        if let value = value as? NSNumber { return value.intValue }
        return nil
    }

    private static func unsignedInteger(_ value: Any?) -> UInt64? {
        if let value = value as? UInt64 { return value }
        if let value = value as? Int, value >= 0 { return UInt64(value) }
        if let value = value as? NSNumber, value.int64Value >= 0 {
            return UInt64(value.int64Value)
        }
        return nil
    }
}
