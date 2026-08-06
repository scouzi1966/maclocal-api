import Foundation

/// A zero-copy projection of AFM safetensor shards into the contiguous address
/// layout expected by DwarfStar. The metadata file is small; tensor bytes stay
/// in their original shard files.
public struct AFMDwarfStarProjection: Sendable {
    public struct Region: Sendable, Equatable {
        public let path: String
        public let virtualOffset: UInt64
        public let fileOffset: UInt64
        public let length: UInt64
    }

    public enum ProjectionError: LocalizedError {
        case invalidTemplate(String)
        case incompatibleTensor(String)

        public var errorDescription: String? {
            switch self {
            case .invalidTemplate(let message), .incompatibleTensor(let message): message
            }
        }
    }

    public let metadataPath: String
    public let virtualSize: UInt64
    public let regions: [Region]

    /// Copies only the GGUF metadata and tensor descriptors needed to project
    /// AFM safetensor shards. The resulting file is typically a few MiB and
    /// makes an executor checkpoint self-contained without duplicating weights.
    public static func writeMetadataTemplate(
        from sourceGGUF: URL,
        to outputURL: URL
    ) throws {
        let template = try GGUFTemplate(url: sourceGGUF)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try template.prefix.write(to: outputURL, options: .atomic)
    }

    public static func build(
        checkpointURL: URL,
        templateGGUF: URL,
        metadataOutputURL: URL
    ) throws -> Self {
        let catalog = try AFMDwarfStarCheckpointCatalog(checkpointURL: checkpointURL)
        try catalog.requireExecutorReady()
        let template = try GGUFTemplate(url: templateGGUF)
        let page = UInt64(getpagesize())

        var shardBases: [String: UInt64] = [:]
        var regions: [Region] = []
        var cursor = align(UInt64(template.prefix.count), to: page)
        for path in catalog.shardPaths.sorted() {
            let size = try fileSize(URL(fileURLWithPath: path))
            shardBases[path] = cursor
            let length = align(size, to: page)
            regions.append(Region(
                path: path,
                virtualOffset: cursor,
                fileOffset: 0,
                length: length))
            cursor += length
        }

        var prefix = template.prefix
        for tensor in template.tensors {
            guard let afmName = afmTensorName(for: tensor.name),
                  let location = catalog.tensor(named: afmName),
                  let shardBase = shardBases[location.shardPath]
            else {
                throw ProjectionError.incompatibleTensor(
                    "No AFM tensor mapping for DwarfStar tensor \(tensor.name).")
            }
            guard location.byteCount == tensor.byteCount else {
                throw ProjectionError.incompatibleTensor(
                    "Tensor \(tensor.name) expects \(tensor.byteCount) bytes but \(afmName) stores \(location.byteCount).")
            }
            let absoluteOffset = shardBase + location.fileOffset
            guard absoluteOffset >= template.tensorDataOffset else {
                throw ProjectionError.invalidTemplate("Tensor offset precedes GGUF data section.")
            }
            prefix.replaceLittleEndian(
                absoluteOffset - template.tensorDataOffset,
                at: tensor.relativeOffsetPosition)
        }
        prefix.append(contentsOf: repeatElement(
            UInt8(0), count: Int(align(UInt64(prefix.count), to: page) - UInt64(prefix.count))))
        try FileManager.default.createDirectory(
            at: metadataOutputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try prefix.write(to: metadataOutputURL, options: .atomic)

        return Self(
            metadataPath: metadataOutputURL.path,
            virtualSize: align(cursor, to: page),
            regions: regions)
    }

    /// Builds a projection that maps an existing GGUF's tensor payload through
    /// the external-region loader. This does not copy model data and is useful
    /// for proving that projected and conventional mappings are equivalent.
    public static func buildGGUFAlias(
        ggufURL: URL,
        metadataOutputURL: URL
    ) throws -> Self {
        let template = try GGUFTemplate(url: ggufURL)
        let page = UInt64(getpagesize())
        let sourceSize = try fileSize(ggufURL)
        let regionBase = align(UInt64(template.prefix.count), to: page)

        var prefix = template.prefix
        for tensor in template.tensors {
            let sourceAbsoluteOffset = template.tensorDataOffset + tensor.relativeOffset
            guard sourceAbsoluteOffset <= sourceSize,
                  tensor.byteCount <= sourceSize - sourceAbsoluteOffset
            else {
                throw ProjectionError.invalidTemplate(
                    "Tensor \(tensor.name) points outside the GGUF source file.")
            }
            let projectedAbsoluteOffset = regionBase + sourceAbsoluteOffset
            prefix.replaceLittleEndian(
                projectedAbsoluteOffset - template.tensorDataOffset,
                at: tensor.relativeOffsetPosition)
        }
        prefix.append(contentsOf: repeatElement(
            UInt8(0), count: Int(align(UInt64(prefix.count), to: page) - UInt64(prefix.count))))
        try FileManager.default.createDirectory(
            at: metadataOutputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try prefix.write(to: metadataOutputURL, options: .atomic)

        let regionLength = align(sourceSize, to: page)
        return Self(
            metadataPath: metadataOutputURL.path,
            virtualSize: regionBase + regionLength,
            regions: [Region(
                path: ggufURL.path,
                virtualOffset: regionBase,
                fileOffset: 0,
                length: regionLength,
            )]
        )
    }

    private static func afmTensorName(for name: String) -> String? {
        let top: [String: String] = [
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
            "output_hc_base.weight": "model.hc_head.hc_head_base",
            "output_hc_fn.weight": "model.hc_head.hc_head_fn",
            "output_hc_scale.weight": "model.hc_head.hc_head_scale",
        ]
        if let mapped = top[name] { return mapped }
        guard name.hasPrefix("blk."),
              let separator = name.dropFirst(4).firstIndex(of: "."),
              let layer = Int(name[name.index(name.startIndex, offsetBy: 4)..<separator])
        else { return nil }
        let rest = String(name[name.index(after: separator)...])
        let prefix = "model.layers.\(layer)."
        let regular: [String: String] = [
            "hc_attn_base.weight": "attn_hc.base",
            "hc_attn_fn.weight": "attn_hc.fn",
            "hc_attn_scale.weight": "attn_hc.scale",
            "hc_ffn_base.weight": "ffn_hc.base",
            "hc_ffn_fn.weight": "ffn_hc.fn",
            "hc_ffn_scale.weight": "ffn_hc.scale",
            "attn_sinks.weight": "self_attn.attn_sink",
            "attn_q_a.weight": "self_attn.wq_a.weight",
            "attn_q_b.weight": "self_attn.wq_b.weight",
            "attn_q_a_norm.weight": "self_attn.q_norm.weight",
            "attn_kv.weight": "self_attn.wkv.weight",
            "attn_kv_a_norm.weight": "self_attn.kv_norm.weight",
            "attn_output_a.weight": "self_attn.wo_a.weight",
            "attn_output_b.weight": "self_attn.wo_b.weight",
            "attn_compressor_ape.weight": "self_attn.compressor.ape",
            "attn_compressor_kv.weight": "self_attn.compressor.wkv.weight",
            "attn_compressor_gate.weight": "self_attn.compressor.wgate.weight",
            "attn_compressor_norm.weight": "self_attn.compressor.norm.weight",
            "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
            "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
            "indexer_compressor_ape.weight": "self_attn.indexer.compressor.ape",
            "indexer_compressor_kv.weight": "self_attn.indexer.compressor.wkv.weight",
            "indexer_compressor_gate.weight": "self_attn.indexer.compressor.wgate.weight",
            "indexer_compressor_norm.weight": "self_attn.indexer.compressor.norm.weight",
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
            "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
            "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
            "ffn_gate_inp.weight": "mlp.gate.weight",
            "exp_probs_b.bias": "mlp.gate.bias",
            "ffn_gate_tid2eid.weight": "mlp.gate.tid2eid",
        ]
        if let mapped = regular[rest] { return prefix + mapped }
        let experts: [(String, String)] = [
            ("ffn_gate_exps.weight", "mlp.switch_mlp.gate_proj.weight"),
            ("ffn_down_exps.weight", "mlp.switch_mlp.down_proj.weight"),
            ("ffn_up_exps.weight", "mlp.switch_mlp.up_proj.weight"),
        ]
        for (source, target) in experts where rest == source { return prefix + target }
        return nil
    }

    fileprivate static func align(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        let remainder = value % alignment
        return remainder == 0 ? value : value + alignment - remainder
    }

    private static func fileSize(_ url: URL) throws -> UInt64 {
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
        guard size >= 0 else { throw ProjectionError.invalidTemplate("Cannot stat \(url.path).") }
        return UInt64(size)
    }
}

private struct GGUFTemplate {
    struct Tensor {
        let name: String
        let byteCount: UInt64
        let relativeOffsetPosition: Int
        let relativeOffset: UInt64
    }

    let prefix: Data
    let tensors: [Tensor]
    let tensorDataOffset: UInt64

    init(url: URL) throws {
        let reader = try BinaryReader(url: url)
        guard try reader.bytes(4) == Data("GGUF".utf8), try reader.u32() == 3 else {
            throw AFMDwarfStarProjection.ProjectionError.invalidTemplate("Expected a GGUF v3 template.")
        }
        let tensorCount = try reader.u64()
        let metadataCount = try reader.u64()
        var alignment: UInt64 = 32
        for _ in 0..<metadataCount {
            let key = try reader.string()
            let type = try reader.u32()
            if key == "general.alignment", type == 4 {
                alignment = UInt64(try reader.u32())
            } else {
                try reader.skipValue(type: type)
            }
        }
        var parsed: [Tensor] = []
        parsed.reserveCapacity(Int(tensorCount))
        for _ in 0..<tensorCount {
            let name = try reader.string()
            let dimensions = try reader.u32()
            guard dimensions > 0, dimensions <= 4 else {
                throw AFMDwarfStarProjection.ProjectionError.invalidTemplate("Invalid tensor rank.")
            }
            var elements: UInt64 = 1
            for _ in 0..<dimensions { elements *= try reader.u64() }
            let type = try reader.u32()
            let offsetPosition = reader.position
            let relativeOffset = try reader.u64()
            parsed.append(Tensor(
                name: name,
                byteCount: try Self.byteCount(type: type, elements: elements),
                relativeOffsetPosition: offsetPosition,
                relativeOffset: relativeOffset))
        }
        tensorDataOffset = AFMDwarfStarProjection.align(UInt64(reader.position), to: alignment)
        prefix = try reader.prefix(count: Int(tensorDataOffset))
        tensors = parsed
    }

    private static func byteCount(type: UInt32, elements: UInt64) throws -> UInt64 {
        let block: (UInt64, UInt64)
        switch type {
        case 0: block = (1, 4)
        case 1: block = (1, 2)
        case 8: block = (32, 34)
        case 26: block = (1, 4)
        case 39: block = (32, 17)
        default:
            throw AFMDwarfStarProjection.ProjectionError.invalidTemplate(
                "Unsupported DwarfStar tensor type \(type).")
        }
        guard elements.isMultiple(of: block.0) else {
            throw AFMDwarfStarProjection.ProjectionError.invalidTemplate("Invalid tensor block geometry.")
        }
        return elements / block.0 * block.1
    }
}

private final class BinaryReader {
    private let handle: FileHandle
    private(set) var position = 0

    init(url: URL) throws { handle = try FileHandle(forReadingFrom: url) }
    deinit { try? handle.close() }

    func bytes(_ count: Int) throws -> Data {
        let data = try handle.read(upToCount: count) ?? Data()
        guard data.count == count else { throw CocoaError(.fileReadCorruptFile) }
        position += count
        return data
    }
    func u32() throws -> UInt32 { try integer(UInt32.self) }
    func u64() throws -> UInt64 { try integer(UInt64.self) }
    func string() throws -> String {
        let count = try u64()
        guard count <= UInt64(Int.max), let value = String(data: try bytes(Int(count)), encoding: .utf8)
        else { throw CocoaError(.fileReadCorruptFile) }
        return value
    }
    func skipValue(type: UInt32, depth: Int = 0) throws {
        guard depth <= 8 else { throw CocoaError(.fileReadCorruptFile) }
        let scalar: [UInt32: Int] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]
        if let count = scalar[type] { _ = try bytes(count); return }
        if type == 8 { _ = try string(); return }
        if type == 9 {
            let elementType = try u32(), count = try u64()
            for _ in 0..<count { try skipValue(type: elementType, depth: depth + 1) }
            return
        }
        throw CocoaError(.fileReadCorruptFile)
    }
    func prefix(count: Int) throws -> Data {
        try handle.seek(toOffset: 0)
        position = 0
        return try bytes(count)
    }
    private func integer<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let data = try bytes(MemoryLayout<T>.size)
        return data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
    }
}

private extension Data {
    mutating func replaceLittleEndian(_ value: UInt64, at offset: Int) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { replaceSubrange(offset..<(offset + 8), with: $0) }
    }
}
