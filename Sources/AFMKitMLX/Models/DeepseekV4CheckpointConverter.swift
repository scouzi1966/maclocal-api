import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Resumable, shard-streaming conversion of an official DeepSeek V4 checkpoint
/// into the tensor layout consumed directly by AFMKit's MLX provider.
public struct DeepseekV4CheckpointConverter {
    public typealias ProgressHandler = (String) -> Void

    public enum Profile: String, Codable, CaseIterable, Sendable {
        case native
        case dwarfstarQ8 = "dwarfstar-q8"
        case dwarfstarSymmetricQ8 = "dwarfstar-symmetric-q8"
        case dwarfstarSymmetricQ8InterleavedMXFP4 =
            "dwarfstar-symmetric-q8-interleaved-mxfp4"
        case dwarfstarSymmetricQ8AlignedMXFP4 =
            "dwarfstar-symmetric-q8-aligned-mxfp4"
    }

    struct Quantization: Codable {
        let groupSize: Int
        let bits: Int
        let mode: String

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    struct CompletedShard: Codable {
        let sourceSize: Int64
        let sourceModificationTime: TimeInterval
        let outputSize: Int64
    }

    struct State: Codable {
        var formatVersion = 1
        var profile: String?
        var completed: [String: CompletedShard] = [:]
        var quantization: [String: Quantization] = [:]
        var weightMap: [String: String] = [:]
    }

    enum ConversionError: LocalizedError {
        case invalidSource(String)
        case unsafeOutput(String)
        case outputExists(String)
        case unsupportedQuantization(String)

        var errorDescription: String? {
            switch self {
            case .invalidSource(let message), .unsafeOutput(let message),
                 .outputExists(let message), .unsupportedQuantization(let message):
                return message
            }
        }
    }

    let source: URL
    let output: URL
    let overwrite: Bool
    let profile: Profile
    let progress: ProgressHandler?

    private var stateURL: URL { output.appendingPathComponent(".afm-mlx-conversion.json") }

    public init(
        source: URL,
        output: URL,
        overwrite: Bool = false,
        profile: Profile = .native,
        progress: ProgressHandler? = nil
    ) {
        self.source = source
        self.output = output
        self.overwrite = overwrite
        self.profile = profile
        self.progress = progress
    }

    public func run() throws {
        let fm = FileManager.default
        let sourceURL = source.standardizedFileURL
        let outputURL = output.standardizedFileURL
        guard sourceURL != outputURL else {
            throw ConversionError.unsafeOutput("Conversion output must differ from the source directory.")
        }
        guard !outputURL.path.hasPrefix(sourceURL.path + "/") else {
            throw ConversionError.unsafeOutput("Conversion output cannot be inside the source checkpoint.")
        }

        try MLXMetalLibrary.ensureAvailable(verbose: true)

        let configURL = sourceURL.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: configURL.path) else {
            throw ConversionError.invalidSource("Missing source config.json at \(configURL.path)")
        }
        let configData = try Data(contentsOf: configURL)
        guard var configObject = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              configObject["model_type"] as? String == "deepseek_v4"
        else {
            throw ConversionError.invalidSource("mlx-convert currently requires model_type deepseek_v4.")
        }

        if overwrite, fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        if fm.fileExists(atPath: outputURL.path), !fm.fileExists(atPath: stateURL.path) {
            let existingItems = try fm.contentsOfDirectory(atPath: outputURL.path)
            guard existingItems.isEmpty else {
                throw ConversionError.outputExists(
                    "Output exists but is not a resumable AFM conversion. Use --overwrite to replace it.")
            }
        }
        try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)

        var state = try loadState()
        if state.profile != nil || !state.completed.isEmpty {
            let storedProfile = state.profile ?? Profile.native.rawValue
            guard storedProfile == profile.rawValue else {
                throw ConversionError.outputExists(
                    "Conversion profile mismatch: output contains \(storedProfile), requested \(profile.rawValue). Use --overwrite or choose another output directory.")
            }
        }
        state.profile = profile.rawValue
        let decoder = JSONDecoder()
        var modelConfig = try decoder.decode(DeepseekV4Configuration.self, from: configData)
        modelConfig.afmNativeCheckpoint = false
        let model = DeepseekV4Model(modelConfig)

        let shards = try fm.contentsOfDirectory(
            at: sourceURL,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else {
            throw ConversionError.invalidSource("No .safetensors shards found in \(sourceURL.path)")
        }

        report("Converting \(shards.count) DeepSeek V4 shards")
        report("  source: \(sourceURL.path)")
        report("  output: \(outputURL.path)")
        report("  profile: \(profile.rawValue)")

        for (index, shard) in shards.enumerated() {
            let name = shard.lastPathComponent
            let values = try shard.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            let sourceSize = Int64(values.fileSize ?? 0)
            let sourceTime = values.contentModificationDate?.timeIntervalSince1970 ?? 0
            let destination = outputURL.appendingPathComponent(name)

            if let completed = state.completed[name],
               completed.sourceSize == sourceSize,
               completed.sourceModificationTime == sourceTime,
               fm.fileExists(atPath: destination.path),
               (try destination.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1)
                    == Int(completed.outputSize)
            {
                report("[\(index + 1)/\(shards.count)] \(name): already converted")
                continue
            }

            report("[\(index + 1)/\(shards.count)] \(name): loading")
            let (weights, metadata) = try loadArraysAndMetadata(url: shard)
            var converted = model.sanitize(weights: weights)
            if profile == .dwarfstarQ8 {
                converted = try convertDwarfstarQ8Control(converted)
            } else if profile == .dwarfstarSymmetricQ8
                        || profile == .dwarfstarSymmetricQ8InterleavedMXFP4
                        || profile == .dwarfstarSymmetricQ8AlignedMXFP4 {
                converted = try convertDwarfstarSymmetricQ8(converted)
                if profile == .dwarfstarSymmetricQ8InterleavedMXFP4 {
                    converted = try convertRoutedMXFP4ToDwarfstarLayout(converted)
                } else if profile == .dwarfstarSymmetricQ8AlignedMXFP4 {
                    converted = try convertRoutedMXFP4ToAlignedSuperblocks(converted)
                }
            }
            try collectMetadata(converted, shard: name, state: &state)

            let partial = outputURL.appendingPathComponent(".\(name).partial.safetensors")
            try? fm.removeItem(at: partial)
            report("[\(index + 1)/\(shards.count)] \(name): writing \(converted.count) tensors")
            try save(arrays: converted, metadata: metadata, url: partial)
            try replaceOrMove(partial, to: destination)
            let outputSize = Int64(
                try destination.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            state.completed[name] = CompletedShard(
                sourceSize: sourceSize,
                sourceModificationTime: sourceTime,
                outputSize: outputSize)
            try saveState(state)
            Memory.clearCache()
        }

        try copySupportFiles(from: sourceURL, to: outputURL)
        configObject["afm_native_checkpoint"] = true
        configObject["afm_symmetric_q8"] = profile == .dwarfstarSymmetricQ8
            || profile == .dwarfstarSymmetricQ8InterleavedMXFP4
            || profile == .dwarfstarSymmetricQ8AlignedMXFP4
        configObject["afm_dwarfstar_mxfp4_layout"] =
            profile == .dwarfstarSymmetricQ8InterleavedMXFP4
        configObject["afm_aligned_mxfp4_layout"] =
            profile == .dwarfstarSymmetricQ8AlignedMXFP4
        configObject["quantization"] = quantizationJSON(state.quantization)
        let convertedConfig = try JSONSerialization.data(
            withJSONObject: configObject, options: [.prettyPrinted, .sortedKeys])
        try writeAtomically(convertedConfig, to: outputURL.appendingPathComponent("config.json"))

        let totalSize = state.completed.values.reduce(Int64(0)) { $0 + $1.outputSize }
        let index: [String: Any] = [
            "metadata": ["total_size": totalSize],
            "weight_map": state.weightMap,
        ]
        let indexData = try JSONSerialization.data(
            withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
        try writeAtomically(
            indexData, to: outputURL.appendingPathComponent("model.safetensors.index.json"))

        report("Conversion complete: \(outputURL.path)")
        report("Run: afm mlx -m \(outputURL.path)")
    }

    private func report(_ message: String) {
        progress?(message)
    }

    private func collectMetadata(
        _ weights: [String: MLXArray], shard: String, state: inout State
    ) throws {
        for (key, value) in weights {
            state.weightMap[key] = shard
            guard key.hasSuffix(".scales") else { continue }
            let base = String(key.dropLast(".scales".count))
            if (profile == .dwarfstarSymmetricQ8
                    || profile == .dwarfstarSymmetricQ8InterleavedMXFP4
                    || profile == .dwarfstarSymmetricQ8AlignedMXFP4),
               Self.usesDwarfstarQ8Control(base)
            {
                state.quantization[base] = Quantization(
                    groupSize: 32, bits: 8, mode: "affine")
                continue
            }
            if profile == .dwarfstarSymmetricQ8AlignedMXFP4,
               base.contains(".switch_mlp.")
            {
                state.quantization[base] = Quantization(
                    groupSize: 32, bits: 4, mode: "mxfp4")
                continue
            }
            guard let weight = weights["\(base).weight"],
                  let inferred = inferOfficialBlockQuantization(
                    weightShape: weight.shape, scaleShape: value.shape)
            else {
                throw ConversionError.unsupportedQuantization(
                    "Cannot infer MXFP layout for \(base): weight=\(weights["\(base).weight"]?.shape ?? []) scales=\(value.shape)")
            }
            let mode = weights["\(base).biases"] == nil ? inferred.mode : .affine
            state.quantization[base] = Quantization(
                groupSize: inferred.groupSize,
                bits: inferred.bits,
                mode: mode.rawValue)
        }
    }

    /// Reproduces the broad tensor-format split advertised by DwarfStar's
    /// DeepSeek V4 MXFP4 package: routed experts remain MXFP4 while attention,
    /// shared-expert, and output matrices use 8-bit affine kernels. This is a
    /// performance-control profile, not AFM's default accuracy profile.
    private func convertDwarfstarQ8Control(
        _ weights: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var result = weights
        let bases = weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        }

        for base in bases.sorted() where Self.usesDwarfstarQ8Control(base) {
            guard let weight = weights["\(base).weight"],
                  let scales = weights["\(base).scales"],
                  let source = inferOfficialBlockQuantization(
                    weightShape: weight.shape, scaleShape: scales.shape)
            else {
                throw ConversionError.unsupportedQuantization(
                    "Cannot convert \(base) to affine Q8")
            }

            let dequantized = MLX.dequantized(
                weight,
                scales: scales,
                biases: weights["\(base).biases"],
                groupSize: source.groupSize,
                bits: source.bits,
                mode: source.mode,
                dtype: .float16)
            let quantized = MLX.quantized(
                dequantized, groupSize: 32, bits: 8, mode: .affine)
            var arrays = [quantized.wq, quantized.scales]
            if let biases = quantized.biases { arrays.append(biases) }
            MLX.eval(arrays)

            result["\(base).weight"] = quantized.wq
            result["\(base).scales"] = quantized.scales
            if let biases = quantized.biases {
                result["\(base).biases"] = biases
            } else {
                result.removeValue(forKey: "\(base).biases")
            }
            report("    affine Q8: \(base)")
        }
        return result
    }

    /// Converts DwarfStar's dense tensor subset to signed symmetric Q8 blocks.
    /// Four signed bytes are packed into each UInt32 while one FP16 scale is
    /// stored per 32 weights. The explicit checkpoint capability selects the
    /// matching custom runtime; ordinary MLX affine-Q8 models are unaffected.
    private func convertDwarfstarSymmetricQ8(
        _ weights: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var result = weights
        let bases = weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        }

        for base in bases.sorted() where Self.usesDwarfstarQ8Control(base) {
            guard let weight = weights["\(base).weight"],
                  let scales = weights["\(base).scales"],
                  let source = inferOfficialBlockQuantization(
                    weightShape: weight.shape, scaleShape: scales.shape)
            else {
                throw ConversionError.unsupportedQuantization(
                    "Cannot convert \(base) to symmetric Q8")
            }

            let dequantized = MLX.dequantized(
                weight,
                scales: scales,
                biases: weights["\(base).biases"],
                groupSize: source.groupSize,
                bits: source.bits,
                mode: source.mode,
                dtype: .float16)
            guard dequantized.ndim == 2, dequantized.dim(1).isMultiple(of: 32) else {
                throw ConversionError.unsupportedQuantization(
                    "Symmetric Q8 requires a 2-D matrix with 32-aligned input: \(base) \(dequantized.shape)")
            }

            let outputRows = dequantized.dim(0)
            let inputDimensions = dequantized.dim(1)
            let groups = inputDimensions / 32
            let grouped = dequantized.reshaped([outputRows, groups, 32])
            let scale = maximum(
                abs(grouped).max(axis: 2, keepDims: true) / 127,
                MLXArray(Float(1.0e-8)))
            let signed = clip(round(grouped / scale), min: -127, max: 127)
                .asType(.int32)
            let bytes = bitwiseAnd(signed.asType(.uint32), 0xff)
                .reshaped([outputRows, inputDimensions / 4, 4])
            let packed = bitwiseOr(
                bitwiseOr(bytes[0..., 0..., 0], leftShift(bytes[0..., 0..., 1], 8)),
                bitwiseOr(leftShift(bytes[0..., 0..., 2], 16),
                    leftShift(bytes[0..., 0..., 3], 24)))
            let q8Scales = scale.reshaped([outputRows, groups]).asType(.float16)
            MLX.eval(packed, q8Scales)

            result["\(base).weight"] = packed
            result["\(base).scales"] = q8Scales
            result.removeValue(forKey: "\(base).biases")
            report("    symmetric Q8: \(base)")
        }
        return result
    }

    /// Reorders each 32-value routed MXFP4 block into DwarfStar's lane-oriented
    /// byte layout. Quantized values and E8M0 scales are unchanged.
    private func convertRoutedMXFP4ToDwarfstarLayout(
        _ weights: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var result = weights
        let bases = weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        }

        for base in bases.sorted() where base.contains(".switch_mlp.") {
            guard let weight = weights["\(base).weight"],
                  let scales = weights["\(base).scales"],
                  weight.dtype == .uint32,
                  scales.dtype == .uint8,
                  weight.size == scales.size * 4
            else {
                throw ConversionError.unsupportedQuantization(
                    "DwarfStar MXFP4 layout requires uint32 weights and one uint8 scale per four words: \(base)")
            }

            let rows = scales.size
            let words = weight.reshaped([rows, 4])
            var bytes: [MLXArray] = []
            bytes.reserveCapacity(16)
            for index in 0..<16 {
                let low = bitwiseAnd(
                    rightShift(words[0..., index / 8], (index % 8) * 4), 0xf)
                let highIndex = index + 16
                let high = bitwiseAnd(
                    rightShift(words[0..., highIndex / 8], (highIndex % 8) * 4), 0xf)
                bytes.append(bitwiseOr(low, leftShift(high, 4)))
            }
            var repacked: [MLXArray] = []
            repacked.reserveCapacity(4)
            for index in stride(from: 0, to: 16, by: 4) {
                repacked.append(bitwiseOr(
                    bitwiseOr(bytes[index], leftShift(bytes[index + 1], 8)),
                    bitwiseOr(leftShift(bytes[index + 2], 16),
                        leftShift(bytes[index + 3], 24))))
            }
            let reordered = stacked(repacked, axis: 1).reshaped(weight.shape)
            MLX.eval(reordered)
            result["\(base).weight"] = reordered
            report("    DwarfStar MXFP4 layout: \(base)")
        }
        return result
    }

    /// Packs 16 routed MXFP4 groups into a 272-byte aligned superblock:
    /// 16 E8M0 scale bytes followed by 64 original UInt32 weight words.
    /// The original nibble ordering is retained, so the optimized kernel can
    /// use aligned word loads without changing accumulation order.
    private func convertRoutedMXFP4ToAlignedSuperblocks(
        _ weights: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var result = weights
        let bases = weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        }

        for base in bases.sorted() where base.contains(".switch_mlp.") {
            guard let weight = weights["\(base).weight"],
                  let scales = weights["\(base).scales"],
                  weight.dtype == .uint32,
                  scales.dtype == .uint8,
                  weight.size == scales.size * 4,
                  let groups = scales.shape.last,
                  groups.isMultiple(of: 16)
            else {
                throw ConversionError.unsupportedQuantization(
                    "Aligned MXFP4 layout requires UInt32 weights and E8M0 groups divisible by 16: \(base)")
            }

            let aligned = try Self.alignedMXFP4Superblocks(
                weight: weight, scales: scales)
            MLX.eval(aligned)
            result["\(base).weight"] = aligned
            report("    aligned MXFP4 superblocks: \(base)")
        }
        return result
    }

    static func alignedMXFP4Superblocks(
        weight: MLXArray, scales: MLXArray
    ) throws -> MLXArray {
        guard weight.dtype == .uint32,
              scales.dtype == .uint8,
              weight.size == scales.size * 4,
              let groups = scales.shape.last,
              groups.isMultiple(of: 16)
        else {
            throw ConversionError.unsupportedQuantization(
                "Aligned MXFP4 packing requires UInt32 weights and E8M0 groups divisible by 16.")
        }
        let rows = scales.size / groups
        let superblocks = groups / 16
        let scaleBytes = scales.reshaped([rows, superblocks, 4, 4]).asType(.uint32)
        let scaleWords = bitwiseOr(
            bitwiseOr(scaleBytes[0..., 0..., 0..., 0],
                leftShift(scaleBytes[0..., 0..., 0..., 1], 8)),
            bitwiseOr(leftShift(scaleBytes[0..., 0..., 0..., 2], 16),
                leftShift(scaleBytes[0..., 0..., 0..., 3], 24)))
        let payload = weight.reshaped([rows, superblocks, 64])
        return concatenated([scaleWords, payload], axis: 2)
            .reshaped(Array(weight.shape.dropLast()) + [superblocks * 68])
    }

    static func usesDwarfstarQ8Control(_ base: String) -> Bool {
        base == "lm_head"
            || (base.contains(".self_attn.")
                && !base.contains(".compressor.")
                && !base.contains(".indexer."))
            || base.contains(".mlp.shared_experts.")
    }

    private func quantizationJSON(_ entries: [String: Quantization]) -> [String: Any] {
        var result: [String: Any] = [
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
        ]
        for (key, value) in entries where value.bits != 4 || value.mode != "mxfp4" {
            result[key] = [
                "group_size": value.groupSize,
                "bits": value.bits,
                "mode": value.mode,
            ]
        }
        return result
    }

    private func loadState() throws -> State {
        guard FileManager.default.fileExists(atPath: stateURL.path) else { return State() }
        return try JSONDecoder().decode(State.self, from: Data(contentsOf: stateURL))
    }

    private func saveState(_ state: State) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try writeAtomically(try encoder.encode(state), to: stateURL)
    }

    private func writeAtomically(_ data: Data, to url: URL) throws {
        let partial = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).partial")
        try data.write(to: partial, options: .atomic)
        try replaceOrMove(partial, to: url)
    }

    private func replaceOrMove(_ source: URL, to destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            _ = try fm.replaceItemAt(destination, withItemAt: source)
        } else {
            try fm.moveItem(at: source, to: destination)
        }
    }

    private func copySupportFiles(from source: URL, to output: URL) throws {
        let fm = FileManager.default
        let names = [
            "LICENSE", "README.md", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "tokenizer.model", "tokenizer_config.jinja", "encoding",
        ]
        for name in names {
            let from = source.appendingPathComponent(name)
            let to = output.appendingPathComponent(name)
            guard fm.fileExists(atPath: from.path) else { continue }
            if fm.fileExists(atPath: to.path) { try fm.removeItem(at: to) }
            try fm.copyItem(at: from, to: to)
        }
    }
}
