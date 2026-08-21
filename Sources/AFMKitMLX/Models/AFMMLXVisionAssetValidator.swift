import Foundation
import MLXVLM

public final class AFMMLXVisionAssetValidator: @unchecked Sendable {
    private struct SafetensorEvidence {
        struct TensorMetadata {
            let dtype: String
            let shape: [Int]
        }

        let tensors: [String: TensorMetadata]

        var tensorNames: Set<String> { Set(tensors.keys) }
    }

    private struct SnapshotFingerprint: Hashable {
        struct FileEvidence: Hashable {
            let name: String
            let size: Int64
            let modifiedAt: TimeInterval
        }

        let directory: String
        let files: [FileEvidence]

        var identity: String {
            let evidence = files.map {
                "\($0.name):\($0.size):\($0.modifiedAt)"
            }.joined(separator: "|")
            return "\(directory)|\(evidence)"
        }
    }

    private let lock = NSLock()
    private var cache: [SnapshotFingerprint: AFMMLXVisionAssetQualification] = [:]

    public init() {}

    public func qualify(
        modelDirectory: URL,
        architecture: AFMMLXModelArchitecturePreflight
    ) -> AFMMLXVisionAssetQualification {
        let fingerprint = snapshotFingerprint(for: modelDirectory)
        if let cached = withLock({ cache[fingerprint] }) {
            return cached
        }

        let qualification = Self.inspect(
            modelDirectory: modelDirectory,
            architecture: architecture,
            snapshotIdentity: fingerprint.identity
        )
        return withLock {
            if let cached = cache[fingerprint] {
                return cached
            }
            cache[fingerprint] = qualification
            return qualification
        }
    }

    private static func inspect(
        modelDirectory: URL,
        architecture: AFMMLXModelArchitecturePreflight,
        snapshotIdentity: String
    ) -> AFMMLXVisionAssetQualification {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let configData = try? Data(contentsOf: configURL)
        let config = jsonObject(at: configURL) ?? [:]
        let isConditionalGeneration = conditionalGenerationArchitecture(in: config)
        let isQwenConditional = isQwenConditionalModelType(
            architecture.canonicalModelType
        ) && isConditionalGeneration
        let hasVisionConfiguration: Bool
        if isQwenConditional {
            hasVisionConfiguration = configData.flatMap {
                try? JSONDecoder().decode(Qwen3_5MoEVLConfiguration.self, from: $0)
            } != nil
        } else {
            hasVisionConfiguration = config["vision_config"] is [String: Any]
        }
        let hasImageTokenIdentifiers = integer(config["image_token_id"]) != nil
            && integer(config["vision_start_token_id"]) != nil
            && integer(config["vision_end_token_id"]) != nil
        let processorClass = selectedProcessorClass(
            modelDirectory: modelDirectory,
            canonicalModelType: architecture.canonicalModelType
        )
        let visionTensorNames = visionTensorNames(
            in: modelDirectory,
            config: config,
            requiresCompleteQwenTower: isQwenConditional
        )
        let visionTensorCount = visionTensorNames.count

        var missing = Set<AFMMLXVisionAssetIssue>()
        if !isConditionalGeneration {
            missing.insert(.conditionalGenerationArchitecture)
        }
        if !hasVisionConfiguration {
            missing.insert(.visionConfiguration)
        }
        if !hasImageTokenIdentifiers {
            missing.insert(.imageTokenIdentifiers)
        }
        if processorClass == nil {
            missing.insert(.processorConfiguration)
        }
        if visionTensorCount == 0 {
            missing.insert(.visionWeights)
        }

        return AFMMLXVisionAssetQualification(
            snapshotIdentity: snapshotIdentity,
            modelType: architecture.modelType,
            canonicalModelType: architecture.canonicalModelType,
            isConditionalGeneration: isConditionalGeneration,
            declaresVision: architecture.isVisionConfiguration,
            processorClass: processorClass,
            visionTensorCount: visionTensorCount,
            missingAssets: missing
        )
    }

    private static func conditionalGenerationArchitecture(
        in config: [String: Any]
    ) -> Bool {
        guard let architectures = config["architectures"] as? [String] else {
            return false
        }
        return architectures.contains { architecture in
            let normalized = architecture
                .lowercased()
                .replacingOccurrences(of: "_", with: "")
                .replacingOccurrences(of: "-", with: "")
            return normalized.hasPrefix("qwen35")
                && normalized.hasSuffix("forconditionalgeneration")
        }
    }

    private static func isQwenConditionalModelType(_ canonicalModelType: String) -> Bool {
        canonicalModelType == "qwen3_5" || canonicalModelType == "qwen3_5_moe"
    }

    private static func selectedProcessorClass(
        modelDirectory: URL,
        canonicalModelType: String
    ) -> String? {
        let preprocessor = modelDirectory.appendingPathComponent(
            "preprocessor_config.json"
        )
        let processor = modelDirectory.appendingPathComponent("processor_config.json")
        let selectedURL = FileManager.default.fileExists(atPath: preprocessor.path)
            ? preprocessor
            : processor
        guard let data = try? Data(contentsOf: selectedURL),
              let baseConfig = try? JSONDecoder().decode(
                  BaseProcessorConfiguration.self,
                  from: data
              ),
              !baseConfig.processorClass.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }

        if canonicalModelType == "qwen3_5" || canonicalModelType == "qwen3_5_moe" {
            guard (try? JSONDecoder().decode(
                Qwen3VLProcessorConfiguration.self,
                from: data
            )) != nil else { return nil }
            return "Qwen3VLProcessor"
        }
        return baseConfig.processorClass
    }

    private static func visionTensorNames(
        in modelDirectory: URL,
        config: [String: Any],
        requiresCompleteQwenTower: Bool
    ) -> Set<String> {
        let indexURL = modelDirectory.appendingPathComponent(
            "model.safetensors.index.json"
        )
        let tensors: [String: SafetensorEvidence.TensorMetadata]
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let index = jsonObject(at: indexURL),
                  let rawWeightMap = index["weight_map"] as? [String: Any]
            else { return [] }
            let weightMap = rawWeightMap.compactMapValues { $0 as? String }
            guard weightMap.count == rawWeightMap.count, !weightMap.isEmpty else {
                return []
            }
            var shardEvidence: [String: SafetensorEvidence] = [:]
            for shardName in Set(weightMap.values) {
                let shardURL = modelDirectory.appendingPathComponent(shardName)
                guard shardURL.standardizedFileURL.deletingLastPathComponent()
                        == modelDirectory.standardizedFileURL,
                      let evidence = safetensorEvidence(in: shardURL)
                else { return [] }
                shardEvidence[shardName] = evidence
            }
            guard weightMap.allSatisfy({ tensorName, shardName in
                shardEvidence[shardName]?.tensors[tensorName] != nil
            }) else { return [] }
            var discovered: [String: SafetensorEvidence.TensorMetadata] = [:]
            for (tensorName, shardName) in weightMap where isVisionTensorName(tensorName) {
                guard let metadata = shardEvidence[shardName]?.tensors[tensorName] else {
                    return []
                }
                discovered[normalizedVisionTensorName(tensorName)] = metadata
            }
            tensors = discovered
        } else {
            guard let files = try? FileManager.default.contentsOfDirectory(
                at: modelDirectory,
                includingPropertiesForKeys: nil
            ) else { return [] }
            let weightFiles = files.filter {
                $0.pathExtension == "safetensors"
                    && $0.lastPathComponent != "mtp.safetensors"
            }
            guard !weightFiles.isEmpty else { return [] }
            var discovered: [String: SafetensorEvidence.TensorMetadata] = [:]
            for file in weightFiles {
                guard let evidence = safetensorEvidence(in: file) else { return [] }
                for (name, metadata) in evidence.tensors where isVisionTensorName(name) {
                    discovered[normalizedVisionTensorName(name)] = metadata
                }
            }
            tensors = discovered
        }

        if requiresCompleteQwenTower {
            guard let required = requiredQwenVisionTensorNames(config: config),
                  required.isSubset(of: Set(tensors.keys)),
                  hasCompleteQwenQuantizationCompanions(
                    required: required,
                    tensors: tensors,
                    config: config
                  )
            else { return [] }
        }
        return Set(tensors.keys)
    }

    private static func isVisionTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") || name.hasPrefix("model.visual")
    }

    private static func normalizedVisionTensorName(_ name: String) -> String {
        if name.hasPrefix("model.visual.") {
            return "vision_tower." + name.dropFirst("model.visual.".count)
        }
        return name
    }

    private static func requiredQwenVisionTensorNames(
        config: [String: Any]
    ) -> Set<String>? {
        guard let vision = config["vision_config"] as? [String: Any],
              let depth = integer(vision["depth"]), depth > 0
        else { return nil }
        let deepstackIndexes = (vision["deepstack_visual_indexes"] as? [Any])?
            .compactMap(integer) ?? []
        guard deepstackIndexes.allSatisfy({ $0 >= 0 && $0 < depth }) else {
            return nil
        }

        var required: Set<String> = [
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.patch_embed.proj.bias",
            "vision_tower.pos_embed.weight",
        ]
        let blockSuffixes = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight",
            "mlp.linear_fc2.bias", "mlp.linear_fc2.weight",
            "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        for block in 0..<depth {
            for suffix in blockSuffixes {
                required.insert("vision_tower.blocks.\(block).\(suffix)")
            }
        }
        let mergerSuffixes = [
            "linear_fc1.bias", "linear_fc1.weight",
            "linear_fc2.bias", "linear_fc2.weight", "norm.bias", "norm.weight",
        ]
        for suffix in mergerSuffixes {
            required.insert("vision_tower.merger.\(suffix)")
        }
        for index in deepstackIndexes.indices {
            for suffix in mergerSuffixes {
                required.insert("vision_tower.deepstack_merger_list.\(index).\(suffix)")
            }
        }
        return required
    }

    private static func hasCompleteQwenQuantizationCompanions(
        required: Set<String>,
        tensors: [String: SafetensorEvidence.TensorMetadata],
        config: [String: Any]
    ) -> Bool {
        let quantization = (config["quantization_config"] as? [String: Any])
            ?? (config["quantization"] as? [String: Any])
        let mode = (quantization?["mode"] as? String)?.lowercased()
        let isMXFP = mode == "mxfp4" || mode == "mxfp8"

        for weightName in required where weightName.hasSuffix(".weight") {
            guard let metadata = tensors[weightName] else { return false }
            let base = String(weightName.dropLast(".weight".count))
            let hasScales = tensors["\(base).scales"] != nil
            let hasBiases = tensors["\(base).biases"] != nil
            let packedWeight = ["U8", "U16", "U32", "I8", "I16", "I32"]
                .contains(metadata.dtype.uppercased())
            let hasQuantizedRepresentation = packedWeight || hasScales || hasBiases
            guard !hasQuantizedRepresentation || quantization != nil else {
                return false
            }
            guard hasQuantizedRepresentation else { continue }
            guard hasScales else { return false }
            if isMXFP {
                guard !hasBiases else { return false }
            } else {
                guard hasBiases else { return false }
            }
        }
        return true
    }

    private static func safetensorEvidence(in url: URL) -> SafetensorEvidence? {
        let maximumHeaderBytes = 64 * 1_024 * 1_024
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8),
              prefix.count == 8
        else { return nil }

        let headerSize = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard headerSize <= UInt64(maximumHeaderBytes),
              let header = try? handle.read(upToCount: Int(headerSize)),
              header.count == Int(headerSize),
              let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = (attributes[.size] as? NSNumber)?.uint64Value
        else { return nil }
        let payloadStart = UInt64(8) + headerSize
        guard payloadStart <= fileSize else { return nil }

        var tensors: [String: SafetensorEvidence.TensorMetadata] = [:]
        var maximumEnd = UInt64(0)
        for (name, rawMetadata) in object where name != "__metadata__" {
            guard let metadata = rawMetadata as? [String: Any],
                  let dtype = metadata["dtype"] as? String,
                  let rawShape = metadata["shape"] as? [Any],
                  let shape = integerShape(rawShape),
                  let offsets = metadata["data_offsets"] as? [Any],
                  offsets.count == 2,
                  let start = unsignedInteger(offsets[0]),
                  let end = unsignedInteger(offsets[1]),
                  start <= end,
                  let expectedBytes = tensorByteCount(dtype: dtype, shape: shape),
                  end - start == expectedBytes
            else { return nil }
            maximumEnd = max(maximumEnd, end)
            tensors[name] = SafetensorEvidence.TensorMetadata(
                dtype: dtype,
                shape: shape
            )
        }
        guard !tensors.isEmpty,
              maximumEnd <= fileSize - payloadStart
        else { return nil }
        return SafetensorEvidence(tensors: tensors)
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func integer(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        return (value as? NSNumber)?.intValue
    }

    private static func unsignedInteger(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber, number.int64Value >= 0 else {
            return nil
        }
        return number.uint64Value
    }

    private static func integerShape(_ values: [Any]) -> [Int]? {
        let shape = values.compactMap(integer)
        guard shape.count == values.count,
              shape.allSatisfy({ $0 >= 0 })
        else { return nil }
        return shape
    }

    private static func tensorByteCount(dtype: String, shape: [Int]) -> UInt64? {
        let byteWidth: UInt64
        switch dtype.uppercased() {
        case "BOOL", "I8", "U8", "F8_E4M3", "F8_E5M2", "F8_E8M0":
            byteWidth = 1
        case "I16", "U16", "F16", "BF16":
            byteWidth = 2
        case "I32", "U32", "F32":
            byteWidth = 4
        case "I64", "U64", "F64":
            byteWidth = 8
        default:
            return nil
        }
        var elements: UInt64 = 1
        for dimension in shape {
            let (next, overflow) = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !overflow else { return nil }
            elements = next
        }
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: byteWidth)
        return overflow ? nil : bytes
    }

    private func snapshotFingerprint(for modelDirectory: URL) -> SnapshotFingerprint {
        let directory = modelDirectory.standardizedFileURL.path
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        )) ?? []
        let evidence = files
            .filter(Self.isQualificationInput)
            .compactMap { url -> SnapshotFingerprint.FileEvidence? in
                guard let values = try? url.resourceValues(forKeys: keys) else {
                    return nil
                }
                return SnapshotFingerprint.FileEvidence(
                    name: url.lastPathComponent,
                    size: Int64(values.fileSize ?? 0),
                    modifiedAt: values.contentModificationDate?.timeIntervalSince1970 ?? 0
                )
            }
            .sorted { $0.name < $1.name }
        return SnapshotFingerprint(directory: directory, files: evidence)
    }

    private static func isQualificationInput(_ url: URL) -> Bool {
        switch url.lastPathComponent {
        case "config.json", "preprocessor_config.json", "processor_config.json",
             "model.safetensors.index.json":
            return true
        default:
            return url.pathExtension == "safetensors"
        }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}
