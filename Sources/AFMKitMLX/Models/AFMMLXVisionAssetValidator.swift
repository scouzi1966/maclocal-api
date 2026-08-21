import Foundation
import MLXVLM

public final class AFMMLXVisionAssetValidator: @unchecked Sendable {
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
        let config = jsonObject(
            at: modelDirectory.appendingPathComponent("config.json")
        ) ?? [:]
        let isConditionalGeneration = conditionalGenerationArchitecture(in: config)
        let hasVisionConfiguration = config["vision_config"] is [String: Any]
        let hasImageTokenIdentifiers = integer(config["image_token_id"]) != nil
            && integer(config["vision_start_token_id"]) != nil
            && integer(config["vision_end_token_id"]) != nil
        let processorClass = selectedProcessorClass(
            modelDirectory: modelDirectory,
            canonicalModelType: architecture.canonicalModelType
        )
        let visionTensorCount = visionTensorNames(in: modelDirectory).count

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

    private static func visionTensorNames(in modelDirectory: URL) -> Set<String> {
        let indexURL = modelDirectory.appendingPathComponent(
            "model.safetensors.index.json"
        )
        if FileManager.default.fileExists(atPath: indexURL.path) {
            guard let index = jsonObject(at: indexURL),
                  let weightMap = index["weight_map"] as? [String: Any]
            else { return [] }

            let shardNames = Set(weightMap.values.compactMap { $0 as? String })
            guard !shardNames.isEmpty,
                  shardNames.allSatisfy({ shardName in
                      FileManager.default.fileExists(
                          atPath: modelDirectory.appendingPathComponent(shardName).path
                      )
                  })
            else { return [] }
            return Set(weightMap.keys.filter(isVisionTensorName))
        }

        guard let files = try? FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        ) else { return [] }
        let weightFiles = files.filter {
            $0.pathExtension == "safetensors"
                && $0.lastPathComponent != "mtp.safetensors"
        }
        var names = Set<String>()
        for file in weightFiles {
            guard let tensorNames = safetensorNames(in: file) else { return [] }
            names.formUnion(tensorNames.filter(isVisionTensorName))
        }
        return names
    }

    private static func isVisionTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") || name.hasPrefix("model.visual")
    }

    private static func safetensorNames(in url: URL) -> Set<String>? {
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
              let object = try? JSONSerialization.jsonObject(with: header)
                as? [String: Any]
        else { return nil }
        return Set(object.keys.filter { $0 != "__metadata__" })
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private static func integer(_ value: Any?) -> Int? {
        if let value = value as? Int { return value }
        return (value as? NSNumber)?.intValue
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
