import Foundation

public enum AFMMLXSpeculativeRuntimeResourceResolver {
    public struct MTPQuantization: Equatable, Sendable {
        public let groupSize: Int
        public let bits: Int
        public let mode: String

        public init(groupSize: Int, bits: Int, mode: String) {
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
        }
    }

    public static let mtpSidecarFilename = "mtp.safetensors"
    public static let repositorySidecarFilename = "model.safetensors"

    /// Resolve the separately published Qwen 3.8 MTP head that matches the
    /// base checkpoint's quantization. Detection comes from config.json rather
    /// than the repository name so imported and renamed checkpoints work too.
    public static func automaticMTPRepositoryID(modelDirectory: URL) -> String? {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = root["text_config"] as? [String: Any],
              (root["model_type"] as? String) == "qwen3_5",
              (text["model_type"] as? String) == "qwen3_5_text",
              (text["mtp_num_hidden_layers"] as? NSNumber)?.intValue ?? 0 > 0,
              (text["hidden_size"] as? NSNumber)?.intValue == 5_120,
              (text["num_hidden_layers"] as? NSNumber)?.intValue == 64
        else {
            return nil
        }

        let quantization = (root["quantization"] as? [String: Any])
            ?? (root["quantization_config"] as? [String: Any])
        let variant: String
        if let mode = quantization?["mode"] as? String {
            switch mode.lowercased() {
            case "mxfp4", "mxfp8", "nvfp4":
                variant = mode.lowercased()
            case "affine":
                guard let bits = (quantization?["bits"] as? NSNumber)?.intValue,
                      bits == 4 || bits == 8 else { return nil }
                variant = "\(bits)bit"
            default:
                return nil
            }
        } else {
            variant = "bf16"
        }

        return "mlx-community/Qwen3.8-27B-MTP-\(variant)"
    }

    /// Read the quantization layout used by a standalone MTP checkpoint.
    /// The loader must use these values; a suffix such as `4bit` is not enough
    /// to infer the group size or quantization mode.
    public static func mtpQuantization(resourceDirectory: URL) -> MTPQuantization? {
        let configURL = resourceDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quantization = (root["quantization"] as? [String: Any])
                ?? (root["quantization_config"] as? [String: Any]),
              let groupSize = (quantization["group_size"] as? NSNumber)?.intValue,
              let bits = (quantization["bits"] as? NSNumber)?.intValue,
              let mode = quantization["mode"] as? String
        else {
            return nil
        }
        return MTPQuantization(groupSize: groupSize, bits: bits, mode: mode.lowercased())
    }

    public static func currentLoadedModelDirectory(
        loadedModelRepoID: String?,
        repositoryDirectory: (String) -> URL?
    ) -> URL? {
        guard let loadedModelRepoID else {
            return nil
        }

        let trimmed = loadedModelRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }

        if trimmed.hasPrefix("/") {
            return URL(fileURLWithPath: trimmed)
        }

        return repositoryDirectory(trimmed)
    }

    public static func mtpSidecarPath(
        modelDirectory: URL?,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) -> String? {
        guard let modelDirectory else {
            return nil
        }

        let path = modelDirectory
            .appendingPathComponent(mtpSidecarFilename)
            .path

        return fileExists(path) ? path : nil
    }

    /// Resolve either the legacy in-model sidecar name or the filename used by
    /// the standalone Qwen 3.8 MTP repositories.
    public static func mtpSidecarPath(
        resourceDirectory: URL?,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) -> String? {
        guard let resourceDirectory else { return nil }
        for filename in [mtpSidecarFilename, repositorySidecarFilename] {
            let path = resourceDirectory.appendingPathComponent(filename).path
            if fileExists(path) { return path }
        }
        return nil
    }
}
