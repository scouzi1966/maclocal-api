import AFMKitMLX
import CryptoKit
import Foundation

/// Application-level storage policy for AFMKit checkpoint conversion.
/// Conversion math and checkpoint ownership remain in AFMKit; maclocal-api
/// validates the local source and destination volume before starting a job.
public enum MLXConversionStoragePreflight {
    private static let resumeAtomicMarginBytes: Int64 = 64_000_000_000

    public struct PathReport: Sendable, Equatable {
        public let source: URL
        public let output: URL
        public let capacityProbe: URL
    }

    public struct Report: Sendable, Equatable {
        public let source: URL
        public let output: URL
        public let capacityProbe: URL
        public let availableBytes: Int64?
        public let requiredBytes: Int64?
    }

    public enum PreflightError: LocalizedError, Equatable {
        case missingLocalSource(String)
        case unsafeOutput(String)
        case invalidOption(String)
        case capacityUnavailable(String)
        case insufficientCapacity(required: Int64, available: Int64, volume: String)

        public var errorDescription: String? {
            switch self {
            case .missingLocalSource(let message), .unsafeOutput(let message),
                 .invalidOption(let message), .capacityUnavailable(let message):
                message
            case .insufficientCapacity(let required, let available, let volume):
                "Conversion requires at least \(Self.bytes(required)) free at \(volume), but only \(Self.bytes(available)) is available."
            }
        }

        private static func bytes(_ value: Int64) -> String {
            ByteCountFormatter.string(fromByteCount: value, countStyle: .decimal)
        }
    }

    public static func validate(
        source: URL,
        output: URL,
        inspection: AFMMLXCheckpointConverter.Inspection,
        overwrite: Bool = false,
        capacity: ((URL) throws -> Int64?)? = nil
    ) throws -> Report {
        let paths = try validateLocalPaths(source: source, output: output)
        let required = try effectiveRequiredBytes(
            output: paths.output,
            inspection: inspection,
            overwrite: overwrite)
        guard let required else {
            return Report(
                source: paths.source,
                output: paths.output,
                capacityProbe: paths.capacityProbe,
                availableBytes: nil,
                requiredBytes: nil)
        }

        let available: Int64?
        if let capacity {
            available = try capacity(paths.capacityProbe)
        } else {
            let values = try paths.capacityProbe.resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey,
                .volumeAvailableCapacityKey,
            ])
            if let important = values.volumeAvailableCapacityForImportantUsage {
                available = important
            } else if let ordinary = values.volumeAvailableCapacity {
                available = Int64(ordinary)
            } else {
                available = nil
            }
        }
        guard let available else {
            throw PreflightError.capacityUnavailable(
                "Cannot determine free capacity for conversion destination \(paths.capacityProbe.path).")
        }
        guard available >= required else {
            throw PreflightError.insufficientCapacity(
                required: required, available: available, volume: paths.capacityProbe.path)
        }
        return Report(
            source: paths.source,
            output: paths.output,
            capacityProbe: paths.capacityProbe,
            availableBytes: available,
            requiredBytes: required)
    }

    public static func validateLocalPaths(source: URL, output: URL) throws -> PathReport {
        let fm = FileManager.default
        let sourceURL = source.standardizedFileURL
        let outputURL = output.standardizedFileURL
        var isDirectory: ObjCBool = false
        guard sourceURL.isFileURL,
              fm.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw PreflightError.missingLocalSource(
                "--source must be an existing local checkpoint directory; automatic model download is disabled for conversion.")
        }
        guard outputURL.isFileURL else {
            throw PreflightError.unsafeOutput(
                "--output must be a local filesystem directory.")
        }
        let resolvedSource = sourceURL.resolvingSymlinksInPath()
        let resolvedOutput = outputURL.resolvingSymlinksInPath()
        guard !contains(resolvedSource, resolvedOutput),
              !contains(resolvedOutput, resolvedSource)
        else {
            throw PreflightError.unsafeOutput(
                "Conversion source and output must be separate directories; neither may contain the other, including through symlinks.")
        }

        let probe = try nearestExistingDirectory(to: outputURL)
        return PathReport(
            source: sourceURL,
            output: outputURL,
            capacityProbe: probe)
    }

    public static func validateProfileName(_ profile: String?) throws {
        guard let profile else { return }
        let supported = Set(
            DeepseekV4CheckpointConverter.Profile.allCases.map(\.rawValue)
                + GLM5NextCheckpointConverter.Profile.allCases.map(\.rawValue))
        guard supported.contains(profile) else {
            throw PreflightError.invalidOption(
                "Unknown conversion profile '\(profile)'. Expected one of: \(supported.sorted().joined(separator: ", ")).")
        }
    }

    public static func validateTemplateFile(_ path: String?) throws -> URL? {
        guard let path else { return nil }
        guard !path.isEmpty else {
            throw PreflightError.missingLocalSource(
                "--template-gguf must name an existing local GGUF file.")
        }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue
        else {
            throw PreflightError.missingLocalSource(
                "--template-gguf must name an existing local GGUF file.")
        }
        return url
    }

    private static func effectiveRequiredBytes(
        output: URL,
        inspection: AFMMLXCheckpointConverter.Inspection,
        overwrite: Bool
    ) throws -> Int64? {
        guard let initial = inspection.requiredDestinationFreeBytes else { return nil }
        guard !overwrite,
              let completed = try verifiedCompletedBytes(
                output: output, sourceRevision: inspection.sourceRevision),
              completed > 0
        else { return initial }
        let remainingEstimate = max(0, (inspection.estimatedOutputBytes ?? 0) - completed)
        return max(
            0,
            max(initial - completed, remainingEstimate + resumeAtomicMarginBytes))
    }

    private static func verifiedCompletedBytes(
        output: URL,
        sourceRevision: String?
    ) throws -> Int64? {
        let stateURL = output.appendingPathComponent(".afm-mlx-conversion.json")
        guard FileManager.default.fileExists(atPath: stateURL.path),
              let object = try JSONSerialization.jsonObject(
                with: Data(contentsOf: stateURL)) as? [String: Any],
              object["sourceRevision"] as? String == sourceRevision,
              let completed = object["completed"] as? [String: Any]
        else { return nil }
        var seen = Set<String>()
        var total: Int64 = 0
        for raw in completed.values {
            guard let item = raw as? [String: Any],
                  let name = item["outputFile"] as? String,
                  URL(fileURLWithPath: name).lastPathComponent == name,
                  let expectedSize = (item["outputSize"] as? NSNumber)?.int64Value,
                  let expectedSHA256 = item["outputSHA256"] as? String,
                  expectedSize >= 0,
                  expectedSHA256.count == 64,
                  expectedSHA256.allSatisfy(\.isHexDigit),
                  seen.insert(name).inserted
            else { return nil }
            let url = output.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path),
                  Int64(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1)
                    == expectedSize,
                  try sha256File(url) == expectedSHA256
            else { return nil }
            let sum = total.addingReportingOverflow(expectedSize)
            guard !sum.overflow else { return nil }
            total = sum.partialValue
        }
        return total
    }

    private static func sha256File(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let chunk = try handle.read(upToCount: 4 * 1024 * 1024), !chunk.isEmpty {
            digest.update(data: chunk)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func contains(_ directory: URL, _ candidate: URL) -> Bool {
        directory == candidate || candidate.path.hasPrefix(directory.path + "/")
    }

    private static func nearestExistingDirectory(to output: URL) throws -> URL {
        let fm = FileManager.default
        var candidate = output
        while !fm.fileExists(atPath: candidate.path) {
            let parent = candidate.deletingLastPathComponent()
            guard parent.path != candidate.path else {
                throw PreflightError.capacityUnavailable(
                    "Cannot find an existing parent directory for \(output.path).")
            }
            candidate = parent
        }
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw PreflightError.capacityUnavailable(
                "Capacity probe path is not a directory: \(candidate.path).")
        }
        return candidate
    }
}
