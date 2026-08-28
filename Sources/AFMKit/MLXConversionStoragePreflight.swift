import AFMKitMLX
import Foundation

/// Application-level storage policy for AFMKit checkpoint conversion.
/// Conversion math and checkpoint ownership remain in AFMKit; maclocal-api
/// validates the local source and destination volume before starting a job.
public enum MLXConversionStoragePreflight {
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
        case capacityUnavailable(String)
        case insufficientCapacity(required: Int64, available: Int64, volume: String)

        public var errorDescription: String? {
            switch self {
            case .missingLocalSource(let message), .unsafeOutput(let message),
                 .capacityUnavailable(let message):
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
        capacity: ((URL) throws -> Int64?)? = nil
    ) throws -> Report {
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
        guard outputURL != sourceURL,
              !outputURL.path.hasPrefix(sourceURL.path + "/")
        else {
            throw PreflightError.unsafeOutput(
                "Conversion output must differ from and cannot be inside the source checkpoint.")
        }

        let probe = try nearestExistingDirectory(to: outputURL)
        let resolvedProbe = probe.resolvingSymlinksInPath()
        guard resolvedProbe != resolvedSource,
              !resolvedProbe.path.hasPrefix(resolvedSource.path + "/")
        else {
            throw PreflightError.unsafeOutput(
                "Conversion output resolves inside the source checkpoint.")
        }

        let required = inspection.requiredDestinationFreeBytes
        let available: Int64?
        if let capacity {
            available = try capacity(probe)
        } else {
            let values = try probe.resourceValues(forKeys: [
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
        if let required {
            guard let available else {
                throw PreflightError.capacityUnavailable(
                    "Cannot determine free capacity for conversion destination \(probe.path).")
            }
            guard available >= required else {
                throw PreflightError.insufficientCapacity(
                    required: required, available: available, volume: probe.path)
            }
        }
        return Report(
            source: sourceURL,
            output: outputURL,
            capacityProbe: probe,
            availableBytes: available,
            requiredBytes: required)
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
