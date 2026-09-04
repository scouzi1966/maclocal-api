import AFMKitMLX
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
        verifiedCompletedOutputBytes: Int64 = 0,
        capacity: ((URL) throws -> Int64?)? = nil
    ) throws -> Report {
        let paths = try validateLocalPaths(source: source, output: output)
        let required = try effectiveRequiredBytes(
            inspection: inspection,
            overwrite: overwrite,
            verifiedCompletedOutputBytes: verifiedCompletedOutputBytes)
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
        guard !isFilesystemOrVolumeRoot(resolvedOutput),
              !contains(resolvedSource, resolvedOutput),
              !contains(resolvedOutput, resolvedSource)
        else {
            throw PreflightError.unsafeOutput(
                "Conversion output cannot be a filesystem or volume root, and source/output must be separate directories with neither containing the other, including through symlinks.")
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
                + GLM5NextCheckpointConverter.Profile.allCases.map(\.rawValue)
                + Qwen4ExpCheckpointConverter.Profile.allCases.map(\.rawValue))
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
        inspection: AFMMLXCheckpointConverter.Inspection,
        overwrite: Bool,
        verifiedCompletedOutputBytes: Int64
    ) throws -> Int64? {
        guard let initial = inspection.requiredDestinationFreeBytes else { return nil }
        guard verifiedCompletedOutputBytes >= 0 else {
            throw PreflightError.invalidOption(
                "Provider-verified completed output bytes cannot be negative.")
        }
        guard !overwrite, verifiedCompletedOutputBytes > 0
        else { return initial }
        guard let estimated = inspection.estimatedOutputBytes else { return initial }
        let completed = verifiedCompletedOutputBytes
        let remainingEstimate = max(0, estimated - completed)
        return max(
            0,
            max(initial - completed, remainingEstimate + resumeAtomicMarginBytes))
    }

    private static func contains(_ directory: URL, _ candidate: URL) -> Bool {
        let parent = directory.standardizedFileURL.pathComponents
        let child = candidate.standardizedFileURL.pathComponents
        guard parent.count <= child.count else { return false }
        return Array(child.prefix(parent.count)) == parent
    }

    static func isFilesystemOrVolumeRoot(
        _ url: URL,
        mountedVolumes: [URL]? = nil
    ) -> Bool {
        let resolvedPath = url.standardizedFileURL.resolvingSymlinksInPath().path
        if resolvedPath == "/" { return true }
        let volumes = mountedVolumes ?? FileManager.default.mountedVolumeURLs(
            includingResourceValuesForKeys: nil,
            options: []) ?? []
        return volumes.contains {
            $0.standardizedFileURL.resolvingSymlinksInPath().path == resolvedPath
        }
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
