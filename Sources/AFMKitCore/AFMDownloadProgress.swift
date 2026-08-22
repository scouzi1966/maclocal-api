import Foundation
import os

/// Provider-neutral metadata attached to aggregate model download progress.
public enum AFMDownloadProgressUserInfo {
    public static let currentFiles = ProgressUserInfoKey("com.maclocal.afm.download.current-files")
    public static let completedFiles = ProgressUserInfoKey("com.maclocal.afm.download.completed-files")
    public static let totalFiles = ProgressUserInfoKey("com.maclocal.afm.download.total-files")
    public static let currentTransports = ProgressUserInfoKey("com.maclocal.afm.download.current-transports")

    public struct File: @unchecked Sendable {
        public let path: String
        public let expectedBytes: Int64
        public let destination: URL?
        public let progress: Progress
        private let transport: OSAllocatedUnfairLock<String>

        public init(
            path: String,
            expectedBytes: Int64,
            destination: URL?,
            progress: Progress,
            transport: String = "pending"
        ) {
            self.path = path
            self.expectedBytes = expectedBytes
            self.destination = destination
            self.progress = progress
            self.transport = OSAllocatedUnfairLock(initialState: transport)
        }

        public func setTransport(_ value: String) {
            transport.withLock { $0 = value }
        }

        public var currentTransport: String {
            transport.withLock { $0 }
        }
    }

    public static func enrich(_ progress: Progress, files: [File]) {
        guard !files.isEmpty else { return }
        var completed = 0
        var completedBytes: Int64 = 0
        var activePairs: [(path: String, transport: String)] = []
        var firstPendingPair: (path: String, transport: String)?
        for file in files {
            let child = file.progress
            if let destination = file.destination,
               let attributes = try? FileManager.default.attributesOfItem(atPath: destination.path),
               let size = (attributes[.size] as? NSNumber)?.int64Value,
               size > child.completedUnitCount {
                child.completedUnitCount = min(size, file.expectedBytes)
            }
            if child.totalUnitCount > 0, child.completedUnitCount >= child.totalUnitCount {
                completed += 1
            } else {
                let pair = (file.path, file.currentTransport)
                if firstPendingPair == nil { firstPendingPair = pair }
                if child.completedUnitCount > 0 { activePairs.append(pair) }
            }
            completedBytes += min(child.completedUnitCount, file.expectedBytes)
        }
        if activePairs.isEmpty, let firstPendingPair { activePairs = [firstPendingPair] }
        progress.completedUnitCount = min(completedBytes, progress.totalUnitCount)
        progress.setUserInfoObject(activePairs.map(\.path), forKey: currentFiles)
        progress.setUserInfoObject(activePairs.map(\.transport), forKey: currentTransports)
        progress.setUserInfoObject(completed, forKey: completedFiles)
        progress.setUserInfoObject(files.count, forKey: totalFiles)
    }
}
