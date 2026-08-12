import Foundation
import os

/// AFM metadata attached to Hugging Face snapshot `Progress` instances.
///
/// The official Hub API reports aggregate bytes. AFM augments that progress
/// with the public repository manifest so terminal clients can also identify
/// active files without replacing the Hub's Xet/LFS transport or cache logic.
public enum AFMDownloadProgressUserInfo {
    public static let currentFiles = ProgressUserInfoKey("com.maclocal.afm.download.current-files")
    public static let completedFiles = ProgressUserInfoKey("com.maclocal.afm.download.completed-files")
    public static let totalFiles = ProgressUserInfoKey("com.maclocal.afm.download.total-files")
    public static let currentTransports = ProgressUserInfoKey("com.maclocal.afm.download.current-transports")

    struct File: @unchecked Sendable {
        let path: String
        let expectedBytes: Int64
        let destination: URL?
        let progress: Progress
        let transport: OSAllocatedUnfairLock<String>

        init(
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

        func setTransport(_ value: String) {
            transport.withLock { $0 = value }
        }

        var currentTransport: String {
            transport.withLock { $0 }
        }
    }

    static func enrich(_ progress: Progress, files: [File]) {
        guard !files.isEmpty else { return }
        var completed = 0
        var completedBytes: Int64 = 0
        var activePairs: [(path: String, transport: String)] = []
        var firstPendingPair: (path: String, transport: String)?
        for file in files {
            let path = file.path
            let child = file.progress
            let transport = file.currentTransport
            if let destination = file.destination,
               let attributes = try? FileManager.default.attributesOfItem(atPath: destination.path),
               let size = (attributes[.size] as? NSNumber)?.int64Value,
               size > child.completedUnitCount {
                child.completedUnitCount = min(size, file.expectedBytes)
            }
            if child.totalUnitCount > 0,
               child.completedUnitCount >= child.totalUnitCount {
                completed += 1
            } else {
                if firstPendingPair == nil {
                    firstPendingPair = (path, transport)
                }
                if child.completedUnitCount > 0 {
                    activePairs.append((path, transport))
                }
            }
            completedBytes += min(child.completedUnitCount, file.expectedBytes)
        }
        if activePairs.isEmpty, let firstPendingPair {
            activePairs = [firstPendingPair]
        }
        let active = activePairs.map(\.path)
        let activeTransports = activePairs.map(\.transport)
        progress.completedUnitCount = min(completedBytes, progress.totalUnitCount)
        progress.setUserInfoObject(active, forKey: currentFiles)
        progress.setUserInfoObject(activeTransports, forKey: currentTransports)
        progress.setUserInfoObject(completed, forKey: completedFiles)
        progress.setUserInfoObject(files.count, forKey: totalFiles)
    }
}
