import Foundation

/// AFM metadata attached to Hugging Face snapshot `Progress` instances.
///
/// The official Hub API reports aggregate bytes. AFM augments that progress
/// with the public repository manifest so terminal clients can also identify
/// active files without replacing the Hub's Xet/LFS transport or cache logic.
public enum AFMDownloadProgressUserInfo {
    public static let currentFiles = ProgressUserInfoKey("com.maclocal.afm.download.current-files")
    public static let completedFiles = ProgressUserInfoKey("com.maclocal.afm.download.completed-files")
    public static let totalFiles = ProgressUserInfoKey("com.maclocal.afm.download.total-files")

    struct File: @unchecked Sendable {
        let path: String
        let expectedBytes: Int64
        let destination: URL?
        let progress: Progress
    }

    static func enrich(_ progress: Progress, files: [File]) {
        guard !files.isEmpty else { return }
        var completed = 0
        var completedBytes: Int64 = 0
        var active: [String] = []
        var firstPending: String?
        for file in files {
            let path = file.path
            let child = file.progress
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
                if firstPending == nil { firstPending = path }
                if child.completedUnitCount > 0 { active.append(path) }
            }
            completedBytes += min(child.completedUnitCount, file.expectedBytes)
        }
        if active.isEmpty, let firstPending { active = [firstPending] }
        progress.completedUnitCount = min(completedBytes, progress.totalUnitCount)
        progress.setUserInfoObject(active, forKey: currentFiles)
        progress.setUserInfoObject(completed, forKey: completedFiles)
        progress.setUserInfoObject(files.count, forKey: totalFiles)
    }
}
