import Foundation

enum CacheProfileExporter {
    private static let lock = NSLock()

    static func append(record: [String: Any], to path: String) {
        guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) else {
            return
        }

        lock.lock()
        defer { lock.unlock() }

        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }

        let url = URL(fileURLWithPath: path)
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }

        do {
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.write(contentsOf: Data([0x0A]))
        } catch {
            return
        }
    }
}
