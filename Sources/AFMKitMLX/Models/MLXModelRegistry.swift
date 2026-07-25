import Foundation

struct MLXRegistryEntry: Codable, Hashable {
    let id: String
    let downloadedAt: Date
}

struct MLXModelRegistry {
    private let fileURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        self.fileURL = home.appendingPathComponent(".afm/mlx-model-registry.json")
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }

    func list() -> [MLXRegistryEntry] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return (try? decoder.decode([MLXRegistryEntry].self, from: data)) ?? []
    }

    func listModelIDs() -> [String] {
        list().map(\.id).sorted()
    }

    func registerModel(_ id: String) throws {
        var current = Dictionary(uniqueKeysWithValues: list().map { ($0.id, $0) })
        if current[id] == nil {
            current[id] = MLXRegistryEntry(id: id, downloadedAt: Date())
            try persist(Array(current.values).sorted { $0.id < $1.id })
        }
    }

    func revalidate(using resolver: MLXCacheResolver) throws -> [String] {
        let current = list()
        let valid = current.filter { resolver.localModelDirectory(repoId: $0.id) != nil }
        if valid.count != current.count {
            try persist(valid)
        }
        return valid.map(\.id).sorted()
    }

    private func persist(_ entries: [MLXRegistryEntry]) throws {
        let parent = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let data = try encoder.encode(entries)
        let tmp = fileURL.appendingPathExtension("tmp")
        try data.write(to: tmp, options: .atomic)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            try FileManager.default.removeItem(at: fileURL)
        }
        try FileManager.default.moveItem(at: tmp, to: fileURL)
    }
}

