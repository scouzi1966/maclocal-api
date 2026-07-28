import Foundation

public struct AFMMLXDownloadedModelRegistryLoad<Model> {
    public let models: [Model]
    public let originalCount: Int

    public var removedCount: Int {
        max(0, originalCount - models.count)
    }

    public init(models: [Model], originalCount: Int) {
        self.models = models
        self.originalCount = originalCount
    }
}

public struct AFMMLXDownloadedModelRegistryStore {
    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public func save<Model: Encodable>(_ models: [Model], forKey key: String) throws {
        let data = try JSONEncoder().encode(models)
        defaults.set(data, forKey: key)
    }

    public func load<Model: Decodable>(_ type: Model.Type, forKey key: String) throws -> [Model] {
        guard let data = defaults.data(forKey: key) else {
            return []
        }

        return try JSONDecoder().decode([Model].self, from: data)
    }

    public func loadCleaned<Model: Decodable>(
        _ type: Model.Type,
        forKey key: String,
        id: (Model) -> String,
        displayName: (Model) -> String,
        isCurated: (String) -> Bool
    ) throws -> AFMMLXDownloadedModelRegistryLoad<Model> {
        let decoded = try load(type, forKey: key)
        let cleaned = AFMMLXModelStore.cleanedPersistedModelRecords(
            decoded,
            id: id,
            displayName: displayName,
            isCurated: isCurated
        )

        return AFMMLXDownloadedModelRegistryLoad(
            models: cleaned,
            originalCount: decoded.count
        )
    }
}
