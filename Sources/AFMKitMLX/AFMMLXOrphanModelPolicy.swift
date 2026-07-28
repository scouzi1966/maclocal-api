import Foundation
import AFMKitCore

public struct AFMMLXOrphanModelCandidate: Hashable, Sendable {
    public var id: String
    public var name: String
    public var author: String
    public var packageDirectory: URL
    public var sizeBytes: Int64

    public init(
        id: String,
        name: String,
        author: String,
        packageDirectory: URL,
        sizeBytes: Int64
    ) {
        self.id = id
        self.name = name
        self.author = author
        self.packageDirectory = packageDirectory
        self.sizeBytes = sizeBytes
    }
}

public enum AFMMLXOrphanModelPolicy {
    public static func candidate(
        from discovered: AFMMLXDiscoveredModel,
        includeSpecialty: Bool,
        registeredModelIDs: Set<String>,
        isCurated: (String) -> Bool = isCuratedRepositoryID
    ) -> AFMMLXOrphanModelCandidate? {
        let repoID = discovered.id.rawValue
        let isSpecialty = AFMMLXModelStore.isSpecialtyModelIdentifier(repoID)
        guard isSpecialty == includeSpecialty else { return nil }
        guard includeSpecialty || !registeredModelIDs.contains(repoID) else {
            return nil
        }
        guard includeSpecialty || !isCurated(repoID) else {
            return nil
        }

        let parts = repoID.split(separator: "/", maxSplits: 1).map(String.init)
        let author = parts.first ?? ""
        let name = parts.count > 1 ? parts[1] : discovered.descriptor.displayName
        return AFMMLXOrphanModelCandidate(
            id: repoID,
            name: name,
            author: author,
            packageDirectory: discovered.packageDirectory,
            sizeBytes: discovered.sizeBytes
        )
    }

    public static func candidates(
        from discoveredModels: [AFMMLXDiscoveredModel],
        includeSpecialty: Bool,
        registeredModelIDs: Set<String>,
        isCurated: (String) -> Bool = isCuratedRepositoryID
    ) -> [AFMMLXOrphanModelCandidate] {
        discoveredModels
            .compactMap {
                candidate(
                    from: $0,
                    includeSpecialty: includeSpecialty,
                    registeredModelIDs: registeredModelIDs,
                    isCurated: isCurated
                )
            }
            .sorted { $0.sizeBytes > $1.sizeBytes }
    }

    public static func isCuratedRepositoryID(_ repoID: String) -> Bool {
        AFMMLXModelCatalog.availableModels.contains { $0.repoID == repoID }
    }
}
