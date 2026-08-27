import Foundation

package enum AFMDwarfStarModelIdentity {
    package static func advertisedModelID(
        requestedModel: String,
        checkpointPath: String
    ) -> String {
        advertisedModelID(
            requestedModel: requestedModel,
            checkpointPath: checkpointPath,
            requestedPathExists: { FileManager.default.fileExists(atPath: $0) }
        )
    }

    package static func advertisedModelID(
        requestedModel: String,
        checkpointPath: String,
        requestedPathExists: (String) -> Bool
    ) -> String {
        let trimmed = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let expandedPath = (trimmed as NSString).expandingTildeInPath
        let explicitlyLocal = trimmed.hasPrefix("/")
            || trimmed.hasPrefix("./")
            || trimmed.hasPrefix("../")
            || requestedPathExists(expandedPath)
        guard !explicitlyLocal else {
            return URL(fileURLWithPath: checkpointPath).lastPathComponent
        }

        let components = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        let isRepositoryID = components.count == 2
            && components.allSatisfy { !$0.isEmpty }
        return isRepositoryID
            ? trimmed
            : URL(fileURLWithPath: checkpointPath).lastPathComponent
    }
}
