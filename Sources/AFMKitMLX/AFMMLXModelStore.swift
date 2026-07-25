import Foundation
import AFMKitCore

/// Shared MLX model discovery and validation for AFMKit consumers.
///
/// Hosts may supply model identifiers from their own curated lists or UI
/// registries, then use this store as the single source of truth for whether
/// those identifiers resolve to complete local model assets.
public struct AFMMLXModelStore: Sendable {
    private let resolver: MLXCacheResolver

    public init(resolver: MLXCacheResolver = .init()) {
        self.resolver = resolver
    }

    /// Returns the complete local model directory for an identifier or path.
    public func localDirectory(for modelID: String) -> URL? {
        resolver.localModelDirectory(repoId: modelID)
    }

    /// Returns whether an identifier resolves to complete local model assets.
    public func isAvailableLocally(_ modelID: String) -> Bool {
        localDirectory(for: modelID) != nil
    }

    /// Describes a model using the same assets and capability inference as the
    /// AFMKit MLX provider.
    public func descriptor(for modelID: String) -> AFMModelDescriptor {
        AFMMLXModelDescriptor.describe(modelID: modelID, resolver: resolver)
    }

    /// Returns locally available descriptors from a host-provided candidate
    /// list, preserving order while removing duplicate identifiers.
    public func localDescriptors<S: Sequence>(
        for modelIDs: S
    ) -> [AFMModelDescriptor] where S.Element == String {
        var seen = Set<String>()
        return modelIDs.compactMap { modelID in
            let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  seen.insert(trimmed).inserted,
                  isAvailableLocally(trimmed) else {
                return nil
            }
            return descriptor(for: trimmed)
        }
    }
}
