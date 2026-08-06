#if canImport(FoundationModels)
import Foundation

@MainActor
@available(macOS 27.0, *)
public final class AFMFoundationActiveGenerationCoordinator<ProviderID: Hashable & Sendable> {
    public struct Generation: Equatable, Sendable {
        public let id: UUID
        public let provider: ProviderID

        public init(id: UUID, provider: ProviderID) {
            self.id = id
            self.provider = provider
        }
    }

    public struct BeginResult: Equatable, Sendable {
        public let generation: Generation
        public let replacedProvider: ProviderID?

        public init(generation: Generation, replacedProvider: ProviderID?) {
            self.generation = generation
            self.replacedProvider = replacedProvider
        }
    }

    private(set) public var activeGeneration: Generation?

    public var activeProvider: ProviderID? {
        activeGeneration?.provider
    }

    public init() {}

    public func begin(provider: ProviderID) -> BeginResult {
        let replacedProvider = activeGeneration?.provider
        let generation = Generation(id: UUID(), provider: provider)
        activeGeneration = generation
        return BeginResult(generation: generation, replacedProvider: replacedProvider)
    }

    public func isActive(_ generation: Generation) -> Bool {
        activeGeneration == generation
    }

    @discardableResult
    public func finish(_ generation: Generation) -> Bool {
        guard isActive(generation) else { return false }
        activeGeneration = nil
        return true
    }

    @discardableResult
    public func cancelActiveGeneration() -> ProviderID? {
        defer { activeGeneration = nil }
        return activeGeneration?.provider
    }

    public func reset() {
        activeGeneration = nil
    }
}
#endif
