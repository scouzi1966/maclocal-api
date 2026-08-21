import Foundation

/// Provider-specific values supplied by an application at model construction time.
public struct AFMProviderConfiguration: Hashable, Sendable {
    public var values: [String: AFMJSONValue]

    public init(values: [String: AFMJSONValue] = [:]) {
        self.values = values
    }
}

public protocol AFMModel: Sendable {
    var descriptor: AFMModelDescriptor { get }
    func availability() async -> AFMModelAvailability
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor
    func respond(to request: AFMRequest) async throws -> AFMModelResponse
    func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error>
    func unload() async
}

/// Optional capability for providers that expose their model tokenizer.
///
/// Tokenization is intentionally separate from `AFMModel`: Apple's Foundation
/// Models API does not expose token IDs, while local providers such as MLX do.
public protocol AFMTextTokenizing: Sendable {
    func tokenize(text: String) async throws -> [Int]
}

public extension AFMModel {
    func load() async throws -> AFMModelDescriptor {
        try await load(progress: nil)
    }

    func unload() async {}
}

public struct AnyAFMModel: AFMModel, Sendable {
    public let rawTextGenerator: AnyAFMRawTextGenerator?
    public let generationAdmitter: AnyAFMGenerationAdmitter?

    private let descriptorValue: AFMModelDescriptor
    private let availabilityOperation: @Sendable () async -> AFMModelAvailability
    private let loadOperation:
        @Sendable ((@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor
    private let respondOperation: @Sendable (AFMRequest) async throws -> AFMModelResponse
    private let streamOperation:
        @Sendable (AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error>
    private let unloadOperation: @Sendable () async -> Void

    public init<Model: AFMModel>(_ model: Model) {
        if let generator = model as? any AFMRawTextGenerating {
            rawTextGenerator = AnyAFMRawTextGenerator(generator)
        } else {
            rawTextGenerator = nil
        }
        if let admitter = model as? any AFMGenerationAdmitting {
            generationAdmitter = AnyAFMGenerationAdmitter { timeout in
                try await admitter.admitGeneration(timeout: timeout)
            }
        } else {
            generationAdmitter = nil
        }
        descriptorValue = model.descriptor
        availabilityOperation = { await model.availability() }
        loadOperation = { progress in try await model.load(progress: progress) }
        respondOperation = { request in try await model.respond(to: request) }
        streamOperation = { request in model.streamResponse(to: request) }
        unloadOperation = { await model.unload() }
    }

    public var descriptor: AFMModelDescriptor { descriptorValue }

    public func availability() async -> AFMModelAvailability {
        await availabilityOperation()
    }

    public func load(
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> AFMModelDescriptor {
        try await loadOperation(progress)
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        try await respondOperation(request)
    }

    public func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        streamOperation(request)
    }

    public func unload() async {
        await unloadOperation()
    }
}

public protocol AFMProviderFactory: Sendable {
    var descriptor: AFMProviderDescriptor { get }
    func modelDescriptors() async throws -> [AFMModelDescriptor]
    func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel
}

public struct AnyAFMProviderFactory: AFMProviderFactory, Sendable {
    private let descriptorValue: AFMProviderDescriptor
    private let descriptorsOperation: @Sendable () async throws -> [AFMModelDescriptor]
    private let makeOperation:
        @Sendable (AFMModelID, AFMProviderConfiguration) throws -> AnyAFMModel

    public init<Factory: AFMProviderFactory>(_ factory: Factory) {
        descriptorValue = factory.descriptor
        descriptorsOperation = { try await factory.modelDescriptors() }
        makeOperation = { id, configuration in
            try factory.makeModel(id: id, configuration: configuration)
        }
    }

    public init(
        descriptor: AFMProviderDescriptor,
        modelDescriptors:
            @escaping @Sendable () async throws -> [AFMModelDescriptor],
        makeModel:
            @escaping @Sendable (AFMModelID, AFMProviderConfiguration) throws -> AnyAFMModel
    ) {
        descriptorValue = descriptor
        descriptorsOperation = modelDescriptors
        makeOperation = makeModel
    }

    public var descriptor: AFMProviderDescriptor { descriptorValue }

    public func modelDescriptors() async throws -> [AFMModelDescriptor] {
        try await descriptorsOperation()
    }

    public func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel {
        try makeOperation(id, configuration)
    }
}

public final class AFMProviderRegistry: @unchecked Sendable {
    public static let shared = AFMProviderRegistry()

    private let lock = NSLock()
    private var factories: [AFMProviderID: AnyAFMProviderFactory] = [:]

    public init() {}

    public func register<Factory: AFMProviderFactory>(_ factory: Factory) throws {
        try register(AnyAFMProviderFactory(factory))
    }

    public func register(_ factory: AnyAFMProviderFactory) throws {
        try lock.withLock {
            let id = factory.descriptor.id
            guard factories[id] == nil else {
                throw AFMError.providerAlreadyRegistered(id)
            }
            factories[id] = factory
        }
    }

    public func replace(_ factory: AnyAFMProviderFactory) {
        lock.withLock {
            factories[factory.descriptor.id] = factory
        }
    }

    @discardableResult
    public func unregister(_ id: AFMProviderID) -> AnyAFMProviderFactory? {
        lock.withLock {
            factories.removeValue(forKey: id)
        }
    }

    public func providerDescriptors() -> [AFMProviderDescriptor] {
        lock.withLock {
            factories.values.map(\.descriptor).sorted {
                $0.id.rawValue < $1.id.rawValue
            }
        }
    }

    public func factory(for id: AFMProviderID) -> AnyAFMProviderFactory? {
        lock.withLock { factories[id] }
    }

    public func makeModel(
        providerID: AFMProviderID,
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration = .init()
    ) throws -> AnyAFMModel {
        guard let factory = factory(for: providerID) else {
            throw AFMError.providerNotRegistered(providerID)
        }
        return try factory.makeModel(id: modelID, configuration: configuration)
    }
}
