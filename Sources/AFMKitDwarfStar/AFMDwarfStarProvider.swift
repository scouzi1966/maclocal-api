import Foundation
import AFMKitCore
import CDwarfStar

public struct AFMDwarfStarProviderFactory: AFMProviderFactory {
    public static let providerID: AFMProviderID = "dwarfstar"

    public init() {}

    public var descriptor: AFMProviderDescriptor {
        AFMProviderDescriptor(
            id: Self.providerID,
            displayName: "DwarfStar",
            privacyBoundary: .device,
            configurationKeys: [
                "modelPath",
                "templateGGUF",
                "projectionMetadataPath",
                "externalMapGGUF",
                "contextWindow",
                "prefillChunk",
                "powerPercent",
                "dsparkSupportPath",
                "dsparkDraftTokens",
                "dsparkConfidenceThreshold",
                "dsparkStrict",
                "enablePrefixCaching",
                "maxConcurrent"
            ],
            metadata: [
                "runtime": .string("in-process-ds4"),
                "execution": .string("fixed-metal-schedule")
            ]
        )
    }

    public func modelDescriptors() async throws -> [AFMModelDescriptor] {
        []
    }

    public func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel {
        let modelPath = configuration.string("modelPath") ?? id.rawValue
        guard !modelPath.isEmpty else {
            throw AFMError.invalidRequest("DwarfStar requires a model or checkpoint path.")
        }
        return AnyAFMModel(
            AFMDwarfStarModel(
                modelID: id,
                modelPath: modelPath,
                templateGGUF: configuration.string("templateGGUF"),
                projectionMetadataPath: configuration.string("projectionMetadataPath"),
                externalMapGGUF: configuration.boolean("externalMapGGUF") ?? false,
                contextWindow: configuration.integer("contextWindow") ?? 32_768,
                prefillChunk: configuration.integer("prefillChunk") ?? 0,
                powerPercent: configuration.integer("powerPercent") ?? 100,
                dsparkSupportPath: configuration.string("dsparkSupportPath"),
                dsparkDraftTokens: configuration.integer("dsparkDraftTokens") ?? 5,
                dsparkConfidenceThreshold: configuration.number("dsparkConfidenceThreshold") ?? 0.7,
                dsparkStrict: configuration.boolean("dsparkStrict") ?? false,
                enablePrefixCaching: configuration.boolean("enablePrefixCaching") ?? false,
                maxConcurrent: configuration.integer("maxConcurrent") ?? 1
            )
        )
    }
}

public final class AFMDwarfStarModel: AFMModel, @unchecked Sendable {
    public let descriptor: AFMModelDescriptor

    private let modelPath: String
    private let templateGGUF: String?
    private let projectionMetadataPath: String?
    private let externalMapGGUF: Bool
    private let contextWindow: Int
    private let prefillChunk: Int
    private let powerPercent: Int
    private let dsparkSupportPath: String?
    private let dsparkDraftTokens: Int
    private let dsparkConfidenceThreshold: Double
    private let dsparkStrict: Bool
    private let enablePrefixCaching: Bool
    private let maxConcurrent: Int
    private let runtime: AFMDwarfStarRuntimeCoordinator

    public init(
        modelID: AFMModelID,
        modelPath: String,
        templateGGUF: String? = nil,
        projectionMetadataPath: String? = nil,
        externalMapGGUF: Bool = false,
        contextWindow: Int = 32_768,
        prefillChunk: Int = 0,
        powerPercent: Int = 100,
        dsparkSupportPath: String? = nil,
        dsparkDraftTokens: Int = 5,
        dsparkConfidenceThreshold: Double = 0.7,
        dsparkStrict: Bool = false,
        enablePrefixCaching: Bool = false,
        maxConcurrent: Int = 1,
        runtime: AFMDwarfStarRuntimeCoordinator = .shared
    ) {
        self.modelPath = modelPath
        let modelURL = URL(fileURLWithPath: modelPath)
        let bundledTemplate = modelURL.appendingPathComponent(
            AFMDwarfStarCheckpointCatalog.bundledTemplateFilename,
            isDirectory: false)
        self.templateGGUF = templateGGUF ?? (
            FileManager.default.fileExists(atPath: bundledTemplate.path)
                ? bundledTemplate.path
                : nil)
        self.projectionMetadataPath = projectionMetadataPath
        self.externalMapGGUF = externalMapGGUF
        self.contextWindow = contextWindow
        self.prefillChunk = prefillChunk
        self.powerPercent = powerPercent
        self.dsparkSupportPath = dsparkSupportPath
        self.dsparkDraftTokens = max(1, min(16, dsparkDraftTokens))
        self.dsparkConfidenceThreshold = max(0, min(1, dsparkConfidenceThreshold))
        self.dsparkStrict = dsparkStrict
        self.enablePrefixCaching = enablePrefixCaching
        self.maxConcurrent = max(1, maxConcurrent)
        self.runtime = runtime
        self.descriptor = AFMModelDescriptor(
            providerID: AFMDwarfStarProviderFactory.providerID,
            modelID: modelID,
            displayName: URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent,
            capabilities: [.text, .streaming, .prefixCaching],
            contextWindow: contextWindow,
            privacyBoundary: .device,
            requiresNetwork: false,
            metadata: [
                "runtime": .string("dwarfstar"),
                "backend": .string("metal"),
                "modelPath": .string(modelPath),
                "dsparkEnabled": .bool(dsparkSupportPath != nil),
                "dsparkDraftTokens": .integer(max(1, min(16, dsparkDraftTokens))),
                "dsparkConfidenceThreshold": .number(max(0, min(1, dsparkConfidenceThreshold))),
                "dsparkStrict": .bool(dsparkStrict),
                "enablePrefixCaching": .bool(enablePrefixCaching),
                "maxConcurrent": .integer(max(1, maxConcurrent))
            ]
        )
    }

    public func availability() async -> AFMModelAvailability {
        FileManager.default.fileExists(atPath: modelPath)
            ? .available
            : .unavailable(reason: "Model or checkpoint does not exist at \(modelPath)")
    }

    public func load(
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> AFMModelDescriptor {
        progress?(0)
        try await runtime.load(
            modelPath: modelPath,
            templateGGUF: templateGGUF,
            projectionMetadataPath: projectionMetadataPath,
            externalMapGGUF: externalMapGGUF,
            contextWindow: contextWindow,
            prefillChunk: prefillChunk,
            powerPercent: powerPercent,
            dsparkSupportPath: dsparkSupportPath,
            dsparkDraftTokens: dsparkDraftTokens,
            dsparkConfidenceThreshold: dsparkConfidenceThreshold,
            dsparkStrict: dsparkStrict,
            enablePrefixCaching: enablePrefixCaching,
            maxConcurrent: maxConcurrent
        )
        progress?(1)
        return descriptor
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        _ = try await load(progress: nil)
        let result = try await runtime.generate(request: request) { _, _ in }
        return result.response(modelID: descriptor.modelID.rawValue)
    }

    public func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await load(progress: nil)
                    let result = try await runtime.generate(request: request) { text, count in
                        continuation.yield(
                            .responseText(action: .append, text: text, tokenCount: count)
                        )
                    }
                    continuation.yield(.usage(result.usage))
                    continuation.yield(.metadata(result.metadata))
                    continuation.yield(.completed(result.finishReason))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.completed(.cancelled))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func unload() async {
        await runtime.unload(modelPath: modelPath)
    }
}

private extension AFMProviderConfiguration {
    func string(_ key: String) -> String? {
        guard case .string(let value) = values[key] else { return nil }
        return value
    }

    func integer(_ key: String) -> Int? {
        guard case .integer(let value) = values[key] else { return nil }
        return value
    }

    func boolean(_ key: String) -> Bool? {
        guard case .bool(let value) = values[key] else { return nil }
        return value
    }

    func number(_ key: String) -> Double? {
        switch values[key] {
        case .number(let value): return value
        case .integer(let value): return Double(value)
        default: return nil
        }
    }
}
