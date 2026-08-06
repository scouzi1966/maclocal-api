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
                "powerPercent"
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
                powerPercent: configuration.integer("powerPercent") ?? 100
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
                "modelPath": .string(modelPath)
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
            powerPercent: powerPercent
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

public actor AFMDwarfStarRuntimeCoordinator {
    public static let shared = AFMDwarfStarRuntimeCoordinator()

    private var engine: OpaquePointer?
    private var session: OpaquePointer?
    private var loadedModelPath: String?
    private var loadedMappingIdentity: String?
    private var loadedContextWindow = 0

    public init() {}

    isolated deinit {
        if let session { ds4_session_free(session) }
        if let engine { ds4_engine_close(engine) }
    }

    public func load(
        modelPath: String,
        templateGGUF: String? = nil,
        projectionMetadataPath: String? = nil,
        externalMapGGUF: Bool = false,
        contextWindow: Int,
        prefillChunk: Int,
        powerPercent: Int
    ) throws {
        let mappingIdentity = [
            externalMapGGUF ? "external-gguf" : "normal",
            templateGGUF ?? "",
            projectionMetadataPath ?? "",
        ].joined(separator: "|")
        if engine != nil,
           session != nil,
           loadedModelPath == modelPath,
           loadedMappingIdentity == mappingIdentity,
           loadedContextWindow == contextWindow {
            return
        }

        unloadCurrent()

        guard let sourceRoot = AFMDwarfStarRuntime.metalSourceDirectory?.path else {
            throw AFMError.loadingFailed("Bundled DwarfStar Metal sources are missing.")
        }
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw AFMError.loadingFailed("Model or checkpoint does not exist at \(modelPath)")
        }

        var openedEngine: OpaquePointer?
        var error = [CChar](repeating: 0, count: 512)
        let modelURL = URL(fileURLWithPath: modelPath)
        let isDirectory = (try? modelURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory)
            == true
        let projection: AFMDwarfStarProjection?
        if isDirectory {
            guard let templateGGUF, !templateGGUF.isEmpty else {
                throw AFMError.loadingFailed(
                    "An AFM DwarfStar checkpoint requires a metadata template GGUF.")
            }
            let metadataURL = URL(fileURLWithPath: projectionMetadataPath
                ?? modelURL.appendingPathComponent(".afm-dwarfstar-projection.gguf").path)
            projection = try AFMDwarfStarProjection.build(
                checkpointURL: modelURL,
                templateGGUF: URL(fileURLWithPath: templateGGUF),
                metadataOutputURL: metadataURL)
        } else if externalMapGGUF {
            guard let projectionMetadataPath, !projectionMetadataPath.isEmpty else {
                throw AFMError.loadingFailed(
                    "External GGUF mapping requires a projection metadata path.")
            }
            projection = try AFMDwarfStarProjection.buildGGUFAlias(
                ggufURL: modelURL,
                metadataOutputURL: URL(fileURLWithPath: projectionMetadataPath))
        } else {
            projection = nil
        }

        let status: Int32
        if let projection {
            let pathPointers = projection.regions.map { strdup($0.path) }
            defer { pathPointers.forEach { free($0) } }
            let regions = zip(projection.regions, pathPointers).map { region, path in
                ds4_model_map_region(
                    path: UnsafePointer(path),
                    virtual_offset: region.virtualOffset,
                    file_offset: region.fileOffset,
                    length: region.length)
            }
            status = projection.metadataPath.withCString { metadataPointer in
                sourceRoot.withCString { sourceRootPointer in
                    regions.withUnsafeBufferPointer { regionBuffer in
                        afm_ds4_engine_open_mapped(
                            &openedEngine,
                            metadataPointer,
                            projection.virtualSize,
                            regionBuffer.baseAddress,
                            regionBuffer.count,
                            Int32(contextWindow),
                            UInt32(clamping: prefillChunk),
                            Int32(powerPercent),
                            sourceRootPointer,
                            &error,
                            error.count)
                    }
                }
            }
        } else {
            status = modelPath.withCString { modelPathPointer in
                sourceRoot.withCString { sourceRootPointer in
                    afm_ds4_engine_open(
                        &openedEngine,
                        modelPathPointer,
                        Int32(contextWindow),
                        UInt32(clamping: prefillChunk),
                        Int32(powerPercent),
                        sourceRootPointer,
                        &error,
                        error.count)
                }
            }
        }
        guard status == 0, let openedEngine else {
            throw AFMError.loadingFailed(Self.errorText(error))
        }

        var openedSession: OpaquePointer?
        guard ds4_session_create(&openedSession, openedEngine, Int32(contextWindow)) == 0,
              let openedSession else {
            ds4_engine_close(openedEngine)
            throw AFMError.loadingFailed("DwarfStar failed to allocate its inference session.")
        }
        ds4_session_gpu_warmup(openedSession)

        engine = openedEngine
        session = openedSession
        loadedModelPath = modelPath
        loadedMappingIdentity = mappingIdentity
        loadedContextWindow = contextWindow
    }

    public func unload(modelPath: String) {
        guard loadedModelPath == modelPath else { return }
        unloadCurrent()
    }

    func generate(
        request: AFMRequest,
        onText: @Sendable (String, Int) -> Void
    ) throws -> AFMDwarfStarGenerationResult {
        guard let engine, let session else {
            throw AFMError.unavailable("DwarfStar is not loaded.")
        }
        guard request.tools.isEmpty else {
            throw AFMError.unsupportedCapability("tool calling in the DwarfStar runtime")
        }

        var prompt = ds4_tokens()
        afm_ds4_tokens_init(&prompt)
        defer { ds4_tokens_free(&prompt) }

        ds4_chat_begin(engine, &prompt)
        for message in request.messages {
            let text = try Self.textContent(of: message)
            message.role.rawValue.withCString { rolePointer in
                text.withCString { textPointer in
                    ds4_chat_append_message(engine, &prompt, rolePointer, textPointer)
                }
            }
        }
        ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE)
        Self.tracePromptIfRequested(request: request, prompt: prompt)

        var error = [CChar](repeating: 0, count: 512)
        let prefillStart = ContinuousClock.now
        let syncStatus = ds4_session_sync(session, &prompt, &error, error.count)
        guard syncStatus == 0 else {
            if syncStatus == DS4_SESSION_SYNC_INTERRUPTED || Task.isCancelled {
                throw CancellationError()
            }
            throw AFMError.generationFailed(Self.errorText(error))
        }
        let promptSeconds = Self.seconds(since: prefillStart)

        let maximumTokens = max(0, request.options.maximumResponseTokens ?? 512)
        let temperature = Float(request.options.temperature ?? 0)
        let topK = Int32(request.options.topK ?? 0)
        let topP = Float(request.options.topP ?? 1)
        let minP = Float(request.options.minP ?? 0.05)
        var randomState = UInt64(bitPattern: Int64(request.options.seed ?? 0x5eed))
        var generatedText = ""
        var pendingUTF8 = Data()
        var outputTokens = 0
        var finishReason: AFMFinishReason = .length
        let generationStart = ContinuousClock.now

        generationLoop: while outputTokens < maximumTokens {
            try Task.checkCancellation()

            let token = temperature <= 0
                ? ds4_session_argmax(session)
                : ds4_session_sample(session, temperature, topK, topP, minP, &randomState)
            if ds4_token_is_stop_for_think_mode(engine, token, DS4_THINK_NONE) {
                finishReason = .stop
                break
            }

            var byteCount = 0
            guard let bytes = ds4_token_text(engine, token, &byteCount) else {
                throw AFMError.generationFailed("DwarfStar returned an invalid token piece.")
            }
            pendingUTF8.append(
                UnsafeRawPointer(bytes).assumingMemoryBound(to: UInt8.self),
                count: byteCount
            )
            afm_ds4_free(bytes)
            outputTokens += 1

            if let piece = String(data: pendingUTF8, encoding: .utf8) {
                pendingUTF8.removeAll(keepingCapacity: true)
                generatedText += piece
                onText(piece, outputTokens)
                if request.options.stopSequences.contains(where: generatedText.hasSuffix) {
                    finishReason = .stop
                    break generationLoop
                }
            }

            if outputTokens < maximumTokens {
                let evalStatus = ds4_session_eval(session, token, &error, error.count)
                guard evalStatus == 0 else {
                    if Task.isCancelled { throw CancellationError() }
                    throw AFMError.generationFailed(Self.errorText(error))
                }
            }
        }

        if !pendingUTF8.isEmpty {
            let piece = String(decoding: pendingUTF8, as: UTF8.self)
            generatedText += piece
            onText(piece, outputTokens)
        }

        let generationSeconds = Self.seconds(since: generationStart)
        return AFMDwarfStarGenerationResult(
            text: generatedText,
            usage: AFMUsage(inputTokens: Int(prompt.len), outputTokens: outputTokens),
            finishReason: finishReason,
            metadata: [
                "runtime": .string("dwarfstar"),
                "backend": .string("metal"),
                "promptTime": .number(promptSeconds),
                "generateTime": .number(generationSeconds),
                "tokensPerSecond": .number(
                    generationSeconds > 0 ? Double(outputTokens) / generationSeconds : 0
                ),
                "modelPath": .string(loadedModelPath ?? "")
            ]
        )
    }

    private func unloadCurrent() {
        if let session { ds4_session_free(session) }
        if let engine { ds4_engine_close(engine) }
        session = nil
        engine = nil
        loadedModelPath = nil
        loadedMappingIdentity = nil
        loadedContextWindow = 0
    }

    private static func textContent(of message: AFMMessage) throws -> String {
        var result = ""
        for part in message.content {
            guard case .text(let text) = part else {
                throw AFMError.unsupportedCapability("non-text DwarfStar input")
            }
            result += text
        }
        return result
    }

    private static func tracePromptIfRequested(
        request: AFMRequest,
        prompt: ds4_tokens
    ) {
        guard ProcessInfo.processInfo.environment["AFM_DWARFSTAR_TRACE_PROMPT"] == "1" else {
            return
        }
        let roles = request.messages.map(\.role.rawValue).joined(separator: ",")
        let texts = request.messages.map { message in
            (try? textContent(of: message)) ?? "<non-text>"
        }
        let tokenIDs: [Int32]
        if let values = prompt.v {
            tokenIDs = (0..<Int(prompt.len)).map { Int32(values[$0]) }
        } else {
            tokenIDs = []
        }
        let line = "[DwarfStarPrompt] roles=\(roles) texts=\(texts.debugDescription) "
            + "tokens=\(tokenIDs)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    private static func errorText(_ buffer: [CChar]) -> String {
        String(
            decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) },
            as: UTF8.self
        )
    }
}

public struct AFMDwarfStarGenerationResult: Sendable {
    public var text: String
    public var usage: AFMUsage
    public var finishReason: AFMFinishReason
    public var metadata: [String: AFMJSONValue]

    public func response(modelID: String) -> AFMModelResponse {
        var metadata = metadata
        metadata["modelID"] = .string(modelID)
        return AFMModelResponse(
            text: text,
            usage: usage,
            finishReason: finishReason,
            metadata: metadata
        )
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
}
