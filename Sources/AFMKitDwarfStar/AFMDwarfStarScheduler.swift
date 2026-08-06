import Foundation
import AFMKitCore
import CDwarfStar

enum AFMDwarfStarSlotPolicy {
    static func bestSlot(
        commonPrefixes: [Int?],
        prefixCachingEnabled: Bool
    ) -> Int? {
        let available = commonPrefixes.indices.filter { commonPrefixes[$0] != nil }
        guard prefixCachingEnabled else { return available.first }
        return available.max { lhs, rhs in
            let left = commonPrefixes[lhs] ?? Int.min
            let right = commonPrefixes[rhs] ?? Int.min
            return left == right ? lhs > rhs : left < right
        }
    }
}

public actor AFMDwarfStarRuntimeCoordinator {
    public static let shared = AFMDwarfStarRuntimeCoordinator()

    private final class GenerationJob: @unchecked Sendable {
        let id: UUID
        let request: AFMRequest
        let onText: @Sendable (String, Int) -> Void
        let continuation: CheckedContinuation<AFMDwarfStarGenerationResult, any Error>
        var prompt: ds4_tokens
        var promptReleased = false
        var prefilled = false
        var cancelled = false
        var cachedInputTokens = 0
        var promptSeconds = 0.0
        var generationStart: ContinuousClock.Instant?
        var generatedText = ""
        var pendingUTF8 = Data()
        var outputTokens = 0
        var randomState: UInt64
        var peakBatchSize = 1

        init(
            id: UUID,
            request: AFMRequest,
            prompt: ds4_tokens,
            onText: @escaping @Sendable (String, Int) -> Void,
            continuation: CheckedContinuation<AFMDwarfStarGenerationResult, any Error>
        ) {
            self.id = id
            self.request = request
            self.prompt = prompt
            self.onText = onText
            self.continuation = continuation
            randomState = UInt64(bitPattern: Int64(request.options.seed ?? 0x5eed))
        }

        var maximumTokens: Int {
            max(0, request.options.maximumResponseTokens ?? 512)
        }

        func releasePrompt() {
            guard !promptReleased else { return }
            ds4_tokens_free(&prompt)
            afm_ds4_tokens_init(&prompt)
            promptReleased = true
        }
    }

    private struct RuntimeSlot {
        let session: OpaquePointer
        var job: GenerationJob?
    }

    private var engine: OpaquePointer?
    private var slots: [RuntimeSlot] = []
    private var pendingJobs: [GenerationJob] = []
    private var schedulerTask: Task<Void, Never>?
    private var loadedModelPath: String?
    private var loadedMappingIdentity: String?
    private var loadedContextWindow = 0
    private var loadedMaxConcurrent = 0
    private var prefixCachingEnabled = false

    public init() {}

    isolated deinit {
        schedulerTask?.cancel()
        pendingJobs.forEach { $0.releasePrompt() }
        for slot in slots {
            slot.job?.releasePrompt()
            ds4_session_free(slot.session)
        }
        if let engine { ds4_engine_close(engine) }
    }

    public func load(
        modelPath: String,
        templateGGUF: String? = nil,
        projectionMetadataPath: String? = nil,
        externalMapGGUF: Bool = false,
        contextWindow: Int,
        prefillChunk: Int,
        powerPercent: Int,
        enablePrefixCaching: Bool,
        maxConcurrent: Int
    ) throws {
        let residentSessions = max(1, maxConcurrent)
        let mappingIdentity = [
            externalMapGGUF ? "external-gguf" : "normal",
            templateGGUF ?? "",
            projectionMetadataPath ?? "",
        ].joined(separator: "|")
        if engine != nil,
           slots.count == residentSessions,
           loadedModelPath == modelPath,
           loadedMappingIdentity == mappingIdentity,
           loadedContextWindow == contextWindow,
           loadedMaxConcurrent == residentSessions,
           prefixCachingEnabled == enablePrefixCaching {
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

        var openedSlots: [RuntimeSlot] = []
        for _ in 0..<residentSessions {
            var openedSession: OpaquePointer?
            guard ds4_session_create(&openedSession, openedEngine, Int32(contextWindow)) == 0,
                  let openedSession else {
                openedSlots.forEach { ds4_session_free($0.session) }
                ds4_engine_close(openedEngine)
                throw AFMError.loadingFailed(
                    "DwarfStar failed to allocate \(residentSessions) inference sessions.")
            }
            openedSlots.append(RuntimeSlot(session: openedSession))
        }
        if let first = openedSlots.first {
            ds4_session_gpu_warmup(first.session)
        }

        engine = openedEngine
        slots = openedSlots
        loadedModelPath = modelPath
        loadedMappingIdentity = mappingIdentity
        loadedContextWindow = contextWindow
        loadedMaxConcurrent = residentSessions
        prefixCachingEnabled = enablePrefixCaching
    }

    public func unload(modelPath: String) {
        guard loadedModelPath == modelPath else { return }
        unloadCurrent()
    }

    func generate(
        request: AFMRequest,
        onText: @escaping @Sendable (String, Int) -> Void
    ) async throws -> AFMDwarfStarGenerationResult {
        guard let engine, !slots.isEmpty else {
            throw AFMError.unavailable("DwarfStar is not loaded.")
        }
        guard request.tools.isEmpty else {
            throw AFMError.unsupportedCapability("tool calling in the DwarfStar runtime")
        }

        let id = UUID()
        var prompt = try Self.makePrompt(engine: engine, request: request)
        Self.tracePromptIfRequested(request: request, prompt: prompt)

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if Task<Never, Never>.isCancelled {
                    ds4_tokens_free(&prompt)
                    continuation.resume(throwing: CancellationError())
                    return
                }
                pendingJobs.append(
                    GenerationJob(
                        id: id,
                        request: request,
                        prompt: prompt,
                        onText: onText,
                        continuation: continuation)
                )
                startSchedulerIfNeeded()
            }
        } onCancel: {
            Task { await self.cancelGeneration(id: id) }
        }
    }

    private func startSchedulerIfNeeded() {
        guard schedulerTask == nil else { return }
        schedulerTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .milliseconds(2))
            await self.runScheduler()
        }
    }

    private func runScheduler() async {
        while !Task<Never, Never>.isCancelled {
            assignPendingJobs()
            if let prefillIndex = slots.firstIndex(where: { $0.job?.prefilled == false }) {
                prefill(slotIndex: prefillIndex)
            } else if slots.contains(where: { $0.job != nil }) {
                decodeCycle()
            } else if pendingJobs.isEmpty {
                break
            }
            await Task.yield()
        }
        schedulerTask = nil
        if !pendingJobs.isEmpty || slots.contains(where: { $0.job != nil }) {
            startSchedulerIfNeeded()
        }
    }

    private func assignPendingJobs() {
        while let job = pendingJobs.first {
            let commonPrefixes: [Int?] = slots.map { slot in
                guard slot.job == nil else { return nil }
                guard prefixCachingEnabled else { return 0 }
                let liveTokens = Int(ds4_session_pos(slot.session))
                let commonTokens = Int(ds4_session_common_prefix(slot.session, &job.prompt))
                // DwarfStar can extend the live graph state without replay only
                // when the complete resident checkpoint prefixes the new prompt.
                // An earlier common prefix still requires rebuilding compressor
                // and KV frontiers, so it must not be reported as a cache hit.
                guard commonTokens == liveTokens, Int(job.prompt.len) >= liveTokens else {
                    return 0
                }
                return liveTokens
            }
            guard let slotIndex = AFMDwarfStarSlotPolicy.bestSlot(
                commonPrefixes: commonPrefixes,
                prefixCachingEnabled: prefixCachingEnabled
            ) else {
                return
            }
            pendingJobs.removeFirst()
            if job.cancelled {
                finishPending(job, throwing: CancellationError())
                continue
            }
            if !prefixCachingEnabled {
                ds4_session_invalidate(slots[slotIndex].session)
            }
            job.cachedInputTokens = prefixCachingEnabled ? (commonPrefixes[slotIndex] ?? 0) : 0
            slots[slotIndex].job = job
        }
    }

    private func prefill(slotIndex: Int) {
        guard let job = slots[slotIndex].job else { return }
        if job.cancelled {
            finish(slotIndex: slotIndex, throwing: CancellationError())
            return
        }

        var error = [CChar](repeating: 0, count: 512)
        let started = ContinuousClock.now
        let status = ds4_session_sync(
            slots[slotIndex].session,
            &job.prompt,
            &error,
            error.count)
        job.promptSeconds = Self.seconds(since: started)
        guard status == 0 else {
            if status == DS4_SESSION_SYNC_INTERRUPTED || job.cancelled {
                finish(slotIndex: slotIndex, throwing: CancellationError())
            } else {
                finish(
                    slotIndex: slotIndex,
                    throwing: AFMError.generationFailed(Self.errorText(error)))
            }
            return
        }
        job.prefilled = true
        job.generationStart = .now
        if job.maximumTokens == 0 {
            finish(slotIndex: slotIndex, reason: .length)
        }
    }

    private func decodeCycle() {
        guard let engine else { return }
        var evalItems: [ds4_decode_item] = []
        var evalSlots: [Int] = []

        for slotIndex in slots.indices {
            guard let job = slots[slotIndex].job, job.prefilled else { continue }
            if job.cancelled {
                finish(slotIndex: slotIndex, throwing: CancellationError())
                continue
            }

            let temperature = Float(job.request.options.temperature ?? 0)
            let token = temperature <= 0
                ? ds4_session_argmax(slots[slotIndex].session)
                : ds4_session_sample(
                    slots[slotIndex].session,
                    temperature,
                    Int32(job.request.options.topK ?? 0),
                    Float(job.request.options.topP ?? 1),
                    Float(job.request.options.minP ?? 0.05),
                    &job.randomState)
            if ds4_token_is_stop_for_think_mode(engine, token, DS4_THINK_NONE) {
                finish(slotIndex: slotIndex, reason: .stop)
                continue
            }

            var byteCount = 0
            guard let bytes = ds4_token_text(engine, token, &byteCount) else {
                finish(
                    slotIndex: slotIndex,
                    throwing: AFMError.generationFailed(
                        "DwarfStar returned an invalid token piece."))
                continue
            }
            job.pendingUTF8.append(
                UnsafeRawPointer(bytes).assumingMemoryBound(to: UInt8.self),
                count: byteCount)
            afm_ds4_free(bytes)
            job.outputTokens += 1

            if let piece = String(data: job.pendingUTF8, encoding: .utf8) {
                job.pendingUTF8.removeAll(keepingCapacity: true)
                job.generatedText += piece
                job.onText(piece, job.outputTokens)
                if job.request.options.stopSequences.contains(where: job.generatedText.hasSuffix) {
                    finish(slotIndex: slotIndex, reason: .stop)
                    continue
                }
            }

            if job.outputTokens >= job.maximumTokens {
                finish(slotIndex: slotIndex, reason: .length)
                continue
            }

            var item = ds4_decode_item()
            item.session = slots[slotIndex].session
            item.token = token
            evalItems.append(item)
            evalSlots.append(slotIndex)
        }

        guard !evalItems.isEmpty else { return }
        let batchSize = evalItems.count
        for slotIndex in evalSlots {
            slots[slotIndex].job?.peakBatchSize = max(
                slots[slotIndex].job?.peakBatchSize ?? 1,
                batchSize)
        }
        var error = [CChar](repeating: 0, count: 512)
        let status = evalItems.withUnsafeMutableBufferPointer { items in
            ds4_sessions_eval_batch(
                items.baseAddress,
                Int32(items.count),
                &error,
                error.count)
        }
        guard status != 0 else { return }

        let failure = AFMError.generationFailed(Self.errorText(error))
        for slotIndex in evalSlots {
            guard slots[slotIndex].job != nil else { continue }
            ds4_session_invalidate(slots[slotIndex].session)
            finish(slotIndex: slotIndex, throwing: failure)
        }
    }

    private func finish(slotIndex: Int, reason: AFMFinishReason) {
        guard let job = slots[slotIndex].job else { return }
        flushPendingUTF8(job)
        let generationSeconds = job.generationStart.map(Self.seconds(since:)) ?? 0
        let result = AFMDwarfStarGenerationResult(
            text: job.generatedText,
            usage: AFMUsage(
                inputTokens: Int(job.prompt.len),
                cachedInputTokens: job.cachedInputTokens,
                outputTokens: job.outputTokens),
            finishReason: reason,
            metadata: [
                "runtime": .string("dwarfstar"),
                "backend": .string("metal"),
                "promptTime": .number(job.promptSeconds),
                "generateTime": .number(generationSeconds),
                "tokensPerSecond": .number(
                    generationSeconds > 0 ? Double(job.outputTokens) / generationSeconds : 0),
                "modelPath": .string(loadedModelPath ?? ""),
                "cachedInputTokens": .integer(job.cachedInputTokens),
                "peakBatchSize": .integer(job.peakBatchSize),
            ])
        slots[slotIndex].job = nil
        job.releasePrompt()
        job.continuation.resume(returning: result)
    }

    private func finish(slotIndex: Int, throwing error: any Error) {
        guard let job = slots[slotIndex].job else { return }
        slots[slotIndex].job = nil
        job.releasePrompt()
        job.continuation.resume(throwing: error)
    }

    private func finishPending(_ job: GenerationJob, throwing error: any Error) {
        job.releasePrompt()
        job.continuation.resume(throwing: error)
    }

    private func flushPendingUTF8(_ job: GenerationJob) {
        guard !job.pendingUTF8.isEmpty else { return }
        let piece = String(decoding: job.pendingUTF8, as: UTF8.self)
        job.pendingUTF8.removeAll(keepingCapacity: false)
        job.generatedText += piece
        job.onText(piece, job.outputTokens)
    }

    private func cancelGeneration(id: UUID) {
        if let pending = pendingJobs.first(where: { $0.id == id }) {
            pending.cancelled = true
            return
        }
        for index in slots.indices where slots[index].job?.id == id {
            slots[index].job?.cancelled = true
            return
        }
    }

    private func unloadCurrent() {
        schedulerTask?.cancel()
        schedulerTask = nil
        for job in pendingJobs {
            finishPending(job, throwing: CancellationError())
        }
        pendingJobs.removeAll(keepingCapacity: false)
        for index in slots.indices {
            if slots[index].job != nil {
                finish(slotIndex: index, throwing: CancellationError())
            }
            ds4_session_free(slots[index].session)
        }
        slots.removeAll(keepingCapacity: false)
        if let engine { ds4_engine_close(engine) }
        engine = nil
        loadedModelPath = nil
        loadedMappingIdentity = nil
        loadedContextWindow = 0
        loadedMaxConcurrent = 0
        prefixCachingEnabled = false
    }

    private static func makePrompt(
        engine: OpaquePointer,
        request: AFMRequest
    ) throws -> ds4_tokens {
        var prompt = ds4_tokens()
        afm_ds4_tokens_init(&prompt)
        ds4_chat_begin(engine, &prompt)
        do {
            for message in request.messages {
                let text = try textContent(of: message)
                message.role.rawValue.withCString { rolePointer in
                    text.withCString { textPointer in
                        ds4_chat_append_message(engine, &prompt, rolePointer, textPointer)
                    }
                }
            }
            ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE)
            return prompt
        } catch {
            ds4_tokens_free(&prompt)
            throw error
        }
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
            as: UTF8.self)
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
            metadata: metadata)
    }
}
