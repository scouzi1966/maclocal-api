import Foundation
import AFMKitCore
import CDwarfStar

enum AFMDwarfStarReasoningMode: String, Equatable, Sendable {
    case chat
    case low
    case high
    case max

    static func resolve(metadata: [String: AFMJSONValue]) -> Self {
        guard case .object(let kwargs)? = metadata["chatTemplateKwargs"] else {
            return .chat
        }
        if case .bool(false)? = kwargs["enable_thinking"] {
            return .chat
        }
        let rawEffort: String?
        if case .string(let value)? = kwargs["reasoning_effort"] {
            rawEffort = value
        } else {
            rawEffort = nil
        }
        if let rawEffort, let effort = Self(rawValue: rawEffort.lowercased()) {
            return effort
        }
        if case .bool(true)? = kwargs["enable_thinking"] {
            return .low
        }
        return .chat
    }

    var thinkMode: ds4_think_mode {
        switch self {
        case .chat:
            return DS4_THINK_NONE
        case .low, .high:
            return DS4_THINK_HIGH
        case .max:
            return DS4_THINK_MAX
        }
    }

    var promptPrefix: String? {
        switch self {
        case .chat, .low:
            return nil
        case .high:
            return Self.highPrefix
        case .max:
            return Self.maxPrefix
        }
    }

    // Verbatim DeepSeek-V4-Flash-0731 prompt contract.
    private static let highPrefix = """
        Reasoning Effort: Absolute maximum with no shortcuts permitted.
        You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
        Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.


        """

    private static let maxPrefix = """
        Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.
        You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.
        Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.


        """
}

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

enum AFMDwarfStarSchedulingPolicy {
    static let idlePrefillQuantum = 2_048
    static let mixedPrefillQuantum = 128

    static func prefillQuantum(activeDecodeCount: Int) -> Int {
        activeDecodeCount > 0 ? mixedPrefillQuantum : idlePrefillQuantum
    }

    static func canMixPrefill(currentPosition: Int, activeDecodeCount: Int) -> Bool {
        currentPosition > 0 && activeDecodeCount > 0
    }

    static func nextPrefillSlot(lastSlot: Int, waiting: [Bool]) -> Int? {
        guard !waiting.isEmpty else { return nil }
        for offset in 1...waiting.count {
            let index = (lastSlot + offset) % waiting.count
            if waiting[index] { return index }
        }
        return nil
    }
}

enum AFMDwarfStarSpeculativePolicy {
    static func isAvailable(requested: Bool, draftTokenCount: Int) -> Bool {
        requested && draftTokenCount > 0
    }
}

enum AFMDwarfStarStoppingPolicy {
    static func shouldStop(
        isEndOfSequence: Bool,
        isRuntimeStop: Bool,
        ignoreEndOfSequence: Bool
    ) -> Bool {
        isRuntimeStop && !(ignoreEndOfSequence && isEndOfSequence)
    }

    static func shouldExposeToken(
        isEndOfSequence: Bool,
        ignoreEndOfSequence: Bool
    ) -> Bool {
        !(ignoreEndOfSequence && isEndOfSequence)
    }
}

enum AFMDwarfStarRawStopPolicy {
    struct Result: Equatable {
        var visibleText: String
        var stopped: Bool
    }

    static func consume(
        buffer: inout String,
        piece: String,
        stopSequences: [String]
    ) -> Result {
        buffer += piece
        let stops = stopSequences.filter { !$0.isEmpty }
        guard !stops.isEmpty else {
            return Result(visibleText: drain(buffer: &buffer), stopped: false)
        }

        let matches = stops.compactMap { buffer.range(of: $0) }
        if let first = matches.min(by: { $0.lowerBound < $1.lowerBound }) {
            let visible = String(buffer[..<first.lowerBound])
            buffer.removeAll(keepingCapacity: true)
            return Result(visibleText: visible, stopped: true)
        }

        let retain = stops.reduce(0) { current, stop in
            let limit = min(buffer.count, max(0, stop.count - 1))
            guard limit > current else { return current }
            for length in stride(from: limit, through: current + 1, by: -1) {
                if buffer.suffix(length) == stop.prefix(length) {
                    return length
                }
            }
            return current
        }
        let visibleCount = buffer.count - retain
        guard visibleCount > 0 else {
            return Result(visibleText: "", stopped: false)
        }
        let split = buffer.index(buffer.startIndex, offsetBy: visibleCount)
        let visible = String(buffer[..<split])
        buffer = String(buffer[split...])
        return Result(visibleText: visible, stopped: false)
    }

    static func drain(buffer: inout String) -> String {
        let text = buffer
        buffer.removeAll(keepingCapacity: false)
        return text
    }
}

enum AFMDwarfStarPrefixCachePolicy {
    static let defaultBudgetMB: UInt64 = 4_096

    static func checkpointKey(path: String, size: UInt64, modified: TimeInterval) -> String {
        let identity = "\(URL(fileURLWithPath: path).standardizedFileURL.path)|\(size)|\(modified)"
        let digest = identity.utf8.reduce(UInt64(1_469_598_103_934_665_603)) { partial, byte in
            (partial ^ UInt64(byte)) &* 1_099_511_628_211
        }
        return String(digest, radix: 16)
    }

    static func budgetMB(environment: [String: String]) -> UInt64 {
        guard let raw = environment["AFM_DWARFSTAR_PREFIX_CACHE_MB"],
              let value = UInt64(raw), value > 0 else {
            return defaultBudgetMB
        }
        return value
    }
}

public actor AFMDwarfStarRuntimeCoordinator {
    public static let shared = AFMDwarfStarRuntimeCoordinator()

    private final class GenerationJob: @unchecked Sendable {
        let id: UUID
        let request: AFMRequest
        let onEvent: @Sendable (AFMGenerationEvent) -> Void
        let continuation: CheckedContinuation<AFMDwarfStarGenerationResult, any Error>
        let telemetryObserver: any AFMInferenceTelemetryObserving
        let telemetryToken: AFMInferenceRequestToken
        var prompt: ds4_tokens
        var promptReleased = false
        var prefilled = false
        var cancelled = false
        var cachedInputTokens = 0
        var promptSeconds = 0.0
        var generationStart: ContinuousClock.Instant?
        var generatedText = ""
        var rawEmissionBuffer = ""
        var generatedReasoning = ""
        var toolCalls: [AFMToolCall] = []
        var toolParser: AFMDwarfStarToolCodec.StreamParser
        var pendingUTF8 = Data()
        var outputTokens = 0
        var randomState: UInt64
        var peakBatchSize = 1
        let reasoningMode: AFMDwarfStarReasoningMode
        var speculativeCycles = 0
        var speculativeAcceptedTokens = 0
        var persistedPrefixTokens = 0

        init(
            id: UUID,
            request: AFMRequest,
            prompt: ds4_tokens,
            telemetryObserver: any AFMInferenceTelemetryObserving,
            telemetryToken: AFMInferenceRequestToken,
            onEvent: @escaping @Sendable (AFMGenerationEvent) -> Void,
            continuation: CheckedContinuation<AFMDwarfStarGenerationResult, any Error>
        ) {
            self.id = id
            self.request = request
            self.prompt = prompt
            self.telemetryObserver = telemetryObserver
            self.telemetryToken = telemetryToken
            self.onEvent = onEvent
            self.continuation = continuation
            randomState = UInt64(bitPattern: Int64(request.options.seed ?? 0x5eed))
            reasoningMode = .resolve(metadata: request.metadata)
            toolParser = AFMDwarfStarToolCodec.StreamParser(
                startsInReasoning: reasoningMode != .chat
            )
        }

        var maximumTokens: Int {
            max(0, request.options.maximumResponseTokens ?? 512)
        }

        var isRawPrompt: Bool {
            if case .string = request.metadata["afm.rawPrompt"] { return true }
            return false
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
    private var prefixCache: OpaquePointer?
    private var slots: [RuntimeSlot] = []
    private var pendingJobs: [GenerationJob] = []
    private var schedulerTask: Task<Void, Never>?
    private var loadedModelPath: String?
    private var loadedMappingIdentity: String?
    private var loadedContextWindow = 0
    private var loadedMaxConcurrent = 0
    private var prefixCachingEnabled = false
    private var dsparkEnabled = false
    private var lastPrefillSlot = -1

    public init() {}

    isolated deinit {
        schedulerTask?.cancel()
        pendingJobs.forEach { $0.releasePrompt() }
        for slot in slots {
            slot.job?.releasePrompt()
            ds4_session_free(slot.session)
        }
        if let prefixCache { afm_ds4_prefix_cache_close(prefixCache) }
        if let engine { ds4_engine_close(engine) }
    }

    public func load(
        modelPath: String,
        contextWindow: Int,
        prefillChunk: Int,
        powerPercent: Int,
        dsparkSupportPath: String? = nil,
        dsparkDraftTokens: Int = 5,
        dsparkConfidenceThreshold: Double = 0.7,
        dsparkStrict: Bool = false,
        enablePrefixCaching: Bool,
        maxConcurrent: Int
    ) throws {
        let residentSessions = max(1, maxConcurrent)
        let mappingIdentity = [
            dsparkSupportPath ?? "",
            String(dsparkDraftTokens),
            String(dsparkConfidenceThreshold),
            String(dsparkStrict),
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
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw AFMError.loadingFailed(
                "DwarfStar requires a native GGUF file; AFM checkpoint directories run through MLX.")
        }
        if let dsparkSupportPath,
           !FileManager.default.fileExists(atPath: dsparkSupportPath) {
            throw AFMError.loadingFailed(
                "DSpark support checkpoint does not exist at \(dsparkSupportPath)")
        }

        var openedEngine: OpaquePointer?
        var error = [CChar](repeating: 0, count: 512)
        let openModel: (UnsafePointer<CChar>?) -> Int32 = { supportPointer in
            modelPath.withCString { modelPathPointer in
                sourceRoot.withCString { sourceRootPointer in
                    afm_ds4_engine_open(
                        &openedEngine,
                        modelPathPointer,
                        Int32(contextWindow),
                        UInt32(clamping: prefillChunk),
                        Int32(powerPercent),
                        supportPointer,
                        Int32(clamping: dsparkDraftTokens),
                        Float(dsparkConfidenceThreshold),
                        dsparkStrict ? 1 : 0,
                        sourceRootPointer,
                        &error,
                        error.count)
                }
            }
        }
        let status = if let dsparkSupportPath {
            dsparkSupportPath.withCString(openModel)
        } else {
            openModel(nil)
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

        var openedPrefixCache: OpaquePointer?
        if enablePrefixCaching {
            let cacheDirectory = try Self.prefixCacheDirectory(modelPath: modelPath)
            let cacheBudget = Self.prefixCacheBudgetMB()
            let cacheStatus = cacheDirectory.withCString { directoryPointer in
                afm_ds4_prefix_cache_open(
                    &openedPrefixCache,
                    directoryPointer,
                    cacheBudget,
                    &error,
                    error.count)
            }
            guard cacheStatus == 0, openedPrefixCache != nil else {
                openedSlots.forEach { ds4_session_free($0.session) }
                ds4_engine_close(openedEngine)
                throw AFMError.loadingFailed(Self.errorText(error))
            }
        }

        engine = openedEngine
        prefixCache = openedPrefixCache
        slots = openedSlots
        loadedModelPath = modelPath
        loadedMappingIdentity = mappingIdentity
        loadedContextWindow = contextWindow
        loadedMaxConcurrent = residentSessions
        prefixCachingEnabled = enablePrefixCaching
        dsparkEnabled = AFMDwarfStarSpeculativePolicy.isAvailable(
            requested: dsparkSupportPath != nil,
            draftTokenCount: Int(ds4_engine_mtp_draft_tokens(openedEngine)))
    }

    public func unload(modelPath: String) {
        guard loadedModelPath == modelPath else { return }
        unloadCurrent()
    }

    func generate(
        request: AFMRequest,
        onEvent: @escaping @Sendable (AFMGenerationEvent) -> Void
    ) async throws -> AFMDwarfStarGenerationResult {
        try await generate(
            request: request,
            telemetryObserver: AFMNoopInferenceTelemetryObserver(),
            onEvent: onEvent
        )
    }

    func generate(
        request: AFMRequest,
        telemetryObserver: any AFMInferenceTelemetryObserving,
        onEvent: @escaping @Sendable (AFMGenerationEvent) -> Void
    ) async throws -> AFMDwarfStarGenerationResult {
        guard let engine, !slots.isEmpty else {
            throw AFMError.unavailable("DwarfStar is not loaded.")
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
                let acceptedAt = Date().timeIntervalSince1970
                let telemetryToken = telemetryObserver.requestAccepted(at: acceptedAt)
                pendingJobs.append(
                    GenerationJob(
                        id: id,
                        request: request,
                        prompt: prompt,
                        telemetryObserver: telemetryObserver,
                        telemetryToken: telemetryToken,
                        onEvent: onEvent,
                        continuation: continuation)
                )
                publishProviderState()
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
            if let prefillIndex = nextPrefillSlot() {
                prefillCycle(slotIndex: prefillIndex)
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

    private func nextPrefillSlot() -> Int? {
        let waiting = slots.map { $0.job?.prefilled == false }
        let index = AFMDwarfStarSchedulingPolicy.nextPrefillSlot(
            lastSlot: lastPrefillSlot,
            waiting: waiting)
        if let index { lastPrefillSlot = index }
        return index
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
            var cachedTokens = prefixCachingEnabled ? (commonPrefixes[slotIndex] ?? 0) : 0
            if prefixCachingEnabled,
               cachedTokens == 0,
               let engine,
               let prefixCache,
               slots.allSatisfy({ $0.job == nil }) {
                var error = [CChar](repeating: 0, count: 512)
                let restored = afm_ds4_prefix_cache_restore(
                    prefixCache,
                    engine,
                    slots[slotIndex].session,
                    &job.prompt,
                    &error,
                    error.count)
                if restored > 0 {
                    cachedTokens = Int(restored)
                } else if restored < 0 {
                    Self.logPrefixCache(Self.errorText(error))
                    ds4_session_invalidate(slots[slotIndex].session)
                }
            }
            if !prefixCachingEnabled || cachedTokens == 0 {
                ds4_session_invalidate(slots[slotIndex].session)
            }
            job.cachedInputTokens = cachedTokens
            slots[slotIndex].job = job
            let now = Date().timeIntervalSince1970
            job.telemetryObserver.requestStarted(job.telemetryToken, at: now)
            job.telemetryObserver.promptTokensProcessed(
                job.telemetryToken,
                fullPromptTokens: Int(job.prompt.len),
                computedPromptTokens: max(0, Int(job.prompt.len) - cachedTokens),
                at: now
            )
            job.telemetryObserver.prefixCacheObserved(
                queriedTokens: Int(job.prompt.len),
                hitTokens: cachedTokens
            )
        }
        publishProviderState()
    }

    /// Advance one bounded prefill quantum. When other slots are decoding, use
    /// DwarfStar's native mixed prefill/decode entry point so a long prompt
    /// cannot stall every resident generation until its entire prefill ends.
    private func prefillCycle(slotIndex: Int) {
        guard let job = slots[slotIndex].job else { return }
        if job.cancelled {
            finish(slotIndex: slotIndex, throwing: CancellationError())
            return
        }

        guard let engine else { return }
        let session = slots[slotIndex].session
        let current = max(0, Int(ds4_session_pos(session)))
        // The upstream mixed API extends a valid checkpoint. A fresh session
        // must establish its first checkpoint before decode rows can share a
        // scheduling epoch with subsequent prefill quanta.
        let decodeBatch = current > 0
            ? prepareDecodeBatch(engine: engine, excluding: slotIndex)
            : DecodeBatch(items: [], slotIndices: [])
        guard slots[slotIndex].job != nil else { return }

        let fullLength = Int(job.prompt.len)
        let quantum = AFMDwarfStarSchedulingPolicy.prefillQuantum(
            activeDecodeCount: decodeBatch.items.count)
        let target = min(fullLength, max(current + quantum, 1))
        var promptPrefix = job.prompt
        promptPrefix.len = Int32(target)

        var error = [CChar](repeating: 0, count: 512)
        let started = ContinuousClock.now
        let status: Int32
        if !AFMDwarfStarSchedulingPolicy.canMixPrefill(
            currentPosition: current,
            activeDecodeCount: decodeBatch.items.count) {
            status = ds4_session_sync(session, &promptPrefix, &error, error.count)
        } else {
            var items = decodeBatch.items
            status = items.withUnsafeMutableBufferPointer { buffer in
                ds4_sessions_eval_batch_with_prefill(
                    buffer.baseAddress,
                    Int32(buffer.count),
                    session,
                    &promptPrefix,
                    &error,
                    error.count)
            }
        }
        job.promptSeconds += Self.seconds(since: started)
        guard status == 0 else {
            if status == DS4_SESSION_SYNC_INTERRUPTED || job.cancelled {
                finish(slotIndex: slotIndex, throwing: CancellationError())
            } else {
                finish(
                    slotIndex: slotIndex,
                    throwing: AFMError.generationFailed(Self.errorText(error)))
            }
            let failure = AFMError.generationFailed(Self.errorText(error))
            for decodeSlot in decodeBatch.slotIndices where slots[decodeSlot].job != nil {
                ds4_session_invalidate(slots[decodeSlot].session)
                finish(slotIndex: decodeSlot, throwing: failure)
            }
            return
        }

        recordBatchSize(decodeBatch.slotIndices)
        if Int(ds4_session_pos(session)) >= fullLength {
            job.prefilled = true
            // Persist the exact user prompt while the live checkpoint is still
            // at that boundary. A post-generation checkpoint includes assistant
            // output and cannot satisfy a later request that repeats or extends
            // only the original prompt.
            persistPrefixIfIdle(slotIndex: slotIndex, job: job, reason: "cold")
            job.generationStart = .now
            if job.maximumTokens == 0 {
                finish(slotIndex: slotIndex, reason: .length)
            }
        }
        publishProviderState()
    }

    private func decodeCycle() {
        guard let engine else { return }
        let batch = prepareDecodeBatch(engine: engine, excluding: nil)
        guard !batch.items.isEmpty else { return }
        recordBatchSize(batch.slotIndices)

        var items = batch.items
        var error = [CChar](repeating: 0, count: 512)
        let status = items.withUnsafeMutableBufferPointer { buffer in
            ds4_sessions_eval_batch(
                buffer.baseAddress,
                Int32(buffer.count),
                &error,
                error.count)
        }
        guard status != 0 else { return }

        let failure = AFMError.generationFailed(Self.errorText(error))
        for slotIndex in batch.slotIndices {
            guard slots[slotIndex].job != nil else { continue }
            ds4_session_invalidate(slots[slotIndex].session)
            finish(slotIndex: slotIndex, throwing: failure)
        }
    }

    private struct DecodeBatch {
        var items: [ds4_decode_item]
        var slotIndices: [Int]
    }

    private func prepareDecodeBatch(
        engine: OpaquePointer,
        excluding excludedSlot: Int?
    ) -> DecodeBatch {
        var evalItems: [ds4_decode_item] = []
        var evalSlots: [Int] = []

        for slotIndex in slots.indices {
            if slotIndex == excludedSlot { continue }
            guard let job = slots[slotIndex].job, job.prefilled else { continue }
            if job.cancelled {
                finish(slotIndex: slotIndex, throwing: CancellationError())
                continue
            }
            if Int(ds4_session_pos(slots[slotIndex].session)) >= loadedContextWindow {
                finish(slotIndex: slotIndex, reason: .length)
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

            let isEOS = token == ds4_token_eos(engine)
            if AFMDwarfStarStoppingPolicy.shouldStop(
                isEndOfSequence: isEOS,
                isRuntimeStop: ds4_token_is_stop_for_think_mode(
                    engine,
                    token,
                    job.reasoningMode.thinkMode
                ),
                ignoreEndOfSequence: job.request.options.ignoreEndOfSequence
            ) {
                finish(slotIndex: slotIndex, reason: .stop)
                continue
            }

            if dsparkEnabled,
               temperature <= 0,
               ds4_engine_mtp_draft_tokens(engine) > 0 {
                let remaining = job.maximumTokens - job.outputTokens
                let capacity = max(1, min(16, remaining))
                var accepted = [Int32](repeating: 0, count: capacity)
                var error = [CChar](repeating: 0, count: 512)
                let acceptedCount = accepted.withUnsafeMutableBufferPointer { buffer in
                    ds4_session_eval_speculative_argmax(
                        slots[slotIndex].session,
                        token,
                        Int32(remaining),
                        job.request.options.ignoreEndOfSequence ? -1 : ds4_token_eos(engine),
                        buffer.baseAddress,
                        Int32(buffer.count),
                        &error,
                        error.count)
                }
                guard acceptedCount >= 0 else {
                    ds4_session_invalidate(slots[slotIndex].session)
                    finish(
                        slotIndex: slotIndex,
                        throwing: AFMError.generationFailed(Self.errorText(error)))
                    continue
                }
                if acceptedCount > 0 {
                    job.speculativeCycles += 1
                    job.speculativeAcceptedTokens += max(0, Int(acceptedCount) - 1)
                    job.telemetryObserver.speculativeRound(
                        draftTokens: min(
                            max(0, Int(ds4_engine_mtp_draft_tokens(engine))),
                            max(0, remaining - 1)
                        ),
                        acceptedTokens: max(0, Int(acceptedCount) - 1)
                    )
                    for acceptedToken in accepted.prefix(Int(acceptedCount)) {
                        guard slots[slotIndex].job != nil else { break }
                        emit(token: acceptedToken, engine: engine, slotIndex: slotIndex)
                    }
                    continue
                }
            }

            emit(token: token, engine: engine, slotIndex: slotIndex)
            guard slots[slotIndex].job != nil else {
                continue
            }

            var item = ds4_decode_item()
            item.session = slots[slotIndex].session
            item.token = token
            evalItems.append(item)
            evalSlots.append(slotIndex)
        }
        return DecodeBatch(items: evalItems, slotIndices: evalSlots)
    }

    private func recordBatchSize(_ slotIndices: [Int]) {
        let batchSize = slotIndices.count
        guard batchSize > 0 else { return }
        for slotIndex in slotIndices {
            slots[slotIndex].job?.peakBatchSize = max(
                slots[slotIndex].job?.peakBatchSize ?? 1,
                batchSize)
        }
    }

    private func emit(token: Int32, engine: OpaquePointer, slotIndex: Int) {
        guard let job = slots[slotIndex].job else { return }
        let isEOS = token == ds4_token_eos(engine)
        if AFMDwarfStarStoppingPolicy.shouldStop(
            isEndOfSequence: isEOS,
            isRuntimeStop: ds4_token_is_stop_for_think_mode(
                engine,
                token,
                job.reasoningMode.thinkMode
            ),
            ignoreEndOfSequence: job.request.options.ignoreEndOfSequence
        ) {
            finish(slotIndex: slotIndex, reason: .stop)
            return
        }
        if !AFMDwarfStarStoppingPolicy.shouldExposeToken(
            isEndOfSequence: isEOS,
            ignoreEndOfSequence: job.request.options.ignoreEndOfSequence
        ) {
            return
        }

        var byteCount = 0
        guard let bytes = ds4_token_text(engine, token, &byteCount) else {
            finish(
                slotIndex: slotIndex,
                throwing: AFMError.generationFailed(
                    "DwarfStar returned an invalid token piece."))
            return
        }
        job.pendingUTF8.append(
            UnsafeRawPointer(bytes).assumingMemoryBound(to: UInt8.self),
            count: byteCount)
        afm_ds4_free(bytes)
        job.outputTokens += 1
        job.telemetryObserver.outputToken(
            job.telemetryToken,
            at: Date().timeIntervalSince1970
        )

        if let piece = String(data: job.pendingUTF8, encoding: .utf8) {
            job.pendingUTF8.removeAll(keepingCapacity: true)
            if job.isRawPrompt {
                if processRawText(piece, for: job, tokenCount: 1) {
                    finish(slotIndex: slotIndex, reason: .stop)
                    return
                }
            } else {
                do {
                    let completedToolCall = process(
                        try job.toolParser.consume(piece),
                        for: job,
                        tokenCount: 1
                    )
                    if completedToolCall {
                        finish(slotIndex: slotIndex, reason: .toolCalls)
                        return
                    }
                } catch {
                    finish(slotIndex: slotIndex, throwing: error)
                    return
                }
            }
            if !job.isRawPrompt,
               job.request.options.stopSequences.contains(where: job.generatedText.hasSuffix) {
                finish(slotIndex: slotIndex, reason: .stop)
                return
            }
        }

        if job.outputTokens >= job.maximumTokens {
            finish(slotIndex: slotIndex, reason: .length)
        }
    }

    private func finish(slotIndex: Int, reason: AFMFinishReason) {
        guard let job = slots[slotIndex].job else { return }
        flushPendingUTF8(job)
        flushRawEmissionBuffer(job)
        if !job.isRawPrompt {
            _ = process(job.toolParser.finish(), for: job, tokenCount: 0)
        }
        persistPrefixIfIdle(slotIndex: slotIndex, job: job, reason: "continued")
        let generationSeconds = job.generationStart.map(Self.seconds(since:)) ?? 0
        let result = AFMDwarfStarGenerationResult(
            text: job.generatedText,
            reasoning: job.generatedReasoning.isEmpty ? nil : job.generatedReasoning,
            usage: AFMUsage(
                inputTokens: Int(job.prompt.len),
                cachedInputTokens: job.cachedInputTokens,
                outputTokens: job.outputTokens),
            toolCalls: job.toolCalls,
            finishReason: job.toolCalls.isEmpty ? reason : .toolCalls,
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
                "reasoningEffort": .string(job.reasoningMode.rawValue),
                "dsparkEnabled": .bool(dsparkEnabled),
                "speculativeCycles": .integer(job.speculativeCycles),
                "speculativeAcceptedTokens": .integer(job.speculativeAcceptedTokens),
                "persistedPrefixTokens": .integer(job.persistedPrefixTokens),
            ])
        slots[slotIndex].job = nil
        _ = job.telemetryObserver.requestFinished(
            job.telemetryToken,
            observation: AFMInferenceRequestFinishObservation(
                reason: Self.telemetryFinishReason(reason),
                completedAt: Date().timeIntervalSince1970,
                fullPromptTokens: Int(job.prompt.len),
                computedPromptTokens: max(0, Int(job.prompt.len) - job.cachedInputTokens),
                generatedTokens: job.outputTokens,
                maximumOutputTokens: job.maximumTokens
            )
        )
        publishProviderState(additionalObservers: [job.telemetryObserver])
        job.releasePrompt()
        job.continuation.resume(returning: result)
    }

    private func finish(slotIndex: Int, throwing error: any Error) {
        guard let job = slots[slotIndex].job else { return }
        slots[slotIndex].job = nil
        _ = job.telemetryObserver.requestFailed(
            job.telemetryToken,
            reason: error is CancellationError ? .cancelled : .inference,
            at: Date().timeIntervalSince1970
        )
        publishProviderState(additionalObservers: [job.telemetryObserver])
        job.releasePrompt()
        job.continuation.resume(throwing: error)
    }

    private func finishPending(_ job: GenerationJob, throwing error: any Error) {
        _ = job.telemetryObserver.requestFailed(
            job.telemetryToken,
            reason: error is CancellationError ? .cancelled : .inference,
            at: Date().timeIntervalSince1970
        )
        publishProviderState(additionalObservers: [job.telemetryObserver])
        job.releasePrompt()
        job.continuation.resume(throwing: error)
    }

    private func flushPendingUTF8(_ job: GenerationJob) {
        guard !job.pendingUTF8.isEmpty else { return }
        let piece = String(decoding: job.pendingUTF8, as: UTF8.self)
        job.pendingUTF8.removeAll(keepingCapacity: false)
        if job.isRawPrompt {
            _ = processRawText(piece, for: job, tokenCount: 0)
        } else {
            guard let outputs = try? job.toolParser.consume(piece) else { return }
            _ = process(outputs, for: job, tokenCount: 0)
        }
    }

    /// Raw completion streams must not expose caller stop sequences. Retain
    /// only the suffix that could still become a stop prefix on the next token.
    private func processRawText(
        _ piece: String,
        for job: GenerationJob,
        tokenCount: Int
    ) -> Bool {
        let result = AFMDwarfStarRawStopPolicy.consume(
            buffer: &job.rawEmissionBuffer,
            piece: piece,
            stopSequences: job.request.options.stopSequences
        )
        emitRawText(result.visibleText, for: job, tokenCount: tokenCount)
        return result.stopped
    }

    private func flushRawEmissionBuffer(_ job: GenerationJob, tokenCount: Int = 0) {
        let text = AFMDwarfStarRawStopPolicy.drain(buffer: &job.rawEmissionBuffer)
        emitRawText(text, for: job, tokenCount: tokenCount)
    }

    private func emitRawText(_ text: String, for job: GenerationJob, tokenCount: Int) {
        guard !text.isEmpty else { return }
        job.generatedText += text
        job.onEvent(.responseText(action: .append, text: text, tokenCount: tokenCount))
    }

    @discardableResult
    private func process(
        _ outputs: [AFMDwarfStarToolCodec.StreamOutput],
        for job: GenerationJob,
        tokenCount: Int
    ) -> Bool {
        for output in outputs {
            switch output {
            case .text(let text):
                job.generatedText += text
                job.onEvent(.responseText(action: .append, text: text, tokenCount: tokenCount))
            case .reasoning(let text):
                job.generatedReasoning += text
                job.onEvent(.reasoningText(action: .append, text: text, tokenCount: tokenCount))
            case .toolCalls(let calls):
                job.toolCalls += calls
                for call in calls {
                    job.onEvent(.toolCall(call: call, stage: .started))
                    job.onEvent(
                        .toolCall(call: call, stage: .argumentsDelta(call.arguments))
                    )
                    job.onEvent(.toolCall(call: call, stage: .completed))
                }
                return true
            }
        }
        return false
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
        let abandonedPendingJobs = pendingJobs
        pendingJobs.removeAll(keepingCapacity: false)
        for job in abandonedPendingJobs {
            finishPending(job, throwing: CancellationError())
        }
        for index in slots.indices {
            if slots[index].job != nil {
                finish(slotIndex: index, throwing: CancellationError())
            }
            ds4_session_free(slots[index].session)
        }
        slots.removeAll(keepingCapacity: false)
        if let prefixCache { afm_ds4_prefix_cache_close(prefixCache) }
        prefixCache = nil
        if let engine { ds4_engine_close(engine) }
        engine = nil
        loadedModelPath = nil
        loadedMappingIdentity = nil
        loadedContextWindow = 0
        loadedMaxConcurrent = 0
        prefixCachingEnabled = false
        dsparkEnabled = false
        lastPrefillSlot = -1
    }

    private func publishProviderState(
        additionalObservers: [any AFMInferenceTelemetryObserving] = []
    ) {
        let activeSlots = slots.indices.filter { slots[$0].job != nil }
        let state = AFMInferenceProviderState(
            runningRequests: activeSlots.count,
            waitingRequests: pendingJobs.count,
            activeLogicalCachePositions: activeSlots.reduce(into: 0) { total, index in
                total += max(0, Int(ds4_session_pos(slots[index].session)))
            },
            logicalCacheCapacity: max(0, loadedContextWindow) * slots.count
        )
        let observers = additionalObservers
            + pendingJobs.map(\.telemetryObserver)
            + activeSlots.compactMap { slots[$0].job?.telemetryObserver }
        for observer in observers {
            observer.updateProviderState(state)
        }
    }

    private static func telemetryFinishReason(
        _ reason: AFMFinishReason
    ) -> AFMInferenceFinishReason {
        switch reason {
        case .stop, .toolCalls, .contentFilter:
            return .stop
        case .length:
            return .length
        case .cancelled:
            return .abort
        case .error, .unknown:
            return .error
        }
    }

    /// Disk serialization can synchronize GPU state. Keep it off the hot path
    /// whenever another request is queued or running; resident sessions still
    /// provide exact-prefix reuse during continuous traffic.
    private func persistPrefixIfIdle(
        slotIndex: Int,
        job: GenerationJob,
        reason: String
    ) {
        guard prefixCachingEnabled,
              pendingJobs.isEmpty,
              slots.indices.allSatisfy({ $0 == slotIndex || slots[$0].job == nil }),
              let engine,
              let prefixCache else {
            return
        }
        var error = [CChar](repeating: 0, count: 512)
        let stored = afm_ds4_prefix_cache_store_session(
            prefixCache,
            engine,
            slots[slotIndex].session,
            reason,
            &error,
            error.count)
        if stored > 0 {
            job.persistedPrefixTokens = Int(stored)
        } else if !Self.errorText(error).isEmpty {
            Self.logPrefixCache(Self.errorText(error))
        }
    }

    private static func prefixCacheDirectory(modelPath: String) throws -> String {
        let fileURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = AFMDwarfStarPrefixCachePolicy.checkpointKey(
            path: fileURL.path,
            size: size,
            modified: modified)
        let root: URL
        if let configured = ProcessInfo.processInfo.environment["AFM_DWARFSTAR_PREFIX_CACHE"],
           !configured.isEmpty {
            root = URL(fileURLWithPath: configured, isDirectory: true)
        } else {
            root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("AFM/DwarfStarPrefixCache", isDirectory: true)
        }
        return root.appendingPathComponent(key, isDirectory: true).path
    }

    private static func prefixCacheBudgetMB() -> UInt64 {
        AFMDwarfStarPrefixCachePolicy.budgetMB(
            environment: ProcessInfo.processInfo.environment)
    }

    private static func logPrefixCache(_ message: String) {
        FileHandle.standardError.write(Data("[DwarfStarPrefixCache] \(message)\n".utf8))
    }

    private static func makePrompt(
        engine: OpaquePointer,
        request: AFMRequest
    ) throws -> ds4_tokens {
        var prompt = ds4_tokens()
        afm_ds4_tokens_init(&prompt)
        if case .string(let rawPrompt)? = request.metadata["afm.rawPrompt"] {
            rawPrompt.withCString { ds4_tokenize_text(engine, $0, &prompt) }
            return prompt
        }
        ds4_chat_begin(engine, &prompt)
        do {
            let reasoningMode = AFMDwarfStarReasoningMode.resolve(metadata: request.metadata)
            if let prefix = reasoningMode.promptPrefix {
                prefix.withCString { ds4_tokenize_text(engine, $0, &prompt) }
            }
            if !request.tools.isEmpty {
                let required = request.metadata["toolCallingMode"] == .string("required")
                let toolsPrompt = try AFMDwarfStarToolCodec.systemPrompt(
                    for: request.tools,
                    toolCallingRequired: required
                )
                toolsPrompt.withCString {
                    ds4_tokenize_rendered_chat(engine, $0, &prompt)
                }
            }
            for message in request.messages {
                let text = try AFMDwarfStarToolCodec.textContent(of: message)
                message.role.rawValue.withCString { rolePointer in
                    text.withCString { textPointer in
                        ds4_chat_append_message(engine, &prompt, rolePointer, textPointer)
                    }
                }
                if message.role == .assistant {
                    let suffix = try AFMDwarfStarToolCodec.assistantReplaySuffix(for: message)
                    suffix.withCString {
                        ds4_tokenize_rendered_chat(engine, $0, &prompt)
                    }
                }
            }
            ds4_chat_append_assistant_prefix(engine, &prompt, reasoningMode.thinkMode)
            return prompt
        } catch {
            ds4_tokens_free(&prompt)
            throw error
        }
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
            (try? AFMDwarfStarToolCodec.textContent(of: message)) ?? "<non-text>"
        }
        let reasoningMode = AFMDwarfStarReasoningMode.resolve(metadata: request.metadata)
        let tokenIDs: [Int32]
        if let values = prompt.v {
            tokenIDs = (0..<Int(prompt.len)).map { Int32(values[$0]) }
        } else {
            tokenIDs = []
        }
        let line = "[DwarfStarPrompt] reasoning=\(reasoningMode.rawValue) "
            + "roles=\(roles) texts=\(texts.debugDescription) "
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
    public var reasoning: String?
    public var usage: AFMUsage
    public var toolCalls: [AFMToolCall]
    public var finishReason: AFMFinishReason
    public var metadata: [String: AFMJSONValue]

    public func response(modelID: String) -> AFMModelResponse {
        var metadata = metadata
        metadata["modelID"] = .string(modelID)
        return AFMModelResponse(
            text: text,
            reasoning: reasoning,
            toolCalls: toolCalls,
            usage: usage,
            finishReason: finishReason,
            metadata: metadata)
    }
}
