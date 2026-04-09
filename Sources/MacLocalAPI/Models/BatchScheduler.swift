import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import os

private let _batchTsFormatter: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
    f.locale = Locale(identifier: "en_US_POSIX")
    return f
}()

private func batchTs() -> String { _batchTsFormatter.string(from: Date()) }

/// Manages concurrent generation with dynamic slot allocation and request queuing.
///
/// **Phase 2: Dense Batched Decoding**
///
/// All GPU operations run in a **single serial loop**. During decode, multiple
/// sequences are packed into a single `model([B, 1])` call that returns
/// `[B, 1, vocabSize]` logits. Per-sequence sampling, detokenization, and
/// stream dispatch happen after each batched step.
///
/// Prefill runs individually per sequence (B=1), then the per-layer KV caches
/// are merged into `BatchKVCacheSimple` for batched decode. Dynamic slot
/// add/remove uses `extend()` and `filter()` on the batch cache.
actor BatchScheduler {
    struct ConstraintRuntimeConfiguration: @unchecked Sendable {
        let mode: String
        let matcherHandle: GrammarMatcherHandle?
    }

    struct ToolCallRuntimeConfiguration: @unchecked Sendable {
        let startTag: String
        let endTag: String
        let parser: String?
        let tools: [RequestTool]?
    }

    /// Default maximum concurrent generations.
    static let defaultMaxConcurrent = 8

    let maxConcurrent: Int
    private let model: any LanguageModel
    /// Tokenizer — nonisolated so concurrent requests can access without actor hop.
    nonisolated let tokenizer: Tokenizer
    private let processor: any UserInputProcessor
    private let configuration: ModelConfiguration
    private let cacheProfilePath: String?

    /// EOS token IDs built once at init.
    private let eosTokenIds: Set<Int>

    /// Shared prefix cache for all slots (safe: all access is serialized inside this actor).
    private let radixCache: RadixTreeCache?

    // MARK: - Slot State

    /// Per-request state for batched generation.
    /// Each slot holds its own sampler/processor (since each request can have
    /// different temperature, top_p, penalties etc.).
    private class SlotState {
        let id: UUID
        let requestId: String
        let continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
        let promptTokenCount: Int
        /// Set to the moment prefill begins (prefillStart), so elapsed includes prefill + decode.
        let startTime: Date
        let prefillTime: TimeInterval
        var tokenCount = 0
        var firstTokenTime: TimeInterval = 0
        let inputTokens: [Int]
        let cachedTokens: Int
        /// Snapshotted per-layer KV state from prefill (for prefix cache save).
        /// Stored as arrays rather than live KVCache references to avoid mutation
        /// by the decode loop (batchCaches shares objects in .unbatched mode).
        let prefillCaches: [KVCache]
        let prefillStates: [[MLXArray]]
        let prefillMetaStates: [[String]]

        // Per-sequence decode state
        var lastTokenId: Int
        /// The sampled token as an MLXArray — used directly as model input on the
        /// next decode step (avoids Int→MLXArray roundtrip and enables overlap).
        /// Set from the evaluated first token during prefill, then from the lazy
        /// sampled token after each decode step.
        var lastTokenArray: MLXArray
        let sampler: LogitSampler
        var processor: LogitProcessor?
        var detokenizer: NaiveStreamingDetokenizer
        let maxTokens: Int?
        let toolRuntime: ToolCallStreamingRuntime?
        let constraintRuntime: ConstraintRuntimeConfiguration?

        /// Lazy token array from the previous decode step (for deferred dispatch).
        /// Set after asyncEval; materialized via .item() after the NEXT model call.
        var pendingTokenArray: MLXArray? = nil
        /// Per-slot logprob configuration
        let computeLogprobs: Bool
        let topLogprobsCount: Int
        let temperatureForLogprobs: Float
        /// Buffered logprob data pending dispatch (one token behind, like single-sequence path)
        var pendingLogprobData: TokenLogprobData? = nil
        let activeStops: [String]
        let maxStopLength: Int
        let thinkStartTag: String?
        let thinkEndTag: String?
        var stopBuffer = ""
        var insideThink = false
        var stoppedBySequence = false
        /// Set to true to signal this slot should stop generating.
        var isCancelled = false

        init(
            requestId: String = "",
            continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation,
            promptTokenCount: Int,
            startTime: Date,
            prefillTime: TimeInterval,
            inputTokens: [Int],
            cachedTokens: Int,
            prefillCaches: [KVCache],
            lastTokenId: Int,
            lastTokenArray: MLXArray,
            sampler: LogitSampler,
            processor: LogitProcessor?,
            detokenizer: NaiveStreamingDetokenizer,
            maxTokens: Int?,
            toolRuntime: ToolCallStreamingRuntime?,
            constraintRuntime: ConstraintRuntimeConfiguration?,
            activeStops: [String],
            thinkStartTag: String?,
            thinkEndTag: String?,
            computeLogprobs: Bool = false,
            topLogprobsCount: Int = 0,
            temperatureForLogprobs: Float = 1.0
        ) {
            self.id = UUID()
            self.requestId = requestId
            self.continuation = continuation
            self.promptTokenCount = promptTokenCount
            self.prefillTime = prefillTime
            self.startTime = startTime
            self.inputTokens = inputTokens
            self.cachedTokens = cachedTokens
            self.prefillCaches = prefillCaches
            // Snapshot prefill KV state for radix cache save.
            // Must contiguous-copy the arrays — MLX sliced views share the underlying
            // buffer with batchCaches, so batched decode mutations would corrupt the snapshot.
            self.prefillStates = prefillCaches.map { cache in
                cache.state.map { MLX.contiguous($0) }
            }
            self.prefillMetaStates = prefillCaches.map { $0.metaState }
            self.lastTokenId = lastTokenId
            self.lastTokenArray = lastTokenArray
            self.sampler = sampler
            self.processor = processor
            self.detokenizer = detokenizer
            self.maxTokens = maxTokens
            self.toolRuntime = toolRuntime
            self.constraintRuntime = constraintRuntime
            self.activeStops = activeStops
            self.maxStopLength = activeStops.map(\.count).max() ?? 0
            self.thinkStartTag = thinkStartTag
            self.thinkEndTag = thinkEndTag
            self.computeLogprobs = computeLogprobs
            self.topLogprobsCount = topLogprobsCount
            self.temperatureForLogprobs = temperatureForLogprobs
        }
    }

    /// Active slots in the batched decode loop.
    private var slots: [SlotState] = []

    /// Per-layer batch caches. Each element is a `BatchKVCacheSimple` with
    /// `batchSize == slots.count`. Empty when no slots are active.
    private var batchCaches: [KVCache] = []

    /// Model output state (e.g. cross-attention states for VLMs).
    /// For most LLMs (including Qwen3.5-MoE), this is nil — recurrent
    /// state lives inside the KV cache, not in LMOutput.State.
    private var batchState: LMOutput.State? = nil

    /// Total tokens generated across all slots (for periodic cache clearing).
    private var totalTokensGenerated = 0

    private func hasRecurrentLayers(_ cache: [KVCache]) -> Bool {
        cache.contains { $0 is ArraysCache || $0 is CacheList }
    }

    private func unsafeExactReplaySuffix() -> Int? {
        let env = ProcessInfo.processInfo.environment
        guard let rawValue = env["AFM_PREFIX_CACHE_ALLOW_UNSAFE_EXACT_REPLAY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !rawValue.isEmpty else {
            return nil
        }
        let value = rawValue.lowercased()
        if value == "true" || value == "yes" {
            return 1
        }
        if let suffix = Int(value), suffix >= 1 {
            return suffix
        }
        return nil
    }

    private func supportsPhysicalTruncation(_ cache: KVCache) -> Bool {
        !(cache is RotatingKVCache)
    }

    private func restoredMetaState(for cache: KVCache, savedMetaState: [String]?) -> [String]? {
        guard let savedMetaState else { return nil }
        if cache is RotatingKVCache, savedMetaState.count == 5 {
            return savedMetaState
        }
        return savedMetaState
    }

    private func shouldTraceReplayBoundary() -> Bool {
        let env = ProcessInfo.processInfo.environment
        let value = env["AFM_PREFIX_CACHE_TRACE_BOUNDARY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return value == "1" || value == "true" || value == "yes"
    }

    private func logReplayBoundary(inputTokens: [Int], effectivePrefix: Int) {
        guard shouldTraceReplayBoundary() else { return }

        let split = max(0, min(effectivePrefix, inputTokens.count))
        let prefixTail = Array(inputTokens[max(0, split - 8)..<split])
        let suffixHead = Array(inputTokens[split..<min(inputTokens.count, split + 16)])
        let prefixTailDecoded = prefixTail.isEmpty ? "" : tokenizer.decode(tokens: prefixTail)
        let suffixHeadDecoded = suffixHead.isEmpty ? "" : tokenizer.decode(tokens: suffixHead)

        print(
            "[\(batchTs())] [PrefixCache] Boundary: mode=batch | prefix_tokens=\(split) | " +
                "suffix_tokens=\(inputTokens.count - split) | prefix_tail_ids=\(prefixTail) | " +
                "suffix_head_ids=\(suffixHead) | prefix_tail_decoded=\(String(reflecting: prefixTailDecoded)) | " +
                "suffix_head_decoded=\(String(reflecting: suffixHeadDecoded))"
        )
    }

    private func logCacheProfile(
        phase: String,
        mode: String,
        outcome: String,
        inputTokenCount: Int,
        cachedTokenCount: Int,
        promptTime: Double,
        lookupTime: Double? = nil,
        restoreTime: Double? = nil,
        trimTime: Double? = nil,
        truncateTime: Double? = nil,
        insertTime: Double? = nil
    ) {
        let lookup = lookupTime ?? 0
        let restore = restoreTime ?? 0
        let trim = trimTime ?? 0
        let truncate = truncateTime ?? 0
        let insert = insertTime ?? 0
        let cacheOverhead = lookup + restore + trim + truncate + insert
        let reuseRatio = inputTokenCount > 0 ? Double(cachedTokenCount) / Double(inputTokenCount) : 0
        let overheadShare = promptTime > 0 ? cacheOverhead / promptTime : 0
        func formatCacheSeconds(_ value: Double?) -> String {
            String(format: "%.6f", value ?? 0)
        }

        print(
            "[\(batchTs())] [CacheProfile] phase=\(phase) | mode=\(mode) | outcome=\(outcome) | " +
                "input_tokens=\(inputTokenCount) | cached_tokens=\(cachedTokenCount) | " +
                "reuse_ratio=\(String(format: "%.3f", reuseRatio)) | " +
                "lookup=\(formatCacheSeconds(lookupTime))s | " +
                "restore=\(formatCacheSeconds(restoreTime))s | " +
                "trim=\(formatCacheSeconds(trimTime))s | " +
                "truncate=\(formatCacheSeconds(truncateTime))s | " +
                "insert=\(formatCacheSeconds(insertTime))s | " +
                "cache_overhead=\(formatCacheSeconds(cacheOverhead))s | " +
                "prompt_time=\(String(format: "%.3f", promptTime))s | " +
                "overhead_share=\(String(format: "%.6f", overheadShare))"
        )

        let environment = ProcessInfo.processInfo.environment
        guard let path = cacheProfilePath ?? environment["MACAFM_CACHE_PROFILE_PATH"] ?? environment["AFM_CACHE_PROFILE_PATH"] else {
            return
        }

        CacheProfileExporter.append(record: [
            "timestamp": batchTs(),
            "phase": phase,
            "mode": mode,
            "outcome": outcome,
            "input_tokens": inputTokenCount,
            "cached_tokens": cachedTokenCount,
            "reuse_ratio": reuseRatio,
            "lookup_s": lookup,
            "restore_s": restore,
            "trim_s": trim,
            "truncate_s": truncate,
            "insert_s": insert,
            "cache_overhead_s": cacheOverhead,
            "prompt_time_s": promptTime,
            "overhead_share": overheadShare,
        ], to: path)
    }

    struct PendingRequest: @unchecked Sendable {
        let requestId: String
        let input: LMInput
        let parameters: GenerateParameters
        let promptTokens: Int
        let toolCallRuntimeConfig: ToolCallRuntimeConfiguration?
        let constraintRuntimeConfig: ConstraintRuntimeConfiguration?
        let stopSequences: [String]
        let thinkStartTag: String?
        let thinkEndTag: String?
        let continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
    }

    /// Thread-safe request queue — accessed without actor isolation.
    /// submit() pushes here (nonisolated); generationLoop() drains (actor-isolated).
    private let _pendingQueue = OSAllocatedUnfairLock(initialState: [PendingRequest]())

    /// Thread-safe shutdown flag — accessed without actor isolation.
    private let _isShutdown = OSAllocatedUnfairLock(initialState: false)

    /// Thread-safe in-flight counter (pending + active). Incremented in submit(),
    /// decremented in finishSlot() and error paths. Used for capacity checks.
    private let _inFlightCount = OSAllocatedUnfairLock(initialState: 0)

    /// Atomically reserve a slot if under capacity. Returns true if reserved.
    nonisolated func tryReserve() -> Bool {
        _inFlightCount.withLock { count in
            if count >= maxConcurrent { return false }
            count += 1
            return true
        }
    }

    /// Exponential backoff constants for slot polling.
    private static let initialPollInterval: UInt64 = 10_000_000   // 10ms
    private static let maxPollInterval: UInt64     = 500_000_000  // 500ms

    /// Wait up to `timeout` seconds for a slot to become available.
    /// Returns true if a slot was reserved, false if timed out, cancelled, or shutdown.
    nonisolated func waitForSlot(timeout: TimeInterval = 30) async -> Bool {
        if Task.isCancelled || _isShutdown.withLock({ $0 }) { return false }
        if tryReserve() { return true }

        let deadline = ContinuousClock.now + .seconds(timeout)
        var delay = Self.initialPollInterval

        while ContinuousClock.now < deadline {
            if Task.isCancelled || _isShutdown.withLock({ $0 }) { return false }
            try? await Task.sleep(nanoseconds: delay)
            if Task.isCancelled || _isShutdown.withLock({ $0 }) { return false }
            if tryReserve() { return true }
            delay = min(delay * 2, Self.maxPollInterval)
        }
        return false
    }

    /// Release a reserved slot (call if request fails before reaching submit).
    nonisolated func releaseReservation() {
        _inFlightCount.withLock { $0 = max($0 - 1, 0) }
    }

    /// Atomically reserve N slots. Returns true if all N were reserved,
    /// false if insufficient capacity (no slots reserved in that case).
    nonisolated func tryReserveMultiple(count: Int) -> Bool {
        _inFlightCount.withLock { current in
            if current + count <= maxConcurrent {
                current += count
                return true
            }
            return false
        }
    }

    /// Release N slot reservations at once.
    nonisolated func releaseMultipleReservations(count: Int) {
        _inFlightCount.withLock { current in
            current = max(0, current - count)
        }
    }

    /// Current number of active + pending slots (for teardown decisions).
    nonisolated var activeSlotCount: Int {
        _inFlightCount.withLock { $0 }
    }

    /// Cancel all slots matching the given IDs.
    func cancelSlots(ids: Set<UUID>) {
        for slot in slots where ids.contains(slot.id) {
            slot.isCancelled = true
        }
    }

    private var loopTask: Task<Void, Never>?

    /// Tracks concrete type in batchCaches to avoid per-step type checks.
    private enum CacheMode {
        case empty      // batchCaches is []
        case unbatched  // batchCaches contains KVCacheSimple (B=1, zero overhead)
        case batched    // batchCaches contains BatchKVCacheSimple (B≥1)
    }
    private var cacheMode: CacheMode = .empty

    init(
        model: any LanguageModel,
        tokenizer: Tokenizer,
        processor: any UserInputProcessor,
        configuration: ModelConfiguration,
        maxConcurrent: Int = BatchScheduler.defaultMaxConcurrent,
        enablePrefixCaching: Bool = false,
        cacheProfilePath: String? = nil
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.configuration = configuration
        self.maxConcurrent = maxConcurrent
        self.cacheProfilePath = cacheProfilePath

        let debug = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
        self.radixCache = RadixTreeCache(
            modelID: configuration.name,
            maxEntries: 64,
            debugLogging: debug
        )

        var eos = configuration.eosTokenIds
        if let tokenizerEos = tokenizer.eosTokenId {
            eos.insert(tokenizerEos)
        }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                eos.insert(id)
            }
        }
        self.eosTokenIds = eos

        // Wire gauge readers into the global stats aggregator. `num_running`
        // is (inflight - waiting), since _inFlightCount covers both actively
        // decoding and still-pending requests. `num_waiting` reads the
        // pending queue directly. Both closures read nonisolated state
        // through existing unfair locks — no actor hop, no async.
        //
        // These closures hold weak references to `self` via the unowned
        // nonisolated lock pointers, so they do not extend scheduler
        // lifetime beyond MLXModelService's hold.
        let pending = self._pendingQueue
        let inflight = self._inFlightCount
        StatsAggregator.shared.registerGaugeReaders(
            running: {
                let total = inflight.withLock { $0 }
                let queued = pending.withLock { $0.count }
                return max(0, total - queued)
            },
            waiting: {
                return pending.withLock { $0.count }
            }
        )
    }

    /// Number of requests currently generating.
    var activeCount: Int { slots.count }

    /// Prepare UserInput into LMInput (tokenization + chat template).
    /// Actor-isolated — processor.prepare may not be thread-safe despite Sendable conformance.
    func prepareInput(_ userInput: UserInput) async throws -> LMInput {
        try await processor.prepare(input: userInput)
    }

    /// Submit a generation request. Returns a stream of StreamChunks immediately.
    /// Nonisolated — no actor hop needed. Pushes to lock-protected queue.
    nonisolated func submit(
        input: LMInput,
        parameters: GenerateParameters,
        promptTokens: Int,
        toolCallRuntimeConfig: ToolCallRuntimeConfiguration? = nil,
        constraintRuntimeConfig: ConstraintRuntimeConfiguration? = nil,
        stopSequences: [String] = [],
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        requestId: String = ""
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        if _isShutdown.withLock({ $0 }) {
            return AsyncThrowingStream { $0.finish(throwing: MLXServiceError.serviceShuttingDown) }
        }

        let (stream, continuation) = AsyncThrowingStream<StreamChunk, Error>.makeStream()

        // Note: slot already reserved by tryReserve() in the controller layer.
        _pendingQueue.withLock {
            $0.append(PendingRequest(
                requestId: requestId,
                input: input,
                parameters: parameters,
                promptTokens: promptTokens,
                toolCallRuntimeConfig: toolCallRuntimeConfig,
                constraintRuntimeConfig: constraintRuntimeConfig,
                stopSequences: stopSequences,
                thinkStartTag: thinkStartTag,
                thinkEndTag: thinkEndTag,
                continuation: continuation
            ))
        }

        StatsAggregator.shared.requestStarted()
        DebugLogger.log("[BatchScheduler] Request enqueued req=\(requestId) (\(_inFlightCount.withLock { $0 })/\(maxConcurrent))")
        Task { await self.ensureLoopRunning() }

        return stream
    }

    /// Drain all pending requests from the nonisolated queue.
    /// Called from within the actor-isolated generationLoop.
    private func drainPendingQueue() -> [PendingRequest] {
        _pendingQueue.withLock { q in
            let result = q
            q.removeAll()
            return result
        }
    }

    /// Gracefully shut down.
    func shutdown() async {
        _isShutdown.withLock { $0 = true }

        // Drain and cancel any pending requests
        let pending = _pendingQueue.withLock { q in
            let result = q; q.removeAll(); return result
        }
        for req in pending {
            req.continuation.finish(throwing: MLXServiceError.serviceShuttingDown)
        }

        if let task = loopTask {
            task.cancel()
            await task.value
            loopTask = nil
        }

        for slot in slots {
            slot.continuation.finish(throwing: MLXServiceError.serviceShuttingDown)
            slot.constraintRuntime?.matcherHandle?.release()
        }
        // Reset in-flight counter
        _inFlightCount.withLock { $0 = 0 }
        slots.removeAll()
        batchCaches = []
        batchState = nil
        cacheMode = .empty
    }

    // MARK: - Private

    private func ensureLoopRunning() {
        guard loopTask == nil else { return }
        loopTask = Task { [weak self] in
            await self?.generationLoop()
        }
    }

    /// Main generation loop: prefill pending → batched decode → dispatch.
    /// ALL GPU operations happen here — never from concurrent Tasks.
    ///
    /// **Pipelined decode with CPU-GPU overlap** (extends serial TokenIterator pattern):
    /// 1. Build computation graph lazily: model(lastTokenArray) → sample (CPU, ~2.7ms)
    /// 2. asyncEval — GPU starts computing
    /// 3. While GPU runs: dispatch PREVIOUS step's tokens (.item() instant, detokenize, yield)
    /// 4. Update lastTokenArray + stash for next iteration, loop back to 1
    ///
    /// This overlaps ~1-2ms of CPU dispatch work with GPU compute that would
    /// otherwise be idle time. The model input uses `lastTokenArray` (MLXArray)
    /// directly — no Int→MLXArray roundtrip needed.
    private func generationLoop() async {
        var stepCount = 0
        // Previous step's sampled tokens + slot IDs + logits, for deferred dispatch.
        // nil on the first iteration (newly prefilled slots have no pending work).
        var dispatchTokens: [MLXArray]? = nil
        var dispatchSlotIDs: [UUID]? = nil
        var dispatchLogits: [MLXArray?]? = nil

        // Decode-step timing accumulators (debug only)
        let debugTiming = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
        var stepTimeAccum: Double = 0
        var modelTimeAccum: Double = 0
        var dispatchTimeAccum: Double = 0
        var timingStepCount = 0

        // For top-of-loop gap detection: wall-clock of the previous iteration-top
        // and per-phase wall times of the previous iteration so we can attribute
        // any gap to model() / dispatch() / housekeeping.
        var _dbgPrevLoopTopAt: Date? = nil
        var _dbgPrevModelWall: Double = 0
        var _dbgPrevDispatchWall: Double = 0
        var _dbgPrevWasMerge: Bool = false
        while !_isShutdown.withLock({ $0 }) {
            let stepStart = debugTiming ? Date() : Date.distantPast
            if debugTiming {
                if let prev = _dbgPrevLoopTopAt {
                    let gap = stepStart.timeIntervalSince(prev)
                    if gap > 0.500 {
                        let other = max(0, gap - _dbgPrevModelWall - _dbgPrevDispatchWall)
                        let merge = _dbgPrevWasMerge ? "yes" : "no"
                        print("[\(batchTs())] [BatchScheduler] LOOP GAP: B=\(slots.count), pending=\(_pendingQueue.withLock { $0.count }), gap=\(String(format: "%.3f", gap))s | model=\(String(format: "%.3f", _dbgPrevModelWall))s dispatch=\(String(format: "%.3f", _dbgPrevDispatchWall))s other=\(String(format: "%.3f", other))s merge=\(merge)")
                    }
                }
                _dbgPrevLoopTopAt = stepStart
                _dbgPrevWasMerge = false
            }

            // Drain nonisolated queue — no Task.yield() needed for request pickup
            let newRequests = drainPendingQueue()
            if debugTiming && !newRequests.isEmpty {
                _dbgPrevWasMerge = true
            }
            if !newRequests.isEmpty {
                // Separate capacity-limited requests from overflow
                var accepted: [PendingRequest] = []
                for req in newRequests {
                    if slots.count + accepted.count < maxConcurrent {
                        accepted.append(req)
                    } else {
                        _pendingQueue.withLock { $0.append(req) }
                    }
                }

                if accepted.count > 1 {
                    // Batch all non-multimodal requests together. Prefix cache hits
                    // are skipped inside prefillBatch (fresh caches only). The batch
                    // speedup from a single B=N forward pass outweighs small partial
                    // cache hits (e.g. shared chat template prefix).
                    var batchEligible: [PendingRequest] = []
                    var individual: [PendingRequest] = []

                    for req in accepted {
                        let isMultimodal = req.input.image != nil || req.input.video != nil
                        if isMultimodal {
                            individual.append(req)
                        } else {
                            batchEligible.append(req)
                        }
                    }

                    for req in individual { prefillOne(req) }

                    if batchEligible.count > 1 {
                        prefillBatch(batchEligible)
                    } else if let req = batchEligible.first {
                        prefillOne(req)
                    }
                } else if let req = accepted.first {
                    prefillOne(req)
                }
            }

            if slots.isEmpty { break }
            if Task.isCancelled {
                // Drain any in-flight GPU work before exiting so synchronize() won't trap.
                Stream.gpu.synchronize()
                return
            }

            // --- Batched decode: build graph from lastTokenArray (CPU, ~2.7ms) ---
            let activeB = slots.count
            let tokens: MLXArray
            if activeB == 1 {
                tokens = slots[0].lastTokenArray.reshaped([1, 1])
            } else {
                tokens = stacked(slots.map { $0.lastTokenArray.reshaped([1]) }).reshaped([activeB, 1])
            }
            let input = LMInput.Text(tokens: tokens)

            let modelStart = debugTiming ? Date() : Date.distantPast
            let output = model(input, cache: batchCaches, state: batchState)
            batchState = output.state

            let logits = output.logits[0..., -1, 0...]

            // Lazy sampling — build graph without eval
            // Also store processed logits per-slot for logprob computation during dispatch
            var tokenArrays = [MLXArray]()
            var slotLogitsForLogprobs = [MLXArray?]()
            for i in 0..<activeB {
                let slot = slots[i]
                let slotLogits = activeB == 1 ? logits : logits[i]
                let processed = slot.processor?.process(logits: slotLogits) ?? slotLogits
                let sampled = slot.sampler.sample(logits: processed)
                tokenArrays.append(sampled)
                slotLogitsForLogprobs.append(slot.computeLogprobs ? processed : nil)
            }

            // Kick off async evaluation — GPU starts computing current step
            asyncEval(tokenArrays)
            if debugTiming {
                let modelW = Date().timeIntervalSince(modelStart)
                modelTimeAccum += modelW
                _dbgPrevModelWall = modelW
            }

            // Update lastTokenArray BEFORE dispatch (indices still match tokenArrays).
            // These lazy MLXArrays are being asyncEval'd — by next iteration they'll
            // be evaluated and ready for model input.
            for i in 0..<activeB {
                slots[i].lastTokenArray = tokenArrays[i]
            }

            // Save previous dispatch info, then stash current step (while indices match).
            // Must happen before dispatch can remove slots and shift indices.
            let toDispatchTokens = dispatchTokens
            let toDispatchSlotIDs = dispatchSlotIDs
            let toDispatchLogits = dispatchLogits
            dispatchTokens = tokenArrays
            dispatchSlotIDs = slots.map { $0.id }  // order matches tokenArrays
            dispatchLogits = slotLogitsForLogprobs

            // --- While GPU runs: dispatch PREVIOUS step's tokens ---
            // .item() is instant because these were asyncEval'd last iteration.
            let dispatchStart = debugTiming ? Date() : Date.distantPast
            if let prevTokens = toDispatchTokens, let prevIDs = toDispatchSlotIDs {
                var completedIndices: [Int] = []
                for j in 0..<prevIDs.count {
                    guard let i = slots.firstIndex(where: { $0.id == prevIDs[j] }) else { continue }
                    let slot = slots[i]

                    // Check for per-slot cancellation
                    if slot.isCancelled {
                        completedIndices.append(i)
                        continue
                    }

                    let tokenArray = prevTokens[j]

                    let token = tokenArray.item(Int.self)
                    slot.processor?.didSample(token: tokenArray)

                    // Promote pending logprobs (one token behind — describes THIS token being dispatched)
                    let logprobsForThisToken: [ResolvedLogprob]?
                    if let pending = slot.pendingLogprobData {
                        logprobsForThisToken = [resolveLogprob(pending)]
                    } else {
                        logprobsForThisToken = nil
                    }

                    // Compute logprobs for the NEXT dispatch from this step's logits
                    if slot.computeLogprobs, let prevLogits = toDispatchLogits?[j] {
                        slot.pendingLogprobData = computeTokenLogprobs(
                            logits: prevLogits, tokenId: token,
                            temperature: slot.temperatureForLogprobs,
                            topK: slot.topLogprobsCount
                        )
                    } else if slot.computeLogprobs {
                        slot.pendingLogprobData = nil
                    }

                    if slot.firstTokenTime == 0 {
                        slot.firstTokenTime = Date().timeIntervalSince(slot.startTime)
                    }

                    if token == tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                        completedIndices.append(i)
                        continue
                    }

                    if let max = slot.maxTokens, slot.tokenCount >= max {
                        completedIndices.append(i)
                        continue
                    }

                    slot.lastTokenId = token

                    slot.detokenizer.append(token: token)
                    if let chunk = slot.detokenizer.next() {
                        slot.tokenCount += 1
                        StatsAggregator.shared.addGenTokens(1)
                        if yieldTextChunk(chunk, for: slot, logprobs: logprobsForThisToken) {
                            completedIndices.append(i)
                        }
                    }
                }

                for i in completedIndices.sorted().reversed() {
                    finishSlot(at: i)
                }
            }

            if debugTiming {
                let dispatchW = Date().timeIntervalSince(dispatchStart)
                dispatchTimeAccum += dispatchW
                _dbgPrevDispatchWall = dispatchW
                stepTimeAccum += Date().timeIntervalSince(stepStart)
                timingStepCount += 1
                if timingStepCount % 200 == 0 {
                    let avgStep = stepTimeAccum / Double(timingStepCount) * 1000
                    let avgModel = modelTimeAccum / Double(timingStepCount) * 1000
                    let avgDispatch = dispatchTimeAccum / Double(timingStepCount) * 1000
                    DebugLogger.log("[BatchScheduler] Decode timing (\(timingStepCount) steps, B=\(activeB)): step=\(String(format: "%.2f", avgStep))ms model=\(String(format: "%.2f", avgModel))ms dispatch=\(String(format: "%.2f", avgDispatch))ms")
                }
            }

            totalTokensGenerated += activeB
            if totalTokensGenerated % 1024 < activeB {
                let clearStart = debugTiming ? Date() : Date.distantPast
                Memory.clearCache()
                if debugTiming {
                    let clearWall = Date().timeIntervalSince(clearStart)
                    if clearWall > 0.010 {
                        print("[\(batchTs())] [BatchScheduler] MEMORY CLEARCACHE: B=\(slots.count), wall=\(String(format: "%.3f", clearWall))s")
                    }
                }
            }

            // Periodically eval cache arrays to collapse the lazy compute graph.
            // Without this, slice-assignment in cache.update() accumulates ~120 new
            // Metal buffers per step (60 layers × 2 K/V). At the OS limit of 499K
            // buffers, the server crashes after ~4000 steps. Eval every 512 steps
            // materializes the arrays and releases graph intermediates.
            if stepCount % 512 == 0 {
                let flushStart = debugTiming ? Date() : Date.distantPast
                eval(batchCaches.flatMap { $0.innerState() })
                if debugTiming {
                    let flushWall = Date().timeIntervalSince(flushStart)
                    print("[\(batchTs())] [BatchScheduler] EVAL FLUSH (512-step): B=\(slots.count), wall=\(String(format: "%.3f", flushWall))s")
                }
            }

            // Yield rarely — just for cooperative scheduling / graceful shutdown.
            // submit() no longer needs actor hop, so no yield needed for request pickup.
            stepCount += 1
            if stepCount % 256 == 0 {
                await Task.yield()
            }
        }

        // Flush: dispatch any remaining tokens from the last step
        if let prevTokens = dispatchTokens, let prevIDs = dispatchSlotIDs {
            for j in 0..<prevIDs.count {
                guard let i = slots.firstIndex(where: { $0.id == prevIDs[j] }) else { continue }
                let slot = slots[i]
                let token = prevTokens[j].item(Int.self)
                slot.processor?.didSample(token: prevTokens[j])

                // Promote pending logprobs for this token
                let logprobsForToken: [ResolvedLogprob]?
                if let pending = slot.pendingLogprobData {
                    logprobsForToken = [resolveLogprob(pending)]
                    slot.pendingLogprobData = nil
                } else {
                    logprobsForToken = nil
                }

                if token == tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                    finishSlot(at: i)
                    continue
                }
                if let max = slot.maxTokens, slot.tokenCount >= max {
                    finishSlot(at: i)
                    continue
                }

                slot.lastTokenId = token
                slot.detokenizer.append(token: token)
                if let chunk = slot.detokenizer.next() {
                    slot.tokenCount += 1
                    if yieldTextChunk(chunk, for: slot, logprobs: logprobsForToken) {
                        finishSlot(at: i)
                        continue
                    }
                }
            }
        }

        loopTask = nil
    }

    // MARK: - Prefill

    /// Prefill a single request (B=1), then merge its cache into the batch.
    /// Prefill a single request (B=1 forward pass).
    /// When `deferMerge` is true, the cache is appended to `deferredCaches`
    /// instead of being immediately merged into the decode batch. The caller
    /// is responsible for batch-merging all deferred caches afterward.
    @discardableResult
    private func prefillOne(
        _ req: PendingRequest,
        deferMerge: Bool = false,
        deferredCaches: UnsafeMutablePointer<[([KVCache], LMOutput.State?)]>? = nil
    ) -> Void {
        var cache = model.newCache(parameters: req.parameters)
        var generateInput = req.input
        var cachedTokens = 0
        var cacheOutcome = "miss"
        var cacheLookupTime: Double? = nil
        var cacheRestoreTime: Double? = nil
        var cacheTrimTime: Double? = nil
        var cacheTruncateTime: Double? = nil

        // Extract token array for prefix cache lookup
        let inputTokens = req.input.text.tokens.reshaped(-1).asArray(Int.self)
        let isMultimodal = req.input.image != nil || req.input.video != nil
        if isMultimodal {
            cacheOutcome = "multimodal-skip"
        }

        // Prefix cache: restore KV state if available
        if !isMultimodal, let radix = radixCache {
            let tLookup0 = Date.timeIntervalSinceReferenceDate
            let (prefixLen, layerStates, layerMetaStates) = radix.findPrefix(inputTokens)
            let tLookup1 = Date.timeIntervalSinceReferenceDate
            cacheLookupTime = tLookup1 - tLookup0
            let forcedSuffix = unsafeExactReplaySuffix()
            let effectivePrefix: Int
            if prefixLen == inputTokens.count && hasRecurrentLayers(cache) && forcedSuffix == nil {
                effectivePrefix = 0
                if prefixLen > 0 {
                    cacheOutcome = "exact-replay-bypass"
                }
            } else if prefixLen == inputTokens.count, let forcedSuffix {
                effectivePrefix = max(0, inputTokens.count - forcedSuffix)
            } else {
                let minSuffix = 16
                effectivePrefix = min(prefixLen, max(0, inputTokens.count - minSuffix))
            }

            if effectivePrefix > 0, let states = layerStates {
                let tRestore0 = Date.timeIntervalSinceReferenceDate
                for i in 0..<cache.count where i < states.count {
                    cache[i].state = states[i]
                    let savedMetaState = layerMetaStates.flatMap { i < $0.count ? $0[i] : nil }
                    if let adjustedMetaState = restoredMetaState(
                        for: cache[i],
                        savedMetaState: savedMetaState
                    ) {
                        cache[i].metaState = adjustedMetaState
                    }
                }
                let tRestore1 = Date.timeIntervalSinceReferenceDate
                for i in 0..<cache.count {
                    let excess = cache[i].offset - effectivePrefix
                    if excess > 0 { cache[i].trim(excess) }
                }
                let tTrim = Date.timeIntervalSinceReferenceDate
                // Physically truncate trimmed cache arrays to eliminate stale data. (#47)
                for i in 0..<cache.count {
                    if cache[i].isTrimmable && cache[i].offset > 0
                        && supportsPhysicalTruncation(cache[i])
                    {
                        cache[i].truncateToOffset()
                    }
                }
                let tRoundtrip = Date.timeIntervalSinceReferenceDate
                let suffixTokens = Array(inputTokens[effectivePrefix...])
                generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                cachedTokens = effectivePrefix
                cacheOutcome = "hit"
                cacheRestoreTime = tRestore1 - tRestore0
                cacheTrimTime = tTrim - tRestore1
                cacheTruncateTime = tRoundtrip - tTrim
                logReplayBoundary(inputTokens: inputTokens, effectivePrefix: effectivePrefix)
                DebugLogger.log("[BatchScheduler] Prefix cache hit: \(effectivePrefix)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix")
                print("[\(batchTs())] [PrefixCache] Prefill: mode=batch | outcome=hit | input_tokens=\(inputTokens.count) | cached_tokens=\(effectivePrefix) | suffix_tokens=\(suffixTokens.count) | radix_entries=\(radix.count) | lookup=\(String(format: "%.6f", cacheLookupTime ?? 0))s | restore=\(String(format: "%.6f", cacheRestoreTime ?? 0))s | trim=\(String(format: "%.6f", cacheTrimTime ?? 0))s | truncate=\(String(format: "%.6f", cacheTruncateTime ?? 0))s")
            }
        }

        let prefillStart = Date()

        // Set up per-sequence processor and sampler
        var logitProcessor = req.parameters.processor()
        let logitSampler = req.parameters.sampler()
        logitProcessor?.prompt(generateInput.text.tokens)

        // Run model on full prompt (B=1)
        let prefillInput = generateInput.text[text: .newAxis]  // [1, seqLen]
        let result = model(prefillInput, cache: cache, state: nil)

        // Extract last-position logits and sample first token.
        // Use [0, -1, 0...] to collapse batch dim → [vocabSize] (scalar sample output).
        // This matches decode path (logits[i] → [vocabSize]) so stacked() shapes agree.
        let logits = result.logits[0, -1, 0...]
        let processed = logitProcessor?.process(logits: logits) ?? logits
        let tokenArray = logitSampler.sample(logits: processed)
        logitProcessor?.didSample(token: tokenArray)

        // Materialize cache and token
        var evalArrays: [MLXArray] = [tokenArray]
        evalArrays.append(contentsOf: cache.flatMap { $0.innerState() })
        if let s = result.state?.crossAttentionStates { evalArrays.append(s) }
        eval(evalArrays)

        let firstToken = tokenArray.item(Int.self)
        let prefillTime = Date().timeIntervalSince(prefillStart)

        // RotatingKVCache models with keep>0 (e.g. some Mistral variants) must stay serial.
        // Models with keep=0 (Gemma 4) can batch via BatchRotatingKVCache.
        let hasUnbatchableRotating = cache.contains { c in
            if let rc = c as? RotatingKVCache { return !rc.isBatchable }
            return false
        }
        if hasUnbatchableRotating {
            print("[\(batchTs())] [BatchScheduler] RotatingKVCache (keep>0) detected — running serial")
            runSerialGeneration(
                req: req, cache: cache, modelState: result.state,
                firstToken: tokenArray, inputTokens: inputTokens,
                cachedTokens: cachedTokens, prefillStart: prefillStart, prefillTime: prefillTime,
                cacheOutcome: cacheOutcome, cacheLookupTime: cacheLookupTime,
                cacheRestoreTime: cacheRestoreTime, cacheTrimTime: cacheTrimTime,
                cacheTruncateTime: cacheTruncateTime,
                logitProcessor: logitProcessor, logitSampler: logitSampler
            )
            return
        }

        if deferMerge && cacheMode == .batched, let dc = deferredCaches {
            // Defer the merge — caller will batch-merge all deferred caches at once
            dc.pointee.append((cache, result.state))
        } else {
            mergeCacheIntoBatch(individualCache: cache, modelState: result.state)
        }

        // Only count the un-cached suffix in prompt_tokens_total so the
        // counter reflects tokens the GPU actually prefilled, not tokens
        // served from the radix cache. This matches vLLM semantics.
        let suffixPromptTokens = max(0, inputTokens.count - cachedTokens)
        StatsAggregator.shared.addPromptTokens(suffixPromptTokens)
        if cachedTokens > 0 {
            StatsAggregator.shared.cacheHit()
        } else if !isMultimodal {
            StatsAggregator.shared.cacheMiss()
        }

        let slot = SlotState(
            requestId: req.requestId,
            continuation: req.continuation,
            promptTokenCount: inputTokens.count,
            startTime: prefillStart,
            prefillTime: prefillTime,
            inputTokens: inputTokens,
            cachedTokens: cachedTokens,
            prefillCaches: cache,
            lastTokenId: firstToken,
            lastTokenArray: tokenArray,  // already eval'd — used as model input for first decode step
            sampler: logitSampler,
            processor: logitProcessor,
            detokenizer: NaiveStreamingDetokenizer(tokenizer: tokenizer),
            maxTokens: req.parameters.maxTokens,
            toolRuntime: req.toolCallRuntimeConfig.map { config in
                ToolCallStreamingRuntime(
                    toolCallStartTag: config.startTag,
                    toolCallEndTag: config.endTag,
                    toolCallParser: config.parser,
                    tools: config.tools,
                    applyFixToolArgs: { rtc in
                        Self.applyFixToolArgs(rtc, tools: config.tools)
                    },
                    remapSingleKey: { key, toolName in
                        Self.remapSingleKey(key, toolName: toolName, tools: config.tools)
                    }
                )
            },
            constraintRuntime: req.constraintRuntimeConfig,
            activeStops: req.stopSequences,
            thinkStartTag: req.thinkStartTag,
            thinkEndTag: req.thinkEndTag,
            computeLogprobs: req.parameters.computeLogprobs,
            topLogprobsCount: req.parameters.topLogprobsCount,
            temperatureForLogprobs: req.parameters.temperature
        )

        // Compute logprobs for the first token (from prefill)
        if slot.computeLogprobs {
            slot.pendingLogprobData = computeTokenLogprobs(
                logits: logits, tokenId: firstToken,
                temperature: slot.temperatureForLogprobs,
                topK: slot.topLogprobsCount
            )
        }

        // Dispatch the first token (from prefill) to the detokenizer.
        // The generationLoop dispatch starts from the SECOND token, so without this
        // the first generated token would be lost (e.g., `{` in JSON output).
        // Guard: don't dispatch EOS or unknown tokens as text.
        if !eosTokenIds.contains(firstToken) && firstToken != tokenizer.unknownTokenId {
            slot.detokenizer.append(token: firstToken)
            if let firstChunk = slot.detokenizer.next() {
                slot.tokenCount += 1
                _ = yieldTextChunk(firstChunk, for: slot)
            }
        }

        // Emit cached token count so the controller can include it in usage
        if cachedTokens > 0 {
            req.continuation.yield(StreamChunk(text: "", cachedTokens: cachedTokens))
        }
        if let constraintRuntime = req.constraintRuntimeConfig {
            DebugLogger.log("[BatchScheduler] Constrained slot \(slot.id.uuidString.prefix(8)) mode=\(constraintRuntime.mode)")
        }
        let radixEntries = radixCache.map { String($0.count) } ?? "nil"
        print("[\(batchTs())] [PrefixCache] Prefill complete: mode=batch | outcome=\(cacheOutcome) | input_tokens=\(inputTokens.count) | cached_tokens=\(cachedTokens) | suffix_tokens=\(inputTokens.count - cachedTokens) | radix_entries=\(radixEntries) | lookup=\(String(format: "%.6f", cacheLookupTime ?? 0))s | prefill=\(String(format: "%.3f", prefillTime))s")
        logCacheProfile(
            phase: "restore",
            mode: "batch",
            outcome: cacheOutcome,
            inputTokenCount: inputTokens.count,
            cachedTokenCount: cachedTokens,
            promptTime: prefillTime,
            lookupTime: cacheLookupTime,
            restoreTime: cacheRestoreTime,
            trimTime: cacheTrimTime,
            truncateTime: cacheTruncateTime
        )
        print("[\(batchTs())] [ChunkStats] stage=preliminary | stream=true | cached_tokens=\(cachedTokens) | prompt_tokens=pending | completion_tokens=pending | prompt_time=pending | generate_time=pending")

        slots.append(slot)
        DebugLogger.log("[BatchScheduler] Prefilled slot req=\(slot.requestId) (B=\(slots.count), \(cachedTokens > 0 ? "cache hit \(cachedTokens) tokens" : "full prefill"), \(String(format: "%.0f", prefillTime * 1000))ms)")
    }

    // MARK: - Serial Generation (CacheList models)

    /// Run full generation for a CacheList model using the existing generate() API.
    /// These models can't be merged into BatchKVCacheSimple due to asymmetric K/V shapes.
    /// See: https://github.com/scouzi1966/maclocal-api/issues/66
    private func runSerialGeneration(
        req: PendingRequest, cache: [KVCache], modelState: LMOutput.State?,
        firstToken: MLXArray, inputTokens: [Int],
        cachedTokens: Int, prefillStart: Date, prefillTime: Double,
        cacheOutcome: String, cacheLookupTime: Double?,
        cacheRestoreTime: Double?, cacheTrimTime: Double?,
        cacheTruncateTime: Double?,
        logitProcessor: LogitProcessor?, logitSampler: LogitSampler
    ) {
        // Emit cached token count
        if cachedTokens > 0 {
            req.continuation.yield(StreamChunk(text: "", cachedTokens: cachedTokens))
        }

        print("[\(batchTs())] [BatchScheduler] CacheList serial generation: prompt=\(inputTokens.count) tokens")
        // Update prompt/cache stats for /metrics (serial path was missing these)
        let suffixTokens = inputTokens.count - cachedTokens
        StatsAggregator.shared.addPromptTokens(suffixTokens)
        if cachedTokens > 0 {
            StatsAggregator.shared.cacheHit()
        } else {
            StatsAggregator.shared.cacheMiss()
        }

        // Build EOS set
        var eosTokenIds = configuration.eosTokenIds
        if let tokenizerEos = tokenizer.eosTokenId { eosTokenIds.insert(tokenizerEos) }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) { eosTokenIds.insert(id) }
        }

        // Decode loop — mirrors the batch decode step logic but for a single sequence
        var processor = logitProcessor
        var currentToken = firstToken
        var state = modelState
        var tokenCount = 0
        let maxTokens = req.parameters.maxTokens ?? 4096
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        let genStart = Date()

        while tokenCount < maxTokens {
            let tokenId = currentToken.item(Int.self)
            if eosTokenIds.contains(tokenId) { break }

            tokenCount += 1
            StatsAggregator.shared.addGenTokens(1)
            detokenizer.append(token: tokenId)
            if let chunk = detokenizer.next() {
                req.continuation.yield(StreamChunk(text: chunk))
            }

            // Next token — shape [1, 1] (batch=1, seqLen=1)
            let input = LMInput.Text(tokens: currentToken.reshaped([1, 1]))
            let output = model(input, cache: cache, state: state)
            state = output.state

            let logits = output.logits[0, -1, 0...]
            let processed = processor?.process(logits: logits) ?? logits
            currentToken = logitSampler.sample(logits: processed)
            processor?.didSample(token: currentToken)
            asyncEval(currentToken)
            _ = currentToken.item(Int.self)
        }

        let generateTime = Date().timeIntervalSince(genStart)
        print("[\(batchTs())] [BatchScheduler] CacheList serial done: \(tokenCount) tokens, \(String(format: "%.1f", Double(tokenCount) / generateTime)) tok/s")

        // Signal completion
        req.continuation.yield(StreamChunk(
            text: "",
            promptTokens: inputTokens.count,
            completionTokens: tokenCount,
            cachedTokens: cachedTokens,
            promptTime: prefillTime,
            generateTime: generateTime,
            stoppedBySequence: false
        ))
        req.continuation.finish()

        // Skip prefix cache for CacheList models — their flattened state layout
        // (4 arrays per layer) is incompatible with KVCacheSimple restore (expects 2).
        // TODO: Support CacheList prefix caching (issue #66)
    }

    // MARK: - Batched Prefill

    /// Prefill multiple requests in a single batched forward pass (B=N).
    ///
    /// All requests must have fresh (empty) caches — no prefix cache hits.
    /// Uses left-padding to handle variable-length prompts in a dense array.
    /// `BatchKVCacheSimple` handles attention mask creation; `MambaCache` handles SSM masks.
    ///
    /// For cold-start concurrent arrivals (the primary use case), this replaces
    /// N sequential B=1 prefills with a single B=N forward pass.
    private func prefillBatch(_ requests: [PendingRequest]) {
        let B = requests.count
        guard B > 0 else { return }

        // Verify cache types support batching
        let templateCache = model.newCache(parameters: requests[0].parameters)
        // Models with RotatingKVCache layers (Gemma 4) use individual prefill + merge.
        // MLX's lazy graph overflows SmallVector during batched prefill eval with
        // BatchRotatingKVCache. The decode loop handles them correctly in batch.
        // Individual prefill adds <1% overhead (63ms for 15 requests vs 17s decode).
        let hasRotatingLayers = templateCache.contains { $0 is RotatingKVCache }
        let canBatch = !hasRotatingLayers && templateCache.allSatisfy { c in
            c is KVCacheSimple || c is ArraysCache || c is CacheList
            || (c as? RotatingKVCache)?.isBatchable == true
        }
        if !canBatch {
            // Individual B=1 prefills, but BULK merge at the end to avoid
            // N per-slot concats. Collect per-request caches, then extend
            // the decode batch once per layer.
            var deferredCaches: [([KVCache], LMOutput.State?)] = []
            for req in requests {
                withUnsafeMutablePointer(to: &deferredCaches) { ptr in
                    prefillOne(req, deferMerge: true, deferredCaches: ptr)
                }
            }
            if !deferredCaches.isEmpty && cacheMode == .batched && !batchCaches.isEmpty {
                // Build ONE batched cache from all N deferred caches and extend decode batch
                let debugTiming = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
                let bulkStart = debugTiming ? Date() : Date.distantPast
                let bBefore = slots.count - deferredCaches.count  // slots were already appended
                let N = deferredCaches.count

                for layer in 0..<batchCaches.count {
                    let individualCaches = deferredCaches.map { $0.0[layer] }

                    if let batchBKV = batchCaches[layer] as? BatchKVCacheSimple {
                        let merged = BatchKVCacheSimple.merge(individualCaches)
                        batchBKV.extend(with: merged)
                    } else if let batchBRC = batchCaches[layer] as? BatchRotatingKVCache {
                        let merged = BatchRotatingKVCache.merge(individualCaches)
                        batchBRC.extend(with: merged)
                    } else if let batchAC = batchCaches[layer] as? ArraysCache {
                        // Merge all individual ArraysCaches into one, then extend
                        for ic in individualCaches {
                            if let iac = ic as? ArraysCache {
                                batchAC.addSlotsBulk(from: iac)
                            }
                        }
                    }
                }
                // Merge cross-attention state from first request (if any)
                if let firstState = deferredCaches.first?.1,
                   let newCAS = firstState.crossAttentionStates {
                    if let existingCAS = batchState?.crossAttentionStates {
                        batchState = .init(crossAttentionStates: concatenated([existingCAS, newCAS], axis: 0))
                    } else {
                        batchState = firstState
                    }
                }
                if debugTiming {
                    let bulkWall = Date().timeIntervalSince(bulkStart)
                    print("[\(batchTs())] [BatchScheduler] BULK COHORT MERGE (prefillOne): adding=\(N) slots, B_before=\(bBefore), layers=\(batchCaches.count), wall=\(String(format: "%.3f", bulkWall))s")
                }
            } else {
                // First slots or mode transition — individual merges already happened inside prefillOne
            }
            return
        }

        let prefillStart = Date()

        // Phase 1: Prepare per-request data
        var allInputTokens: [[Int]] = []
        var logitProcessors: [LogitProcessor?] = []
        var logitSamplers: [LogitSampler] = []

        for req in requests {
            let inputTokens = req.input.text.tokens.reshaped(-1).asArray(Int.self)
            allInputTokens.append(inputTokens)

            var processor = req.parameters.processor()
            let sampler = req.parameters.sampler()
            processor?.prompt(req.input.text.tokens)
            logitProcessors.append(processor)
            logitSamplers.append(sampler)
        }

        // Phase 2: Left-pad and stack
        let lengths = allInputTokens.map { $0.count }
        let maxLen = lengths.max()!
        var leftPads: [Int] = []
        var paddedTokenRows: [[Int32]] = []

        for tokens in allInputTokens {
            let pad = maxLen - tokens.count
            leftPads.append(pad)
            paddedTokenRows.append(Array(repeating: Int32(0), count: pad) + tokens.map { Int32($0) })
        }

        let batchTokens = stacked(paddedTokenRows.map { MLXArray($0) })  // [B, maxLen]

        // Phase 3: Create batched caches
        // Attention layers -> BatchKVCacheSimple with leftPadding (handles causal+padding masks)
        // SSM layers -> MambaCache with leftPadding (handles SSM masks)
        // CacheList layers -> BatchCacheList (batches each sub-cache independently)
        let prefillCaches: [KVCache] = templateCache.map { layerCache in
            if let cl = layerCache as? CacheList {
                return BatchCacheList.forPrefill(template: cl, batchSize: B, leftPadding: leftPads) as KVCache
            } else if layerCache is ArraysCache {
                return MambaCache(leftPadding: leftPads) as KVCache
            } else if let rc = layerCache as? RotatingKVCache, rc.isBatchable {
                return BatchRotatingKVCache(batchSize: B, leftPadding: leftPads,
                                            maxCacheSize: rc.cacheSize, keep: rc.keepCount) as KVCache
            } else {
                return BatchKVCacheSimple(batchSize: B, leftPadding: leftPads) as KVCache
            }
        }

        // Phase 4: Single model forward pass
        if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" {
            print("[prefillBatch] B=\(B) maxLen=\(maxLen) tokens=\(batchTokens.shape) caches=\(prefillCaches.count) types=\(prefillCaches.prefix(3).map { type(of: $0) })")
            fflush(stdout)
        }
        let input = LMInput.Text(tokens: batchTokens)
        let result = model(input, cache: prefillCaches, state: nil)

        // Phase 5: Per-request logit processing and sampling
        // With left-padding, position -1 is the last real token for all sequences
        let logits = result.logits  // [B, maxLen, vocabSize]
        var tokenArrays: [MLXArray] = []

        for i in 0..<B {
            let seqLogits = logits[i, -1, 0...]
            let processed = logitProcessors[i]?.process(logits: seqLogits) ?? seqLogits
            let tokenArray = logitSamplers[i].sample(logits: processed)
            logitProcessors[i]?.didSample(token: tokenArray)
            tokenArrays.append(tokenArray)
        }

        // Materialize caches and tokens (MLX eval for GPU synchronization)
        var evalArrays: [MLXArray] = tokenArrays
        evalArrays.append(contentsOf: prefillCaches.flatMap { $0.innerState() })
        if let s = result.state?.crossAttentionStates { evalArrays.append(s) }
        MLX.eval(evalArrays)

        let prefillTime = Date().timeIntervalSince(prefillStart)

        // Phase 6: Extract individual caches per request (for prefix cache save in finishSlot)
        var perRequestCaches: [[KVCache]] = (0..<B).map { _ in [KVCache]() }
        for cache in prefillCaches {
            if let bclCache = cache as? BatchCacheList {
                for i in 0..<B {
                    perRequestCaches[i].append(bclCache.extract(i) as KVCache)
                }
            } else if let brcCache = cache as? BatchRotatingKVCache {
                for i in 0..<B {
                    let (k, v, tokenCount) = brcCache.extract(i)
                    let individual = RotatingKVCache(maxSize: brcCache.maxCacheSize, keep: brcCache.keep)
                    individual.state = [k, v]
                    // Restore metaState with correct offset/idx from extraction
                    individual.metaState = [
                        "\(brcCache.keep)", "\(brcCache.maxCacheSize)",
                        "\(BatchRotatingKVCache.preAllocStep)", "\(tokenCount)", "\(tokenCount)"
                    ]
                    perRequestCaches[i].append(individual as KVCache)
                }
            } else if let bkvCache = cache as? BatchKVCacheSimple {
                for i in 0..<B {
                    let (k, v, _) = bkvCache.extract(i)
                    let individual = KVCacheSimple()
                    individual.state = [k, v]
                    perRequestCaches[i].append(individual as KVCache)
                }
            } else if let acCache = cache as? ArraysCache {
                let cacheState = acCache.state
                for i in 0..<B {
                    let individual = MambaCache()
                    if !cacheState.isEmpty {
                        individual.state = cacheState.map { $0[i ..< i + 1] }
                    }
                    perRequestCaches[i].append(individual as KVCache)
                }
            }
        }

        // Merge into decode batch
        if slots.isEmpty {
            // No existing decode slots -- use prefill caches directly for decode
            batchCaches = prefillCaches
            cacheMode = .batched
            batchState = result.state
        } else if cacheMode == .batched && batchCaches.count == prefillCaches.count {
            // Bulk cohort merge: extend each layer's batch cache with the
            // corresponding batched prefill cache in ONE concat per layer.
            // This replaces the per-slot loop that did N separate concats per
            // layer per slot — the main source of the merge cliff stalls.
            let debugTiming = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
            let bulkStart = debugTiming ? Date() : Date.distantPast
            let bBefore = slots.count

            for layer in 0..<batchCaches.count {
                if let batchBKV = batchCaches[layer] as? BatchKVCacheSimple,
                   let prefillBKV = prefillCaches[layer] as? BatchKVCacheSimple {
                    batchBKV.extend(with: prefillBKV)
                } else if let batchAC = batchCaches[layer] as? ArraysCache,
                          let prefillAC = prefillCaches[layer] as? ArraysCache {
                    batchAC.addSlotsBulk(from: prefillAC)
                } else if let batchBRC = batchCaches[layer] as? BatchRotatingKVCache,
                          let prefillBRC = prefillCaches[layer] as? BatchRotatingKVCache {
                    batchBRC.extend(with: prefillBRC)
                } else if let batchBCL = batchCaches[layer] as? BatchCacheList,
                          let prefillBCL = prefillCaches[layer] as? BatchCacheList {
                    // Extend each sub-cache within the CacheList
                    for subIdx in 0..<Swift.min(batchBCL.count, prefillBCL.count) {
                        if let batchACSub = batchBCL[subIdx] as? ArraysCache,
                           let prefillACSub = prefillBCL[subIdx] as? ArraysCache {
                            batchACSub.addSlotsBulk(from: prefillACSub)
                        } else if let batchBKVSub = batchBCL[subIdx] as? BatchKVCacheSimple,
                                  let prefillBKVSub = prefillBCL[subIdx] as? BatchKVCacheSimple {
                            batchBKVSub.extend(with: prefillBKVSub)
                        }
                    }
                }
            }
            // Merge cross-attention state
            if let newCAS = result.state?.crossAttentionStates {
                if let existingCAS = batchState?.crossAttentionStates {
                    batchState = .init(crossAttentionStates: concatenated([existingCAS, newCAS], axis: 0))
                } else {
                    batchState = result.state
                }
            }
            if debugTiming {
                let bulkWall = Date().timeIntervalSince(bulkStart)
                print("[\(batchTs())] [BatchScheduler] BULK COHORT MERGE: adding=\(B) slots, B_before=\(bBefore), layers=\(batchCaches.count), wall=\(String(format: "%.3f", bulkWall))s")
            }
        } else {
            // Fallback: per-slot merge (for cache mode transitions or mismatched types)
            for i in 0..<B {
                mergeCacheIntoBatch(individualCache: perRequestCaches[i], modelState: i == 0 ? result.state : nil)
            }
        }

        // Create SlotState for each request
        for i in 0..<B {
            let req = requests[i]
            let firstToken = tokenArrays[i].item(Int.self)

            let slot = SlotState(
                requestId: req.requestId,
                continuation: req.continuation,
                promptTokenCount: allInputTokens[i].count,
                startTime: prefillStart,
                prefillTime: prefillTime,
                inputTokens: allInputTokens[i],
                cachedTokens: 0,
                prefillCaches: perRequestCaches[i],
                lastTokenId: firstToken,
                lastTokenArray: tokenArrays[i],
                sampler: logitSamplers[i],
                processor: logitProcessors[i],
                detokenizer: NaiveStreamingDetokenizer(tokenizer: tokenizer),
                maxTokens: req.parameters.maxTokens,
                toolRuntime: req.toolCallRuntimeConfig.map { config in
                    ToolCallStreamingRuntime(
                        toolCallStartTag: config.startTag,
                        toolCallEndTag: config.endTag,
                        toolCallParser: config.parser,
                        tools: config.tools,
                        applyFixToolArgs: { rtc in
                            Self.applyFixToolArgs(rtc, tools: config.tools)
                        },
                        remapSingleKey: { key, toolName in
                            Self.remapSingleKey(key, toolName: toolName, tools: config.tools)
                        }
                    )
                },
                constraintRuntime: req.constraintRuntimeConfig,
                activeStops: req.stopSequences,
                thinkStartTag: req.thinkStartTag,
                thinkEndTag: req.thinkEndTag,
                computeLogprobs: req.parameters.computeLogprobs,
                topLogprobsCount: req.parameters.topLogprobsCount,
                temperatureForLogprobs: req.parameters.temperature
            )

            // Compute logprobs for the first token (from prefill)
            if slot.computeLogprobs {
                let seqLogits = logits[i, -1, 0...]
                let processed = logitProcessors[i]?.process(logits: seqLogits) ?? seqLogits
                slot.pendingLogprobData = computeTokenLogprobs(
                    logits: processed, tokenId: firstToken,
                    temperature: slot.temperatureForLogprobs,
                    topK: slot.topLogprobsCount
                )
            }

            // Dispatch prefill first token to detokenizer (same fix as prefillOne)
            if !eosTokenIds.contains(firstToken) && firstToken != tokenizer.unknownTokenId {
                slot.detokenizer.append(token: firstToken)
                if let firstChunk = slot.detokenizer.next() {
                    slot.tokenCount += 1
                    _ = yieldTextChunk(firstChunk, for: slot)
                }
            }

            if let constraintRuntime = req.constraintRuntimeConfig {
                DebugLogger.log("[BatchScheduler] Constrained slot \(slot.id.uuidString.prefix(8)) mode=\(constraintRuntime.mode)")
            }

            slots.append(slot)
        }

        let totalInputTokens = lengths.reduce(0, +)
        print("[\(batchTs())] [BatchScheduler] Batched prefill: B=\(B), maxLen=\(maxLen), totalTokens=\(totalInputTokens), leftPads=\(leftPads), time=\(String(format: "%.3f", prefillTime))s (\(String(format: "%.0f", Double(totalInputTokens) / prefillTime)) tok/s)")
        DebugLogger.log("[BatchScheduler] Batched prefill complete: B=\(B), \(String(format: "%.0f", prefillTime * 1000))ms")
        StatsAggregator.shared.addPromptTokens(totalInputTokens)
        // Batched prefill requires fresh (empty) caches — every request in
        // it is a cache miss by definition. Requests that hit the radix
        // cache go through prefillOne and are counted there.
        for _ in 0..<B { StatsAggregator.shared.cacheMiss() }
    }

    /// Merge a newly-prefilled individual cache into the batch.
    ///
    /// Handles CacheMode transitions:
    /// - empty → unbatched: Keep KVCacheSimple at B=1 (zero batch overhead)
    /// - unbatched → batched: Promote to BatchKVCacheSimple when B≥2
    /// - batched → batched: Extend existing BatchKVCacheSimple
    private func mergeCacheIntoBatch(individualCache: [KVCache], modelState: LMOutput.State?) {
        switch cacheMode {
        case .empty:
            // B=0 → B=1: Keep individual caches if all are simple/batchable types (zero overhead)
            let allSimple = individualCache.allSatisfy {
                $0 is KVCacheSimple || $0 is ArraysCache || $0 is CacheList
                || ($0 as? RotatingKVCache)?.isBatchable == true
            }
            if allSimple {
                batchCaches = individualCache.map { layerCache in
                    if layerCache is ArraysCache && !(layerCache is CacheList) {
                        let copy = MambaCache()
                        copy.state = layerCache.state
                        return copy as KVCache
                    } else {
                        return layerCache  // Keep as KVCacheSimple, RotatingKVCache, or CacheList
                    }
                }
                cacheMode = .unbatched
            } else {
                // QuantizedKVCache etc. — promote to batch immediately
                batchCaches = individualCache.map { layerCache in
                    if layerCache is ArraysCache && !(layerCache is CacheList) {
                        let copy = MambaCache()
                        copy.state = layerCache.state
                        return copy as KVCache
                    } else if layerCache is CacheList {
                        return BatchCacheList.merge([layerCache], leftPadding: [0]) as KVCache
                    } else if let rc = layerCache as? RotatingKVCache, rc.isBatchable {
                        return BatchRotatingKVCache.merge([layerCache]) as KVCache
                    } else {
                        return BatchKVCacheSimple.merge([layerCache]) as KVCache
                    }
                }
                cacheMode = .batched
            }

        case .unbatched:
            // B=1 → B≥2: Promote existing + new to batch caches
            Stream.gpu.synchronize()  // Ensure B=1 cache arrays are fully evaluated
            batchCaches = zip(batchCaches, individualCache).map { existing, new in
                if let existingCL = existing as? CacheList, new is CacheList {
                    return BatchCacheList.merge([existingCL, new], leftPadding: [0, 0]) as KVCache
                } else if let existingAC = existing as? ArraysCache,
                   let newAC = new as? ArraysCache {
                    existingAC.extend(other: newAC)
                    return existingAC as KVCache
                } else if existing is RotatingKVCache || new is RotatingKVCache {
                    return BatchRotatingKVCache.merge([existing, new]) as KVCache
                } else {
                    return BatchKVCacheSimple.merge([existing, new]) as KVCache
                }
            }
            cacheMode = .batched

        case .batched:
            // B≥2 → B+1: Extend existing batch caches
            for layer in 0..<batchCaches.count where layer < individualCache.count {
                if let batchBCL = batchCaches[layer] as? BatchCacheList,
                   let newCL = individualCache[layer] as? CacheList {
                    for subIdx in 0..<batchBCL.count where subIdx < newCL.count {
                        if let batchAC = batchBCL[subIdx] as? ArraysCache,
                           let newAC = newCL[subIdx] as? ArraysCache {
                            batchAC.extend(other: newAC)
                        } else if let batchBKV = batchBCL[subIdx] as? BatchKVCacheSimple {
                            let single = BatchKVCacheSimple.merge([newCL[subIdx]])
                            batchBKV.extend(with: single)
                        }
                    }
                } else if let batchAC = batchCaches[layer] as? ArraysCache,
                   let newAC = individualCache[layer] as? ArraysCache {
                    batchAC.extend(other: newAC)
                } else if let batchBRC = batchCaches[layer] as? BatchRotatingKVCache {
                    let single = BatchRotatingKVCache.merge([individualCache[layer]])
                    batchBRC.extend(with: single)
                } else {
                    let singleBatch = BatchKVCacheSimple.merge([individualCache[layer]])
                    (batchCaches[layer] as! BatchKVCacheSimple).extend(with: singleBatch)
                }
            }
        }

        // Merge model cross-attention state if present
        if let newCAS = modelState?.crossAttentionStates {
            if let existingCAS = batchState?.crossAttentionStates {
                batchState = .init(crossAttentionStates: concatenated([existingCAS, newCAS], axis: 0))
            } else {
                batchState = modelState
            }
        }
    }

    // MARK: - Slot Completion

    /// Finish a completed slot: save prefix cache, yield timing info, remove from batch.
    private func finishSlot(at index: Int) {
        let debugTiming = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
        let finishStart = debugTiming ? Date() : Date.distantPast
        let slot = slots[index]
        let elapsed = Date().timeIntervalSince(slot.startTime)
        let generateTime = elapsed - slot.prefillTime

        if let trailingEvents = slot.toolRuntime?.finishIncompleteToolCall(), !trailingEvents.isEmpty {
            yieldToolRuntimeEvents(trailingEvents, to: slot)
        }
        if !slot.activeStops.isEmpty && !slot.stopBuffer.isEmpty && !slot.stoppedBySequence {
            slot.continuation.yield(StreamChunk(text: slot.stopBuffer))
            slot.stopBuffer = ""
        }

        // Save prompt KV state to prefix cache before removal.
        // Use the original per-layer prefill caches (retained on the slot).
        let cacheStart = debugTiming ? Date() : Date.distantPast
        var saveTrimTime: Double? = nil
        var saveTruncateTime: Double? = nil
        var saveInsertTime: Double? = nil
        if let radix = radixCache, !slot.inputTokens.isEmpty {
            let tSave0 = Date.timeIntervalSinceReferenceDate
            // Use pre-snapshotted state from prefill time — immune to decode mutations.
            let layerStates = slot.prefillStates
            let layerMetaStates = slot.prefillMetaStates
            let tSaveTrim = tSave0
            let tSaveTruncate = tSave0
            radix.insert(
                tokens: slot.inputTokens,
                layerStates: layerStates,
                layerMetaStates: layerMetaStates
            )
            let tSaveInsert = Date.timeIntervalSinceReferenceDate
            saveTrimTime = tSaveTrim - tSave0
            saveTruncateTime = tSaveTruncate - tSaveTrim
            saveInsertTime = tSaveInsert - tSaveTruncate
            let activeLayers = layerStates.reduce(into: 0) { count, state in
                if !state.isEmpty { count += 1 }
            }
            let maxOffset = layerStates.map { $0.isEmpty ? 0 : $0[0].dim(2) }.max() ?? 0
            print(
                "[\(batchTs())] [PrefixCache] Save complete: mode=batch | stored_tokens=\(slot.inputTokens.count) | " +
                    "radix_entries=\(radix.count) | trim=\(String(format: "%.6f", saveTrimTime ?? 0))s | " +
                    "truncate=\(String(format: "%.6f", saveTruncateTime ?? 0))s | insert=\(String(format: "%.6f", saveInsertTime ?? 0))s | " +
                    "layers=\(layerStates.count) active_layers=\(activeLayers) max_offset=\(maxOffset)"
            )
            DebugLogger.log("[BatchScheduler] Prefix cache save: \(slot.inputTokens.count) tokens")
        }
        logCacheProfile(
            phase: "save",
            mode: "batch",
            outcome: slot.inputTokens.isEmpty ? "skip" : "save",
            inputTokenCount: slot.inputTokens.count,
            cachedTokenCount: slot.cachedTokens,
            promptTime: slot.prefillTime,
            trimTime: saveTrimTime,
            truncateTime: saveTruncateTime,
            insertTime: saveInsertTime
        )

        slot.continuation.yield(StreamChunk(
            text: "",
            promptTokens: slot.promptTokenCount,
            completionTokens: slot.tokenCount,
            cachedTokens: slot.cachedTokens,
            promptTime: slot.prefillTime,
            generateTime: generateTime,
            stoppedBySequence: slot.stoppedBySequence
        ))
        print("[\(batchTs())] [ChunkStats] stage=final | stream=true | cached_tokens=\(slot.cachedTokens) | prompt_tokens=\(slot.promptTokenCount) | completion_tokens=\(slot.tokenCount) | prompt_time=\(String(format: "%.3f", slot.prefillTime))s | generate_time=\(String(format: "%.3f", generateTime))s")
        slot.continuation.finish()
        slot.constraintRuntime?.matcherHandle?.release()
        _inFlightCount.withLock { $0 = max($0 - 1, 0) }
        StatsAggregator.shared.requestCompleted()

        DebugLogger.log("[BatchScheduler] Finished slot req=\(slot.requestId) (\(slot.tokenCount) tok, \(String(format: "%.2f", elapsed))s, in-flight: \(_inFlightCount.withLock { $0 })/\(maxConcurrent))")

        // Remove from batch cache and state
        let keepIndices = (0..<slots.count).filter { $0 != index }
        let filterStart = debugTiming ? Date() : Date.distantPast
        let bBefore = slots.count
        // Track the max seqLen across BatchKVCacheSimple layers so the probe can
        // report the B × seqLen product that drives the cost.
        var probeMaxSeqLen = 0
        if debugTiming {
            for c in batchCaches {
                if let bkv = c as? BatchKVCacheSimple {
                    probeMaxSeqLen = max(probeMaxSeqLen, bkv.offset)
                }
            }
        }
        if keepIndices.isEmpty {
            batchCaches = []
            batchState = nil
            cacheMode = .empty
        } else {
            switch cacheMode {
            case .unbatched:
                break  // B=1 finishing but keepIndices not empty — unreachable
            case .batched:
                let idxArray = MLXArray(keepIndices.map { Int32($0) })
                for layer in 0..<batchCaches.count {
                    if let bclCache = batchCaches[layer] as? BatchCacheList {
                        // Filter each sub-cache inside the CacheList
                        for subIdx in 0..<bclCache.count {
                            if let bkvSub = bclCache[subIdx] as? BatchKVCacheSimple {
                                bkvSub.filter(keepIndices)
                            } else if let acSub = bclCache[subIdx] as? ArraysCache {
                                acSub.filter(batchIndices: idxArray)
                            }
                        }
                    } else if let brcCache = batchCaches[layer] as? BatchRotatingKVCache {
                        brcCache.filter(keepIndices)
                    } else if let bkvCache = batchCaches[layer] as? BatchKVCacheSimple {
                        bkvCache.filter(keepIndices)
                    } else if let acCache = batchCaches[layer] as? ArraysCache {
                        acCache.filter(batchIndices: idxArray)
                    }
                }
                if let cas = batchState?.crossAttentionStates {
                    batchState = .init(crossAttentionStates: cas[idxArray])
                }
                // Stay .batched — no demotion (overhead negligible at B=1 after filter)
            case .empty:
                break
            }
        }

        Stream.gpu.synchronize()
        if debugTiming {
            let filterWall = Date().timeIntervalSince(filterStart)
            // Only log "expensive" filters (>10ms) to avoid flooding the log
            // with the thousands of fast filters at small B / low seqLen.
            if filterWall > 0.010 {
                print("[\(batchTs())] [BatchScheduler] FILTER: B_before=\(bBefore) → \(keepIndices.count), seqLen=\(probeMaxSeqLen), wall=\(String(format: "%.3f", filterWall))s")
            }
        }
        if debugTiming {
            let totalTime = Date().timeIntervalSince(finishStart) * 1000
            let cacheTime = Date().timeIntervalSince(cacheStart) * 1000
            DebugLogger.log("[BatchScheduler] finishSlot timing: total=\(String(format: "%.1f", totalTime))ms cache_save=\(String(format: "%.1f", cacheTime))ms")
        }
        slots.remove(at: index)
    }

    /// Compute per-token logprob data from processed logits and sampled token.
    /// Matches the algorithm in Evaluate.swift convertToToken() for consistency.
    private func computeTokenLogprobs(
        logits: MLXArray, tokenId: Int, temperature: Float, topK: Int
    ) -> TokenLogprobData {
        var lpLogits = logits
        if lpLogits.dtype == .bfloat16 { lpLogits = lpLogits.asType(.float32) }
        if temperature > 0 { lpLogits = lpLogits / MLXArray(temperature) }
        let probs = softmax(lpLogits, axis: -1)
        let logProbs = log(probs)
        let flat = logProbs.reshaped(-1)

        let tokenLogprob = flat[tokenId].item(Float.self)

        var topIds = [Int]()
        var topLps = [Float]()
        if topK > 0 {
            let vocabSize = flat.dim(0)
            let n = Swift.min(topK, vocabSize)
            let sorted = argSort(flat, axis: -1)
            for i in 0..<n {
                let idx = sorted[vocabSize - 1 - i].item(Int.self)
                topIds.append(idx)
                topLps.append(flat[idx].item(Float.self))
            }
        }

        return TokenLogprobData(
            tokenId: tokenId,
            logprob: tokenLogprob,
            topTokenIds: topIds,
            topLogprobs: topLps
        )
    }

    /// Resolve a single TokenLogprobData to ResolvedLogprob.
    private func resolveLogprob(_ data: TokenLogprobData) -> ResolvedLogprob {
        let token = tokenizer.decode(tokens: [data.tokenId])
        let topTokens = zip(data.topTokenIds, data.topLogprobs).map { (id, lp) in
            (token: tokenizer.decode(tokens: [id]), tokenId: id, logprob: lp)
        }
        return ResolvedLogprob(
            token: token,
            tokenId: data.tokenId,
            logprob: data.logprob,
            topTokens: topTokens
        )
    }

    private func yieldTextChunk(_ chunk: String, for slot: SlotState, logprobs: [ResolvedLogprob]? = nil) -> Bool {
        // Yield logprobs before the text chunk they correspond to (matches single-sequence pattern)
        if let logprobs, !logprobs.isEmpty {
            slot.continuation.yield(StreamChunk(text: "", logprobs: logprobs))
        }
        if let toolRuntime = slot.toolRuntime {
            let output = toolRuntime.process(piece: chunk)
            if output.handled {
                yieldToolRuntimeEvents(output.events, to: slot)
                return false
            }
        }
        let stopResult = Self.stopChunksToEmit(
            from: chunk,
            stopBuffer: &slot.stopBuffer,
            activeStops: slot.activeStops,
            maxStopLength: slot.maxStopLength,
            insideThink: &slot.insideThink,
            thinkStartTag: slot.thinkStartTag,
            thinkEndTag: slot.thinkEndTag
        )
        for emit in stopResult.chunks {
            slot.continuation.yield(emit)
        }
        if stopResult.stopped {
            slot.stoppedBySequence = true
        }
        return stopResult.stopped
    }

    private func yieldToolRuntimeEvents(_ events: [ToolCallStreamingEvent], to slot: SlotState) {
        for chunk in Self.streamChunksToEmit(from: events) {
            slot.continuation.yield(chunk)
        }
    }

    static func streamChunksToEmit(from events: [ToolCallStreamingEvent]) -> [StreamChunk] {
        var chunks = [StreamChunk]()
        let deltas = deltaToolCallsToEmit(from: events)
        if !deltas.isEmpty {
            chunks.append(StreamChunk(text: "", toolCallDeltas: deltas))
        }
        for toolCall in completedToolCallsToEmit(from: events) {
            chunks.append(StreamChunk(text: "", toolCalls: [toolCall]))
        }
        return chunks
    }

    static func deltaToolCallsToEmit(from events: [ToolCallStreamingEvent]) -> [StreamDeltaToolCall] {
        var emitted = [StreamDeltaToolCall]()
        for event in events {
            if case .delta(let delta) = event {
                emitted.append(delta)
            }
        }
        return emitted
    }

    static func completedToolCallsToEmit(from events: [ToolCallStreamingEvent]) -> [ResponseToolCall] {
        var emitted = [ResponseToolCall]()
        for event in events {
            switch event {
            case .started, .delta:
                continue
            case .appendCollected(let toolCall):
                if !toolCall.function.arguments.isEmpty {
                    emitted.append(toolCall)
                }
            case .replaceCollected(_, let toolCall):
                emitted.append(toolCall)
            }
        }
        return emitted
    }

    static func stopChunksToEmit(
        from text: String,
        stopBuffer: inout String,
        activeStops: [String],
        maxStopLength: Int,
        insideThink: inout Bool,
        thinkStartTag: String?,
        thinkEndTag: String?
    ) -> (chunks: [StreamChunk], stopped: Bool) {
        guard !activeStops.isEmpty else {
            return ([StreamChunk(text: text)], false)
        }

        var chunks = [StreamChunk]()
        let wasInsideThink = insideThink
        if let thinkStartTag, text.contains(thinkStartTag) {
            insideThink = true
        }
        if let thinkEndTag, text.contains(thinkEndTag) {
            insideThink = false
        }

        if !insideThink {
            if wasInsideThink, let thinkEndTag, let range = text.range(of: thinkEndTag) {
                let afterThink = String(text[range.upperBound...])
                if !afterThink.isEmpty {
                    stopBuffer += afterThink
                }
            } else {
                stopBuffer += text
            }

            if let match = activeStops.first(where: { stopBuffer.contains($0) }),
               let range = stopBuffer.range(of: match) {
                let before = String(stopBuffer[..<range.lowerBound])
                chunks.append(StreamChunk(text: before, stoppedBySequence: true))
                return (chunks, true)
            }

            if stopBuffer.count > maxStopLength {
                let flushEnd = stopBuffer.index(stopBuffer.endIndex, offsetBy: -maxStopLength)
                let flushText = String(stopBuffer[..<flushEnd])
                stopBuffer = String(stopBuffer[flushEnd...])
                chunks.append(StreamChunk(text: flushText))
            }
        } else {
            chunks.append(StreamChunk(text: text))
        }

        return (chunks, false)
    }

    private static func remapSingleKey(_ key: String, toolName: String, tools: [RequestTool]?) -> String {
        guard let tools, !tools.isEmpty else { return key }
        let dummy: [String: any Sendable] = [key: ""]
        let remapped = MLXModelService.remapArgumentKeys(dummy, toolName: toolName, tools: tools)
        return remapped.keys.first ?? key
    }

    private static func applyFixToolArgs(_ rtc: ResponseToolCall, tools: [RequestTool]?) -> ResponseToolCall {
        guard let tools, !tools.isEmpty else { return rtc }
        guard let data = rtc.function.arguments.data(using: .utf8),
              let argsDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return rtc }
        var sendableArgs = [String: any Sendable]()
        for (key, value) in argsDict { sendableArgs[key] = value }
        let remapped = MLXModelService.remapArgumentKeys(sendableArgs, toolName: rtc.function.name, tools: tools)
        let remappedAny = remapped.mapValues { $0 as Any }
        guard let newData = try? JSONSerialization.data(withJSONObject: remappedAny, options: [.sortedKeys]),
              let newString = String(data: newData, encoding: .utf8) else { return rtc }
        return ResponseToolCall(
            index: rtc.index,
            id: rtc.id,
            type: rtc.type,
            function: ResponseToolCallFunction(name: rtc.function.name, arguments: newString)
        )
    }
}
