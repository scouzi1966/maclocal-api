import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import os

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

    /// Default maximum concurrent generations.
    static let defaultMaxConcurrent = 8

    let maxConcurrent: Int
    private let model: any LanguageModel
    private let tokenizer: Tokenizer
    private let processor: any UserInputProcessor
    private let configuration: ModelConfiguration

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
        let continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
        let promptTokenCount: Int
        /// Set to the moment prefill begins (prefillStart), so elapsed includes prefill + decode.
        let startTime: Date
        let prefillTime: TimeInterval
        var tokenCount = 0
        var firstTokenTime: TimeInterval = 0
        let inputTokens: [Int]
        let cachedTokens: Int
        /// The individual per-layer caches from prefill (retained for prefix cache save).
        let prefillCaches: [KVCache]

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

        /// Lazy token array from the previous decode step (for deferred dispatch).
        /// Set after asyncEval; materialized via .item() after the NEXT model call.
        var pendingTokenArray: MLXArray? = nil

        init(
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
            maxTokens: Int?
        ) {
            self.id = UUID()
            self.continuation = continuation
            self.promptTokenCount = promptTokenCount
            self.prefillTime = prefillTime
            self.startTime = startTime
            self.inputTokens = inputTokens
            self.cachedTokens = cachedTokens
            self.prefillCaches = prefillCaches
            self.lastTokenId = lastTokenId
            self.lastTokenArray = lastTokenArray
            self.sampler = sampler
            self.processor = processor
            self.detokenizer = detokenizer
            self.maxTokens = maxTokens
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

    struct PendingRequest: @unchecked Sendable {
        let input: LMInput
        let parameters: GenerateParameters
        let promptTokens: Int
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

    /// Release a reserved slot (call if request fails before reaching submit).
    nonisolated func releaseReservation() {
        _inFlightCount.withLock { $0 = max($0 - 1, 0) }
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
        enablePrefixCaching: Bool = false
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.configuration = configuration
        self.maxConcurrent = maxConcurrent

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
    }

    /// Number of requests currently generating.
    var activeCount: Int { slots.count }

    /// Prepare UserInput into LMInput (tokenization + chat template).
    func prepareInput(_ userInput: UserInput) async throws -> LMInput {
        try await processor.prepare(input: userInput)
    }

    /// Submit a generation request. Returns a stream of StreamChunks immediately.
    /// Nonisolated — no actor hop needed. Pushes to lock-protected queue.
    nonisolated func submit(
        input: LMInput,
        parameters: GenerateParameters,
        promptTokens: Int
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        if _isShutdown.withLock({ $0 }) {
            return AsyncThrowingStream { $0.finish(throwing: MLXServiceError.serviceShuttingDown) }
        }

        let (stream, continuation) = AsyncThrowingStream<StreamChunk, Error>.makeStream()

        // Note: slot already reserved by tryReserve() in the controller layer.
        _pendingQueue.withLock {
            $0.append(PendingRequest(
                input: input,
                parameters: parameters,
                promptTokens: promptTokens,
                continuation: continuation
            ))
        }

        DebugLogger.log("[BatchScheduler] Request enqueued (\(_inFlightCount.withLock { $0 })/\(maxConcurrent))")
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
        // Previous step's sampled tokens + slot IDs, for deferred dispatch.
        // nil on the first iteration (newly prefilled slots have no pending work).
        var dispatchTokens: [MLXArray]? = nil
        var dispatchSlotIDs: [UUID]? = nil

        // Decode-step timing accumulators (debug only)
        let debugTiming = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
        var stepTimeAccum: Double = 0
        var modelTimeAccum: Double = 0
        var dispatchTimeAccum: Double = 0
        var timingStepCount = 0

        while !_isShutdown.withLock({ $0 }) {
            let stepStart = debugTiming ? Date() : Date.distantPast

            // Drain nonisolated queue — no Task.yield() needed for request pickup
            let newRequests = drainPendingQueue()
            for req in newRequests {
                if slots.count < maxConcurrent {
                    prefillOne(req)
                } else {
                    // Re-enqueue if at capacity (shouldn't happen often)
                    _pendingQueue.withLock { $0.append(req) }
                }
            }

            if slots.isEmpty { break }
            if Task.isCancelled { return }

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
            var tokenArrays = [MLXArray]()
            for i in 0..<activeB {
                let slot = slots[i]
                let slotLogits = activeB == 1 ? logits : logits[i]
                let processed = slot.processor?.process(logits: slotLogits) ?? slotLogits
                tokenArrays.append(slot.sampler.sample(logits: processed))
            }

            // Kick off async evaluation — GPU starts computing current step
            asyncEval(tokenArrays)
            if debugTiming {
                modelTimeAccum += Date().timeIntervalSince(modelStart)
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
            dispatchTokens = tokenArrays
            dispatchSlotIDs = slots.map { $0.id }  // order matches tokenArrays

            // --- While GPU runs: dispatch PREVIOUS step's tokens ---
            // .item() is instant because these were asyncEval'd last iteration.
            let dispatchStart = debugTiming ? Date() : Date.distantPast
            if let prevTokens = toDispatchTokens, let prevIDs = toDispatchSlotIDs {
                var completedIndices: [Int] = []
                for j in 0..<prevIDs.count {
                    guard let i = slots.firstIndex(where: { $0.id == prevIDs[j] }) else { continue }
                    let slot = slots[i]
                    let tokenArray = prevTokens[j]

                    let token = tokenArray.item(Int.self)
                    slot.processor?.didSample(token: tokenArray)

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
                        slot.continuation.yield(StreamChunk(text: chunk))
                    }
                }

                for i in completedIndices.sorted().reversed() {
                    finishSlot(at: i)
                }
            }

            if debugTiming {
                dispatchTimeAccum += Date().timeIntervalSince(dispatchStart)
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
                Memory.clearCache()
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
                    slot.continuation.yield(StreamChunk(text: chunk))
                }
            }
        }

        loopTask = nil
    }

    // MARK: - Prefill

    /// Prefill a single request (B=1), then merge its cache into the batch.
    private func prefillOne(_ req: PendingRequest) {
        var cache = model.newCache(parameters: req.parameters)
        var generateInput = req.input
        var cachedTokens = 0

        // Extract token array for prefix cache lookup
        let inputTokens = req.input.text.tokens.reshaped(-1).asArray(Int.self)
        let isMultimodal = req.input.image != nil || req.input.video != nil

        // Prefix cache: restore KV state if available
        if !isMultimodal, let radix = radixCache {
            let (prefixLen, layerStates) = radix.findPrefix(inputTokens)
            let minSuffix = 16
            let effectivePrefix = min(prefixLen, max(0, inputTokens.count - minSuffix))

            if effectivePrefix > 0, let states = layerStates {
                for i in 0..<cache.count where i < states.count {
                    cache[i].state = states[i]
                }
                for i in 0..<cache.count {
                    let excess = cache[i].offset - effectivePrefix
                    if excess > 0 { cache[i].trim(excess) }
                }
                for i in 0..<cache.count {
                    if cache[i].isTrimmable && cache[i].offset > 0 {
                        cache[i].state = cache[i].state
                    }
                }
                let suffixTokens = Array(inputTokens[effectivePrefix...])
                generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                cachedTokens = effectivePrefix
                DebugLogger.log("[BatchScheduler] Prefix cache hit: \(effectivePrefix)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix")
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

        // Merge individual caches into batch
        mergeCacheIntoBatch(individualCache: cache, modelState: result.state)

        let slot = SlotState(
            continuation: req.continuation,
            promptTokenCount: req.promptTokens,
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
            maxTokens: req.parameters.maxTokens
        )

        // Emit cached token count so the controller can include it in usage
        if cachedTokens > 0 {
            req.continuation.yield(StreamChunk(text: "", cachedTokens: cachedTokens))
        }

        slots.append(slot)
        DebugLogger.log("[BatchScheduler] Prefilled slot \(slot.id.uuidString.prefix(8)) (B=\(slots.count), \(cachedTokens > 0 ? "cache hit \(cachedTokens) tokens" : "full prefill"), \(String(format: "%.0f", prefillTime * 1000))ms)")
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
            // B=0 → B=1: Keep individual caches if all are KVCacheSimple (zero overhead)
            let allSimple = individualCache.allSatisfy { $0 is KVCacheSimple || $0 is ArraysCache }
            if allSimple {
                batchCaches = individualCache.map { layerCache in
                    if layerCache is ArraysCache {
                        let copy = MambaCache()
                        copy.state = layerCache.state
                        return copy as KVCache
                    } else {
                        return layerCache  // Keep as KVCacheSimple — zero batch overhead
                    }
                }
                cacheMode = .unbatched
            } else {
                // QuantizedKVCache etc. — promote to BatchKVCacheSimple immediately
                batchCaches = individualCache.map { layerCache in
                    if layerCache is ArraysCache {
                        let copy = MambaCache()
                        copy.state = layerCache.state
                        return copy as KVCache
                    } else {
                        return BatchKVCacheSimple.merge([layerCache]) as KVCache
                    }
                }
                cacheMode = .batched
            }

        case .unbatched:
            // B=1 → B≥2: Promote existing + new to BatchKVCacheSimple
            Stream.gpu.synchronize()  // Ensure B=1 cache arrays are fully evaluated
            batchCaches = zip(batchCaches, individualCache).map { existing, new in
                if let existingAC = existing as? ArraysCache,
                   let newAC = new as? ArraysCache {
                    existingAC.extend(other: newAC)
                    return existingAC as KVCache
                } else {
                    return BatchKVCacheSimple.merge([existing, new]) as KVCache
                }
            }
            cacheMode = .batched

        case .batched:
            // B≥2 → B+1: Extend existing BatchKVCacheSimple
            for layer in 0..<batchCaches.count where layer < individualCache.count {
                if let batchAC = batchCaches[layer] as? ArraysCache,
                   let newAC = individualCache[layer] as? ArraysCache {
                    batchAC.extend(other: newAC)
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

        // Save prompt KV state to prefix cache before removal.
        // Use the original per-layer prefill caches (retained on the slot).
        let cacheStart = debugTiming ? Date() : Date.distantPast
        if let radix = radixCache, !slot.inputTokens.isEmpty {
            let promptLen = slot.inputTokens.count
            var cache = slot.prefillCaches
            for layer in cache {
                let excess = layer.offset - promptLen
                if excess > 0 { layer.trim(excess) }
            }
            for i in 0..<cache.count {
                if cache[i].isTrimmable && cache[i].offset > 0 {
                    cache[i].state = cache[i].state
                }
            }
            let layerStates = cache.map { $0.state }
            radix.insert(tokens: slot.inputTokens, layerStates: layerStates)
            DebugLogger.log("[BatchScheduler] Prefix cache save: \(slot.inputTokens.count) tokens")
        }

        slot.continuation.yield(StreamChunk(
            text: "",
            promptTokens: slot.promptTokenCount,
            completionTokens: slot.tokenCount,
            promptTime: slot.prefillTime,
            generateTime: generateTime
        ))
        slot.continuation.finish()
        _inFlightCount.withLock { $0 = max($0 - 1, 0) }

        DebugLogger.log("[BatchScheduler] Finished slot \(slot.id.uuidString.prefix(8)) (\(slot.tokenCount) tok, \(String(format: "%.2f", elapsed))s, in-flight: \(_inFlightCount.withLock { $0 })/\(maxConcurrent))")

        // Remove from batch cache and state
        let keepIndices = (0..<slots.count).filter { $0 != index }
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
                    if let bkvCache = batchCaches[layer] as? BatchKVCacheSimple {
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
            let totalTime = Date().timeIntervalSince(finishStart) * 1000
            let cacheTime = Date().timeIntervalSince(cacheStart) * 1000
            DebugLogger.log("[BatchScheduler] finishSlot timing: total=\(String(format: "%.1f", totalTime))ms cache_save=\(String(format: "%.1f", cacheTime))ms")
        }
        slots.remove(at: index)
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
}
