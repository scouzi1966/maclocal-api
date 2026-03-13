import Foundation
import MLX
import MLXLMCommon
import Tokenizers

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
        let sampler: LogitSampler
        var processor: LogitProcessor?
        var detokenizer: NaiveStreamingDetokenizer
        let maxTokens: Int?

        init(
            continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation,
            promptTokenCount: Int,
            prefillTime: TimeInterval,
            inputTokens: [Int],
            cachedTokens: Int,
            prefillCaches: [KVCache],
            lastTokenId: Int,
            sampler: LogitSampler,
            processor: LogitProcessor?,
            detokenizer: NaiveStreamingDetokenizer,
            maxTokens: Int?
        ) {
            self.id = UUID()
            self.continuation = continuation
            self.promptTokenCount = promptTokenCount
            self.prefillTime = prefillTime
            self.startTime = Date()
            self.inputTokens = inputTokens
            self.cachedTokens = cachedTokens
            self.prefillCaches = prefillCaches
            self.lastTokenId = lastTokenId
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

    private var pendingRequests: [PendingRequest] = []
    private var isShutdown = false
    private var loopTask: Task<Void, Never>?

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

        if enablePrefixCaching {
            let debug = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"
            self.radixCache = RadixTreeCache(
                modelID: configuration.name,
                maxEntries: 64,
                debugLogging: debug
            )
        } else {
            self.radixCache = nil
        }

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
    func submit(
        input: LMInput,
        parameters: GenerateParameters,
        promptTokens: Int
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        if isShutdown {
            return AsyncThrowingStream { $0.finish(throwing: MLXServiceError.serviceShuttingDown) }
        }

        let (stream, continuation) = AsyncThrowingStream<StreamChunk, Error>.makeStream()

        pendingRequests.append(PendingRequest(
            input: input,
            parameters: parameters,
            promptTokens: promptTokens,
            continuation: continuation
        ))

        DebugLogger.log("[BatchScheduler] Request submitted (\(pendingRequests.count) pending, \(slots.count) active)")
        ensureLoopRunning()

        return stream
    }

    /// Gracefully shut down.
    func shutdown() async {
        isShutdown = true

        for req in pendingRequests {
            req.continuation.finish(throwing: MLXServiceError.serviceShuttingDown)
        }
        pendingRequests.removeAll()

        if let task = loopTask {
            task.cancel()
            await task.value
            loopTask = nil
        }

        for slot in slots {
            slot.continuation.finish(throwing: MLXServiceError.serviceShuttingDown)
        }
        slots.removeAll()
        batchCaches = []
        batchState = nil
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
    private func generationLoop() async {
        var stepCount = 0
        while !isShutdown {
            // Prefill pending requests (up to capacity, one at a time)
            prefillPending()

            if slots.isEmpty { break }
            if Task.isCancelled { return }

            // --- Batched decode step ---
            let B = slots.count
            let tokens = MLXArray(slots.map { Int32($0.lastTokenId) }).reshaped([B, 1])
            let input = LMInput.Text(tokens: tokens)

            // Single batched model call — B sequences at once
            let output = model(input, cache: batchCaches, state: batchState)
            batchState = output.state

            // Extract last-position logits: [B, 1, V] → [B, V]
            let logits = output.logits[0..., -1, 0...]

            // --- Lazy sampling: build the full computation graph without eval ---
            // Each token flows: model → logits → process → sample → tokenArray.
            // Cache state is a dependency of the logits graph, so when we
            // asyncEval the token arrays, MLX evaluates everything (model +
            // cache updates) in one shot — matching the serial TokenIterator
            // pattern and avoiding the two-sync-point overhead of
            // eval(logits) + asyncEval(cacheArrays).
            var tokenArrays = [MLXArray]()
            for i in 0..<B {
                let slot = slots[i]
                let slotLogits = B == 1 ? logits : logits[i]
                let processed = slot.processor?.process(logits: slotLogits) ?? slotLogits
                tokenArrays.append(slot.sampler.sample(logits: processed))
            }

            // Single async eval — cache arrays evaluate as graph dependencies
            asyncEval(tokenArrays)

            // Materialize and dispatch
            var completedIndices: [Int] = []

            for i in 0..<B {
                let slot = slots[i]
                let token = tokenArrays[i].item(Int.self)

                slot.processor?.didSample(token: tokenArrays[i])

                // Track time to first decode token
                if slot.firstTokenTime == 0 {
                    slot.firstTokenTime = Date().timeIntervalSince(slot.startTime)
                }

                // EOS check
                if token == tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                    completedIndices.append(i)
                    continue
                }

                // Max tokens check
                if let max = slot.maxTokens, slot.tokenCount >= max {
                    completedIndices.append(i)
                    continue
                }

                // Update slot state
                slot.lastTokenId = token

                // Detokenize and yield
                slot.detokenizer.append(token: token)
                if let chunk = slot.detokenizer.next() {
                    slot.tokenCount += 1
                    slot.continuation.yield(StreamChunk(text: chunk))
                }
            }

            totalTokensGenerated += B
            if totalTokensGenerated % 1024 < B {
                Memory.clearCache()
            }

            // Finish completed slots (reverse order to preserve indices)
            for i in completedIndices.reversed() {
                finishSlot(at: i)
            }

            // Yield periodically to allow submit() calls to be processed
            stepCount += 1
            if stepCount % 16 == 0 || !pendingRequests.isEmpty {
                await Task.yield()
            }
        }

        loopTask = nil
    }

    // MARK: - Prefill

    /// Prefill pending requests up to capacity. Each request is processed
    /// individually (B=1), then its cache is merged into the batch.
    private func prefillPending() {
        while !pendingRequests.isEmpty && slots.count < maxConcurrent && !isShutdown {
            let req = pendingRequests.removeFirst()

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

            // Extract last-position logits and sample first token
            let logits = result.logits[0..., -1, 0...]
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
                prefillTime: prefillTime,
                inputTokens: inputTokens,
                cachedTokens: cachedTokens,
                prefillCaches: cache,
                lastTokenId: firstToken,
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
    }

    /// Merge a newly-prefilled individual cache into the batch.
    ///
    /// Handles mixed cache types: KVCacheSimple → BatchKVCacheSimple (batched K/V),
    /// MambaCache/ArraysCache → kept as-is with native extend/filter (recurrent state).
    private func mergeCacheIntoBatch(individualCache: [KVCache], modelState: LMOutput.State?) {
        if batchCaches.isEmpty {
            // First sequence: promote each per-layer cache appropriately
            batchCaches = individualCache.map { layerCache in
                if layerCache is ArraysCache {
                    // MambaCache/ArraysCache: create a copy so slot.prefillCaches stays independent
                    let copy = MambaCache()
                    copy.state = layerCache.state
                    return copy as KVCache
                } else {
                    return BatchKVCacheSimple.merge([layerCache]) as KVCache
                }
            }
        } else {
            // Extend existing batch with the new sequence
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
        let slot = slots[index]
        let elapsed = Date().timeIntervalSince(slot.startTime)
        let generateTime = elapsed - slot.prefillTime

        // Save prompt KV state to prefix cache before removal.
        // Use the original per-layer prefill caches (retained on the slot).
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

        DebugLogger.log("[BatchScheduler] Finished slot \(slot.id.uuidString.prefix(8)) (\(slot.tokenCount) tok, \(String(format: "%.2f", elapsed))s)")

        // Remove from batch cache and state
        let keepIndices = (0..<slots.count).filter { $0 != index }
        if keepIndices.isEmpty {
            batchCaches = []
            batchState = nil
        } else {
            let idxArray = MLXArray(keepIndices.map { Int32($0) })
            for layer in 0..<batchCaches.count {
                if let bkvCache = batchCaches[layer] as? BatchKVCacheSimple {
                    bkvCache.filter(keepIndices)
                } else if let acCache = batchCaches[layer] as? ArraysCache {
                    acCache.filter(batchIndices: idxArray)
                }
            }
            if let cas = batchState?.crossAttentionStates {
                let idxArray = MLXArray(keepIndices.map { Int32($0) })
                batchState = .init(crossAttentionStates: cas[idxArray])
            }
        }

        Stream.gpu.synchronize()
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
