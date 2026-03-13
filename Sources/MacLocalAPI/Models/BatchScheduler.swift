import Foundation
import MLX
import MLXLMCommon
import Tokenizers

/// Manages concurrent generation with dynamic slot allocation and request queuing.
///
/// All GPU operations (prefill + decode) run in a **single serial loop** to avoid
/// Metal command buffer conflicts. MLX operations are NOT thread-safe across
/// concurrent Tasks — multiple `model()` calls from different threads crash in
/// `mlx_quantized_matmul`.
///
/// CPU/GPU interleaving happens naturally within the round-robin loop:
/// `iterator.next()` for slot N submits GPU work via `asyncEval()` and returns
/// immediately (the result is read on the NEXT call). While the GPU processes
/// slot N's token, the CPU builds slot N+1's computation graph.
///
/// Phase 2 (future): Upgrade to dense batched decoding where multiple sequences
/// are packed into a single `model(batch_input, batch_cache)` call.
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

    /// Per-request state for the round-robin generation loop.
    /// Wraps TokenIterator (a value type) in a class for reference semantics.
    private class SlotState {
        let id: UUID
        var iterator: TokenIterator
        var detokenizer: NaiveStreamingDetokenizer
        let continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation
        let promptTokenCount: Int
        let startTime: Date
        /// Time spent in prefill (TokenIterator init), measured externally.
        let prefillTime: TimeInterval
        var tokenCount = 0
        /// Wall-clock time from slot start to first decode token.
        var firstTokenTime: TimeInterval = 0
        /// Full prompt token array (for prefix cache save on completion).
        let inputTokens: [Int]
        /// KV cache reference (shared with TokenIterator, for prefix cache save).
        let generationCache: [KVCache]
        /// Number of tokens restored from prefix cache (0 = full prefill).
        let cachedTokens: Int

        init(
            iterator: TokenIterator,
            detokenizer: NaiveStreamingDetokenizer,
            continuation: AsyncThrowingStream<StreamChunk, Error>.Continuation,
            promptTokenCount: Int,
            prefillTime: TimeInterval,
            inputTokens: [Int],
            generationCache: [KVCache],
            cachedTokens: Int
        ) {
            self.id = UUID()
            self.iterator = iterator
            self.detokenizer = detokenizer
            self.continuation = continuation
            self.promptTokenCount = promptTokenCount
            self.prefillTime = prefillTime
            self.startTime = Date()
            self.inputTokens = inputTokens
            self.generationCache = generationCache
            self.cachedTokens = cachedTokens
        }
    }

    /// Active slots in the round-robin decode loop.
    private var slots: [SlotState] = []

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
            slot.iterator.teardown()
            slot.continuation.finish(throwing: MLXServiceError.serviceShuttingDown)
        }
        slots.removeAll()
    }

    // MARK: - Private

    private func ensureLoopRunning() {
        guard loopTask == nil else { return }
        loopTask = Task { [weak self] in
            await self?.generationLoop()
        }
    }

    /// Single serial loop: prefill pending requests, then round-robin decode.
    /// ALL GPU operations happen here — never from concurrent Tasks.
    private func generationLoop() async {
        while !isShutdown {
            // Prefill pending requests (up to capacity, one at a time)
            prefillPending()

            if slots.isEmpty { break }

            // Round-robin: one decode step per active slot
            var completedIndices: [Int] = []

            for i in 0..<slots.count {
                if Task.isCancelled { return }
                let slot = slots[i]

                guard let token = slot.iterator.next() else {
                    completedIndices.append(i)
                    continue
                }

                // Track time to first decode token
                if slot.firstTokenTime == 0 {
                    slot.firstTokenTime = Date().timeIntervalSince(slot.startTime)
                }

                // EOS check
                if token == tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                    completedIndices.append(i)
                    continue
                }

                // Resolve logprobs if available
                var resolved: [ResolvedLogprob]? = nil
                if let lpInfo = slot.iterator.lastLogprobInfo {
                    resolved = [resolveLogprob(lpInfo)]
                }

                // Detokenize and yield
                slot.detokenizer.append(token: token)
                if let chunk = slot.detokenizer.next() {
                    slot.tokenCount += 1
                    slot.continuation.yield(StreamChunk(text: chunk, logprobs: resolved))
                }
            }

            // Finish completed slots (reverse order to preserve indices)
            for i in completedIndices.reversed() {
                finishSlot(at: i)
            }

            // Yield to allow submit() calls to be processed by the actor
            await Task.yield()
        }

        loopTask = nil
    }

    /// Prefill pending requests up to capacity. Runs on the actor (GPU-serial).
    private func prefillPending() {
        while !pendingRequests.isEmpty && slots.count < maxConcurrent && !isShutdown {
            let req = pendingRequests.removeFirst()

            do {
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
                        // Restore per-layer KV cache state
                        for i in 0..<cache.count where i < states.count {
                            cache[i].state = states[i]
                        }
                        // Trim to effective prefix length
                        for i in 0..<cache.count {
                            let excess = cache[i].offset - effectivePrefix
                            if excess > 0 { cache[i].trim(excess) }
                        }
                        // Physically truncate trimmed arrays (#47)
                        for i in 0..<cache.count {
                            if cache[i].isTrimmable && cache[i].offset > 0 {
                                cache[i].state = cache[i].state
                            }
                        }
                        // Create suffix-only input
                        let suffixTokens = Array(inputTokens[effectivePrefix...])
                        generateInput = LMInput(text: .init(tokens: MLXArray(suffixTokens)))
                        cachedTokens = effectivePrefix
                        DebugLogger.log("[BatchScheduler] Prefix cache hit: \(effectivePrefix)/\(inputTokens.count) tokens cached, processing \(suffixTokens.count) suffix")
                    }
                }

                let prefillStart = Date()
                let iterator = try TokenIterator(
                    input: generateInput,
                    model: model,
                    cache: cache,
                    parameters: req.parameters
                )
                let prefillTime = Date().timeIntervalSince(prefillStart)
                let detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

                let slot = SlotState(
                    iterator: iterator,
                    detokenizer: detokenizer,
                    continuation: req.continuation,
                    promptTokenCount: req.promptTokens,
                    prefillTime: prefillTime,
                    inputTokens: inputTokens,
                    generationCache: cache,
                    cachedTokens: cachedTokens
                )

                // Emit cached token count so the controller can include it in usage
                if cachedTokens > 0 {
                    req.continuation.yield(StreamChunk(text: "", cachedTokens: cachedTokens))
                }

                slots.append(slot)
                DebugLogger.log("[BatchScheduler] Prefilled slot \(slot.id.uuidString.prefix(8)) (\(slots.count) active, \(cachedTokens > 0 ? "cache hit \(cachedTokens) tokens" : "full prefill"))")
            } catch {
                req.continuation.finish(throwing: error)
                DebugLogger.log("[BatchScheduler] Prefill error: \(error.localizedDescription)")
            }
        }
    }

    /// Finish a completed slot: save prefix cache, yield timing info, teardown, remove.
    private func finishSlot(at index: Int) {
        let slot = slots[index]
        let elapsed = Date().timeIntervalSince(slot.startTime)
        let generateTime = elapsed - slot.prefillTime

        // Save prompt KV state to prefix cache before teardown
        if let radix = radixCache, !slot.inputTokens.isEmpty {
            let promptLen = slot.inputTokens.count
            var cache = slot.generationCache
            // Trim generated tokens beyond prompt length
            for layer in cache {
                let excess = layer.offset - promptLen
                if excess > 0 { layer.trim(excess) }
            }
            // Physically truncate trimmed arrays (#47)
            for i in 0..<cache.count {
                if cache[i].isTrimmable && cache[i].offset > 0 {
                    cache[i].state = cache[i].state
                }
            }
            let layerStates = cache.map { $0.state }
            radix.insert(tokens: slot.inputTokens, layerStates: layerStates)
            DebugLogger.log("[BatchScheduler] Prefix cache save: \(slot.inputTokens.count) tokens, \(cache.count) layers")
        }

        slot.continuation.yield(StreamChunk(
            text: "",
            promptTokens: slot.promptTokenCount,
            completionTokens: slot.tokenCount,
            promptTime: slot.prefillTime,
            generateTime: generateTime
        ))

        slot.iterator.teardown()
        Stream.gpu.synchronize()
        slot.continuation.finish()

        DebugLogger.log("[BatchScheduler] Finished slot \(slot.id.uuidString.prefix(8)) (\(slot.tokenCount) tok, \(String(format: "%.2f", elapsed))s)")
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
