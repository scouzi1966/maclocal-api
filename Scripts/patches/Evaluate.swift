// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import Tokenizers

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
public protocol LogitSampler {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let y = sampler.sample(logits: logits)
/// processor?.didSample(token: y)
/// ```
///
/// See also: ``LogitSampler``
public protocol LogitProcessor: Sendable {

    /// called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// called to visit and possibly modify the logits
    func process(logits: MLXArray) -> MLXArray

    /// called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Per-token log probability data produced during generation.
public struct TokenLogprobData: Sendable {
    /// The token ID that was sampled
    public let tokenId: Int
    /// Log probability of the sampled token
    public let logprob: Float
    /// Token IDs of the top alternatives (sorted by descending probability)
    public let topTokenIds: [Int]
    /// Log probabilities corresponding to topTokenIds
    public let topLogprobs: [Float]
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.
public struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    public var prefillStepSize: Int

    /// Maximum tokens to generate
    public var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries (except the first 4 tokens) will be overwritten.
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    public var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    public var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    public var kvGroupSize: Int

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    public var quantizedKVStart: Int

    /// sampling temperature
    public var temperature: Float

    /// top p sampling
    public var topP: Float

    /// penalty factor for repeating tokens
    public var repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    public var repetitionContextSize: Int

    /// top-k sampling: keep only the k most likely tokens (0 = disabled)
    public var topK: Int

    /// min-p sampling: filter tokens with probability < minP * maxProb (0.0 = disabled)
    public var minP: Float

    /// presence penalty: flat additive penalty for tokens already generated (0.0 = disabled)
    public var presencePenalty: Float

    /// random seed for reproducible sampling (nil = non-deterministic)
    public var seed: UInt64?

    /// whether to compute per-token log probabilities (default: false)
    public var computeLogprobs: Bool

    /// number of top alternative tokens to include in logprob output (0-20, default: 0)
    public var topLogprobsCount: Int

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        topK: Int = 0,
        minP: Float = 0.0,
        presencePenalty: Float = 0.0,
        seed: UInt64? = nil,
        computeLogprobs: Bool = false,
        topLogprobsCount: Int = 0,
        prefillStepSize: Int = 512
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.computeLogprobs = computeLogprobs
        self.topLogprobsCount = min(max(topLogprobsCount, 0), 20)
        self.prefillStepSize = prefillStepSize
    }

    public func sampler() -> LogitSampler {
        if temperature == 0 {
            return ArgMaxSampler()
        } else if topP > 0 && topP < 1 {
            return TopPSampler(temperature: temperature, topP: topP, seed: seed)
        } else {
            return CategoricalSampler(temperature: temperature, seed: seed)
        }
    }

    public func processor() -> LogitProcessor? {
        // Build processors in llama.cpp sampler chain order:
        // penalties (repetition, presence) → top_k → min_p
        // (temperature is handled by the sampler, not a processor)
        var processors: [LogitProcessor] = []

        if let repetitionPenalty, repetitionContextSize > 0, repetitionPenalty != 1.0 {
            processors.append(RepetitionContext(
                repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize))
        }

        if presencePenalty != 0.0 {
            processors.append(PresenceContext(
                presencePenalty: presencePenalty, contextSize: repetitionContextSize))
        }

        if topK > 0 {
            processors.append(TopKProcessor(k: topK))
        }

        if minP > 0 && minP < 1 {
            processors.append(MinPProcessor(minP: minP))
        }

        switch processors.count {
        case 0: return nil
        case 1: return processors[0]
        default: return CompositeLogitProcessor(processors)
        }
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
public struct ArgMaxSampler: LogitSampler {
    public init() {}

    public func sample(logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses `topP` and `temperature` to sample the logits.
public struct TopPSampler: LogitSampler {
    let temp: MLXArray
    let topP: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, topP: Float, seed: UInt64? = nil) {
        self.temp = MLXArray(temperature)
        self.topP = MLXArray(topP)
        if let seed {
            self.randomState = MLXRandom.RandomState(seed: seed)
        } else {
            self.randomState = MLXRandom.RandomState()
        }
    }

    public func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            let probs = softmax(logits / temp, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)

            // probs shape is [B,V] and after take it will be [1, B, V], so we squeeze it back to [B, V]
            let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)

            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))

            let sortedToken = categorical(log(topProbs))
            return sortedIndices.squeezed(axis: 0)[sortedToken]
        }
    }
}

/// Sampler that uses `temperature` to sample the logits.
public struct CategoricalSampler: LogitSampler {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    public init(temperature: Float, seed: UInt64? = nil) {
        self.temp = MLXArray(temperature)
        if let seed {
            self.randomState = MLXRandom.RandomState(seed: seed)
        } else {
            self.randomState = MLXRandom.RandomState()
        }
    }

    public func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// Processor that applies a flat additive penalty for tokens already generated.
///
/// Follows the OpenAI / vLLM / SGLang convention: for every unique token that
/// has appeared in the generated context, subtract `presencePenalty` from its logit.
/// Unlike ``RepetitionContext`` (which is multiplicative and sign-aware), the penalty
/// is always subtracted, matching the OpenAI API specification.
public struct PresenceContext: LogitProcessor {
    /// tokens in the context sliding window
    var tokens = [Int]()

    /// current write index into the tokens circular array
    var index = 0

    /// additive penalty for tokens already present in context
    let presencePenalty: Float

    /// number of tokens to consider
    let contextSize: Int

    public init(presencePenalty: Float, contextSize: Int = 64) {
        precondition(contextSize > 0)
        self.presencePenalty = presencePenalty
        self.contextSize = contextSize
    }

    mutating public func prompt(_ prompt: MLXArray) {
        if prompt.shape[0] <= contextSize {
            self.tokens = prompt.asArray(Int.self)
        } else {
            self.tokens = prompt[(-contextSize)...].asArray(Int.self)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        guard !tokens.isEmpty else { return logits }
        let uniqueTokens = Array(Set(tokens))
        let indices = MLXArray(uniqueTokens.map { UInt32($0) })
        let penalty = MLXArray(presencePenalty)
        logits[0..., indices] = logits[0..., indices] - penalty
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        if tokens.count >= contextSize {
            tokens[index] = token.item(Int.self)
            index = (index + 1) % contextSize
        } else {
            tokens.append(token.item(Int.self))
        }
    }
}

/// Processor that keeps only the top-K logits, setting the rest to `-Float.infinity`.
///
/// When `k <= 0` or `k >= vocabSize` the processor is effectively a no-op.
/// Follows the same algorithm used by llama.cpp and vLLM.
public struct TopKProcessor: LogitProcessor {
    let k: Int

    public init(k: Int) {
        self.k = k
    }

    public func prompt(_ prompt: MLXArray) {}
    public func didSample(token: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        let vocabSize = logits.dim(-1)
        guard k > 0, k < vocabSize else { return logits }

        // Sort descending along the last axis to find the k-th largest value
        let sorted = MLX.sorted(logits, axis: -1)
        // k-th largest is at index [vocabSize - k] in ascending-sorted array
        let threshold = sorted[0..., vocabSize - k]

        return MLX.where(logits .>= threshold, logits, MLXArray(-Float.infinity))
    }
}

/// Processor that filters tokens whose logit is below `min_p * max_logit` (in log-space).
///
/// This avoids a full softmax: instead of computing probabilities, we use the
/// identity `p(x) >= min_p * p_max` ⟺ `logit(x) >= max_logit + log(min_p)`.
/// Follows the llama.cpp / vLLM log-space approach.
public struct MinPProcessor: LogitProcessor {
    let minP: Float

    public init(minP: Float) {
        self.minP = minP
    }

    public func prompt(_ prompt: MLXArray) {}
    public func didSample(token: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        guard minP > 0, minP < 1 else { return logits }

        let maxLogit = MLX.max(logits, axis: -1, keepDims: true)
        let threshold = maxLogit + MLXArray(log(minP))

        return MLX.where(logits .>= threshold, logits, MLXArray(-Float.infinity))
    }
}

/// Wraps multiple ``LogitProcessor`` instances into a single processor.
///
/// The processors are called in order during `process(logits:)`, matching
/// the llama.cpp sampler chain convention: penalties → top_k → top_p → min_p → temperature.
public struct CompositeLogitProcessor: LogitProcessor {
    var processors: [LogitProcessor]

    public init(_ processors: [LogitProcessor]) {
        self.processors = processors
    }

    mutating public func prompt(_ prompt: MLXArray) {
        for i in processors.indices {
            processors[i].prompt(prompt)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        for processor in processors {
            logits = processor.process(logits: logits)
        }
        return logits
    }

    mutating public func didSample(token: MLXArray) {
        for i in processors.indices {
            processors[i].didSample(token: token)
        }
    }
}

/// Processor that implements a `repetitionPenalty`
public struct RepetitionContext: LogitProcessor {
    /// tokens in the repetition context sliding window
    var tokens = [Int]()

    /// current write index into the tokens circular array
    var index = 0

    /// penalty factor for repeating tokens
    let repetitionPenalty: Float

    /// number of tokens to consider for repetition penalty
    let repetitionContextSize: Int

    public init(repetitionPenalty: Float, repetitionContextSize: Int) {
        precondition(repetitionContextSize > 0)
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }

    mutating public func prompt(_ prompt: MLXArray) {
        if prompt.shape[0] <= repetitionContextSize {
            self.tokens = prompt.asArray(Int.self)
        } else {
            self.tokens = prompt[(-repetitionContextSize)...].asArray(Int.self)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        if tokens.count > 0 {
            let indices = MLXArray(tokens.map { UInt32($0) })
            var selectedLogits = logits[0..., indices]

            selectedLogits = MLX.where(
                selectedLogits .< 0, selectedLogits * repetitionPenalty,
                selectedLogits / repetitionPenalty)

            logits[0..., indices] = selectedLogits
            return logits
        }

        return logits
    }

    mutating public func didSample(token: MLXArray) {
        if tokens.count >= repetitionContextSize {
            tokens[index] = token.item(Int.self)
            index = (index + 1) % repetitionContextSize
        } else {
            tokens.append(token.item(Int.self))
        }
    }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>`.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: generateParameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
public struct TokenIterator: Sequence, IteratorProtocol {
    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int

    // Internal metrics
    var promptPrefillTime: TimeInterval = 0.0

    // Logprob computation
    let computeLogprobs: Bool
    let topLogprobsCount: Int
    let temperatureForLogprobs: Float
    /// Logprob info for the token that will be returned by the current `next()` call.
    public private(set) var lastLogprobInfo: TokenLogprobData?
    /// Logprob info computed during `convertToToken()`, pending for the next `next()` return.
    private var pendingLogprobInfo: TokenLogprobData?

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:parameters:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    public init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        self.computeLogprobs = parameters.computeLogprobs
        self.topLogprobsCount = parameters.topLogprobsCount
        self.temperatureForLogprobs = parameters.temperature

        self.promptPrefillTime = try measure {
            try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
        }
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        self.computeLogprobs = parameters.computeLogprobs
        self.topLogprobsCount = parameters.topLogprobsCount
        self.temperatureForLogprobs = parameters.temperature

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefillStepSize: optional prefill step size
    ///   - maxTokens: maximum number of tokens to generate
    public init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler, prefillStepSize: Int = 512,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = 64
        self.quantizedKVStart = 0

        self.computeLogprobs = false
        self.topLogprobsCount = 0
        self.temperatureForLogprobs = 0

        self.promptPrefillTime = try measure {
            try prepare(input: input, windowSize: prefillStepSize)
        }
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)

        case .logits(let result):
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)

            break
        }
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        // process the logits (one hot array of possible tokens)
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits

        // transform logits back to a token
        let y = sampler.sample(logits: logits)

        processor?.didSample(token: y)

        // Compute per-token log probabilities if requested
        if computeLogprobs {
            var lpLogits = logits
            if lpLogits.dtype == .bfloat16 {
                lpLogits = lpLogits.asType(.float32)
            }
            // Apply temperature scaling (same as the sampler uses)
            if temperatureForLogprobs > 0 {
                lpLogits = lpLogits / MLXArray(temperatureForLogprobs)
            }
            let probs = softmax(lpLogits, axis: -1)
            let logProbs = log(probs)
            let flatLogProbs = logProbs.reshaped(-1)

            let tokenId = y.item(Int.self)
            let tokenLogprob = flatLogProbs[tokenId].item(Float.self)

            var topIds = [Int]()
            var topLps = [Float]()
            if topLogprobsCount > 0 {
                let vocabSize = flatLogProbs.dim(0)
                let n = Swift.min(topLogprobsCount, vocabSize)
                let sorted = argSort(flatLogProbs, axis: -1)
                for i in 0..<n {
                    let idx = sorted[vocabSize - 1 - i].item(Int.self)
                    topIds.append(idx)
                    topLps.append(flatLogProbs[idx].item(Float.self))
                }
            }

            pendingLogprobInfo = TokenLogprobData(
                tokenId: tokenId,
                logprob: tokenLogprob,
                topTokenIds: topIds,
                topLogprobs: topLps
            )
        } else {
            pendingLogprobInfo = nil
        }

        return y
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        let result = model(
            previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )

        return convertToToken(logits: result.logits)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // Promote pending logprob info: the logprob computed during the PREVIOUS
        // step() corresponds to the token we are about to return (previousY).
        lastLogprobInfo = pendingLogprobInfo

        // save current value -- this will be returned
        let previousY = y

        // compute the next state and async eval the next token
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)

        tokenCount += 1

        // Periodically clear GPU memory cache to prevent fragmentation,
        // especially important for large MoE models. Matches mlx_lm behavior.
        if tokenCount % 256 == 0 {
            Memory.clearCache()
        }

        return previousY.tokens.item(Int.self)
    }
}

/// Result of a call to a deprecated callback-based generate function.
public struct GenerateResult {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokens: The array of tokens generated.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    public init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokens = tokens
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    /// input (prompt, images, etc.)
    public let inputText: LMInput.Text

    @available(*, deprecated, message: "use inputText")
    public var promptTokens: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    /// output tokens
    public let tokens: [Int]

    /// output text
    public let output: String

    /// The number of tokens included in the input prompt.
    public var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    public var generationTokenCount: Int { tokens.count }

    /// time to process the prompt / generate the first token
    public let promptTime: TimeInterval

    /// time to generate the remaining tokens
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(tokens.count) / generateTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in deprecated callback-based generate functions.
public enum GenerateDisposition: Sendable {
    /// keep producing tokens until an EOS token is produced
    case more

    /// stop producing tokens, e.g. a token limit has been hit
    case stop
}

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognize special tokens
///   - extraEOSTokens: any additional stop tokens
///   - didGenerate: visitor for the tokens as they are generated
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    // this is a compatibility cover -- create the required values
    // for the iteration
    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    // Build complete EOS token set from all sources
    var eosTokenIds = context.configuration.eosTokenIds
    if let tokenizerEos = context.tokenizer.eosTokenId {
        eosTokenIds.insert(tokenizerEos)
    }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            eosTokenIds.insert(id)
        }
    }

    var tokens = [Int]()

    for token in iterator {
        // compute the timing for the prompt
        if tokens.isEmpty {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        if token == context.tokenizer.unknownTokenId || eosTokenIds.contains(token) {
            break
        }
        tokens.append(token)

        if didGenerate(tokens) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // TokenIterator uses `asyncEval()` to keep the pipeline full. If the caller
    // exits the program right away, those tasks will still be executing and will
    // hit assertions as the mlx scheduler is torn down. Synchronize with the stream
    // to make sure it is complete.
    Stream().synchronize()

    return GenerateResult(
        inputText: input.text, tokens: tokens,
        output: context.tokenizer.decode(tokens: tokens),
        promptTime: promptTime + iterator.promptPrefillTime,
        generateTime: generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:parameters:context:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    // Build complete EOS token set from all sources
    var eosTokenIds = context.configuration.eosTokenIds
    if let tokenizerEos = context.tokenizer.eosTokenId {
        eosTokenIds.insert(tokenizerEos)
    }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) {
            eosTokenIds.insert(id)
        }
    }

    var tokenCount = 0

    for token in iterator {
        // Compute the timing for the prompt
        if promptTime == 0 {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        // Check for end-of-sequence tokens
        if token == context.tokenizer.unknownTokenId || eosTokenIds.contains(token) {
            break
        }

        tokenCount += 1

        // Invoke the callback with the current token
        if didGenerate(token) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // Synchronize with the stream to ensure tasks are completed
    Stream().synchronize()

    return GenerateCompletionInfo(
        promptTokenCount: input.text.tokens.size,
        generationTokenCount: tokenCount,
        promptTime: promptTime + iterator.promptPrefillTime,
        generationTime: generateTime
    )
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent text chunks, tool calls, or summary
/// completion information.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  This is typically OK for
/// one-shot calls, but for "chat session" type calls consider using
/// ``generateTask(promptTokenCount:context:iterator:)``
/// so that the end of the generation task can be observed.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated text chunks (`.chunk`),
///   tool calls (`.toolCall`), and completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: generateParameters, context: context)
///
/// // Process the stream asynchronously to handle text chunks and completion info.
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     }
/// }
/// ```
public func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext
) throws -> AsyncStream<Generation> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, parameters: parameters)
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator)
    return stream
}

@available(
    *, deprecated,
    message: "use a higher level generate() call or use generateTask() for fine grained control"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator
) -> AsyncStream<Generation> {
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator)
    return stream
}

/// Low-level token generation using a ``TokenIterator``, returning an
/// `AsyncStream<Generation>` and a `Task`.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  Callers can await
/// the `task` to observe when the use of the parameters is complete.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
/// - Returns: An `AsyncStream` that emits `Generation` values and a `Task`
public func generateTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator
) -> (AsyncStream<Generation>, Task<Void, Never>) {

    let (stream, continuation) = AsyncStream<Generation>.makeStream()

    let iterator = SendableBox(iterator)

    // Launch a Task to perform iteration asynchronously.
    let task = Task {
        var iterator = iterator.consume()

        var start = Date.timeIntervalSinceReferenceDate
        var promptTime: TimeInterval = 0

        // Build complete EOS token set from all sources
        var eosTokenIds = modelConfiguration.eosTokenIds
        if let tokenizerEos = tokenizer.eosTokenId {
            eosTokenIds.insert(tokenizerEos)
        }
        for token in modelConfiguration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                eosTokenIds.insert(id)
            }
        }

        var tokenCount = 0
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        let toolCallProcessor = ToolCallProcessor(
            format: modelConfiguration.toolCallFormat ?? .json
        )
        var pendingLogprobs = [TokenLogprobData]()

        while let token = iterator.next() {

            // Check for cancellation on every loop iteration.
            if Task.isCancelled {
                break
            }

            if promptTime == 0 {
                let now = Date.timeIntervalSinceReferenceDate
                promptTime = now - start
                start = now
            }

            if token == tokenizer.unknownTokenId || eosTokenIds.contains(token) {
                break
            }

            // Buffer logprob data for this token
            if let lpInfo = iterator.lastLogprobInfo {
                pendingLogprobs.append(lpInfo)
            }

            detokenizer.append(token: token)
            if let chunk = detokenizer.next() {
                tokenCount += 1

                // Process chunk through the tool call processor
                if let textToYield = toolCallProcessor.processChunk(chunk) {
                    // Yield buffered logprobs before the chunk they belong to
                    if !pendingLogprobs.isEmpty {
                        continuation.yield(.tokenLogprobs(pendingLogprobs))
                        pendingLogprobs = []
                    }
                    if case .terminated = continuation.yield(.chunk(textToYield)) {
                        break
                    }
                }

                // Check if we have a complete tool call
                if let toolCall = toolCallProcessor.toolCalls.popLast() {
                    if case .terminated = continuation.yield(.toolCall(toolCall)) {
                        break
                    }
                }
            }
        }

        let now = Date.timeIntervalSinceReferenceDate
        let generateTime = now - start

        let info = GenerateCompletionInfo(
            promptTokenCount: promptTokenCount,
            generationTokenCount: tokenCount,
            promptTime: promptTime + iterator.promptPrefillTime,
            generationTime: generateTime
        )
        continuation.yield(.info(info))

        // Synchronize with the stream to ensure tasks are completed
        Stream().synchronize()

        // Finalize the stream
        continuation.finish()
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { _ in
        task.cancel()
    }

    return (stream, task)
}

/// Represents metadata and statistics related to token generation.
///
/// Provides information about the number of tokens processed during both the prompt and generation phases, as well as the time taken for each phase.
public struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    public let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    public let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    public let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    public let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    public var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    public var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
    }

    public func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s, \(promptTime.formatted())s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Represents the different stages or outputs of the token generation process.
///
/// This enum distinguishes between the following:
/// - `.chunk`: A decoded string from one or more tokens generated by the language model.
/// - `.toolCall`: A tool call parsed from the generated output.
/// - `.info`: Metadata and performance statistics about the generation process.
public enum Generation: Sendable {
    /// A generated text chunk as a String.
    case chunk(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// A tool call from the language model.
    case toolCall(ToolCall)

    /// Per-token log probability data for the preceding chunk.
    case tokenLogprobs([TokenLogprobData])

    /// Generated text or nil
    public var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .info: nil
        case .toolCall: nil
        case .tokenLogprobs: nil
        }
    }

    /// Completion info or nil
    public var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .info(let info): info
        case .toolCall: nil
        case .tokenLogprobs: nil
        }
    }

    /// Tool call or nil
    public var toolCall: ToolCall? {
        switch self {
        case .chunk: nil
        case .info: nil
        case .toolCall(let toolCall): toolCall
        case .tokenLogprobs: nil
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    public static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}

/// Measures the execution time of a closure.
private func measure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}
