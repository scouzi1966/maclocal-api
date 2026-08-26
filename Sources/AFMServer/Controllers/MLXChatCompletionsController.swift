import Vapor
import AFMKit
import AFMKitCore
import AFMKitMLX
import Foundation

struct FinalizedAssistantTurn {
    let finishReason: String
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ResponseToolCall]?
}

/// Swift 6 does not infer Sendable conformance for the transport tuple used by
/// AFMChatServing, even though every stored component is immutable and
/// Sendable. Keep that compatibility tuple local to this explicit task box.
private struct CancellableGenerationResult: @unchecked Sendable {
    let value: AFMChatGenerationResult
}

/// Prevents API stop sequences from reaching an SSE client, including when a
/// delimiter is split across provider chunks. The provider may report that it
/// stopped only after yielding the token that completed the delimiter, so the
/// controller must retain any suffix that could still become a stop sequence.
struct StreamingStopSequenceFilter {
    private let stopSequences: [String]
    private var pending = ""
    private(set) var stopped = false

    init(stopSequences: [String]?) {
        self.stopSequences = Array(Set((stopSequences ?? []).filter { !$0.isEmpty }))
    }

    mutating func consume(_ text: String) -> String {
        guard !stopped else { return "" }
        guard !stopSequences.isEmpty else { return text }
        pending += text

        var earliestStop: Range<String.Index>?
        for stop in stopSequences {
            guard let range = pending.range(of: stop) else { continue }
            if earliestStop == nil || range.lowerBound < earliestStop!.lowerBound {
                earliestStop = range
            }
        }
        if let earliestStop {
            let output = String(pending[..<earliestStop.lowerBound])
            pending = ""
            stopped = true
            return output
        }

        var retainedCount = 0
        let maximumCandidateLength = min(
            pending.count,
            stopSequences.map(\.count).max() ?? 0
        )
        if maximumCandidateLength > 0 {
            for length in stride(from: maximumCandidateLength, through: 1, by: -1) {
                let suffix = pending.suffix(length)
                if stopSequences.contains(where: { $0.hasPrefix(suffix) }) {
                    retainedCount = length
                    break
                }
            }
        }

        guard retainedCount > 0 else {
            defer { pending = "" }
            return pending
        }
        let boundary = pending.index(pending.endIndex, offsetBy: -retainedCount)
        let output = String(pending[..<boundary])
        pending = String(pending[boundary...])
        return output
    }

    mutating func flush() -> String {
        guard !stopped else { return "" }
        defer { pending = "" }
        return pending
    }
}

struct MLXChatCompletionsController: RouteCollection {
    private static let degenerateTailRegex = try! NSRegularExpression(pattern: "([!?.:,;`~_\\-*=|])\\1{79,}$")

    /// Max time (seconds) to wait for a concurrent slot before returning 503.
    /// RotatingKVCache models run serial, so queued requests can wait a long time.
    private static let slotQueueTimeout: TimeInterval = 240

    private let streamingEnabled: Bool
    private let modelID: String
    private let service: any AFMChatServing
    private let temperature: Double?
    private let topP: Double?
    private let maxTokens: Int?
    private let repetitionPenalty: Double?
    private let topK: Int?
    private let minP: Double?
    private let presencePenalty: Double?
    private let seed: Int?
    private let maxLogprobs: Int
    private let veryVerbose: Bool
    private let trace: Bool
    private let rawOutput: Bool
    private let stop: String?

    init(
        streamingEnabled: Bool = true,
        modelID: String,
        service: any AFMChatServing,
        temperature: Double?,
        topP: Double? = nil,
        maxTokens: Int? = nil,
        repetitionPenalty: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        maxLogprobs: Int = 20,
        veryVerbose: Bool = false,
        trace: Bool = false,
        rawOutput: Bool = false,
        stop: String? = nil
    ) {
        self.streamingEnabled = streamingEnabled
        self.modelID = modelID
        self.service = service
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.maxLogprobs = maxLogprobs
        self.veryVerbose = veryVerbose
        self.trace = trace
        self.rawOutput = rawOutput
        self.stop = stop
    }

    /// Merge CLI --stop sequences with API-level stop sequences, deduplicating.
    private func mergeStopSequences(cliStop: String?, apiStop: [String]?) -> [String]? {
        var merged: [String] = []
        if let cliStopString = cliStop {
            let cliArray = cliStopString.split(separator: ",").map { String($0.trimmingCharacters(in: .whitespaces)) }
            merged.append(contentsOf: cliArray)
        }
        if let apiArray = apiStop { merged.append(contentsOf: apiArray) }
        guard !merged.isEmpty else { return nil }
        var seen = Set<String>()
        return merged.filter { seen.insert($0).inserted }
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "chat", "completions", body: .collect(maxSize: "100mb"), use: chatCompletions)
        v1.on(.OPTIONS, "chat", "completions", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization, X-AFM-Profile")
        return response
    }

    private static let debugPipeline = ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1"

    static func containsDeclaredMedia(_ messages: [Message]) -> Bool {
        messages.contains { message in
            guard let content = message.content, case .parts(let parts) = content else {
                return false
            }
            return parts.contains { part in
                part.type == "image_url" || part.type == "input_audio"
            }
        }
    }

    static func redactedRequestJSON(_ request: ChatCompletionRequest) -> String {
        guard
            let encoded = try? JSONEncoder().encode(request),
            var object = try? JSONSerialization.jsonObject(with: encoded) as? [String: Any],
            var messages = object["messages"] as? [[String: Any]]
        else {
            return "<request unavailable>"
        }

        for messageIndex in messages.indices {
            guard var parts = messages[messageIndex]["content"] as? [[String: Any]] else {
                continue
            }
            for partIndex in parts.indices {
                switch parts[partIndex]["type"] as? String {
                case "image_url":
                    if var imageURL = parts[partIndex]["image_url"] as? [String: Any] {
                        imageURL["url"] = "<redacted-media-reference>"
                        parts[partIndex]["image_url"] = imageURL
                    }
                case "input_audio":
                    if var inputAudio = parts[partIndex]["input_audio"] as? [String: Any] {
                        inputAudio["data"] = "<redacted-audio-data>"
                        parts[partIndex]["input_audio"] = inputAudio
                    }
                default:
                    break
                }
            }
            messages[messageIndex]["content"] = parts
        }
        object["messages"] = messages

        guard
            let redacted = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
            ),
            let json = String(data: redacted, encoding: .utf8)
        else {
            return "<request unavailable>"
        }
        return json
    }

    func chatCompletions(req: Request) async throws -> Response {
        // RequestIDMiddleware sets this; controller piggybacks for log correlation. (T1.1)
        let reqId = req.afmRequestID
        let inflightRegistry = req.application.inflightRegistry
        let cancelHandle = CancellableTaskHandle()
        let requestRegistration: InflightRequestRegistry.Registration?
        if !reqId.isEmpty {
            requestRegistration = await inflightRegistry.register(
                id: reqId,
                cancel: { cancelHandle.cancel() }
            )
        } else {
            requestRegistration = nil
        }
        var requestRegistered = requestRegistration != nil
        do {
            let httpArrival = Self.debugPipeline ? Date() : Date.distantPast
            let chatRequest = try req.content.decode(ChatCompletionRequest.self)
            if veryVerbose {
                print("\(Self.pink)[\(Self.timestamp())] RECV MLX full request:\n\(Self.redactedRequestJSON(chatRequest))\(Self.reset)"); fflush(stdout)
                if let lastUser = chatRequest.messages.last(where: { $0.role == "user" }) {
                    let prompt = lastUser.textContent
                    let truncated = prompt.count > 500 ? String(prompt.prefix(500)) + "..." : prompt
                    print("\(Self.red)[\(Self.timestamp())] RECV MLX user prompt:\n  \(truncated)\(Self.reset)"); fflush(stdout)
                }
            }
            // Apply server-level --guided-json default when request omits response_format (#97)
            let effectiveResponseFormat = service.effectiveResponseFormat(requestFormat: chatRequest.responseFormat)

            // Detect strict-mode downgrade: user requested grammar enforcement but admin didn't enable the engine
            let grammarDowngraded = service.shouldDowngradeGrammarConstraints(
                responseFormat: effectiveResponseFormat,
                tools: chatRequest.tools
            )

            guard !chatRequest.messages.isEmpty else {
                return try await createErrorResponse(req: req, error: OpenAIError(message: "At least one message is required"), status: .badRequest)
            }

            // Validate top_logprobs against server max (vLLM-compatible)
            if let requestedTopLogprobs = chatRequest.topLogprobs, requestedTopLogprobs > maxLogprobs {
                return try await createErrorResponse(
                    req: req,
                    error: OpenAIError(
                        message: "top_logprobs must be <= \(maxLogprobs). Received \(requestedTopLogprobs). Use --max-logprobs to increase the server limit.",
                        type: "invalid_request_error"
                    ),
                    status: .badRequest
                )
            }

            if let requestedModelRaw = chatRequest.model?.trimmingCharacters(in: .whitespacesAndNewlines),
               !requestedModelRaw.isEmpty,
               service.normalizeModel(requestedModelRaw) != modelID {
                // WebUI may send transformed model identifiers; afm mlx always serves the active model.
                print("[\(Self.timestamp())] MLX request model '\(requestedModelRaw)' does not match active model '\(modelID)'; serving active model"); fflush(stdout)
            }

            let effectiveTools = try Self.resolveEffectiveTools(
                chatRequest.tools,
                toolChoice: chatRequest.toolChoice
            )

            let hasTools = effectiveTools != nil && !(effectiveTools?.isEmpty ?? true)
            if hasTools && veryVerbose {
                let toolNames = chatRequest.tools!.map { $0.function.name }.joined(separator: ", ")
                print("\(Self.gold)[\(Self.timestamp())] RECV tools: [\(toolNames)]\(Self.reset)")
            }
            // -VV: Full tool schemas as received from client
            if hasTools && trace {
                for tool in effectiveTools! {
                    let schemaJSON: String
                    if let params = tool.function.parameters,
                       let data = try? JSONSerialization.data(withJSONObject: params.toJinjaCompatible(), options: [.prettyPrinted, .sortedKeys]),
                       let str = String(data: data, encoding: .utf8) {
                        schemaJSON = str
                    } else {
                        schemaJSON = "(no parameters)"
                    }
                    print("\(Self.cyan)[\(Self.timestamp())] [VV] RECV tool schema: \(tool.function.name)\n\(schemaJSON)\(Self.reset)")
                }
                fflush(stdout)
            }

            let tJsonParse = Self.debugPipeline ? Date() : Date.distantPast

            let containsMedia = Self.containsDeclaredMedia(chatRequest.messages)
            let mediaServing = containsMedia
                ? service as? any AFMMLXMediaRequestServing
                : nil
            if containsMedia {
                guard let mediaServing else {
                    throw MLXServiceError.unsupportedMediaInput(
                        model: modelID,
                        kind: "media"
                    )
                }
                try mediaServing.validateMediaRequestCapabilities(
                    model: modelID,
                    messages: chatRequest.messages
                )
            }

            let admissionTask = Task<(reserved: Bool, media: AFMMLXResolvedMediaRequest?), Error> {
                guard await service.waitForSlot(timeout: Self.slotQueueTimeout) else {
                    try Task.checkCancellation()
                    return (false, nil)
                }
                do {
                    let preflighted = try await mediaServing?.preflightMediaRequest(
                        model: modelID,
                        messages: chatRequest.messages
                    )
                    try Task.checkCancellation()
                    return (true, preflighted)
                } catch {
                    service.releaseSlot()
                    throw error
                }
            }
            cancelHandle.assign(admissionTask)
            let admission = try await withTaskCancellationHandler {
                try await admissionTask.value
            } onCancel: {
                cancelHandle.cancel()
            }

            guard admission.reserved else {
                if requestRegistered {
                    await inflightRegistry.release(id: reqId, registration: requestRegistration)
                    requestRegistered = false
                }
                let peer = req.peerAddress?.description ?? "unknown"
                let ua = req.headers.first(name: .userAgent) ?? "unknown"
                req.logger.warning("Connection refused: at capacity after \(Int(Self.slotQueueTimeout))s wait (\(service.maxConcurrent)/\(service.maxConcurrent)) — client=\(peer) ua=\(ua)")
                let response = Response(status: .serviceUnavailable)
                response.headers.add(name: .contentType, value: "application/json")
                response.headers.add(name: .accessControlAllowOrigin, value: "*")
                response.headers.add(name: "Retry-After", value: "2")
                try response.content.encode(OpenAIError(
                    message: "Server at capacity (\(service.maxConcurrent) concurrent requests). Please retry shortly.",
                    type: "server_busy"
                ))
                return response
            }
            let preflightedMedia = admission.media

            // Reset peak memory before each request so usage.peak_memory_gib
            // reflects this request only (matches mlx_lm's mx.reset_peak_memory())
            service.resetRequestPeakMemory()

            let isWebUI = req.headers.first(name: .origin) != nil
            let extractThinking = !rawOutput || isWebUI

            if chatRequest.stream == true && streamingEnabled {
                return try await createStreamingResponse(
                    req: req,
                    chatRequest: chatRequest,
                    preflightedMedia: preflightedMedia,
                    cancelHandle: cancelHandle,
                    extractThinking: extractThinking,
                    effectiveResponseFormat: effectiveResponseFormat,
                    grammarDowngraded: grammarDowngraded,
                    requestId: reqId,
                    requestRegistration: requestRegistration
                )
            }

            // The controller owns the reservation until generation either uses
            // the serial path or successfully returns a stream. A returned
            // stream releases its transferred reservation on termination.
            var reservationTransferredToStream = false
            defer {
                if !reservationTransferredToStream {
                    service.releaseSlot()
                }
            }

            // AFM Profile: start GPU monitoring if client requests it
            let profileHeader = req.headers.first(name: "X-AFM-Profile")?.lowercased()
            let wantProfile = profileHeader == "true" || profileHeader == "extended"
            let wantExtended = profileHeader == "extended"
            if wantProfile { service.startAPIProfile() }

            let effectiveTemp = chatRequest.temperature ?? temperature
            let effectiveTopP = chatRequest.topP ?? topP
            let effectiveMaxTokens = normalizedMaxTokens(chatRequest.effectiveMaxTokens)
            let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? repetitionPenalty
            let effectiveTopK = chatRequest.topK ?? topK
            let effectiveMinP = chatRequest.minP ?? minP
            let effectivePresencePenalty = chatRequest.presencePenalty ?? presencePenalty
            let effectiveSeed = chatRequest.seed ?? seed
            let effectiveStop = mergeStopSequences(cliStop: stop, apiStop: chatRequest.stop)
            if veryVerbose {
                let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                let stopDesc = effectiveStop.map { $0.map { $0.debugDescription }.joined(separator: ", ") }
                print(
                    "\(Self.orange)[\(Self.timestamp())] MLX start: stream=false\n  prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens)\n  temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none")\n  top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none")\n  seed=\(effectiveSeed?.description ?? "none") stop=\(stopDesc ?? "none")\(Self.reset)"
                ); fflush(stdout)
            }
            if Self.debugPipeline {
                let tSlotReserved = Date()
                let jsonMs = tJsonParse.timeIntervalSince(httpArrival) * 1000
                let slotMs = tSlotReserved.timeIntervalSince(tJsonParse) * 1000
                let totalMs = tSlotReserved.timeIntervalSince(httpArrival) * 1000
                print("[\(Self.timestamp())] [HTTPPipeline] req=\(reqId) json_parse=\(String(format: "%.1f", jsonMs))ms slot_wait=\(String(format: "%.1f", slotMs))ms total=\(String(format: "%.1f", totalMs))ms")
            }

            let result: AFMChatGenerationResult
            if service.maxConcurrent >= 2 {
                // Keep batch generation and stream collection under the same
                // cancellable task so the registry owns GPU work, not merely
                // the already-completed admission phase.
                reservationTransferredToStream = true
                let generationTask = Task<CancellableGenerationResult, Error> {
                    var streamOwnsReservation = false
                    defer {
                        if !streamOwnsReservation {
                            service.releaseSlot()
                        }
                    }
                    let streamResult: AFMChatStreamingResult = try await withPreflightedMedia(
                        preflightedMedia,
                        messages: chatRequest.messages
                    ) { trustedMessages in
                        try await service.generateStreaming(
                        model: modelID,
                        messages: trustedMessages,
                        temperature: effectiveTemp,
                        maxTokens: effectiveMaxTokens,
                        topP: effectiveTopP,
                        repetitionPenalty: effectiveRepetitionPenalty,
                        topK: effectiveTopK,
                        minP: effectiveMinP,
                        presencePenalty: effectivePresencePenalty,
                        seed: effectiveSeed,
                        logprobs: chatRequest.logprobs,
                        topLogprobs: chatRequest.topLogprobs,
                        tools: effectiveTools,
                        toolChoice: chatRequest.toolChoice,
                        parallelToolCalls: chatRequest.parallelToolCalls,
                        stop: effectiveStop,
                        responseFormat: effectiveResponseFormat,
                        chatTemplateKwargs: chatRequest.effectiveChatTemplateKwargs,
                        preserveStructuralTags: !extractThinking,
                        requestId: reqId
                        )
                    }
                    streamOwnsReservation = true

                // Collect stream into complete response
                var fullText = ""
                var allLogprobs: [AFMServerResolvedLogprob] = []
                var finalToolCalls: [ResponseToolCall]? = nil
                var promptTokens = streamResult.promptTokens
                var completionTokens = 0
                var cachedTokens = 0
                var promptTime: Double = 0
                var generateTime: Double = 0
                var stoppedBySequence = false

                for try await chunk in streamResult.stream {
                    fullText += chunk.text
                    if let lp = chunk.logprobs { allLogprobs.append(contentsOf: lp) }
                    if let tc = chunk.toolCalls { finalToolCalls = tc }
                    if let pt = chunk.promptTokens { promptTokens = pt }
                    if let ct = chunk.completionTokens { completionTokens = ct }
                    if let cached = chunk.cachedTokens { cachedTokens = cached }
                    if let pt = chunk.promptTime { promptTime = pt }
                    if let gt = chunk.generateTime { generateTime = gt }
                    if let sbs = chunk.stoppedBySequence { stoppedBySequence = sbs }
                }

                // FIX: Vendor ToolCallProcessor can append XML tag remnants to tool names
                // for zero-parameter calls (e.g. "todoread</function"). Strip them.
                // See: opencode promptfoo test #20/#33 — todoread</function bug.
                if let tcs = finalToolCalls {
                    finalToolCalls = tcs.map { tc in
                        var name = tc.function.name
                        if let tagIdx = name.range(of: "</") {
                            name = String(name[..<tagIdx.lowerBound])
                        }
                        guard name != tc.function.name else { return tc }
                        return ResponseToolCall(
                            index: tc.index, id: tc.id, type: tc.type,
                            function: ResponseToolCallFunction(name: name, arguments: tc.function.arguments)
                        )
                    }
                }

                // AFMKit owns model-specific parsing. The HTTP layer only serializes
                // the typed tool calls emitted by the provider.
                    return CancellableGenerationResult(
                        value: (
                            modelID: streamResult.modelID,
                            content: fullText,
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            tokenLogprobs: allLogprobs.isEmpty ? nil : allLogprobs,
                            toolCalls: finalToolCalls,
                            cachedTokens: cachedTokens,
                            promptTime: promptTime,
                            generateTime: generateTime,
                            stoppedBySequence: stoppedBySequence
                        )
                    )
                }
                cancelHandle.assign(generationTask)
                result = try await withTaskCancellationHandler {
                    try await generationTask.value.value
                } onCancel: {
                    cancelHandle.cancel()
                }
            } else {
                // Serial mode: use existing generate() path
                let generationTask = Task<AFMChatGenerationResult, Error> {
                    try await withPreflightedMedia(
                        preflightedMedia,
                        messages: chatRequest.messages
                    ) { trustedMessages in
                        try await service.generate(
                            model: modelID,
                            messages: trustedMessages,
                            temperature: effectiveTemp,
                            maxTokens: effectiveMaxTokens,
                            topP: effectiveTopP,
                            repetitionPenalty: effectiveRepetitionPenalty,
                            topK: effectiveTopK,
                            minP: effectiveMinP,
                            presencePenalty: effectivePresencePenalty,
                            seed: effectiveSeed,
                            logprobs: chatRequest.logprobs,
                            topLogprobs: chatRequest.topLogprobs,
                            tools: effectiveTools,
                            toolChoice: chatRequest.toolChoice,
                            parallelToolCalls: chatRequest.parallelToolCalls,
                            stop: effectiveStop,
                            responseFormat: effectiveResponseFormat,
                            chatTemplateKwargs: chatRequest.effectiveChatTemplateKwargs
                        )
                    }
                }
                cancelHandle.assign(generationTask)
                result = try await withTaskCancellationHandler {
                    try await generationTask.value
                } onCancel: {
                    cancelHandle.cancel()
                }
            }
            let completionTok = result.completionTokens
            let promptTime = result.promptTime
            let generateTime = result.generateTime
            let tokPerSec = generateTime > 0 ? Double(completionTok) / generateTime : 0
            let promptTokPerSec = promptTime > 0 ? Double(result.promptTokens) / promptTime : 0
            let structuralStripTags = extractThinking ? service.structuralStripTags : []
            let sanitizeContent: (String) -> String = {
                Self.sanitizeStructuredOutput(
                    self.sanitizeDegenerateTail(Self.stripStructuralTags($0, tags: structuralStripTags)),
                    responseFormat: effectiveResponseFormat
                )
            }
            let finalizedTurn = Self.finalizeAssistantTurn(
                content: result.content,
                toolCalls: result.toolCalls,
                toolChoice: chatRequest.toolChoice,
                parallelToolCalls: chatRequest.parallelToolCalls,
                extractThinking: extractThinking,
                thinkStartTag: service.thinkStartTag ?? "<think>",
                thinkEndTag: service.thinkEndTag ?? "</think>",
                stoppedBySequence: result.stoppedBySequence,
                completionTokens: completionTok,
                maxTokens: effectiveMaxTokens,
                sanitizeContent: sanitizeContent,
                responseChannelFormat: service.responseChannelFormat,
                stopSequences: effectiveStop
                )

            // If we got tool calls, return a tool_calls response
            if let toolCalls = finalizedTurn.toolCalls, !toolCalls.isEmpty {
                if veryVerbose {
                    print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  prompt=\(String(format: "%.2f", promptTime))s gen=\(String(format: "%.2f", generateTime))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)"); fflush(stdout)
                    for tc in toolCalls {
                        print("\(Self.gold)[\(Self.timestamp())] SEND tool_call: \(tc.function.name)\n  id=\(tc.id)\n  args=\(tc.function.arguments)\(Self.reset)")
                    }
                    fflush(stdout)
                }

                let choiceLogprobs = Self.buildChoiceLogprobs(result.tokenLogprobs)
                let timings = StreamTimings(prompt_n: result.promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTok, predicted_ms: generateTime * 1000)
                let extended = wantExtended ? service.stopAPIProfileExtended(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil
                let profile = wantExtended ? nil : (wantProfile ? service.stopAPIProfile(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil)
                let response = ChatCompletionResponse(
                    model: result.modelID,
                    toolCalls: toolCalls,
                    logprobs: choiceLogprobs,
                    promptTokens: result.promptTokens,
                    completionTokens: completionTok,
                    cachedTokens: result.cachedTokens,
                    completionTime: generateTime,
                    promptTime: promptTime,
                    peakMemoryGib: service.currentRequestPeakMemoryGib(),
                    timings: timings,
                    afmProfile: profile,
                    afmProfileExtended: extended
                )
                let cacheInfo1 = Self.cacheStatsSummary(
                    cachedTokens: result.cachedTokens,
                    totalPromptTokens: result.promptTokens
                )
                print("\(Self.orange)[\(Self.timestamp())] [STATS] pp: \(result.promptTokens) tok, \(String(format: "%.2f", promptTime))s (\(String(format: "%.1f", promptTokPerSec)) tok/s) | tg: \(completionTok) tok, \(String(format: "%.2f", generateTime))s (\(String(format: "%.1f", tokPerSec)) tok/s)\(cacheInfo1) stream=false\(Self.reset)")
                let tcSummary = toolCalls.map { "\($0.function.name)(\(Self.argKeysPreview($0.function.arguments)))" }.joined(separator: ", ")
                print("\(Self.gold)[\(Self.timestamp())] [TOOL_CALLS] \(toolCalls.count) call(s): \(tcSummary)\(Self.reset)")
                fflush(stdout)
                if veryVerbose {
                    print("\(Self.teal)[\(Self.timestamp())] SEND full response:\n\(encodeJSON(response))\(Self.reset)"); fflush(stdout)
                }
                // -VV: Non-streaming tool call details
                if trace {
                    for tc in toolCalls {
                        print("\(Self.cyan)[\(Self.timestamp())] [VV] SEND→CLIENT (non-stream) tool_call \(tc.function.name):\n  \(tc.function.arguments)\(Self.reset)")
                    }
                    fflush(stdout)
                }
                if requestRegistered {
                    await inflightRegistry.release(id: reqId, registration: requestRegistration)
                    requestRegistered = false
                }
                return try await createSuccessResponse(req: req, response: response, grammarDowngraded: grammarDowngraded)
            }

            let stopReason = finalizedTurn.finishReason
            if veryVerbose {
                print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=false\n  prompt_tokens=\(result.promptTokens) completion_tokens=\(completionTok)\n  prompt=\(String(format: "%.2f", promptTime))s gen=\(String(format: "%.2f", generateTime))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(stopReason)\(Self.reset)"); fflush(stdout)
            }

            let choiceLogprobs = Self.buildChoiceLogprobs(result.tokenLogprobs)
            let timings = StreamTimings(prompt_n: result.promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTok, predicted_ms: generateTime * 1000)
            let extended = wantExtended ? service.stopAPIProfileExtended(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil
            let profile = wantExtended ? nil : (wantProfile ? service.stopAPIProfile(promptTokens: result.promptTokens, completionTokens: completionTok, promptTime: promptTime, generateTime: generateTime) : nil)
            let response = ChatCompletionResponse(
                model: result.modelID,
                content: finalizedTurn.content ?? "",
                reasoningContent: finalizedTurn.reasoningContent,
                logprobs: choiceLogprobs,
                finishReason: stopReason,
                promptTokens: result.promptTokens,
                completionTokens: completionTok,
                cachedTokens: result.cachedTokens,
                completionTime: generateTime,
                promptTime: promptTime,
                peakMemoryGib: service.currentRequestPeakMemoryGib(),
                timings: timings,
                afmProfile: profile,
                afmProfileExtended: extended
            )
            let cacheInfo2 = Self.cacheStatsSummary(
                cachedTokens: result.cachedTokens,
                totalPromptTokens: result.promptTokens
            )
            print("\(Self.orange)[\(Self.timestamp())] [STATS] pp: \(result.promptTokens) tok, \(String(format: "%.2f", promptTime))s (\(String(format: "%.1f", promptTokPerSec)) tok/s) | tg: \(completionTok) tok, \(String(format: "%.2f", generateTime))s (\(String(format: "%.1f", tokPerSec)) tok/s)\(cacheInfo2) stream=false\(Self.reset)")
            fflush(stdout)
            if veryVerbose {
                print("\(Self.teal)[\(Self.timestamp())] SEND full response:\n\(encodeJSON(response))\(Self.reset)"); fflush(stdout)
            }
            if requestRegistered {
                await inflightRegistry.release(id: reqId, registration: requestRegistration)
                requestRegistered = false
            }
            return try await createSuccessResponse(req: req, response: response, grammarDowngraded: grammarDowngraded)
        } catch let serviceError as MLXServiceError {
            if requestRegistered {
                await inflightRegistry.release(id: reqId, registration: requestRegistration)
            }
            let code: String
            switch serviceError {
            case .visionAssetsUnavailable:
                code = "vision_assets_unavailable"
            case .unsupportedMediaInput:
                code = "unsupported_media_input"
            case .invalidMediaInput:
                code = "invalid_media_input"
            default:
                code = "mlx_error"
            }
            req.logger.error("[\(Self.timestamp())] MLX request error: \(serviceError.localizedDescription)")
            return try await createErrorResponse(
                req: req,
                error: OpenAIError(
                    message: serviceError.localizedDescription,
                    type: code == "mlx_error" ? "mlx_error" : "invalid_request_error",
                    code: code,
                    requestId: reqId.isEmpty ? nil : reqId
                ),
                status: .badRequest
            )
        } catch let abort as Abort {
            if requestRegistered {
                await inflightRegistry.release(id: reqId, registration: requestRegistration)
            }
            req.logger.error("[\(Self.timestamp())] MLX completions error: \(abort)")
            return try await createErrorResponse(
                req: req,
                error: OpenAIError(
                    message: abort.reason,
                    type: abort.status == .badRequest ? "invalid_request_error" : "mlx_error"
                ),
                status: abort.status
            )
        } catch {
            if requestRegistered {
                await inflightRegistry.release(id: reqId, registration: requestRegistration)
            }
            req.logger.error("[\(Self.timestamp())] MLX completions error: \(error)")
            return try await createErrorResponse(req: req, error: OpenAIError(message: error.localizedDescription, type: "mlx_error"), status: .badRequest)
        }
    }

    private func withPreflightedMedia<Result: Sendable>(
        _ request: AFMMLXResolvedMediaRequest?,
        messages: [Message],
        operation: ([Message]) async throws -> Result
    ) async throws -> Result {
        guard let request else {
            return try await operation(messages)
        }
        guard let mediaServing = service as? any AFMMLXMediaRequestServing else {
            throw MLXServiceError.unsupportedMediaInput(model: modelID, kind: "media")
        }
        return try await mediaServing.withPreflightedMediaRequest(
            request,
            operation: operation
        )
    }

    private func createStreamingResponse(
        req: Request,
        chatRequest: ChatCompletionRequest,
        preflightedMedia: AFMMLXResolvedMediaRequest?,
        cancelHandle: CancellableTaskHandle,
        extractThinking: Bool,
        effectiveResponseFormat: ResponseFormat?,
        grammarDowngraded: Bool = false,
        requestId: String = "",
        requestRegistration: InflightRequestRegistry.Registration? = nil
    ) async throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "text/event-stream")
        httpResponse.headers.add(name: .cacheControl, value: "no-cache")
        httpResponse.headers.add(name: .connection, value: "keep-alive")
        httpResponse.headers.add(name: "Access-Control-Allow-Origin", value: "*")
        httpResponse.headers.add(name: "Access-Control-Allow-Headers", value: "Content-Type, X-AFM-Profile")
        httpResponse.headers.add(name: "X-Accel-Buffering", value: "no")
        if grammarDowngraded {
            httpResponse.headers.add(name: "X-Grammar-Constraints", value: "downgraded")
        }

        let streamId = UUID().uuidString

        // AFM Profile: start GPU monitoring if client requests it
        let streamProfileHeader = req.headers.first(name: "X-AFM-Profile")?.lowercased()
        let wantStreamProfile = streamProfileHeader == "true" || streamProfileHeader == "extended"
        let wantStreamExtended = streamProfileHeader == "extended"
        if wantStreamProfile { service.startAPIProfile() }

        let streamReqId = requestId
        // Capture the registry on the request thread so the asyncStream closure
        // can register a cancel hook without re-resolving it.
        let inflightRegistry = req.application.inflightRegistry
        // Register the cancel hook BEFORE the asyncStream closure spawns the
        // body Task. Eliminates the race where a cancel arrives between the
        // Response being returned and the closure firing. (T1.4/T1.5 fix)
        httpResponse.body = .init(asyncStream: { writer in
            // T1.4/T1.5: Wrap the streaming body in an explicit Task so we can
            // cancel it from outside (cancel endpoint, client disconnect detection).
            // Cooperative cancellation propagates through the AsyncThrowingStream
            // iterator (`res.stream` below) — when the body Task is cancelled,
            // its `next()` throws `CancellationError`, the iterator deinits, and
            // its `onTermination` fires `task.cancel()` on the underlying model
            // generator (BatchScheduler / MLX serial path), stopping GPU work.
            let bodyTask = Task<Void, Never> {
            var reservationTransferredToStream = false
            defer {
                if !reservationTransferredToStream {
                    self.service.releaseSlot()
                }
            }
            // PR #122: Streaming routes account for their own
            // afm:num_active_connections — ActiveConnectionsMiddleware filters
            // them because its defer fires when the controller returns, not
            // when the SSE body finishes.
            ActiveConnectionTracker.shared.connectionStarted()
            defer { ActiveConnectionTracker.shared.connectionEnded() }
            let encoder = JSONEncoder()
            var fullContent = ""
            let started = Date()
            let effectiveTemp = chatRequest.temperature ?? self.temperature
            let effectiveTopP = chatRequest.topP ?? self.topP
            let effectiveMaxTokens = self.normalizedMaxTokens(chatRequest.effectiveMaxTokens)
            let effectiveRepetitionPenalty = chatRequest.effectiveRepetitionPenalty ?? self.repetitionPenalty
            let effectiveTopK = chatRequest.topK ?? self.topK
            let effectiveMinP = chatRequest.minP ?? self.minP
            let effectivePresencePenalty = chatRequest.presencePenalty ?? self.presencePenalty
            let effectiveSeed = chatRequest.seed ?? self.seed
            let effectiveStop = self.mergeStopSequences(cliStop: self.stop, apiStop: chatRequest.stop)
            let deferStructuredOutputContent = Self.requiresStructuredOutputSanitization(effectiveResponseFormat)
            let structuralStripTags = extractThinking ? service.structuralStripTags : []
            let sanitizeContent: (String) -> String = {
                Self.sanitizeStructuredOutput(
                    self.sanitizeDegenerateTail(Self.stripStructuralTags($0, tags: structuralStripTags)),
                    responseFormat: effectiveResponseFormat
                )
            }

            do {
                let effectiveTools = try Self.resolveEffectiveTools(
                    chatRequest.tools,
                    toolChoice: chatRequest.toolChoice
                )
                if self.veryVerbose {
                    let promptChars = chatRequest.messages.map { $0.textContent.count }.reduce(0, +)
                    let stopDesc = effectiveStop.map { $0.map { $0.debugDescription }.joined(separator: ", ") }
                    print(
                        "\(Self.orange)[\(Self.timestamp())] MLX start: stream=true\n  prompt_chars=\(promptChars) max_tokens=\(effectiveMaxTokens)\n  temperature=\(effectiveTemp?.description ?? "default") top_p=\(effectiveTopP?.description ?? "default") rep_penalty=\(effectiveRepetitionPenalty?.description ?? "none")\n  top_k=\(effectiveTopK?.description ?? "none") min_p=\(effectiveMinP?.description ?? "none") presence_penalty=\(effectivePresencePenalty?.description ?? "none")\n  seed=\(effectiveSeed?.description ?? "none") stop=\(stopDesc ?? "none")\(Self.reset)"
                    ); fflush(stdout)
                }
                let res = try await self.withPreflightedMedia(
                    preflightedMedia,
                    messages: chatRequest.messages
                ) { trustedMessages in
                    try await service.generateStreaming(
                        model: modelID,
                        messages: trustedMessages,
                        temperature: effectiveTemp,
                        maxTokens: effectiveMaxTokens,
                        topP: effectiveTopP,
                        repetitionPenalty: effectiveRepetitionPenalty,
                        topK: effectiveTopK,
                        minP: effectiveMinP,
                        presencePenalty: effectivePresencePenalty,
                        seed: effectiveSeed,
                        logprobs: chatRequest.logprobs,
                        topLogprobs: chatRequest.topLogprobs,
                        tools: effectiveTools,
                        toolChoice: chatRequest.toolChoice,
                        parallelToolCalls: chatRequest.parallelToolCalls,
                        stop: effectiveStop,
                        responseFormat: effectiveResponseFormat,
                        chatTemplateKwargs: chatRequest.effectiveChatTemplateKwargs,
                        preserveStructuralTags: !extractThinking,
                        requestId: streamReqId
                    )
                }
                reservationTransferredToStream = true
                // Emit an initial assistant delta so clients always open a response container.
                let initialChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFirst: true
                )
                let initialData = try encoder.encode(initialChunk)
                if let jsonString = String(data: initialData, encoding: .utf8) {
                    try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }

                // State for <think> tag extraction (Qwen, DeepSeek R1, etc.)
                var insideThinkBlock = false
                var thinkBuffer = ""
                // State for response-channel parsing (gpt-oss Harmony or Muse). Mutually exclusive with thinkBuffer.
                let responseChannelFormat = self.service.responseChannelFormat
                let harmonyChannels = responseChannelFormat == .harmony
                let museChannels = responseChannelFormat == .muse
                var harmonyState = HarmonyState()
                var harmonyBuffer = ""
                var museState = MuseResponseChannelState()
                var museBuffer = ""
                var verboseReasoningBuf = ""
                var verboseContentBuf = ""
                var logprobBuffer = [AFMServerResolvedLogprob]()
                var collectedToolCalls = [ResponseToolCall]()
                var hasToolCalls = false
                let allowedToolName: String? = {
                    if case .function(let functionChoice) = chatRequest.toolChoice {
                        return functionChoice.function.name
                    }
                    return nil
                }()
                var permittedToolIndices = Set<Int>()
                // The batch scheduler emits argument deltas followed by a completed
                // call for final-state collection. Do not put those arguments on the
                // wire twice: OpenAI clients concatenate argument delta strings.
                var streamedVendorToolIndices = Set<Int>()
                var emittedToolCallDeltaIndices = Set<Int>()
                var streamedToolCallNames = [Int: String]()
                var streamedToolCallIDs = [Int: String]()
                var streamedToolCallTypes = [Int: String]()
                var streamedToolArgumentBuffers = [Int: String]()
                var completedStreamedToolArgumentIndices = Set<Int>()
                var suppressRestartedToolArgumentIndices = Set<Int>()
                func shouldEmitToolDelta(_ delta: StreamDeltaToolCall) -> Bool {
                    guard let arguments = delta.function?.arguments, !arguments.isEmpty else {
                        return true
                    }

                    let index = delta.index
                    let existing = streamedToolArgumentBuffers[index] ?? ""
                    let trimmed = arguments.trimmingCharacters(in: .whitespacesAndNewlines)

                    if completedStreamedToolArgumentIndices.contains(index) {
                        return false
                    }

                    if suppressRestartedToolArgumentIndices.contains(index) {
                        let candidate = existing + arguments
                        if Self.isCompleteJSONToolArguments(candidate) {
                            streamedToolArgumentBuffers[index] = candidate
                            completedStreamedToolArgumentIndices.insert(index)
                            suppressRestartedToolArgumentIndices.remove(index)
                            return true
                        }
                        return false
                    }

                    // Some models emit a valid incremental XML/function stream and then
                    // restart the same arguments as an inline JSON object. OpenAI clients
                    // concatenate argument fragments, so a same-index restart corrupts
                    // the stream. Treat `{` after prior argument bytes as a restart,
                    // not as a legal continuation.
                    if !existing.isEmpty && trimmed.hasPrefix("{") {
                        suppressRestartedToolArgumentIndices.insert(index)
                        return false
                    }

                    let candidate = existing + arguments
                    streamedToolArgumentBuffers[index] = candidate
                    if Self.isCompleteJSONToolArguments(candidate) {
                        completedStreamedToolArgumentIndices.insert(index)
                    }
                    return true
                }
                var stoppedBySequence = false
                var streamingStopFilter = StreamingStopSequenceFilter(stopSequences: effectiveStop)
                var providerStopReached = false
                var realPromptTokens: Int? = nil
                var realCompletionTokens: Int? = nil
                var realCachedTokens: Int? = nil
                var realPromptTime: Double? = nil
                var realGenerateTime: Double? = nil

                let thinkStartTag = res.thinkStartTag
                let thinkEndTag = res.thinkEndTag

                var pendingRawTag: String? = nil
                for try await streamChunk in res.stream {
                    let piece = streamChunk.text
                    let suppressPayload = providerStopReached || streamingStopFilter.stopped
                    let providerReportedStop = streamChunk.stoppedBySequence == true
                    providerStopReached = providerStopReached || providerReportedStop

                    // Capture real token counts and timing from the info chunk
                    if let pt = streamChunk.promptTokens { realPromptTokens = pt }
                    if let ct = streamChunk.completionTokens { realCompletionTokens = ct }
                    if let cached = streamChunk.cachedTokens { realCachedTokens = cached }
                    if streamChunk.stoppedBySequence == true { stoppedBySequence = true }
                    if let pt = streamChunk.promptTime { realPromptTime = pt }
                    if let gt = streamChunk.generateTime { realGenerateTime = gt }

                    // Once either the provider or the visible-content filter has
                    // stopped, drain only telemetry from subsequent chunks.
                    guard !suppressPayload else { continue }

                    var stopPreview = streamingStopFilter
                    if !extractThinking { _ = stopPreview.consume(piece) }
                    let allowCurrentSemanticPayload = !providerReportedStop && !stopPreview.stopped

                    if allowCurrentSemanticPayload,
                       let deltas = streamChunk.toolCallDeltas, !deltas.isEmpty {
                        let filtered = deltas.filter {
                            Self.isToolDeltaAllowed(
                                $0,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            )
                        }
                        var emitDeltas = [StreamDeltaToolCall]()
                        for delta in filtered where shouldEmitToolDelta(delta) {
                            if let name = delta.function?.name { streamedToolCallNames[delta.index] = name }
                            if let id = delta.id { streamedToolCallIDs[delta.index] = id }
                            if let type = delta.type { streamedToolCallTypes[delta.index] = type }
                            emitDeltas.append(delta)
                        }
                        guard !emitDeltas.isEmpty else { continue }
                        streamedVendorToolIndices.formUnion(emitDeltas.map(\.index))
                        emittedToolCallDeltaIndices.formUnion(emitDeltas.map(\.index))
                        hasToolCalls = true
                        let tcChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            toolCalls: emitDeltas
                        )
                        let tcData = try encoder.encode(tcChunk)
                        if let jsonString = String(data: tcData, encoding: .utf8) {
                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                        continue
                    }

                    // Handle tool call chunks from the vendor parser
                    if allowCurrentSemanticPayload,
                       let tcs = streamChunk.toolCalls, !tcs.isEmpty {
                        for tc in tcs {
                            // Coerce argument types before emitting (Gemma 4 escape markers, etc.)
                            let coercedToolCall = service.coerceToolCallArguments(tc, tools: effectiveTools)
                            guard Self.isToolCallAllowed(
                                coercedToolCall,
                                toolChoice: chatRequest.toolChoice,
                                allowedFunctionName: allowedToolName,
                                permittedToolIndices: &permittedToolIndices
                            ) else { continue }
                            hasToolCalls = true
                            let toolIndex = coercedToolCall.index ?? streamedVendorToolIndices.min() ?? collectedToolCalls.count
                            if toolIndex < collectedToolCalls.count {
                                collectedToolCalls[toolIndex] = coercedToolCall
                            } else {
                                collectedToolCalls.append(coercedToolCall)
                            }
                            if streamedVendorToolIndices.contains(toolIndex) {
                                continue
                            }
                            if self.veryVerbose {
                                print("\(Self.gold)[\(Self.timestamp())] SEND tool_call (vendor): \(coercedToolCall.function.name)\n  id=\(coercedToolCall.id)\n  args=\(coercedToolCall.function.arguments)\(Self.reset)")
                                fflush(stdout)
                            }
                            let delta = StreamDeltaToolCall(
                                index: toolIndex,
                                id: coercedToolCall.id,
                                type: coercedToolCall.type,
                                function: StreamDeltaFunction(
                                    name: coercedToolCall.function.name,
                                    arguments: coercedToolCall.function.arguments
                                )
                            )
                            let tcChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                toolCalls: [delta]
                            )
                            let tcData = try encoder.encode(tcChunk)
                            if let jsonString = String(data: tcData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                        continue
                    }

                    fullContent += piece

                    if allowCurrentSemanticPayload, let lps = streamChunk.logprobs {
                        logprobBuffer.append(contentsOf: lps)
                    }

                    // Detect RAW think tags but defer logging until after extraction flush
                    let tst = thinkStartTag ?? "<think>"
                    let tet = thinkEndTag ?? "</think>"
                    if self.veryVerbose && (piece.contains(tst) || piece.contains(tet)) {
                        pendingRawTag = piece.debugDescription
                    }

                    if extractThinking && harmonyChannels {
                        harmonyBuffer += piece
                        let extracted = Self.extractHarmonyChannels(
                            buffer: &harmonyBuffer,
                            state: &harmonyState
                        )
                        var emitContent = extracted.content
                        let emitReasoning = extracted.reasoning
                        if let content = emitContent {
                            emitContent = streamingStopFilter.consume(content)
                            if emitContent != content { logprobBuffer = [] }
                            if streamingStopFilter.stopped { stoppedBySequence = true }
                        }
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        let hasReasoning = emitReasoning != nil
                        let hasContent = emitContent != nil
                        if !deferStructuredOutputContent && (hasReasoning || hasContent || flushLogprobs != nil) {
                            if self.veryVerbose {
                                if let r = emitReasoning { verboseReasoningBuf += r }
                                if let c = emitContent { verboseContentBuf += c }
                                if verboseReasoningBuf.hasSuffix("\n") || verboseReasoningBuf.count > 200 {
                                    print("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)"); fflush(stdout)
                                    verboseReasoningBuf = ""
                                }
                                if verboseContentBuf.hasSuffix("\n") || verboseContentBuf.count > 200 {
                                    print("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)"); fflush(stdout)
                                    verboseContentBuf = ""
                                }
                            }
                            logprobBuffer = []
                            let contentChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                content: emitContent ?? "",
                                reasoningContent: emitReasoning,
                                logprobs: flushLogprobs,
                                isFirst: false
                            )
                            let chunkData = try encoder.encode(contentChunk)
                            if let jsonString = String(data: chunkData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                        if harmonyState.stopReached {
                            stoppedBySequence = true
                            break
                        }
                    } else if extractThinking && museChannels {
                        museBuffer += piece
                        let extracted = Self.extractMuseResponseChannels(
                            buffer: &museBuffer,
                            state: &museState
                        )

                        var emitContent = extracted.content
                        let emitReasoning = extracted.reasoning
                        if let content = emitContent {
                            emitContent = streamingStopFilter.consume(content)
                            if emitContent != content { logprobBuffer = [] }
                            if streamingStopFilter.stopped { stoppedBySequence = true }
                        }
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        let hasReasoning = emitReasoning != nil
                        let hasContent = emitContent != nil
                        if !deferStructuredOutputContent && (hasReasoning || hasContent || flushLogprobs != nil) {
                            if self.veryVerbose {
                                if let r = emitReasoning { verboseReasoningBuf += r }
                                if let c = emitContent { verboseContentBuf += c }
                                if verboseReasoningBuf.hasSuffix("\n") || verboseReasoningBuf.count > 200 {
                                    print("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)"); fflush(stdout)
                                    verboseReasoningBuf = ""
                                }
                                if verboseContentBuf.hasSuffix("\n") || verboseContentBuf.count > 200 {
                                    print("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)"); fflush(stdout)
                                    verboseContentBuf = ""
                                }
                            }
                            logprobBuffer = []
                            let contentChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                content: emitContent ?? "",
                                reasoningContent: emitReasoning,
                                logprobs: flushLogprobs,
                                isFirst: false
                            )
                            let chunkData = try encoder.encode(contentChunk)
                            if let jsonString = String(data: chunkData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                        if museState.stopReached {
                            stoppedBySequence = true
                            break
                        }
                    } else if extractThinking {
                        // Drop structural wrapper tokens (e.g. Cohere <|START_TEXT|>) before
                        // buffering. Safe per-piece: special tokens detokenize atomically,
                        // the same assumption the start-tag comparison below relies on. (#148)
                        let piece = Self.stripStructuralTags(piece, tags: structuralStripTags)
                        // If the piece is exactly the think start tag (template-injected or
                        // model-generated), just flip the state without adding the literal
                        // tag to the buffer. Prevents double-tag leaks when the template
                        // injects a think tag and the model also generates one.
                        let trimmedPiece = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                        if let tst = thinkStartTag, trimmedPiece == tst && !insideThinkBlock {
                            insideThinkBlock = true
                        } else if !piece.isEmpty {
                            thinkBuffer += piece
                        }

                        let extracted = Self.extractThinkTags(
                            buffer: &thinkBuffer,
                            insideThinkBlock: &insideThinkBlock,
                            startTag: thinkStartTag ?? "<think>",
                            endTag: thinkEndTag ?? "</think>"
                        )

                        var emitContent = extracted.content
                        let emitReasoning = extracted.reasoning
                        if let content = emitContent {
                            emitContent = streamingStopFilter.consume(content)
                            if emitContent != content { logprobBuffer = [] }
                            if streamingStopFilter.stopped { stoppedBySequence = true }
                        }

                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)

                        // Emit a chunk whenever we have visible content, extracted reasoning,
                        // or buffered logprobs. Without this, per-token logprobs can be lost
                        // when detokenized content is still buffered for think-tag extraction.
                        let hasReasoning = emitReasoning != nil
                        let hasContent = emitContent != nil
                        if !deferStructuredOutputContent && (hasReasoning || hasContent || flushLogprobs != nil) {
                            if self.veryVerbose {
                                if let r = emitReasoning { verboseReasoningBuf += r }
                                if let c = emitContent { verboseContentBuf += c }
                                // Flush verbose log on newlines or when buffer gets large
                                if verboseReasoningBuf.hasSuffix("\n") || verboseReasoningBuf.count > 200 {
                                    print("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)"); fflush(stdout)
                                    verboseReasoningBuf = ""
                                }
                                if verboseContentBuf.hasSuffix("\n") || verboseContentBuf.count > 200 {
                                    print("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)"); fflush(stdout)
                                    verboseContentBuf = ""
                                }
                            }
                            logprobBuffer = []
                            let contentChunk = ChatCompletionStreamResponse(
                                id: streamId,
                                model: res.modelID,
                                content: emitContent ?? "",
                                reasoningContent: emitReasoning,
                                logprobs: flushLogprobs,
                                isFirst: false
                            )
                            let chunkData = try encoder.encode(contentChunk)
                            if let jsonString = String(data: chunkData, encoding: .utf8) {
                                try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                            }
                        }
                        // Log RAW think tag AFTER extracted reasoning/content is flushed
                        if let tag = pendingRawTag {
                            print("\(Self.purple)[\(Self.timestamp())] MLX RAW token: \(tag)\(Self.reset)")
                            fflush(stdout)
                            pendingRawTag = nil
                        }
                    } else {
                        if deferStructuredOutputContent {
                            continue
                        }
                        let visiblePiece = streamingStopFilter.consume(piece)
                        if streamingStopFilter.stopped { stoppedBySequence = true }
                        let flushLogprobs = visiblePiece == piece && !providerReportedStop && !logprobBuffer.isEmpty
                            ? Self.buildChoiceLogprobs(logprobBuffer)
                            : nil
                        logprobBuffer = []
                        let contentChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: visiblePiece,
                            logprobs: flushLogprobs,
                            isFirst: false
                        )
                        let chunkData = try encoder.encode(contentChunk)
                        if let jsonString = String(data: chunkData, encoding: .utf8) {
                            try await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }

                if !emittedToolCallDeltaIndices.isEmpty { hasToolCalls = true }

                if hasToolCalls && collectedToolCalls.isEmpty && !emittedToolCallDeltaIndices.isEmpty {
                    for index in emittedToolCallDeltaIndices.sorted() {
                        guard let name = streamedToolCallNames[index] else { continue }
                        collectedToolCalls.append(ResponseToolCall(
                            index: index,
                            id: streamedToolCallIDs[index] ?? "call_\(index)",
                            type: streamedToolCallTypes[index] ?? "function",
                            function: ResponseToolCallFunction(
                                name: name,
                                arguments: streamedToolArgumentBuffers[index] ?? ""
                            )
                        ))
                    }
                }

                // === LOG IMMEDIATELY after generation, BEFORE any writer.write() calls ===
                if self.veryVerbose {
                    if !verboseReasoningBuf.isEmpty { print("\(Self.purple)[\(Self.timestamp())] SEND reasoning:\n  \(verboseReasoningBuf)\(Self.reset)") }
                    if !verboseContentBuf.isEmpty { print("\(Self.teal)[\(Self.timestamp())] SEND content (chunk):\n  \(verboseContentBuf)\(Self.reset)") }
                }
                let promptTokens = realPromptTokens ?? res.promptTokens
                let completionTokens = realCompletionTokens ?? self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
                let tokPerSec = generationDuration > 0 ? Double(completionTokens) / generationDuration : 0
                let finalizedTurn = Self.finalizeAssistantTurn(
                    content: fullContent,
                    toolCalls: hasToolCalls ? collectedToolCalls : nil,
                    toolChoice: chatRequest.toolChoice,
                    parallelToolCalls: chatRequest.parallelToolCalls,
                    extractThinking: extractThinking,
                    thinkStartTag: thinkStartTag ?? "<think>",
                    thinkEndTag: thinkEndTag ?? "</think>",
                    stoppedBySequence: stoppedBySequence,
                    completionTokens: completionTokens,
                    maxTokens: effectiveMaxTokens,
                    sanitizeContent: sanitizeContent,
                    responseChannelFormat: responseChannelFormat,
                    stopSequences: effectiveStop
                )
                let finishReason = finalizedTurn.finishReason
                if self.veryVerbose {
                    if finishReason == "tool_calls" {
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  prompt_tokens=\(promptTokens) completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=tool_calls\(Self.reset)")
                    } else {
                        let trimmedAnswer = (finalizedTurn.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmedAnswer.isEmpty {
                            print("\(Self.teal)[\(Self.timestamp())] MLX full answer:\n  \(trimmedAnswer)\(Self.reset)")
                        }
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  prompt_tokens=\(promptTokens) completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  finish_reason=\(finishReason)\(Self.reset)")
                    }
                }
                let sPromptTime = realPromptTime ?? 0
                let sPromptTokPerSec = sPromptTime > 0 ? Double(promptTokens) / sPromptTime : 0
                let sGenTime = realGenerateTime ?? generationDuration
                let sGenTokPerSec = sGenTime > 0 ? Double(completionTokens) / sGenTime : 0
                let sCached = realCachedTokens ?? 0
                let sCacheInfo = Self.cacheStatsSummary(
                    cachedTokens: sCached,
                    totalPromptTokens: promptTokens
                )
                print("\(Self.orange)[\(Self.timestamp())] [STATS] pp: \(promptTokens) tok, \(String(format: "%.2f", sPromptTime))s (\(String(format: "%.1f", sPromptTokPerSec)) tok/s) | tg: \(completionTokens) tok, \(String(format: "%.2f", sGenTime))s (\(String(format: "%.1f", sGenTokPerSec)) tok/s)\(sCacheInfo) stream=true\(Self.reset)")
                if let finalToolCalls = finalizedTurn.toolCalls, !finalToolCalls.isEmpty {
                    let tcSummary = finalToolCalls.map { "\($0.function.name)(\(Self.argKeysPreview($0.function.arguments)))" }.joined(separator: ", ")
                    print("\(Self.gold)[\(Self.timestamp())] [TOOL_CALLS] \(finalToolCalls.count) call(s): \(tcSummary)\(Self.reset)")
                }
                if self.veryVerbose {
                    let usageLog = StreamUsage(promptTokens: promptTokens, completionTokens: completionTokens, completionTime: generationDuration, promptTime: 0)
                    print("\(Self.teal)[\(Self.timestamp())] SEND usage:\n  \(self.encodeJSON(usageLog))\(Self.reset)")
                }
                fflush(stdout)

                // === Now flush remaining buffer to client (writer calls may hang/throw) ===
                // Flush remaining harmony buffer (#121). When the stream ends mid-channel
                // without a closing <|end|>/<|return|>, emit whatever was being accumulated.
                if !deferStructuredOutputContent && extractThinking && harmonyChannels && !harmonyBuffer.isEmpty {
                    var remaining: String?
                    let remainingReasoning: String?
                    switch harmonyState.channel {
                    case .analysis:
                        remainingReasoning = harmonyBuffer
                        remaining = nil
                    case .final:
                        remaining = harmonyBuffer
                        remainingReasoning = nil
                    default:
                        remaining = nil
                        remainingReasoning = nil
                    }
                    if let content = remaining {
                        remaining = streamingStopFilter.consume(content)
                        if remaining != content { logprobBuffer = [] }
                        if streamingStopFilter.stopped { stoppedBySequence = true }
                    }
                    if remaining != nil || remainingReasoning != nil {
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        logprobBuffer = []
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: remaining ?? "",
                            reasoningContent: remainingReasoning,
                            logprobs: flushLogprobs,
                            isFirst: false
                        )
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }
                // Flush remaining Muse response-template channel content.
                if !deferStructuredOutputContent && extractThinking && museChannels && !museBuffer.isEmpty {
                    let remaining = Self.flushMuseResponseChannelRemainder(
                        buffer: &museBuffer,
                        state: &museState
                    )
                    var remainingContent = remaining.content
                    if let content = remainingContent {
                        remainingContent = streamingStopFilter.consume(content)
                        if remainingContent != content { logprobBuffer = [] }
                        if streamingStopFilter.stopped { stoppedBySequence = true }
                    }
                    if remainingContent != nil || remaining.reasoning != nil {
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        logprobBuffer = []
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: remainingContent ?? "",
                            reasoningContent: remaining.reasoning,
                            logprobs: flushLogprobs,
                            isFirst: false
                        )
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }
                // Flush remaining thinkBuffer content (tool call tags are handled
                // above and never enter the thinkBuffer, so this is safe).
                if !deferStructuredOutputContent && extractThinking && responseChannelFormat == .none && !thinkBuffer.isEmpty {
                    var remaining: String?
                    let remainingReasoning: String?
                    if insideThinkBlock {
                        remainingReasoning = thinkBuffer
                        remaining = nil
                    } else {
                        remaining = thinkBuffer
                        remainingReasoning = nil
                    }
                    if let content = remaining {
                        remaining = streamingStopFilter.consume(content)
                        if remaining != content { logprobBuffer = [] }
                        if streamingStopFilter.stopped { stoppedBySequence = true }
                    }
                    if remaining != nil || remainingReasoning != nil {
                        let flushLogprobs = logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer)
                        logprobBuffer = []
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: remaining ?? "",
                            reasoningContent: remainingReasoning,
                            logprobs: flushLogprobs,
                            isFirst: false
                        )
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    } else if !logprobBuffer.isEmpty {
                        let flushChunk = ChatCompletionStreamResponse(
                            id: streamId,
                            model: res.modelID,
                            content: "",
                            logprobs: Self.buildChoiceLogprobs(logprobBuffer),
                            isFirst: false
                        )
                        logprobBuffer = []
                        if let flushData = try? encoder.encode(flushChunk),
                           let jsonString = String(data: flushData, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                        }
                    }
                }

                if deferStructuredOutputContent,
                   finalizedTurn.toolCalls == nil,
                   finalizedTurn.content != nil || finalizedTurn.reasoningContent != nil || !logprobBuffer.isEmpty {
                    let visibleContent = streamingStopFilter.consume(finalizedTurn.content ?? "")
                    if visibleContent != (finalizedTurn.content ?? "") { logprobBuffer = [] }
                    if streamingStopFilter.stopped { stoppedBySequence = true }
                    let contentChunk = ChatCompletionStreamResponse(
                        id: streamId,
                        model: res.modelID,
                        content: visibleContent,
                        reasoningContent: finalizedTurn.reasoningContent,
                        logprobs: logprobBuffer.isEmpty ? nil : Self.buildChoiceLogprobs(logprobBuffer),
                        isFirst: false
                    )
                    logprobBuffer = []
                    let contentData = try encoder.encode(contentChunk)
                    if let jsonString = String(data: contentData, encoding: .utf8) {
                        try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                    }
                }

                let withheldContent = streamingStopFilter.flush()
                if !withheldContent.isEmpty {
                    let flushChunk = ChatCompletionStreamResponse(
                        id: streamId,
                        model: res.modelID,
                        content: withheldContent,
                        isFirst: false
                    )
                    let flushData = try encoder.encode(flushChunk)
                    if let jsonString = String(data: flushData, encoding: .utf8) {
                        try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                    }
                }

                let promptTime = realPromptTime ?? 0
                let generateTime = realGenerateTime ?? generationDuration
                let usage = StreamUsage(
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    completionTime: generateTime,
                    promptTime: promptTime,
                    cachedTokens: realCachedTokens
                )
                let finalChunk = ChatCompletionStreamResponse(
                    id: streamId,
                    model: res.modelID,
                    content: "",
                    isFinished: true,
                    finishReason: finishReason
                )
                let finalData = try encoder.encode(finalChunk)
                if let jsonString = String(data: finalData, encoding: .utf8) {
                    try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                }
                // Gate the usage chunk on stream_options.include_usage. (T1.2)
                if chatRequest.includeStreamingUsage {
                    let usageChunk = ChatCompletionStreamResponse(
                        id: streamId,
                        model: res.modelID,
                        usage: usage,
                        timings: StreamTimings(prompt_n: promptTokens, prompt_ms: promptTime * 1000, predicted_n: completionTokens, predicted_ms: generateTime * 1000)
                    )
                    let usageData = try encoder.encode(usageChunk)
                    if let jsonString = String(data: usageData, encoding: .utf8) {
                        try? await writer.write(.buffer(.init(string: "data: \(jsonString)\n\n")))
                    }
                }
                // AFM Profile: send as a final SSE event before [DONE]
                if wantStreamProfile {
                    if wantStreamExtended {
                        let extended = self.service.stopAPIProfileExtended(
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            promptTime: promptTime,
                            generateTime: generateTime
                        )
                        if let data = try? JSONEncoder().encode(["afm_profile_extended": extended]),
                           let json = String(data: data, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                        }
                    } else {
                        let profile = self.service.stopAPIProfile(
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            promptTime: promptTime,
                            generateTime: generateTime
                        )
                        if let data = try? JSONEncoder().encode(["afm_profile": profile]),
                           let json = String(data: data, encoding: .utf8) {
                            try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                        }
                    }
                }
                try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                try? await writer.write(.end)
            } catch {
                // Cleanup profile timer on error to prevent leak
                if wantStreamProfile || wantStreamExtended {
                    _ = self.service.stopAPIProfile(promptTokens: 0, completionTokens: 0, promptTime: 0, generateTime: 0)
                }
                // Distinguish cooperative cancellation (T1.4/T1.5) from genuine
                // errors. Cancellation must NOT emit a "⚠️ Error" content chunk;
                // the stream should end cleanly with finish_reason="cancelled".
                let isCancellation = (error is CancellationError) || Task.isCancelled
                let completionTokens = self.estimateTokens(fullContent)
                let generationDuration = max(Date().timeIntervalSince(started), 0.001)
                let tokPerSec = generationDuration > 0 ? Double(completionTokens) / generationDuration : 0
                if isCancellation {
                    if self.veryVerbose {
                        print("\(Self.orange)[\(Self.timestamp())] MLX cancelled: stream=true completion_tokens=\(completionTokens) elapsed=\(String(format: "%.2f", generationDuration))s\(Self.reset)")
                        fflush(stdout)
                    }
                    let cancelledFinal = ChatCompletionStreamResponse(
                        id: streamId,
                        model: self.modelID,
                        content: "",
                        isFinished: true,
                        finishReason: "cancelled"
                    )
                    if let data = try? encoder.encode(cancelledFinal), let json = String(data: data, encoding: .utf8) {
                        try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                    }
                } else {
                    if self.veryVerbose {
                        let (finalAnswer, _) = Self.extractThinkContent(from: fullContent, startTag: self.service.thinkStartTag ?? "<think>", endTag: self.service.thinkEndTag ?? "</think>")
                        let trimmedAnswer = finalAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmedAnswer.isEmpty {
                            print("\(Self.teal)[\(Self.timestamp())] MLX full answer (before error):\n  \(trimmedAnswer)\(Self.reset)")
                        }
                        print("\(Self.orange)[\(Self.timestamp())] MLX done: stream=true\n  completion_tokens=\(completionTokens)\n  elapsed=\(String(format: "%.2f", generationDuration))s tok/s=\(String(format: "%.1f", tokPerSec))\n  error=\(error.localizedDescription)\(Self.reset)")
                        fflush(stdout)
                    }
                    req.logger.error("[\(Self.timestamp())] MLX stream error: \(error)")
                    let streamError: OpenAIError
                    if let serviceError = error as? MLXServiceError {
                        let code: String
                        switch serviceError {
                        case .visionAssetsUnavailable:
                            code = "vision_assets_unavailable"
                        case .unsupportedMediaInput:
                            code = "unsupported_media_input"
                        case .invalidMediaInput:
                            code = "invalid_media_input"
                        default:
                            code = "mlx_error"
                        }
                        streamError = OpenAIError(
                            message: serviceError.localizedDescription,
                            type: code == "mlx_error" ? "mlx_error" : "invalid_request_error",
                            code: code,
                            requestId: streamReqId.isEmpty ? nil : streamReqId
                        )
                    } else {
                        streamError = OpenAIError(
                            message: error.localizedDescription,
                            type: "mlx_error",
                            code: "stream_error",
                            requestId: streamReqId.isEmpty ? nil : streamReqId
                        )
                    }
                    if let data = try? encoder.encode(streamError), let json = String(data: data, encoding: .utf8) {
                        try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
                    }
                }
                try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                try? await writer.write(.end)
            }
            } // end bodyTask
            // T1.4/T1.5: Bridge bodyTask into the pre-registered cancel handle,
            // await completion, then release. Pre-registration eliminates the
            // race where cancel arrives before the closure fires.
            cancelHandle.assign(bodyTask)
            _ = await bodyTask.value
            if !streamReqId.isEmpty {
                await inflightRegistry.release(
                    id: streamReqId,
                    registration: requestRegistration
                )
            }
        })

        return httpResponse
    }

    private func normalizedMaxTokens(_ requested: Int?) -> Int {
        Self.resolveEffectiveMaxTokens(requested: requested, serverDefault: maxTokens)
    }

    /// Fallback when neither the request nor the server sets max_tokens.
    /// Deliberately larger than mlx_lm.server's 512 so interactive clients that
    /// omit max_tokens don't get truncated thinking/code output; parity
    /// benchmarks should pass an explicit max_tokens on both servers.
    static let defaultMaxCompletionTokens = 8_192

    static func resolveEffectiveMaxTokens(requested: Int?, serverDefault: Int?) -> Int {
        if let requested, requested > 0 { return requested }
        if let serverDefault, serverDefault > 0 { return serverDefault }
        return Self.defaultMaxCompletionTokens
    }

    private func sanitizeDegenerateTail(_ text: String) -> String {
        var cleaned = text

        if let badChar = cleaned.lastIndex(of: "�"), cleaned.distance(from: badChar, to: cleaned.endIndex) < 512 {
            cleaned = String(cleaned[..<badChar])
        }

        let nsrange = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
        guard let match = Self.degenerateTailRegex.firstMatch(in: cleaned, range: nsrange),
              let range = Range(match.range, in: cleaned) else {
            return cleaned
        }

        return String(cleaned[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func requiresStructuredOutputSanitization(_ responseFormat: ResponseFormat?) -> Bool {
        OpenAIResponseFormatPolicy.requiresStructuredOutputSanitization(responseFormat)
    }

    static func sanitizeStructuredOutput(_ text: String, responseFormat: ResponseFormat?) -> String {
        OpenAIResponseFormatPolicy.sanitizeStructuredOutput(text, responseFormat: responseFormat)
    }

    private static func isCompleteJSONToolArguments(_ text: String) -> Bool {
        guard let data = text.data(using: .utf8) else { return false }
        do {
            let object = try JSONSerialization.jsonObject(with: data)
            return object is [String: Any]
        } catch {
            return false
        }
    }

    private func createSuccessResponse(req: Request, response: ChatCompletionResponse, grammarDowngraded: Bool = false) async throws -> Response {
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        if grammarDowngraded {
            httpResponse.headers.add(name: "X-Grammar-Constraints", value: "downgraded")
        }
        try httpResponse.content.encode(response)
        return httpResponse
    }

    private func createErrorResponse(req: Request, error: OpenAIError, status: HTTPStatus) async throws -> Response {
        let httpResponse = Response(status: status)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(error)
        return httpResponse
    }

    private func estimateTokens(_ text: String) -> Int {
        let words = text.split(whereSeparator: \.isWhitespace).count
        return Int(max(Double(text.count) / 4.0, Double(words) / 0.75))
    }

    private static let isoFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    private static func timestamp() -> String {
        isoFormatter.string(from: Date())
    }

    /// Fuzzy-match a hallucinated tool name against valid candidates.
    /// Returns the best match if edit distance ≤ 3, otherwise nil.
    private static func fuzzyMatchToolName(_ name: String, candidates: [String]) -> String? {
        var bestMatch: String?
        var bestDist = Int.max
        for candidate in candidates {
            let d = editDistance(name.lowercased(), candidate.lowercased())
            if d < bestDist {
                bestDist = d
                bestMatch = candidate
            }
        }
        return bestDist <= 3 ? bestMatch : nil
    }

    /// Levenshtein edit distance between two strings.
    private static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        let m = a.count, n = b.count
        if m == 0 { return n }
        if n == 0 { return m }
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                curr[j] = a[i-1] == b[j-1] ? prev[j-1] : 1 + Swift.min(prev[j], curr[j-1], prev[j-1])
            }
            prev = curr
        }
        return prev[n]
    }

    /// Extract argument key names from a JSON arguments string for log preview.
    private static func argKeysPreview(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json.count > 40 ? "\(json.prefix(40))..." : json
        }
        return dict.keys.sorted().joined(separator: ", ")
    }

    private static func cacheStatsSummary(cachedTokens: Int, totalPromptTokens: Int) -> String {
        let total = max(totalPromptTokens, cachedTokens)
        let suffix = max(0, total - cachedTokens)
        guard cachedTokens > 0 else { return " | cache: MISS suffix=\(suffix)" }
        let ratio = total > 0 ? Int(Double(cachedTokens) / Double(total) * 100) : 0
        return " | cache: HIT \(cachedTokens)/\(total) (\(ratio)%) suffix=\(suffix)"
    }

    /// ANSI color codes
    private static let orange = "\u{1B}[38;5;208m"
    private static let pink = "\u{1B}[38;5;213m"
    private static let red = "\u{1B}[38;5;196m"
    private static let teal = "\u{1B}[38;5;43m"
    private static let purple = "\u{1B}[38;5;135m"
    private static let gold = "\u{1B}[38;5;178m"
    private static let cyan = "\u{1B}[38;5;87m"   // -VV trace logging
    private static let reset = "\u{1B}[0m"

    // MARK: - Harmony channel parsing (gpt-oss) (#121)

    enum HarmonyChannel {
        case none           // Awaiting <|channel|>
        case awaitingName   // Saw <|channel|>; reading name until <|message|>
        case analysis       // Inside analysis -> reasoning_content
        case final          // Inside final -> content
        case commentary     // Inside commentary (tool calls) — discarded for now
        case done           // After <|return|>; stop
    }

    struct HarmonyState {
        var channel: HarmonyChannel = .none
        var nameBuf: String = ""
        var stopReached: Bool = false
    }

    enum MuseResponseChannel {
        case none
        case awaitingName
        case reasoning
        case content
        case discard
        case done
    }

    struct MuseResponseChannelState {
        var channel: MuseResponseChannel = .none
        var nameBuf: String = ""
        var stopReached: Bool = false
    }

    /// Length of the longest harmony control token. Used for boundary handling
    /// so we never flush a tail that could still match a control token.
    private static let harmonyMaxControlLen = 11   // "<|channel|>" / "<|message|>"

    /// Extract harmony channel content from a streaming buffer.
    /// Routes `<|channel|>analysis<|message|>...<|end|>` to reasoning,
    /// `<|channel|>final<|message|>...<|return|>/<|end|>` to content,
    /// drops `commentary` (tool-call channel), and strips control tokens.
    /// Sets `state.stopReached = true` on `<|return|>`. The buffer retains
    /// any partial control-token fragment for the next call. (#121)
    static func extractHarmonyChannels(
        buffer: inout String,
        state: inout HarmonyState
    ) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""

        parseLoop: while !buffer.isEmpty && !state.stopReached {
            switch state.channel {
            case .done:
                buffer = ""
                break parseLoop

            case .none:
                if let r = buffer.range(of: "<|channel|>") {
                    // Discard preamble (e.g. "<|start|>assistant") before the channel marker
                    buffer = String(buffer[r.upperBound...])
                    state.channel = .awaitingName
                    state.nameBuf = ""
                } else if buffer.count > Self.harmonyMaxControlLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -Self.harmonyMaxControlLen)
                    buffer = String(buffer[safeEnd...])
                    break parseLoop
                } else {
                    break parseLoop
                }

            case .awaitingName:
                if let r = buffer.range(of: "<|message|>") {
                    state.nameBuf += String(buffer[..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    let name = state.nameBuf.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    state.nameBuf = ""
                    switch name {
                    case "analysis": state.channel = .analysis
                    case "final": state.channel = .final
                    case "commentary": state.channel = .commentary
                    default: state.channel = .commentary
                    }
                } else if buffer.count > Self.harmonyMaxControlLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -Self.harmonyMaxControlLen)
                    state.nameBuf += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break parseLoop
                } else {
                    break parseLoop
                }

            case .analysis, .final, .commentary:
                let endRange = buffer.range(of: "<|end|>")
                let returnRange = buffer.range(of: "<|return|>")
                let nextMarker: (range: Range<String.Index>, isReturn: Bool)?
                if let e = endRange, let r = returnRange {
                    nextMarker = (e.lowerBound < r.lowerBound) ? (e, false) : (r, true)
                } else if let e = endRange {
                    nextMarker = (e, false)
                } else if let r = returnRange {
                    nextMarker = (r, true)
                } else {
                    nextMarker = nil
                }

                if let marker = nextMarker {
                    let text = String(buffer[..<marker.range.lowerBound])
                    switch state.channel {
                    case .analysis: reasoning += text
                    case .final: content += text
                    default: break  // commentary discarded
                    }
                    buffer = String(buffer[marker.range.upperBound...])
                    if marker.isReturn {
                        state.channel = .done
                        state.stopReached = true
                    } else {
                        state.channel = .none
                    }
                } else if buffer.count > Self.harmonyMaxControlLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -Self.harmonyMaxControlLen)
                    let text = String(buffer[buffer.startIndex..<safeEnd])
                    switch state.channel {
                    case .analysis: reasoning += text
                    case .final: content += text
                    default: break
                    }
                    buffer = String(buffer[safeEnd...])
                    break parseLoop
                } else {
                    break parseLoop
                }
            }
        }

        let r: String? = reasoning.isEmpty ? nil : reasoning
        let c: String? = content.isEmpty ? nil : content
        return (reasoning: r, content: c)
    }

    /// Whole-text harmony extraction for non-streaming responses. (#121)
    static func extractHarmonyContent(from text: String) -> (content: String, reasoning: String?) {
        var buffer = text
        var state = HarmonyState()
        var allReasoning = ""
        var allContent = ""
        while !buffer.isEmpty {
            let extracted = extractHarmonyChannels(buffer: &buffer, state: &state)
            if let r = extracted.reasoning { allReasoning += r }
            if let c = extracted.content { allContent += c }
            if extracted.reasoning == nil && extracted.content == nil { break }
        }
        // Flush remainder for the channel we ended in (no terminator before EOS).
        if !buffer.isEmpty {
            switch state.channel {
            case .analysis: allReasoning += buffer
            case .final: allContent += buffer
            default: break
            }
        }
        let reasoning: String? = allReasoning.isEmpty ? nil : allReasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        let content = allContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return (content, reasoning)
    }

    private static let museMaxControlLen = 18   // "to=self<|message|>" / "to=user<|message|>"

    /// Extract Muse response-template channels. Muse may emit `to=self<|message|>`
    /// for internal reasoning and `to=user<|message|>` for visible assistant text.
    /// Routes those channels to OpenAI `reasoning_content` and `content`, strips
    /// channel controls, and preserves only a short suffix across chunk boundaries.
    static func extractMuseResponseChannels(
        buffer: inout String,
        state: inout MuseResponseChannelState
    ) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""

        parseLoop: while !buffer.isEmpty && !state.stopReached {
            switch state.channel {
            case .done:
                buffer = ""
                break parseLoop

            case .none:
                if let r = buffer.range(of: "to=") {
                    buffer = String(buffer[r.upperBound...])
                    state.channel = .awaitingName
                    state.nameBuf = ""
                } else if buffer.count > Self.museMaxControlLen {
                    // Muse content should start with a channel. If it does not, keep only
                    // enough tail to detect a split `to=` marker later.
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -Self.museMaxControlLen)
                    buffer = String(buffer[safeEnd...])
                    break parseLoop
                } else {
                    break parseLoop
                }

            case .awaitingName:
                if let r = buffer.range(of: "<|message|>") {
                    state.nameBuf += String(buffer[..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    let name = state.nameBuf.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    state.nameBuf = ""
                    switch name {
                    case "self": state.channel = .reasoning
                    case "user", "assistant": state.channel = .content
                    default: state.channel = .discard
                    }
                } else if buffer.count > Self.museMaxControlLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -Self.museMaxControlLen)
                    state.nameBuf += String(buffer[buffer.startIndex..<safeEnd])
                    buffer = String(buffer[safeEnd...])
                    break parseLoop
                } else {
                    break parseLoop
                }

            case .reasoning, .content, .discard:
                let eomRange = buffer.range(of: "<|eom|>")
                let returnRange = buffer.range(of: "<|return|>")
                let nextMarker: (range: Range<String.Index>, isReturn: Bool)?
                if let e = eomRange, let r = returnRange {
                    nextMarker = (e.lowerBound < r.lowerBound) ? (e, false) : (r, true)
                } else if let e = eomRange {
                    nextMarker = (e, false)
                } else if let r = returnRange {
                    nextMarker = (r, true)
                } else {
                    nextMarker = nil
                }

                if let marker = nextMarker {
                    let text = String(buffer[..<marker.range.lowerBound])
                    switch state.channel {
                    case .reasoning: reasoning += text
                    case .content: content += text
                    default: break
                    }
                    buffer = String(buffer[marker.range.upperBound...])
                    if marker.isReturn {
                        state.channel = .done
                        state.stopReached = true
                    } else {
                        state.channel = .none
                    }
                } else if buffer.count > Self.museMaxControlLen {
                    let safeEnd = buffer.index(buffer.endIndex, offsetBy: -Self.museMaxControlLen)
                    let text = String(buffer[buffer.startIndex..<safeEnd])
                    switch state.channel {
                    case .reasoning: reasoning += text
                    case .content: content += text
                    default: break
                    }
                    buffer = String(buffer[safeEnd...])
                    break parseLoop
                } else {
                    break parseLoop
                }
            }
        }

        return (
            reasoning: reasoning.isEmpty ? nil : reasoning,
            content: content.isEmpty ? nil : content
        )
    }

    static func flushMuseResponseChannelRemainder(
        buffer: inout String,
        state: inout MuseResponseChannelState
    ) -> (reasoning: String?, content: String?) {
        let extracted = extractMuseResponseChannels(buffer: &buffer, state: &state)
        var reasoning = extracted.reasoning ?? ""
        var content = extracted.content ?? ""
        if !buffer.isEmpty {
            switch state.channel {
            case .reasoning: reasoning += buffer
            case .content: content += buffer
            default: break
            }
            buffer = ""
        }
        return (
            reasoning: reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : reasoning,
            content: content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : content
        )
    }

    static func extractMuseResponseContent(from text: String) -> (content: String, reasoning: String?) {
        var buffer = text
        var state = MuseResponseChannelState()
        var allReasoning = ""
        var allContent = ""
        while !buffer.isEmpty {
            let extracted = extractMuseResponseChannels(buffer: &buffer, state: &state)
            if let r = extracted.reasoning { allReasoning += r }
            if let c = extracted.content { allContent += c }
            if extracted.reasoning == nil && extracted.content == nil { break }
        }
        let remaining = flushMuseResponseChannelRemainder(buffer: &buffer, state: &state)
        if let r = remaining.reasoning { allReasoning += r }
        if let c = remaining.content { allContent += c }
        return (
            allContent.trimmingCharacters(in: .whitespacesAndNewlines),
            allReasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : allReasoning.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    /// Extract `<think>...</think>` content from a streaming buffer.
    /// Returns any reasoning and regular content that can be flushed.
    /// The buffer retains incomplete tag fragments for the next call.
    /// Longest suffix of `buffer` that is a proper prefix of `tag` (capped at tag.count-1).
    /// Streaming withholds exactly this many trailing chars so it can emit everything else
    /// immediately while never splitting a boundary tag across chunks. Returns 0 when the
    /// buffer tail can't begin the tag (the common case → emit with zero added latency).
    static func partialBoundaryHoldback(_ buffer: String, _ tag: String) -> Int {
        let b = Array(buffer), t = Array(tag)
        var k = Swift.min(b.count, t.count - 1)
        while k > 0 {
            if Array(b.suffix(k)) == Array(t.prefix(k)) { return k }
            k -= 1
        }
        return 0
    }

    /// Remove structural wrapper tokens (service.structuralStripTags — e.g. Cohere's
    /// `<|START_TEXT|>`/`<|END_TEXT|>`) that are neither think tags nor stop sequences
    /// but would otherwise leak into content/reasoning. No-op for the empty list. (#148)
    static func stripStructuralTags(_ text: String, tags: [String]) -> String {
        guard !tags.isEmpty, !text.isEmpty else { return text }
        var out = text
        for tag in tags where out.contains(tag) {
            out = out.replacingOccurrences(of: tag, with: "")
        }
        return out
    }

    static func extractThinkTags(
        buffer: inout String,
        insideThinkBlock: inout Bool,
        startTag: String = "<think>",
        endTag: String = "</think>"
    ) -> (reasoning: String?, content: String?) {
        var reasoning = ""
        var content = ""

        while !buffer.isEmpty {
            if insideThinkBlock {
                if let endRange = buffer.range(of: endTag) {
                    reasoning += String(buffer[buffer.startIndex..<endRange.lowerBound])
                    buffer = String(buffer[endRange.upperBound...])
                    insideThinkBlock = false
                } else {
                    // Emit reasoning eagerly: withhold only the trailing chars that could be the
                    // start of a partial end tag (a prefix of "</think>"), instead of a fixed
                    // endTagLen. For typical reasoning text (not ending mid-tag) this holds back 0
                    // chars and streams immediately, cutting first-reasoning-token latency (TTFT)
                    // by several tokens. Correctness preserved: a partial end tag is never emitted.
                    let hb = Self.partialBoundaryHoldback(buffer, endTag)
                    if buffer.count > hb {
                        let safeEnd = buffer.index(buffer.endIndex, offsetBy: -hb)
                        reasoning += String(buffer[buffer.startIndex..<safeEnd])
                        buffer = String(buffer[safeEnd...])
                    }
                    break
                }
            } else {
                let startRange = buffer.range(of: startTag)
                let endRange = buffer.range(of: endTag)
                // Orphan end tag with no preceding start tag: the model reasoned without
                // emitting the start tag (the template opened or pre-closed the block —
                // e.g. cohere2_moe under enable_thinking=false). The text before it has
                // already streamed as content, so just drop the literal tag. (#148)
                if let startRange {
                    if let endRange, endRange.lowerBound < startRange.lowerBound {
                        content += String(buffer[buffer.startIndex..<endRange.lowerBound])
                        buffer = String(buffer[endRange.upperBound...])
                    } else {
                        let before = String(buffer[buffer.startIndex..<startRange.lowerBound])
                        content += before
                        buffer = String(buffer[startRange.upperBound...])
                        insideThinkBlock = true
                    }
                } else if let endRange {
                    content += String(buffer[buffer.startIndex..<endRange.lowerBound])
                    buffer = String(buffer[endRange.upperBound...])
                } else {
                    // Same eager-emit optimization for the pre-think content path: withhold only a
                    // partial start- or end-tag prefix, not a fixed startTagLen.
                    let hb = Swift.max(
                        Self.partialBoundaryHoldback(buffer, startTag),
                        Self.partialBoundaryHoldback(buffer, endTag)
                    )
                    if buffer.count > hb {
                        let safeEnd = buffer.index(buffer.endIndex, offsetBy: -hb)
                        content += String(buffer[buffer.startIndex..<safeEnd])
                        buffer = String(buffer[safeEnd...])
                    }
                    break
                }
            }
        }

        let r: String? = reasoning.isEmpty ? nil : reasoning
        let c: String? = content.isEmpty ? nil : content

        if r == nil && c == nil {
            return (reasoning: nil, content: nil)
        }

        return (reasoning: r, content: c)
    }

    /// Extract think tags from a complete (non-streaming) response.
    static func extractThinkContent(from text: String, startTag: String = "<think>", endTag: String = "</think>") -> (content: String, reasoning: String?) {
        AFMReasoningOutputExtractor.extractThinkContent(
            from: text,
            startTag: startTag,
            endTag: endTag
        )
    }

    static func finalizeAssistantTurn(
        content: String,
        toolCalls: [ResponseToolCall]?,
        toolChoice: ToolChoice?,
        parallelToolCalls: Bool? = nil,
        extractThinking: Bool,
        thinkStartTag: String,
        thinkEndTag: String,
        stoppedBySequence: Bool,
        completionTokens: Int,
        maxTokens: Int,
        sanitizeContent: (String) -> String,
        responseChannelFormat: AFMResponseChannelFormat = .none,
        stopSequences: [String]? = nil
    ) -> FinalizedAssistantTurn {
        var effectiveToolCalls = applyToolChoice(toolCalls, toolChoice: toolChoice)
        // Honor parallel_tool_calls=false by truncating to the first call. (T1.3)
        if parallelToolCalls == false,
           let calls = effectiveToolCalls,
           calls.count > 1 {
            effectiveToolCalls = [calls[0]]
        }
        if let effectiveToolCalls, !effectiveToolCalls.isEmpty {
            return FinalizedAssistantTurn(
                finishReason: "tool_calls",
                content: nil,
                reasoningContent: nil,
                toolCalls: effectiveToolCalls
            )
        }

        let cleanedContent = sanitizeContent(content)
        let finalContent: String
        let reasoningContent: String?
        if extractThinking && responseChannelFormat == .harmony {
            (finalContent, reasoningContent) = extractHarmonyContent(from: cleanedContent)
        } else if extractThinking && responseChannelFormat == .muse {
            (finalContent, reasoningContent) = extractMuseResponseContent(from: cleanedContent)
        } else if extractThinking {
            (finalContent, reasoningContent) = extractThinkContent(
                from: cleanedContent,
                startTag: thinkStartTag,
                endTag: thinkEndTag
            )
        } else {
            finalContent = cleanedContent
            reasoningContent = nil
        }
        let visibleStop = trimAtFirstStop(finalContent, stopSequences: stopSequences)

        let finishReason = (stoppedBySequence || visibleStop.stopped) ? "stop" : (completionTokens >= maxTokens ? "length" : "stop")
        return FinalizedAssistantTurn(
            finishReason: finishReason,
            content: visibleStop.text,
            reasoningContent: reasoningContent,
            toolCalls: nil
        )
    }

    static func trimAtFirstStop(_ text: String, stopSequences: [String]?) -> (text: String, stopped: Bool) {
        guard let stopSequences else { return (text, false) }
        var earliest: Range<String.Index>?
        for stop in stopSequences where !stop.isEmpty {
            guard let range = text.range(of: stop) else { continue }
            if earliest == nil || range.lowerBound < earliest!.lowerBound {
                earliest = range
            }
        }
        guard let earliest else { return (text, false) }
        return (String(text[..<earliest.lowerBound]), true)
    }

    static func applyToolChoice(_ toolCalls: [ResponseToolCall]?, toolChoice: ToolChoice?) -> [ResponseToolCall]? {
        guard let toolCalls, !toolCalls.isEmpty else { return nil }
        guard let toolChoice else { return toolCalls }

        switch toolChoice {
        case .mode(let mode):
            return mode == "none" ? nil : toolCalls
        case .function(let functionChoice):
            let name = functionChoice.function.name
            let filtered = toolCalls.filter { $0.function.name == name }
            return filtered.isEmpty ? nil : filtered
        }
    }

    static func resolveEffectiveTools(_ tools: [RequestTool]?, toolChoice: ToolChoice?) throws -> [RequestTool]? {
        guard let tools, !tools.isEmpty else { return nil }
        guard let toolChoice else { return tools }

        switch toolChoice {
        case .mode(let mode):
            return mode == "none" ? nil : tools
        case .function(let functionChoice):
            let requestedName = functionChoice.function.name
            let filtered = tools.filter { $0.function.name == requestedName }
            guard !filtered.isEmpty else {
                throw Abort(
                    .badRequest,
                    reason: "tool_choice specifies function '\(requestedName)', but that tool was not provided"
                )
            }
            return filtered
        }
    }

    static func isToolCallAllowed(
        _ toolCall: ResponseToolCall,
        toolChoice: ToolChoice?,
        allowedFunctionName: String?,
        permittedToolIndices: inout Set<Int>
    ) -> Bool {
        switch toolChoice {
        case .mode(let mode) where mode == "none":
            return false
        case .function:
            guard let allowedFunctionName else { return false }
            let isAllowed = toolCall.function.name == allowedFunctionName
            if isAllowed {
                permittedToolIndices.insert(toolCall.index ?? permittedToolIndices.count)
            }
            return isAllowed
        default:
            if let index = toolCall.index {
                permittedToolIndices.insert(index)
            }
            return true
        }
    }

    static func isToolDeltaAllowed(
        _ delta: StreamDeltaToolCall,
        toolChoice: ToolChoice?,
        allowedFunctionName: String?,
        permittedToolIndices: inout Set<Int>
    ) -> Bool {
        switch toolChoice {
        case .mode(let mode) where mode == "none":
            return false
        case .function:
            if let name = delta.function?.name {
                let isAllowed = name == allowedFunctionName
                if isAllowed {
                    permittedToolIndices.insert(delta.index)
                }
                return isAllowed
            }
            return permittedToolIndices.contains(delta.index)
        default:
            permittedToolIndices.insert(delta.index)
            return true
        }
    }

    static func buildChoiceLogprobs(_ resolved: [AFMServerResolvedLogprob]?) -> ChoiceLogprobs? {
        guard let resolved, !resolved.isEmpty else { return nil }
        let content = resolved.map { entry in
            let topEntries = entry.topTokens.map { top in
                TopLogprobEntry(
                    token: top.token,
                    logprob: Double(top.logprob),
                    bytes: Array(top.token.utf8).map { Int($0) }
                )
            }
            return TokenLogprobContent(
                token: entry.token,
                logprob: Double(entry.logprob),
                bytes: Array(entry.token.utf8).map { Int($0) },
                topLogprobs: topEntries
            )
        }
        return ChoiceLogprobs(content: content)
    }

    /// Look up the schema type for a parameter in a tool's function schema.
    static func schemaTypeForParam(_ paramName: String, toolName: String, tools: [RequestTool]?) -> String? {
        guard let tools else { return nil }
        guard let tool = tools.first(where: { $0.function.name == toolName }),
              let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
              let props = paramsAny["properties"] as? [String: Any],
              let propSchema = props[paramName] as? [String: Any],
              let schemaType = propSchema["type"] as? String else { return nil }
        return schemaType
    }

    /// JSON-encode a parameter value: if it parses as a JSON array or object,
    /// return it as-is (structured); otherwise encode as a JSON string.
    static func jsonEncodeValue(_ s: String) -> String {
        if let data = s.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data),
           (parsed is [Any] || parsed is [String: Any]),
           let reencoded = try? JSONSerialization.data(withJSONObject: parsed),
           let result = String(data: reencoded, encoding: .utf8) {
            return result
        }
        return jsonEncodeString(s)
    }

    /// JSON-encode a string value with proper escaping, including surrounding quotes.
    static func jsonEncodeString(_ s: String) -> String {
        // Wrap in array so JSONSerialization accepts it, then strip the brackets.
        if let data = try? JSONSerialization.data(withJSONObject: [s]),
           let str = String(data: data, encoding: .utf8),
           str.hasPrefix("["), str.hasSuffix("]") {
            // str is e.g. ["hello \"world\""] — strip [ and ]
            let inner = str.dropFirst().dropLast()
            return String(inner)
        }
        // Fallback: manual escaping
        let escaped = s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escaped)\""
    }

    /// Escape a JSON object key (minimal: just backslash and quote).
    static func jsonEscapeKey(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\")
         .replacingOccurrences(of: "\"", with: "\\\"")
    }

    /// Convert camelCase to snake_case (e.g. "filePath" → "file_path").
    /// Used to build reverse mapping for Qwen3-Coder's parameter name conversion.
    static func toSnakeCase(_ s: String) -> String {
        var result = ""
        for (i, char) in s.enumerated() {
            if char.isUppercase {
                if i > 0 { result += "_" }
                result += char.lowercased()
            } else {
                result += String(char)
            }
        }
        return result
    }

    private func encodeJSON<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value),
              let text = String(data: data, encoding: .utf8) else {
            return "<json-encode-failed>"
        }
        return text
    }
}
