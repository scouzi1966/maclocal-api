import Vapor
import Foundation
import os

struct BatchCompletionsController: RouteCollection {
    private let service: any MLXChatServing
    private let modelID: String
    private let temperature: Double?
    private let topP: Double?
    private let maxTokens: Int?
    private let repetitionPenalty: Double?
    private let topK: Int?
    private let minP: Double?
    private let presencePenalty: Double?
    private let seed: Int?
    private let maxLogprobs: Int

    init(
        service: any MLXChatServing,
        modelID: String,
        temperature: Double? = nil,
        topP: Double? = nil,
        maxTokens: Int? = nil,
        repetitionPenalty: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        maxLogprobs: Int = 20
    ) {
        self.service = service
        self.modelID = modelID
        self.temperature = temperature
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.maxLogprobs = maxLogprobs
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "batch", "completions", body: .collect(maxSize: "100mb"), use: batchCompletions)
    }

    func batchCompletions(req: Request) async throws -> Response {
        let batchReq = try req.content.decode(BatchCompletionRequest.self)

        // Validation
        guard !batchReq.requests.isEmpty else {
            throw Abort(.badRequest, reason: "Batch must contain at least one request")
        }
        guard batchReq.requests.count <= 64 else {
            throw Abort(.badRequest, reason: "Batch size exceeds maximum of 64 requests")
        }

        let ids = batchReq.requests.map(\.customId)
        guard ids.allSatisfy({ !$0.isEmpty }) else {
            throw Abort(.badRequest, reason: "All requests must have a non-empty custom_id")
        }
        guard Set(ids).count == ids.count else {
            throw Abort(.badRequest, reason: "Duplicate custom_id values")
        }

        // Auto-promote if needed
        try await service.ensureBatchMode(concurrency: batchReq.requests.count)

        let response = Response(status: .ok)
        response.headers.replaceOrAdd(name: .contentType, value: "text/event-stream")
        response.headers.replaceOrAdd(name: .cacheControl, value: "no-cache")
        response.headers.replaceOrAdd(name: "X-Accel-Buffering", value: "no")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")

        // Grammar constraint header: check if any request has strict tools/schema
        let anyStrict = batchReq.requests.contains { item in
            MLXModelService.shouldDowngradeGrammarConstraints(
                responseFormat: item.body.responseFormat,
                tools: item.body.tools,
                supportsStrictToolGrammar: service.supportsStrictToolGrammar,
                enableGrammarConstraints: service.enableGrammarConstraints
            )
        }
        if anyStrict {
            response.headers.add(name: "X-Grammar-Constraints", value: "downgraded")
        }

        let svc = service
        let mdlID = modelID
        let temp = temperature
        let tp = topP
        let mt = maxTokens
        let rp = repetitionPenalty
        let tk = topK
        let mp = minP
        let pp = presencePenalty
        let sd = seed

        response.body = .init(asyncStream: { writer in
            let encoder = JSONEncoder()

            await withTaskGroup(of: Void.self) { group in
                // Per-request streams feed into a shared async channel
                let (mergedStream, mergedContinuation) = AsyncStream<String>.makeStream()
                let activeCount = OSAllocatedUnfairLock(initialState: batchReq.requests.count)

                for item in batchReq.requests {
                    group.addTask {
                        await self.processRequest(
                            item: item,
                            service: svc,
                            modelID: mdlID,
                            temperature: temp,
                            topP: tp,
                            maxTokens: mt,
                            repetitionPenalty: rp,
                            topK: tk,
                            minP: mp,
                            presencePenalty: pp,
                            seed: sd,
                            encoder: encoder,
                            continuation: mergedContinuation,
                            activeCount: activeCount
                        )
                    }
                }

                // Writer task: read from merged stream and write to SSE
                group.addTask {
                    for await sseData in mergedStream {
                        do {
                            try await writer.write(.buffer(.init(string: sseData)))
                        } catch {
                            break
                        }
                    }
                    // Write final DONE
                    try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                    try? await writer.write(.end)
                }
            }

            svc.releaseBatchReference()
        })

        return response
    }

    private func processRequest(
        item: BatchRequestItem,
        service: any MLXChatServing,
        modelID: String,
        temperature: Double?,
        topP: Double?,
        maxTokens: Int?,
        repetitionPenalty: Double?,
        topK: Int?,
        minP: Double?,
        presencePenalty: Double?,
        seed: Int?,
        encoder: JSONEncoder,
        continuation: AsyncStream<String>.Continuation,
        activeCount: OSAllocatedUnfairLock<Int>
    ) async {
        let chatReq = item.body
        let customId = item.customId
        let isStreaming = chatReq.stream ?? false

        do {
            guard service.tryReserveSlot() else {
                let errorEvent = makeErrorEvent(customId: customId, message: "Server at capacity", type: "server_error", encoder: encoder)
                continuation.yield(errorEvent)
                decrementAndFinishIfDone(activeCount: activeCount, continuation: continuation)
                return
            }
            defer { service.releaseSlot() }

            let effectiveModel = service.normalizeModel(chatReq.model ?? modelID)
            let effectiveMaxTokens = chatReq.effectiveMaxTokens ?? maxTokens ?? Int.max

            let streamResult = try await service.generateStreaming(
                model: effectiveModel,
                messages: chatReq.messages,
                temperature: chatReq.temperature ?? temperature,
                maxTokens: effectiveMaxTokens,
                topP: chatReq.topP ?? topP,
                repetitionPenalty: chatReq.effectiveRepetitionPenalty ?? repetitionPenalty,
                topK: chatReq.topK ?? topK,
                minP: chatReq.minP ?? minP,
                presencePenalty: chatReq.presencePenalty ?? presencePenalty,
                seed: chatReq.seed ?? seed,
                logprobs: chatReq.logprobs,
                topLogprobs: chatReq.topLogprobs,
                tools: chatReq.tools,
                stop: chatReq.stop,
                responseFormat: chatReq.responseFormat,
                chatTemplateKwargs: chatReq.chatTemplateKwargs
            )

            let extractThinking = streamResult.thinkStartTag != nil
            let thinkStart = streamResult.thinkStartTag ?? "<think>"
            let thinkEnd = streamResult.thinkEndTag ?? "</think>"

            if isStreaming {
                // Emit streaming chunks tagged with custom_id
                // Includes per-chunk think extraction, logprobs, and tool call deltas
                let completionId = "chatcmpl-\(UUID().uuidString.lowercased().prefix(12))"
                var tokenCount = 0
                var promptTokens = streamResult.promptTokens
                var hasToolCalls = false
                var fullText = ""
                let deferStructuredOutputContent =
                    MLXChatCompletionsController.requiresStructuredOutputSanitization(chatReq.responseFormat)

                // Think extraction state
                var thinkBuffer = ""
                var insideThinkBlock = false

                for try await chunk in streamResult.stream {
                    tokenCount += 1
                    fullText += chunk.text

                    var event: [String: Any] = [
                        "custom_id": customId,
                        "id": completionId,
                        "object": "chat.completion.chunk",
                        "created": Int(Date().timeIntervalSince1970),
                        "model": effectiveModel,
                    ]

                    var delta: [String: Any] = [:]
                    if tokenCount == 1 { delta["role"] = "assistant" }

                    // Think extraction on each chunk
                    if extractThinking && !chunk.text.isEmpty {
                        thinkBuffer += chunk.text
                        let extracted = MLXChatCompletionsController.extractThinkTags(
                            buffer: &thinkBuffer,
                            insideThinkBlock: &insideThinkBlock,
                            startTag: thinkStart,
                            endTag: thinkEnd
                        )
                        if let reasoning = extracted.reasoning {
                            delta["reasoning_content"] = reasoning
                        }
                        if !deferStructuredOutputContent,
                           let content = extracted.content, !content.isEmpty {
                            delta["content"] = content
                        }
                    } else if !chunk.text.isEmpty && !deferStructuredOutputContent {
                        delta["content"] = chunk.text
                    }

                    // Tool call deltas
                    if let deltas = chunk.toolCallDeltas {
                        var tcArray: [[String: Any]] = []
                        for d in deltas {
                            var tc: [String: Any] = ["index": d.index]
                            var fn: [String: Any] = [:]
                            if let name = d.function?.name { fn["name"] = name }
                            if let args = d.function?.arguments { fn["arguments"] = args }
                            if !fn.isEmpty { tc["function"] = fn }
                            if let id = d.id { tc["id"] = id }
                            if let type = d.type { tc["type"] = type }
                            tcArray.append(tc)
                        }
                        delta["tool_calls"] = tcArray
                        hasToolCalls = true
                    }

                    // Logprobs
                    if let lps = chunk.logprobs {
                        let choiceLogprobs = MLXChatCompletionsController.buildChoiceLogprobs(lps)
                        if let clp = choiceLogprobs, let content = clp.content {
                            event["logprobs"] = ["content": content.map { lp in
                                ["token": lp.token, "logprob": lp.logprob, "bytes": lp.bytes as Any, "top_logprobs": lp.topLogprobs.map { t in
                                    ["token": t.token, "logprob": t.logprob, "bytes": t.bytes as Any] as [String: Any]
                                }] as [String: Any]
                            }]
                        }
                    }

                    var choiceDict: [String: Any] = ["index": 0, "delta": delta]

                    if let ct = chunk.completionTokens {
                        // Final chunk with usage
                        // Flush remaining think buffer
                        if extractThinking && !thinkBuffer.isEmpty {
                            insideThinkBlock = false
                            let flushed = MLXChatCompletionsController.extractThinkTags(
                                buffer: &thinkBuffer,
                                insideThinkBlock: &insideThinkBlock,
                                startTag: thinkStart,
                                endTag: thinkEnd
                            )
                            if let r = flushed.reasoning { delta["reasoning_content"] = r }
                            if !deferStructuredOutputContent,
                               let c = flushed.content, !c.isEmpty { delta["content"] = c }
                            // Flush any remaining buffer as content
                            if !deferStructuredOutputContent && !thinkBuffer.isEmpty {
                                let existing = delta["content"] as? String ?? ""
                                delta["content"] = existing + thinkBuffer
                                thinkBuffer = ""
                            }
                            choiceDict["delta"] = delta
                        }

                        if deferStructuredOutputContent && !hasToolCalls && chunk.toolCalls == nil {
                            let sanitized = MLXChatCompletionsController.sanitizeStructuredOutput(
                                fullText,
                                responseFormat: chatReq.responseFormat
                            )
                            if !sanitized.isEmpty {
                                delta["content"] = sanitized
                                choiceDict["delta"] = delta
                            }
                        }

                        choiceDict["finish_reason"] = (hasToolCalls || chunk.toolCalls != nil) ? "tool_calls" : "stop"
                        if let pt = chunk.promptTokens { promptTokens = pt }
                        event["usage"] = [
                            "prompt_tokens": promptTokens,
                            "completion_tokens": ct,
                            "total_tokens": promptTokens + ct
                        ]
                    }

                    event["choices"] = [choiceDict]

                    if let jsonData = try? JSONSerialization.data(withJSONObject: event),
                       let jsonStr = String(data: jsonData, encoding: .utf8) {
                        continuation.yield("data: \(jsonStr)\n\n")
                    }
                }
            } else {
                // Non-streaming: use StreamCollector for full post-processing
                let collected = try await StreamCollector.collect(
                    from: streamResult,
                    extractThinking: extractThinking,
                    thinkStartTag: thinkStart,
                    thinkEndTag: thinkEnd,
                    maxTokens: effectiveMaxTokens
                )

                let choiceLogprobs = MLXChatCompletionsController.buildChoiceLogprobs(collected.logprobs)

                var message: [String: Any] = ["role": "assistant"]
                if let tc = collected.toolCalls, !tc.isEmpty {
                    message["content"] = NSNull()
                    message["tool_calls"] = tc.map { toolCall in
                        [
                            "id": toolCall.id,
                            "type": toolCall.type,
                            "function": [
                                "name": toolCall.function.name,
                                "arguments": toolCall.function.arguments
                            ]
                        ] as [String: Any]
                    }
                } else {
                    message["content"] = MLXChatCompletionsController.sanitizeStructuredOutput(
                        collected.content ?? "",
                        responseFormat: chatReq.responseFormat
                    )
                    if let reasoning = collected.reasoningContent {
                        message["reasoning_content"] = reasoning
                    }
                }

                var choiceDict: [String: Any] = [
                    "index": 0,
                    "message": message,
                    "finish_reason": collected.finishReason
                ]

                if let clp = choiceLogprobs {
                    choiceDict["logprobs"] = ["content": clp.content?.map { lp in
                        ["token": lp.token, "logprob": lp.logprob, "bytes": lp.bytes as Any, "top_logprobs": lp.topLogprobs.map { t in
                            ["token": t.token, "logprob": t.logprob, "bytes": t.bytes as Any] as [String: Any]
                        }] as [String: Any]
                    } as Any]
                }

                let event: [String: Any] = [
                    "custom_id": customId,
                    "object": "chat.completion",
                    "id": "chatcmpl-\(UUID().uuidString.lowercased().prefix(12))",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": effectiveModel,
                    "choices": [choiceDict],
                    "usage": [
                        "prompt_tokens": collected.promptTokens,
                        "completion_tokens": collected.completionTokens,
                        "total_tokens": collected.promptTokens + collected.completionTokens
                    ]
                ]

                if let jsonData = try? JSONSerialization.data(withJSONObject: event),
                   let jsonStr = String(data: jsonData, encoding: .utf8) {
                    continuation.yield("data: \(jsonStr)\n\n")
                }
            }
        } catch {
            let errorEvent = makeErrorEvent(customId: customId, message: error.localizedDescription, type: "server_error", encoder: encoder)
            continuation.yield(errorEvent)
        }

        decrementAndFinishIfDone(activeCount: activeCount, continuation: continuation)
    }

    private func makeErrorEvent(customId: String, message: String, type: String, encoder: JSONEncoder) -> String {
        let event: [String: Any] = [
            "custom_id": customId,
            "object": "batch.error",
            "error": ["message": message, "type": type]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: event),
           let str = String(data: data, encoding: .utf8) {
            return "data: \(str)\n\n"
        }
        return "data: {\"custom_id\":\"\(customId)\",\"object\":\"batch.error\",\"error\":{\"message\":\"Internal error\",\"type\":\"server_error\"}}\n\n"
    }

    private func decrementAndFinishIfDone(activeCount: OSAllocatedUnfairLock<Int>, continuation: AsyncStream<String>.Continuation) {
        let remaining = activeCount.withLock { count -> Int in
            count -= 1
            return count
        }
        if remaining == 0 {
            continuation.finish()
        }
    }
}
