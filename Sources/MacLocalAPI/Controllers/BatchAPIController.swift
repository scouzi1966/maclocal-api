import Vapor
import Foundation

struct BatchAPIController: RouteCollection {
    private let service: any MLXChatServing
    private let store: BatchStore
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
        store: BatchStore,
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
        self.store = store
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

        // File endpoints
        v1.on(.POST, "files", body: .collect(maxSize: "100mb"), use: uploadFile)
        v1.get("files", ":file_id", use: getFile)
        v1.get("files", ":file_id", "content", use: getFileContent)
        v1.delete("files", ":file_id", use: deleteFile)

        // Batch endpoints
        v1.post("batches", use: createBatch)
        v1.get("batches", ":batch_id", use: getBatch)
        v1.get("batches", use: listBatches)
        v1.post("batches", ":batch_id", "cancel", use: cancelBatch)
    }

    // MARK: - File Endpoints

    func uploadFile(req: Request) async throws -> FileObject {
        guard let body = req.body.data else {
            throw Abort(.badRequest, reason: "No file data provided")
        }

        // Parse multipart: extract file and purpose
        let purpose = try? req.content.get(String.self, at: "purpose")
        guard purpose == "batch" else {
            throw Abort(.badRequest, reason: "Only purpose='batch' is supported")
        }

        // Try multipart parsing
        if let file = try? req.content.get(Vapor.File.self, at: "file") {
            let data = Data(buffer: file.data)
            return await store.storeFile(filename: file.filename, purpose: "batch", data: data)
        }

        // Fallback: treat raw body as file data
        let data = Data(buffer: body)
        return await store.storeFile(filename: "upload.jsonl", purpose: "batch", data: data)
    }

    func getFile(req: Request) async throws -> FileObject {
        guard let fileId = req.parameters.get("file_id") else {
            throw Abort(.badRequest, reason: "Missing file_id")
        }
        guard let file = await store.getFile(fileId) else {
            throw Abort(.notFound, reason: "File not found: \(fileId)")
        }
        return file
    }

    func getFileContent(req: Request) async throws -> Response {
        guard let fileId = req.parameters.get("file_id") else {
            throw Abort(.badRequest, reason: "Missing file_id")
        }
        guard let data = await store.getFileData(fileId) else {
            throw Abort(.notFound, reason: "File not found: \(fileId)")
        }
        let response = Response(status: .ok, body: .init(data: data))
        response.headers.contentType = .init(type: "application", subType: "jsonl")
        return response
    }

    func deleteFile(req: Request) async throws -> FileDeleteResponse {
        guard let fileId = req.parameters.get("file_id") else {
            throw Abort(.badRequest, reason: "Missing file_id")
        }
        guard await store.deleteFile(fileId) else {
            throw Abort(.notFound, reason: "File not found: \(fileId)")
        }
        return FileDeleteResponse(id: fileId)
    }

    // MARK: - Batch Endpoints

    func createBatch(req: Request) async throws -> BatchObject {
        let createReq = try req.content.decode(BatchCreateRequest.self)

        guard createReq.endpoint == "/v1/chat/completions" else {
            throw Abort(.badRequest, reason: "Only /v1/chat/completions endpoint is supported")
        }

        // Get input file
        guard let fileData = await store.getFileData(createReq.inputFileId) else {
            throw Abort(.notFound, reason: "Input file not found: \(createReq.inputFileId)")
        }

        // Parse JSONL
        guard let content = String(data: fileData, encoding: .utf8) else {
            throw Abort(.badRequest, reason: "Input file is not valid UTF-8")
        }

        let decoder = JSONDecoder()
        var inputLines: [BatchInputLine] = []
        for (lineIndex, line) in content.split(separator: "\n").enumerated() where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            guard let lineData = line.data(using: .utf8) else { continue }
            let parsed = try decoder.decode(BatchInputLine.self, from: lineData)
            // Validate per-line method and url per OpenAI Batch API spec
            guard parsed.method == "POST" else {
                throw Abort(.badRequest, reason: "Line \(lineIndex + 1): method must be POST, got \(parsed.method)")
            }
            guard parsed.url == "/v1/chat/completions" else {
                throw Abort(.badRequest, reason: "Line \(lineIndex + 1): url must be /v1/chat/completions, got \(parsed.url)")
            }
            inputLines.append(parsed)
        }

        guard !inputLines.isEmpty else {
            throw Abort(.badRequest, reason: "Input file contains no valid request lines")
        }
        guard inputLines.count <= 64 else {
            throw Abort(.badRequest, reason: "Batch size exceeds maximum of 64 requests")
        }

        // Check for duplicate custom_ids
        let ids = inputLines.map(\.customId)
        guard Set(ids).count == ids.count else {
            throw Abort(.badRequest, reason: "Duplicate custom_id values in input file")
        }

        // Create batch
        let batchId = await store.createBatch(
            inputFileId: createReq.inputFileId,
            endpoint: createReq.endpoint,
            totalRequests: inputLines.count
        )

        // Auto-promote to batch mode
        do {
            try await service.ensureBatchMode(concurrency: inputLines.count)
        } catch {
            await store.markBatchFailed(batchId, error: BatchError(message: error.localizedDescription, type: "server_error"))
            guard let obj = await store.getBatch(batchId) else {
                throw Abort(.internalServerError)
            }
            return obj
        }

        // Mark in_progress and dispatch
        await store.markBatchInProgress(batchId)

        // Dispatch all requests in background.
        // Each request individually reserves a slot via tryReserveSlot() in processOneRequest.
        // This allows partial batch execution — some requests may get 503 while others succeed,
        // which is acceptable per OpenAI batch semantics (per-request errors don't fail the batch).
        Task {
            await self.dispatchBatchRequests(batchId: batchId, requests: inputLines)
            service.releaseBatchReference()
        }

        guard let obj = await store.getBatch(batchId) else {
            throw Abort(.internalServerError)
        }
        return obj
    }

    func getBatch(req: Request) async throws -> BatchObject {
        guard let batchId = req.parameters.get("batch_id") else {
            throw Abort(.badRequest, reason: "Missing batch_id")
        }
        guard let batch = await store.getBatch(batchId) else {
            throw Abort(.notFound, reason: "Batch not found: \(batchId)")
        }
        return batch
    }

    func listBatches(req: Request) async throws -> BatchListResponse {
        BatchListResponse(data: await store.listBatches())
    }

    func cancelBatch(req: Request) async throws -> BatchObject {
        guard let batchId = req.parameters.get("batch_id") else {
            throw Abort(.badRequest, reason: "Missing batch_id")
        }
        guard let batch = await store.getBatch(batchId) else {
            throw Abort(.notFound, reason: "Batch not found: \(batchId)")
        }
        guard batch.status == "in_progress" else {
            throw Abort(.badRequest, reason: "Batch is not in_progress (status: \(batch.status))")
        }

        await store.markBatchCancelling(batchId)

        // Actually cancel in-flight scheduler slots
        let slotIds = await store.getSlotIds(batchId)
        if !slotIds.isEmpty {
            await service.cancelBatchSlots(ids: Set(slotIds))
        }

        await store.markBatchCancelled(batchId)

        guard let updated = await store.getBatch(batchId) else {
            throw Abort(.internalServerError)
        }
        return updated
    }

    // MARK: - Dispatch Logic

    private func dispatchBatchRequests(batchId: String, requests: [BatchInputLine]) async {
        await withTaskGroup(of: Void.self) { group in
            for inputLine in requests {
                group.addTask {
                    await self.processOneRequest(batchId: batchId, inputLine: inputLine)
                }
            }
        }
    }

    private func processOneRequest(batchId: String, inputLine: BatchInputLine) async {
        let requestId = "req_\(UUID().uuidString.lowercased().prefix(12))"
        let resultId = "batch_req_\(UUID().uuidString.lowercased().prefix(12))"
        let chatReq = inputLine.body

        do {
            // Reserve slot
            guard service.tryReserveSlot() else {
                let result = BatchResultLine(
                    id: resultId, customId: inputLine.customId,
                    response: nil,
                    error: BatchError(message: "Server at capacity", type: "server_error")
                )
                await store.recordResult(batchId, result: result)
                return
            }
            defer { service.releaseSlot() }

            let effectiveModel = service.normalizeModel(chatReq.model ?? modelID)
            let effectiveMaxTokens = chatReq.effectiveMaxTokens ?? maxTokens ?? Int.max

            // Use generateStreaming + StreamCollector for full post-processing
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

            // Determine if model supports thinking
            let extractThinking = streamResult.thinkStartTag != nil

            let collected = try await StreamCollector.collect(
                from: streamResult,
                extractThinking: extractThinking,
                thinkStartTag: streamResult.thinkStartTag ?? "<think>",
                thinkEndTag: streamResult.thinkEndTag ?? "</think>",
                maxTokens: effectiveMaxTokens
            )

            let choiceLogprobs = StreamCollector.buildChoiceLogprobs(collected.logprobs)

            // Build response with full post-processing
            let response: ChatCompletionResponse
            if let toolCalls = collected.toolCalls, !toolCalls.isEmpty {
                response = ChatCompletionResponse(
                    model: effectiveModel,
                    toolCalls: toolCalls,
                    logprobs: choiceLogprobs,
                    promptTokens: collected.promptTokens,
                    completionTokens: collected.completionTokens,
                    cachedTokens: collected.cachedTokens > 0 ? collected.cachedTokens : nil
                )
            } else {
                response = ChatCompletionResponse(
                    model: effectiveModel,
                    content: collected.content ?? "",
                    reasoningContent: collected.reasoningContent,
                    logprobs: choiceLogprobs,
                    finishReason: collected.finishReason,
                    promptTokens: collected.promptTokens,
                    completionTokens: collected.completionTokens,
                    cachedTokens: collected.cachedTokens > 0 ? collected.cachedTokens : nil
                )
            }

            let result = BatchResultLine(
                id: resultId, customId: inputLine.customId,
                response: BatchResultResponse(
                    statusCode: 200,
                    requestId: requestId,
                    body: response
                ),
                error: nil
            )
            await store.recordResult(batchId, result: result)

        } catch {
            let result = BatchResultLine(
                id: resultId, customId: inputLine.customId,
                response: nil,
                error: BatchError(message: error.localizedDescription, type: "server_error")
            )
            await store.recordResult(batchId, result: result)
        }
    }
}
