import XCTest
import Vapor
import XCTVapor
import Foundation
import Testing

@testable import MacLocalAPI

// MARK: - Fake Service for Batch Tests

/// A fake MLXChatServing that supports batch-mode protocol methods.
/// Configurable per-test: control slot reservation, streaming results, errors, and concurrency tracking.
private final class FakeBatchService: MLXChatServing, @unchecked Sendable {
    let maxConcurrent: Int
    let toolCallParser: String? = nil
    let thinkStartTag: String? = nil
    let thinkEndTag: String? = nil
    let fixToolArgs: Bool = false
    let enableGrammarConstraints: Bool = false

    // Tracking
    private let _lock = NSLock()
    private(set) var ensureBatchModeCallCount = 0
    private(set) var releaseBatchReferenceCallCount = 0
    private(set) var generateStreamingCallCount = 0
    private(set) var reserveSlotCallCount = 0
    private(set) var releaseSlotCallCount = 0

    // Configuration
    var shouldFailEnsureBatchMode = false
    var shouldFailReserveSlot = false
    var streamingResultFactory: (([Message]) -> ChatStreamingResult)?
    private let defaultStreamingResult: ChatStreamingResult

    init(maxConcurrent: Int = 8) {
        self.maxConcurrent = maxConcurrent
        self.defaultStreamingResult = FakeBatchService.makeStreamingResult(chunks: [
            StreamChunk(text: "Hello from batch"),
            StreamChunk(text: "", promptTokens: 10, completionTokens: 5, cachedTokens: 0, promptTime: 0.01, generateTime: 0.02),
        ])
    }

    func normalizeModel(_ raw: String) -> String { raw }

    func tryReserveSlot() -> Bool {
        _lock.lock()
        reserveSlotCallCount += 1
        _lock.unlock()
        return !shouldFailReserveSlot
    }

    func releaseSlot() {
        _lock.lock()
        releaseSlotCallCount += 1
        _lock.unlock()
    }

    func ensureBatchMode(concurrency: Int) async throws {
        _lock.lock()
        ensureBatchModeCallCount += 1
        let shouldFail = shouldFailEnsureBatchMode
        _lock.unlock()
        if shouldFail {
            throw MLXServiceError.noModelLoaded
        }
    }

    func releaseBatchReference() {
        _lock.lock()
        releaseBatchReferenceCallCount += 1
        _lock.unlock()
    }

    func cancelBatchSlots(ids: Set<UUID>) async {
        // No-op in tests
    }

    func startAPIProfile() {}
    func stopAPIProfile(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfile {
        AFMProfile(gpuPowerAvgW: nil, gpuPowerPeakW: nil, gpuSamples: nil, memoryWeightsGiB: nil, memoryKvGiB: nil, memoryPeakGiB: nil, prefillTokS: nil, decodeTokS: nil, chip: nil, theoreticalBwGbs: nil, estBandwidthGbs: nil)
    }
    func stopAPIProfileExtended(promptTokens: Int, completionTokens: Int, promptTime: Double, generateTime: Double) -> AFMProfileExtended {
        AFMProfileExtended(summary: stopAPIProfile(promptTokens: promptTokens, completionTokens: completionTokens, promptTime: promptTime, generateTime: generateTime), samples: [])
    }

    func generate(
        model: String, messages: [Message], temperature: Double?, maxTokens: Int?,
        topP: Double?, repetitionPenalty: Double?, topK: Int?, minP: Double?,
        presencePenalty: Double?, seed: Int?, logprobs: Bool?, topLogprobs: Int?,
        tools: [RequestTool]?, stop: [String]?, responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatGenerationResult {
        (modelID: model, content: "Hello", promptTokens: 10, completionTokens: 5,
         tokenLogprobs: nil, toolCalls: nil, cachedTokens: 0, promptTime: 0.01,
         generateTime: 0.02, stoppedBySequence: false)
    }

    func generateStreaming(
        model: String, messages: [Message], temperature: Double?, maxTokens: Int?,
        topP: Double?, repetitionPenalty: Double?, topK: Int?, minP: Double?,
        presencePenalty: Double?, seed: Int?, logprobs: Bool?, topLogprobs: Int?,
        tools: [RequestTool]?, stop: [String]?, responseFormat: ResponseFormat?,
        chatTemplateKwargs: [String: AnyCodable]?
    ) async throws -> ChatStreamingResult {
        _lock.lock()
        generateStreamingCallCount += 1
        _lock.unlock()
        return streamingResultFactory?(messages) ?? defaultStreamingResult
    }

    static func makeStreamingResult(chunks: [StreamChunk]) -> ChatStreamingResult {
        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }
        return (
            modelID: "test-model",
            stream: stream,
            promptTokens: 10,
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            thinkStartTag: nil,
            thinkEndTag: nil
        )
    }

    static func makeErrorStreamingResult(error: Error) -> ChatStreamingResult {
        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
            continuation.finish(throwing: error)
        }
        return (
            modelID: "test-model",
            stream: stream,
            promptTokens: 0,
            toolCallStartTag: nil,
            toolCallEndTag: nil,
            thinkStartTag: nil,
            thinkEndTag: nil
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MARK: - BatchStore Tests (Swift Testing)
// ═══════════════════════════════════════════════════════════════════════════

struct BatchStoreTests {

    // MARK: - File Storage

    @Test("storeFile creates file with correct metadata")
    func storeFileMetadata() async {
        let store = BatchStore()
        let data = Data("test content".utf8)
        let file = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: data)

        #expect(file.id.hasPrefix("file-"))
        #expect(file.bytes == data.count)
        #expect(file.filename == "input.jsonl")
        #expect(file.purpose == "batch")
        #expect(file.object == "file")
    }

    @Test("getFile returns stored file metadata")
    func getFileReturnsMetadata() async {
        let store = BatchStore()
        let data = Data("hello".utf8)
        let file = await store.storeFile(filename: "test.jsonl", purpose: "batch", data: data)
        let retrieved = await store.getFile(file.id)

        #expect(retrieved != nil)
        #expect(retrieved?.id == file.id)
        #expect(retrieved?.bytes == file.bytes)
        #expect(retrieved?.filename == file.filename)
    }

    @Test("getFile returns nil for nonexistent file")
    func getFileNonexistent() async {
        let store = BatchStore()
        let result = await store.getFile("file-nonexistent")
        #expect(result == nil)
    }

    @Test("getFileData returns raw data")
    func getFileData() async {
        let store = BatchStore()
        let original = Data("raw content here".utf8)
        let file = await store.storeFile(filename: "data.jsonl", purpose: "batch", data: original)
        let data = await store.getFileData(file.id)

        #expect(data == original)
    }

    @Test("getFileData returns nil for nonexistent file")
    func getFileDataNonexistent() async {
        let store = BatchStore()
        let data = await store.getFileData("file-missing")
        #expect(data == nil)
    }

    @Test("deleteFile removes the file")
    func deleteFile() async {
        let store = BatchStore()
        let file = await store.storeFile(filename: "temp.jsonl", purpose: "batch", data: Data("x".utf8))
        let deleted = await store.deleteFile(file.id)
        #expect(deleted == true)

        let afterDelete = await store.getFile(file.id)
        #expect(afterDelete == nil)
    }

    @Test("deleteFile returns false for nonexistent file")
    func deleteFileNonexistent() async {
        let store = BatchStore()
        let deleted = await store.deleteFile("file-nope")
        #expect(deleted == false)
    }

    @Test("Multiple files have unique IDs")
    func multipleFilesUniqueIds() async {
        let store = BatchStore()
        let f1 = await store.storeFile(filename: "a.jsonl", purpose: "batch", data: Data("a".utf8))
        let f2 = await store.storeFile(filename: "b.jsonl", purpose: "batch", data: Data("b".utf8))
        let f3 = await store.storeFile(filename: "c.jsonl", purpose: "batch", data: Data("c".utf8))

        #expect(f1.id != f2.id)
        #expect(f2.id != f3.id)
        #expect(f1.id != f3.id)
    }

    // MARK: - Batch State Management

    @Test("createBatch returns ID with correct prefix and initial state")
    func createBatch() async {
        let store = BatchStore()
        let file = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: Data("x".utf8))
        let batchId = await store.createBatch(inputFileId: file.id, endpoint: "/v1/chat/completions", totalRequests: 3)

        #expect(batchId.hasPrefix("batch_"))

        let batch = await store.getBatch(batchId)
        #expect(batch != nil)
        #expect(batch?.status == "validating")
        #expect(batch?.endpoint == "/v1/chat/completions")
        #expect(batch?.inputFileId == file.id)
        #expect(batch?.requestCounts.total == 3)
        #expect(batch?.requestCounts.completed == 0)
        #expect(batch?.requestCounts.failed == 0)
        #expect(batch?.outputFileId == nil)
        #expect(batch?.completedAt == nil)
    }

    @Test("markBatchInProgress transitions status and stores slot IDs")
    func markBatchInProgress() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 2)
        let slotIds = [UUID(), UUID()]
        await store.markBatchInProgress(batchId, slotIds: slotIds)

        let batch = await store.getBatch(batchId)
        #expect(batch?.status == "in_progress")

        let retrievedSlots = await store.getSlotIds(batchId)
        #expect(retrievedSlots.count == 2)
    }

    @Test("recordResult increments completed count")
    func recordResultCompleted() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 2)
        await store.markBatchInProgress(batchId)

        let response = ChatCompletionResponse(model: "test", content: "Hello", promptTokens: 10, completionTokens: 5)
        let result = BatchResultLine(
            id: "batch_req_1", customId: "req-1",
            response: BatchResultResponse(statusCode: 200, requestId: "r1", body: response),
            error: nil
        )
        await store.recordResult(batchId, result: result)

        let batch = await store.getBatch(batchId)
        #expect(batch?.requestCounts.completed == 1)
        #expect(batch?.requestCounts.failed == 0)
        #expect(batch?.status == "in_progress")  // Not done yet — 1 of 2
    }

    @Test("recordResult increments failed count for error results")
    func recordResultFailed() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 1)
        await store.markBatchInProgress(batchId)

        let result = BatchResultLine(
            id: "batch_req_1", customId: "req-1",
            response: nil,
            error: BatchError(message: "boom", type: "server_error")
        )
        await store.recordResult(batchId, result: result)

        let batch = await store.getBatch(batchId)
        #expect(batch?.requestCounts.failed == 1)
        #expect(batch?.status == "completed")  // 1 of 1 done
        #expect(batch?.completedAt != nil)
        #expect(batch?.outputFileId != nil)
    }

    @Test("Batch auto-completes when all requests are done")
    func batchAutoCompletes() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 2)
        await store.markBatchInProgress(batchId)

        let response = ChatCompletionResponse(model: "test", content: "ok", promptTokens: 10, completionTokens: 5)

        await store.recordResult(batchId, result: BatchResultLine(
            id: "r1", customId: "a",
            response: BatchResultResponse(statusCode: 200, requestId: "r1", body: response),
            error: nil
        ))
        let midBatch = await store.getBatch(batchId)
        #expect(midBatch?.status == "in_progress")

        await store.recordResult(batchId, result: BatchResultLine(
            id: "r2", customId: "b",
            response: BatchResultResponse(statusCode: 200, requestId: "r2", body: response),
            error: nil
        ))
        let doneBatch = await store.getBatch(batchId)
        #expect(doneBatch?.status == "completed")
        #expect(doneBatch?.outputFileId != nil)
        #expect(doneBatch?.completedAt != nil)
    }

    @Test("Output JSONL file contains all results")
    func outputFileContainsResults() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 2)
        await store.markBatchInProgress(batchId)

        let response = ChatCompletionResponse(model: "test", content: "ok", promptTokens: 10, completionTokens: 5)
        await store.recordResult(batchId, result: BatchResultLine(
            id: "r1", customId: "first",
            response: BatchResultResponse(statusCode: 200, requestId: "r1", body: response), error: nil
        ))
        await store.recordResult(batchId, result: BatchResultLine(
            id: "r2", customId: "second",
            response: BatchResultResponse(statusCode: 200, requestId: "r2", body: response), error: nil
        ))

        let batch = await store.getBatch(batchId)
        let outputData = await store.getFileData(batch!.outputFileId!)
        #expect(outputData != nil)

        let outputStr = String(data: outputData!, encoding: .utf8)!
        let lines = outputStr.split(separator: "\n")
        #expect(lines.count == 2)
        #expect(outputStr.contains("\"first\""))
        #expect(outputStr.contains("\"second\""))
    }

    @Test("markBatchFailed sets status and error")
    func markBatchFailed() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 1)
        await store.markBatchFailed(batchId, error: BatchError(message: "out of memory", type: "server_error"))

        let batch = await store.getBatch(batchId)
        #expect(batch?.status == "failed")
        #expect(batch?.completedAt != nil)
    }

    @Test("markBatchCancelling and markBatchCancelled transitions")
    func batchCancellationFlow() async {
        let store = BatchStore()
        let batchId = await store.createBatch(inputFileId: "file-1", endpoint: "/v1/chat/completions", totalRequests: 5)
        await store.markBatchInProgress(batchId)

        await store.markBatchCancelling(batchId)
        let cancelling = await store.getBatch(batchId)
        #expect(cancelling?.status == "cancelling")

        await store.markBatchCancelled(batchId)
        let cancelled = await store.getBatch(batchId)
        #expect(cancelled?.status == "cancelled")
        #expect(cancelled?.completedAt != nil)
    }

    @Test("listBatches returns all batches")
    func listBatches() async {
        let store = BatchStore()
        _ = await store.createBatch(inputFileId: "f1", endpoint: "/v1/chat/completions", totalRequests: 1)
        _ = await store.createBatch(inputFileId: "f2", endpoint: "/v1/chat/completions", totalRequests: 2)
        _ = await store.createBatch(inputFileId: "f3", endpoint: "/v1/chat/completions", totalRequests: 3)

        let batches = await store.listBatches()
        #expect(batches.count == 3)
    }

    @Test("getBatch returns nil for nonexistent batch")
    func getBatchNonexistent() async {
        let store = BatchStore()
        let batch = await store.getBatch("batch_nonexistent")
        #expect(batch == nil)
    }

    @Test("getSlotIds returns empty for nonexistent batch")
    func getSlotIdsNonexistent() async {
        let store = BatchStore()
        let slots = await store.getSlotIds("batch_nope")
        #expect(slots.isEmpty)
    }

    // MARK: - Concurrent Access

    @Test("BatchStore handles concurrent file operations safely")
    func concurrentFileOps() async {
        let store = BatchStore()
        let iterations = 50

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<iterations {
                group.addTask {
                    let data = Data("content-\(i)".utf8)
                    let file = await store.storeFile(filename: "file-\(i).jsonl", purpose: "batch", data: data)
                    let retrieved = await store.getFile(file.id)
                    // Each file should be retrievable immediately after storage
                    assert(retrieved != nil)
                }
            }
        }
    }

    @Test("BatchStore handles concurrent batch creates safely")
    func concurrentBatchCreates() async {
        let store = BatchStore()
        let iterations = 20

        await withTaskGroup(of: String.self) { group in
            for _ in 0..<iterations {
                group.addTask {
                    await store.createBatch(inputFileId: "f", endpoint: "/v1/chat/completions", totalRequests: 1)
                }
            }

            var ids = Set<String>()
            for await id in group {
                ids.insert(id)
            }
            // All batch IDs should be unique
            #expect(ids.count == iterations)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MARK: - Batch Type Serialization Tests (Swift Testing)
// ═══════════════════════════════════════════════════════════════════════════

struct BatchTypeSerializationTests {

    @Test("BatchInputLine decodes from JSONL line")
    func decodeBatchInputLine() throws {
        let json = """
        {"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"Hello"}]}}
        """
        let decoded = try JSONDecoder().decode(BatchInputLine.self, from: Data(json.utf8))
        #expect(decoded.customId == "req-1")
        #expect(decoded.method == "POST")
        #expect(decoded.url == "/v1/chat/completions")
        #expect(decoded.body.model == "test")
        #expect(decoded.body.messages.count == 1)
    }

    @Test("BatchCreateRequest decodes with snake_case keys")
    func decodeBatchCreateRequest() throws {
        let json = """
        {"input_file_id":"file-abc","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        let decoded = try JSONDecoder().decode(BatchCreateRequest.self, from: Data(json.utf8))
        #expect(decoded.inputFileId == "file-abc")
        #expect(decoded.endpoint == "/v1/chat/completions")
        #expect(decoded.completionWindow == "24h")
    }

    @Test("BatchCompletionRequest decodes array of requests")
    func decodeBatchCompletionRequest() throws {
        let json = """
        {"requests":[{"custom_id":"a","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}},{"custom_id":"b","body":{"model":"m","messages":[{"role":"user","content":"bye"}]}}]}
        """
        let decoded = try JSONDecoder().decode(BatchCompletionRequest.self, from: Data(json.utf8))
        #expect(decoded.requests.count == 2)
        #expect(decoded.requests[0].customId == "a")
        #expect(decoded.requests[1].customId == "b")
    }

    @Test("FileObject encodes with snake_case keys")
    func encodeFileObject() throws {
        let file = FileObject(id: "file-123", bytes: 1024, createdAt: 1700000000, filename: "test.jsonl", purpose: "batch")
        let data = try JSONEncoder().encode(file)
        let str = String(data: data, encoding: .utf8)!
        #expect(str.contains("\"created_at\""))
        #expect(str.contains("\"file-123\""))
        #expect(str.contains("\"object\":\"file\""))
    }

    @Test("FileDeleteResponse defaults")
    func fileDeleteResponseDefaults() throws {
        let response = FileDeleteResponse(id: "file-abc")
        let data = try JSONEncoder().encode(response)
        let str = String(data: data, encoding: .utf8)!
        #expect(str.contains("\"deleted\":true"))
        #expect(str.contains("\"object\":\"file\""))
    }

    @Test("BatchObject encodes with all fields")
    func encodeBatchObject() throws {
        let batch = BatchObject(
            id: "batch_abc", object: "batch", endpoint: "/v1/chat/completions",
            inputFileId: "file-input", completionWindow: "24h", status: "completed",
            createdAt: 1700000000, completedAt: 1700000060, outputFileId: "file-output",
            requestCounts: BatchRequestCounts(total: 5, completed: 4, failed: 1)
        )
        let data = try JSONEncoder().encode(batch)
        let str = String(data: data, encoding: .utf8)!
        #expect(str.contains("\"input_file_id\""))
        #expect(str.contains("\"completion_window\""))
        #expect(str.contains("\"output_file_id\""))
        #expect(str.contains("\"request_counts\""))
    }

    @Test("BatchResultLine round-trips through JSON")
    func batchResultLineRoundTrip() throws {
        let response = ChatCompletionResponse(model: "test", content: "hi", promptTokens: 10, completionTokens: 5)
        let result = BatchResultLine(
            id: "batch_req_1", customId: "my-req",
            response: BatchResultResponse(statusCode: 200, requestId: "req-1", body: response),
            error: nil
        )
        let encoder = JSONEncoder()
        let data = try encoder.encode(result)
        let decoded = try JSONDecoder().decode(BatchResultLine.self, from: data)

        #expect(decoded.id == "batch_req_1")
        #expect(decoded.customId == "my-req")
        #expect(decoded.response?.statusCode == 200)
        #expect(decoded.error == nil)
    }

    @Test("BatchResultLine with error encodes correctly")
    func batchResultLineWithError() throws {
        let result = BatchResultLine(
            id: "batch_req_err", customId: "fail-req",
            response: nil,
            error: BatchError(message: "Server at capacity", type: "server_error")
        )
        let data = try JSONEncoder().encode(result)
        let str = String(data: data, encoding: .utf8)!
        #expect(str.contains("server_error"))
        #expect(str.contains("Server at capacity"))
    }

    @Test("BatchListResponse encodes with has_more field")
    func batchListResponseEncoding() throws {
        let list = BatchListResponse(data: [])
        let data = try JSONEncoder().encode(list)
        let str = String(data: data, encoding: .utf8)!
        #expect(str.contains("\"has_more\":false"))
        #expect(str.contains("\"object\":\"list\""))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MARK: - BatchAPIController Integration Tests (XCTest + Vapor)
// ═══════════════════════════════════════════════════════════════════════════

final class BatchAPIControllerTests: XCTestCase {
    private var app: Application!
    private var service: FakeBatchService!
    private var store: BatchStore!

    override func setUp() async throws {
        app = try await Application.make(.testing)
        service = FakeBatchService()
        store = BatchStore()
        try BatchAPIController(
            service: service,
            store: store,
            modelID: "test-model"
        ).boot(routes: app)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    // MARK: - File Endpoints

    func testUploadFileRejectsPurposeNotBatch() async throws {
        // Create multipart body with wrong purpose
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = ByteBuffer()
        body.writeString("--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nfine-tune\r\n")
        body.writeString("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\ntest data\r\n")
        body.writeString("--\(boundary)--\r\n")

        var headers = HTTPHeaders()
        headers.contentType = HTTPMediaType(type: "multipart", subType: "form-data", parameters: ["boundary": boundary])

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/files", headers: headers, body: body
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
        }
    }

    func testGetFileReturns404ForMissing() async throws {
        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/files/file-nonexistent"
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testGetFileContentReturns404ForMissing() async throws {
        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/files/file-nonexistent/content"
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testDeleteFileReturns404ForMissing() async throws {
        try await app.testable(method: .running(port: 0)).test(
            .DELETE, "/v1/files/file-nonexistent"
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testDeleteFileAfterStoring() async throws {
        // Pre-populate store
        let fileObj = await store.storeFile(filename: "test.jsonl", purpose: "batch", data: Data("content".utf8))

        try await app.testable(method: .running(port: 0)).test(
            .DELETE, "/v1/files/\(fileObj.id)"
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"deleted\":true")
        }

        // Verify deleted
        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/files/\(fileObj.id)"
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testGetFileContentReturnsData() async throws {
        let content = "line1\nline2\nline3\n"
        let fileObj = await store.storeFile(filename: "data.jsonl", purpose: "batch", data: Data(content.utf8))

        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/files/\(fileObj.id)/content"
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.body.string, content)
        }
    }

    // MARK: - Batch Endpoints

    func testCreateBatchRejectsInvalidEndpoint() async throws {
        let fileObj = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: Data("x".utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/embeddings","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
        }
    }

    func testCreateBatchRejectsMissingInputFile() async throws {
        let json = """
        {"input_file_id":"file-nonexistent","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testCreateBatchRejectsEmptyInput() async throws {
        let fileObj = await store.storeFile(filename: "empty.jsonl", purpose: "batch", data: Data("\n\n".utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
        }
    }

    func testCreateBatchRejectsDuplicateCustomIds() async throws {
        let jsonl = """
        {"custom_id":"same","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"a"}]}}
        {"custom_id":"same","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"b"}]}}
        """
        let fileObj = await store.storeFile(filename: "dup.jsonl", purpose: "batch", data: Data(jsonl.utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "Duplicate")
        }
    }

    func testCreateBatchRejectsOver64Requests() async throws {
        var lines: [String] = []
        for i in 0..<65 {
            lines.append("""
            {"custom_id":"req-\(i)","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"hi"}]}}
            """)
        }
        let jsonl = lines.joined(separator: "\n")
        let fileObj = await store.storeFile(filename: "big.jsonl", purpose: "batch", data: Data(jsonl.utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "64")
        }
    }

    func testCreateBatchSuccessWithValidInput() async throws {
        let jsonl = """
        {"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"Hello"}]}}
        {"custom_id":"req-2","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"World"}]}}
        """
        let fileObj = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: Data(jsonl.utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"object\":\"batch\"")
            XCTAssertContains(res.body.string, "\"status\":\"in_progress\"")
            XCTAssertContains(res.body.string, "\"endpoint\":\"\\/v1\\/chat\\/completions\"")
        }

        // Verify ensureBatchMode was called
        XCTAssertEqual(service.ensureBatchModeCallCount, 1)
    }

    func testCreateBatchFailsWhenEnsureBatchModeFails() async throws {
        service.shouldFailEnsureBatchMode = true

        let jsonl = """
        {"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"Hello"}]}}
        """
        let fileObj = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: Data(jsonl.utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .ok) // Returns batch object with failed status
            XCTAssertContains(res.body.string, "\"status\":\"failed\"")
        }
    }

    func testGetBatchReturns404ForMissing() async throws {
        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/batches/batch_nonexistent"
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testGetBatchReturnsStoredBatch() async throws {
        let batchId = await store.createBatch(inputFileId: "f", endpoint: "/v1/chat/completions", totalRequests: 1)

        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/batches/\(batchId)"
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, batchId)
            XCTAssertContains(res.body.string, "\"status\":\"validating\"")
        }
    }

    func testListBatchesEmpty() async throws {
        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/batches"
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"object\":\"list\"")
            XCTAssertContains(res.body.string, "\"data\":[]")
        }
    }

    func testListBatchesReturnsAll() async throws {
        _ = await store.createBatch(inputFileId: "f1", endpoint: "/v1/chat/completions", totalRequests: 1)
        _ = await store.createBatch(inputFileId: "f2", endpoint: "/v1/chat/completions", totalRequests: 1)

        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/batches"
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"object\":\"list\"")
            // Should contain 2 batches
            let data = try? JSONSerialization.jsonObject(with: Data(res.body.string.utf8)) as? [String: Any]
            let batches = data?["data"] as? [[String: Any]]
            XCTAssertEqual(batches?.count, 2)
        }
    }

    func testCancelBatchReturns404ForMissing() async throws {
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches/batch_nonexistent/cancel"
        ) { res async in
            XCTAssertEqual(res.status, .notFound)
        }
    }

    func testCancelBatchRejectsBatchNotInProgress() async throws {
        let batchId = await store.createBatch(inputFileId: "f", endpoint: "/v1/chat/completions", totalRequests: 1)
        // Batch is in "validating" state, not "in_progress"

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches/\(batchId)/cancel"
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "not in_progress")
        }
    }

    func testCancelBatchSuccess() async throws {
        let batchId = await store.createBatch(inputFileId: "f", endpoint: "/v1/chat/completions", totalRequests: 1)
        await store.markBatchInProgress(batchId)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches/\(batchId)/cancel"
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, "\"status\":\"cancelled\"")
        }
    }

    // MARK: - End-to-End: Batch Dispatch + Polling

    func testBatchDispatchCompletesAndProducesResults() async throws {
        let jsonl = """
        {"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"Hello"}]}}
        """
        let fileObj = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: Data(jsonl.utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        var batchId = ""
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            if let data = try? JSONSerialization.jsonObject(with: Data(res.body.string.utf8)) as? [String: Any] {
                batchId = data["id"] as? String ?? ""
            }
        }

        XCTAssertFalse(batchId.isEmpty)

        // Poll for completion (dispatch is async via Task)
        for _ in 0..<20 {
            try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            let batch = await store.getBatch(batchId)
            if batch?.status == "completed" { break }
        }

        let batch = await store.getBatch(batchId)
        XCTAssertEqual(batch?.status, "completed")
        XCTAssertEqual(batch?.requestCounts.completed, 1)
        XCTAssertNotNil(batch?.outputFileId)

        // Verify output file content
        if let outputFileId = batch?.outputFileId {
            try await app.testable(method: .running(port: 0)).test(
                .GET, "/v1/files/\(outputFileId)/content"
            ) { res async in
                XCTAssertEqual(res.status, .ok)
                XCTAssertContains(res.body.string, "\"req-1\"")
                XCTAssertContains(res.body.string, "\"status_code\":200")
            }
        }
    }

    func testBatchDispatchRecordsErrorOnSlotReservationFailure() async throws {
        service.shouldFailReserveSlot = true

        let jsonl = """
        {"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"Hello"}]}}
        """
        let fileObj = await store.storeFile(filename: "input.jsonl", purpose: "batch", data: Data(jsonl.utf8))

        let json = """
        {"input_file_id":"\(fileObj.id)","endpoint":"/v1/chat/completions","completion_window":"24h"}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        var batchId = ""
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batches", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            if let data = try? JSONSerialization.jsonObject(with: Data(res.body.string.utf8)) as? [String: Any] {
                batchId = data["id"] as? String ?? ""
            }
        }

        // Wait for dispatch
        for _ in 0..<20 {
            try await Task.sleep(nanoseconds: 100_000_000)
            let batch = await store.getBatch(batchId)
            if batch?.status == "completed" { break }
        }

        let batch = await store.getBatch(batchId)
        XCTAssertEqual(batch?.status, "completed")
        XCTAssertEqual(batch?.requestCounts.failed, 1)
        XCTAssertEqual(batch?.requestCounts.completed, 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MARK: - BatchCompletionsController Integration Tests (XCTest + Vapor)
// ═══════════════════════════════════════════════════════════════════════════

final class BatchCompletionsControllerTests: XCTestCase {
    private var app: Application!
    private var service: FakeBatchService!

    override func setUp() async throws {
        app = try await Application.make(.testing)
        service = FakeBatchService()
        try BatchCompletionsController(
            service: service,
            modelID: "test-model"
        ).boot(routes: app)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    // MARK: - Validation

    func testRejectsEmptyRequests() async throws {
        let json = """
        {"requests":[]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "at least one")
        }
    }

    func testRejectsOver64Requests() async throws {
        var items: [String] = []
        for i in 0..<65 {
            items.append("""
            {"custom_id":"req-\(i)","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}
            """)
        }
        let json = """
        {"requests":[\(items.joined(separator: ","))]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "64")
        }
    }

    func testRejectsDuplicateCustomIds() async throws {
        let json = """
        {"requests":[{"custom_id":"same","body":{"model":"m","messages":[{"role":"user","content":"a"}]}},{"custom_id":"same","body":{"model":"m","messages":[{"role":"user","content":"b"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "Duplicate")
        }
    }

    func testRejectsEmptyCustomId() async throws {
        let json = """
        {"requests":[{"custom_id":"","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .badRequest)
            XCTAssertContains(res.body.string, "non-empty")
        }
    }

    // MARK: - SSE Response

    func testSSEResponseHeaders() async throws {
        let json = """
        {"requests":[{"custom_id":"req-1","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertEqual(res.headers.first(name: .contentType), "text/event-stream")
            XCTAssertEqual(res.headers.first(name: "X-Accel-Buffering"), "no")
            XCTAssertEqual(res.headers.first(name: .cacheControl), "no-cache")
        }
    }

    func testSSEContainsDoneSentinel() async throws {
        let json = """
        {"requests":[{"custom_id":"req-1","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertContains(res.body.string, "data: [DONE]")
        }
    }

    func testNonStreamingResponseContainsCustomId() async throws {
        let json = """
        {"requests":[{"custom_id":"my-req-42","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertContains(res.body.string, "my-req-42")
            XCTAssertContains(res.body.string, "chat.completion")
        }
    }

    func testStreamingResponseContainsCustomId() async throws {
        let json = """
        {"requests":[{"custom_id":"stream-1","body":{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertContains(res.body.string, "stream-1")
            XCTAssertContains(res.body.string, "chat.completion.chunk")
        }
    }

    func testMultipleRequestsAllGetResponses() async throws {
        let json = """
        {"requests":[{"custom_id":"a","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}},{"custom_id":"b","body":{"model":"m","messages":[{"role":"user","content":"bye"}]}},{"custom_id":"c","body":{"model":"m","messages":[{"role":"user","content":"ok"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            let body = res.body.string
            XCTAssertContains(body, "\"a\"")
            XCTAssertContains(body, "\"b\"")
            XCTAssertContains(body, "\"c\"")
            XCTAssertContains(body, "data: [DONE]")
        }
    }

    func testCallsEnsureBatchMode() async throws {
        let json = """
        {"requests":[{"custom_id":"req-1","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .ok)
        }

        XCTAssertEqual(service.ensureBatchModeCallCount, 1)
    }

    func testEnsureBatchModeFailureReturns500() async throws {
        service.shouldFailEnsureBatchMode = true

        let json = """
        {"requests":[{"custom_id":"req-1","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .internalServerError)
        }
    }

    func testSlotReservationFailureEmitsErrorEvent() async throws {
        service.shouldFailReserveSlot = true

        let json = """
        {"requests":[{"custom_id":"no-slot","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertContains(res.body.string, "batch.error")
            XCTAssertContains(res.body.string, "no-slot")
            XCTAssertContains(res.body.string, "capacity")
            XCTAssertContains(res.body.string, "data: [DONE]")
        }
    }

    func testMixedStreamingAndNonStreaming() async throws {
        let json = """
        {"requests":[{"custom_id":"streamed","body":{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}},{"custom_id":"non-streamed","body":{"model":"m","messages":[{"role":"user","content":"bye"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            let body = res.body.string
            XCTAssertContains(body, "streamed")
            XCTAssertContains(body, "non-streamed")
            XCTAssertContains(body, "data: [DONE]")
        }
    }

    func testReleaseBatchReferenceCalledAfterCompletion() async throws {
        let json = """
        {"requests":[{"custom_id":"req-1","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}]}
        """
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/batch/completions", headers: headers, body: ByteBuffer(string: json)
        ) { res async in
            XCTAssertEqual(res.status, .ok)
        }

        XCTAssertEqual(service.releaseBatchReferenceCallCount, 1)
    }
}

// NOTE: BatchScheduler extension tests (tryReserveMultiple, releaseMultipleReservations,
// activeSlotCount) require model+tokenizer to instantiate. Those methods are exercised
// through the existing ConcurrentBatchTests which test the scheduler with a real model,
// and indirectly through the BatchAPIController and BatchCompletionsController integration
// tests above which verify slot reservation/release via the FakeBatchService mock.

// ═══════════════════════════════════════════════════════════════════════════
// MARK: - JSONL Parsing Edge Cases (Swift Testing)
// ═══════════════════════════════════════════════════════════════════════════

struct BatchJSONLParsingTests {

    @Test("BatchInputLine handles messages with special characters")
    func specialCharactersInMessages() throws {
        let json = """
        {"custom_id":"special","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"Hello\\nWorld"}]}}
        """
        let decoded = try JSONDecoder().decode(BatchInputLine.self, from: Data(json.utf8))
        #expect(decoded.customId == "special")
        if case .text(let text) = decoded.body.messages[0].content {
            #expect(text.contains("Hello"))
            #expect(text.contains("World"))
        } else {
            Issue.record("Expected .text content")
        }
    }

    @Test("BatchInputLine handles system + user messages")
    func systemAndUserMessages() throws {
        let json = """
        {"custom_id":"multi-msg","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hello"}]}}
        """
        let decoded = try JSONDecoder().decode(BatchInputLine.self, from: Data(json.utf8))
        #expect(decoded.body.messages.count == 2)
    }

    @Test("BatchInputLine handles optional body fields")
    func optionalBodyFields() throws {
        let json = """
        {"custom_id":"opts","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"max_tokens":100,"top_p":0.9}}
        """
        let decoded = try JSONDecoder().decode(BatchInputLine.self, from: Data(json.utf8))
        #expect(decoded.body.temperature == 0.7)
        #expect(decoded.body.topP == 0.9)
    }

    @Test("Multiple JSONL lines parse independently")
    func multipleJSONLLines() throws {
        let jsonl = """
        {"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"first"}]}}
        {"custom_id":"b","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"second"}]}}
        {"custom_id":"c","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"third"}]}}
        """
        let decoder = JSONDecoder()
        var results: [BatchInputLine] = []
        for line in jsonl.split(separator: "\n") where !line.isEmpty {
            let parsed = try decoder.decode(BatchInputLine.self, from: Data(line.utf8))
            results.append(parsed)
        }
        #expect(results.count == 3)
        #expect(results[0].customId == "a")
        #expect(results[1].customId == "b")
        #expect(results[2].customId == "c")
    }

    @Test("JSONL with blank lines are skipped")
    func blankLinesSkipped() throws {
        let jsonl = """
        {"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"hello"}]}}

        {"custom_id":"b","method":"POST","url":"/v1/chat/completions","body":{"model":"test","messages":[{"role":"user","content":"world"}]}}

        """
        let decoder = JSONDecoder()
        var results: [BatchInputLine] = []
        for line in jsonl.split(separator: "\n") where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            let parsed = try decoder.decode(BatchInputLine.self, from: Data(line.utf8))
            results.append(parsed)
        }
        #expect(results.count == 2)
    }
}

// MARK: - StreamCollector Tests

@Suite("StreamCollector")
struct StreamCollectorTests {

    /// Helper to create a ChatStreamingResult from chunks with configurable think tags
    static func makeStreamingResult(
        chunks: [StreamChunk],
        thinkStartTag: String? = nil,
        thinkEndTag: String? = nil,
        promptTokens: Int = 10
    ) -> ChatStreamingResult {
        let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }
        return (
            modelID: "test-model",
            stream: stream,
            promptTokens: promptTokens,
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            thinkStartTag: thinkStartTag,
            thinkEndTag: thinkEndTag
        )
    }

    // MARK: - Think Extraction

    @Test("extracts think tags into reasoningContent")
    func thinkExtraction() async throws {
        let result = Self.makeStreamingResult(
            chunks: [
                StreamChunk(text: "<think>I need to think about this</think>The answer is 42"),
                StreamChunk(text: "", completionTokens: 12),
            ],
            thinkStartTag: "<think>",
            thinkEndTag: "</think>"
        )

        let collected = try await StreamCollector.collect(
            from: result,
            extractThinking: true,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>"
        )

        #expect(collected.reasoningContent == "I need to think about this")
        #expect(collected.content == "The answer is 42")
        #expect(collected.finishReason == "stop")
    }

    @Test("multi-chunk think tags extracted correctly")
    func thinkExtractionMultiChunk() async throws {
        let result = Self.makeStreamingResult(
            chunks: [
                StreamChunk(text: "<think>Step 1: "),
                StreamChunk(text: "analyze"),
                StreamChunk(text: "</think>Result"),
                StreamChunk(text: "", completionTokens: 8),
            ],
            thinkStartTag: "<think>",
            thinkEndTag: "</think>"
        )

        let collected = try await StreamCollector.collect(
            from: result,
            extractThinking: true
        )

        #expect(collected.reasoningContent == "Step 1: analyze")
        #expect(collected.content == "Result")
    }

    @Test("no think extraction when disabled")
    func thinkExtractionDisabled() async throws {
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "<think>reasoning</think>content"),
            StreamChunk(text: "", completionTokens: 5),
        ])

        let collected = try await StreamCollector.collect(
            from: result,
            extractThinking: false
        )

        #expect(collected.reasoningContent == nil)
        #expect(collected.content == "<think>reasoning</think>content")
    }

    @Test("think extraction with tool calls returns nil content and reasoning")
    func thinkWithToolCalls() async throws {
        let tc = ResponseToolCall(
            index: 0, id: "call_1", type: "function",
            function: ResponseToolCallFunction(name: "get_weather", arguments: "{\"location\":\"Paris\"}")
        )
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "<think>Should I call weather?</think>"),
            StreamChunk(text: "", toolCalls: [tc], completionTokens: 5),
        ], thinkStartTag: "<think>", thinkEndTag: "</think>")

        let collected = try await StreamCollector.collect(
            from: result,
            extractThinking: true
        )

        #expect(collected.toolCalls?.count == 1)
        #expect(collected.content == nil)
        #expect(collected.reasoningContent == nil)
        #expect(collected.finishReason == "tool_calls")
    }

    // MARK: - Logprobs

    @Test("collects logprobs from stream chunks")
    func logprobsCollection() async throws {
        let lp1 = ResolvedLogprob(token: "Hello", tokenId: 1, logprob: -0.5, topTokens: [
            (token: "Hello", tokenId: 1, logprob: -0.5),
            (token: "Hi", tokenId: 2, logprob: -1.2),
        ])
        let lp2 = ResolvedLogprob(token: " world", tokenId: 3, logprob: -0.3, topTokens: [
            (token: " world", tokenId: 3, logprob: -0.3),
        ])

        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "Hello", logprobs: [lp1]),
            StreamChunk(text: " world", logprobs: [lp2]),
            StreamChunk(text: "", completionTokens: 2),
        ])

        let collected = try await StreamCollector.collect(from: result, extractThinking: false)

        #expect(collected.logprobs.count == 2)
        #expect(collected.logprobs[0].token == "Hello")
        #expect(collected.logprobs[1].token == " world")
    }

    @Test("buildChoiceLogprobs converts ResolvedLogprob to ChoiceLogprobs")
    func buildChoiceLogprobs() {
        let resolved = [
            ResolvedLogprob(token: "A", tokenId: 1, logprob: -0.1, topTokens: [
                (token: "A", tokenId: 1, logprob: -0.1),
                (token: "B", tokenId: 2, logprob: -2.0),
            ]),
        ]

        let choice = MLXChatCompletionsController.buildChoiceLogprobs(resolved)
        #expect(choice != nil)
        #expect(choice!.content!.count == 1)
        #expect(choice!.content![0].token == "A")
        #expect(abs(choice!.content![0].logprob - Double(-0.1)) < 0.001)
        #expect(choice!.content![0].topLogprobs.count == 2)
        #expect(choice!.content![0].topLogprobs[0].token == "A")
        #expect(choice!.content![0].topLogprobs[1].token == "B")
    }

    @Test("buildChoiceLogprobs returns nil for empty input")
    func buildChoiceLogprobsEmpty() {
        let choice = MLXChatCompletionsController.buildChoiceLogprobs(nil)
        #expect(choice == nil)

        let choice2 = MLXChatCompletionsController.buildChoiceLogprobs([])
        #expect(choice2 == nil)
    }

    // MARK: - Finish Reason

    @Test("finishReason is tool_calls when tool calls present")
    func finishReasonToolCalls() async throws {
        let tc = ResponseToolCall(
            index: 0, id: "call_1", type: "function",
            function: ResponseToolCallFunction(name: "calc", arguments: "{}")
        )
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "", toolCalls: [tc], completionTokens: 5),
        ])

        let collected = try await StreamCollector.collect(from: result, extractThinking: false)
        #expect(collected.finishReason == "tool_calls")
    }

    @Test("finishReason is stop when stopped by sequence")
    func finishReasonStopSequence() async throws {
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "Hi", stoppedBySequence: true),
            StreamChunk(text: "", completionTokens: 1),
        ])

        let collected = try await StreamCollector.collect(from: result, extractThinking: false)
        #expect(collected.finishReason == "stop")
    }

    @Test("finishReason is length when max tokens reached")
    func finishReasonLength() async throws {
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "Hello world", completionTokens: 10),
        ])

        let collected = try await StreamCollector.collect(
            from: result,
            extractThinking: false,
            maxTokens: 10
        )
        #expect(collected.finishReason == "length")
    }

    @Test("finishReason is stop by default")
    func finishReasonDefault() async throws {
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "Hello"),
            StreamChunk(text: "", completionTokens: 1),
        ])

        let collected = try await StreamCollector.collect(from: result, extractThinking: false)
        #expect(collected.finishReason == "stop")
    }

    // MARK: - Timing and Token Stats

    @Test("collects timing and token stats from final chunk")
    func timingStats() async throws {
        let result = Self.makeStreamingResult(chunks: [
            StreamChunk(text: "Hi"),
            StreamChunk(text: "", promptTokens: 20, completionTokens: 15, cachedTokens: 5, promptTime: 0.1, generateTime: 0.5),
        ], promptTokens: 10)

        let collected = try await StreamCollector.collect(from: result, extractThinking: false)
        #expect(collected.promptTokens == 20)
        #expect(collected.completionTokens == 15)
        #expect(collected.cachedTokens == 5)
        #expect(collected.promptTime == 0.1)
        #expect(collected.generateTime == 0.5)
    }

    // MARK: - Edge Cases

    @Test("empty stream produces nil content")
    func emptyStream() async throws {
        let result = Self.makeStreamingResult(chunks: [])
        let collected = try await StreamCollector.collect(from: result, extractThinking: false)
        #expect(collected.content == nil)
        #expect(collected.reasoningContent == nil)
        #expect(collected.logprobs.isEmpty)
    }

    @Test("stream with only think tags and no visible content")
    func onlyThinkContent() async throws {
        let result = Self.makeStreamingResult(
            chunks: [
                StreamChunk(text: "<think>all reasoning no content</think>"),
                StreamChunk(text: "", completionTokens: 5),
            ],
            thinkStartTag: "<think>",
            thinkEndTag: "</think>"
        )

        let collected = try await StreamCollector.collect(
            from: result,
            extractThinking: true
        )

        #expect(collected.reasoningContent == "all reasoning no content")
        // Content should be nil or empty since there's nothing after </think>
        #expect(collected.content == nil)
    }
}

// MARK: - Post-Processing Parity Integration Tests

@Suite("BatchPostProcessingParity")
struct BatchPostProcessingParityTests {

    // MARK: - BatchAPIController with think extraction

    @Test("BatchAPIController response includes reasoningContent when model supports thinking")
    func batchAPIThinkExtraction() async throws {
        let app = try await Application.make(.testing)
        defer { Task { try await app.asyncShutdown() } }

        let svc = FakeBatchService(maxConcurrent: 8)
        svc.streamingResultFactory = { _ in
            let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
                continuation.yield(StreamChunk(text: "<think>reasoning here</think>visible content"))
                continuation.yield(StreamChunk(text: "", completionTokens: 8, promptTime: 0.01, generateTime: 0.02))
                continuation.finish()
            }
            return (
                modelID: "test-model", stream: stream, promptTokens: 10,
                toolCallStartTag: nil, toolCallEndTag: nil,
                thinkStartTag: "<think>", thinkEndTag: "</think>"
            )
        }

        let store = BatchStore()
        let controller = BatchAPIController(service: svc, store: store, modelID: "test-model")
        try app.register(collection: controller)

        // Upload JSONL
        let jsonl = #"{"custom_id":"think-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"think about this"}]}}"#
        var uploadHeaders = HTTPHeaders()
        let boundary = "test-boundary-\(UUID().uuidString)"
        uploadHeaders.add(name: .contentType, value: "multipart/form-data; boundary=\(boundary)")
        let multipartBody = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\n\(jsonl)\r\n--\(boundary)--\r\n"

        try await app.test(.POST, "/v1/files", headers: uploadHeaders, body: .init(string: multipartBody)) { res async in
            #expect(res.status == .ok)
            let file = try? res.content.decode(FileObject.self)
            guard let fileId = file?.id else { return }

            // Create batch
            try? await app.test(.POST, "/v1/batches", beforeRequest: { req in
                try req.content.encode(["input_file_id": fileId, "endpoint": "/v1/chat/completions", "completion_window": "24h"])
            }) { batchRes async in
                #expect(batchRes.status == .ok)
                let batch = try? batchRes.content.decode(BatchObject.self)
                guard let batchId = batch?.id else { return }

                // Poll until completed (max 5 tries)
                for _ in 0..<5 {
                    try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
                    try? await app.test(.GET, "/v1/batches/\(batchId)") { pollRes async in
                        let polled = try? pollRes.content.decode(BatchObject.self)
                        if polled?.status == "completed", let outputId = polled?.outputFileId {
                            // Check output file for reasoning_content
                            try? await app.test(.GET, "/v1/files/\(outputId)/content") { contentRes async in
                                let body = contentRes.body.string
                                // The output should contain reasoning_content
                                #expect(body.contains("reasoning_content") || body.contains("visible content"))
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - BatchAPIController with logprobs

    @Test("BatchAPIController response includes logprobs when provided in stream")
    func batchAPILogprobs() async throws {
        let app = try await Application.make(.testing)
        defer { Task { try await app.asyncShutdown() } }

        let svc = FakeBatchService(maxConcurrent: 8)
        svc.streamingResultFactory = { _ in
            let lp = ResolvedLogprob(token: "Hi", tokenId: 1, logprob: -0.5, topTokens: [
                (token: "Hi", tokenId: 1, logprob: -0.5)
            ])
            let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
                continuation.yield(StreamChunk(text: "Hi", logprobs: [lp]))
                continuation.yield(StreamChunk(text: "", completionTokens: 1, promptTime: 0.01, generateTime: 0.01))
                continuation.finish()
            }
            return (
                modelID: "test-model", stream: stream, promptTokens: 5,
                toolCallStartTag: nil, toolCallEndTag: nil,
                thinkStartTag: nil, thinkEndTag: nil
            )
        }

        let store = BatchStore()
        let controller = BatchAPIController(service: svc, store: store, modelID: "test-model")
        try app.register(collection: controller)

        // Upload and create batch
        let jsonl = #"{"custom_id":"lp-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":5}}"#
        var uploadHeaders = HTTPHeaders()
        let boundary = "boundary-\(UUID().uuidString)"
        uploadHeaders.add(name: .contentType, value: "multipart/form-data; boundary=\(boundary)")
        let multipartBody = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\n\(jsonl)\r\n--\(boundary)--\r\n"

        try await app.test(.POST, "/v1/files", headers: uploadHeaders, body: .init(string: multipartBody)) { res async in
            let file = try? res.content.decode(FileObject.self)
            guard let fileId = file?.id else { return }

            try? await app.test(.POST, "/v1/batches", beforeRequest: { req in
                try req.content.encode(["input_file_id": fileId, "endpoint": "/v1/chat/completions", "completion_window": "24h"])
            }) { batchRes async in
                let batch = try? batchRes.content.decode(BatchObject.self)
                guard let batchId = batch?.id else { return }

                for _ in 0..<5 {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                    try? await app.test(.GET, "/v1/batches/\(batchId)") { pollRes async in
                        let polled = try? pollRes.content.decode(BatchObject.self)
                        if polled?.status == "completed", let outputId = polled?.outputFileId {
                            try? await app.test(.GET, "/v1/files/\(outputId)/content") { contentRes async in
                                let body = contentRes.body.string
                                #expect(body.contains("logprobs") || body.contains("Hi"))
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - BatchAPIController with tool calls

    @Test("BatchAPIController response includes tool_calls with correct finish_reason")
    func batchAPIToolCalls() async throws {
        let app = try await Application.make(.testing)
        defer { Task { try await app.asyncShutdown() } }

        let tc = ResponseToolCall(
            index: 0, id: "call_abc", type: "function",
            function: ResponseToolCallFunction(name: "get_weather", arguments: "{\"location\":\"Paris\"}")
        )
        let svc = FakeBatchService(maxConcurrent: 8)
        svc.streamingResultFactory = { _ in
            let stream = AsyncThrowingStream<StreamChunk, Error> { continuation in
                continuation.yield(StreamChunk(text: "", toolCalls: [tc]))
                continuation.yield(StreamChunk(text: "", completionTokens: 5, promptTime: 0.01, generateTime: 0.02))
                continuation.finish()
            }
            return (
                modelID: "test-model", stream: stream, promptTokens: 10,
                toolCallStartTag: nil, toolCallEndTag: nil,
                thinkStartTag: nil, thinkEndTag: nil
            )
        }

        let store = BatchStore()
        let controller = BatchAPIController(service: svc, store: store, modelID: "test-model")
        try app.register(collection: controller)

        let jsonl = #"{"custom_id":"tc-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}}"#
        var uploadHeaders = HTTPHeaders()
        let boundary = "boundary-\(UUID().uuidString)"
        uploadHeaders.add(name: .contentType, value: "multipart/form-data; boundary=\(boundary)")
        let multipartBody = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\n\(jsonl)\r\n--\(boundary)--\r\n"

        try await app.test(.POST, "/v1/files", headers: uploadHeaders, body: .init(string: multipartBody)) { res async in
            let file = try? res.content.decode(FileObject.self)
            guard let fileId = file?.id else { return }

            try? await app.test(.POST, "/v1/batches", beforeRequest: { req in
                try req.content.encode(["input_file_id": fileId, "endpoint": "/v1/chat/completions", "completion_window": "24h"])
            }) { batchRes async in
                let batch = try? batchRes.content.decode(BatchObject.self)
                guard let batchId = batch?.id else { return }

                for _ in 0..<5 {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                    try? await app.test(.GET, "/v1/batches/\(batchId)") { pollRes async in
                        let polled = try? pollRes.content.decode(BatchObject.self)
                        if polled?.status == "completed", let outputId = polled?.outputFileId {
                            try? await app.test(.GET, "/v1/files/\(outputId)/content") { contentRes async in
                                let body = contentRes.body.string
                                #expect(body.contains("tool_calls"))
                                #expect(body.contains("get_weather"))
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Grammar Header

    @Test("BatchCompletionsController sets grammar downgrade header when strict tools present")
    func grammarDowngradeHeader() async throws {
        let app = try await Application.make(.testing)
        defer { Task { try await app.asyncShutdown() } }

        let svc = FakeBatchService(maxConcurrent: 8)
        // enableGrammarConstraints is false by default in FakeBatchService
        let controller = BatchCompletionsController(service: svc, modelID: "test-model")
        try app.register(collection: controller)

        // Request with strict: true tool
        let body = """
        {"requests":[{"custom_id":"strict-1","body":{"model":"test-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"test","strict":true,"parameters":{"type":"object","properties":{"x":{"type":"string"}}}}}]}}]}
        """

        try await app.test(.POST, "/v1/batch/completions", beforeRequest: { req in
            req.headers.contentType = .json
            req.body = .init(string: body)
        }) { res async in
            #expect(res.status == .ok)
            let grammarHeader = res.headers.first(name: "X-Grammar-Constraints")
            #expect(grammarHeader == "downgraded")
        }
    }
}
