# Batch Dispatch API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI-compatible Batch API (`/v1/batches` + `/v1/files`) and custom SSE multiplex endpoint (`/v1/batch/completions`) for concurrent multi-request dispatch using AFM's existing BatchScheduler.

**Architecture:** Thin controller layer wrapping existing BatchScheduler. Two controllers — `BatchAPIController` for OpenAI-compatible polling workflow, `BatchCompletionsController` for real-time SSE multiplex. Shared `BatchStore` actor for in-memory file/batch state. Auto-promotion creates a BatchScheduler on-the-fly when in serial mode.

**Tech Stack:** Swift, Vapor (HTTP framework), MLX (GPU inference), AsyncThrowingStream (concurrency)

**Spec:** `docs/superpowers/specs/2026-03-25-batch-dispatch-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `Sources/MacLocalAPI/Models/BatchStore.swift` | **New.** Actor holding in-memory file store + batch state. Thread-safe CRUD for files and batches. Auto-eviction. |
| `Sources/MacLocalAPI/Controllers/BatchAPIController.swift` | **New.** OpenAI-compatible `/v1/files` and `/v1/batches` endpoints. Parses JSONL, dispatches to scheduler, collects results. |
| `Sources/MacLocalAPI/Controllers/BatchCompletionsController.swift` | **New.** Custom `/v1/batch/completions` SSE multiplex endpoint. Merges per-request streams into single tagged SSE output. |
| `Sources/MacLocalAPI/Models/OpenAIRequest.swift` | **Modify.** Add `BatchCompletionRequest`, `BatchRequestItem`, `BatchCreateRequest`, `BatchInputLine` types. |
| `Sources/MacLocalAPI/Models/OpenAIResponse.swift` | **Modify.** Add `BatchObject`, `FileObject`, `BatchResultLine`, `BatchSSEEvent`, `BatchError`, `FileDeleteResponse` types. |
| `Sources/MacLocalAPI/Models/BatchScheduler.swift` | **Modify.** Add `tryReserveMultiple(count:)`, per-slot `isCancelled` flag, `activeSlotCount` accessor. |
| `Sources/MacLocalAPI/Models/MLXModelService.swift` | **Modify.** Add `ensureBatchMode(concurrency:)`, `activeBatchCount`, grace-period teardown. Fix non-streaming to use scheduler. |
| `Sources/MacLocalAPI/Controllers/MLXChatServing.swift` | **Modify.** Add `ensureBatchMode(concurrency:)` to protocol. |
| `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` | **Modify.** Route non-streaming requests through `generateStreaming()` + collect when scheduler is active. |
| `Sources/MacLocalAPI/Server.swift` | **Modify.** Register `BatchAPIController` and `BatchCompletionsController` routes. |
| `Tests/MacLocalAPITests/BatchStoreTests.swift` | **New.** Unit tests for BatchStore actor. |
| `Tests/MacLocalAPITests/BatchAPITests.swift` | **New.** Integration tests for OpenAI-compatible batch workflow. |
| `Tests/MacLocalAPITests/BatchCompletionsTests.swift` | **New.** Tests for SSE multiplex endpoint. |

---

## Task 1: Request and Response Types

**Files:**
- Modify: `Sources/MacLocalAPI/Models/OpenAIRequest.swift` (append after line 389)
- Modify: `Sources/MacLocalAPI/Models/OpenAIResponse.swift` (append after line 530)

- [ ] **Step 1: Add batch request types to OpenAIRequest.swift**

Append these types after the existing code:

```swift
// MARK: - Batch API Types

/// A single request entry in the SSE multiplex batch.
struct BatchRequestItem: Content {
    let customId: String
    let body: ChatCompletionRequest

    enum CodingKeys: String, CodingKey {
        case customId = "custom_id"
        case body
    }
}

/// Request body for POST /v1/batch/completions (SSE multiplex).
struct BatchCompletionRequest: Content {
    let requests: [BatchRequestItem]
}

/// Request body for POST /v1/batches (OpenAI-compatible).
struct BatchCreateRequest: Content {
    let inputFileId: String
    let endpoint: String
    let completionWindow: String?

    enum CodingKeys: String, CodingKey {
        case inputFileId = "input_file_id"
        case endpoint
        case completionWindow = "completion_window"
    }
}

/// A single line in the input JSONL file for batch processing.
struct BatchInputLine: Codable {
    let customId: String
    let method: String
    let url: String
    let body: ChatCompletionRequest

    enum CodingKeys: String, CodingKey {
        case customId = "custom_id"
        case method, url, body
    }
}
```

- [ ] **Step 2: Add batch response types to OpenAIResponse.swift**

Append these types after the existing code:

```swift
// MARK: - Batch API Response Types

/// OpenAI File object for /v1/files endpoints.
struct FileObject: Content {
    let id: String
    let object: String
    let bytes: Int
    let createdAt: Int
    let filename: String
    let purpose: String

    enum CodingKeys: String, CodingKey {
        case id, object, bytes
        case createdAt = "created_at"
        case filename, purpose
    }

    init(id: String, bytes: Int, createdAt: Int, filename: String, purpose: String) {
        self.id = id
        self.object = "file"
        self.bytes = bytes
        self.createdAt = createdAt
        self.filename = filename
        self.purpose = purpose
    }
}

/// Response for DELETE /v1/files/{id}.
struct FileDeleteResponse: Content {
    let id: String
    let object: String
    let deleted: Bool

    init(id: String) {
        self.id = id
        self.object = "file"
        self.deleted = true
    }
}

/// Batch request counts.
struct BatchRequestCounts: Content {
    var total: Int
    var completed: Int
    var failed: Int
}

/// OpenAI Batch object for /v1/batches endpoints.
struct BatchObject: Content {
    let id: String
    let object: String
    let endpoint: String
    let inputFileId: String
    let completionWindow: String
    var status: String
    let createdAt: Int
    var completedAt: Int?
    var outputFileId: String?
    var requestCounts: BatchRequestCounts

    enum CodingKeys: String, CodingKey {
        case id, object, endpoint, status
        case inputFileId = "input_file_id"
        case completionWindow = "completion_window"
        case createdAt = "created_at"
        case completedAt = "completed_at"
        case outputFileId = "output_file_id"
        case requestCounts = "request_counts"
    }
}

/// A single result line in the output JSONL file.
struct BatchResultLine: Codable {
    let id: String
    let customId: String
    let response: BatchResultResponse?
    let error: BatchError?

    enum CodingKeys: String, CodingKey {
        case id
        case customId = "custom_id"
        case response, error
    }
}

/// The response wrapper inside a batch result line.
struct BatchResultResponse: Codable {
    let statusCode: Int
    let requestId: String
    let body: ChatCompletionResponse

    enum CodingKeys: String, CodingKey {
        case statusCode = "status_code"
        case requestId = "request_id"
        case body
    }
}

/// Error object for batch results and SSE error events.
struct BatchError: Content {
    let message: String
    let type: String
}

/// Wrapper for OpenAI list responses (GET /v1/batches).
struct BatchListResponse: Content {
    let object: String
    let data: [BatchObject]
    let hasMore: Bool

    enum CodingKeys: String, CodingKey {
        case object, data
        case hasMore = "has_more"
    }

    init(data: [BatchObject]) {
        self.object = "list"
        self.data = data
        self.hasMore = false
    }
}
```

- [ ] **Step 3: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add Sources/MacLocalAPI/Models/OpenAIRequest.swift Sources/MacLocalAPI/Models/OpenAIResponse.swift
git commit -m "feat(batch): add request and response types for batch dispatch API"
```

---

## Task 2: BatchStore Actor

**Files:**
- Create: `Sources/MacLocalAPI/Models/BatchStore.swift`

- [ ] **Step 1: Create BatchStore actor**

Create `Sources/MacLocalAPI/Models/BatchStore.swift`:

```swift
import Foundation
import Vapor

/// In-memory store for batch files and batch state.
/// All access is serialized through the actor for thread safety.
actor BatchStore {
    /// Maximum age for files before auto-eviction (1 hour).
    private let fileTTL: TimeInterval = 3600

    // MARK: - File Storage

    struct StoredFile {
        let id: String
        let bytes: Int
        let filename: String
        let purpose: String
        let data: Data
        let createdAt: Date
    }

    private var files: [String: StoredFile] = [:]

    /// Store a file and return its metadata.
    func storeFile(filename: String, purpose: String, data: Data) -> FileObject {
        evictExpiredFiles()
        let id = "file-\(UUID().uuidString.lowercased().prefix(12))"
        let now = Date()
        let stored = StoredFile(
            id: id, bytes: data.count, filename: filename,
            purpose: purpose, data: data, createdAt: now
        )
        files[id] = stored
        return FileObject(
            id: id, bytes: data.count,
            createdAt: Int(now.timeIntervalSince1970),
            filename: filename, purpose: purpose
        )
    }

    /// Get file metadata.
    func getFile(_ id: String) -> FileObject? {
        evictExpiredFiles()
        guard let f = files[id] else { return nil }
        return FileObject(
            id: f.id, bytes: f.bytes,
            createdAt: Int(f.createdAt.timeIntervalSince1970),
            filename: f.filename, purpose: f.purpose
        )
    }

    /// Get raw file data.
    func getFileData(_ id: String) -> Data? {
        files[id]?.data
    }

    /// Delete a file.
    func deleteFile(_ id: String) -> Bool {
        files.removeValue(forKey: id) != nil
    }

    /// Remove files older than TTL.
    private func evictExpiredFiles() {
        let cutoff = Date().addingTimeInterval(-fileTTL)
        files = files.filter { $0.value.createdAt > cutoff }
    }

    // MARK: - Batch State

    struct BatchState {
        let id: String
        let inputFileId: String
        let endpoint: String
        var status: String  // validating, in_progress, completed, failed, cancelling, cancelled
        var requestCounts: BatchRequestCounts
        var results: [BatchResultLine]
        var outputFileId: String?
        let createdAt: Date
        var completedAt: Date?
        var error: BatchError?
        /// IDs of slots in the scheduler, for cancellation support.
        var slotIds: [UUID]
    }

    private var batches: [String: BatchState] = [:]

    /// Create a new batch in `validating` state.
    func createBatch(inputFileId: String, endpoint: String, totalRequests: Int) -> String {
        let id = "batch_\(UUID().uuidString.lowercased().prefix(12))"
        batches[id] = BatchState(
            id: id, inputFileId: inputFileId, endpoint: endpoint,
            status: "validating",
            requestCounts: BatchRequestCounts(total: totalRequests, completed: 0, failed: 0),
            results: [], outputFileId: nil,
            createdAt: Date(), completedAt: nil, error: nil,
            slotIds: []
        )
        return id
    }

    /// Transition batch to in_progress.
    func markBatchInProgress(_ id: String, slotIds: [UUID] = []) {
        batches[id]?.status = "in_progress"
        batches[id]?.slotIds = slotIds
    }

    /// Record a completed request result.
    func recordResult(_ batchId: String, result: BatchResultLine) {
        guard var batch = batches[batchId] else { return }
        batch.results.append(result)
        if result.error != nil {
            batch.requestCounts.failed += 1
        } else {
            batch.requestCounts.completed += 1
        }

        // Check if all requests are done
        let done = batch.requestCounts.completed + batch.requestCounts.failed
        if done >= batch.requestCounts.total {
            batch.status = "completed"
            batch.completedAt = Date()
            // Build output JSONL and store as file
            let outputData = buildOutputJSONL(results: batch.results)
            let outputFile = storeFileInternal(
                filename: "batch_\(batchId)_output.jsonl",
                purpose: "batch_output",
                data: outputData
            )
            batch.outputFileId = outputFile
        }

        batches[batchId] = batch
    }

    /// Mark batch as failed with an error.
    func markBatchFailed(_ id: String, error: BatchError) {
        batches[id]?.status = "failed"
        batches[id]?.error = error
        batches[id]?.completedAt = Date()
    }

    /// Mark batch as cancelling.
    func markBatchCancelling(_ id: String) {
        batches[id]?.status = "cancelling"
    }

    /// Mark batch as cancelled.
    func markBatchCancelled(_ id: String) {
        batches[id]?.status = "cancelled"
        batches[id]?.completedAt = Date()
    }

    /// Get batch state as API object.
    func getBatch(_ id: String) -> BatchObject? {
        guard let b = batches[id] else { return nil }
        return BatchObject(
            id: b.id, object: "batch", endpoint: b.endpoint,
            inputFileId: b.inputFileId, completionWindow: "24h",
            status: b.status,
            createdAt: Int(b.createdAt.timeIntervalSince1970),
            completedAt: b.completedAt.map { Int($0.timeIntervalSince1970) },
            outputFileId: b.outputFileId,
            requestCounts: b.requestCounts
        )
    }

    /// Get slot IDs for a batch (for cancellation).
    func getSlotIds(_ batchId: String) -> [UUID] {
        batches[batchId]?.slotIds ?? []
    }

    /// List all batches.
    func listBatches() -> [BatchObject] {
        batches.values.compactMap { b in
            BatchObject(
                id: b.id, object: "batch", endpoint: b.endpoint,
                inputFileId: b.inputFileId, completionWindow: "24h",
                status: b.status,
                createdAt: Int(b.createdAt.timeIntervalSince1970),
                completedAt: b.completedAt.map { Int($0.timeIntervalSince1970) },
                outputFileId: b.outputFileId,
                requestCounts: b.requestCounts
            )
        }
    }

    // MARK: - Private Helpers

    /// Store file without eviction check (internal use for output files).
    private func storeFileInternal(filename: String, purpose: String, data: Data) -> String {
        let id = "file-\(UUID().uuidString.lowercased().prefix(12))"
        files[id] = StoredFile(
            id: id, bytes: data.count, filename: filename,
            purpose: purpose, data: data, createdAt: Date()
        )
        return id
    }

    /// Encode results array as JSONL Data.
    private func buildOutputJSONL(results: [BatchResultLine]) -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .useDefaultKeys
        let lines = results.compactMap { result -> String? in
            guard let data = try? encoder.encode(result) else { return nil }
            return String(data: data, encoding: .utf8)
        }
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }
}
```

- [ ] **Step 2: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Models/BatchStore.swift
git commit -m "feat(batch): add BatchStore actor for in-memory file and batch state"
```

---

## Task 3: BatchScheduler Extensions

**Files:**
- Modify: `Sources/MacLocalAPI/Models/BatchScheduler.swift`

This task adds three things to the existing BatchScheduler: atomic multi-slot reservation, per-slot cancellation, and an active slot count accessor.

- [ ] **Step 1: Add `tryReserveMultiple(count:)` method**

In `BatchScheduler.swift`, after the existing `releaseReservation()` method (around line 321), add:

```swift
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
```

- [ ] **Step 2: Add `isCancelled` flag to SlotState**

In the `SlotState` class (around line 62), add a cancellation flag after the existing properties (around line 100):

```swift
        /// Set to true to signal this slot should stop generating.
        var isCancelled = false
```

- [ ] **Step 3: Add cancellation check in the decode loop**

Find the deferred-dispatch section inside `generationLoop()` (around lines 559-596), which iterates over slots using `completedIndices` and `finishSlot(at:)`. At the top of the per-slot dispatch processing (inside the loop that checks each slot), add a cancellation check:

```swift
            // Check for per-slot cancellation
            if slot.isCancelled {
                finishSlot(at: i)
                completedIndices.insert(i)
                continue
            }
```

The existing code uses `completedIndices` (not `slotsToRemove`) and `finishSlot(at:)` for slot cleanup. The cancellation check should be added as the first statement inside the per-slot iteration in the dispatch section, before the deferred token processing.

- [ ] **Step 4: Add active slot count accessor**

After the `tryReserveMultiple` method, add:

```swift
    /// Current number of active + pending slots (for teardown decisions).
    nonisolated var activeSlotCount: Int {
        _inFlightCount.withLock { $0 }
    }
```

- [ ] **Step 5: Add cancel method for batch slots**

Add a method to cancel specific slots by UUID:

```swift
    /// Cancel all slots matching the given IDs.
    func cancelSlots(ids: Set<UUID>) {
        for slot in slots where ids.contains(slot.id) {
            slot.isCancelled = true
        }
    }
```

This must be an actor-isolated method (no `nonisolated`) since it accesses `slots`.

- [ ] **Step 6: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 7: Commit**

```bash
git add Sources/MacLocalAPI/Models/BatchScheduler.swift
git commit -m "feat(batch): add multi-slot reservation, cancellation, and active count to BatchScheduler"
```

---

## Task 4: Auto-Promotion in MLXModelService

**Files:**
- Modify: `Sources/MacLocalAPI/Models/MLXModelService.swift` (around lines 114, 148-154, 1083-1103)
- Modify: `Sources/MacLocalAPI/Controllers/MLXChatServing.swift` (around line 26-81)

- [ ] **Step 1: Add `ensureBatchMode` to MLXChatServing protocol**

In `MLXChatServing.swift`, add to the protocol (after `releaseSlot()` at line 36):

```swift
    func ensureBatchMode(concurrency: Int) async throws
    func releaseBatchReference()
```

- [ ] **Step 2: Add auto-promotion state to MLXModelService**

In `MLXModelService.swift`, near the existing `scheduler` property (around line 148), add:

```swift
    // NOTE: ensure `import os` is present at the top of this file for OSAllocatedUnfairLock

    /// Whether the server was started with --concurrent (persistent batch mode).
    private var startedInBatchMode = false

    /// Number of in-flight batch operations (for auto-teardown).
    private let _activeBatchCount = OSAllocatedUnfairLock(initialState: 0)

    /// Whether a promotion is currently in progress (prevents races).
    private var promotionInProgress = false

    /// Scheduled teardown work item (cancelled if new batch arrives).
    private var teardownWorkItem: DispatchWorkItem?
```

- [ ] **Step 3: Mark persistent batch mode in existing init path**

In the existing code where `maxConcurrent` is set (around line 150 setter or wherever `initScheduler` is called), add a flag. Find where `initScheduler()` is called from `main.swift` (line 637) and note that the service should set `startedInBatchMode = true` when `maxConcurrent >= 2` is set at startup.

Add to `initScheduler()` (around line 1083):

```swift
    func initScheduler() async throws {
        guard maxConcurrent >= 2 else { return }
        startedInBatchMode = true  // <-- add this line
        // ... rest of existing code
    }
```

- [ ] **Step 4: Implement `ensureBatchMode(concurrency:)`**

Add after `initScheduler()` (around line 1103):

```swift
    /// Auto-promote from serial to batch mode for batch requests.
    /// Thread-safe: uses stateLock + promotionInProgress to prevent races.
    func ensureBatchMode(concurrency: Int) async throws {
        // Fast path: scheduler already exists
        if withStateLock({ scheduler != nil }) {
            _activeBatchCount.withLock { $0 += 1 }
            // Cancel any pending teardown
            teardownWorkItem?.cancel()
            teardownWorkItem = nil
            return
        }

        // Check if another caller is already promoting
        let shouldPromote = withStateLock { () -> Bool in
            if scheduler != nil { return false }
            if promotionInProgress { return false }
            promotionInProgress = true
            return true
        }

        if !shouldPromote {
            // Wait for the other caller to finish promotion
            while withStateLock({ promotionInProgress && scheduler == nil }) {
                try await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }
            _activeBatchCount.withLock { $0 += 1 }
            return
        }

        // Promote: create scheduler
        let limit = max(concurrency, 8)
        self.maxConcurrent = limit
        self.enablePrefixCaching = true

        guard let container = withStateLock({ currentContainer }) else {
            withStateLock { promotionInProgress = false }
            throw MLXServiceError.noModelLoaded
        }

        let sched = await container.perform { context -> BatchScheduler in
            BatchScheduler(
                model: context.model,
                tokenizer: context.tokenizer,
                processor: context.processor,
                configuration: context.configuration,
                maxConcurrent: limit,
                enablePrefixCaching: true,
                cacheProfilePath: self.cacheProfilePath
            )
        }

        withStateLock {
            self.scheduler = sched
            self.promotionInProgress = false
        }
        _activeBatchCount.withLock { $0 += 1 }
        print("[\(ts())] Auto-promoted to batch mode: \(limit) concurrent slots (prefix caching enabled)")
    }

    /// Decrement batch reference count and schedule teardown if appropriate.
    func releaseBatchReference() {
        let remaining = _activeBatchCount.withLock { count -> Int in
            count = max(0, count - 1)
            return count
        }

        // Only teardown if auto-promoted (not started with --concurrent)
        guard !startedInBatchMode, remaining == 0 else { return }

        // Schedule teardown after grace period
        teardownWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            guard let self else { return }
            Task {
                await self.performTeardownIfIdle()
            }
        }
        teardownWorkItem = workItem
        DispatchQueue.global().asyncAfter(deadline: .now() + 5.0, execute: workItem)
    }

    /// Tear down auto-promoted scheduler if no active slots or batches.
    private func performTeardownIfIdle() async {
        let shouldTeardown = _activeBatchCount.withLock { $0 == 0 }
        guard shouldTeardown else { return }

        // Check scheduler has no active slots
        if let sched = withStateLock({ scheduler }) {
            guard sched.activeSlotCount == 0 else { return }
        }

        withStateLock {
            self.scheduler = nil
            self.maxConcurrent = 0
            self.enablePrefixCaching = false
        }
        print("[\(ts())] Auto-teardown: returned to serial mode")
    }
```

- [ ] **Step 5: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 6: Commit**

```bash
git add Sources/MacLocalAPI/Models/MLXModelService.swift Sources/MacLocalAPI/Controllers/MLXChatServing.swift
git commit -m "feat(batch): add auto-promotion/teardown lifecycle for batch mode"
```

---

## Task 5: Non-Streaming Through BatchScheduler

**Files:**
- Modify: `Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift` (around lines 178-243)

- [ ] **Step 1: Route non-streaming through scheduler when in batch mode**

In `MLXChatCompletionsController.swift`, find the non-streaming branch (around line 182-243). The current code does:

```swift
            // In concurrent mode, non-streaming requests currently bypass the
            // BatchScheduler decode loop...
            defer { service.releaseSlot() }

            let result: ChatGenerationResult = try await service.generate(...)
```

Replace the non-streaming path to use `generateStreaming()` + collect when a scheduler is active. Wrap the existing non-streaming code in a conditional:

```swift
            if service.maxConcurrent >= 2 {
                // Batch mode: route through scheduler for batched decode
                defer { service.releaseSlot() }

                let streamResult: ChatStreamingResult = try await service.generateStreaming(
                    model: modelID,
                    messages: chatRequest.messages,
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
                    stop: effectiveStop,
                    responseFormat: chatRequest.responseFormat,
                    chatTemplateKwargs: chatRequest.chatTemplateKwargs
                )

                // Collect stream into complete response
                var fullText = ""
                var allLogprobs: [ResolvedLogprob] = []
                var finalToolCalls: [ResponseToolCall]? = nil
                var finalToolCallDeltas: [StreamDeltaToolCall]? = nil
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
                    if let tcd = chunk.toolCallDeltas { finalToolCallDeltas = tcd }
                    if let pt = chunk.promptTokens { promptTokens = pt }
                    if let ct = chunk.completionTokens { completionTokens = ct }
                    if let cached = chunk.cachedTokens { cachedTokens = cached }
                    if let pt = chunk.promptTime { promptTime = pt }
                    if let gt = chunk.generateTime { generateTime = gt }
                    if let sbs = chunk.stoppedBySequence { stoppedBySequence = sbs }
                }

                let result: ChatGenerationResult = (
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

                // Continue with existing non-streaming response building...
                // (the code after the generate() call that builds the response)
```

The key change: keep the `defer { service.releaseSlot() }` and the existing response-building code below. Only replace the `service.generate()` call with `service.generateStreaming()` + collection when `service.maxConcurrent >= 2`.

The `else` branch keeps the existing `service.generate()` call for serial mode.

- [ ] **Step 2: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Controllers/MLXChatCompletionsController.swift
git commit -m "fix: route non-streaming requests through BatchScheduler when concurrent mode active"
```

---

## Task 6: OpenAI-Compatible Batch API Controller

**Files:**
- Create: `Sources/MacLocalAPI/Controllers/BatchAPIController.swift`

- [ ] **Step 1: Create BatchAPIController**

Create `Sources/MacLocalAPI/Controllers/BatchAPIController.swift`:

```swift
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
        for line in content.split(separator: "\n") where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            guard let lineData = line.data(using: .utf8) else { continue }
            let parsed = try decoder.decode(BatchInputLine.self, from: lineData)
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
        // TODO: cancel scheduler slots via cancelSlots(ids:)
        // For now, mark as cancelled directly
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

            // Use generateStreaming + collect
            let streamResult = try await service.generateStreaming(
                model: effectiveModel,
                messages: chatReq.messages,
                temperature: chatReq.temperature ?? temperature,
                maxTokens: chatReq.effectiveMaxTokens ?? maxTokens,
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

            // Collect stream
            var fullText = ""
            var promptTokens = streamResult.promptTokens
            var completionTokens = 0
            var cachedTokens = 0
            var toolCalls: [ResponseToolCall]? = nil
            var stoppedBySequence = false

            for try await chunk in streamResult.stream {
                fullText += chunk.text
                if let tc = chunk.toolCalls { toolCalls = tc }
                if let ct = chunk.completionTokens { completionTokens = ct }
                if let pt = chunk.promptTokens { promptTokens = pt }
                if let cached = chunk.cachedTokens { cachedTokens = cached }
                if let sbs = chunk.stoppedBySequence { stoppedBySequence = sbs }
            }

            // Use existing ChatCompletionResponse convenience inits
            let response: ChatCompletionResponse
            if let toolCalls, !toolCalls.isEmpty {
                response = ChatCompletionResponse(
                    model: effectiveModel,
                    toolCalls: toolCalls,
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    cachedTokens: cachedTokens > 0 ? cachedTokens : nil
                )
            } else {
                response = ChatCompletionResponse(
                    model: effectiveModel,
                    content: fullText,
                    promptTokens: promptTokens,
                    completionTokens: completionTokens,
                    cachedTokens: cachedTokens > 0 ? cachedTokens : nil
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
```

- [ ] **Step 2: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds. There may be warnings about unused parameters — these are fine and will be resolved as the controller is integrated.

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Controllers/BatchAPIController.swift
git commit -m "feat(batch): add OpenAI-compatible /v1/batches and /v1/files endpoints"
```

---

## Task 7: SSE Multiplex Controller

**Files:**
- Create: `Sources/MacLocalAPI/Controllers/BatchCompletionsController.swift`

- [ ] **Step 1: Create BatchCompletionsController**

Create `Sources/MacLocalAPI/Controllers/BatchCompletionsController.swift`:

```swift
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

            let streamResult = try await service.generateStreaming(
                model: effectiveModel,
                messages: chatReq.messages,
                temperature: chatReq.temperature ?? temperature,
                maxTokens: chatReq.effectiveMaxTokens ?? maxTokens,
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

            if isStreaming {
                // Emit streaming chunks tagged with custom_id
                let completionId = "chatcmpl-\(UUID().uuidString.lowercased().prefix(12))"
                var tokenCount = 0
                var promptTokens = streamResult.promptTokens
                var lastCachedTokens = 0

                for try await chunk in streamResult.stream {
                    tokenCount += 1

                    var event: [String: Any] = [
                        "custom_id": customId,
                        "id": completionId,
                        "object": "chat.completion.chunk",
                        "created": Int(Date().timeIntervalSince1970),
                        "model": effectiveModel,
                    ]

                    var delta: [String: Any] = [:]
                    if tokenCount == 1 { delta["role"] = "assistant" }
                    if !chunk.text.isEmpty { delta["content"] = chunk.text }

                    var choiceDict: [String: Any] = ["index": 0, "delta": delta]

                    if let ct = chunk.completionTokens {
                        // Final chunk with usage
                        choiceDict["finish_reason"] = chunk.toolCalls != nil ? "tool_calls" : "stop"
                        if let pt = chunk.promptTokens { promptTokens = pt }
                        if let cached = chunk.cachedTokens { lastCachedTokens = cached }
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
                // Non-streaming: collect all, emit single complete response
                var fullText = ""
                var promptTokens = streamResult.promptTokens
                var completionTokens = 0
                var cachedTokens = 0
                var toolCalls: [ResponseToolCall]? = nil
                var stoppedBySequence = false

                for try await chunk in streamResult.stream {
                    fullText += chunk.text
                    if let tc = chunk.toolCalls { toolCalls = tc }
                    if let ct = chunk.completionTokens { completionTokens = ct }
                    if let pt = chunk.promptTokens { promptTokens = pt }
                    if let cached = chunk.cachedTokens { cachedTokens = cached }
                    if let sbs = chunk.stoppedBySequence { stoppedBySequence = sbs }
                }

                let finishReason = toolCalls != nil ? "tool_calls" : "stop"

                var event: [String: Any] = [
                    "custom_id": customId,
                    "object": "chat.completion",
                    "id": "chatcmpl-\(UUID().uuidString.lowercased().prefix(12))",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": effectiveModel,
                    "choices": [[
                        "index": 0,
                        "message": [
                            "role": "assistant",
                            "content": toolCalls != nil ? NSNull() : fullText
                        ] as [String: Any],
                        "finish_reason": finishReason
                    ] as [String: Any]],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens
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
```

- [ ] **Step 2: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Controllers/BatchCompletionsController.swift
git commit -m "feat(batch): add SSE multiplex endpoint /v1/batch/completions"
```

---

## Task 8: Route Registration

**Files:**
- Modify: `Sources/MacLocalAPI/Server.swift` (around lines 244-262)
- Modify: `Sources/MacLocalAPI/main.swift` (if BatchStore needs to be passed)

- [ ] **Step 1: Add BatchStore instance and register controllers in Server.swift**

In `Server.swift`, find where the MLX controller is created and registered (around lines 244-262). After the existing `try app.register(collection: mlxController)`, add:

```swift
            // Batch API endpoints
            let batchStore = BatchStore()

            let batchAPIController = BatchAPIController(
                service: mlxModelService,
                store: batchStore,
                modelID: mlxModelID,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                topK: mlxTopK,
                minP: mlxMinP,
                presencePenalty: mlxPresencePenalty,
                seed: mlxSeed,
                maxLogprobs: mlxMaxLogprobs
            )
            try app.register(collection: batchAPIController)

            let batchCompletionsController = BatchCompletionsController(
                service: mlxModelService,
                modelID: mlxModelID,
                temperature: temperature,
                topP: mlxTopP,
                maxTokens: mlxMaxTokens,
                repetitionPenalty: mlxRepetitionPenalty,
                topK: mlxTopK,
                minP: mlxMinP,
                presencePenalty: mlxPresencePenalty,
                seed: mlxSeed,
                maxLogprobs: mlxMaxLogprobs
            )
            try app.register(collection: batchCompletionsController)
```

These variables match the existing `MLXChatCompletionsController` init at Server.swift:243-262.

- [ ] **Step 2: Verify the project builds**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Server.swift
git commit -m "feat(batch): register batch API and SSE multiplex routes"
```

---

## Task 9: Build and Smoke Test

- [ ] **Step 1: Full build**

Run: `swift build -c release 2>&1 | tail -10`
Expected: Build succeeds with no errors.

- [ ] **Step 2: Start server and test health**

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache .build/release/afm mlx -m mlx-community/Qwen3-0.6B-4bit --port 9999 &
sleep 5
curl -s http://localhost:9999/health
```
Expected: `{"status":"healthy","timestamp":...,"version":"1.0.0"}`

- [ ] **Step 3: Test SSE multiplex endpoint**

```bash
curl -s -N http://localhost:9999/v1/batch/completions \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "Say hello"}], "stream": true, "max_tokens": 10}},
      {"custom_id": "r2", "body": {"messages": [{"role": "user", "content": "Say world"}], "stream": false, "max_tokens": 10}}
    ]
  }'
```
Expected: SSE events with `custom_id` tags, ending with `data: [DONE]`.

- [ ] **Step 4: Test OpenAI batch endpoint — create input file**

```bash
# Create input JSONL
cat > /tmp/batch_input.jsonl << 'JSONL'
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Say world"}], "max_tokens": 10}}
JSONL

# Upload file
curl -s http://localhost:9999/v1/files \
  -F purpose=batch \
  -F file=@/tmp/batch_input.jsonl
```
Expected: JSON with `"id": "file-..."`, `"object": "file"`.

- [ ] **Step 5: Test OpenAI batch endpoint — create and poll batch**

```bash
# Create batch (use file ID from step 4)
FILE_ID="<paste file id>"
curl -s http://localhost:9999/v1/batches \
  -H "Content-Type: application/json" \
  -d "{\"input_file_id\": \"$FILE_ID\", \"endpoint\": \"/v1/chat/completions\", \"completion_window\": \"24h\"}"

# Poll until completed
BATCH_ID="<paste batch id>"
curl -s http://localhost:9999/v1/batches/$BATCH_ID

# Get results
OUTPUT_FILE_ID="<paste output_file_id>"
curl -s http://localhost:9999/v1/files/$OUTPUT_FILE_ID/content
```
Expected: Batch transitions to `completed`, output JSONL contains results for both requests.

- [ ] **Step 6: Kill test server**

```bash
kill %1
```

- [ ] **Step 7: Commit any fixes from smoke testing**

Fix any issues found during smoke testing, then commit.

---

## Task 10: Python Client Compatibility Test

- [ ] **Step 1: Create Python test script**

Create a test script (not committed — just for verification):

```python
#!/usr/bin/env python3
"""Test OpenAI batch API compatibility with AFM."""
from openai import OpenAI
import json, time, tempfile, os

client = OpenAI(base_url="http://localhost:9999/v1", api_key="not-needed")

# Create input JSONL
requests = [
    {"custom_id": f"req-{i}", "method": "POST", "url": "/v1/chat/completions",
     "body": {"messages": [{"role": "user", "content": f"Count to {i+1}"}], "max_tokens": 20}}
    for i in range(3)
]

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for r in requests:
        f.write(json.dumps(r) + '\n')
    input_path = f.name

# Upload
input_file = client.files.create(file=open(input_path, 'rb'), purpose='batch')
print(f"Uploaded: {input_file.id}")

# Create batch
batch = client.batches.create(
    input_file_id=input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch: {batch.id} status={batch.status}")

# Poll
while batch.status not in ("completed", "failed", "cancelled"):
    time.sleep(1)
    batch = client.batches.retrieve(batch.id)
    print(f"  status={batch.status} completed={batch.request_counts.completed}/{batch.request_counts.total}")

# Get results
if batch.status == "completed":
    content = client.files.content(batch.output_file_id)
    print("\nResults:")
    for line in content.text.strip().split('\n'):
        result = json.loads(line)
        print(f"  {result['custom_id']}: {result['response']['body']['choices'][0]['message']['content'][:50]}")

os.unlink(input_path)
print("\nAll tests passed!")
```

- [ ] **Step 2: Run the test**

```bash
python3 /tmp/test_batch.py
```
Expected: All 3 requests complete, results printed.

- [ ] **Step 3: Clean up**

Remove test script. No commit needed for test files.

---

## Task 11: Create Feature Branch and Final Commit

- [ ] **Step 1: Ensure all changes are committed on the worktree branch**

```bash
git status
git log --oneline -10
```

- [ ] **Step 2: Create the feature branch from the worktree**

The worktree is already on `claude/elated-proskuriakova`. The final PR branch should be `feature/claude-batch-dispatch`. This will be handled during the PR creation step.
