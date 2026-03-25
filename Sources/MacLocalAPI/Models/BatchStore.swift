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
