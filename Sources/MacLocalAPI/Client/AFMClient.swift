// MARK: - AFMClient — Zero-dependency OpenAI-compatible API client for AFM
//
// Uses only URLSession and Foundation. Thread-safe, works from any actor context.
// Covers 100% of AFM's API contract: health, models, chat completions (streaming
// and non-streaming), batch API, SSE multiplex, multimodal, tool calling, logprobs,
// GPU profiling, and think/reasoning extraction.
//
// All nested types live in `extension AFMClient { ... }` in AFMClientTypes.swift
// to avoid name collisions with the server's internal types in the same module.
// Users reference them as `AFMClient.ChatRequest`, `AFMClient.Usage`, etc.

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// MARK: - AFMClient

/// A thread-safe, zero-dependency OpenAI-compatible API client for AFM servers.
///
/// All request/response types are nested inside this class to avoid collisions
/// with the server-side types. Reference them as `AFMClient.ChatRequest`, etc.
///
/// Usage:
/// ```swift
/// let client = AFMClient(baseURL: "http://localhost:9998")
///
/// // Non-streaming
/// let response = try await client.chatCompletion(
///     .init(messages: [AFMClient.userMessage("Hello!")])
/// )
/// print(response.text ?? "")
///
/// // Streaming
/// let stream = try await client.chatCompletionStream(
///     .init(messages: [AFMClient.userMessage("Hello!")])
/// )
/// for try await chunk in stream {
///     print(chunk.text ?? "", terminator: "")
/// }
/// ```
public final class AFMClient: @unchecked Sendable {

    // MARK: - Properties

    /// Base URL of the AFM server (e.g. "http://localhost:9998").
    public let baseURL: URL

    /// Optional API key sent as `Authorization: Bearer <key>`.
    public let apiKey: String?

    /// Additional headers to include in every request.
    public let customHeaders: [String: String]

    /// URLSession used for all requests.
    public let session: URLSession

    /// JSON decoder. Uses explicit CodingKeys in all types for reliable mapping.
    public let decoder: JSONDecoder

    /// JSON encoder. Uses explicit CodingKeys in all types for reliable mapping.
    public let encoder: JSONEncoder

    // MARK: - Init

    /// Creates a new AFM API client.
    ///
    /// - Parameters:
    ///   - baseURL: Server URL (default: "http://localhost:9998").
    ///   - apiKey: Optional Bearer token for authentication.
    ///   - customHeaders: Additional headers for every request.
    ///   - session: URLSession to use (default: `.shared`).
    public init(
        baseURL: String = "http://localhost:9998",
        apiKey: String? = nil,
        customHeaders: [String: String] = [:],
        session: URLSession = .shared
    ) {
        self.baseURL = URL(string: baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL)!
        self.apiKey = apiKey
        self.customHeaders = customHeaders
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
    }

    // MARK: - Health

    /// Check server health.
    ///
    /// - Returns: Health status including version and timestamp.
    public func health() async throws -> HealthResponse {
        return try await get(path: "/health")
    }

    // MARK: - Models

    /// List available models.
    ///
    /// - Returns: List of model objects.
    public func models() async throws -> ModelsResponse {
        return try await get(path: "/v1/models")
    }

    // MARK: - Chat Completions (Non-Streaming)

    /// Send a chat completion request (non-streaming).
    ///
    /// - Parameter request: The chat request. `stream` will be forced to `false`.
    /// - Returns: Complete chat completion response.
    public func chatCompletion(_ request: ChatRequest) async throws -> ChatCompletionResponse {
        var req = request
        req.stream = false
        return try await post(path: "/v1/chat/completions", body: req)
    }

    /// Send a chat completion request with GPU profiling (non-streaming).
    ///
    /// - Parameters:
    ///   - request: The chat request.
    ///   - profile: Profiling level (`basic` or `extended`).
    /// - Returns: Complete chat completion response with profiling data.
    public func chatCompletion(
        _ request: ChatRequest,
        profile: ProfileLevel?
    ) async throws -> ChatCompletionResponse {
        var req = request
        req.stream = false
        var extraHeaders: [String: String] = [:]
        if let profile {
            extraHeaders["X-AFM-Profile"] = profile.rawValue
        }
        return try await post(path: "/v1/chat/completions", body: req, extraHeaders: extraHeaders)
    }

    // MARK: - Chat Completions (Streaming)

    /// Send a streaming chat completion request.
    ///
    /// - Parameter request: The chat request. `stream` will be forced to `true`.
    /// - Returns: An async stream of chat completion chunks. Completes on `[DONE]`.
    public func chatCompletionStream(
        _ request: ChatRequest
    ) async throws -> AsyncThrowingStream<ChatCompletionChunk, Error> {
        var req = request
        req.stream = true
        return try await postSSE(path: "/v1/chat/completions", body: req)
    }

    /// Send a streaming chat completion request with GPU profiling.
    ///
    /// - Parameters:
    ///   - request: The chat request.
    ///   - profile: Profiling level (`basic` or `extended`).
    /// - Returns: An async stream of chat completion chunks.
    public func chatCompletionStream(
        _ request: ChatRequest,
        profile: ProfileLevel?
    ) async throws -> AsyncThrowingStream<ChatCompletionChunk, Error> {
        var req = request
        req.stream = true
        var extraHeaders: [String: String] = [:]
        if let profile {
            extraHeaders["X-AFM-Profile"] = profile.rawValue
        }
        return try await postSSE(path: "/v1/chat/completions", body: req, extraHeaders: extraHeaders)
    }

    // MARK: - File API

    /// Upload a file for batch processing.
    ///
    /// - Parameters:
    ///   - data: File contents (typically JSONL).
    ///   - filename: Name of the file.
    ///   - purpose: Purpose string (e.g. "batch").
    /// - Returns: The created file object.
    public func uploadFile(
        data: Data,
        filename: String,
        purpose: String = "batch"
    ) async throws -> FileObject {
        let boundary = "AFMClient-\(UUID().uuidString)"
        var bodyData = Data()

        // purpose field
        bodyData.afmAppendMultipartField(name: "purpose", value: purpose, boundary: boundary)
        // file field
        bodyData.afmAppendMultipartFile(name: "file", filename: filename, data: data, boundary: boundary)
        // closing boundary
        bodyData.append("--\(boundary)--\r\n".data(using: .utf8)!)

        let url = baseURL.appendingPathComponent("/v1/files")
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = bodyData
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        applyCommonHeaders(to: &urlRequest)

        return try await executeRequest(urlRequest)
    }

    /// Retrieve a file object by ID.
    ///
    /// - Parameter id: The file ID.
    /// - Returns: The file object.
    public func getFile(id: String) async throws -> FileObject {
        return try await get(path: "/v1/files/\(id)")
    }

    /// Retrieve the content of a file.
    ///
    /// - Parameter id: The file ID.
    /// - Returns: Raw file content bytes.
    public func getFileContent(id: String) async throws -> Data {
        let url = baseURL.appendingPathComponent("/v1/files/\(id)/content")
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"
        applyCommonHeaders(to: &urlRequest)

        let (data, response) = try await performRequest(urlRequest)
        try validateHTTPResponse(response, data: data)
        return data
    }

    /// Delete a file.
    ///
    /// - Parameter id: The file ID.
    /// - Returns: Deletion confirmation.
    public func deleteFile(id: String) async throws -> FileDeleteResponse {
        return try await delete(path: "/v1/files/\(id)")
    }

    // MARK: - Batch API

    /// Create a batch processing job.
    ///
    /// - Parameters:
    ///   - inputFileId: ID of the uploaded input file.
    ///   - endpoint: The endpoint to process (e.g. "/v1/chat/completions").
    ///   - completionWindow: Optional completion window (e.g. "24h").
    /// - Returns: The created batch object.
    public func createBatch(
        inputFileId: String,
        endpoint: String = "/v1/chat/completions",
        completionWindow: String? = nil
    ) async throws -> BatchObject {
        let body = BatchCreateBody(
            inputFileId: inputFileId,
            endpoint: endpoint,
            completionWindow: completionWindow
        )
        return try await post(path: "/v1/batches", body: body)
    }

    /// Retrieve a batch by ID.
    ///
    /// - Parameter id: The batch ID.
    /// - Returns: The batch object.
    public func getBatch(id: String) async throws -> BatchObject {
        return try await get(path: "/v1/batches/\(id)")
    }

    /// List all batches.
    ///
    /// - Returns: Paginated list of batch objects.
    public func listBatches() async throws -> BatchListResponse {
        return try await get(path: "/v1/batches")
    }

    /// Cancel a batch.
    ///
    /// - Parameter id: The batch ID.
    /// - Returns: The updated batch object.
    public func cancelBatch(id: String) async throws -> BatchObject {
        return try await post(path: "/v1/batches/\(id)/cancel", body: AFMEmptyBody())
    }

    /// Stream batch completions via SSE multiplex.
    ///
    /// - Parameter requests: Array of batch request items.
    /// - Returns: Async stream of chat completion chunks (interleaved from all requests).
    public func batchCompletionStream(
        requests: [BatchRequestItem]
    ) async throws -> AsyncThrowingStream<ChatCompletionChunk, Error> {
        let body = BatchCompletionBody(requests: requests)
        return try await postSSE(path: "/v1/batch/completions", body: body)
    }

    // MARK: - Convenience Message Builders

    /// Create a message with any role and text content.
    public static func message(_ role: String, _ content: String) -> ChatMessage {
        ChatMessage(role: role, text: content)
    }

    /// Create a user message with text content.
    public static func userMessage(_ text: String) -> ChatMessage {
        ChatMessage(role: "user", text: text)
    }

    /// Create a user message with multimodal content parts.
    public static func userMessage(_ parts: [ContentPart]) -> ChatMessage {
        ChatMessage(role: "user", content: .parts(parts))
    }

    /// Create a system message.
    public static func systemMessage(_ text: String) -> ChatMessage {
        ChatMessage(role: "system", text: text)
    }

    /// Create an assistant message.
    public static func assistantMessage(_ text: String) -> ChatMessage {
        ChatMessage(role: "assistant", text: text)
    }

    /// Create a tool result message.
    public static func toolResult(callId: String, content: String) -> ChatMessage {
        ChatMessage(role: "tool", content: .text(content), toolCallId: callId)
    }

    // MARK: - Image Helpers

    /// Create an image content part from a URL.
    public static func imageFromURL(_ url: String, detail: ImageDetail? = nil) -> ContentPart {
        .imageURL(url: url, detail: detail)
    }

    /// Create an image content part from base64-encoded data.
    public static func imageFromBase64(
        _ base64: String,
        mediaType: String = "image/png",
        detail: ImageDetail? = nil
    ) -> ContentPart {
        .imageData(base64: base64, mediaType: mediaType, detail: detail)
    }

    /// Create an image content part from raw `Data`.
    public static func imageFromData(
        _ data: Data,
        mediaType: String = "image/png",
        detail: ImageDetail? = nil
    ) -> ContentPart {
        .imageData(base64: data.base64EncodedString(), mediaType: mediaType, detail: detail)
    }

    /// Create a sequence of image content parts representing video frames.
    ///
    /// - Parameters:
    ///   - frames: Array of (data, mediaType) tuples for each frame.
    ///   - detail: Image detail level for all frames.
    /// - Returns: Array of image content parts.
    public static func videoFrames(
        _ frames: [(data: Data, mediaType: String)],
        detail: ImageDetail? = nil
    ) -> [ContentPart] {
        frames.map { frame in
            .imageData(
                base64: frame.data.base64EncodedString(),
                mediaType: frame.mediaType,
                detail: detail
            )
        }
    }
}

// ============================================================================
// MARK: - Internal Request Body Types
// ============================================================================

extension AFMClient {

    /// Body for POST /v1/batches.
    struct BatchCreateBody: Encodable {
        var inputFileId: String
        var endpoint: String
        var completionWindow: String?

        enum CodingKeys: String, CodingKey {
            case inputFileId = "input_file_id"
            case endpoint
            case completionWindow = "completion_window"
        }
    }

    /// Body for POST /v1/batch/completions.
    struct BatchCompletionBody: Encodable {
        var requests: [BatchRequestItem]
    }
}

/// Empty encodable body for POST endpoints that don't require a body.
private struct AFMEmptyBody: Encodable {}

// ============================================================================
// MARK: - Internal HTTP Layer
// ============================================================================

extension AFMClient {

    // MARK: GET

    private func get<T: Decodable>(path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"
        applyCommonHeaders(to: &urlRequest)
        return try await executeRequest(urlRequest)
    }

    // MARK: POST (JSON body)

    private func post<B: Encodable, T: Decodable>(
        path: String,
        body: B,
        extraHeaders: [String: String] = [:]
    ) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        applyCommonHeaders(to: &urlRequest)
        for (key, value) in extraHeaders {
            urlRequest.setValue(value, forHTTPHeaderField: key)
        }

        do {
            urlRequest.httpBody = try encoder.encode(body)
        } catch {
            throw AFMClientError.decodingError(error)
        }

        return try await executeRequest(urlRequest)
    }

    // MARK: DELETE

    private func delete<T: Decodable>(path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "DELETE"
        applyCommonHeaders(to: &urlRequest)
        return try await executeRequest(urlRequest)
    }

    // MARK: POST -> SSE Stream

    private func postSSE<B: Encodable>(
        path: String,
        body: B,
        extraHeaders: [String: String] = [:]
    ) async throws -> AsyncThrowingStream<ChatCompletionChunk, Error> {
        let url = baseURL.appendingPathComponent(path)
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        applyCommonHeaders(to: &urlRequest)
        for (key, value) in extraHeaders {
            urlRequest.setValue(value, forHTTPHeaderField: key)
        }

        do {
            urlRequest.httpBody = try encoder.encode(body)
        } catch {
            throw AFMClientError.decodingError(error)
        }

        let (asyncBytes, response) = try await performBytes(urlRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AFMClientError.connectionError(
                URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "Non-HTTP response"])
            )
        }

        // For non-2xx on SSE, try to read the body as an error
        guard (200...299).contains(httpResponse.statusCode) else {
            var errorData = Data()
            for try await byte in asyncBytes {
                errorData.append(byte)
                // Cap at 64KB to prevent memory issues
                if errorData.count > 65536 { break }
            }
            let detail = try? decoder.decode(APIError.self, from: errorData).error
            throw AFMClientError.httpError(statusCode: httpResponse.statusCode, detail: detail)
        }

        let dec = self.decoder
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let parser = SSEParser(bytes: asyncBytes, decoder: dec)
                    for try await chunk in parser {
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    // MARK: Common Helpers

    private func applyCommonHeaders(to request: inout URLRequest) {
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        for (key, value) in customHeaders {
            request.setValue(value, forHTTPHeaderField: key)
        }
    }

    private func executeRequest<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await performRequest(request)
        try validateHTTPResponse(response, data: data)

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw AFMClientError.decodingError(error)
        }
    }

    private func performRequest(_ request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await session.data(for: request)
        } catch let error as URLError {
            throw AFMClientError.connectionError(error)
        } catch {
            throw AFMClientError.connectionError(error)
        }
    }

    private func performBytes(_ request: URLRequest) async throws -> (URLSession.AsyncBytes, URLResponse) {
        do {
            return try await session.bytes(for: request)
        } catch let error as URLError {
            throw AFMClientError.connectionError(error)
        } catch {
            throw AFMClientError.connectionError(error)
        }
    }

    private func validateHTTPResponse(_ response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AFMClientError.connectionError(
                URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "Non-HTTP response"])
            )
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            let detail = try? decoder.decode(APIError.self, from: data).error
            throw AFMClientError.httpError(statusCode: httpResponse.statusCode, detail: detail)
        }
    }
}

// ============================================================================
// MARK: - SSE Parser
// ============================================================================

extension AFMClient {

    /// Internal SSE (Server-Sent Events) line parser.
    ///
    /// Reads `URLSession.AsyncBytes` line-by-line, accumulates `data:` lines,
    /// yields parsed `ChatCompletionChunk` on blank line boundaries, and
    /// completes on the `data: [DONE]` sentinel.
    struct SSEParser: AsyncSequence {
        typealias Element = ChatCompletionChunk

        let bytes: URLSession.AsyncBytes
        let decoder: JSONDecoder

        struct AsyncIterator: AsyncIteratorProtocol {
            var lineIterator: AsyncLineSequence<URLSession.AsyncBytes>.AsyncIterator
            let decoder: JSONDecoder
            var dataBuffer: String = ""
            var done: Bool = false

            mutating func next() async throws -> ChatCompletionChunk? {
                guard !done else { return nil }

                while let line = try await lineIterator.next() {
                    // SSE spec: lines starting with ":" are comments, skip them
                    if line.hasPrefix(":") {
                        continue
                    }

                    // data: line -- parse immediately (AFM sends one JSON per data: line)
                    if line.hasPrefix("data:") {
                        let payload = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)

                        // [DONE] sentinel -- end of stream
                        if payload == "[DONE]" {
                            done = true
                            return nil
                        }

                        // Empty data line — skip
                        if payload.isEmpty {
                            continue
                        }

                        // Try to parse as a complete JSON chunk immediately
                        // (AFM sends one complete JSON object per data: line)
                        if payload.hasPrefix("{") {
                            if let jsonData = payload.data(using: .utf8) {
                                do {
                                    return try decoder.decode(ChatCompletionChunk.self, from: jsonData)
                                } catch {
                                    // If decoding fails, fall through to buffer accumulation
                                    // (may be a multi-line data payload)
                                }
                            }
                        }

                        // Accumulate data lines (SSE spec allows multi-line data)
                        if !dataBuffer.isEmpty {
                            dataBuffer += "\n"
                        }
                        dataBuffer += payload
                        continue
                    }

                    // event: line -- we don't use event types but consume them
                    if line.hasPrefix("event:") {
                        continue
                    }

                    // id: line -- SSE last event ID, we don't use it
                    if line.hasPrefix("id:") {
                        continue
                    }

                    // retry: line -- SSE reconnection interval, skip
                    if line.hasPrefix("retry:") {
                        continue
                    }

                    // Blank line = event boundary -- process accumulated data
                    if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        if let chunk = try processBuffer() {
                            return chunk
                        }
                        continue
                    }

                    // Unknown line format -- skip silently (SSE spec says ignore unknown fields)
                }

                // Stream ended without [DONE] -- process any remaining buffer
                if !dataBuffer.isEmpty {
                    let result = try processBuffer()
                    done = true
                    return result
                }

                done = true
                return nil
            }

            /// Decode the accumulated data buffer into a chunk. Clears the buffer.
            private mutating func processBuffer() throws -> ChatCompletionChunk? {
                guard !dataBuffer.isEmpty else { return nil }

                let payload = dataBuffer
                dataBuffer = ""

                if payload == "[DONE]" {
                    done = true
                    return nil
                }

                guard let jsonData = payload.data(using: .utf8) else {
                    throw AFMClientError.streamError("Invalid UTF-8 in SSE data")
                }

                do {
                    return try decoder.decode(ChatCompletionChunk.self, from: jsonData)
                } catch {
                    throw AFMClientError.decodingError(error)
                }
            }
        }

        func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(
                lineIterator: bytes.lines.makeAsyncIterator(),
                decoder: decoder
            )
        }
    }
}

// ============================================================================
// MARK: - Multipart Form Data Helpers
// ============================================================================

// Prefixed with `afm` to avoid collisions with any Data extensions elsewhere
// in the module. These are internal helpers for the file upload endpoint.

extension Data {
    /// Append a simple form field to multipart body.
    mutating func afmAppendMultipartField(name: String, value: String, boundary: String) {
        append("--\(boundary)\r\n".data(using: .utf8)!)
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n".data(using: .utf8)!)
        append("\r\n".data(using: .utf8)!)
        append("\(value)\r\n".data(using: .utf8)!)
    }

    /// Append a file field to multipart body.
    mutating func afmAppendMultipartFile(name: String, filename: String, data: Data, boundary: String) {
        append("--\(boundary)\r\n".data(using: .utf8)!)
        append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        append("Content-Type: application/octet-stream\r\n".data(using: .utf8)!)
        append("\r\n".data(using: .utf8)!)
        append(data)
        append("\r\n".data(using: .utf8)!)
    }
}
