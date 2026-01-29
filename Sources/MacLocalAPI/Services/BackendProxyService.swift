import Vapor
import Foundation

/// Hardcoded API key sent to all discovered backends.
/// NOT a security measure — purely for compatibility with backends that require any key.
let afmAPIKey = "afmapikeyunsafe"

actor BackendProxyService {
    private let logger: Logger
    /// Tracks the last model used for proxied requests.
    /// When the model changes, conversation history is stripped to give the new model a clean context.
    private var lastProxiedModel: String?

    init(logger: Logger) {
        self.logger = logger
    }

    /// Notify that a non-proxied model (e.g. Foundation) was used,
    /// so the next proxied request knows the model changed.
    func notifyModelUsed(_ modelId: String) {
        lastProxiedModel = modelId
    }

    /// Proxy a non-streaming chat completion request to an external backend
    func proxyRequest(to baseURL: String, originalModelId: String, backendName: String, request: Request) async throws -> Response {
        let targetURL = URL(string: "\(baseURL)/v1/chat/completions")!

        // Get the raw request body and rewrite the model name to the original backend ID
        guard let body = request.body.data else {
            throw Abort(.badRequest, reason: "Missing request body")
        }
        let stripHistory = lastProxiedModel != nil && lastProxiedModel != originalModelId
        lastProxiedModel = originalModelId
        let bodyData = rewriteModelInBody(Data(buffer: body), to: originalModelId, stripHistory: stripHistory, backendName: backendName)
        if stripHistory {
            logger.info("Model changed from '\(lastProxiedModel ?? "")' — stripping conversation history")
        }
        logger.info("Proxying to \(baseURL) with model: \(originalModelId)")

        var urlRequest = URLRequest(url: targetURL)
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = bodyData
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("Bearer \(afmAPIKey)", forHTTPHeaderField: "Authorization")
        urlRequest.timeoutInterval = 120

        let (data, urlResponse) = try await URLSession.shared.data(for: urlRequest)

        let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 500
        let response = Response(status: HTTPResponseStatus(statusCode: statusCode))
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(data: data)

        return response
    }

    /// Proxy a streaming chat completion request, forwarding SSE lines as they arrive
    func proxyStreamingRequest(to baseURL: String, originalModelId: String, backendName: String, request: Request) async throws -> Response {
        let targetURL = URL(string: "\(baseURL)/v1/chat/completions")!

        guard let body = request.body.data else {
            throw Abort(.badRequest, reason: "Missing request body")
        }
        let stripHistory = lastProxiedModel != nil && lastProxiedModel != originalModelId
        lastProxiedModel = originalModelId
        let bodyData = rewriteModelInBody(Data(buffer: body), to: originalModelId, stripHistory: stripHistory, backendName: backendName)
        if stripHistory {
            logger.info("Model changed — stripping conversation history for clean context")
        }
        logger.info("Streaming proxy to \(baseURL) with model: \(originalModelId)")

        var urlRequest = URLRequest(url: targetURL)
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = bodyData
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("Bearer \(afmAPIKey)", forHTTPHeaderField: "Authorization")
        urlRequest.timeoutInterval = 300

        let loggerCopy = self.logger

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "text/event-stream")
        response.headers.add(name: .cacheControl, value: "no-cache")
        response.headers.add(name: .connection, value: "keep-alive")
        response.headers.add(name: "Access-Control-Allow-Origin", value: "*")
        response.headers.add(name: "Access-Control-Allow-Headers", value: "Content-Type")
        response.headers.add(name: "X-Accel-Buffering", value: "no")

        // Capture the URL request for use in the closure
        let capturedURLRequest = urlRequest

        response.body = .init(asyncStream: { writer in
            do {
                let (bytes, urlResponse) = try await URLSession.shared.bytes(for: capturedURLRequest)

                guard let httpResponse = urlResponse as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 500
                    loggerCopy.error("Backend returned status \(statusCode)")
                    // Read error body for details
                    var errorData = Data()
                    for try await byte in bytes {
                        errorData.append(byte)
                        if errorData.count > 4096 { break }
                    }
                    let detail = String(data: errorData, encoding: .utf8) ?? ""
                    let errorMsg: String
                    switch statusCode {
                    case 401:
                        errorMsg = "⚠️ **Backend Authentication Required (HTTP 401)**\n\nThe backend at \(capturedURLRequest.url?.host ?? ""):\(capturedURLRequest.url?.port ?? 0) requires authentication.\n\nSet the API key to `\(afmAPIKey)` in the backend settings, or disable API key authentication."
                    case 403:
                        errorMsg = "⚠️ **Backend Access Denied (HTTP 403)**\n\n\(detail.isEmpty ? "The backend refused the request." : detail)"
                    case 404:
                        errorMsg = "⚠️ **Model Not Found (HTTP 404)**\n\nThe backend does not have this model available.\n\n\(detail.isEmpty ? "" : detail)"
                    case 500...599:
                        errorMsg = "⚠️ **Backend Error (HTTP \(statusCode))**\n\n\(detail.isEmpty ? "The backend encountered an internal error." : detail)"
                    default:
                        errorMsg = "⚠️ **Backend Error (HTTP \(statusCode))**\n\n\(detail.isEmpty ? "Unexpected error from backend." : detail)"
                    }
                    // Send error as visible chat content via SSE so the user sees it
                    try await Self.writeSSEError(errorMsg, writer: writer)
                    return
                }

                // Track timing for backends that don't provide it (e.g. Ollama)
                let streamStart = Date()
                var firstTokenTime: Date? = nil
                var promptTokens = 0
                var completionTokens = 0
                var lastDataLine: String? = nil

                for try await line in bytes.lines {
                    // Check for stream termination
                    if line.contains("[DONE]") {
                        // Inject timings into the last data chunk if the backend didn't provide them
                        if let last = lastDataLine {
                            let injected = Self.injectTimingsIfMissing(
                                line: last,
                                streamStart: streamStart,
                                firstTokenTime: firstTokenTime,
                                promptTokens: promptTokens,
                                completionTokens: completionTokens
                            )
                            try await writer.write(.buffer(.init(string: "data: \(injected)\n\n")))
                        }

                        try await writer.write(.buffer(.init(string: "\(line)\n\n")))
                        break
                    }

                    // Parse data chunks for timing and token tracking
                    if line.hasPrefix("data:") {
                        let jsonStr = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)

                        if let data = jsonStr.data(using: .utf8),
                           let chunk = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {

                            // Track first-token time from content deltas
                            if let choices = chunk["choices"] as? [[String: Any]],
                               let delta = choices.first?["delta"] as? [String: Any],
                               let content = delta["content"] as? String, !content.isEmpty {
                                if firstTokenTime == nil { firstTokenTime = Date() }
                            }

                            // Read real token counts from usage chunk (sent by Ollama with stream_options.include_usage)
                            if let usage = chunk["usage"] as? [String: Any] {
                                if let pt = usage["prompt_tokens"] as? Int { promptTokens = pt }
                                if let ct = usage["completion_tokens"] as? Int { completionTokens = ct }
                            }
                        }

                        // Buffer the last data line (don't forward yet — we may need to inject timings)
                        if let prev = lastDataLine {
                            try await writer.write(.buffer(.init(string: "data: \(prev)\n\n")))
                        }
                        lastDataLine = jsonStr
                    } else {
                        // Forward non-data lines as-is
                        try await writer.write(.buffer(.init(string: "\(line)\n")))
                    }
                }

                try await writer.write(.end)
            } catch {
                loggerCopy.error("Streaming proxy error: \(error)")
                let errorMsg = "⚠️ **Backend Connection Error**\n\n\(error.localizedDescription)"
                try? await Self.writeSSEError(errorMsg, writer: writer)
            }
        })

        return response
    }

    /// Send an error message as a visible SSE chat response, then close the stream.
    private static func writeSSEError(_ message: String, writer: any AsyncBodyStreamWriter) async throws {
        let streamId = UUID().uuidString
        let chunk: [String: Any] = [
            "id": streamId,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": "error",
            "choices": [[
                "index": 0,
                "delta": ["role": "assistant", "content": message],
                "finish_reason": NSNull()
            ]]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: chunk),
           let json = String(data: data, encoding: .utf8) {
            try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
        }
        // Send finish chunk
        let finish: [String: Any] = [
            "id": streamId,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": "error",
            "choices": [[
                "index": 0,
                "delta": ["content": ""],
                "finish_reason": "stop"
            ]]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: finish),
           let json = String(data: data, encoding: .utf8) {
            try? await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
        }
        try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
        try? await writer.write(.end)
    }

    /// Rewrite the "model" field in the JSON body to the original backend model ID.
    /// When stripHistory is true, removes prior assistant/user exchanges and keeps only
    /// system messages and the last user message, giving the new model a clean context.
    /// For Ollama backends, repacks llama.cpp-specific sampling parameters into the
    /// "options" dict so they aren't silently ignored on the OpenAI-compat endpoint.
    private nonisolated func rewriteModelInBody(_ data: Data, to originalModelId: String, stripHistory: Bool = false, backendName: String = "") -> Data {
        guard var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return data
        }
        json["model"] = originalModelId

        if stripHistory, let messages = json["messages"] as? [[String: Any]] {
            var trimmed: [[String: Any]] = []
            // Keep all system messages
            for msg in messages {
                if (msg["role"] as? String) == "system" {
                    trimmed.append(msg)
                }
            }
            // Keep only the last user message
            if let lastUser = messages.last(where: { ($0["role"] as? String) == "user" }) {
                trimmed.append(lastUser)
            }
            json["messages"] = trimmed
        }

        // Ollama's OpenAI-compat endpoint ignores non-standard top-level params.
        // Repack them into the "options" dict that Ollama does honour.
        if backendName.lowercased() == "ollama" {
            repackOllamaOptions(&json)
        }

        guard let rewritten = try? JSONSerialization.data(withJSONObject: json) else {
            return data
        }
        return rewritten
    }

    /// Inject llama.cpp-style `timings` into the final SSE chunk JSON if the backend didn't include them.
    /// Uses real token counts from the backend's usage data and wall-clock timing from the proxy.
    private static func injectTimingsIfMissing(line: String, streamStart: Date, firstTokenTime: Date?, promptTokens: Int, completionTokens: Int) -> String {
        guard let data = line.data(using: .utf8),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return line
        }

        // Don't override if backend already provides timings
        if json["timings"] != nil { return line }

        let now = Date()
        let promptMs = (firstTokenTime ?? now).timeIntervalSince(streamStart) * 1000
        let predictedMs = now.timeIntervalSince(firstTokenTime ?? streamStart) * 1000

        json["timings"] = [
            "prompt_n": promptTokens,
            "prompt_ms": promptMs,
            "predicted_n": completionTokens,
            "predicted_ms": predictedMs
        ]

        guard let rewritten = try? JSONSerialization.data(withJSONObject: json),
              let result = String(data: rewritten, encoding: .utf8) else {
            return line
        }
        return result
    }

    /// Move llama.cpp sampling parameters from top-level into Ollama's "options" dict.
    /// Top-level OpenAI-standard keys (temperature, top_p, etc.) are kept as-is since
    /// Ollama reads those natively; the non-standard ones are moved into "options".
    private nonisolated func repackOllamaOptions(_ json: inout [String: Any]) {
        // Map from llama.cpp/webui top-level key → Ollama options key
        let paramMapping: [(webui: String, ollama: String)] = [
            ("top_k",          "top_k"),
            ("min_p",          "min_p"),
            ("repeat_penalty", "repeat_penalty"),
            ("repeat_last_n",  "repeat_last_n"),
            ("typical_p",      "typical_p"),
            ("typ_p",          "typical_p"),   // webui alias
            ("mirostat",       "mirostat"),
            ("mirostat_tau",   "mirostat_tau"),
            ("mirostat_eta",   "mirostat_eta"),
            ("num_predict",    "num_predict"),
            ("tfs_z",          "tfs_z"),
        ]

        var options = json["options"] as? [String: Any] ?? [:]

        for (webuiKey, ollamaKey) in paramMapping {
            if let value = json[webuiKey] {
                options[ollamaKey] = value
                json.removeValue(forKey: webuiKey)
            }
        }

        // Also copy temperature and seed into options so Ollama uses them
        // (Ollama reads these from both top-level and options)
        if let temp = json["temperature"] {
            options["temperature"] = temp
        }
        if let seed = json["seed"] {
            options["seed"] = seed
        }

        if !options.isEmpty {
            json["options"] = options
        }

        // Request usage data in streaming responses so we get real token counts
        if json["stream"] as? Bool == true {
            json["stream_options"] = ["include_usage": true]
        }
    }
}
