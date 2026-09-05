import Foundation
import Vapor

/// Anthropic Messages compatibility over AFM's Chat Completions transport.
/// External SSE is deliberately buffered: the internal Chat request is always
/// non-streaming, then emitted as one complete, valid Anthropic lifecycle.
struct MessagesController: RouteCollection {
    typealias ChatHandler = @Sendable (Request) async throws -> Response

    private static let successfulStatusCodes: Range<UInt> = 200..<300
    private static let serverErrorStatusCodes: Range<UInt> = 500..<600
    private static let overloadedStatusCode: UInt = 529

    let model: String
    let chatHandler: ChatHandler

    func boot(routes: RoutesBuilder) throws {
        routes.grouped("v1").on(.POST, "messages", body: .collect(maxSize: "100mb"), use: create)
    }

    func create(req: Request) async throws -> Response {
        do {
            guard let data = req.body.data.map({ Data(buffer: $0) }) else {
                return try Self.error("request body is required")
            }
            let body = try JSONDecoder().decode(ResponsesJSON.self, from: data)
            guard let object = body.objectValue,
                  let sourceMessages = object["messages"]?.arrayValue,
                  let maxTokens = object["max_tokens"]?.intValue else {
                return try Self.error("messages and max_tokens are required")
            }
            let chatRequest = try Self.makeChatRequest(
                object: object, sourceMessages: sourceMessages,
                maxTokens: maxTokens, defaultModel: model
            )
            let result = try await executeChat(chatRequest, from: req)
            let resource = try Self.makeMessage(chat: result, request: object, defaultModel: model)
            return object["stream"]?.boolValue == true
                ? try Self.streamingResponse(resource)
                : try Self.json(resource)
        } catch let abort as AbortError {
            return try Self.error(abort.reason, status: abort.status)
        } catch {
            return try Self.error("invalid_request_error: \(error)")
        }
    }

    private func executeChat(_ body: ResponsesJSON, from request: Request) async throws -> ResponsesJSON {
        let encodedBody = try JSONEncoder().encode(body)
        var buffer = request.byteBufferAllocator.buffer(capacity: encodedBody.count)
        buffer.writeBytes(encodedBody)
        var headers = request.headers
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: String(encodedBody.count))
        headers.replaceOrAdd(name: "X-AFM-Report-Matched-Stop", value: "1")
        let chatRequest = Request(
            application: request.application, method: .POST, url: URI(path: "/v1/chat/completions"),
            version: request.version, headersNoUpdate: headers, collectedBody: buffer,
            remoteAddress: request.remoteAddress, peerCertificateChain: request.peerCertificateChain,
            logger: request.logger, byteBufferAllocator: request.byteBufferAllocator, on: request.eventLoop
        )
        let response = try await chatHandler(chatRequest)
        guard let output = try await response.body.collect(on: chatRequest.eventLoop).get() else {
            throw Abort(.internalServerError, reason: "chat backend returned an empty response")
        }
        let outputData = Data(buffer: output)
        guard Self.successfulStatusCodes.contains(response.status.code) else {
            let decoded = try? JSONDecoder().decode(ResponsesJSON.self, from: outputData)
            let reason = decoded?["error"]?["message"]?.stringValue
                ?? decoded?["detail"]?.stringValue
                ?? "chat backend failed with status \(response.status.code)"
            throw Abort(response.status, reason: reason)
        }
        guard let decoded = try? JSONDecoder().decode(ResponsesJSON.self, from: outputData) else {
            throw Abort(.internalServerError, reason: "chat backend returned malformed JSON")
        }
        guard let encodedStop = response.headers.first(name: "X-AFM-Matched-Stop-Base64"),
              let stopData = Data(base64Encoded: encodedStop),
              let stop = String(data: stopData, encoding: .utf8),
              var object = decoded.objectValue else {
            return decoded
        }
        object["_afm_matched_stop"] = .string(stop)
        return .object(object)
    }

    static func makeChatRequest(object: [String: ResponsesJSON], sourceMessages: [ResponsesJSON], maxTokens: Int, defaultModel: String) throws -> ResponsesJSON {
        var messages: [ResponsesJSON] = []
        if let system = object["system"] {
            let systemText = text(from: system)
            if !systemText.isEmpty {
                messages.append(.object(["role": .string("system"), "content": .string(systemText)]))
            }
        }
        let assistantPrefill = sourceMessages.last.flatMap(prefillText(from:))
        let translatedSources = assistantPrefill == nil ? sourceMessages : Array(sourceMessages.dropLast())
        for source in translatedSources {
            guard let message = source.objectValue,
                  let role = message["role"]?.stringValue,
                  ["user", "assistant"].contains(role) else {
                throw Abort(.badRequest, reason: "messages must contain user or assistant turns")
            }
            messages.append(contentsOf: try chatMessages(from: message, role: role))
        }
        if let assistantPrefill {
            // AFMKit's current ChatCompletionRequest contract has no
            // `continue_final_message` flag. Convert Anthropic's required
            // unfinished-final-assistant semantics into an explicit final turn
            // instead of sending a completed assistant turn that native chat
            // templates close before adding their generation prompt.
            messages.append(.object([
                "role": .string("user"),
                "content": .string(
                    "Continue the unfinished assistant answer below. Return only the exact continuation after the prefix; do not repeat the prefix or start a new answer.\n\nUnfinished assistant prefix:\n\(assistantPrefill)"
                )
            ]))
        }
        var request: [String: ResponsesJSON] = [
            "model": object["model"] ?? .string(defaultModel), "messages": .array(messages),
            "max_tokens": .number(Double(maxTokens)), "stream": .bool(false)
        ]
        for key in ["temperature", "top_p", "top_k", "stop_sequences"] {
            if let value = object[key] { request[key == "stop_sequences" ? "stop" : key] = value }
        }
        if object["thinking"]?["type"]?.stringValue == "enabled" {
            request["reasoning_effort"] = .string("low")
        }
        if assistantPrefill != nil {
            // A final assistant turn is continuation text, not a new reasoning
            // turn. Thinking models can otherwise spend the entire response
            // budget on a hidden scratchpad before emitting the few characters
            // that complete the prefix.
            request["chat_template_kwargs"] = .object([
                "enable_thinking": .bool(false)
            ])
        }
        if let tools = object["tools"]?.arrayValue {
            request["tools"] = .array(tools.compactMap { tool in
                guard let tool = tool.objectValue, let name = tool["name"] else { return nil }
                return .object([
                    "type": .string("function"),
                    "function": .object([
                        "name": name,
                        "description": tool["description"] ?? .null,
                        "parameters": tool["input_schema"] ?? .object([:])
                    ])
                ])
            })
        }
        if let choice = object["tool_choice"] {
            request["tool_choice"] = chatToolChoice(choice)
            if choice["disable_parallel_tool_use"]?.boolValue == true {
                request["parallel_tool_calls"] = .bool(false)
            }
        }
        return .object(request)
    }

    /// Anthropic defines a final assistant text turn as a generation prefill.
    /// Tool-use turns are structured history, not a textual prefill.
    static func prefillText(from source: ResponsesJSON) -> String? {
        guard let message = source.objectValue,
              message["role"]?.stringValue == "assistant" else { return nil }
        if let blocks = message["content"]?.arrayValue,
           blocks.contains(where: { $0["type"]?.stringValue == "tool_use" }) {
            return nil
        }
        let value = text(from: message["content"] ?? .null)
        return value.isEmpty ? nil : value
    }

    /// Converts Message blocks without discarding tool or image semantics.
    static func chatMessages(from message: [String: ResponsesJSON], role: String) throws -> [ResponsesJSON] {
        let content = message["content"] ?? .null
        guard let blocks = content.arrayValue else {
            return [.object(["role": .string(role), "content": content])]
        }
        var textParts: [ResponsesJSON] = []
        var toolCalls: [ResponsesJSON] = []
        var toolResults: [ResponsesJSON] = []
        for block in blocks {
            guard let block = block.objectValue else { continue }
            switch block["type"]?.stringValue {
            case "text", "thinking":
                let value = block["text"] ?? block["thinking"] ?? .string("")
                textParts.append(.object(["type": .string("text"), "text": value]))
            case "image":
                if let image = chatImagePart(block) { textParts.append(image) }
            case "tool_use":
                let input = block["input"] ?? .object([:])
                let arguments = String(decoding: try JSONEncoder().encode(input), as: UTF8.self)
                toolCalls.append(.object([
                    "id": block["id"] ?? .string("call_\(UUID().uuidString.lowercased().prefix(12))"),
                    "type": .string("function"),
                    "function": .object(["name": block["name"] ?? .string(""), "arguments": .string(arguments)])
                ]))
            case "tool_result":
                toolResults.append(.object([
                    "role": .string("tool"),
                    "tool_call_id": block["tool_use_id"] ?? .string(""),
                    "content": .string(text(from: block["content"] ?? .null))
                ]))
            default:
                continue
            }
        }
        var result = toolResults
        let translatedContent: ResponsesJSON = textParts.count == 1 && textParts[0]["type"]?.stringValue == "text"
            ? textParts[0]["text"] ?? .string("") : .array(textParts)
        if !toolCalls.isEmpty {
            result.append(.object([
                "role": .string("assistant"),
                "content": textParts.isEmpty ? .null : translatedContent,
                "tool_calls": .array(toolCalls)
            ]))
        } else if !textParts.isEmpty || toolResults.isEmpty {
            result.append(.object(["role": .string(role), "content": translatedContent]))
        }
        return result
    }

    static func chatImagePart(_ block: [String: ResponsesJSON]) -> ResponsesJSON? {
        guard let source = block["source"]?.objectValue else { return nil }
        if source["type"]?.stringValue == "base64", let data = source["data"]?.stringValue {
            let mediaType = source["media_type"]?.stringValue ?? "image/png"
            return .object(["type": .string("image_url"), "image_url": .object(["url": .string("data:\(mediaType);base64,\(data)")])])
        }
        if source["type"]?.stringValue == "url", let url = source["url"] {
            return .object(["type": .string("image_url"), "image_url": .object(["url": url])])
        }
        return nil
    }

    static func chatToolChoice(_ choice: ResponsesJSON) -> ResponsesJSON {
        guard let object = choice.objectValue else { return choice }
        switch object["type"]?.stringValue {
        case "any": return .string("required")
        case "tool":
            return .object(["type": .string("function"), "function": .object(["name": object["name"] ?? .string("")])])
        case "auto", "none": return object["type"] ?? .string("auto")
        default: return choice
        }
    }

    static func makeMessage(chat: ResponsesJSON, request: [String: ResponsesJSON], defaultModel: String) throws -> ResponsesJSON {
        guard let choice = chat["choices"]?.arrayValue?.first?.objectValue else {
            throw Abort(.internalServerError, reason: "chat backend returned no choices")
        }
        let output = choice["message"]?.objectValue ?? [:]
        let visibleText = output["content"]?.stringValue ?? ""
        let reasoning = output["reasoning_content"]?.stringValue
        let thinkingEnabled = request["thinking"]?["type"]?.stringValue == "enabled"
        var content: [ResponsesJSON] = []
        if thinkingEnabled, let reasoning, !reasoning.isEmpty {
            content.append(.object(["type": .string("thinking"), "thinking": .string(reasoning)]))
        }
        if !visibleText.isEmpty || output["tool_calls"] == nil {
            content.append(.object(["type": .string("text"), "text": .string(visibleText)]))
        }
        if let calls = output["tool_calls"]?.arrayValue {
            for call in calls {
                guard let call = call.objectValue, let function = call["function"]?.objectValue else { continue }
                let rawInput = function["arguments"]?.stringValue ?? "{}"
                let input = (try? JSONDecoder().decode(ResponsesJSON.self, from: Data(rawInput.utf8))) ?? .object([:])
                content.append(.object([
                    "type": .string("tool_use"),
                    "id": call["id"] ?? .string("toolu_\(UUID().uuidString.lowercased().prefix(12))"),
                    "name": function["name"] ?? .string(""),
                    "input": input
                ]))
            }
        }
        let finish = choice["finish_reason"]?.stringValue ?? "stop"
        let stops = request["stop_sequences"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let matchedStop = chat["_afm_matched_stop"]?.stringValue
            ?? stops.first(where: { visibleText.contains($0) })
        let reason = matchedStop != nil ? "stop_sequence" : finish == "length" ? "max_tokens" : finish == "tool_calls" ? "tool_use" : "end_turn"
        let usage = chat["usage"]?.objectValue ?? [:]
        return .object([
            "id": .string("msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"),
            "type": .string("message"), "role": .string("assistant"),
            "model": chat["model"] ?? request["model"] ?? .string(defaultModel), "content": .array(content),
            "stop_reason": .string(reason), "stop_sequence": matchedStop.map(ResponsesJSON.string) ?? .null,
            "usage": .object(["input_tokens": usage["prompt_tokens"] ?? .number(0), "output_tokens": usage["completion_tokens"] ?? .number(0)])
        ])
    }

    static func text(from value: ResponsesJSON) -> String {
        if let text = value.stringValue { return text }
        return value.arrayValue?.compactMap { $0["text"]?.stringValue ?? $0["thinking"]?.stringValue }.joined(separator: "\n") ?? ""
    }

    static func streamingEvents(for message: ResponsesJSON) -> [ResponsesJSON] {
        let content = message["content"]?.arrayValue ?? []
        let finalUsage = message["usage"]?.objectValue ?? [:]
        var initialMessage = message.objectValue ?? [:]
        initialMessage["content"] = .array([])
        initialMessage["stop_reason"] = .null
        initialMessage["stop_sequence"] = .null
        initialMessage["usage"] = .object([
            "input_tokens": finalUsage["input_tokens"] ?? .number(0),
            "output_tokens": .number(0)
        ])
        var events: [ResponsesJSON] = [.object([
            "type": .string("message_start"),
            "message": .object(initialMessage)
        ])]
        for (index, block) in content.enumerated() {
            let type = block["type"]?.stringValue ?? "text"
            let field = type == "thinking" ? "thinking" : type == "tool_use" ? "input" : "text"
            let value = block[field]?.stringValue ?? ""
            var start: [String: ResponsesJSON] = ["type": .string(type)]
            if type == "tool_use" {
                start["id"] = block["id"] ?? .string("")
                start["name"] = block["name"] ?? .string("")
                start["input"] = .object([:])
            } else {
                start[field] = .string("")
            }
            events.append(.object(["type": .string("content_block_start"), "index": .number(Double(index)), "content_block": .object(start)]))
            if type == "tool_use" {
                let input = block["input"] ?? .object([:])
                let partial = (try? String(decoding: JSONEncoder().encode(input), as: UTF8.self)) ?? "{}"
                events.append(.object(["type": .string("content_block_delta"), "index": .number(Double(index)), "delta": .object(["type": .string("input_json_delta"), "partial_json": .string(partial)])]))
            } else if !value.isEmpty {
                events.append(.object(["type": .string("content_block_delta"), "index": .number(Double(index)), "delta": .object(["type": .string(type == "thinking" ? "thinking_delta" : "text_delta"), field: .string(value)])]))
            }
            events.append(.object(["type": .string("content_block_stop"), "index": .number(Double(index))]))
        }
        events.append(.object([
            "type": .string("message_delta"),
            "delta": .object([
                "stop_reason": message["stop_reason"] ?? .string("end_turn"),
                "stop_sequence": message["stop_sequence"] ?? .null
            ]),
            "usage": .object(["output_tokens": finalUsage["output_tokens"] ?? .number(0)])
        ]))
        events.append(.object(["type": .string("message_stop")]))
        return events
    }

    static func streamingResponse(_ message: ResponsesJSON) throws -> Response {
        let frames = try streamingEvents(for: message).map { event in
            let type = event["type"]?.stringValue ?? "message"
            return "event: \(type)\ndata: \(String(decoding: try JSONEncoder().encode(event), as: UTF8.self))\n\n"
        }
        let response = Response(status: .ok)
        response.headers.replaceOrAdd(name: .contentType, value: "text/event-stream")
        response.headers.replaceOrAdd(name: .cacheControl, value: "no-cache")
        response.body = .init(string: frames.joined())
        return response
    }

    static func json(_ value: ResponsesJSON) throws -> Response {
        let response = Response(status: .ok)
        response.headers.contentType = .json
        response.body = .init(data: try JSONEncoder().encode(value))
        return response
    }

    static func error(_ message: String, status: HTTPStatus = .badRequest) throws -> Response {
        let response = Response(status: status)
        response.headers.contentType = .json
        response.body = .init(data: try JSONEncoder().encode(.object([
            "type": .string("error"), "error": .object([
                "type": .string(errorType(for: status)),
                "message": .string(message)
            ])
        ]) as ResponsesJSON))
        return response
    }

    private static func errorType(for status: HTTPStatus) -> String {
        if status.code == overloadedStatusCode { return "overloaded_error" }
        if status == .tooManyRequests { return "rate_limit_error" }
        if serverErrorStatusCodes.contains(status.code) { return "api_error" }
        return "invalid_request_error"
    }
}
