import Vapor
import Foundation

/// A small, transport-owned JSON value used by the Responses adapter. Keeping
/// this type in AFMServer avoids adding Responses API wire types to AFMKit's
/// provider contracts.
enum ResponsesJSON: Codable, Sendable, Equatable {
    case object([String: ResponsesJSON])
    case array([ResponsesJSON])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([ResponsesJSON].self) { self = .array(value) }
        else { self = .object(try container.decode([String: ResponsesJSON].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    var objectValue: [String: ResponsesJSON]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    var arrayValue: [ResponsesJSON]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    var intValue: Int? {
        guard case .number(let value) = self else { return nil }
        return Int(value)
    }

    subscript(_ key: String) -> ResponsesJSON? { objectValue?[key] }
}

struct StoredResponse: Sendable {
    let messages: [ResponsesJSON]
    let resource: ResponsesJSON
}

actor ResponsesStore {
    private static let maximumStoredResponses = 1_024
    private var responses: [String: StoredResponse] = [:]
    private var insertionOrder: [String] = []

    func get(_ id: String) -> StoredResponse? { responses[id] }

    func put(id: String, messages: [ResponsesJSON], resource: ResponsesJSON) {
        if responses[id] == nil {
            insertionOrder.append(id)
        }
        responses[id] = StoredResponse(messages: messages, resource: resource)
        while insertionOrder.count > Self.maximumStoredResponses {
            responses.removeValue(forKey: insertionOrder.removeFirst())
        }
    }
}

/// Implements the OpenAI Responses surface by translating it at the HTTP
/// boundary into the already-qualified Chat Completions pipeline. This keeps
/// model loading, media preflight, structured output, tools, and reasoning in
/// one inference path.
struct ResponsesController: RouteCollection {
    typealias ChatHandler = @Sendable (Request) async throws -> Response

    private let defaultModelID: String
    private let chatHandler: ChatHandler
    private let store: ResponsesStore

    init(
        defaultModelID: String,
        store: ResponsesStore = ResponsesStore(),
        chatHandler: @escaping ChatHandler
    ) {
        self.defaultModelID = defaultModelID
        self.store = store
        self.chatHandler = chatHandler
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "responses", body: .collect(maxSize: "100mb"), use: createResponse)
        v1.get("responses", ":response_id", use: getResponse)
        v1.on(.OPTIONS, "responses", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, GET, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization, X-AFM-Profile")
        return response
    }

    func getResponse(req: Request) async throws -> Response {
        guard let id = req.parameters.get("response_id"),
              let stored = await store.get(id) else {
            throw Abort(.notFound, reason: "Response not found")
        }
        return try encodeJSONResponse(stored.resource)
    }

    func createResponse(req: Request) async throws -> Response {
        guard let inputData = req.body.data.map({ Data(buffer: $0) }) else {
            throw Abort(.badRequest, reason: "Missing request body")
        }
        let request = try JSONDecoder().decode(ResponsesJSON.self, from: inputData)
        guard let body = request.objectValue else {
            throw Abort(.badRequest, reason: "Request body must be a JSON object")
        }

        let responseID = "resp_\(UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: ""))"
        let model = body["model"]?.stringValue ?? defaultModelID
        let previousID = body["previous_response_id"]?.stringValue
        let priorMessages: [ResponsesJSON]
        if let previousID {
            guard let previous = await store.get(previousID) else {
                throw Abort(.notFound, reason: "Unknown previous_response_id: \(previousID)")
            }
            priorMessages = previous.messages
        } else {
            priorMessages = []
        }

        let newMessages = try Self.translateInput(body)
        let messages = priorMessages + newMessages
        let chatBody = try Self.makeChatBody(body, model: model, messages: messages)
        let background = body["background"]?.boolValue ?? false

        if background {
            let initial = Self.makeResource(
                request: body,
                id: responseID,
                model: model,
                previousID: previousID,
                status: "in_progress",
                output: [],
                usage: nil,
                background: true
            )
            await store.put(id: responseID, messages: messages, resource: initial)

            let handler = chatHandler
            let responseStore = store
            Task {
                do {
                    let completed = try await Self.executeChat(
                        request: req,
                        chatBody: chatBody,
                        requestBody: body,
                        responseID: responseID,
                        model: model,
                        previousID: previousID,
                        background: true,
                        handler: handler
                    )
                    await responseStore.put(
                        id: responseID,
                        messages: messages + completed.assistantMessages,
                        resource: completed.resource
                    )
                } catch {
                    let failed = Self.makeResource(
                        request: body,
                        id: responseID,
                        model: model,
                        previousID: previousID,
                        status: "failed",
                        output: [],
                        usage: nil,
                        background: true,
                        error: error.localizedDescription
                    )
                    await responseStore.put(id: responseID, messages: messages, resource: failed)
                }
            }
            return try encodeJSONResponse(initial, status: .accepted)
        }

        let completed = try await Self.executeChat(
            request: req,
            chatBody: chatBody,
            requestBody: body,
            responseID: responseID,
            model: model,
            previousID: previousID,
            background: false,
            handler: chatHandler
        )
        await store.put(
            id: responseID,
            messages: messages + completed.assistantMessages,
            resource: completed.resource
        )

        if body["stream"]?.boolValue == true {
            return try Self.makeStreamingResponse(completed.resource)
        }
        return try encodeJSONResponse(completed.resource)
    }

    private struct CompletedChat: Sendable {
        let resource: ResponsesJSON
        let assistantMessages: [ResponsesJSON]
    }

    private static func executeChat(
        request: Request,
        chatBody: ResponsesJSON,
        requestBody: [String: ResponsesJSON],
        responseID: String,
        model: String,
        previousID: String?,
        background: Bool,
        handler: ChatHandler
    ) async throws -> CompletedChat {
        let encodedBody = try JSONEncoder().encode(chatBody)
        var buffer = request.byteBufferAllocator.buffer(capacity: encodedBody.count)
        buffer.writeBytes(encodedBody)
        var headers = request.headers
        headers.contentType = .json
        headers.replaceOrAdd(name: .contentLength, value: String(encodedBody.count))
        let chatRequest = Request(
            application: request.application,
            method: .POST,
            url: URI(path: "/v1/chat/completions"),
            version: request.version,
            headersNoUpdate: headers,
            collectedBody: buffer,
            remoteAddress: request.remoteAddress,
            peerCertificateChain: request.peerCertificateChain,
            logger: request.logger,
            byteBufferAllocator: request.byteBufferAllocator,
            on: request.eventLoop
        )
        let chatResponse = try await handler(chatRequest)
        guard (200..<300).contains(chatResponse.status.code) else {
            throw Abort(chatResponse.status, reason: "Chat generation failed")
        }
        guard let buffer = try await chatResponse.body.collect(on: chatRequest.eventLoop).get() else {
            throw Abort(.internalServerError, reason: "Chat generation returned an empty body")
        }
        let envelope = try JSONDecoder().decode(ResponsesJSON.self, from: Data(buffer: buffer))
        guard let choice = envelope["choices"]?.arrayValue?.first?.objectValue,
              let message = choice["message"]?.objectValue else {
            throw Abort(.internalServerError, reason: "Chat generation returned an invalid response")
        }

        let output = makeOutput(message: message)
        let usage = makeUsage(envelope["usage"])
        let finishReason = choice["finish_reason"]?.stringValue
        let responseStatus = finishReason == "length" ? "incomplete" : "completed"
        let resource = makeResource(
            request: requestBody,
            id: responseID,
            model: envelope["model"]?.stringValue ?? model,
            previousID: previousID,
            status: responseStatus,
            output: output,
            usage: usage,
            background: background,
            incompleteReason: finishReason == "length" ? "max_output_tokens" : nil
        )
        return CompletedChat(resource: resource, assistantMessages: [makeStoredAssistantMessage(message)])
    }

    private static func translateInput(_ body: [String: ResponsesJSON]) throws -> [ResponsesJSON] {
        var messages: [ResponsesJSON] = []
        if let instructions = body["instructions"]?.stringValue, !instructions.isEmpty {
            messages.append(.object(["role": .string("system"), "content": .string(instructions)]))
        }

        guard let input = body["input"] else {
            throw Abort(.badRequest, reason: "input is required")
        }
        if let text = input.stringValue {
            messages.append(.object(["role": .string("user"), "content": .string(text)]))
            return messages
        }
        guard let items = input.arrayValue else {
            throw Abort(.badRequest, reason: "input must be a string or an array")
        }

        for item in items {
            guard let object = item.objectValue else { continue }
            switch object["type"]?.stringValue {
            case "function_call":
                let callID = object["call_id"]?.stringValue ?? "call_\(UUID().uuidString.lowercased().prefix(12))"
                messages.append(.object([
                    "role": .string("assistant"),
                    "content": .null,
                    "tool_calls": .array([.object([
                        "id": .string(callID),
                        "type": .string("function"),
                        "function": .object([
                            "name": object["name"] ?? .string(""),
                            "arguments": object["arguments"] ?? .string("{}")
                        ])
                    ])])
                ]))
            case "function_call_output":
                messages.append(.object([
                    "role": .string("tool"),
                    "tool_call_id": object["call_id"] ?? .string(""),
                    "content": object["output"] ?? .string("")
                ]))
            default:
                let role = object["role"]?.stringValue ?? "user"
                let content = translateContent(object["content"])
                messages.append(.object(["role": .string(role), "content": content]))
            }
        }
        return messages
    }

    private static func translateContent(_ value: ResponsesJSON?) -> ResponsesJSON {
        if let text = value?.stringValue { return .string(text) }
        guard let parts = value?.arrayValue else { return .string("") }
        return .array(parts.compactMap { part in
            guard let object = part.objectValue else { return nil }
            switch object["type"]?.stringValue {
            case "input_image":
                guard let imageURL = object["image_url"] else { return nil }
                return .object([
                    "type": .string("image_url"),
                    "image_url": .object(["url": imageURL])
                ])
            case "input_text", "output_text", "text":
                return .object([
                    "type": .string("text"),
                    "text": object["text"] ?? .string("")
                ])
            default:
                return nil
            }
        })
    }

    private static func makeChatBody(
        _ request: [String: ResponsesJSON],
        model: String,
        messages: [ResponsesJSON]
    ) throws -> ResponsesJSON {
        var body: [String: ResponsesJSON] = [
            "model": .string(model),
            "messages": .array(messages),
            "stream": .bool(false)
        ]
        let mappings = [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_output_tokens", "max_tokens"),
            ("parallel_tool_calls", "parallel_tool_calls"),
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty")
        ]
        for (source, destination) in mappings where request[source] != nil {
            body[destination] = request[source]
        }
        if let effort = request["reasoning"]?["effort"]?.stringValue {
            body["reasoning_effort"] = .string(effort)
        }
        if let tools = request["tools"]?.arrayValue {
            body["tools"] = .array(tools.compactMap { tool in
                guard let object = tool.objectValue,
                      object["type"]?.stringValue == "function" else { return nil }
                return .object([
                    "type": .string("function"),
                    "function": .object([
                        "name": object["name"] ?? .string(""),
                        "description": object["description"] ?? .null,
                        "parameters": object["parameters"] ?? .null,
                        "strict": object["strict"] ?? .null
                    ])
                ])
            })
        }
        if let choice = request["tool_choice"] {
            if let object = choice.objectValue, object["type"]?.stringValue == "function" {
                body["tool_choice"] = .object([
                    "type": .string("function"),
                    "function": .object(["name": object["name"] ?? .string("")])
                ])
            } else {
                body["tool_choice"] = choice
            }
        }
        if let format = request["text"]?["format"]?.objectValue,
           let type = format["type"]?.stringValue {
            if type == "json_schema" {
                body["response_format"] = .object([
                    "type": .string("json_schema"),
                    "json_schema": .object([
                        "name": format["name"] ?? .string("response"),
                        "description": format["description"] ?? .null,
                        "schema": format["schema"] ?? .object([:]),
                        "strict": format["strict"] ?? .bool(true)
                    ])
                ])
            } else if type == "json_object" {
                body["response_format"] = .object(["type": .string("json_object")])
            }
        }
        return .object(body)
    }

    private static func makeOutput(message: [String: ResponsesJSON]) -> [ResponsesJSON] {
        var output: [ResponsesJSON] = []
        if let reasoning = message["reasoning_content"]?.stringValue, !reasoning.isEmpty {
            output.append(.object([
                "type": .string("reasoning"),
                "id": .string("rs_\(UUID().uuidString.lowercased().prefix(12))"),
                "content": .array([.object(["type": .string("reasoning_text"), "text": .string(reasoning)])]),
                "summary": .array([])
            ]))
        }
        if let toolCalls = message["tool_calls"]?.arrayValue {
            for call in toolCalls {
                guard let object = call.objectValue,
                      let function = object["function"]?.objectValue else { continue }
                let callID = object["id"]?.stringValue ?? "call_\(UUID().uuidString.lowercased().prefix(12))"
                output.append(.object([
                    "type": .string("function_call"),
                    "id": .string("fc_\(UUID().uuidString.lowercased().prefix(12))"),
                    "call_id": .string(callID),
                    "name": function["name"] ?? .string(""),
                    "arguments": function["arguments"] ?? .string("{}"),
                    "status": .string("completed")
                ]))
            }
        }
        if let content = message["content"]?.stringValue, !content.isEmpty || output.isEmpty {
            output.append(.object([
                "type": .string("message"),
                "id": .string("msg_\(UUID().uuidString.lowercased().prefix(12))"),
                "status": .string("completed"),
                "role": .string("assistant"),
                "content": .array([.object([
                    "type": .string("output_text"),
                    "text": .string(content),
                    "annotations": .array([])
                ])])
            ]))
        }
        return output
    }

    private static func makeStoredAssistantMessage(_ message: [String: ResponsesJSON]) -> ResponsesJSON {
        var stored: [String: ResponsesJSON] = ["role": .string("assistant")]
        stored["content"] = message["content"] ?? .null
        if let toolCalls = message["tool_calls"] { stored["tool_calls"] = toolCalls }
        if let reasoning = message["reasoning_content"] { stored["reasoning_content"] = reasoning }
        return .object(stored)
    }

    private static func makeUsage(_ chatUsage: ResponsesJSON?) -> ResponsesJSON? {
        guard let usage = chatUsage?.objectValue else { return nil }
        let input = usage["prompt_tokens"]?.intValue ?? 0
        let output = usage["completion_tokens"]?.intValue ?? 0
        let cached = usage["prompt_tokens_details"]?["cached_tokens"]?.intValue ?? 0
        return .object([
            "input_tokens": .number(Double(input)),
            "output_tokens": .number(Double(output)),
            "total_tokens": .number(Double(input + output)),
            "input_tokens_details": .object(["cached_tokens": .number(Double(cached))]),
            "output_tokens_details": .object(["reasoning_tokens": .number(0)])
        ])
    }

    private static func makeResource(
        request: [String: ResponsesJSON],
        id: String,
        model: String,
        previousID: String?,
        status: String,
        output: [ResponsesJSON],
        usage: ResponsesJSON?,
        background: Bool,
        error: String? = nil,
        incompleteReason: String? = nil
    ) -> ResponsesJSON {
        let now = Int(Date().timeIntervalSince1970)
        let tools = request["tools"]?.arrayValue?.map { tool -> ResponsesJSON in
            guard var object = tool.objectValue else { return tool }
            if object["description"] == nil { object["description"] = .null }
            if object["parameters"] == nil { object["parameters"] = .null }
            if object["strict"] == nil { object["strict"] = .null }
            return .object(object)
        } ?? []
        let text = request["text"] ?? .object(["format": .object(["type": .string("text")])])
        let reasoning: ResponsesJSON
        if var reasoningObject = request["reasoning"]?.objectValue {
            if reasoningObject["effort"] == nil { reasoningObject["effort"] = .null }
            if reasoningObject["summary"] == nil { reasoningObject["summary"] = .null }
            reasoning = .object(reasoningObject)
        } else {
            reasoning = .null
        }
        let errorValue: ResponsesJSON = error.map {
            .object(["code": .string("server_error"), "message": .string($0)])
        } ?? .null
        return .object([
            "id": .string(id),
            "object": .string("response"),
            "created_at": .number(Double(now)),
            "completed_at": status == "completed" || status == "failed" ? .number(Double(now)) : .null,
            "status": .string(status),
            "incomplete_details": incompleteReason.map {
                .object(["reason": .string($0)])
            } ?? .null,
            "model": .string(model),
            "previous_response_id": previousID.map(ResponsesJSON.string) ?? .null,
            "instructions": request["instructions"] ?? .null,
            "output": .array(output),
            "error": errorValue,
            "tools": .array(tools),
            "tool_choice": request["tool_choice"] ?? .string("auto"),
            "truncation": request["truncation"] ?? .string("disabled"),
            "parallel_tool_calls": request["parallel_tool_calls"] ?? .bool(true),
            "text": text,
            "top_p": request["top_p"] ?? .number(1),
            "presence_penalty": request["presence_penalty"] ?? .number(0),
            "frequency_penalty": request["frequency_penalty"] ?? .number(0),
            "top_logprobs": request["top_logprobs"] ?? .number(0),
            "temperature": request["temperature"] ?? .number(1),
            "reasoning": reasoning,
            "usage": usage ?? .null,
            "max_output_tokens": request["max_output_tokens"] ?? .null,
            "max_tool_calls": request["max_tool_calls"] ?? .null,
            "store": request["store"] ?? .bool(true),
            "background": .bool(background),
            "service_tier": request["service_tier"] ?? .string("default"),
            "metadata": request["metadata"] ?? .object([:]),
            "safety_identifier": request["safety_identifier"] ?? .null,
            "prompt_cache_key": request["prompt_cache_key"] ?? .null
        ])
    }

    private static func makeStreamingResponse(_ resource: ResponsesJSON) throws -> Response {
        guard let object = resource.objectValue,
              let output = object["output"]?.arrayValue else {
            throw Abort(.internalServerError, reason: "Invalid Responses resource")
        }
        var created = object
        created["status"] = .string("in_progress")
        created["completed_at"] = .null
        created["output"] = .array([])
        created["usage"] = .null

        var sequence = 0
        var events: [ResponsesJSON] = [event("response.created", sequence: &sequence, fields: ["response": .object(created)])]
        events.append(event("response.in_progress", sequence: &sequence, fields: ["response": .object(created)]))

        for (index, item) in output.enumerated() {
            guard let itemObject = item.objectValue else { continue }
            let itemID = itemObject["id"]?.stringValue ?? "item_\(index)"
            events.append(event("response.output_item.added", sequence: &sequence, fields: [
                "output_index": .number(Double(index)), "item": item
            ]))
            if itemObject["type"]?.stringValue == "message",
               let part = itemObject["content"]?.arrayValue?.first,
               let text = part["text"]?.stringValue {
                let emptyPart: ResponsesJSON = .object([
                    "type": .string("output_text"), "text": .string(""), "annotations": .array([])
                ])
                events.append(event("response.content_part.added", sequence: &sequence, fields: [
                    "item_id": .string(itemID), "output_index": .number(Double(index)),
                    "content_index": .number(0), "part": emptyPart
                ]))
                if !text.isEmpty {
                    events.append(event("response.output_text.delta", sequence: &sequence, fields: [
                        "item_id": .string(itemID), "output_index": .number(Double(index)),
                        "content_index": .number(0), "delta": .string(text)
                    ]))
                }
                events.append(event("response.output_text.done", sequence: &sequence, fields: [
                    "item_id": .string(itemID), "output_index": .number(Double(index)),
                    "content_index": .number(0), "text": .string(text)
                ]))
                events.append(event("response.content_part.done", sequence: &sequence, fields: [
                    "item_id": .string(itemID), "output_index": .number(Double(index)),
                    "content_index": .number(0), "part": part
                ]))
            } else if itemObject["type"]?.stringValue == "function_call" {
                let arguments = itemObject["arguments"]?.stringValue ?? "{}"
                events.append(event("response.function_call_arguments.delta", sequence: &sequence, fields: [
                    "item_id": .string(itemID), "output_index": .number(Double(index)), "delta": .string(arguments)
                ]))
                events.append(event("response.function_call_arguments.done", sequence: &sequence, fields: [
                    "item_id": .string(itemID), "output_index": .number(Double(index)), "arguments": .string(arguments)
                ]))
            }
            events.append(event("response.output_item.done", sequence: &sequence, fields: [
                "output_index": .number(Double(index)), "item": item
            ]))
        }
        let terminalType = object["status"]?.stringValue == "incomplete"
            ? "response.incomplete"
            : "response.completed"
        events.append(event(terminalType, sequence: &sequence, fields: ["response": resource]))

        let encoder = JSONEncoder()
        let chunks = try events.map { event -> String in
            let type = event["type"]?.stringValue ?? "message"
            let data = try encoder.encode(event)
            return "event: \(type)\ndata: \(String(decoding: data, as: UTF8.self))\n\n"
        }
        let response = Response(status: .ok)
        response.headers.replaceOrAdd(name: .contentType, value: "text/event-stream")
        response.headers.replaceOrAdd(name: .cacheControl, value: "no-cache")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(string: chunks.joined() + "data: [DONE]\n\n")
        return response
    }

    private static func event(
        _ type: String,
        sequence: inout Int,
        fields: [String: ResponsesJSON]
    ) -> ResponsesJSON {
        var object = fields
        object["type"] = .string(type)
        object["sequence_number"] = .number(Double(sequence))
        sequence += 1
        return .object(object)
    }

    private func encodeJSONResponse(_ value: ResponsesJSON, status: HTTPStatus = .ok) throws -> Response {
        let response = Response(status: status)
        response.headers.contentType = .json
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(data: try JSONEncoder().encode(value))
        return response
    }
}
