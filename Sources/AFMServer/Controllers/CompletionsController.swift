import AFMKitCore
import AFMOpenAICompat
import Foundation
import Vapor

/// OpenAI-compatible raw-prompt completion endpoint. This deliberately does
/// not translate prompts into chat messages or invoke a chat template.
struct CompletionsController: RouteCollection {
    private let modelID: String
    private let generator: AnyAFMRawTextGenerator
    private let telemetry: AFMServerTelemetryAdapter

    init(
        modelID: String,
        generator: AnyAFMRawTextGenerator,
        telemetry: AFMServerTelemetryAdapter = .standalone()
    ) {
        self.modelID = modelID
        self.generator = generator
        self.telemetry = telemetry
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        v1.on(.POST, "completions", body: .collect(maxSize: "100mb"), use: completions)
        v1.on(.OPTIONS, "completions", use: options)
    }

    private func options(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    private func completions(req: Request) async throws -> Response {
        let request: CompletionRequest
        do {
            request = try req.content.decode(CompletionRequest.self)
        } catch {
            telemetry.recordRejection(.decode)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "Invalid completion request: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }

        guard case .text(let prompt) = request.prompt else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "prompt arrays are not supported; prompt must be one string",
                code: "unsupported_prompt_array"
            )
        }
        guard request.maxTokens.map({ $0 >= 0 }) ?? true else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "max_tokens must be greater than or equal to zero",
                code: "invalid_request_error"
            )
        }
        guard (request.n ?? 1) == 1, (request.bestOf ?? 1) == 1 else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "AFM raw completions currently support only n=1 and best_of=1",
                code: "unsupported_completion_multiplicity"
            )
        }

        let providerRequest = AFMRawTextGenerationRequest(
            prompt: prompt,
            modelID: AFMModelID(rawValue: modelID),
            maximumOutputTokens: request.maxTokens,
            stopSequences: request.stop?.sequences ?? [],
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            repetitionPenalty: request.repetitionPenalty,
            presencePenalty: request.presencePenalty,
            seed: request.seed,
            ignoreEndOfSequence: request.ignoreEOS ?? false
        )
        let events = generator.rawTextGenerationEvents(for: providerRequest)

        if request.stream == true {
            return streamingResponse(
                request: req,
                completionRequest: request,
                events: events
            )
        }
        return try await nonStreamingResponse(request: req, events: events)
    }

    private func nonStreamingResponse(
        request: Request,
        events: AsyncStream<AFMRawTextGenerationEvent>
    ) async throws -> Response {
        var text = ""
        var terminal: AFMRawTextGenerationResult?
        for await event in events {
            switch event {
            case .textDelta(let delta, _, _):
                text += delta
            case .completed(let result):
                terminal = result
            case .failed(let reason, let message):
                return try errorResponse(
                    request: request,
                    status: .internalServerError,
                    message: message,
                    type: reason == .cancelled ? "cancelled" : "server_error",
                    code: reason.rawValue
                )
            }
        }
        guard let terminal else {
            return try errorResponse(
                request: request,
                status: .internalServerError,
                message: "Raw completion provider ended without a terminal event",
                type: "server_error",
                code: "missing_terminal_event"
            )
        }

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(CompletionResponse(
            id: completionID(),
            created: Int(Date().timeIntervalSince1970),
            model: modelID,
            choices: [CompletionChoice(
                text: text,
                finishReason: wireFinishReason(terminal.finishReason)
            )],
            usage: usage(terminal)
        ))
        return response
    }

    private func streamingResponse(
        request: Request,
        completionRequest: CompletionRequest,
        events: AsyncStream<AFMRawTextGenerationEvent>
    ) -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "text/event-stream")
        response.headers.add(name: .cacheControl, value: "no-cache")
        response.headers.add(name: .connection, value: "keep-alive")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: "X-Accel-Buffering", value: "no")

        let id = completionID()
        let created = Int(Date().timeIntervalSince1970)
        let includeUsage = completionRequest.includeStreamingUsage
        response.body = .init(asyncStream: { writer in
            let connectionToken = telemetry.connectionOpened()
            defer { telemetry.connectionClosed(connectionToken) }
            let encoder = JSONEncoder()
            var terminalSeen = false

            for await event in events {
                if terminalSeen { continue }
                switch event {
                case .textDelta(let text, _, _):
                    let chunk = CompletionResponse(
                        id: id,
                        created: created,
                        model: modelID,
                        choices: [CompletionChoice(text: text)]
                    )
                    try? await writeEvent(chunk, encoder: encoder, to: writer)
                case .completed(let result):
                    terminalSeen = true
                    let finish = CompletionResponse(
                        id: id,
                        created: created,
                        model: modelID,
                        choices: [CompletionChoice(
                            text: "",
                            finishReason: wireFinishReason(result.finishReason)
                        )]
                    )
                    try? await writeEvent(finish, encoder: encoder, to: writer)
                    if includeUsage {
                        let usageChunk = CompletionResponse(
                            id: id,
                            created: created,
                            model: modelID,
                            choices: [],
                            usage: usage(result)
                        )
                        try? await writeEvent(usageChunk, encoder: encoder, to: writer)
                    }
                    try? await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                case .failed(let reason, let message):
                    terminalSeen = true
                    let error = OpenAIError(
                        message: message,
                        type: reason == .cancelled ? "cancelled" : "server_error",
                        code: reason.rawValue,
                        requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
                    )
                    try? await writeEvent(error, encoder: encoder, to: writer)
                }
            }
            try? await writer.write(.end)
        })
        return response
    }

    private func errorResponse(
        request: Request,
        status: HTTPResponseStatus,
        message: String,
        type: String = "invalid_request_error",
        code: String
    ) throws -> Response {
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(OpenAIError(
            message: message,
            type: type,
            code: code,
            requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
        ))
        return response
    }

    private func writeEvent<Value: Encodable>(
        _ value: Value,
        encoder: JSONEncoder,
        to writer: any AsyncBodyStreamWriter
    ) async throws {
        let data = try encoder.encode(value)
        guard let json = String(data: data, encoding: .utf8) else { return }
        try await writer.write(.buffer(.init(string: "data: \(json)\n\n")))
    }

    private func completionID() -> String {
        "cmpl-\(UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(24))"
    }

    private func usage(_ result: AFMRawTextGenerationResult) -> Usage {
        Usage(
            promptTokens: result.promptTokens,
            completionTokens: result.completionTokens,
            totalTokens: result.totalTokens
        )
    }

    private func wireFinishReason(_ reason: AFMInferenceFinishReason) -> String {
        switch reason {
        case .length: "length"
        case .abort: "cancelled"
        case .error: "error"
        case .stop, .repetition: "stop"
        }
    }
}
