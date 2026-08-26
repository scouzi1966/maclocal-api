import AFMKitCore
import AFMOpenAICompat
import Foundation
import Vapor

/// OpenAI-compatible raw-prompt completion endpoint. This deliberately does
/// not translate prompts into chat messages or invoke a chat template.
struct CompletionsController: RouteCollection {
    private static let defaultSlotQueueTimeout: Duration = .seconds(240)

    private let modelID: String
    private let generator: AnyAFMRawTextGenerator
    private let generationAdmitter: AnyAFMGenerationAdmitter?
    private let slotQueueTimeout: Duration
    private let telemetry: AFMServerTelemetryAdapter

    init(
        modelID: String,
        generator: AnyAFMRawTextGenerator,
        generationAdmitter: AnyAFMGenerationAdmitter? = nil,
        slotQueueTimeout: Duration = Self.defaultSlotQueueTimeout,
        telemetry: AFMServerTelemetryAdapter = .standalone()
    ) {
        self.modelID = modelID
        self.generator = generator
        self.generationAdmitter = generationAdmitter
        self.slotQueueTimeout = slotQueueTimeout
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
                code: "unsupported_prompt_array",
                param: "prompt"
            )
        }
        if let requestedModel = request.model, requestedModel != modelID {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .notFound,
                message: "The model `\(requestedModel)` does not exist on this server",
                code: "model_not_found",
                param: "model"
            )
        }
        guard request.maxTokens.map({ $0 >= 0 }) ?? true else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "max_tokens must be greater than or equal to zero",
                code: "invalid_request_error",
                param: "max_tokens"
            )
        }
        guard !(request.stop?.sequences.contains("") ?? false) else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "stop sequences must not be empty",
                code: "invalid_request_error",
                param: "stop"
            )
        }
        guard (request.n ?? 1) == 1, (request.bestOf ?? 1) == 1 else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "AFM raw completions currently support only n=1 and best_of=1",
                code: "unsupported_completion_multiplicity",
                param: request.n.map({ $0 != 1 }) == true ? "n" : "best_of"
            )
        }
        guard request.echo != true else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "echo=true is not supported for raw completions",
                code: "unsupported_parameter",
                param: "echo"
            )
        }
        guard request.logprobs == nil else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "logprobs is not supported for raw completions",
                code: "unsupported_parameter",
                param: "logprobs"
            )
        }
        guard request.streamOptions?.continuousUsageStats != true else {
            telemetry.recordRejection(.validation)
            return try errorResponse(
                request: req,
                status: .badRequest,
                message: "stream_options.continuous_usage_stats is not supported; use include_usage for exact final usage",
                code: "unsupported_parameter",
                param: "stream_options.continuous_usage_stats"
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
        let acceptedAt = ProcessInfo.processInfo.systemUptime
        let lease: AFMGenerationLease?
        do {
            lease = try await generationAdmitter?.admitGeneration(timeout: slotQueueTimeout)
        } catch {
            return try admissionErrorResponse(request: req, error: error)
        }
        let events: AsyncStream<AFMRawTextGenerationEvent>
        if let lease {
            events = AFMGenerationContext.$admissionLease.withValue(lease) {
                AFMGenerationContext.$telemetryToken.withValue(lease.telemetryToken) {
                    AFMGenerationContext.$acceptedAt.withValue(acceptedAt) {
                        AFMGenerationContext.$requestedMaximumOutputTokens.withValue(
                            request.maxTokens
                        ) {
                            AFMGenerationContext.$ignoreEndOfSequence.withValue(
                                request.ignoreEOS ?? false
                            ) {
                                generator.rawTextGenerationEvents(for: providerRequest)
                            }
                        }
                    }
                }
            }
        } else {
            events = AFMGenerationContext.$requestedMaximumOutputTokens.withValue(
                request.maxTokens
            ) {
                AFMGenerationContext.$ignoreEndOfSequence.withValue(
                    request.ignoreEOS ?? false
                ) {
                    generator.rawTextGenerationEvents(for: providerRequest)
                }
            }
        }

        if request.stream == true {
            return streamingResponse(
                request: req,
                completionRequest: request,
                events: events,
                lease: lease
            )
        }
        defer { lease?.release() }
        return try await nonStreamingResponse(request: req, events: events)
    }

    private func nonStreamingResponse(
        request: Request,
        events: AsyncStream<AFMRawTextGenerationEvent>
    ) async throws -> Response {
        var text = ""
        var terminal: AFMRawTextGenerationResult?
        for await event in events {
            if terminal != nil { break }
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
        events: AsyncStream<AFMRawTextGenerationEvent>,
        lease: AFMGenerationLease?
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
            defer { lease?.release() }
            let connectionToken = telemetry.connectionOpened()
            defer { telemetry.connectionClosed(connectionToken) }
            let encoder = JSONEncoder()
            var terminalSeen = false
            var writeFailed = false

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
                    do {
                        try await writeEvent(chunk, encoder: encoder, to: writer)
                    } catch {
                        writeFailed = true
                    }
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
                    do {
                        try await writeEvent(finish, encoder: encoder, to: writer)
                    } catch {
                        writeFailed = true
                    }
                    if includeUsage && !writeFailed {
                        let usageChunk = CompletionResponse(
                            id: id,
                            created: created,
                            model: modelID,
                            choices: [],
                            usage: usage(result)
                        )
                        do {
                            try await writeEvent(usageChunk, encoder: encoder, to: writer)
                        } catch {
                            writeFailed = true
                        }
                    }
                    if !writeFailed {
                        do {
                            try await writer.write(.buffer(.init(string: "data: [DONE]\n\n")))
                        } catch {
                            writeFailed = true
                        }
                    }
                case .failed(let reason, let message):
                    terminalSeen = true
                    let error = OpenAIError(
                        message: message,
                        type: reason == .cancelled ? "cancelled" : "server_error",
                        code: reason.rawValue,
                        requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
                    )
                    do {
                        try await writeEvent(error, encoder: encoder, to: writer)
                    } catch {
                        writeFailed = true
                    }
                }
                if writeFailed { break }
            }
            if !terminalSeen && !writeFailed {
                let error = OpenAIError(
                    message: "Raw completion provider ended without a terminal event",
                    type: "server_error",
                    code: "missing_terminal_event",
                    requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
                )
                try? await writeEvent(error, encoder: encoder, to: writer)
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
        code: String,
        param: String? = nil
    ) throws -> Response {
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(OpenAIError(
            message: message,
            type: type,
            code: code,
            param: param,
            requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
        ))
        return response
    }

    private func admissionErrorResponse(request: Request, error: Error) throws -> Response {
        switch error as? AFMGenerationAdmissionError {
        case .capacity, .timedOut:
            let response = try errorResponse(
                request: request,
                status: .serviceUnavailable,
                message: "Server at capacity. Please retry shortly.",
                type: "server_busy",
                code: "server_busy"
            )
            response.headers.add(name: "Retry-After", value: "2")
            return response
        case .cancelled:
            return try errorResponse(
                request: request,
                status: HTTPResponseStatus(
                    statusCode: 499,
                    reasonPhrase: "Client Closed Request"
                ),
                message: "Request cancelled while waiting for generation capacity.",
                type: "cancelled",
                code: "cancelled"
            )
        case .internalFailure, .none:
            return try errorResponse(
                request: request,
                status: .internalServerError,
                message: "Generation admission failed.",
                type: "server_error",
                code: "internal_error"
            )
        }
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
