import Vapor
import Foundation

extension SpeechError: AbortError {
    var status: HTTPResponseStatus {
        switch self {
        case .platformUnavailable: return .serviceUnavailable
        case .fileNotFound: return .notFound
        case .unsupportedFormat: return .badRequest
        case .recognitionFailed: return .internalServerError
        case .noSpeechFound: return .unprocessableEntity
        case .onDeviceNotAvailable: return .serviceUnavailable
        case .authorizationDenied: return .forbidden
        }
    }

    var reason: String { errorDescription ?? "Speech error" }
}

struct SpeechTranscriptionResponse: Content {
    let object: String
    let text: String
    let locale: String
}

struct TTSSpeechRequest: Content {
    let input: String
    let voice: String?
    let responseFormat: String?
    let speed: Double?
    let locale: String?
    let appleVoice: String?

    enum CodingKeys: String, CodingKey {
        case input, voice
        case responseFormat = "response_format"
        case speed, locale
        case appleVoice = "apple_voice"
    }
}

struct VerboseTranscriptionResponse: Content {
    let text: String
    let language: String
    let duration: Double
    let words: [TranscriptionWord]?
    let segments: [TranscriptionSegment]?
}

struct SpeechAPIController: RouteCollection {

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        let speech = v1.grouped("audio")
        speech.on(.POST, "transcriptions", body: .collect(maxSize: "50mb"), use: transcribe)
        speech.on(.OPTIONS, "transcriptions", use: handleOptions)
        speech.on(.POST, "speech", body: .collect(maxSize: "1mb"), use: synthesize)
        speech.on(.OPTIONS, "speech", use: handleOptions)
        speech.on(.GET, "voices", use: listVoices)
        speech.on(.OPTIONS, "voices", use: handleOptions)
    }

    private func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "GET, POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    private func createErrorResponse(message: String, status: HTTPStatus, type: String = "invalid_request_error") async throws -> Response {
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(OpenAIError(message: message, type: type))
        return response
    }

    private func transcribe(req: Request) async throws -> Response {
        guard #available(macOS 13.0, *) else {
            return try await createErrorResponse(message: "Speech recognition requires macOS 13.0 or later", status: .serviceUnavailable, type: "speech_unavailable")
        }

        struct TranscriptionRequest: Content {
            let file: String?
            let data: String?
            let format: String?
            let locale: String?
            let model: String?
            let language: String?
            let responseFormat: String?
            let timestampGranularities: [String]?
            enum CodingKeys: String, CodingKey {
                case file, data, format, locale, model, language
                case responseFormat = "response_format"
                case timestampGranularities = "timestamp_granularities"
            }
        }

        let body = try req.content.decode(TranscriptionRequest.self)

        // language takes precedence over locale
        let effectiveLocale: String
        if let language = body.language, !language.isEmpty {
            effectiveLocale = language
        } else {
            effectiveLocale = body.locale ?? "en-US"
        }

        let options = SpeechRequestOptions(locale: effectiveLocale)
        let service = SpeechService()
        var cleanupURLs: [URL] = []

        defer {
            for url in cleanupURLs {
                try? FileManager.default.removeItem(at: url)
            }
        }

        let filePath: String
        if let file = body.file, !file.isEmpty {
            filePath = try Self.sanitizeAudioPath(file)
        } else if let data = body.data, !data.isEmpty {
            let ext = try Self.validatedExtension(body.format ?? "wav")
            let tempURL = try Self.writeTempAudio(base64: data, ext: ext)
            cleanupURLs.append(tempURL)
            filePath = tempURL.path
        } else {
            return try await createErrorResponse(message: "Either 'file' path or 'data' (base64) is required", status: .badRequest)
        }

        let responseFormat = body.responseFormat?.lowercased() ?? "json"

        switch responseFormat {
        case "text":
            let text = try await service.transcribe(from: filePath, options: options)
            let httpResponse = Response(status: .ok)
            httpResponse.headers.add(name: .contentType, value: "text/plain")
            httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
            httpResponse.body = .init(string: text)
            return httpResponse

        case "json":
            let text = try await service.transcribe(from: filePath, options: options)
            let httpResponse = Response(status: .ok)
            httpResponse.headers.add(name: .contentType, value: "application/json")
            httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
            try httpResponse.content.encode(SpeechTranscriptionResponse(
                object: "speech.transcription",
                text: text,
                locale: effectiveLocale
            ))
            return httpResponse

        case "verbose_json":
            let result = try await service.transcribeWithDetails(from: filePath, options: options)
            let granularities = Set(body.timestampGranularities ?? ["segment"])
            let verboseResponse = Self.formatAsVerboseJSON(result: result, granularities: granularities)
            let httpResponse = Response(status: .ok)
            httpResponse.headers.add(name: .contentType, value: "application/json")
            httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
            try httpResponse.content.encode(verboseResponse)
            return httpResponse

        case "srt":
            let result = try await service.transcribeWithDetails(from: filePath, options: options)
            let srtText = result.formatAsSRT()
            let httpResponse = Response(status: .ok)
            httpResponse.headers.add(name: .contentType, value: "text/plain")
            httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
            httpResponse.body = .init(string: srtText)
            return httpResponse

        case "vtt":
            let result = try await service.transcribeWithDetails(from: filePath, options: options)
            let vttText = result.formatAsVTT()
            let httpResponse = Response(status: .ok)
            httpResponse.headers.add(name: .contentType, value: "text/vtt")
            httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
            httpResponse.body = .init(string: vttText)
            return httpResponse

        default:
            return try await createErrorResponse(message: "Unsupported response_format '\(responseFormat)'. Supported: json, text, verbose_json, srt, vtt", status: .badRequest)
        }
    }

    private func synthesize(req: Request) async throws -> Response {
        guard #available(macOS 13.0, *) else {
            return try await createErrorResponse(message: "Text-to-speech requires macOS 13.0 or later", status: .serviceUnavailable, type: "speech_unavailable")
        }

        let body = try req.content.decode(TTSSpeechRequest.self)

        guard !body.input.isEmpty else {
            return try await createErrorResponse(message: "Input text is required and must not be empty", status: .badRequest)
        }
        guard body.input.count <= TTSRequestOptions.maxInputCharacters else {
            return try await createErrorResponse(message: "Input text exceeds maximum of \(TTSRequestOptions.maxInputCharacters) characters", status: .badRequest)
        }

        let format: TTSAudioFormat
        if let fmt = body.responseFormat {
            guard let parsed = TTSAudioFormat(rawValue: fmt.lowercased()) else {
                return try await createErrorResponse(message: "Unsupported response_format '\(fmt)'. Supported: aac, wav, caf", status: .badRequest)
            }
            format = parsed
        } else {
            format = .aac
        }

        let options = TTSRequestOptions(
            voice: body.voice ?? "alloy",
            appleVoice: body.appleVoice,
            locale: body.locale ?? "en-US",
            speed: Float(body.speed ?? 1.0),
            format: format
        )

        let service = SpeechSynthesisService()
        let audioData: Data
        do {
            audioData = try await service.synthesize(text: body.input, options: options)
        } catch let error as SpeechSynthesisError {
            let status: HTTPResponseStatus
            switch error {
            case .voiceNotAvailable: status = .notFound
            case .inputTooLong, .emptyInput, .unsupportedFormat: status = .badRequest
            case .synthesisTimedOut: status = .gatewayTimeout
            case .platformUnavailable: status = .serviceUnavailable
            case .synthesisFailed: status = .internalServerError
            }
            return try await createErrorResponse(message: error.errorDescription ?? "TTS synthesis failed", status: status, type: "speech_error")
        } catch {
            return try await createErrorResponse(message: "TTS synthesis failed: \(error.localizedDescription)", status: .internalServerError, type: "internal_error")
        }

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: format.contentType)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(data: audioData)
        return response
    }

    private func listVoices(req: Request) async throws -> Response {
        guard #available(macOS 13.0, *) else {
            return try await createErrorResponse(message: "Text-to-speech requires macOS 13.0 or later", status: .serviceUnavailable, type: "speech_unavailable")
        }

        let locale = req.query[String.self, at: "locale"]
        let voices = SpeechSynthesisService.listVoices(locale: locale)

        struct VoiceListResponse: Content {
            let voices: [VoiceInfo]
        }

        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(VoiceListResponse(voices: voices))
        return response
    }

    // MARK: - Chat completions integration

    static func extractTranscriptionFromMessages(_ messages: [Message], options: SpeechRequestOptions) async throws -> (messages: [Message], transcriptionTexts: [String], cleanupURLs: [URL]) {
        guard #available(macOS 13.0, *) else {
            return (messages, [], [])
        }

        let service = SpeechService()
        var updatedMessages: [Message] = []
        var transcriptionTexts: [String] = []
        var cleanupURLs: [URL] = []
        var audioIndex = 0

        for message in messages {
            guard let content = message.content, case .parts(let parts) = content else {
                updatedMessages.append(message)
                continue
            }

            // Prepare audio parts for concurrent transcription
            struct AudioWork: Sendable {
                let tempURL: URL
                let options: SpeechRequestOptions
                let index: Int
            }
            var audioWork: [AudioWork] = []
            for part in parts where part.type == "input_audio" {
                guard let inputAudio = part.input_audio else { continue }
                let ext = try validatedExtension(inputAudio.format.isEmpty ? "wav" : inputAudio.format)
                let tempURL = try writeTempAudio(base64: inputAudio.data, ext: ext)
                cleanupURLs.append(tempURL)
                audioIndex += 1
                let perAudioOptions = inputAudio.language.map { SpeechRequestOptions(locale: $0) } ?? options
                audioWork.append(AudioWork(tempURL: tempURL, options: perAudioOptions, index: audioIndex))
            }

            let results = try await withThrowingTaskGroup(of: (Int, String).self) { group in
                for work in audioWork {
                    group.addTask {
                        let transcription = try await service.transcribe(from: work.tempURL.path, options: work.options)
                        return (work.index, "[Apple Speech transcription \(work.index)]\n\(transcription)")
                    }
                }
                var collected: [(Int, String)] = []
                for try await result in group {
                    collected.append(result)
                }
                return collected.sorted { $0.0 < $1.0 }.map(\.1)
            }

            var textChunks = parts.compactMap(\.text)
            textChunks.append(contentsOf: results)
            transcriptionTexts.append(contentsOf: results)

            updatedMessages.append(Message(role: message.role, content: textChunks.joined(separator: "\n\n")))
        }

        return (updatedMessages, transcriptionTexts, cleanupURLs)
    }

    // MARK: - Transcription response formatting

    private static func formatAsVerboseJSON(result: TranscriptionResult, granularities: Set<String>) -> VerboseTranscriptionResponse {
        VerboseTranscriptionResponse(
            text: result.text,
            language: result.language,
            duration: result.duration,
            words: granularities.contains("word") ? result.words : nil,
            segments: granularities.contains("segment") ? result.segments : nil
        )
    }

    // MARK: - Helpers

    /// Validate and resolve a file path for the API endpoint.
    /// Resolves symlinks, rejects directories, and enforces audio extension allowlist.
    private static func sanitizeAudioPath(_ raw: String) throws -> String {
        let expanded = NSString(string: raw).expandingTildeInPath
        let resolved = URL(fileURLWithPath: expanded).resolvingSymlinksInPath().path
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: resolved, isDirectory: &isDir) else {
            throw SpeechError.fileNotFound
        }
        guard !isDir.boolValue else {
            throw SpeechError.unsupportedFormat
        }
        let ext = URL(fileURLWithPath: resolved).pathExtension.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(ext) else {
            throw SpeechError.unsupportedFormat
        }
        return resolved
    }

    /// Validate that ext is a supported audio extension before using it in a filename.
    private static func validatedExtension(_ ext: String) throws -> String {
        let clean = ext.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(clean) else {
            throw SpeechError.unsupportedFormat
        }
        return clean
    }

    private static func writeTempAudio(base64: String, ext: String) throws -> URL {
        guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else {
            throw Abort(.badRequest, reason: "Invalid base64 audio data")
        }
        if data.count > SpeechRequestOptions.defaultMaxFileBytes {
            throw Abort(.payloadTooLarge, reason: "Decoded audio exceeds maximum of \(SpeechRequestOptions.defaultMaxFileBytes / (1024 * 1024))MB")
        }
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm_speech_\(UUID().uuidString).\(ext)")
        try data.write(to: tempURL)
        return tempURL
    }
}
