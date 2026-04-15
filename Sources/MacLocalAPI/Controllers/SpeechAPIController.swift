import Vapor
import Foundation

struct SpeechTranscriptionResponse: Content {
    let object: String
    let text: String
    let locale: String
}

struct SpeechAPIController: RouteCollection {

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1")
        let speech = v1.grouped("audio")
        speech.on(.POST, "transcriptions", body: .collect(maxSize: "50mb"), use: transcribe)
        speech.on(.OPTIONS, "transcriptions", use: handleOptions)
    }

    private func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    private func transcribe(req: Request) async throws -> Response {
        guard #available(macOS 13.0, *) else {
            throw Abort(.serviceUnavailable, reason: "Speech recognition requires macOS 13.0 or later")
        }

        struct TranscriptionRequest: Content {
            let file: String?
            let data: String?
            let format: String?
            let locale: String?
        }

        let body = try req.content.decode(TranscriptionRequest.self)
        let locale = body.locale ?? "en-US"
        let options = SpeechRequestOptions(locale: locale)
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
            throw Abort(.badRequest, reason: "Either 'file' path or 'data' (base64) is required")
        }

        let text = try await service.transcribe(from: filePath, options: options)

        let response = SpeechTranscriptionResponse(
            object: "speech.transcription",
            text: text,
            locale: locale
        )
        let httpResponse = Response(status: .ok)
        httpResponse.headers.add(name: .contentType, value: "application/json")
        httpResponse.headers.add(name: .accessControlAllowOrigin, value: "*")
        try httpResponse.content.encode(response)
        return httpResponse
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

            var textChunks = parts.compactMap(\.text)
            for part in parts where part.type == "input_audio" {
                guard let inputAudio = part.input_audio else { continue }
                let ext = try validatedExtension(inputAudio.format.isEmpty ? "wav" : inputAudio.format)
                let tempURL = try writeTempAudio(base64: inputAudio.data, ext: ext)
                cleanupURLs.append(tempURL)
                let transcription = try await service.transcribe(from: tempURL.path, options: options)
                audioIndex += 1
                let labeled = "[Apple Speech transcription \(audioIndex)]\n\(transcription)"
                textChunks.append(labeled)
                transcriptionTexts.append(labeled)
            }

            updatedMessages.append(Message(role: message.role, content: textChunks.joined(separator: "\n\n")))
        }

        return (updatedMessages, transcriptionTexts, cleanupURLs)
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
            throw SpeechError.requestTooLarge(actualBytes: data.count, maxBytes: SpeechRequestOptions.defaultMaxFileBytes)
        }
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm_speech_\(UUID().uuidString).\(ext)")
        try data.write(to: tempURL)
        return tempURL
    }
}
