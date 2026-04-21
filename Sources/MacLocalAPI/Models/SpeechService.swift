import Foundation
import os
import Speech

enum SpeechError: Error, LocalizedError {
    case platformUnavailable
    case fileNotFound
    case unsupportedFormat
    case recognitionFailed(String)
    case noSpeechFound
    case onDeviceNotAvailable
    case authorizationDenied

    var errorDescription: String? {
        switch self {
        case .platformUnavailable:
            return "Apple Speech framework requires macOS 10.15 or later"
        case .fileNotFound:
            return "The specified audio file was not found"
        case .unsupportedFormat:
            return "Unsupported audio format. Supported formats: WAV, MP3, M4A, CAF, AIFF"
        case .recognitionFailed(let message):
            return "Speech recognition failed: \(message)"
        case .noSpeechFound:
            return "No speech was detected in the audio file"
        case .onDeviceNotAvailable:
            return "On-device speech recognition is not available for the requested locale"
        case .authorizationDenied:
            return "Speech recognition authorization was denied. Grant access in System Settings > Privacy & Security > Speech Recognition"
        }
    }
}

struct SpeechRequestOptions: Sendable {
    static let recognitionTimeoutNs: UInt64 = 120_000_000_000  // 120 seconds
    static let supportedExtensions: Set<String> = ["wav", "mp3", "m4a", "caf", "aiff", "aif"]

    let locale: String

    init(locale: String = "en-US") {
        self.locale = locale
    }
}

struct TranscriptionWord: Sendable, Codable {
    let word: String
    let start: Double
    let end: Double
}

struct TranscriptionSegment: Sendable, Codable {
    let id: Int
    let start: Double
    let end: Double
    let text: String
    let confidence: Float
}

struct TranscriptionResult: Sendable, Codable {
    let text: String
    let language: String
    let duration: Double
    let words: [TranscriptionWord]
    let segments: [TranscriptionSegment]

    func formatAsSRT() -> String {
        segments.enumerated().map { index, seg in
            let startTS = Self.srtTimestamp(seg.start)
            let endTS = Self.srtTimestamp(seg.end)
            return "\(index + 1)\n\(startTS) --> \(endTS)\n\(seg.text)"
        }.joined(separator: "\n\n")
    }

    func formatAsVTT() -> String {
        var lines = ["WEBVTT", ""]
        lines += segments.map { seg in
            let startTS = Self.vttTimestamp(seg.start)
            let endTS = Self.vttTimestamp(seg.end)
            return "\(startTS) --> \(endTS)\n\(seg.text)"
        }
        return lines.joined(separator: "\n\n")
    }

    static func srtTimestamp(_ seconds: Double) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        let ms = Int((seconds.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }

    static func vttTimestamp(_ seconds: Double) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        let ms = Int((seconds.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
    }
}

@available(macOS 13.0, *)
final class SpeechService {

    func transcribe(from filePath: String) async throws -> String {
        try await transcribe(from: filePath, options: SpeechRequestOptions())
    }

    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String {
        let result = try await transcribeWithDetails(from: filePath, options: options)
        return result.text
    }

    func transcribeWithDetails(from filePath: String) async throws -> TranscriptionResult {
        try await transcribeWithDetails(from: filePath, options: SpeechRequestOptions())
    }

    func transcribeWithDetails(from filePath: String, options: SpeechRequestOptions) async throws -> TranscriptionResult {
        let fileURL = URL(fileURLWithPath: filePath)

        // Validate file exists
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw SpeechError.fileNotFound
        }

        // Validate extension
        let ext = fileURL.pathExtension.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(ext) else {
            throw SpeechError.unsupportedFormat
        }

        // Check authorization
        let status = SFSpeechRecognizer.authorizationStatus()
        if status == .denied || status == .restricted {
            throw SpeechError.authorizationDenied
        }
        if status == .notDetermined {
            let granted = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { newStatus in
                    continuation.resume(returning: newStatus == .authorized)
                }
            }
            guard granted else { throw SpeechError.authorizationDenied }
        }

        // Create recognizer and verify on-device support
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: options.locale)) else {
            throw SpeechError.onDeviceNotAvailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw SpeechError.onDeviceNotAvailable
        }

        // Create request
        let request = SFSpeechURLRecognitionRequest(url: fileURL)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false

        // Run recognition with a timeout to prevent hung requests.
        // OSAllocatedUnfairLock guards the one-shot continuation resume.
        let resumed = OSAllocatedUnfairLock(initialState: false)
        // Holds the SFSpeechRecognitionTask so onCancel can reach it.
        let taskRef = OSAllocatedUnfairLock<SFSpeechRecognitionTask?>(initialState: nil)

        let transcriptionResult: TranscriptionResult = try await withThrowingTaskGroup(of: TranscriptionResult.self) { group in
            group.addTask {
                try await withTaskCancellationHandler {
                    try await withCheckedThrowingContinuation { continuation in
                        let queue = OperationQueue()
                        queue.qualityOfService = .userInitiated

                        recognizer.queue = queue
                        let task = recognizer.recognitionTask(with: request) { result, error in
                            if let error {
                                let shouldResume = resumed.withLock { done in
                                    if done { return false }
                                    done = true
                                    return true
                                }
                                guard shouldResume else { return }
                                continuation.resume(throwing: SpeechError.recognitionFailed(error.localizedDescription))
                                return
                            }
                            guard let result, result.isFinal else { return }
                            let shouldResume = resumed.withLock { done in
                                if done { return false }
                                done = true
                                return true
                            }
                            guard shouldResume else { return }
                            let transcription = result.bestTranscription
                            let formatted = transcription.formattedString
                            if formatted.isEmpty {
                                continuation.resume(throwing: SpeechError.noSpeechFound)
                                return
                            }

                            // Extract word-level timing from segments
                            let words = transcription.segments.map { seg in
                                TranscriptionWord(
                                    word: seg.substring,
                                    start: seg.timestamp,
                                    end: seg.timestamp + seg.duration
                                )
                            }

                            // Build a single segment from the full transcription
                            let totalDuration = transcription.segments.last.map { $0.timestamp + $0.duration } ?? 0
                            let avgConfidence = transcription.segments.isEmpty ? Float(0)
                                : transcription.segments.map(\.confidence).reduce(0, +) / Float(transcription.segments.count)
                            let segments = [TranscriptionSegment(
                                id: 0,
                                start: 0,
                                end: totalDuration,
                                text: formatted,
                                confidence: avgConfidence
                            )]

                            // Extract language from locale
                            let lang = options.locale.split(separator: "-").first.map(String.init) ?? options.locale

                            continuation.resume(returning: TranscriptionResult(
                                text: formatted,
                                language: lang,
                                duration: totalDuration,
                                words: words,
                                segments: segments
                            ))
                        }
                        taskRef.withLock { $0 = task }
                        if Task.isCancelled { task.cancel() }
                    }
                } onCancel: {
                    taskRef.withLock { $0?.cancel() }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: SpeechRequestOptions.recognitionTimeoutNs)
                throw SpeechError.recognitionFailed("Recognition timed out")
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }

        return transcriptionResult
    }
}
