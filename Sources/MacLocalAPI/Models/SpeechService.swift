import Foundation
import os
import Speech

enum SpeechError: Error, LocalizedError {
    case platformUnavailable
    case fileNotFound
    case unsupportedFormat
    case requestTooLarge(actualBytes: Int, maxBytes: Int)
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
        case .requestTooLarge(let actualBytes, let maxBytes):
            return "Audio file size \(actualBytes) bytes exceeds the limit of \(maxBytes) bytes"
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
    static let defaultMaxFileBytes = 50 * 1024 * 1024  // 50 MB
    static let recognitionTimeoutNs: UInt64 = 120_000_000_000  // 120 seconds
    static let supportedExtensions: Set<String> = ["wav", "mp3", "m4a", "caf", "aiff", "aif"]

    let locale: String
    let maxFileBytes: Int

    init(
        locale: String = "en-US",
        maxFileBytes: Int = SpeechRequestOptions.defaultMaxFileBytes
    ) {
        self.locale = locale
        self.maxFileBytes = maxFileBytes
    }
}

@available(macOS 13.0, *)
final class SpeechService {

    func transcribe(from filePath: String) async throws -> String {
        try await transcribe(from: filePath, options: SpeechRequestOptions())
    }

    func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String {
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

        // Validate size
        let attrs = try FileManager.default.attributesOfItem(atPath: filePath)
        if let size = attrs[.size] as? Int, size > options.maxFileBytes {
            throw SpeechError.requestTooLarge(actualBytes: size, maxBytes: options.maxFileBytes)
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

        let text: String = try await withThrowingTaskGroup(of: String.self) { group in
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
                            let formatted = result.bestTranscription.formattedString
                            if formatted.isEmpty {
                                continuation.resume(throwing: SpeechError.noSpeechFound)
                            } else {
                                continuation.resume(returning: formatted)
                            }
                        }
                        taskRef.withLock { $0 = task }
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

        return text
    }
}
