import Foundation
import AVFoundation
import os

enum SpeechSynthesisError: Error, LocalizedError {
    case platformUnavailable
    case inputTooLong(actualChars: Int, maxChars: Int)
    case emptyInput
    case voiceNotAvailable(String)
    case synthesisTimedOut
    case synthesisFailed(String)
    case unsupportedFormat(String)

    var errorDescription: String? {
        switch self {
        case .platformUnavailable:
            return "Text-to-speech requires macOS 13.0 or later"
        case .inputTooLong(let actual, let max):
            return "Input text \(actual) characters exceeds the limit of \(max)"
        case .emptyInput:
            return "Input text is empty"
        case .voiceNotAvailable(let name):
            return "Voice '\(name)' is not available on this system"
        case .synthesisTimedOut:
            return "Speech synthesis timed out"
        case .synthesisFailed(let message):
            return "Speech synthesis failed: \(message)"
        case .unsupportedFormat(let format):
            return "Unsupported audio format: \(format). Supported: aac, wav, caf"
        }
    }
}

enum TTSAudioFormat: String, Sendable, CaseIterable {
    case aac
    case wav
    case caf

    var contentType: String {
        switch self {
        case .aac: return "audio/aac"
        case .wav: return "audio/wav"
        case .caf: return "audio/x-caf"
        }
    }

    var fileExtension: String { rawValue }
}

enum OpenAIVoiceName: String, CaseIterable {
    case alloy, echo, fable, nova, onyx, shimmer

    enum Gender { case female, male }

    var gender: Gender {
        switch self {
        case .alloy, .nova, .shimmer: return .female
        case .echo, .fable, .onyx: return .male
        }
    }

    var preferredAppleVoiceNames: [String] {
        switch self {
        case .alloy: return ["Samantha"]
        case .echo: return ["Daniel"]
        case .fable: return ["Tom"]
        case .nova: return ["Karen"]
        case .onyx: return ["Alex"]
        case .shimmer: return ["Ava"]
        }
    }
}

struct TTSRequestOptions: Sendable {
    static let maxInputCharacters = 4096
    static let synthesisTimeoutNs: UInt64 = 120_000_000_000
    static let encodeTimeoutNs: UInt64 = 30_000_000_000  // 30s for afconvert

    let voice: String?
    let appleVoice: String?
    let locale: String
    let speed: Float
    let format: TTSAudioFormat

    init(
        voice: String? = "alloy",
        appleVoice: String? = nil,
        locale: String = "en-US",
        speed: Float = 1.0,
        format: TTSAudioFormat = .aac
    ) {
        self.voice = voice
        self.appleVoice = appleVoice
        self.locale = locale
        self.speed = speed
        self.format = format
    }
}

struct VoiceInfo: Sendable, Codable {
    let id: String
    let name: String
    let locale: String
    let gender: String
    let quality: String
}

@available(macOS 13.0, *)
final class SpeechSynthesisService: NSObject, @unchecked Sendable {

    static let speedMinimum: Float = 0.25
    static let speedMaximum: Float = 4.0

    static func listVoices(locale: String? = nil) -> [VoiceInfo] {
        let voices = AVSpeechSynthesisVoice.speechVoices()
        let filtered: [AVSpeechSynthesisVoice]
        if let locale = locale {
            let prefix = locale.lowercased()
            filtered = voices.filter { $0.language.lowercased().hasPrefix(prefix) }
        } else {
            filtered = voices
        }
        return filtered.map { voice in
            let quality: String
            switch voice.quality {
            case .enhanced: quality = "enhanced"
            case .premium: quality = "premium"
            default: quality = "compact"
            }
            return VoiceInfo(
                id: voice.identifier,
                name: voice.name,
                locale: voice.language,
                gender: voice.gender == .male ? "male" : "female",
                quality: quality
            )
        }.sorted {
            if $0.locale != $1.locale { return $0.locale < $1.locale }
            return $0.name < $1.name
        }
    }

    func synthesize(text: String, options: TTSRequestOptions) async throws -> Data {
        guard !text.isEmpty else {
            throw SpeechSynthesisError.emptyInput
        }
        guard text.count <= TTSRequestOptions.maxInputCharacters else {
            throw SpeechSynthesisError.inputTooLong(
                actualChars: text.count,
                maxChars: TTSRequestOptions.maxInputCharacters
            )
        }
        guard options.speed >= Self.speedMinimum && options.speed <= Self.speedMaximum else {
            throw SpeechSynthesisError.synthesisFailed(
                "Speed must be between \(Self.speedMinimum) and \(Self.speedMaximum)"
            )
        }

        let voice = try resolveVoice(options: options)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = clampRate(options.speed)

        let synthesizer = AVSpeechSynthesizer()
        let allBuffers: [AVAudioPCMBuffer] = try await withThrowingTaskGroup(of: [AVAudioPCMBuffer].self) { group in
            group.addTask {
                let state = OSAllocatedUnfairLock(initialState: (resumed: false, buffers: [AVAudioPCMBuffer](), continuation: nil as CheckedContinuation<[AVAudioPCMBuffer], Error>?))
                return try await withTaskCancellationHandler {
                    try await withCheckedThrowingContinuation { continuation in
                        state.withLock { $0.continuation = continuation }
                        synthesizer.write(utterance) { buffer in
                            if let pcm = buffer as? AVAudioPCMBuffer, pcm.frameLength > 0 {
                                state.withLock { $0.buffers.append(pcm) }
                            } else {
                                state.withLock { s in
                                    guard !s.resumed else { return }
                                    s.resumed = true
                                    s.continuation?.resume(returning: s.buffers)
                                    s.continuation = nil
                                }
                            }
                        }
                        // If already cancelled before we got here, resume immediately
                        state.withLock { s in
                            if Task.isCancelled && !s.resumed {
                                s.resumed = true
                                s.continuation?.resume(throwing: CancellationError())
                                s.continuation = nil
                            }
                        }
                    }
                } onCancel: {
                    synthesizer.stopSpeaking(at: .immediate)
                    // If stopSpeaking doesn't trigger the empty-buffer callback,
                    // resume the continuation to prevent a leak
                    state.withLock { s in
                        guard !s.resumed else { return }
                        s.resumed = true
                        s.continuation?.resume(throwing: CancellationError())
                        s.continuation = nil
                    }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: TTSRequestOptions.synthesisTimeoutNs)
                throw SpeechSynthesisError.synthesisTimedOut
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }

        guard !allBuffers.isEmpty else {
            throw SpeechSynthesisError.synthesisFailed("No audio data generated")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm_tts_\(UUID().uuidString).\(options.format.fileExtension)")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        // Encode collected PCM buffers to the requested format
        try await encodeBuffers(allBuffers, to: tempURL, format: options.format)
        return try Data(contentsOf: tempURL)
    }

    /// Encode PCM buffers to a file. For WAV/CAF, write PCM directly via
    /// AVAudioFile. For AAC, use AVAudioConverter to transcode.
    private func encodeBuffers(
        _ buffers: [AVAudioPCMBuffer],
        to url: URL,
        format: TTSAudioFormat
    ) async throws {
        let inputFormat = buffers[0].format

        switch format {
        case .wav, .caf:
            // PCM output -- write directly
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: inputFormat.sampleRate,
                AVNumberOfChannelsKey: inputFormat.channelCount,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false
            ]
            let outputFile = try AVAudioFile(
                forWriting: url,
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            for buffer in buffers {
                try outputFile.write(from: buffer)
            }

        case .aac:
            // AAC encoding: write PCM to temp WAV, then use afconvert CLI.
            // AVSpeechSynthesizer outputs 22050 Hz; AVAudioConverter's AAC
            // encoder doesn't support non-standard sample rates, so we shell
            // out to afconvert which handles resampling transparently.
            let tempWAV = FileManager.default.temporaryDirectory
                .appendingPathComponent("afm_tts_pcm_\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: tempWAV) }

            let pcmSettings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: inputFormat.sampleRate,
                AVNumberOfChannelsKey: inputFormat.channelCount,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false
            ]
            // Scope the AVAudioFile so it closes before afconvert reads the WAV
            do {
                let pcmFile = try AVAudioFile(
                    forWriting: tempWAV,
                    settings: pcmSettings,
                    commonFormat: .pcmFormatFloat32,
                    interleaved: false
                )
                for buffer in buffers {
                    try pcmFile.write(from: buffer)
                }
            } catch {
                throw SpeechSynthesisError.synthesisFailed("Failed to write PCM temp file: \(error.localizedDescription)")
            }

            // Convert WAV → AAC (ADTS format) via afconvert (ships with macOS).
            // Omit bitrate flag — let CoreAudio pick a suitable rate for the
            // source sample rate (22050 Hz mono can't sustain 128 kbps).
            // Run via terminationHandler to avoid blocking the cooperative thread pool.
            let encode = Process()
            encode.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
            encode.arguments = [
                tempWAV.path, url.path,
                "-d", "aac", "-f", "adts"
            ]
            let errPipe = Pipe()
            encode.standardError = errPipe
            // Collect stderr asynchronously to avoid blocking the termination handler
            // and prevent potential deadlock if afconvert writes more than the pipe buffer
            let stderrData = OSAllocatedUnfairLock(initialState: Data())
            errPipe.fileHandleForReading.readabilityHandler = { handle in
                let chunk = handle.availableData
                if !chunk.isEmpty {
                    stderrData.withLock { $0.append(chunk) }
                }
            }
            let resumed = OSAllocatedUnfairLock(initialState: false)
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                        encode.terminationHandler = { process in
                            errPipe.fileHandleForReading.readabilityHandler = nil
                            let alreadyResumed = resumed.withLock { val -> Bool in
                                if val { return true }
                                val = true
                                return false
                            }
                            guard !alreadyResumed else { return }
                            if process.terminationStatus != 0 {
                                let errMsg = stderrData.withLock { String(data: $0, encoding: .utf8) ?? "unknown" }
                                continuation.resume(throwing: SpeechSynthesisError.synthesisFailed(
                                    "AAC encode failed (\(process.terminationStatus)): \(errMsg)"
                                ))
                            } else {
                                continuation.resume()
                            }
                        }
                        do {
                            try encode.run()
                        } catch {
                            let alreadyResumed = resumed.withLock { val -> Bool in
                                if val { return true }
                                val = true
                                return false
                            }
                            guard !alreadyResumed else { return }
                            continuation.resume(throwing: SpeechSynthesisError.synthesisFailed(
                                "Failed to launch afconvert: \(error.localizedDescription)"
                            ))
                        }
                    }
                }
                group.addTask {
                    try await Task.sleep(nanoseconds: TTSRequestOptions.encodeTimeoutNs)
                    let alreadyResumed = resumed.withLock { val -> Bool in
                        if val { return true }
                        val = true
                        return false
                    }
                    encode.terminate()
                    guard !alreadyResumed else { return }
                    throw SpeechSynthesisError.synthesisTimedOut
                }
                let _ = try await group.next()!
                group.cancelAll()
            }
        }
    }

    private func resolveVoice(options: TTSRequestOptions) throws -> AVSpeechSynthesisVoice {
        // 1. Exact Apple voice identifier
        if let appleVoice = options.appleVoice, !appleVoice.isEmpty {
            if let voice = AVSpeechSynthesisVoice(identifier: appleVoice) {
                return voice
            }
            throw SpeechSynthesisError.voiceNotAvailable(appleVoice)
        }

        // 2. OpenAI voice name mapping
        if let voiceName = options.voice,
           let openAIVoice = OpenAIVoiceName(rawValue: voiceName.lowercased()) {
            let voices = AVSpeechSynthesisVoice.speechVoices()
                .filter { $0.language.lowercased().hasPrefix(options.locale.lowercased().prefix(2).description) }

            // Try preferred Apple voice names first
            for preferred in openAIVoice.preferredAppleVoiceNames {
                // Prefer enhanced > premium > compact
                let sorted = voices.filter { $0.name == preferred }
                    .sorted { qualityRank($0) > qualityRank($1) }
                if let found = sorted.first {
                    return found
                }
            }

            // Fallback: match by gender, prefer higher quality
            let genderMatched = voices
                .filter { voiceMatchesGender($0, openAIVoice.gender) }
                .sorted { qualityRank($0) > qualityRank($1) }
            if let found = genderMatched.first {
                return found
            }

            // Last fallback: any voice for the locale
            if let anyVoice = voices.sorted(by: { qualityRank($0) > qualityRank($1) }).first {
                return anyVoice
            }
        }

        // 3. System default for locale
        if let voice = AVSpeechSynthesisVoice(language: options.locale) {
            return voice
        }

        // 4. Absolute fallback
        if let voice = AVSpeechSynthesisVoice(language: "en-US") {
            return voice
        }

        throw SpeechSynthesisError.voiceNotAvailable(options.locale)
    }

    private func qualityRank(_ voice: AVSpeechSynthesisVoice) -> Int {
        switch voice.quality {
        case .enhanced: return 3
        case .premium: return 2
        default: return 1
        }
    }

    private func voiceMatchesGender(_ voice: AVSpeechSynthesisVoice, _ gender: OpenAIVoiceName.Gender) -> Bool {
        switch gender {
        case .female: return voice.gender == .female
        case .male: return voice.gender == .male
        }
    }

    private func clampRate(_ speed: Float) -> Float {
        let minRate = AVSpeechUtteranceMinimumSpeechRate
        let maxRate = AVSpeechUtteranceMaximumSpeechRate
        let defaultRate = AVSpeechUtteranceDefaultSpeechRate

        // Map OpenAI's 0.25-4.0 range to Apple's rate range
        // speed=1.0 maps to defaultRate
        let normalized = speed * defaultRate
        return Swift.min(Swift.max(normalized, minRate), maxRate)
    }
}
