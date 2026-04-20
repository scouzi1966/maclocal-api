# Vision & Speech Feature Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand AFM's vision and speech capabilities with TTS, transcription enhancements (OpenAI parity), and new vision modes (barcode, classify, saliency, auto-crop).

**Architecture:** Three independent feature areas built on existing VisionService/SpeechService patterns. Each adds service methods, controller routes, and CLI flags. Speech transcription branch must be merged first as a prerequisite. TTS is a new service (`SpeechSynthesisService`). Vision modes extend the existing `VisionService` with per-mode `#available` checks instead of class-level gating.

**Tech Stack:** Swift, Vapor, Apple Vision framework, Apple Speech framework, AVSpeechSynthesizer, AudioToolbox, ArgumentParser

**Spec:** `docs/superpowers/specs/2026-04-20-vision-speech-options-design.md`

---

## Prerequisites

### Task 0: Merge speech transcription branch

**Files:**
- Merge: `feature/speech-transcription` into `main`

The speech transcription feature branch has SpeechService, SpeechAPIController, and SpeechCommand. It must be on main before we can enhance it.

- [ ] **Step 1: Merge the branch**

```bash
git merge feature/speech-transcription --no-edit
```

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit if needed**

The merge commit is automatic. Verify with `git log --oneline -3`.

---

## Part 1: TTS (Text-to-Speech)

### Task 1: Create SpeechSynthesisService

**Files:**
- Create: `Sources/MacLocalAPI/Models/SpeechSynthesisService.swift`

- [ ] **Step 1: Create the service file**

```swift
import Foundation
import AVFoundation

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

    static let speedMinimum: Float = 0.25
    static let speedMaximum: Float = 4.0

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

        // Collect PCM buffers from AVSpeechSynthesizer.write, then encode
        let synthesizer = AVSpeechSynthesizer()
        let allBuffers: [AVAudioPCMBuffer] = await withCheckedContinuation { continuation in
            var buffers: [AVAudioPCMBuffer] = []
            synthesizer.write(utterance) { buffer in
                if let pcm = buffer as? AVAudioPCMBuffer, pcm.frameLength > 0 {
                    buffers.append(pcm)
                } else {
                    // Empty buffer signals completion
                    continuation.resume(returning: buffers)
                }
            }
        }

        guard !allBuffers.isEmpty else {
            throw SpeechSynthesisError.synthesisFailed("No audio data generated")
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm_tts_\(UUID().uuidString).\(options.format.fileExtension)")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        // Encode collected PCM buffers to the requested format
        try encodeBuffers(allBuffers, to: tempURL, format: options.format)
        return try Data(contentsOf: tempURL)
    }

    /// Encode PCM buffers to a file. For WAV/CAF, write PCM directly via
    /// AVAudioFile. For AAC, use AVAudioConverter to transcode.
    private func encodeBuffers(
        _ buffers: [AVAudioPCMBuffer],
        to url: URL,
        format: TTSAudioFormat
    ) throws {
        let inputFormat = buffers[0].format

        switch format {
        case .wav, .caf:
            // PCM output — write directly
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
            // AAC encoding via AVAudioConverter
            let outputSettings: [String: Any] = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: inputFormat.sampleRate,
                AVNumberOfChannelsKey: inputFormat.channelCount,
                AVEncoderBitRateKey: 128_000
            ]
            guard let outputFormat = AVAudioFormat(settings: outputSettings) else {
                throw SpeechSynthesisError.synthesisFailed("Cannot create AAC output format")
            }
            guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
                throw SpeechSynthesisError.synthesisFailed("Cannot create AAC converter")
            }

            let outputFile = try AVAudioFile(
                forWriting: url,
                settings: outputSettings
            )

            // Feed all input buffers through the converter
            var inputBufferIndex = 0
            var inputFrameOffset: AVAudioFrameCount = 0

            let outputBufferSize: AVAudioFrameCount = 4096
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: outputBufferSize
            ) else {
                throw SpeechSynthesisError.synthesisFailed("Cannot allocate output buffer")
            }

            while inputBufferIndex < buffers.count {
                var error: NSError?
                let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                    if inputBufferIndex >= buffers.count {
                        outStatus.pointee = .endOfStream
                        return nil
                    }
                    let currentBuffer = buffers[inputBufferIndex]
                    inputBufferIndex += 1
                    outStatus.pointee = .haveData
                    return currentBuffer
                }

                if let error { throw SpeechSynthesisError.synthesisFailed(error.localizedDescription) }
                if status == .error { throw SpeechSynthesisError.synthesisFailed("AAC conversion error") }

                if outputBuffer.frameLength > 0 {
                    try outputFile.write(from: outputBuffer)
                }

                if status == .endOfStream { break }
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
```

- [ ] **Step 2: Verify build compiles**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Models/SpeechSynthesisService.swift
git commit -m "feat: add SpeechSynthesisService for on-device TTS"
```

### Task 2: Add TTS routes to SpeechAPIController

**Files:**
- Modify: `Sources/MacLocalAPI/Controllers/SpeechAPIController.swift`

- [ ] **Step 1: Add TTS request struct, voice listing route, and speech synthesis route**

Add to `SpeechAPIController`:

1. A `TTSSpeechRequest` struct:
```swift
struct TTSSpeechRequest: Content {
    let model: String?
    let input: String
    let voice: String?
    let responseFormat: String?
    let speed: Double?
    let locale: String?
    let appleVoice: String?

    enum CodingKeys: String, CodingKey {
        case model, input, voice
        case responseFormat = "response_format"
        case speed, locale
        case appleVoice = "apple_voice"
    }
}
```

2. Register new routes in `boot(routes:)`:
```swift
speech.on(.POST, "speech", body: .collect(maxSize: "1mb"), use: synthesize)
speech.on(.OPTIONS, "speech", use: handleOptions)
speech.on(.GET, "voices", use: listVoices)
speech.on(.OPTIONS, "voices", use: handleOptions)
```

3. A `synthesize(req:)` handler that:
   - Decodes `TTSSpeechRequest`
   - Validates `input` is non-empty and under 4096 chars (return 400)
   - Parses `response_format` into `TTSAudioFormat` (default `.aac`)
   - Creates `TTSRequestOptions` from the request fields
   - Calls `SpeechSynthesisService().synthesize(text:options:)`
   - Returns raw audio bytes with correct `Content-Type` header

4. A `listVoices(req:)` handler that:
   - Reads optional `locale` query param
   - Calls `SpeechSynthesisService.listVoices(locale:)`
   - Returns JSON `{"voices": [...]}`

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Controllers/SpeechAPIController.swift
git commit -m "feat: add TTS and voice listing API endpoints"
```

### Task 3: Add TTS CLI commands to SpeechCommand

**Files:**
- Modify: `Sources/MacLocalAPI/SpeechCommand.swift`
- Modify: `Sources/MacLocalAPI/main.swift`

- [ ] **Step 1: Restructure SpeechCommand with subcommands**

Refactor `SpeechCommand` to have two subcommands: `SpeechTranscribeCommand` and `SpeechSynthesizeCommand`, plus a `--list-voices` flag.

The existing `SpeechCommand` fields (`file`, `locale`, `helpJson`) move into `SpeechTranscribeCommand`. Add `SpeechSynthesizeCommand` with `text` (positional), `--voice`, `--locale`, `--format` (aac/wav/caf, default aac), `--speed`, `-o`/`--output`.

Add `--list-voices` and `--locale` to the parent `SpeechCommand`.

Key subcommand structure:
```
afm speech --list-voices [--locale en]
afm speech synthesize "text" [--voice alloy] [--locale en-US] [--format aac] [--speed 1.0] [-o file]
afm speech transcribe -f file.wav [--locale en-US] [--format json]
afm speech -f file.wav  (legacy — dispatches to transcribe)
```

- [ ] **Step 2: Update main.swift dispatch**

The existing `else if CommandLine.arguments[1] == "speech"` block on the feature branch dispatches to `SpeechCommand`. Since we're restructuring SpeechCommand, update the dispatch to handle the new subcommand structure. Keep backward compat: if args contain `-f`, treat as `transcribe`.

Add the speech dispatch block (it's not yet on main — comes from the merge in Task 0):
```swift
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "speech" {
    let args = Array(CommandLine.arguments.dropFirst(2))
    do {
        let cmd = try SpeechCommand.parse(args)
        let group = DispatchGroup()
        var caughtError: Error?
        group.enter()
        Task {
            do {
                try await cmd.run()
            } catch {
                caughtError = error
            }
            group.leave()
        }
        group.wait()
        if let error = caughtError {
            throw error
        }
    } catch {
        SpeechCommand.exit(withError: error)
    }
}
```

- [ ] **Step 3: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 4: Commit**

```bash
git add Sources/MacLocalAPI/SpeechCommand.swift Sources/MacLocalAPI/main.swift
git commit -m "feat: add TTS CLI (afm speech synthesize) and voice listing"
```

---

## Part 2: Transcription Enhancements

### Task 4: Enhance SpeechService with rich results

**Files:**
- Modify: `Sources/MacLocalAPI/Models/SpeechService.swift`

- [ ] **Step 1: Add rich transcription result types and enhance the service**

Add result structs:
```swift
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
}
```

Add a new method `transcribeWithDetails(from:options:)` that returns `TranscriptionResult`. The existing `transcribe(from:options:)` can call it and return just `.text`.

In the recognition task callback, when `result.isFinal`, extract:
- `result.bestTranscription.formattedString` for `text`
- `result.bestTranscription.segments` for word-level timing (each `SFTranscriptionSegment` has `timestamp`, `duration`, `substring`, `confidence`)
- Build segments by grouping words into sentence-like chunks (or treat the whole transcription as one segment)
- `duration` = last segment's `timestamp + duration`

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Models/SpeechService.swift
git commit -m "feat: add rich transcription results with timestamps"
```

### Task 5: Add response format support to SpeechAPIController

**Files:**
- Modify: `Sources/MacLocalAPI/Controllers/SpeechAPIController.swift`

- [ ] **Step 1: Update transcription endpoint**

Update the `TranscriptionRequest` struct to add new fields:
```swift
struct TranscriptionRequest: Content {
    let file: String?
    let data: String?
    let format: String?
    let locale: String?
    let model: String?           // accepted, ignored
    let language: String?        // ISO-639-1, alias for locale
    let responseFormat: String?  // json, text, verbose_json, srt, vtt
    let timestampGranularities: [String]?

    enum CodingKeys: String, CodingKey {
        case file, data, format, locale, model, language
        case responseFormat = "response_format"
        case timestampGranularities = "timestamp_granularities"
    }
}
```

Resolve `language` as alias for `locale` (language takes precedence if both provided).

Add response formatting methods:
- `formatAsJSON(result:)` — `{"text": "..."}`
- `formatAsText(result:)` — plain string
- `formatAsVerboseJSON(result:granularities:)` — full structure with words/segments
- `formatAsSRT(result:)` — standard SRT subtitle format
- `formatAsVTT(result:)` — standard WebVTT subtitle format

Route the controller's transcribe handler to call `transcribeWithDetails` and format based on `response_format`.

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Controllers/SpeechAPIController.swift
git commit -m "feat: add response_format support to transcription endpoint"
```

### Task 6: Update SpeechCommand for new transcription options

**Files:**
- Modify: `Sources/MacLocalAPI/SpeechCommand.swift`

- [ ] **Step 1: Add --format, --language, --timestamps flags to SpeechTranscribeCommand**

```swift
@Option(name: .long, help: "Output format: json, text, verbose_json, srt, vtt (default: text for CLI)")
var format: String = "text"

@Option(name: .long, help: "Language code (ISO-639-1, e.g. 'en', 'ja')")
var language: String?

@Option(name: .long, help: "Timestamp granularities: word, segment, or both (comma-separated)")
var timestamps: String?
```

Update `run()` to use `transcribeWithDetails` when format is not `text`, and format output accordingly. For CLI, default to `text` (just print the transcription). For `verbose_json`, print the full JSON. For `srt`/`vtt`, print subtitle text.

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/SpeechCommand.swift
git commit -m "feat: add format/language/timestamps CLI flags for transcription"
```

---

## Part 3: Vision Enhancements

### Task 7: Relax VisionService macOS gating and add new modes

**Files:**
- Modify: `Sources/MacLocalAPI/Models/VisionService.swift`

- [ ] **Step 1: Remove class-level @available and add per-mode checks**

Change `VisionService` from `@available(macOS 26.0, *)` to no class-level restriction (or `@available(macOS 13.0, *)`). Keep `@available(macOS 26.0, *)` on the text/table methods that use `VNRecognizeTextRequest`.

**Access level change:** Change `validateFile(at:maxBytes:)` from `private` to `internal` so the new mode methods can reuse it.

**macOS version gating change:** Remove `@available(macOS 26.0, *)` from `VisionService` class declaration. Add `@available(macOS 26.0, *)` to individual text/table methods (`extractText`, `extractTextWithDetails`, `extractTables`, `debugRawDetection`, `analyzeDocument`, `makeTextRequest`). The new barcode/classify/saliency methods have no macOS 26 restriction (Vision framework barcode/classification/saliency APIs are available since macOS 13).

**Update `VisionServing` protocol** to add new methods. Update `@available(macOS 26.0, *)` conformance extension to only cover the text/table methods. Add a separate protocol or extend `VisionServing` with default-nil implementations for the new methods so they're accessible on macOS 13+.

**Update `VisionAPIController.defaultVisionServiceFactory()`** to create `VisionService()` on macOS 13+ (not just 26+). For modes that require macOS 26, the individual service methods will check `#available` internally.

Add new result types:
```swift
struct BarcodeResult: Sendable {
    let type: String
    let payload: String
    let boundingBox: CGRect
    let confidence: Float
}

struct ClassificationLabel: Sendable {
    let label: String
    let confidence: Float
}

struct ClassifyResult: Sendable {
    let labels: [ClassificationLabel]
    let salientRegions: [CGRect]
}

struct SaliencyRegion: Sendable {
    let type: String
    let boundingBox: CGRect
}

struct SaliencyResult: Sendable {
    let regions: [SaliencyRegion]
    let heatMapPNG: Data?
}
```

Note: `CGRect` is fine in service-layer structs — the controller maps them to `VisionOCRBoundingBox` (which is `Codable`) when building API responses, following the existing pattern used for `TextBlock`.

Add new methods:
- `detectBarcodes(from:options:)` — uses `VNDetectBarcodesRequest`
- `classifyImage(from:maxLabels:)` — uses `VNClassifyImageRequest` + `VNGenerateAttentionBasedSaliencyImageRequest`
- `detectSaliency(from:type:includeHeatMap:)` — uses `VNGenerateAttentionBasedSaliencyImageRequest` or `VNGenerateObjectnessBasedSaliencyImageRequest`
- `autoCrop(imageData:)` — uses `VNDetectDocumentSegmentationRequest`, returns cropped `Data`

Each method reuses the now-`internal` `validateFile`. The barcode/classify/saliency methods do NOT require macOS 26.

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Models/VisionService.swift
git commit -m "feat: add barcode, classify, saliency modes to VisionService"
```

### Task 8: Update VisionAPIController for new modes

**Files:**
- Modify: `Sources/MacLocalAPI/Controllers/VisionAPIController.swift`

- [ ] **Step 1: Extend request/response types and routing**

Update `VisionOCRRequest` to add new fields:
```swift
let mode: String?           // text, table, barcode, classify, saliency, auto
let detail: String?         // high, low — alias for recognitionLevel
let autoCrop: Bool?         // run document segmentation first
let responseFormat: String? // json, text, verbose_json
let maxLabels: Int?         // for classify mode
let saliencyType: String?   // attention, objectness
let includeHeatMap: Bool?   // for saliency mode
```

Add CodingKeys for `auto_crop`, `response_format`, `max_labels`, `saliency_type`, `include_heat_map`.

Update `VisionServing` protocol to include new methods.

Add response types:
```swift
struct VisionBarcodeResponse: Content { ... }
struct VisionClassifyResponse: Content { ... }
struct VisionSaliencyResponse: Content { ... }
struct VisionAutoResponse: Content { ... }  // combined results with modes_run
```

Update the `ocr` handler to:
1. Resolve `mode` (default `"text"`, honor `table` bool as alias)
2. Resolve `detail` as alias for `recognition_level` (`high`→`accurate`, `low`→`fast`)
3. If `auto_crop` is true, run document segmentation first on the input
4. Route to the correct service method based on mode
5. Format response based on `response_format` (default `json`)
6. For `auto` mode, run text+barcode+classify via `TaskGroup`. On pre-macOS 26, omit text (barcode+classify only). Include `modes_run` array in response indicating which modes executed.
7. Return 501 with message if a specific mode requires macOS 26 and host is pre-26 (e.g., `mode=text` on pre-26)

Empty results (no barcodes found, etc.) return 200 with empty arrays, not errors.

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/Controllers/VisionAPIController.swift
git commit -m "feat: add mode/detail/auto_crop/response_format to vision API"
```

### Task 9: Update VisionCommand for new CLI flags

**Files:**
- Modify: `Sources/MacLocalAPI/VisionCommand.swift`

- [ ] **Step 1: Add new flags to VisionCommand**

Add:
```swift
@Option(name: .long, help: "Vision mode: text, table, barcode, classify, saliency, auto (default: text)")
var mode: String = "text"

@Option(name: .long, help: "Recognition detail: high or low (default: high)")
var detail: String = "high"

@Flag(name: .long, help: "Auto-crop document region before processing")
var autoCrop: Bool = false

@Option(name: .long, help: "Output format: json, text, verbose_json (default: text for CLI)")
var format: String = "text"

@Option(name: .long, help: "Max classification labels to return (default: 5)")
var maxLabels: Int = 5

@Option(name: .long, help: "Saliency type: attention or objectness (default: attention)")
var saliencyType: String = "attention"

@Flag(name: .long, help: "Include saliency heat map as base64 PNG")
var heatMap: Bool = false
```

Update `run()` to:
- When `mode == "table"` or `--table` flag, use table extraction
- When `mode == "barcode"`, call barcode detection
- When `mode == "classify"`, call classification
- When `mode == "saliency"`, call saliency detection
- When `mode == "auto"`, run text+barcode+classify
- Apply `--auto-crop` as preprocessing
- Map `--detail high` → `.accurate`, `--detail low` → `.fast`
- Format output per `--format` (CLI defaults to `text` for readable output)

Keep existing `--table` and `--verbose` flags as aliases for backward compat.

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit**

```bash
git add Sources/MacLocalAPI/VisionCommand.swift
git commit -m "feat: add mode/detail/auto-crop/format CLI flags to vision command"
```

---

## Part 4: Server Registration

### Task 10: Register new routes in Server.swift

**Files:**
- Modify: `Sources/MacLocalAPI/Server.swift`

- [ ] **Step 1: Ensure SpeechAPIController is registered**

After the merge in Task 0, `SpeechAPIController` should already be registered. Verify it's present. If not, add:
```swift
try app.register(collection: SpeechAPIController())
```
right after the `VisionAPIController()` registration (line 243).

The TTS and voice listing routes are already part of `SpeechAPIController.boot()` from Task 2, so no additional registration is needed.

- [ ] **Step 2: Verify build**

Run: `swift build 2>&1 | tail -5`
Expected: Build succeeded

- [ ] **Step 3: Commit (if changes needed)**

```bash
git add Sources/MacLocalAPI/Server.swift
git commit -m "feat: register speech API controller in server"
```

---

## Part 5: Manual Testing

### Task 11: End-to-end smoke test

- [ ] **Step 1: Build release**

```bash
swift build -c release 2>&1 | tail -5
```

- [ ] **Step 2: Test TTS endpoint**

```bash
curl -s http://localhost:9999/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from AFM", "voice": "alloy"}' \
  -o /tmp/tts-test.aac
file /tmp/tts-test.aac
```
Expected: Valid audio file

- [ ] **Step 3: Test voice listing**

```bash
curl -s http://localhost:9999/v1/audio/voices | python3 -m json.tool | head -20
```
Expected: JSON with voices array

- [ ] **Step 4: Test transcription verbose_json**

```bash
curl -s http://localhost:9999/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{"file": "/path/to/test.wav", "response_format": "verbose_json"}' | python3 -m json.tool
```
Expected: JSON with text, duration, words, segments

- [ ] **Step 5: Test vision barcode mode**

```bash
curl -s http://localhost:9999/v1/vision/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "/path/to/qr-code.png", "mode": "barcode"}' | python3 -m json.tool
```
Expected: JSON with barcode results

- [ ] **Step 6: Test vision classify mode**

```bash
curl -s http://localhost:9999/v1/vision/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "/path/to/photo.jpg", "mode": "classify"}' | python3 -m json.tool
```
Expected: JSON with labels and salient_regions

- [ ] **Step 7: Test vision saliency mode**

```bash
curl -s http://localhost:9999/v1/vision/ocr \
  -H "Content-Type: application/json" \
  -d '{"file": "/path/to/photo.jpg", "mode": "saliency"}' | python3 -m json.tool
```
Expected: JSON with regions

- [ ] **Step 8: Test CLI commands**

```bash
afm speech --list-voices --locale en | head -10
afm speech synthesize "Hello world" -o /tmp/cli-tts.aac
afm vision -f /path/to/image.jpg --mode classify
afm vision -f /path/to/image.jpg --mode barcode
afm vision -f /path/to/image.jpg --mode saliency
```
