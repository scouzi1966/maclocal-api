import ArgumentParser
import Foundation

struct SpeechCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech",
        abstract: "Speech synthesis and transcription using Apple frameworks",
        discussion: """
        Use Apple's Speech and AVFoundation frameworks for on-device speech-to-text
        and text-to-speech.

        Subcommands:
          synthesize  — Convert text to speech audio
          transcribe  — Transcribe audio files to text
          voices      — List available synthesis voices

        Legacy shortcuts:
          afm speech file.wav     — Equivalent to: afm speech transcribe -f file.wav
          afm speech -f file.wav  — Equivalent to: afm speech transcribe -f file.wav

        Examples:
          afm speech synthesize "Hello world" -o output.aac
          afm speech synthesize "Hello" --voice nova --format wav
          afm speech voices --locale en
          afm speech transcribe -f recording.wav
          afm speech transcribe -f meeting.mp3 --format verbose_json
          afm speech -f audio.wav                         # legacy transcribe shortcut
        """,
        subcommands: [SpeechSynthesizeCommand.self, SpeechTranscribeCommand.self, SpeechVoicesCommand.self]
    )

    @Flag(name: .long, help: "Print machine-readable JSON capability card for AI agents and exit")
    var helpJson: Bool = false

    func run() async throws {
        if helpJson {
            printHelpJson(command: "afm speech")
            return
        }

        // No subcommand and no flags — show help
        throw CleanExit.helpRequest(self)
    }
}

struct SpeechVoicesCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voices",
        abstract: "List available speech synthesis voices"
    )

    @Option(name: .long, help: "Filter voices by locale prefix (e.g. 'en', 'ja-JP')")
    var locale: String?

    func run() async throws {
        guard #available(macOS 13.0, *) else {
            throw SpeechError.platformUnavailable
        }

        let voices = SpeechSynthesisService.listVoices(locale: locale)
        if voices.isEmpty {
            print("No voices found\(locale.map { " for locale '\($0)'" } ?? "").")
            return
        }
        let maxNameLen = voices.map(\.name.count).max() ?? 10
        let maxLocaleLen = voices.map(\.locale.count).max() ?? 5
        for voice in voices {
            let name = voice.name.padding(toLength: maxNameLen, withPad: " ", startingAt: 0)
            let loc = voice.locale.padding(toLength: maxLocaleLen, withPad: " ", startingAt: 0)
            print("\(name)  \(loc)  \(voice.gender)  \(voice.quality)  \(voice.id)")
        }
    }
}

struct SpeechSynthesizeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "synthesize",
        abstract: "Convert text to speech audio"
    )

    @Argument(help: "Text to synthesize")
    var text: String

    @Option(name: .long, help: "Voice name: alloy, echo, fable, nova, onyx, shimmer (default: alloy)")
    var voice: String = "alloy"

    @Option(name: .long, help: "Locale for voice selection (default: en-US)")
    var locale: String = "en-US"

    @Option(name: .long, help: "Audio format: aac, wav, caf (default: aac)")
    var format: String = "aac"

    @Option(name: .long, help: "Speech speed: 0.25-4.0 (default: 1.0)")
    var speed: Float = 1.0

    @Option(name: [.short, .long], help: "Output file path (default: stdout)")
    var output: String?

    func run() async throws {
        guard #available(macOS 13.0, *) else {
            throw SpeechSynthesisError.platformUnavailable
        }

        guard let audioFormat = TTSAudioFormat(rawValue: format.lowercased()) else {
            print("Error: Unsupported format '\(format)'. Supported: aac, wav, caf")
            throw ExitCode.failure
        }

        let options = TTSRequestOptions(
            voice: voice,
            locale: locale,
            speed: speed,
            format: audioFormat
        )

        do {
            let service = SpeechSynthesisService()
            let audioData = try await service.synthesize(text: text, options: options)

            if let outputPath = output {
                let expandedPath = NSString(string: outputPath).expandingTildeInPath
                let url = URL(fileURLWithPath: expandedPath)
                try audioData.write(to: url)
                print("Wrote \(audioData.count) bytes to \(expandedPath)")
            } else {
                // Write raw audio to stdout
                FileHandle.standardOutput.write(audioData)
            }
        } catch {
            if let synthError = error as? SpeechSynthesisError {
                print("Error: \(synthError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
            throw ExitCode.failure
        }
    }
}

struct SpeechTranscribeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe audio files to text"
    )

    @Option(name: [.short, .long], help: "Path to the audio file")
    var file: String

    @Option(name: .long, help: "Locale for speech recognition (default: en-US)")
    var locale: String = "en-US"

    @Option(name: .long, help: "Output format: text, json, verbose_json, srt, vtt (default: text)")
    var format: String = "text"

    @Option(name: .long, help: "Language code (ISO-639-1, e.g. 'en', 'ja'). Alias for --locale.")
    var language: String?

    @Option(name: .long, help: "Timestamp granularities: word, segment, or both (comma-separated)")
    var timestamps: String?

    func run() async throws {
        let expandedPath = NSString(string: file).expandingTildeInPath
        let resolvedPath = URL(fileURLWithPath: expandedPath).resolvingSymlinksInPath().path

        // language takes precedence over locale; pass bare codes through —
        // SFSpeechRecognizer resolves them to the user's preferred regional variant
        let effectiveLocale = language ?? locale

        do {
            if #available(macOS 13.0, *) {
                let service = SpeechService()
                let options = SpeechRequestOptions(locale: effectiveLocale)

                switch format.lowercased() {
                case "text":
                    let text = try await service.transcribe(from: resolvedPath, options: options)
                    print(text)
                case "json":
                    let text = try await service.transcribe(from: resolvedPath, options: options)
                    let jsonObj: [String: Any] = ["text": text]
                    let jsonData = try JSONSerialization.data(withJSONObject: jsonObj, options: [.prettyPrinted])
                    print(String(data: jsonData, encoding: .utf8) ?? "{}")
                case "verbose_json", "srt", "vtt":
                    let result = try await service.transcribeWithDetails(from: resolvedPath, options: options)
                    let granularities = parseGranularities()
                    switch format.lowercased() {
                    case "verbose_json":
                        let verbose = VerboseTranscriptionResponse(
                            text: result.text,
                            language: result.language,
                            duration: result.duration,
                            words: granularities.contains("word") ? result.words : nil,
                            segments: granularities.contains("segment") ? result.segments : nil
                        )
                        let data = try JSONEncoder.prettyPrinted.encode(verbose)
                        print(String(data: data, encoding: .utf8) ?? "{}")
                    case "srt":
                        print(result.formatAsSRT())
                    case "vtt":
                        print(result.formatAsVTT())
                    default:
                        break
                    }
                default:
                    print("Error: Unsupported format '\(format)'. Supported: text, json, verbose_json, srt, vtt")
                    throw ExitCode.failure
                }
            } else {
                throw SpeechError.platformUnavailable
            }
        } catch {
            if let speechError = error as? SpeechError {
                print("Error: \(speechError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
            throw ExitCode.failure
        }
    }

    private func parseGranularities() -> Set<String> {
        guard let raw = timestamps else { return ["segment"] }
        return Set(raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() })
    }
}

private extension JSONEncoder {
    static let prettyPrinted: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()
}
