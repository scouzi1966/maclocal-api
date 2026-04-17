import ArgumentParser
import Foundation

struct SpeechCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech",
        abstract: "Transcribe audio files using Apple's Speech framework",
        discussion: """
        Use Apple's Speech framework to perform on-device speech-to-text transcription.

        Supported formats: WAV, MP3, M4A, CAF, AIFF

        Examples:
          afm speech -f recording.wav
          afm speech --file /path/to/audio.m4a
          afm speech -f meeting.mp3 --locale ja-JP
        """
    )

    @Option(name: [.short, .long], help: "Path to the audio file")
    var file: String

    @Option(name: .long, help: "Locale for speech recognition (default: en-US)")
    var locale: String = "en-US"

    @Flag(name: .long, help: "Print machine-readable JSON capability card for AI agents and exit")
    var helpJson: Bool = false

    func run() async throws {
        if helpJson {
            printHelpJson(command: "afm speech")
            return
        }

        guard !file.isEmpty else {
            print("Error: File path is required. Use -f or --file to specify the input file.")
            throw ExitCode.failure
        }

        let expandedPath = NSString(string: file).expandingTildeInPath
        let resolvedPath = URL(fileURLWithPath: expandedPath).standardized.path

        do {
            if #available(macOS 13.0, *) {
                let service = SpeechService()
                let options = SpeechRequestOptions(locale: locale)
                let text = try await service.transcribe(from: resolvedPath, options: options)
                print(text)
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
}
