import AFMKitMLX
import ArgumentParser
import Foundation

struct MLXConvertCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-convert",
        abstract: "Convert an official DeepSeek V4 checkpoint to native AFM/MLX format"
    )

    @Option(name: .long, help: "Official DeepSeek V4 checkpoint directory")
    var source: String

    @Option(name: .long, help: "Persistent output directory for the converted checkpoint")
    var output: String

    @Flag(name: .long, help: "Delete and recreate an existing conversion directory")
    var overwrite = false

    @Option(
        name: .long,
        help: "Conversion profile: native, dwarfstar-q8, dwarfstar-symmetric-q8, dwarfstar-symmetric-q8-interleaved-mxfp4, or dwarfstar-symmetric-q8-aligned-mxfp4"
    )
    var profile = DeepseekV4CheckpointConverter.Profile.native.rawValue

    mutating func run() throws {
        guard let conversionProfile = DeepseekV4CheckpointConverter.Profile(rawValue: profile) else {
            throw ValidationError(
                "Unknown profile '\(profile)'. Expected: \(DeepseekV4CheckpointConverter.Profile.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        let converter = DeepseekV4CheckpointConverter(
            source: URL(fileURLWithPath: source),
            output: URL(fileURLWithPath: output),
            overwrite: overwrite,
            profile: conversionProfile,
            progress: { print($0) })
        try converter.run()
    }
}
