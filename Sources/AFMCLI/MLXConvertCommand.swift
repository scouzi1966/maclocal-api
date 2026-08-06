import AFMKitMLX
import AFMKitDwarfStar
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
        help: "Conversion profile: native, dwarfstar-executor, dwarfstar-q8, dwarfstar-q8-0, dwarfstar-symmetric-q8, dwarfstar-symmetric-q8-interleaved-mxfp4, or dwarfstar-symmetric-q8-aligned-mxfp4"
    )
    var profile = DeepseekV4CheckpointConverter.Profile.native.rawValue

    @Option(
        name: .long,
        help: "Reference DS4 GGUF used to bundle the compact metadata template required by dwarfstar-executor"
    )
    var templateGGUF: String?

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

        if conversionProfile == .dwarfstarExecutor {
            guard let templateGGUF, !templateGGUF.isEmpty else {
                throw ValidationError(
                    "--template-gguf is required for dwarfstar-executor so the converted checkpoint is self-contained")
            }
            let templateOutput = URL(fileURLWithPath: output, isDirectory: true)
                .appendingPathComponent(
                    AFMDwarfStarCheckpointCatalog.bundledTemplateFilename,
                    isDirectory: false)
            try AFMDwarfStarProjection.writeMetadataTemplate(
                from: URL(fileURLWithPath: templateGGUF),
                to: templateOutput)
            print("Bundled DwarfStar metadata template: \(templateOutput.path)")
        }
    }
}

struct MLXAlignExecutorCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-align-executor",
        abstract: "Align an existing AFM DwarfStar executor checkpoint for Metal"
    )

    @Option(name: .long, help: "Executor checkpoint directory to upgrade in place")
    var checkpoint: String

    mutating func run() throws {
        try AlignedSafetensorRewriter.rewriteCheckpoint(
            at: URL(fileURLWithPath: checkpoint, isDirectory: true),
            progress: { print($0) })
    }
}
