import AFMKit
import AFMKitMLX
import AFMKitDwarfStar
import ArgumentParser
import Foundation

struct MLXConvertCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-convert",
        abstract: "Convert a supported local checkpoint to native AFM/MLX format"
    )

    @Option(
        name: .long,
        help: "Existing local checkpoint directory (automatic model download is disabled)"
    )
    var source: String

    @Option(name: .long, help: "Persistent output directory for the converted checkpoint")
    var output: String

    @Flag(name: .long, help: "Delete and recreate an existing conversion directory")
    var overwrite = false

    @Option(
        name: .long,
        help: "Conversion profile. Defaults to native for DeepSeek V4 and mlx-affine-4 for GLM-5.3 Flash"
    )
    var profile: String?

    @Option(
        name: .long,
        help: "Pinned 40-character Hugging Face source commit (required for GLM unless inferable from the local snapshot)"
    )
    var sourceRevision: String?

    @Option(
        name: .long,
        help: "Reference DS4 GGUF used to bundle the compact metadata template required by dwarfstar-executor"
    )
    var templateGGUF: String?

    mutating func run() throws {
        let sourceURL = URL(fileURLWithPath: source, isDirectory: true)
        let outputURL = URL(fileURLWithPath: output, isDirectory: true)
        let inspection = try AFMMLXCheckpointConverter.inspect(
            source: sourceURL,
            sourceRevision: sourceRevision)
        let selectedProfile = profile ?? inspection.defaultProfile
        guard inspection.supportedProfiles.contains(selectedProfile) else {
            throw ValidationError(
                "Unknown profile '\(selectedProfile)' for \(inspection.modelKind.rawValue). Expected: \(inspection.supportedProfiles.joined(separator: ", "))")
        }
        let needsDwarfStarTemplate = inspection.modelKind == .deepseekV4
            && selectedProfile
                == DeepseekV4CheckpointConverter.Profile.dwarfstarExecutor.rawValue
        if needsDwarfStarTemplate, templateGGUF?.isEmpty != false {
            throw ValidationError(
                "--template-gguf is required for dwarfstar-executor so the converted checkpoint is self-contained")
        }
        if !needsDwarfStarTemplate, templateGGUF != nil {
            throw ValidationError(
                "--template-gguf is only valid with the DeepSeek dwarfstar-executor profile")
        }
        let storage = try MLXConversionStoragePreflight.validate(
            source: sourceURL,
            output: outputURL,
            inspection: inspection)
        if let required = storage.requiredBytes, let available = storage.availableBytes {
            print("Destination preflight: \(Self.bytes(available)) free; \(Self.bytes(required)) required")
        }
        if let revision = inspection.sourceRevision {
            print("Pinned source revision: \(revision)")
        }

        let converter = AFMMLXCheckpointConverter(
            source: sourceURL,
            output: outputURL,
            overwrite: overwrite,
            profile: selectedProfile,
            sourceRevision: inspection.sourceRevision,
            progress: { print($0) })
        try converter.run()

        if needsDwarfStarTemplate {
            let templateOutput = URL(fileURLWithPath: output, isDirectory: true)
                .appendingPathComponent(
                    AFMDwarfStarCheckpointCatalog.bundledTemplateFilename,
                    isDirectory: false)
            try AFMDwarfStarProjection.writeMetadataTemplate(
                from: URL(fileURLWithPath: templateGGUF!),
                to: templateOutput)
            print("Bundled DwarfStar metadata template: \(templateOutput.path)")
        }
    }

    private static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .decimal)
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
