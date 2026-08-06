#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationStructuredResponseCompletion: Equatable, Sendable {
    public let renderedContent: String
    public let telemetry: AFMFoundationGenerationTelemetry
    public let toolInvocations: [AFMFoundationToolInvocationSnapshot]

    public init(
        renderedContent: String,
        telemetry: AFMFoundationGenerationTelemetry,
        toolInvocations: [AFMFoundationToolInvocationSnapshot]
    ) {
        self.renderedContent = renderedContent
        self.telemetry = telemetry
        self.toolInvocations = toolInvocations
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationStructuredResponseCompleter {
    public static func complete<S: Sequence>(
        rawContent: GeneratedContent,
        label: String,
        usage: LanguageModelSession.Usage,
        transcriptEntries: S,
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        completedAt: ContinuousClock.Instant,
        render: (GeneratedContent) -> String
    ) throws -> AFMFoundationStructuredResponseCompletion where S.Element == Transcript.Entry {
        try complete(
            rawContent: rawContent,
            label: label,
            usage: AFMFoundationGenerationTelemetryCalculator.usage(from: usage),
            transcriptEntries: transcriptEntries,
            contextAction: contextAction,
            startedAt: startedAt,
            completedAt: completedAt,
            render: render
        )
    }

    public static func complete<S: Sequence>(
        rawContent: GeneratedContent,
        label: String,
        usage: AFMFoundationGenerationUsage,
        transcriptEntries: S,
        contextAction: String?,
        startedAt: ContinuousClock.Instant,
        completedAt: ContinuousClock.Instant,
        render: (GeneratedContent) -> String
    ) throws -> AFMFoundationStructuredResponseCompletion where S.Element == Transcript.Entry {
        let rendered = try AFMFoundationGeneratedContentRenderer.nonEmptyRenderedContent(
            rawContent,
            label: label,
            render: render
        )
        let toolInvocations = AFMFoundationTranscriptSnapshotParser.toolInvocations(from: transcriptEntries)
        let telemetry = AFMFoundationGenerationTelemetryCalculator.singleResponseTelemetry(
            usage: usage,
            toolNames: Array(Set(toolInvocations.map(\.name))).sorted(),
            contextAction: contextAction,
            startedAt: startedAt,
            completedAt: completedAt
        )
        return AFMFoundationStructuredResponseCompletion(
            renderedContent: rendered,
            telemetry: telemetry,
            toolInvocations: toolInvocations
        )
    }
}
#endif
