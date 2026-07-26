#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public enum AFMFoundationToolInvocationStatus: String, Sendable {
    case requested
    case completed
}

@available(macOS 27.0, *)
public struct AFMFoundationToolInvocationSnapshot: Equatable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let argumentsJSON: String?
    public let outputPreview: String?
    public let status: AFMFoundationToolInvocationStatus

    public init(
        id: String,
        name: String,
        argumentsJSON: String?,
        outputPreview: String?,
        status: AFMFoundationToolInvocationStatus
    ) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
        self.outputPreview = outputPreview
        self.status = status
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationTranscriptSnapshotParser {
    public static let defaultPreviewLimit = 2_048

    public static func toolInvocations<S: Sequence>(
        from entries: S,
        previewLimit: Int = defaultPreviewLimit
    ) -> [AFMFoundationToolInvocationSnapshot] where S.Element == Transcript.Entry {
        var order: [String] = []
        var invocations: [String: AFMFoundationToolInvocationSnapshot] = [:]

        for entry in entries {
            switch entry {
            case .toolCalls(let calls):
                for call in calls {
                    if invocations[call.id] == nil {
                        order.append(call.id)
                    }
                    let existing = invocations[call.id]
                    invocations[call.id] = AFMFoundationToolInvocationSnapshot(
                        id: call.id,
                        name: call.toolName,
                        argumentsJSON: bounded(call.arguments.jsonString, limit: previewLimit),
                        outputPreview: existing?.outputPreview,
                        status: existing?.status ?? .requested
                    )
                }
            case .toolOutput(let output):
                let invocationID = invocations[output.id] != nil
                    ? output.id
                    : order.last(where: {
                        invocations[$0]?.name == output.toolName
                            && invocations[$0]?.status == .requested
                    }) ?? output.id
                if invocations[invocationID] == nil {
                    order.append(invocationID)
                }
                let existing = invocations[invocationID]
                invocations[invocationID] = AFMFoundationToolInvocationSnapshot(
                    id: invocationID,
                    name: existing?.name ?? output.toolName,
                    argumentsJSON: existing?.argumentsJSON,
                    outputPreview: bounded(render(output.segments), limit: previewLimit),
                    status: .completed
                )
            default:
                continue
            }
        }

        return order.compactMap { invocations[$0] }
    }

    public static func reasoningContent<S: Sequence>(
        from entries: S
    ) -> String where S.Element == Transcript.Entry {
        var reasoningSegments: [String] = []

        for entry in entries {
            switch entry {
            case .prompt:
                reasoningSegments.removeAll()
            case .reasoning(let reasoning):
                let rendered = render(reasoning.segments)
                if !rendered.isEmpty {
                    reasoningSegments.append(rendered)
                }
            default:
                continue
            }
        }

        return reasoningSegments.joined(separator: "\n")
    }

    public static func bounded(_ value: String?, limit: Int = defaultPreviewLimit) -> String? {
        guard let value, !value.isEmpty else { return nil }
        guard value.count > limit else { return value }
        return String(value.prefix(limit)) + "…"
    }

    private static func render(_ segments: [Transcript.Segment]) -> String {
        segments.map { segment in
            switch segment {
            case .text(let text):
                return text.content
            case .structure(let structure):
                return structure.content.jsonString
            case .attachment(let attachment):
                return attachment.label.map { "[Attachment: \($0)]" } ?? "[Attachment]"
            case .custom:
                return "[Custom tool output]"
            @unknown default:
                return "[Unknown tool output]"
            }
        }
        .joined(separator: "\n")
    }
}
#endif
