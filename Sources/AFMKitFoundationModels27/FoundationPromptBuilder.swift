#if canImport(FoundationModels)
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationPromptAttachment: Equatable, Sendable {
    public var url: URL
    public var label: String
    public var instruction: String

    public init(url: URL, label: String, instruction: String? = nil) {
        self.url = url
        self.label = label
        self.instruction = instruction
            ?? "The attached image is labeled '\(label)'. Use that exact label for image tools."
    }
}

@available(macOS 27.0, *)
public enum AFMFoundationPromptBuilder {
    public static func prompt(
        text: String,
        attachment: AFMFoundationPromptAttachment? = nil
    ) -> Prompt {
        Prompt {
            text
            if let attachment {
                attachment.instruction
                Attachment(imageURL: attachment.url).label(attachment.label)
            }
        }
    }
}
#endif
