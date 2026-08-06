#if canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationPromptBuilderTests: XCTestCase {
    func testBuildsTextPrompt() {
        let prompt = AFMFoundationPromptBuilder.prompt(text: "Hello")

        XCTAssertEqual(String(describing: type(of: prompt)), "Prompt")
    }

    func testBuildsAttachmentPromptWithDefaultInstruction() {
        let attachment = AFMFoundationPromptAttachment(
            url: URL(fileURLWithPath: "/tmp/image.png"),
            label: "vesta_image"
        )
        let prompt = AFMFoundationPromptBuilder.prompt(
            text: "Describe the image.",
            attachment: attachment
        )

        XCTAssertEqual(String(describing: type(of: prompt)), "Prompt")
        XCTAssertEqual(
            attachment.instruction,
            "The attached image is labeled 'vesta_image'. Use that exact label for image tools."
        )
    }
}
#endif
