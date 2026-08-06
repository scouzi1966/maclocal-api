#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModels27
import XCTest

@available(macOS 27.0, *)
final class FoundationGeneratedContentReaderTests: XCTestCase {
    func testReadsStructuredGeneratedContentProperties() {
        let content = GeneratedContent(properties: [
            "title": "Sky color",
            "confidence": 92,
            "facts": ["Blue wavelengths scatter", "Sunlight contains many colors"]
        ])

        XCTAssertEqual(AFMFoundationGeneratedContentReader.string("title", in: content), "Sky color")
        XCTAssertEqual(AFMFoundationGeneratedContentReader.number("confidence", in: content), 92)
        XCTAssertEqual(
            AFMFoundationGeneratedContentReader.strings("facts", in: content),
            ["Blue wavelengths scatter", "Sunlight contains many colors"]
        )
        XCTAssertNil(AFMFoundationGeneratedContentReader.string("missing", in: content))
        XCTAssertEqual(AFMFoundationGeneratedContentReader.strings("missing", in: content), [])
    }

    func testJoinedSectionsFallsBackToJsonWhenEmpty() {
        let content = GeneratedContent(properties: [
            "title": "Artifact"
        ])

        XCTAssertEqual(
            AFMFoundationGeneratedContentReader.joinedSections(["One", "Two"], fallback: content),
            "One\n\nTwo"
        )
        XCTAssertEqual(
            AFMFoundationGeneratedContentReader.joinedSections([], fallback: content),
            content.jsonString
        )
    }

    func testNonEmptyRendererReturnsRenderedContent() throws {
        let content = GeneratedContent(properties: [
            "title": "Artifact"
        ])

        let rendered = try AFMFoundationGeneratedContentRenderer.nonEmptyRenderedContent(
            content,
            label: "Artifact"
        ) { _ in
            "Rendered"
        }

        XCTAssertEqual(rendered, "Rendered")
    }

    func testNonEmptyRendererThrowsForBlankContent() {
        let content = GeneratedContent(properties: [
            "title": "Artifact"
        ])

        XCTAssertThrowsError(
            try AFMFoundationGeneratedContentRenderer.nonEmptyRenderedContent(
                content,
                label: "Artifact"
            ) { _ in
                "  \n\t"
            }
        ) { error in
            XCTAssertEqual(
                error as? AFMFoundationStructuredResponseError,
                .emptyRenderedContent(label: "Artifact")
            )
            XCTAssertEqual(
                (error as? LocalizedError)?.errorDescription,
                "Artifact returned an empty structured response."
            )
        }
    }
}
#endif
