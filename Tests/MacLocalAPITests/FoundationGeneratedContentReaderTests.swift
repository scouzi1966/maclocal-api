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
}
#endif
