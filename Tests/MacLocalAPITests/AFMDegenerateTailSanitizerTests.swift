import Foundation
import XCTest
@testable import AFMServer

final class AFMDegenerateTailSanitizerTests: XCTestCase {
    func testMatchesLegacyRegexForBoundedInputs() throws {
        let regex = try NSRegularExpression(pattern: "([!?.:,;`~_\\-*=|])\\1{79,}$")
        for punctuation in "!?.:,;`~_-=*|" {
            for count in [0, 1, 79, 80, 81, 160] {
                for ending in ["", "x", "\n", "\r", "\r\n", "\u{85}", "\u{2028}", "\u{2029}", "\n\n"] {
                    let text = "Answer 🙂  " + String(repeating: String(punctuation), count: count) + ending
                    let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
                    let expected: String
                    if let match, let range = Range(match.range, in: text) {
                        expected = String(text[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                    } else {
                        expected = text
                    }
                    XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize(text), expected)
                }
            }
        }
    }

    func testLongNearMatchDoesNotScanEveryStartingPosition() {
        let text = "Answer " + String(repeating: "!", count: 100_000) + "x"
        let start = Date()
        XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize(text), text)
        XCTAssertLessThan(Date().timeIntervalSince(start), 2)
    }

    func testLongTailAndReplacementCharacterBoundaries() {
        XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize("Answer " + String(repeating: "!", count: 100_000)), "Answer")
        XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize("Answer�broken"), "Answer")
        XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize("Answer�" + String(repeating: "x", count: 510)), "Answer")
        let distant = "Answer�" + String(repeating: "x", count: 511)
        XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize(distant), distant)
        XCTAssertEqual(AFMDegenerateTailSanitizer.sanitize(""), "")
    }
}
