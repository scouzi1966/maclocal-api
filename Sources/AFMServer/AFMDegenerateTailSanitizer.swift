import Foundation

enum AFMDegenerateTailSanitizer {
    private static let punctuation: Set<Character> = Set("!?.:,;`~_-=*|")
    private static let finalLineBreaks: Set<Character> = ["\n", "\r", "\r\n", "\u{85}", "\u{2028}", "\u{2029}"]

    static func sanitize(_ text: String) -> String {
        var cleaned = text
        if let badChar = cleaned.lastIndex(of: "�"),
           cleaned.distance(from: badChar, to: cleaned.endIndex) < 512 {
            cleaned = String(cleaned[..<badChar])
        }

        // Match the old ICU end anchor, which also permits one final line break.
        var end = cleaned.endIndex
        if end > cleaned.startIndex, finalLineBreaks.contains(cleaned[cleaned.index(before: end)]) {
            end = cleaned.index(before: end)
        }
        guard end > cleaned.startIndex else { return cleaned }
        let last = cleaned[cleaned.index(before: end)]
        guard punctuation.contains(last) else { return cleaned }

        // Scan the trailing run once. An unanchored backreference regex can
        // retry every starting position in a long near-match and monopolize CPU.
        var start = end
        var count = 0
        while start > cleaned.startIndex {
            let previous = cleaned.index(before: start)
            guard cleaned[previous] == last else { break }
            start = previous
            count += 1
        }
        guard count >= 80 else { return cleaned }
        return String(cleaned[..<start]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
