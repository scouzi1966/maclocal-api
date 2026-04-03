// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for the documented Gemma 4 function call format.
///
/// Gemma 4 uses:
/// - Start tag: `<|tool_call>`
/// - End tag: `<tool_call|>`
/// - Function body: `call:name{key:value,...}`
/// - Escaped strings: `<|"|>value<|"|>`
///
/// This parser intentionally does not perform schema-aware coercion.
/// It follows the model's surface format and returns raw argument strings.
///
/// Reference: https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
public struct Gemma4FunctionParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"

    private let escapeMarker = "<|\"|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if let startTag, text.hasPrefix(startTag) {
            text.removeFirst(startTag.count)
        }
        if let endTag, text.hasSuffix(endTag) {
            text.removeLast(endTag.count)
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let callRange = text.range(of: "call:") else { return nil }
        let payload = text[callRange.upperBound...]
        guard let braceStart = payload.firstIndex(of: "{"),
              let braceEnd = payload.lastIndex(of: "}") else { return nil }

        let functionName = String(payload[..<braceStart]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !functionName.isEmpty else { return nil }

        let argsBody = String(payload[payload.index(after: braceStart)..<braceEnd])
        return ToolCall(
            function: .init(
                name: functionName,
                arguments: parseArguments(argsBody)
            )
        )
    }

    private func parseArguments(_ text: String) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]
        var index = text.startIndex

        while true {
            skipDelimiters(in: text, index: &index)
            guard index < text.endIndex else { break }

            guard let colonIndex = text[index...].firstIndex(of: ":") else { break }
            let key = String(text[index..<colonIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { break }

            index = text.index(after: colonIndex)
            skipWhitespace(in: text, index: &index)

            let parsed = parseValue(in: text, from: index)
            arguments[key] = parsed.value
            index = parsed.nextIndex
        }

        return arguments
    }

    private func parseValue(
        in text: String,
        from start: String.Index
    ) -> (value: String, nextIndex: String.Index) {
        if text[start...].hasPrefix(escapeMarker) {
            let valueStart = text.index(start, offsetBy: escapeMarker.count)
            guard let closingRange = text.range(of: escapeMarker, range: valueStart..<text.endIndex) else {
                return ("", text.endIndex)
            }
            let value = String(text[valueStart..<closingRange.lowerBound])
            return (value, advancePastComma(in: text, from: closingRange.upperBound))
        }

        let endIndex = findValueEnd(in: text, from: start) ?? text.endIndex
        let value = String(text[start..<endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        return (value, advancePastComma(in: text, from: endIndex))
    }

    private func findValueEnd(
        in text: String,
        from start: String.Index
    ) -> String.Index? {
        var index = start
        var objectDepth = 0
        var arrayDepth = 0
        var inJSONString = false
        var previousWasEscape = false

        while index < text.endIndex {
            let char = text[index]
            if char == "\"" && !previousWasEscape {
                inJSONString.toggle()
            } else if !inJSONString {
                switch char {
                case "{":
                    objectDepth += 1
                case "}":
                    if objectDepth == 0 && arrayDepth == 0 {
                        return index
                    }
                    objectDepth = max(0, objectDepth - 1)
                case "[":
                    arrayDepth += 1
                case "]":
                    arrayDepth = max(0, arrayDepth - 1)
                case ",":
                    if objectDepth == 0 && arrayDepth == 0 {
                        return index
                    }
                default:
                    break
                }
            }
            previousWasEscape = char == "\\" && !previousWasEscape
            if char != "\\" { previousWasEscape = false }
            index = text.index(after: index)
        }

        return text.endIndex
    }

    private func advancePastComma(
        in text: String,
        from start: String.Index
    ) -> String.Index {
        var index = start
        if index < text.endIndex, text[index] == "," {
            index = text.index(after: index)
        }
        skipWhitespace(in: text, index: &index)
        return index
    }

    private func skipDelimiters(in text: String, index: inout String.Index) {
        while index < text.endIndex {
            let char = text[index]
            if char == "," || char.isWhitespace {
                index = text.index(after: index)
            } else {
                return
            }
        }
    }

    private func skipWhitespace(in text: String, index: inout String.Index) {
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
    }
}
