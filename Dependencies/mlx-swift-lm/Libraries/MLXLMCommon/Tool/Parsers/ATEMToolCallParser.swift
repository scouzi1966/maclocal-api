// Copyright © 2026 Soprano Technologies Inc.

import Foundation

/// Parser for the Onyx ATEM format used by Muse Glimmer.
///
/// The model's chat template defines ATEM as a regex-parsed protocol rather
/// than XML, so values may contain unescaped text and span multiple lines.
public struct ATEMToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<atem:function_calls>"
    public let endTag: String? = "</atem:function_calls>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content
        if let startTag, let range = text.range(of: startTag) {
            text = String(text[range.upperBound...])
        }
        if let endTag, let range = text.range(of: endTag) {
            text = String(text[..<range.lowerBound])
        }

        guard let invoke = text.range(
            of: #"<atem:invoke\s+name\s*=\s*[\"'][^\"']+[\"']\s*>"#,
            options: .regularExpression
        ) else { return nil }

        let invokeTag = String(text[invoke])
        guard let functionName = attribute(named: "name", in: invokeTag),
              !functionName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let invokeEnd = text.range(
                of: "</atem:invoke>",
                range: invoke.upperBound..<text.endIndex
              )
        else { return nil }

        let body = String(text[invoke.upperBound..<invokeEnd.lowerBound])
        let parameterConfig = getParameterConfig(funcName: functionName, tools: tools)
        var arguments: [String: any Sendable] = [:]
        var cursor = body.startIndex

        while cursor < body.endIndex,
              let start = body.range(
                of: #"<atem:parameter\s+name\s*=\s*[\"'][^\"']+[\"']\s*>"#,
                options: .regularExpression,
                range: cursor..<body.endIndex
              ) {
            let startTag = String(body[start])
            guard let parameterName = attribute(named: "name", in: startTag),
                  !parameterName.isEmpty,
                  let end = body.range(
                    of: "</atem:parameter>",
                    range: start.upperBound..<body.endIndex
                  )
            else { return nil }

            // ATEM explicitly preserves spaces in string values. Only remove a
            // single formatting newline added around a parameter value.
            var value = String(body[start.upperBound..<end.lowerBound])
            if value.hasPrefix("\n") { value.removeFirst() }
            if value.hasSuffix("\n") { value.removeLast() }

            let schema = parameterConfig[parameterName] as? [String: any Sendable]
            arguments[parameterName] = convertValueWithTypes(
                value,
                types: extractTypesFromSchema(schema)
            )
            cursor = end.upperBound
        }

        return ToolCall(function: .init(name: functionName, arguments: arguments))
    }

    private func attribute(named name: String, in tag: String) -> String? {
        guard let match = tag.range(
            of: #"\b"# + NSRegularExpression.escapedPattern(for: name)
                + #"\s*=\s*([\"'])[^\"']*\1"#,
            options: .regularExpression
        ) else { return nil }

        let assignment = String(tag[match])
        guard let equals = assignment.firstIndex(of: "=") else { return nil }
        let quoted = assignment[assignment.index(after: equals)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard quoted.count >= 2, let quote = quoted.first, quoted.last == quote else { return nil }
        return String(quoted.dropFirst().dropLast())
    }
}
