// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma 4 function call format.
///
/// Gemma 4 uses a different tag scheme than Gemma 3:
/// - Start: `<|tool_call>`  End: `<tool_call|>`
/// - String escaping: `<|"|>` instead of `<escape>`
///
/// Format: `<|tool_call>call:name{key:<|"|>str_value<|"|>,k2:numeric}<tool_call|>`
///
/// Reference: https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
public struct Gemma4FunctionParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"

    private let escapeMarker = "<|\"|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }

        guard let callRange = text.range(of: "call:") else { return nil }
        let remaining = String(text[callRange.upperBound...])
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let funcName = String(remaining[..<braceStart])
        guard !funcName.isEmpty else { return nil }
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        let argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])

        let schemaTypes = schemaTypesByParameterName(for: funcName, tools: tools)
        let arguments = parseArguments(argsStr, schemaTypes: schemaTypes)
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    private func parseArguments(
        _ args: String,
        schemaTypes: [String: String]
    ) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]
        var index = args.startIndex

        while true {
            skipDelimiters(in: args, index: &index)
            guard index < args.endIndex else { break }

            guard let colonIndex = findTopLevelColon(in: args, from: index) else { break }
            let key = String(args[index..<colonIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            index = args.index(after: colonIndex)
            skipWhitespace(in: args, index: &index)

            guard !key.isEmpty else { break }
            let parsed = parseValue(in: args, from: index)
            arguments[key] = coerceValue(
                parsed.value,
                rawValue: parsed.rawValue,
                schemaType: schemaTypes[key]
            )
            index = parsed.nextIndex
        }

        return arguments
    }

    private func parseValue(
        in text: String,
        from start: String.Index
    ) -> (value: any Sendable, rawValue: String, nextIndex: String.Index) {
        if text[start...].hasPrefix(escapeMarker) {
            let valueStart = text.index(start, offsetBy: escapeMarker.count)
            guard let endRange = text.range(of: escapeMarker, range: valueStart..<text.endIndex) else {
                return ("", "", text.endIndex)
            }
            let raw = String(text[valueStart..<endRange.lowerBound])
            let next = advancePastComma(in: text, from: endRange.upperBound)
            return (raw, raw, next)
        }

        if let endIndex = findValueEnd(in: text, from: start) {
            let raw = String(text[start..<endIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            let next = advancePastComma(in: text, from: endIndex)
            if let json = decodeJSON(raw) {
                return (json, raw, next)
            }
            return (raw, raw, next)
        }

        let raw = String(text[start...]).trimmingCharacters(in: .whitespacesAndNewlines)
        if let json = decodeJSON(raw) {
            return (json, raw, text.endIndex)
        }
        return (raw, raw, text.endIndex)
    }

    private func coerceValue(
        _ value: any Sendable,
        rawValue: String,
        schemaType: String?
    ) -> any Sendable {
        guard let schemaType else { return value }
        if !(value is String) { return value }

        switch schemaType {
        case "integer":
            return Int(rawValue) ?? value
        case "number":
            if let number = Double(rawValue) {
                let integer = Int(number)
                return number == Double(integer) ? integer : number
            }
            return value
        case "boolean":
            switch rawValue.lowercased() {
            case "true": return true
            case "false": return false
            default: return value
            }
        case "array", "object":
            return decodeJSON(rawValue) ?? value
        default:
            return value
        }
    }

    private func schemaTypesByParameterName(
        for functionName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: String] {
        guard let tools else { return [:] }
        for tool in tools {
            guard let function = tool["function"] as? [String: any Sendable],
                  let name = function["name"] as? String,
                  name == functionName,
                  let parameters = function["parameters"] as? [String: any Sendable],
                  let properties = parameters["properties"] as? [String: any Sendable] else {
                continue
            }

            var result: [String: String] = [:]
            for (key, value) in properties {
                if let dict = value as? [String: any Sendable],
                   let type = dict["type"] as? String {
                    result[key] = type
                }
            }
            return result
        }
        return [:]
    }

    private func findTopLevelColon(
        in text: String,
        from start: String.Index
    ) -> String.Index? {
        var index = start
        var depth = 0
        var inString = false
        var previousWasEscape = false

        while index < text.endIndex {
            let char = text[index]
            if char == "\"" && !previousWasEscape {
                inString.toggle()
            } else if !inString {
                if char == "{" || char == "[" {
                    depth += 1
                } else if char == "}" || char == "]" {
                    depth = max(0, depth - 1)
                } else if char == ":" && depth == 0 {
                    return index
                }
            }
            previousWasEscape = char == "\\" && !previousWasEscape
            if char != "\\" { previousWasEscape = false }
            index = text.index(after: index)
        }
        return nil
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

    private func decodeJSON(_ raw: String) -> (any Sendable)? {
        guard let data = raw.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) else {
            return nil
        }
        return makeSendableJSON(json)
    }

    private func makeSendableJSON(_ value: Any) -> any Sendable {
        switch value {
        case let string as String:
            return string
        case let int as Int:
            return int
        case let double as Double:
            return double
        case let bool as Bool:
            return bool
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return number.boolValue
            }
            let doubleValue = number.doubleValue
            let integerValue = Int(doubleValue)
            return doubleValue == Double(integerValue) ? integerValue : doubleValue
        case let dict as [String: Any]:
            return dict.mapValues { makeSendableJSON($0) }
        case let array as [Any]:
            return array.map { makeSendableJSON($0) }
        case _ as NSNull:
            return NSNull()
        default:
            return String(describing: value)
        }
    }
}
