import Foundation
import MLXLMCommon

enum ToolCallStreamingEvent: Sendable {
    case started
    case delta(StreamDeltaToolCall)
    case appendCollected(ResponseToolCall)
    case replaceCollected(index: Int, toolCall: ResponseToolCall)
}

struct ToolCallStreamingOutput: Sendable {
    let handled: Bool
    let events: [ToolCallStreamingEvent]
}

final class ToolCallStreamingRuntime {
    private let toolCallStartTag: String
    private let toolCallEndTag: String
    private let toolCallParser: String?
    private let tools: [RequestTool]?
    private(set) var paramNameMapping: [String: String]
    private let applyFixToolArgs: @Sendable (ResponseToolCall) -> ResponseToolCall
    private let remapSingleKey: @Sendable (String, String) -> String

    private(set) var inToolCall = false
    private(set) var madeToolCall = false
    private(set) var hasToolCalls = false

    private var currentToolText = ""
    private var incrementalEmittedFirst = false
    private var incrementalCallId = ""
    private var incrementalFunctionName = ""
    private var incrementalToolIndex = 0
    private var incrementalParamCount = 0
    private var incrementalEmittedKeys = Set<String>()
    private var collectedCount = 0

    init(
        toolCallStartTag: String,
        toolCallEndTag: String,
        toolCallParser: String?,
        tools: [RequestTool]?,
        applyFixToolArgs: @escaping @Sendable (ResponseToolCall) -> ResponseToolCall,
        remapSingleKey: @escaping @Sendable (String, String) -> String
    ) {
        self.toolCallStartTag = toolCallStartTag
        self.toolCallEndTag = toolCallEndTag
        self.toolCallParser = toolCallParser
        self.tools = tools
        self.applyFixToolArgs = applyFixToolArgs
        self.remapSingleKey = remapSingleKey

        var mapping = [String: String]()
        if let tools {
            for tool in tools {
                if let paramsAny = tool.function.parameters?.toSendable() as? [String: Any],
                   let props = paramsAny["properties"] as? [String: Any] {
                    for key in props.keys {
                        let snaked = Self.toSnakeCase(key)
                        if snaked != key {
                            mapping[snaked] = key
                        }
                    }
                }
            }
        }
        self.paramNameMapping = mapping
    }

    func process(piece: String) -> ToolCallStreamingOutput {
        if !inToolCall, let startRange = piece.range(of: toolCallStartTag) {
            inToolCall = true
            madeToolCall = true
            let afterStart = String(piece[startRange.upperBound...])
            return self.consumeToolBodyFragment(afterStart, prependStarted: true)
        }

        guard inToolCall else {
            return ToolCallStreamingOutput(handled: false, events: [])
        }

        return consumeToolBodyFragment(piece, prependStarted: false)
    }

    func finishIncompleteToolCall() -> [ToolCallStreamingEvent] {
        guard inToolCall, !currentToolText.isEmpty else { return [] }

        defer { resetState() }

        if incrementalEmittedFirst {
            var events = [ToolCallStreamingEvent]()
            if let salvaged = salvageUnclosedParameterFragment() {
                events.append(.delta(salvaged))
            }

            let closeArgs = incrementalParamCount == 0 ? "{}" : "}"
            events.append(.delta(StreamDeltaToolCall(
                index: incrementalToolIndex,
                id: nil,
                type: nil,
                function: StreamDeltaFunction(name: nil, arguments: closeArgs)
            )))

            let parsed = parseIncrementalToolCalls(includeTrailingPartial: true)
            for tc in parsed {
                hasToolCalls = true
                let responseToolCall = normalizedToolCall(
                    from: tc,
                    index: incrementalToolIndex
                )
                events.append(.replaceCollected(index: incrementalToolIndex, toolCall: responseToolCall))
            }
            return events
        }

        return emitParsedToolCalls(from: currentToolText)
    }

    private func consumeToolBodyFragment(_ fragment: String, prependStarted: Bool) -> ToolCallStreamingOutput {
        var events = prependStarted ? [ToolCallStreamingEvent.started] : []
        currentToolText += fragment

        if let endRange = currentToolText.range(of: toolCallEndTag) {
            let beforeEnd = String(currentToolText[..<endRange.lowerBound])
            currentToolText = beforeEnd
            events.append(contentsOf: finalizeCurrentToolCall())
            return ToolCallStreamingOutput(handled: true, events: events)
        }

        events.append(contentsOf: scanIncrementalMarkers())
        return ToolCallStreamingOutput(handled: true, events: events)
    }

    private func finalizeCurrentToolCall() -> [ToolCallStreamingEvent] {
        defer { resetState() }

        if incrementalEmittedFirst {
            let parsed = parseIncrementalToolCalls(includeTrailingPartial: false)
            var events = [ToolCallStreamingEvent]()
            for tc in parsed {
                hasToolCalls = true
                let responseToolCall = normalizedToolCall(
                    from: tc,
                    index: incrementalToolIndex
                )
                events.append(.replaceCollected(index: incrementalToolIndex, toolCall: responseToolCall))
                events.append(.delta(StreamDeltaToolCall(
                    index: incrementalToolIndex,
                    id: nil,
                    type: nil,
                    function: StreamDeltaFunction(name: nil, arguments: responseToolCall.function.arguments)
                )))
            }
            return events
        }

        return emitParsedToolCalls(from: currentToolText)
    }

    private func parseIncrementalToolCalls(includeTrailingPartial: Bool) -> [ToolCall] {
        let wrapped = "\(toolCallStartTag)\(currentToolText)\(toolCallEndTag)"
        let (parsed, _) = MLXModelService.extractToolCallsFallback(from: wrapped, tools: tools)
        if !parsed.isEmpty {
            return parsed
        }
        guard let fallback = buildIncrementalToolCall(includeTrailingPartial: includeTrailingPartial) else {
            return []
        }
        return [fallback]
    }

    private func emitParsedToolCalls(from body: String) -> [ToolCallStreamingEvent] {
        let parsed = parseToolCalls(from: body)
        var events = [ToolCallStreamingEvent]()
        for tc in parsed {
            hasToolCalls = true
            let responseToolCall = normalizedToolCall(from: tc, index: collectedCount)
            collectedCount += 1
            events.append(.appendCollected(responseToolCall))
            events.append(.delta(StreamDeltaToolCall(
                index: collectedCount - 1,
                id: responseToolCall.id,
                type: responseToolCall.type,
                function: StreamDeltaFunction(
                    name: responseToolCall.function.name,
                    arguments: responseToolCall.function.arguments
                )
            )))
        }
        return events
    }

    private func parseToolCalls(from body: String) -> [ToolCall] {
        let wrapped = "\(toolCallStartTag)\(body)\(toolCallEndTag)"
        let (parsed, _) = Self.parseCompletedToolCalls(
            from: wrapped,
            toolCallParser: toolCallParser,
            tools: tools
        )
        return parsed
    }

    private func normalizedToolCall(from toolCall: ToolCall, index: Int) -> ResponseToolCall {
        // FIX: Strip XML tag remnants from tool name (e.g. "todoread</function")
        // See: opencode promptfoo test #20/#33 — zero-parameter XML tool call bug
        var cleanedToolCall = toolCall
        if let tagIdx = cleanedToolCall.function.name.range(of: "</") {
            let cleanName = String(cleanedToolCall.function.name[..<tagIdx.lowerBound])
            cleanedToolCall = ToolCall(function: .init(name: cleanName, arguments: cleanedToolCall.function.arguments))
        }
        let responseToolCall = MLXModelService.convertToolCall(
            cleanedToolCall,
            index: index,
            paramNameMapping: paramNameMapping,
            tools: tools
        )
        let fixedToolCall = applyFixToolArgs(responseToolCall)
        return MLXModelService.coerceArgumentTypes(fixedToolCall, tools: tools)
    }

    private func scanIncrementalMarkers() -> [ToolCallStreamingEvent] {
        var events = [ToolCallStreamingEvent]()

        if !incrementalEmittedFirst,
           let funcRange = currentToolText.range(of: #"<function=([^>]+)>"#, options: .regularExpression) {
            let match = String(currentToolText[funcRange])
            if let equalsRange = match.range(of: "="),
               let closeRange = match.range(of: ">", options: .backwards) {
                let functionName = String(match[equalsRange.upperBound..<closeRange.lowerBound])
                if !functionName.contains("\""), !functionName.contains("{") {
                    incrementalCallId = "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
                    incrementalFunctionName = functionName
                    incrementalToolIndex = collectedCount
                    let placeholder = ResponseToolCall(
                        index: incrementalToolIndex,
                        id: incrementalCallId,
                        type: "function",
                        function: ResponseToolCallFunction(name: functionName, arguments: "")
                    )
                    collectedCount += 1
                    hasToolCalls = true
                    incrementalEmittedFirst = true
                    events.append(.appendCollected(placeholder))
                    events.append(.delta(StreamDeltaToolCall(
                        index: incrementalToolIndex,
                        id: incrementalCallId,
                        type: "function",
                        function: StreamDeltaFunction(name: functionName, arguments: "")
                    )))
                }
            }
        }

        if incrementalEmittedFirst {
            let pattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
                let nsText = currentToolText as NSString
                let matches = regex.matches(in: currentToolText, range: NSRange(location: 0, length: nsText.length))
                for match in matches {
                    guard match.numberOfRanges >= 3,
                          let keyRange = Range(match.range(at: 1), in: currentToolText) else {
                        continue
                    }
                    incrementalEmittedKeys.insert(String(currentToolText[keyRange]))
                }
            }
        }

        return events
    }

    private func salvageUnclosedParameterFragment() -> StreamDeltaToolCall? {
        let pattern = #"<parameter=([^>]+)>([\s\S]+)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: currentToolText, range: NSRange(currentToolText.startIndex..., in: currentToolText)),
              let keyRange = Range(match.range(at: 1), in: currentToolText),
              let valueRange = Range(match.range(at: 2), in: currentToolText) else {
            return nil
        }

        let rawKey = String(currentToolText[keyRange])
        guard !incrementalEmittedKeys.contains(rawKey) else { return nil }

        var value = String(currentToolText[valueRange])
        if value.hasPrefix("\n") { value = String(value.dropFirst()) }
        if value.hasSuffix("\n") { value = String(value.dropLast()) }
        value = MLXModelService.decodeJSONEscapes(MLXModelService.decodeXMLEntities(value))
        guard !value.isEmpty else { return nil }

        incrementalEmittedKeys.insert(rawKey)
        var emittedKey = paramNameMapping[rawKey] ?? rawKey
        if emittedKey == rawKey {
            emittedKey = remapSingleKey(rawKey, incrementalFunctionName)
        }

        let jsonValue = Self.jsonEncodeString(value)
        let fragment: String
        if incrementalParamCount == 0 {
            fragment = "{\"\(Self.jsonEscapeKey(emittedKey))\":\(jsonValue)"
        } else {
            fragment = ",\"\(Self.jsonEscapeKey(emittedKey))\":\(jsonValue)"
        }
        incrementalParamCount += 1

        return StreamDeltaToolCall(
            index: incrementalToolIndex,
            id: nil,
            type: nil,
            function: StreamDeltaFunction(name: nil, arguments: fragment)
        )
    }

    private func resolveToolName(_ name: String) -> String {
        let validNames = tools?.map(\.function.name) ?? []
        guard !validNames.isEmpty, !validNames.contains(name) else { return name }
        return Self.fuzzyMatchToolName(name, candidates: validNames) ?? name
    }

    private func buildIncrementalToolCall(includeTrailingPartial: Bool) -> ToolCall? {
        guard !incrementalFunctionName.isEmpty else { return nil }

        var arguments = [String: any Sendable]()
        let pattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
        if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) {
            let nsText = currentToolText as NSString
            let matches = regex.matches(in: currentToolText, range: NSRange(location: 0, length: nsText.length))
            for match in matches {
                guard match.numberOfRanges >= 3,
                      let keyRange = Range(match.range(at: 1), in: currentToolText),
                      let valueRange = Range(match.range(at: 2), in: currentToolText) else {
                    continue
                }
                let key = String(currentToolText[keyRange])
                if arguments[key] == nil {
                    arguments[key] = Self.decodeParameterValue(String(currentToolText[valueRange]))
                }
            }
        }

        if includeTrailingPartial,
           let partial = trailingPartialParameter(),
           arguments[partial.key] == nil {
            arguments[partial.key] = Self.decodeParameterValue(partial.value)
        }

        return ToolCall(function: .init(
            name: resolveToolName(incrementalFunctionName),
            arguments: arguments
        ))
    }

    private func trailingPartialParameter() -> (key: String, value: String)? {
        let pattern = #"<parameter=([^>]+)>([\s\S]+)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: currentToolText, range: NSRange(currentToolText.startIndex..., in: currentToolText)),
              let keyRange = Range(match.range(at: 1), in: currentToolText),
              let valueRange = Range(match.range(at: 2), in: currentToolText) else {
            return nil
        }

        var value = String(currentToolText[valueRange])
        if value.hasPrefix("\n") { value = String(value.dropFirst()) }
        if value.hasSuffix("\n") { value = String(value.dropLast()) }
        guard !value.isEmpty else { return nil }
        return (String(currentToolText[keyRange]), value)
    }

    private func resetState() {
        currentToolText = ""
        inToolCall = false
        incrementalEmittedFirst = false
        incrementalCallId = ""
        incrementalFunctionName = ""
        incrementalParamCount = 0
        incrementalEmittedKeys = Set<String>()
    }

    private static func fuzzyMatchToolName(_ name: String, candidates: [String]) -> String? {
        var bestMatch: String?
        var bestDistance = Int.max
        for candidate in candidates {
            let distance = editDistance(name.lowercased(), candidate.lowercased())
            if distance < bestDistance {
                bestDistance = distance
                bestMatch = candidate
            }
        }
        return bestDistance <= 3 ? bestMatch : nil
    }

    private static func editDistance(_ a: String, _ b: String) -> Int {
        let lhs = Array(a)
        let rhs = Array(b)
        if lhs.isEmpty { return rhs.count }
        if rhs.isEmpty { return lhs.count }

        var previous = Array(0...rhs.count)
        var current = [Int](repeating: 0, count: rhs.count + 1)
        for i in 1...lhs.count {
            current[0] = i
            for j in 1...rhs.count {
                current[j] = lhs[i - 1] == rhs[j - 1]
                    ? previous[j - 1]
                    : 1 + Swift.min(previous[j], current[j - 1], previous[j - 1])
            }
            previous = current
        }
        return previous[rhs.count]
    }

    private static func jsonEncodeString(_ value: String) -> String {
        if let data = try? JSONSerialization.data(withJSONObject: [value], options: []),
           let encoded = String(data: data, encoding: .utf8),
           encoded.count >= 2 {
            return String(encoded.dropFirst().dropLast())
        }
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escaped)\""
    }

    private static func jsonEscapeKey(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    static func parseCompletedToolCalls(
        from text: String,
        toolCallParser: String?,
        tools: [RequestTool]?
    ) -> ([ToolCall], String) {
        if toolCallParser == "afm_adaptive_xml",
           let direct = parseSingleAdaptiveJSONToolCall(from: text, tools: tools) {
            return direct
        }
        let (parsed, remaining) = MLXModelService.extractToolCallsFallback(from: text, tools: tools)
        guard !parsed.isEmpty else { return (parsed, remaining) }
        return (normalizeParsedToolCalls(parsed, toolCallParser: toolCallParser, tools: tools), remaining)
    }

    private static func normalizeParsedToolCalls(
        _ toolCalls: [ToolCall],
        toolCallParser: String?,
        tools: [RequestTool]?
    ) -> [ToolCall] {
        guard toolCallParser == "afm_adaptive_xml" else { return toolCalls }
        let validNames = tools?.map(\.function.name) ?? []
        guard !validNames.isEmpty else { return toolCalls }
        return toolCalls.map { toolCall in
            // FIX: Zero-parameter XML tool calls (e.g. <function=todoread></function>)
            // can have "</function" appended to the name when the streaming XML parser
            // captures past the ">" boundary. Strip any XML tag remnants from the name.
            // See: opencode promptfoo test #20/#33 — todoread</function bug.
            var cleanName = toolCall.function.name
            if let tagIdx = cleanName.range(of: "</") {
                cleanName = String(cleanName[cleanName.startIndex..<tagIdx.lowerBound])
            }
            let resolvedName: String
            if validNames.contains(cleanName) {
                resolvedName = cleanName
            } else {
                resolvedName = fuzzyMatchToolName(cleanName, candidates: validNames) ?? cleanName
            }
            // Use .anyValue to convert JSONValue → plain types before re-init,
            // otherwise JSONValue.from() double-wraps via String(describing:).
            let plainArgs: [String: any Sendable] = toolCall.function.arguments.mapValues { $0.anyValue }
            return ToolCall(function: .init(name: resolvedName, arguments: plainArgs))
        }
    }

    private static func parseSingleAdaptiveJSONToolCall(
        from text: String,
        tools: [RequestTool]?
    ) -> ([ToolCall], String)? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let regex = try? NSRegularExpression(
            pattern: #"<tool_call>\s*(.*?)\s*</tool_call>"#,
            options: [.dotMatchesLineSeparators]
        ),
        let match = regex.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)),
        match.range.location == 0,
        match.range.length == (trimmed as NSString).length,
        let innerRange = Range(match.range(at: 1), in: trimmed),
        let toolCall = parseAdaptiveJSONToolCallBody(String(trimmed[innerRange]), tools: tools) else {
            return nil
        }
        return ([toolCall], "")
    }

    private static func parseAdaptiveJSONToolCallBody(_ body: String, tools: [RequestTool]?) -> ToolCall? {
        let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedBody.hasPrefix("{"),
              let data = trimmedBody.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = json["name"] as? String else {
            return nil
        }
        var arguments: [String: any Sendable] = [:]
        if let args = (json["arguments"] as? [String: Any]) ?? (json["parameters"] as? [String: Any]) {
            for (key, value) in args {
                arguments[key] = makeSendableJSON(value)
            }
        }
        let validNames = tools?.map(\.function.name) ?? []
        let resolvedName: String
        if validNames.isEmpty || validNames.contains(name) {
            resolvedName = name
        } else {
            resolvedName = fuzzyMatchToolName(name, candidates: validNames) ?? name
        }
        return ToolCall(function: .init(name: resolvedName, arguments: arguments))
    }

    private static func toSnakeCase(_ value: String) -> String {
        guard !value.isEmpty else { return value }
        var output = ""
        for scalar in value.unicodeScalars {
            let character = Character(scalar)
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if !output.isEmpty { output.append("_") }
                output.append(character.lowercased())
            } else {
                output.append(character)
            }
        }
        return output
    }

    private static func makeSendableJSON(_ value: Any) -> any Sendable {
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
            if floor(doubleValue) == doubleValue {
                return number.intValue
            }
            return doubleValue
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

    private static func decodeParameterValue(_ value: String) -> any Sendable {
        let decoded = MLXModelService.decodeJSONEscapes(MLXModelService.decodeXMLEntities(value))
        let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        if (trimmed.hasPrefix("{") || trimmed.hasPrefix("[")),
           let data = trimmed.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) {
            return makeSendableJSON(json)
        }
        return decoded
    }
}
