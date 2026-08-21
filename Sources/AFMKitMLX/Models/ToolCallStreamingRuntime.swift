import Foundation
import AFMOpenAICompat
import MLXLMCommon

public enum ToolCallStreamingEvent: Sendable {
    case started
    case delta(StreamDeltaToolCall)
    case appendCollected(ResponseToolCall)
    case replaceCollected(index: Int, toolCall: ResponseToolCall)
}

public struct ToolCallStreamingOutput: Sendable {
    public let handled: Bool
    public let events: [ToolCallStreamingEvent]
    public let passthroughText: String?

    public init(
        handled: Bool,
        events: [ToolCallStreamingEvent],
        passthroughText: String? = nil
    ) {
        self.handled = handled
        self.events = events
        self.passthroughText = passthroughText
    }
}

public final class ToolCallStreamingRuntime {
    private let toolCallStartTag: String
    private let toolCallEndTag: String
    private let toolCallParser: String?
    private let tools: [RequestTool]?
    public private(set) var paramNameMapping: [String: String]
    private let applyFixToolArgs: @Sendable (ResponseToolCall) -> ResponseToolCall
    private let remapSingleKey: @Sendable (String, String) -> String

    public private(set) var inToolCall = false
    public private(set) var madeToolCall = false
    public private(set) var hasToolCalls = false

    private var currentToolText = ""
    private var incrementalEmittedFirst = false
    private var incrementalCallId = ""
    private var incrementalFunctionName = ""
    private var incrementalToolIndex = 0
    private var incrementalParamCount = 0
    private var incrementalArgumentPrefix = ""
    private var incrementalEmittedKeys = Set<String>()
    private var collectedCount = 0
    private var pendingStartProbe = ""
    private var finalizedCurrentToolCall = false

    public init(
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

    public func process(piece: String) -> ToolCallStreamingOutput {
        if !inToolCall {
            let candidate = pendingStartProbe + piece
            if let startRange = candidate.range(of: toolCallStartTag) {
                let prefix = String(candidate[..<startRange.lowerBound])
                pendingStartProbe = ""
                inToolCall = true
                madeToolCall = true
                let afterStart = String(candidate[startRange.upperBound...])
                let output = self.consumeToolBodyFragment(afterStart, prependStarted: true)
                guard !prefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return output
                }
                return ToolCallStreamingOutput(
                    handled: true,
                    events: output.events,
                    passthroughText: prefix
                )
            }

            let retained = Self.partialSuffixLength(in: candidate, matching: toolCallStartTag)
            if retained > 0 {
                let emitEnd = candidate.index(candidate.endIndex, offsetBy: -retained)
                let emit = String(candidate[..<emitEnd])
                pendingStartProbe = String(candidate[emitEnd...])
                return ToolCallStreamingOutput(
                    handled: true,
                    events: [],
                    passthroughText: emit.isEmpty ? nil : emit
                )
            }

            if !pendingStartProbe.isEmpty {
                pendingStartProbe = ""
                return ToolCallStreamingOutput(
                    handled: true,
                    events: [],
                    passthroughText: candidate.isEmpty ? nil : candidate
                )
            }
        }

        guard inToolCall else {
            return ToolCallStreamingOutput(handled: false, events: [])
        }

        return consumeToolBodyFragment(piece, prependStarted: false)
    }

    public func finishIncompleteToolCall() -> [ToolCallStreamingEvent] {
        guard inToolCall, !currentToolText.isEmpty, !finalizedCurrentToolCall else { return [] }

        defer { resetState() }

        if incrementalEmittedFirst {
            var events = [ToolCallStreamingEvent]()
            if let salvaged = salvageUnclosedParameterFragment() {
                events.append(.delta(salvaged))
            }

            let closeArgs = incrementalParamCount == 0 ? "{}" : (needsIncrementalArgumentClose ? "}" : nil)
            if let closeArgs {
                events.append(.delta(StreamDeltaToolCall(
                    index: incrementalToolIndex,
                    id: nil,
                    type: nil,
                    function: StreamDeltaFunction(name: nil, arguments: closeArgs)
                )))
            }

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
            let afterEnd = String(currentToolText[endRange.upperBound...])
            currentToolText = beforeEnd
            finalizedCurrentToolCall = true
            events.append(contentsOf: finalizeCurrentToolCall())

            guard !afterEnd.isEmpty else {
                return ToolCallStreamingOutput(handled: true, events: events)
            }

            let trailing = process(piece: afterEnd)
            events.append(contentsOf: trailing.events)
            return ToolCallStreamingOutput(
                handled: true,
                events: events,
                passthroughText: trailing.handled ? trailing.passthroughText : afterEnd
            )
        }

        events.append(contentsOf: scanIncrementalMarkers())
        return ToolCallStreamingOutput(handled: true, events: events)
    }

    private func finalizeCurrentToolCall() -> [ToolCallStreamingEvent] {
        defer { resetState() }

        if incrementalEmittedFirst {
            let parsed = parseIncrementalToolCalls(includeTrailingPartial: false)
            var events = [ToolCallStreamingEvent]()
            let closeArguments = incrementalParamCount == 0
                ? "{}"
                : (needsIncrementalArgumentClose ? "}" : nil)
            if let closeArguments {
                events.append(.delta(StreamDeltaToolCall(
                    index: incrementalToolIndex,
                    id: nil,
                    type: nil,
                    function: StreamDeltaFunction(name: nil, arguments: closeArguments)
                )))
            }
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

    private func parseIncrementalToolCalls(includeTrailingPartial: Bool) -> [ToolCall] {
        // At end-of-stream, the incremental representation is authoritative: it
        // includes both closed parameters and the salvaged trailing parameter.
        // The general fallback parser only sees closed XML elements and would
        // otherwise produce a replacement snapshot that drops the salvaged value.
        if includeTrailingPartial,
           let fallback = buildIncrementalToolCall(includeTrailingPartial: true) {
            return [fallback]
        }

        let wrapped = "\(toolCallStartTag)\(currentToolText)\(toolCallEndTag)"
        let (parsed, _) = MLXModelService.extractToolCallsFallback(
            from: wrapped, tools: tools,
            allowMalformedRepair: toolCallParser == "afm_adaptive_xml")
        if !parsed.isEmpty {
            return parsed
        }
        guard let fallback = buildIncrementalToolCall(includeTrailingPartial: false) else {
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
        let coerced = MLXModelService.coerceArgumentTypes(fixedToolCall, tools: tools)
        guard coerced.function.arguments
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return coerced
        }
        return ResponseToolCall(
            index: coerced.index,
            id: coerced.id,
            type: coerced.type,
            function: ResponseToolCallFunction(
                name: coerced.function.name,
                arguments: "{}"
            )
        )
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
                    let rawKey = String(currentToolText[keyRange])
                    guard !incrementalEmittedKeys.contains(rawKey),
                          let valueRange = Range(match.range(at: 2), in: currentToolText) else {
                        continue
                    }
                    if let delta = buildParameterDelta(
                        rawKey: rawKey,
                        rawValue: String(currentToolText[valueRange])
                    ) {
                        events.append(.delta(delta))
                    }
                }
            }
        }

        return events
    }

    private func buildParameterDelta(rawKey: String, rawValue: String) -> StreamDeltaToolCall? {
        var emittedKey = paramNameMapping[rawKey] ?? rawKey
        if emittedKey == rawKey {
            emittedKey = remapSingleKey(rawKey, incrementalFunctionName)
        }

        let decodedValue = Self.decodeParameterValue(Self.normalizeParameterBody(rawValue))
        let jsonValue = Self.jsonEncodeValue(coerceIncrementalParameterValue(decodedValue, key: emittedKey))
        let fragment: String
        if incrementalParamCount == 0 {
            fragment = "{\"\(Self.jsonEscapeKey(emittedKey))\":\(jsonValue)"
        } else {
            fragment = ",\"\(Self.jsonEscapeKey(emittedKey))\":\(jsonValue)"
        }
        incrementalParamCount += 1
        incrementalArgumentPrefix += fragment
        incrementalEmittedKeys.insert(rawKey)

        return StreamDeltaToolCall(
            index: incrementalToolIndex,
            id: nil,
            type: nil,
            function: StreamDeltaFunction(name: nil, arguments: fragment)
        )
    }

    private func coerceIncrementalParameterValue(
        _ value: any Sendable,
        key: String
    ) -> any Sendable {
        guard let stringValue = value as? String,
              let tool = tools?.first(where: { $0.function.name == incrementalFunctionName }),
              let parameters = tool.function.parameters?.toSendable() as? [String: Any],
              let properties = parameters["properties"] as? [String: Any],
              let schema = properties[key] as? [String: Any],
              let schemaType = schema["type"] as? String else {
            return value
        }

        switch schemaType {
        case "integer":
            return Int(stringValue) ?? value
        case "number":
            guard let number = Double(stringValue) else { return value }
            let integer = Int(number)
            return number == Double(integer) ? integer : number
        case "boolean":
            switch stringValue.lowercased() {
            case "true": return true
            case "false": return false
            default: return value
            }
        default:
            return value
        }
    }

    private func salvageUnclosedParameterFragment() -> StreamDeltaToolCall? {
        guard let partial = trailingPartialParameter() else { return nil }
        let rawKey = partial.key
        guard !incrementalEmittedKeys.contains(rawKey) else { return nil }
        return buildParameterDelta(rawKey: rawKey, rawValue: partial.value)
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
                    arguments[key] = Self.decodeParameterValue(Self.normalizeParameterBody(String(currentToolText[valueRange])))
                }
            }
        }

        if includeTrailingPartial,
           let partial = trailingPartialParameter(),
           arguments[partial.key] == nil {
            arguments[partial.key] = Self.decodeParameterValue(Self.normalizeParameterBody(partial.value))
        }

        return ToolCall(function: .init(
            name: resolveToolName(incrementalFunctionName),
            arguments: arguments
        ))
    }

    private func trailingPartialParameter() -> (key: String, value: String)? {
        guard let openRange = currentToolText.range(of: "<parameter=", options: .backwards) else {
            return nil
        }
        if let closedRange = currentToolText.range(of: "</parameter>", options: .backwards),
           closedRange.lowerBound > openRange.lowerBound {
            return nil
        }

        let fragment = String(currentToolText[openRange.lowerBound...])
        let pattern = #"^\s*<parameter=([^>]+)>([\s\S]+)$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: fragment, range: NSRange(fragment.startIndex..., in: fragment)),
              let keyRange = Range(match.range(at: 1), in: fragment),
              let valueRange = Range(match.range(at: 2), in: fragment) else {
            return nil
        }

        let key = String(fragment[keyRange])
        var value = String(fragment[valueRange])
        if value.hasPrefix("\n") { value = String(value.dropFirst()) }
        if value.hasSuffix("\n") { value = String(value.dropLast()) }
        guard !value.isEmpty else { return nil }
        return (key, value)
    }

    private func resetState() {
        currentToolText = ""
        inToolCall = false
        pendingStartProbe = ""
        incrementalEmittedFirst = false
        incrementalCallId = ""
        incrementalFunctionName = ""
        incrementalParamCount = 0
        incrementalArgumentPrefix = ""
        incrementalEmittedKeys = Set<String>()
        finalizedCurrentToolCall = false
    }

    private var needsIncrementalArgumentClose: Bool {
        guard incrementalParamCount > 0 else { return false }
        return Self.jsonBraceBalance(in: incrementalArgumentPrefix) > 0
    }

    private static func jsonBraceBalance(in text: String) -> Int {
        var balance = 0
        var inString = false
        var escaped = false
        for char in text {
            if escaped {
                escaped = false
                continue
            }
            if char == "\\" {
                escaped = inString
                continue
            }
            if char == "\"" {
                inString.toggle()
                continue
            }
            guard !inString else { continue }
            if char == "{" {
                balance += 1
            } else if char == "}" {
                balance -= 1
            }
        }
        return balance
    }

    private static func partialSuffixLength(in text: String, matching boundary: String) -> Int {
        let maximum = min(text.count, max(0, boundary.count - 1))
        guard maximum > 0 else { return 0 }
        for length in stride(from: maximum, through: 1, by: -1) {
            if text.suffix(length) == boundary.prefix(length) {
                return length
            }
        }
        return 0
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

    private static func jsonEncodeValue(_ value: any Sendable) -> String {
        switch value {
        case let string as String:
            return jsonEncodeString(string)
        case let bool as Bool:
            return bool ? "true" : "false"
        case let int as Int:
            return String(int)
        case let double as Double:
            guard double.isFinite else { return "null" }
            return String(double)
        case _ as NSNull:
            return "null"
        case let dict as [String: any Sendable]:
            if let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]),
               let encoded = String(data: data, encoding: .utf8) {
                return encoded
            }
        case let array as [any Sendable]:
            if let data = try? JSONSerialization.data(withJSONObject: array, options: [.sortedKeys]),
               let encoded = String(data: data, encoding: .utf8) {
                return encoded
            }
        default:
            break
        }
        return jsonEncodeString(String(describing: value))
    }

    private static func normalizeParameterBody(_ value: String) -> String {
        let decoded = MLXModelService.decodeXMLEntities(value)
        let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") || trimmed.hasPrefix("[") {
            return trimmed
        }
        return trimmed.isEmpty ? decoded : trimmed
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

    public static func parseCompletedToolCalls(
        from text: String,
        toolCallParser: String?,
        tools: [RequestTool]?
    ) -> ([ToolCall], String) {
        if let atem = parseATEMToolCalls(from: text, tools: tools) {
            return atem
        }
        if let dsml = parseDeepseekDSMLToolCalls(from: text, tools: tools) {
            return dsml
        }
        if toolCallParser == "afm_adaptive_xml",
           let direct = parseSingleAdaptiveJSONToolCall(from: text, tools: tools) {
            return direct
        }
        let (parsed, remaining) = MLXModelService.extractToolCallsFallback(
            from: text, tools: tools,
            allowMalformedRepair: toolCallParser == "afm_adaptive_xml")
        guard !parsed.isEmpty else { return (parsed, remaining) }
        return (normalizeParsedToolCalls(parsed, toolCallParser: toolCallParser, tools: tools), remaining)
    }

    /// Parse Muse Glimmer's ATEM envelope. Detection is syntax-based so the
    /// runtime continues to work for converted or renamed checkpoints.
    private static func parseATEMToolCalls(
        from text: String,
        tools: [RequestTool]?
    ) -> ([ToolCall], String)? {
        guard let envelopeRegex = try? NSRegularExpression(
            pattern: #"<atem:function_calls>\s*(.*?)\s*</atem:function_calls>"#,
            options: [.dotMatchesLineSeparators, .caseInsensitive]
        ),
        let invokeRegex = try? NSRegularExpression(
            pattern: #"<atem:invoke\s+name\s*=\s*[\"'][^\"']+[\"']\s*>.*?</atem:invoke>"#,
            options: [.dotMatchesLineSeparators, .caseInsensitive]
        ) else { return nil }

        let source = text as NSString
        let envelopes = envelopeRegex.matches(
            in: text,
            range: NSRange(location: 0, length: source.length)
        )
        guard !envelopes.isEmpty else { return nil }

        let toolSpecs: [[String: any Sendable]]? = tools?.map { tool in
            var function: [String: any Sendable] = ["name": tool.function.name]
            if let parameters = tool.function.parameters {
                function["parameters"] = parameters.toSendable()
            }
            return ["type": tool.type, "function": function]
        }

        let parser = ATEMToolCallParser()
        var calls: [ToolCall] = []
        for envelope in envelopes {
            guard let bodyRange = Range(envelope.range(at: 1), in: text) else { continue }
            let body = String(text[bodyRange])
            let bodySource = body as NSString
            for invoke in invokeRegex.matches(
                in: body,
                range: NSRange(location: 0, length: bodySource.length)
            ) {
                let invokeText = bodySource.substring(with: invoke.range)
                if let call = parser.parse(content: invokeText, tools: toolSpecs) {
                    calls.append(call)
                }
            }
        }
        guard !calls.isEmpty else { return nil }

        let remaining = NSMutableString(string: text)
        for envelope in envelopes.reversed() {
            remaining.replaceCharacters(in: envelope.range, with: "")
        }
        return (calls, remaining.trimmingCharacters(in: .whitespacesAndNewlines))
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
            let plainArgs: [String: any Sendable] = toolCall.function.arguments.mapValues { MLXModelService.asSendableJSON($0.anyValue) }
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

    /// Parse DeepSeek's native DSML envelope. Detection is syntax-based so a
    /// converted checkpoint does not depend on its repository or model name.
    private static func parseDeepseekDSMLToolCalls(
        from text: String,
        tools: [RequestTool]?
    ) -> ([ToolCall], String)? {
        let envelopePattern = #"<[|｜]DSML[|｜]tool_calls>\s*(.*?)\s*</[|｜]DSML[|｜]tool_calls>"#
        guard let envelopeRegex = try? NSRegularExpression(
            pattern: envelopePattern,
            options: [.dotMatchesLineSeparators]
        ) else { return nil }

        let source = text as NSString
        let envelopeMatches = envelopeRegex.matches(
            in: text,
            range: NSRange(location: 0, length: source.length)
        )
        guard !envelopeMatches.isEmpty else { return nil }

        let invokeRegex = try? NSRegularExpression(
            pattern: #"<[|｜]DSML[|｜]invoke\s+name=\"([^\"]+)\">\s*(.*?)\s*</[|｜]DSML[|｜]invoke>"#,
            options: [.dotMatchesLineSeparators]
        )
        let parameterRegex = try? NSRegularExpression(
            pattern: #"<[|｜]DSML[|｜]parameter\s+name=\"([^\"]+)\"(?:\s+string=\"(true|false)\")?>(.*?)</[|｜]DSML[|｜]parameter>"#,
            options: [.dotMatchesLineSeparators, .caseInsensitive]
        )
        guard let invokeRegex, let parameterRegex else { return nil }

        let validNames = tools?.map(\.function.name) ?? []
        var calls: [ToolCall] = []
        for envelope in envelopeMatches {
            guard let bodyRange = Range(envelope.range(at: 1), in: text) else { continue }
            let body = String(text[bodyRange])
            let bodyNSString = body as NSString
            for invoke in invokeRegex.matches(
                in: body,
                range: NSRange(location: 0, length: bodyNSString.length)
            ) {
                guard invoke.numberOfRanges >= 3 else { continue }
                let rawName = bodyNSString.substring(with: invoke.range(at: 1))
                let invokeBody = bodyNSString.substring(with: invoke.range(at: 2))
                let invokeNSString = invokeBody as NSString
                var arguments: [String: any Sendable] = [:]
                for parameter in parameterRegex.matches(
                    in: invokeBody,
                    range: NSRange(location: 0, length: invokeNSString.length)
                ) {
                    let name = MLXModelService.decodeXMLEntities(
                        invokeNSString.substring(with: parameter.range(at: 1))
                    )
                    let forceString = parameter.range(at: 2).location != NSNotFound
                        && invokeNSString.substring(with: parameter.range(at: 2)).lowercased() == "true"
                    let value = invokeNSString.substring(with: parameter.range(at: 3))
                    arguments[name] = decodeDSMLParameterValue(value, forceString: forceString)
                }
                let resolvedName: String
                if validNames.isEmpty || validNames.contains(rawName) {
                    resolvedName = rawName
                } else {
                    resolvedName = fuzzyMatchToolName(rawName, candidates: validNames) ?? rawName
                }
                calls.append(ToolCall(function: .init(name: resolvedName, arguments: arguments)))
            }
        }
        guard !calls.isEmpty else { return nil }

        let mutableRemaining = NSMutableString(string: text)
        for match in envelopeMatches.reversed() {
            mutableRemaining.replaceCharacters(in: match.range, with: "")
        }
        return (calls, mutableRemaining.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func decodeDSMLParameterValue(
        _ value: String,
        forceString: Bool
    ) -> any Sendable {
        let decoded = MLXModelService.decodeJSONEscapes(MLXModelService.decodeXMLEntities(value))
        guard !forceString else { return decoded }
        let trimmed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = trimmed.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
            return makeSendableJSON(json)
        }
        return decoded
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
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return number.boolValue
            }
            let doubleValue = number.doubleValue
            if floor(doubleValue) == doubleValue {
                return number.intValue
            }
            return doubleValue
        case let int as Int:
            return int
        case let double as Double:
            return double
        case let bool as Bool:
            return bool
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
