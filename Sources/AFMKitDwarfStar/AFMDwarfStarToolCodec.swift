import Foundation
import AFMKitCore

enum AFMDwarfStarToolCodec {
    static let blockStart = "<｜DSML｜tool_calls>"
    static let blockEnd = "</｜DSML｜tool_calls>"
    static let thinkStart = "<think>"
    static let thinkEnd = "</think>"
    static let endOfSentence = "<｜end▁of▁sentence｜>"

    enum StreamOutput: Equatable {
        case text(String)
        case reasoning(String)
        case toolCalls([AFMToolCall])
    }

    static func systemPrompt(
        for tools: [AFMToolDefinition],
        toolCallingRequired: Bool = false
    ) throws -> String {
        guard !tools.isEmpty else { return "" }
        let schemas = try tools.map { tool in
            var value: [String: Any] = [
                "name": tool.name,
                "parameters": try jsonObject(tool.inputSchema)
            ]
            if let description = tool.description {
                value["description"] = description
            }
            let data = try JSONSerialization.data(
                withJSONObject: value,
                options: [.sortedKeys]
            )
            return String(decoding: data, as: UTF8.self)
        }.joined(separator: "\n")

        let requirement = toolCallingRequired
            ? "You MUST call at least one available tool before answering.\n\n"
            : ""
        return """
        ## Tools

        You have access to tools. Invoke one or more tools using exactly this syntax:

        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="$TOOL_NAME">
        <｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>

        String parameters use raw text with string="true". Numbers, booleans,
        arrays, objects, and null use JSON with string="false". Use the exact
        tool and parameter names from these schemas:

        \(requirement)\(schemas)
        """
    }

    static func assistantContent(for message: AFMMessage) throws -> String {
        var content = try textContent(of: message)
        guard !message.toolCalls.isEmpty else { return content }
        if !content.isEmpty { content += "\n\n" }
        content += try renderedToolCalls(message.toolCalls)
        return content
    }

    static func assistantReplaySuffix(for message: AFMMessage) throws -> String {
        let calls = message.toolCalls
        guard !calls.isEmpty else { return endOfSentence }
        return "\n\n" + (try renderedToolCalls(calls)) + endOfSentence
    }

    static func renderedToolCalls(_ calls: [AFMToolCall]) throws -> String {
        var content = blockStart + "\n"
        for call in calls {
            content += "<｜DSML｜invoke name=\"\(escapeAttribute(call.name))\">\n"
            let object = try argumentsObject(call.arguments)
            for key in object.keys.sorted() {
                let value = object[key] as Any
                if let string = value as? String {
                    content += "<｜DSML｜parameter name=\"\(escapeAttribute(key))\" string=\"true\">"
                    content += escapeParameter(string)
                } else {
                    let data = try JSONSerialization.data(
                        withJSONObject: value,
                        options: [.sortedKeys, .fragmentsAllowed]
                    )
                    content += "<｜DSML｜parameter name=\"\(escapeAttribute(key))\" string=\"false\">"
                    content += String(decoding: data, as: UTF8.self)
                }
                content += "</｜DSML｜parameter>\n"
            }
            content += "</｜DSML｜invoke>\n"
        }
        content += blockEnd
        return content
    }

    static func textContent(of message: AFMMessage) throws -> String {
        var result = ""
        for part in message.content {
            guard case .text(let text) = part else {
                throw AFMError.unsupportedCapability("non-text DwarfStar input")
            }
            result += text
        }
        return result
    }

    struct StreamParser {
        private enum Channel {
            case reasoning
            case response
        }

        private var buffer = ""
        private var parsingTools = false
        private var completedToolCall = false
        private var nextCallIndex = 0
        private var channel: Channel

        init(startsInReasoning: Bool = false) {
            channel = startsInReasoning ? .reasoning : .response
        }

        mutating func consume(_ text: String) throws -> [StreamOutput] {
            guard !completedToolCall else { return [] }
            buffer += text
            var outputs: [StreamOutput] = []

            while true {
                if parsingTools {
                    guard let end = buffer.range(of: blockEnd) else { return outputs }
                    let block = String(buffer[..<end.upperBound])
                    let calls = try parseToolCalls(block, nextIndex: &nextCallIndex)
                    if !calls.isEmpty {
                        outputs.append(.toolCalls(calls))
                        completedToolCall = true
                    }
                    buffer = ""
                    parsingTools = false
                    return outputs
                }

                switch channel {
                case .reasoning:
                    if buffer.hasPrefix(thinkStart) {
                        buffer.removeFirst(thinkStart.count)
                        continue
                    }
                    if let end = buffer.range(of: thinkEnd) {
                        append(String(buffer[..<end.lowerBound]), to: .reasoning, in: &outputs)
                        buffer = String(buffer[end.upperBound...])
                        channel = .response
                        continue
                    }
                    emitSafePrefix(retaining: [thinkStart, thinkEnd], asReasoning: true, into: &outputs)
                    return outputs

                case .response:
                    let toolRange = buffer.range(of: blockStart)
                    let thinkRange = buffer.range(of: thinkStart)
                    if let next = earliest(toolRange, thinkRange) {
                        append(String(buffer[..<next.lowerBound]), to: .response, in: &outputs)
                        let marker = String(buffer[next])
                        buffer = String(buffer[next.upperBound...])
                        if marker == blockStart {
                            buffer = marker + buffer
                            parsingTools = true
                        } else {
                            channel = .reasoning
                        }
                        continue
                    }
                    emitSafePrefix(retaining: [blockStart, thinkStart], asReasoning: false, into: &outputs)
                    return outputs
                }
            }
        }

        mutating func finish() -> [StreamOutput] {
            guard !completedToolCall else { return [] }
            guard !buffer.isEmpty else { return [] }
            defer { buffer = "" }
            guard !parsingTools else { return [] }
            return channel == .reasoning ? [.reasoning(buffer)] : [.text(buffer)]
        }

        private func earliest(
            _ lhs: Range<String.Index>?,
            _ rhs: Range<String.Index>?
        ) -> Range<String.Index>? {
            switch (lhs, rhs) {
            case (.none, .none): nil
            case (.some(let range), .none), (.none, .some(let range)): range
            case (.some(let left), .some(let right)):
                left.lowerBound < right.lowerBound ? left : right
            }
        }

        private mutating func emitSafePrefix(
            retaining markers: [String],
            asReasoning: Bool,
            into outputs: inout [StreamOutput]
        ) {
            let retained = markers.map { longestMarkerPrefixSuffix(in: buffer, marker: $0) }.max() ?? 0
            let emitCount = buffer.count - retained
            guard emitCount > 0 else { return }
            let split = buffer.index(buffer.startIndex, offsetBy: emitCount)
            let value = String(buffer[..<split])
            outputs.append(asReasoning ? .reasoning(value) : .text(value))
            buffer = String(buffer[split...])
        }

        private func append(
            _ value: String,
            to channel: Channel,
            in outputs: inout [StreamOutput]
        ) {
            guard !value.isEmpty else { return }
            switch channel {
            case .response: outputs.append(.text(value))
            case .reasoning: outputs.append(.reasoning(value))
            }
        }

        private func longestMarkerPrefixSuffix(in text: String, marker: String) -> Int {
            let maximum = min(text.count, marker.count - 1)
            guard maximum > 0 else { return 0 }
            for count in stride(from: maximum, through: 1, by: -1) {
                if text.suffix(count) == marker.prefix(count) { return count }
            }
            return 0
        }
    }

    private static func parseToolCalls(
        _ block: String,
        nextIndex: inout Int
    ) throws -> [AFMToolCall] {
        let invoke = try NSRegularExpression(
            pattern: #"<｜DSML｜invoke\s+name="([^"]+)">(.*?)</｜DSML｜invoke>"#,
            options: [.dotMatchesLineSeparators]
        )
        let parameter = try NSRegularExpression(
            pattern: #"<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</｜DSML｜parameter>"#,
            options: [.dotMatchesLineSeparators]
        )
        let source = block as NSString
        return try invoke.matches(
            in: block,
            range: NSRange(location: 0, length: source.length)
        ).map { match in
            let name = unescape(source.substring(with: match.range(at: 1)))
            let body = source.substring(with: match.range(at: 2))
            let bodySource = body as NSString
            var arguments: [String: Any] = [:]
            for item in parameter.matches(
                in: body,
                range: NSRange(location: 0, length: bodySource.length)
            ) {
                let key = unescape(bodySource.substring(with: item.range(at: 1)))
                let isString = bodySource.substring(with: item.range(at: 2)) == "true"
                let raw = unescape(bodySource.substring(with: item.range(at: 3)))
                if isString {
                    arguments[key] = raw
                } else {
                    let data = Data(raw.utf8)
                    arguments[key] = try JSONSerialization.jsonObject(
                        with: data,
                        options: [.fragmentsAllowed]
                    )
                }
            }
            nextIndex += 1
            let data = try JSONSerialization.data(
                withJSONObject: arguments,
                options: [.sortedKeys]
            )
            return AFMToolCall(
                id: "call_\(nextIndex)",
                name: name,
                arguments: String(decoding: data, as: UTF8.self)
            )
        }
    }

    private static func argumentsObject(_ arguments: String) throws -> [String: Any] {
        guard !arguments.isEmpty else { return [:] }
        let value = try JSONSerialization.jsonObject(with: Data(arguments.utf8))
        guard let object = value as? [String: Any] else {
            throw AFMError.invalidRequest("DwarfStar tool arguments must be a JSON object.")
        }
        return object
    }

    private static func jsonObject(_ value: AFMJSONValue) throws -> Any {
        switch value {
        case .null: return NSNull()
        case .bool(let value): return value
        case .integer(let value): return value
        case .number(let value): return value
        case .string(let value): return value
        case .array(let values): return try values.map(jsonObject)
        case .object(let values): return try values.mapValues(jsonObject)
        }
    }

    private static func escapeAttribute(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    private static func escapeParameter(_ value: String) -> String {
        value.replacingOccurrences(
            of: "</｜DSML｜parameter>",
            with: "&lt;/｜DSML｜parameter>"
        )
    }

    private static func unescape(_ value: String) -> String {
        value.replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&amp;", with: "&")
    }
}
