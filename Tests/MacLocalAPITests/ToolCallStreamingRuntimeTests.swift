import Foundation
import Testing

@testable import MacLocalAPI

struct ToolCallStreamingRuntimeTests {
// dimensions: tool_call_format=xmlFunction/json, streaming=true

    @Test("runtime emits placeholder then final replacement for XML tool call")
    func emitsPlaceholderAndReplacement() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "get_weather", properties: ["location": ["type": "string"]], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        let start = runtime.process(piece: "<tool_call>")
        #expect(start.handled)
        #expect(start.events.count == 1)

        let function = runtime.process(piece: "<function=get_weather>")
        #expect(function.handled)
        #expect(function.events.count == 2)
        #expect(placeholder(from: function.events)?.function.name == "get_weather")
        #expect(delta(from: function.events)?.function?.name == "get_weather")

        let parameter = runtime.process(piece: "<parameter=location>Berlin</parameter>")
        #expect(parameter.handled)

        let end = runtime.process(piece: "</tool_call>")
        #expect(end.handled)
        #expect(replacement(from: end.events)?.function.name == "get_weather")
        #expect(replacement(from: end.events)?.function.arguments == #"{"location":"Berlin"}"#)
        #expect(delta(from: end.events)?.function?.arguments == #"{"location":"Berlin"}"#)
    }

    @Test("runtime parses adaptive xml JSON fallback")
    func parsesAdaptiveXMLJSONFallback() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "get_weather", properties: ["location": ["type": "string"]], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: #"{"name":"get_weather","arguments":{"location":"Berlin"}}"#)
        let end = runtime.process(piece: "</tool_call>")

        let collected = appended(from: end.events)
        #expect(collected?.function.name == "get_weather")
        #expect(collected?.function.arguments == #"{"location":"Berlin"}"#)
        #expect(delta(from: end.events)?.function?.name == "get_weather")
    }

    @Test("runtime preserves nulls in adaptive xml JSON fallback")
    func preservesNullsInAdaptiveXMLJSONFallback() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "search", properties: ["query": ["type": "string"], "cursor": ["type": "string"]], required: ["query"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: #"{"name":"search","arguments":{"query":"docs","cursor":null}}"#)
        let end = runtime.process(piece: "</tool_call>")

        let collected = appended(from: end.events)
        #expect(collected?.function.name == "search")
        #expect(collected?.function.arguments == #"{"cursor":null,"query":"docs"}"#)
    }

    @Test("runtime salvages incomplete tool call at stream end")
    func salvagesIncompleteToolCall() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "get_weather", properties: ["location": ["type": "string"]], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: "<function=get_weather>")
        _ = runtime.process(piece: "<parameter=location>Ber")

        let trailing = runtime.finishIncompleteToolCall()
        #expect(placeholder(from: trailing) == nil)
        #expect(trailing.contains(where: {
            if case .delta(let delta) = $0 {
                return delta.function?.arguments?.hasPrefix(#"{"location":"Ber"#) == true
            }
            return false
        }))
        #expect(trailing.contains(where: {
            if case .delta(let delta) = $0 {
                return delta.function?.arguments == "}"
            }
            return false
        }))
        #expect(replacement(from: trailing)?.function.arguments == #"{"location":"Ber"}"#)
    }

    private func makeTool(name: String, properties: [String: [String: Any]], required: [String]? = nil) -> RequestTool {
        var schemaDict: [String: Any] = [
            "type": "object",
            "properties": properties,
        ]
        if let required {
            schemaDict["required"] = required
        }
        let schemaData = try! JSONSerialization.data(withJSONObject: schemaDict)
        let schema = try! JSONDecoder().decode(AnyCodable.self, from: schemaData)
        return RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: name,
                description: nil,
                parameters: schema,
                strict: nil
            )
        )
    }

    private func placeholder(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        for event in events {
            if case .appendCollected(let toolCall) = event {
                return toolCall
            }
        }
        return nil
    }

    private func appended(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        placeholder(from: events)
    }

    private func replacement(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        for event in events {
            if case .replaceCollected(_, let toolCall) = event {
                return toolCall
            }
        }
        return nil
    }

    private func delta(from events: [ToolCallStreamingEvent]) -> StreamDeltaToolCall? {
        for event in events {
            if case .delta(let delta) = event {
                return delta
            }
        }
        return nil
    }
}
