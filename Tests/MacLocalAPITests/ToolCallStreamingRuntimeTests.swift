import Foundation
import Testing

@testable import AFMKit
@testable import AFMKitMLX
@testable import AFMServer

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
        #expect(emittedArgumentDeltas(parameter.events + end.events).joined() == #"{"location":"Berlin"}"#)
    }

    @Test("runtime does not resend full arguments after incremental XML deltas")
    func doesNotResendFinalArgumentsAfterIncrementalDeltas() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "get_weather", properties: [
                "location": ["type": "string"],
                "unit": ["type": "string"],
            ], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: "<function=get_weather>")
        let location = runtime.process(piece: "<parameter=location>Sydney</parameter>")
        let unit = runtime.process(piece: "<parameter=unit>fahrenheit</parameter>")
        let end = runtime.process(piece: "</tool_call>")

        #expect(replacement(from: end.events)?.function.arguments == #"{"location":"Sydney","unit":"fahrenheit"}"#)
        let deltas = emittedArgumentDeltas(location.events + unit.events + end.events)
        #expect(deltas.joined() == #"{"location":"Sydney","unit":"fahrenheit"}"#)
        #expect(!deltas.contains(#"{"location":"Sydney","unit":"fahrenheit"}"#))
    }

    @Test("runtime ignores duplicate JSON fallback after XML parameter deltas")
    func ignoresDuplicateJSONFallbackAfterXMLParameterDeltas() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "get_weather", properties: [
                "location": ["type": "string"],
                "unit": ["type": "string"],
            ], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: "<function=get_weather>")
        let location = runtime.process(piece: "<parameter=location>\nSydney\n</parameter>")
        let unit = runtime.process(piece: "<parameter=unit>\nfahrenheit\n</parameter>")
        _ = runtime.process(piece: #"{"location":"Sydney","unit":"fahrenheit"}"#)
        let end = runtime.process(piece: "</tool_call>")

        let deltas = emittedArgumentDeltas(location.events + unit.events + end.events)
        #expect(deltas.joined() == #"{"location":"Sydney","unit":"fahrenheit"}"#)
    }

    @Test("runtime does not append an extra brace when a parameter delta is already a full JSON object")
    func doesNotAppendExtraBraceForFullObjectParameterDelta() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [makeTool(name: "get_weather", properties: [
                "location": ["type": "string"],
                "unit": ["type": "string"],
            ], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: "<function=get_weather>")
        let arguments = runtime.process(piece: #"<parameter=arguments>{"location":"Sydney","unit":"fahrenheit"}</parameter>"#)
        let end = runtime.process(piece: "</tool_call>")

        let deltas = emittedArgumentDeltas(arguments.events + end.events)
        #expect(deltas.joined() == #"{"arguments":{"location":"Sydney","unit":"fahrenheit"}}"#)
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

    @Test("completed parser extracts DeepSeek DSML calls by syntax")
    func parsesDeepseekDSMLToolCalls() throws {
        let text = """
        I will update the file.
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="edit">
        <｜DSML｜parameter name="path" string="true">README.md</｜DSML｜parameter>
        <｜DSML｜parameter name="line">42</｜DSML｜parameter>
        <｜DSML｜parameter name="enabled">true</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """
        let tools = [makeTool(
            name: "edit",
            properties: [
                "path": ["type": "string"],
                "line": ["type": "integer"],
                "enabled": ["type": "boolean"],
            ],
            required: ["path"]
        )]

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: nil,
            tools: tools
        )

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "edit")
        #expect(calls[0].function.arguments["path"]?.anyValue as? String == "README.md")
        #expect(calls[0].function.arguments["line"]?.anyValue as? Int == 42)
        #expect(calls[0].function.arguments["enabled"] == .bool(true))
        #expect(remaining == "I will update the file.")
    }

    @Test("completed parser supports ASCII DeepSeek DSML markers and parallel calls")
    func parsesASCIIAndParallelDeepseekDSMLToolCalls() throws {
        let text = """
        <|DSML|tool_calls>
        <|DSML|invoke name="read"><|DSML|parameter name="path" string="true">a.txt</|DSML|parameter></|DSML|invoke>
        <|DSML|invoke name="read"><|DSML|parameter name="path" string="true">b.txt</|DSML|parameter></|DSML|invoke>
        </|DSML|tool_calls>
        """

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: nil,
            tools: [makeTool(name: "read", properties: ["path": ["type": "string"]])]
        )

        #expect(calls.count == 2)
        #expect(calls[0].function.arguments["path"]?.anyValue as? String == "a.txt")
        #expect(calls[1].function.arguments["path"]?.anyValue as? String == "b.txt")
        #expect(remaining.isEmpty)
    }

    @Test("streaming runtime buffers split DeepSeek DSML start tags")
    func buffersSplitDeepseekDSMLStartTags() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<｜DSML｜tool_calls>",
            toolCallEndTag: "</｜DSML｜tool_calls>",
            toolCallParser: "deepseek_dsml",
            tools: [makeTool(name: "get_weather", properties: ["location": ["type": "string"]], required: ["location"])],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        #expect(!runtime.process(piece: "prefix ").handled)
        #expect(runtime.process(piece: "<｜DSML｜").handled)
        #expect(runtime.process(piece: "tool").handled)
        #expect(runtime.process(piece: "_calls>\n").handled)
        _ = runtime.process(piece: "<｜DSML｜invoke name=\"get_weather\">\n")
        _ = runtime.process(piece: "<｜DSML｜parameter name=\"location\" string=\"true\">Toronto</｜DSML｜parameter>\n")
        let end = runtime.process(piece: "</｜DSML｜invoke>\n</｜DSML｜tool_calls>")

        #expect(end.handled)
        #expect(appended(from: end.events)?.function.name == "get_weather")
        #expect(appended(from: end.events)?.function.arguments == #"{"location":"Toronto"}"#)
        #expect(delta(from: end.events)?.function?.arguments == #"{"location":"Toronto"}"#)
    }

    @Test("completed parser extracts parallel Muse ATEM calls by syntax")
    func parsesParallelATEMToolCalls() throws {
        let text = """
        I will check both cities.
        <atem:function_calls>
        <atem:invoke name="get_weather"><atem:parameter name="location">Toronto</atem:parameter></atem:invoke>
        <atem:invoke name='get_weather'><atem:parameter name='location'>New York</atem:parameter></atem:invoke>
        </atem:function_calls>
        """
        let tools = [makeTool(
            name: "get_weather",
            properties: ["location": ["type": "string"]],
            required: ["location"]
        )]

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: nil,
            tools: tools
        )

        #expect(calls.count == 2)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["location"]?.anyValue as? String == "Toronto")
        #expect(calls[1].function.arguments["location"]?.anyValue as? String == "New York")
        #expect(remaining == "I will check both cities.")
    }

    @Test("streaming runtime buffers split Muse ATEM start tags")
    func buffersSplitATEMStartTags() throws {
        let runtime = ToolCallStreamingRuntime(
            toolCallStartTag: "<atem:function_calls>",
            toolCallEndTag: "</atem:function_calls>",
            toolCallParser: nil,
            tools: [makeTool(
                name: "get_weather",
                properties: ["location": ["type": "string"]],
                required: ["location"]
            )],
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )

        #expect(runtime.process(piece: "<atem:fun").handled)
        #expect(runtime.process(piece: "ction_calls>").handled)
        _ = runtime.process(piece: "<atem:invoke name=\"get_weather\">")
        _ = runtime.process(piece: "<atem:parameter name=\"location\">Toronto</atem:parameter>")
        let end = runtime.process(piece: "</atem:invoke></atem:function_calls>")

        #expect(end.handled)
        #expect(appended(from: end.events)?.function.name == "get_weather")
        #expect(appended(from: end.events)?.function.arguments == #"{"location":"Toronto"}"#)
        #expect(delta(from: end.events)?.function?.arguments == #"{"location":"Toronto"}"#)
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

    private func emittedArgumentDeltas(_ events: [ToolCallStreamingEvent]) -> [String] {
        events.compactMap { event -> String? in
            guard case .delta(let delta) = event else { return nil }
            return delta.function?.arguments
        }
    }
}
