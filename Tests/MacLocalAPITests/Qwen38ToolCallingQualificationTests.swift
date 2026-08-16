import Foundation
import Testing

@testable import AFMKit
@testable import AFMKitMLX
@testable import AFMServer

/// Release qualification for Qwen 3.8 tool-call output shapes.
///
/// These fixtures intentionally exercise the parser and API contract without
/// depending on a live model, so regressions identify the AFM layer rather than
/// model selection quality.
struct Qwen38ToolCallingQualificationTests {
    @Test("Qwen 3.8 native JSON-in-XML preserves nested Unicode arguments")
    func nativeJSONInXMLPreservesComplexArguments() throws {
        let text = #"""
        <tool_call>
        {"name":"create_event","arguments":{"title":"Café 🚀","attendees":["Zoë","Miyuki"],"metadata":{"priority":2,"remote":true},"note":null}}
        </tool_call>
        """#

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [eventTool]
        )

        let call = try #require(calls.first)
        #expect(calls.count == 1)
        #expect(call.function.name == "create_event")
        #expect(call.function.arguments["title"]?.anyValue as? String == "Café 🚀")
        #expect(call.function.arguments["attendees"]?.anyValue as? [Any] != nil)
        #expect(call.function.arguments["metadata"]?.anyValue as? [String: Any] != nil)
        #expect(call.function.arguments["note"] == .null)
        #expect(remaining.isEmpty)
    }

    @Test("Qwen 3.8 streaming assembles a JSON tool call split at arbitrary boundaries")
    func streamingAssemblesFragmentedJSONInXML() throws {
        let runtime = makeRuntime(tools: [weatherTool])
        let pieces = [
            "<tool", "_call>\n{\"na", "me\":\"get_weather\",",
            "\"arguments\":{\"location\":\"Montr", "éal\",\"days\":3}}\n",
            "</tool_", "call>",
        ]

        var events: [ToolCallStreamingEvent] = []
        for piece in pieces {
            events.append(contentsOf: runtime.process(piece: piece).events)
        }

        let call = try #require(collectedCall(from: events))
        #expect(call.function.name == "get_weather")
        let arguments = try decodeArguments(call.function.arguments)
        #expect(arguments["location"] as? String == "Montréal")
        #expect(arguments["days"] as? Int == 3)
    }

    @Test("Qwen 3.8 adaptive mode repairs malformed name-equals output without leaking syntax")
    func adaptiveModeRepairsMalformedOutput() throws {
        let text = #"<tool_call>{"name="get_weather", "arguments":{"location":"Toronto","days":2}}</tool_call>"#

        let (strictCalls, strictRemaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: nil,
            tools: [weatherTool]
        )
        #expect(strictCalls.isEmpty)
        #expect(strictRemaining.contains("<tool_call>"))

        let (adaptiveCalls, adaptiveRemaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [weatherTool]
        )
        let call = try #require(adaptiveCalls.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["location"]?.anyValue as? String == "Toronto")
        #expect(call.function.arguments["days"]?.anyValue as? Int == 2)
        #expect(adaptiveRemaining.isEmpty)
    }

    @Test("Qwen 3.8 incomplete stream is salvaged as one valid call")
    func incompleteStreamIsSalvaged() throws {
        let runtime = makeRuntime(tools: [weatherTool])
        _ = runtime.process(piece: "<tool_call>")
        _ = runtime.process(piece: "<function=get_weather>")
        _ = runtime.process(piece: "<parameter=location>Toronto</parameter>")
        _ = runtime.process(piece: "<parameter=days>3")

        let events = runtime.finishIncompleteToolCall()
        let call = try #require(replacementCall(from: events))
        let arguments = try decodeArguments(call.function.arguments)
        #expect(call.function.name == "get_weather")
        #expect(arguments["location"] as? String == "Toronto")
        #expect(call.function.arguments.contains(#""days":3"#))
    }

    @Test("Qwen 3.8 completed parser preserves valid parallel calls")
    func completedParserPreservesParallelCalls() throws {
        let text = #"""
        <tool_call>{"name":"get_weather","arguments":{"location":"Toronto","days":1}}</tool_call>
        <tool_call>{"name":"get_weather","arguments":{"location":"Vancouver","days":2}}</tool_call>
        """#

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [weatherTool]
        )

        #expect(calls.count == 2)
        #expect(calls[0].function.arguments["location"]?.anyValue as? String == "Toronto")
        #expect(calls[1].function.arguments["location"]?.anyValue as? String == "Vancouver")
        #expect(remaining.isEmpty)
    }

    @Test("tool_choice none suppresses otherwise valid Qwen 3.8 calls")
    func toolChoiceNoneSuppressesCalls() {
        let calls = [responseCall(name: "get_weather", index: 0)]
        #expect(MLXChatCompletionsController.applyToolChoice(calls, toolChoice: .mode("none")) == nil)
    }

    @Test("named tool_choice exposes and accepts only the requested function")
    func namedToolChoiceFiltersToolsAndCalls() throws {
        let choice = ToolChoice.function(.init(
            type: "function",
            function: .init(name: "create_event")
        ))
        let effectiveTools = try MLXChatCompletionsController.resolveEffectiveTools(
            [weatherTool, eventTool],
            toolChoice: choice
        )
        let calls = [
            responseCall(name: "get_weather", index: 0),
            responseCall(name: "create_event", index: 1),
        ]
        let effectiveCalls = MLXChatCompletionsController.applyToolChoice(calls, toolChoice: choice)

        #expect(effectiveTools?.map(\.function.name) == ["create_event"])
        #expect(effectiveCalls?.map(\.function.name) == ["create_event"])
    }

    @Test("parallel_tool_calls false returns exactly one structured call")
    func parallelFalseReturnsOneCall() {
        let turn = MLXChatCompletionsController.finalizeAssistantTurn(
            content: "tool syntax must not appear as assistant content",
            toolCalls: [
                responseCall(name: "get_weather", index: 0),
                responseCall(name: "create_event", index: 1),
            ],
            toolChoice: .mode("auto"),
            parallelToolCalls: false,
            extractThinking: false,
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            stoppedBySequence: false,
            completionTokens: 10,
            maxTokens: 128,
            sanitizeContent: { $0 }
        )

        #expect(turn.finishReason == "tool_calls")
        #expect(turn.content == nil)
        #expect(turn.toolCalls?.count == 1)
        #expect(turn.toolCalls?.first?.function.name == "get_weather")
    }

    @Test("ordinary text containing escaped tool syntax remains text")
    func escapedToolSyntaxIsNotParsed() {
        let text = "Document the literal &lt;tool_call&gt; marker without invoking anything."
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: "afm_adaptive_xml",
            tools: [weatherTool]
        )
        #expect(calls.isEmpty)
        #expect(remaining == text)
    }

    private var weatherTool: RequestTool {
        makeTool(
            name: "get_weather",
            properties: [
                "location": ["type": "string"],
                "days": ["type": "integer"],
            ],
            required: ["location"]
        )
    }

    private var eventTool: RequestTool {
        makeTool(
            name: "create_event",
            properties: [
                "title": ["type": "string"],
                "attendees": ["type": "array", "items": ["type": "string"]],
                "metadata": ["type": "object"],
                "note": ["type": ["string", "null"]],
            ],
            required: ["title"]
        )
    }

    private func makeRuntime(tools: [RequestTool]) -> ToolCallStreamingRuntime {
        ToolCallStreamingRuntime(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: tools,
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )
    }

    private func makeTool(
        name: String,
        properties: [String: Any],
        required: [String]
    ) -> RequestTool {
        let schema: [String: Any] = [
            "type": "object",
            "properties": properties,
            "required": required,
        ]
        let data = try! JSONSerialization.data(withJSONObject: schema)
        return RequestTool(
            type: "function",
            function: .init(
                name: name,
                description: nil,
                parameters: try! JSONDecoder().decode(AnyCodable.self, from: data),
                strict: nil
            )
        )
    }

    private func responseCall(name: String, index: Int) -> ResponseToolCall {
        ResponseToolCall(
            index: index,
            id: "call_\(index)",
            type: "function",
            function: .init(name: name, arguments: "{}")
        )
    }

    private func collectedCall(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        for event in events.reversed() {
            switch event {
            case .replaceCollected(_, let call), .appendCollected(let call):
                return call
            default:
                continue
            }
        }
        return nil
    }

    private func replacementCall(from events: [ToolCallStreamingEvent]) -> ResponseToolCall? {
        for event in events.reversed() {
            if case .replaceCollected(_, let call) = event {
                return call
            }
        }
        return nil
    }

    private func decodeArguments(_ arguments: String) throws -> [String: Any] {
        try #require(
            JSONSerialization.jsonObject(with: Data(arguments.utf8)) as? [String: Any]
        )
    }

}
