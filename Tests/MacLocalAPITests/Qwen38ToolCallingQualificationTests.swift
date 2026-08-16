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

    @Test("tool_choice auto and required preserve the complete tool surface")
    func automaticAndRequiredToolChoicePreserveTools() throws {
        let tools = [weatherTool, eventTool]
        let calls = [
            responseCall(name: "get_weather", index: 0),
            responseCall(name: "create_event", index: 1),
        ]

        for mode in ["auto", "required"] {
            let choice = ToolChoice.mode(mode)
            let effectiveTools = try MLXChatCompletionsController.resolveEffectiveTools(
                tools,
                toolChoice: choice
            )
            let effectiveCalls = MLXChatCompletionsController.applyToolChoice(
                calls,
                toolChoice: choice
            )

            #expect(effectiveTools?.map(\.function.name) == ["get_weather", "create_event"])
            #expect(effectiveCalls?.map(\.function.name) == ["get_weather", "create_event"])
        }
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

    @Test("OpenCode grep arguments preserve arrays and optional paths")
    func parsesOpenCodeGrepArguments() throws {
        let tool = makeTool(
            name: "grep",
            properties: [
                "pattern": ["type": "string"],
                "path": ["type": "string"],
                "include": ["type": "array", "items": ["type": "string"]],
            ],
            required: ["pattern"]
        )
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: #"<tool_call>{"name":"grep","arguments":{"pattern":"finalizeAssistantTurn","path":"Sources","include":["*.swift","*.md"]}}</tool_call>"#,
            toolCallParser: "afm_adaptive_xml",
            tools: [tool]
        )
        let call = try #require(calls.first)
        #expect(calls.count == 1)
        #expect(remaining.isEmpty)
        #expect(call.function.name == "grep")
        #expect(call.function.arguments["path"]?.anyValue as? String == "Sources")
        #expect(call.function.arguments["include"]?.anyValue as? [String] == ["*.swift", "*.md"])
    }

    @Test("Pi write arguments preserve camelCase fields and Unicode content")
    func parsesPiWriteArguments() throws {
        let tool = makeTool(
            name: "write",
            properties: [
                "path": ["type": "string"],
                "content": ["type": "string"],
                "createDirectories": ["type": "boolean"],
            ],
            required: ["path", "content"]
        )
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: #"<tool_call>{"name":"write","arguments":{"path":"Notes/café.md","content":"line 1\nline 2 🚀","createDirectories":true}}</tool_call>"#,
            toolCallParser: "afm_adaptive_xml",
            tools: [tool]
        )
        let call = try #require(calls.first)
        #expect(calls.count == 1)
        #expect(remaining.isEmpty)
        #expect(call.function.arguments["path"]?.anyValue as? String == "Notes/café.md")
        #expect(call.function.arguments["content"]?.anyValue as? String == "line 1\nline 2 🚀")
        #expect(call.function.arguments["createDirectories"]?.anyValue as? Bool == true)
    }

    @Test("OpenClaw apply_patch preserves a multiline unified diff")
    func parsesOpenClawApplyPatchArguments() throws {
        let tool = makeTool(
            name: "apply_patch",
            properties: ["patch": ["type": "string"]],
            required: ["patch"]
        )
        let diff = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new"
        let encodedDiff = try JSONEncoder().encode(diff)
        let quotedDiff = try #require(String(data: encodedDiff, encoding: .utf8))
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: "<tool_call>{\"name\":\"apply_patch\",\"arguments\":{\"patch\":\(quotedDiff)}}</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: [tool]
        )
        let call = try #require(calls.first)
        #expect(calls.count == 1)
        #expect(remaining.isEmpty)
        #expect(call.function.arguments["patch"]?.anyValue as? String == diff)
    }

    @Test("Hermes todo preserves nested arrays and nullable metadata")
    func parsesHermesTodoArguments() throws {
        let tool = makeTool(
            name: "todo",
            properties: [
                "items": [
                    "type": "array",
                    "items": [
                        "type": "object",
                        "properties": [
                            "content": ["type": "string"],
                            "status": ["type": "string"],
                            "owner": ["type": ["string", "null"]],
                        ],
                    ],
                ],
            ],
            required: ["items"]
        )
        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: #"<tool_call>{"name":"todo","arguments":{"items":[{"content":"Inspect API","status":"in_progress","owner":null},{"content":"Run tests","status":"pending","owner":"agent"}]}}</tool_call>"#,
            toolCallParser: "afm_adaptive_xml",
            tools: [tool]
        )
        let call = try #require(calls.first)
        #expect(calls.count == 1)
        #expect(remaining.isEmpty)
        let items = try #require(call.function.arguments["items"]?.anyValue as? [[String: Any]])

        #expect(items.count == 2)
        #expect(items[0]["content"] as? String == "Inspect API")
        #expect(items[0]["owner"] is NSNull)
        #expect(items[1]["owner"] as? String == "agent")
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

    private func responseCall(name: String, index: Int, arguments: String = "{}") -> ResponseToolCall {
        ResponseToolCall(
            index: index,
            id: "call_\(index)",
            type: "function",
            function: .init(name: name, arguments: arguments)
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
