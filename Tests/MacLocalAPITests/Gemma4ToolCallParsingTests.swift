import Foundation
import Testing

@testable import MacLocalAPI

struct Gemma4ToolCallParsingTests {
// dimensions: tool_call_format=gemma4, streaming=true/false

    @Test("Gemma4 parser coerces structured arguments using schema types")
    func parserCoercesStructuredArguments() {
        let parser = Gemma4FunctionParser()
        let tool = makeTool(
            name: "update_settings",
            properties: [
                "count": ["type": "integer"],
                "enabled": ["type": "boolean"],
                "tags": ["type": "array"],
                "config": ["type": "object"],
                "name": ["type": "string"],
            ]
        )

        let content = #"<|tool_call>call:update_settings{count:3,enabled:true,tags:["fast","safe"],config:{"theme":"dark","retries":2},name:<|"|>Alice<|"|>}<tool_call|>"#
        let toolCall = parser.parse(content: content, tools: [tool])

        #expect(toolCall?.function.name == "update_settings")
        #expect(toolCall?.function.arguments["count"]?.anyValue as? Int == 3)
        #expect(toolCall?.function.arguments["enabled"]?.anyValue as? Bool == true)
        #expect(toolCall?.function.arguments["name"]?.anyValue as? String == "Alice")

        let tags = toolCall?.function.arguments["tags"]?.anyValue as? [Any]
        #expect(tags?.count == 2)

        let config = toolCall?.function.arguments["config"]?.anyValue as? [String: Any]
        #expect(config?["theme"] as? String == "dark")
        #expect(config?["retries"] as? Int == 2)
    }

    @Test("Gemma4 parser preserves commas inside escaped strings")
    func parserPreservesEscapedStringCommas() {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:write_note{title:<|"|>hello, world<|"|>,priority:2}<tool_call|>"#

        let toolCall = parser.parse(content: content, tools: nil)

        #expect(toolCall?.function.name == "write_note")
        #expect(toolCall?.function.arguments["title"]?.anyValue as? String == "hello, world")
        #expect(toolCall?.function.arguments["priority"]?.anyValue as? Int == 2)
    }

    @Test("extractToolCallsFallback parses Gemma4 wrapper blocks")
    func extractFallbackParsesGemma4Blocks() {
        let text = """
        Before
        <|tool_call>call:get_weather{city:<|"|>Berlin<|"|>,days:5}<tool_call|>
        After
        """

        let (calls, remaining) = MLXModelService.extractToolCallsFallback(from: text)

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["city"]?.anyValue as? String == "Berlin")
        #expect(calls[0].function.arguments["days"]?.anyValue as? Int == 5)
        #expect(remaining.contains("Before"))
        #expect(remaining.contains("After"))
    }

    @Test("streaming runtime fallback parses completed Gemma4 tool call")
    func streamingRuntimeParsesCompletedGemma4ToolCall() {
        let text = #"<|tool_call>call:get_weather{city:<|"|>Berlin<|"|>,days:5}<tool_call|>"#

        let (calls, remaining) = ToolCallStreamingRuntime.parseCompletedToolCalls(
            from: text,
            toolCallParser: nil,
            tools: nil
        )

        #expect(calls.count == 1)
        #expect(calls[0].function.name == "get_weather")
        #expect(calls[0].function.arguments["city"]?.anyValue as? String == "Berlin")
        #expect(remaining.isEmpty)
    }

    private func makeTool(
        name: String,
        properties: [String: [String: String]]
    ) -> [String: any Sendable] {
        var propValues: [String: any Sendable] = [:]
        for (key, value) in properties {
            var sendableValue: [String: any Sendable] = [:]
            for (nestedKey, nestedValue) in value {
                sendableValue[nestedKey] = nestedValue
            }
            propValues[key] = sendableValue
        }
        return [
            "type": "function",
            "function": [
                "name": name,
                "parameters": [
                    "type": "object",
                    "properties": propValues,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
