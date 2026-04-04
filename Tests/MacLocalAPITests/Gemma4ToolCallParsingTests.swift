import Foundation
import Testing
import MLXLMCommon

@testable import MacLocalAPI

struct Gemma4ToolCallParsingTests {

    @Test("Gemma4 parser follows documented wrapper and call syntax")
    func parserParsesDocumentedFormat() {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:get_weather{location:<|"|>Berlin<|"|>,units:<|"|>celsius<|"|>}<tool_call|>"#

        let toolCall = parser.parse(content: content, tools: nil)

        #expect(toolCall?.function.name == "get_weather")
        #expect(toolCall?.function.arguments["location"]?.anyValue as? String == "Berlin")
        #expect(toolCall?.function.arguments["units"]?.anyValue as? String == "celsius")
    }

    @Test("Gemma4 parser preserves commas inside escaped strings")
    func parserPreservesEscapedStringCommas() {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:write_note{title:<|"|>hello, world<|"|>,priority:2}<tool_call|>"#

        let toolCall = parser.parse(content: content, tools: nil)

        #expect(toolCall?.function.name == "write_note")
        #expect(toolCall?.function.arguments["title"]?.anyValue as? String == "hello, world")
        #expect(toolCall?.function.arguments["priority"]?.anyValue as? String == "2")
    }

    @Test("Gemma4 parser leaves nested-looking values as raw strings")
    func parserLeavesNestedValuesUncoerced() {
        let parser = Gemma4FunctionParser()
        let content = #"<|tool_call>call:update_settings{tags:["fast","safe"],config:{"theme":"dark","retries":2}}<tool_call|>"#

        let toolCall = parser.parse(content: content, tools: nil)

        #expect(toolCall?.function.name == "update_settings")
        #expect(toolCall?.function.arguments["tags"]?.anyValue as? String == #"["fast","safe"]"#)
        #expect(toolCall?.function.arguments["config"]?.anyValue as? String == #"{"theme":"dark","retries":2}"#)
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
        #expect(calls[0].function.arguments["days"]?.anyValue as? String == "5")
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
}
