import Testing

@testable import AFMKit
@testable import AFMServer

struct Qwen38ToolCallingQualificationTests {
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

    private var weatherTool: RequestTool {
        makeTool(name: "get_weather")
    }

    private var eventTool: RequestTool {
        makeTool(name: "create_event")
    }

    private func makeTool(name: String) -> RequestTool {
        RequestTool(
            type: "function",
            function: .init(
                name: name,
                description: nil,
                parameters: AnyCodable(["type": "object"]),
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
}
