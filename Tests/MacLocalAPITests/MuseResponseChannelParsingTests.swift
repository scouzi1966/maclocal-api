import Testing
import MLXLMCommon

@testable import AFMKitMLX
@testable import AFMServer

struct MuseResponseChannelParsingTests {
    typealias Controller = MLXChatCompletionsController
    typealias State = Controller.MuseResponseChannelState

    @Test("Muse response-template channels route reasoning and visible content")
    func museChannelsRouteReasoningAndContent() {
        let raw = "to=self<|message|>Need a concise answer.<|eom|>to=user<|message|>The sky is blue because of Rayleigh scattering.<|return|>"

        let parsed = Controller.extractMuseResponseContent(from: raw)

        #expect(parsed.reasoning == "Need a concise answer.")
        #expect(parsed.content == "The sky is blue because of Rayleigh scattering.")
        #expect(!parsed.content.contains("to=self"))
        #expect(!parsed.content.contains("<|message|>"))
    }

    @Test("Muse streaming parser handles split channel controls")
    func museStreamingControlsCanSpanChunks() {
        var state = State()
        var buffer = "to=se"

        var first = Controller.extractMuseResponseChannels(buffer: &buffer, state: &state)
        #expect(first.reasoning == nil)
        #expect(first.content == nil)

        buffer += "lf<|message|>Plan before answer<|eo"
        first = Controller.extractMuseResponseChannels(buffer: &buffer, state: &state)
        #expect(first.reasoning?.contains("Plan") == true)
        #expect(first.content == nil)

        buffer += "m|>to=user<|message|>Visible answer<|return|>"
        let second = Controller.extractMuseResponseChannels(buffer: &buffer, state: &state)

        #expect(second.content == "Visible answer")
        #expect(state.stopReached)
    }

    @Test("Serving configuration preserves explicit Muse response channel format")
    func servingConfigurationPreservesMuseFormat() {
        let configuration = AFMMLXServingConfiguration(responseChannelFormat: .muse)

        #expect(configuration.responseChannelFormat == .muse)
        #expect(configuration.harmonyChannels == false)
    }

    @Test("Harmony compatibility still maps legacy bool to response channel format")
    func harmonyBoolMapsToResponseChannelFormat() {
        let configuration = AFMMLXServingConfiguration(harmonyChannels: true)

        #expect(configuration.responseChannelFormat == .harmony)
    }

    @Test("Muse auto-detects the ATEM tool-call format")
    func museAutoDetectsATEMFormat() {
        #expect(ToolCallFormat.infer(from: "muse_glimmer") == .atem)
        #expect(ToolCallFormat.infer(from: "muse_glimmer_text") == .atem)
    }

    @Test("ATEM parser preserves strings and coerces schema-backed values")
    func atemParserCoercesSchemaBackedValues() {
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "weather.lookup",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"],
                        "days": ["type": "integer"],
                        "metric": ["type": "boolean"],
                        "filters": ["type": "object"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let raw = """
        <atem:function_calls>
        <atem:invoke name="weather.lookup">
        <atem:parameter name="city"> New York </atem:parameter>
        <atem:parameter name="days">3</atem:parameter>
        <atem:parameter name="metric">true</atem:parameter>
        <atem:parameter name="filters">{"alerts":false}</atem:parameter>
        </atem:invoke>
        </atem:function_calls>
        """

        let call = ATEMToolCallParser().parse(content: raw, tools: tools)

        #expect(call?.function.name == "weather.lookup")
        #expect(call?.function.arguments["city"]?.anyValue as? String == " New York ")
        #expect(call?.function.arguments["days"]?.anyValue as? Int == 3)
        #expect(call?.function.arguments["metric"]?.anyValue as? Bool == true)
        let filters = call?.function.arguments["filters"]?.anyValue as? [String: Any]
        #expect(filters?["alerts"] as? Bool == false)
    }

    @Test("ATEM parser supports multiline parameter text")
    func atemParserSupportsMultilineText() {
        let raw = """
        <atem:function_calls>
        <atem:invoke name='write_note'>
        <atem:parameter name='body'>first line
        second line</atem:parameter>
        </atem:invoke>
        </atem:function_calls>
        """

        let call = ATEMToolCallParser().parse(content: raw, tools: nil)

        #expect(call?.function.name == "write_note")
        #expect(call?.function.arguments["body"]?.anyValue as? String == "first line\nsecond line")
    }

    @Test(
        "Muse reasoning effort maps to reasoning_strength",
        arguments: ["low", "high", "max"]
    )
    func museReasoningEffortMapsToReasoningStrength(_ effort: String) {
        let normalized = MLXModelService.normalizeReasoningKwargs(
            ["reasoning_effort": effort],
            canonicalModelType: "muse_glimmer",
            forceDisableThinking: false
        )

        #expect(normalized.kwargs["reasoning_strength"] as? String == effort)
        #expect(normalized.kwargs["reasoning_effort"] == nil)
        #expect(normalized.kwargs["enable_thinking"] == nil)
        #expect(normalized.note == nil)
    }

    @Test("Muse no-think requests lowest reasoning strength")
    func museNoThinkRequestsLowestReasoningStrength() {
        let normalized = MLXModelService.normalizeReasoningKwargs(
            [
                "enable_thinking": false,
                "reasoning_effort": "max",
            ],
            canonicalModelType: "muse_glimmer",
            forceDisableThinking: true
        )

        #expect(normalized.kwargs["reasoning_strength"] as? String == "low")
        #expect(normalized.kwargs["reasoning_effort"] == nil)
        #expect(normalized.kwargs["enable_thinking"] == nil)
        #expect(normalized.note?.contains("does not expose an off switch") == true)
    }

    @Test("Non-Muse reasoning kwargs are preserved")
    func nonMuseReasoningKwargsArePreserved() {
        let normalized = MLXModelService.normalizeReasoningKwargs(
            [
                "enable_thinking": true,
                "reasoning_effort": "high",
            ],
            canonicalModelType: "deepseek_v4",
            forceDisableThinking: false
        )

        #expect(normalized.kwargs["reasoning_effort"] as? String == "high")
        #expect(normalized.kwargs["enable_thinking"] as? Bool == true)
        #expect(normalized.note == nil)
    }
}
