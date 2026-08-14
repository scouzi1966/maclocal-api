import Testing

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
}
