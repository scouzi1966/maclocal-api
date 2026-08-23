import Testing

@testable import AFMServer

struct AFMServerStreamChunkTests {
    @Test("AFMServerStreamChunk defaults to a text-only delta")
    func streamChunkDefaults() {
        let chunk = AFMServerStreamChunk(text: "hello")

        #expect(chunk.text == "hello")
        #expect(chunk.logprobs == nil)
        #expect(chunk.toolCalls == nil)
        #expect(chunk.promptTokens == nil)
        #expect(chunk.completionTokens == nil)
        #expect(chunk.cachedTokens == nil)
        #expect(chunk.promptTime == nil)
        #expect(chunk.generateTime == nil)
    }

    @Test("AFMServerStreamChunk carries timing information")
    func streamChunkWithInfo() {
        let chunk = AFMServerStreamChunk(
            text: "",
            promptTokens: 100,
            completionTokens: 50,
            promptTime: 1.5,
            generateTime: 3.0
        )

        #expect(chunk.promptTokens == 100)
        #expect(chunk.completionTokens == 50)
        #expect(chunk.promptTime == 1.5)
        #expect(chunk.generateTime == 3.0)
    }

    @Test("AFMServerStreamChunk carries cached token count")
    func streamChunkCachedTokens() {
        let chunk = AFMServerStreamChunk(text: "", cachedTokens: 512)

        #expect(chunk.cachedTokens == 512)
    }
}
