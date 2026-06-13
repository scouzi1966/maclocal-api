import Foundation
import Testing

@testable import AFMKit

/// Unit tests for harmony channel parsing (gpt-oss). (#121)
///
/// Covers the streaming `extractHarmonyChannels` state machine and the
/// whole-text `extractHarmonyContent` finalizer. Validates routing of
/// `<|channel|>analysis|message|>` -> reasoning_content,
/// `<|channel|>final|message|>` -> content, control-token stripping,
/// `<|return|>` as a stop signal, and boundary handling for control
/// tokens that span chunk arrivals.
struct HarmonyChannelParsingTests {

    typealias Controller = MLXChatCompletionsController
    typealias State = Controller.HarmonyState

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Whole-text extraction (extractHarmonyContent)
    // ═══════════════════════════════════════════════════════════════════

    @Test("analysis then final routes correctly")
    func analysisAndFinalRouted() {
        let raw = "<|channel|>analysis<|message|>The user asks why the sky is blue. Reason about Rayleigh scattering.<|end|><|channel|>final<|message|>The sky is blue because of Rayleigh scattering.<|return|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(reasoning == "The user asks why the sky is blue. Reason about Rayleigh scattering.")
        #expect(content == "The sky is blue because of Rayleigh scattering.")
    }

    @Test("final-only with no analysis still routes to content")
    func finalOnlyRoutesToContent() {
        let raw = "<|channel|>final<|message|>Mars.<|return|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(reasoning == nil)
        #expect(content == "Mars.")
    }

    @Test("preamble before first <|channel|> is discarded")
    func preambleDiscarded() {
        let raw = "<|start|>assistant<|channel|>final<|message|>Mars.<|return|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(reasoning == nil)
        #expect(content == "Mars.")
    }

    @Test("commentary channel is dropped")
    func commentaryDropped() {
        // commentary is the tool-call channel — out of scope for this fix; should be discarded.
        let raw = "<|channel|>commentary<|message|>tool_use_payload<|end|><|channel|>final<|message|>visible answer<|return|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(reasoning == nil)
        #expect(content == "visible answer")
    }

    @Test("control tokens never leak into output")
    func controlTokensStripped() {
        let raw = "<|channel|>analysis<|message|>thinking text<|end|><|channel|>final<|message|>final answer<|return|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(!(reasoning ?? "").contains("<|"))
        #expect(!content.contains("<|"))
    }

    @Test("return inside analysis terminates without a final channel")
    func returnDuringAnalysisStops() {
        let raw = "<|channel|>analysis<|message|>partial reasoning<|return|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(reasoning == "partial reasoning")
        #expect(content == "")
    }

    @Test("end inside analysis without subsequent channel emits the reasoning so far")
    func endDuringAnalysisFlushesReasoning() {
        let raw = "<|channel|>analysis<|message|>thinking complete<|end|>"
        let (content, reasoning) = Controller.extractHarmonyContent(from: raw)
        #expect(reasoning == "thinking complete")
        #expect(content == "")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Streaming state machine (extractHarmonyChannels)
    // ═══════════════════════════════════════════════════════════════════

    @Test("streaming: <|channel|> split across two chunks is reassembled")
    func channelMarkerSpansChunks() {
        var state = State()
        var buffer = ""

        // Chunk 1 ends mid-control-token. The parser should hold the partial fragment.
        buffer += "<|start|>assistant<|chan"
        let r1 = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        #expect(r1.reasoning == nil)
        #expect(r1.content == nil)

        // Chunk 2 completes the marker plus message.
        buffer += "nel|>final<|message|>visible<|return|>"
        let r2 = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        #expect(r2.content == "visible")
        #expect(r2.reasoning == nil)
        #expect(state.stopReached == true)
    }

    @Test("streaming: <|end|> marker split across chunks closes the channel cleanly")
    func endMarkerSpansChunks() {
        var state = State()
        var buffer = "<|channel|>analysis<|message|>partial reason"

        let r1 = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        #expect(r1.reasoning != nil) // safe portion of analysis flushed

        // Marker arrives split: <|en + d|>
        buffer += "ing<|en"
        _ = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        // Should NOT have closed channel yet because <|en is partial
        #expect(state.channel != .none && state.stopReached == false)

        buffer += "d|><|channel|>final<|message|>answer<|return|>"
        let r3 = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        #expect(state.stopReached == true)
        #expect(r3.content == "answer")
    }

    @Test("streaming: stopReached is set on <|return|>")
    func stopReachedOnReturn() {
        var state = State()
        var buffer = "<|channel|>final<|message|>done<|return|>"
        _ = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        #expect(state.stopReached == true)
        #expect(state.channel == .done)
    }

    @Test("streaming: stopReached stays false when <|end|> closes a channel")
    func stopNotSetOnEnd() {
        var state = State()
        var buffer = "<|channel|>analysis<|message|>thoughts<|end|>"
        _ = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        #expect(state.stopReached == false)
        #expect(state.channel == .none)
    }

    @Test("streaming: long body emits in safe-tail-bounded increments")
    func longAnalysisFlushesIncrementally() {
        var state = State()
        var buffer = "<|channel|>analysis<|message|>"
        // Long body with no closing marker yet.
        let body = String(repeating: "abcdefghij", count: 50) // 500 chars
        buffer += body

        let extracted = Controller.extractHarmonyChannels(buffer: &buffer, state: &state)
        // Most of body must have been emitted as reasoning; only ~11 char tail kept.
        let emitted = extracted.reasoning ?? ""
        #expect(emitted.count >= body.count - 12)
        #expect(buffer.count <= 11)
        #expect(state.channel == .analysis)
    }
}
