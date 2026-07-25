import AFMKitCore
@testable import AFMKitMLX
import AFMOpenAICompat
import XCTest

final class MLXStreamEventTranslatorTests: XCTestCase {
    func testReasoningTagsSplitAcrossChunksProduceSeparateChannels() {
        var translator = MLXStreamEventTranslator(
            thinkStartTag: "<think>",
            thinkEndTag: "</think>",
            maximumResponseTokens: 100
        )

        let events = [
            translator.consume(.init(text: "visible<th")),
            translator.consume(.init(text: "ink>private</thi")),
            translator.consume(.init(text: "nk>answer")),
            translator.finish()
        ].flatMap { $0 }

        XCTAssertEqual(text(from: events), "visibleanswer")
        XCTAssertEqual(reasoning(from: events), "private")
        XCTAssertEqual(tokenCount(from: events), 3)
        XCTAssertEqual(completionReason(from: events), .stop)
    }

    func testToolDeltasAreNotDuplicatedByCompletedCall() {
        var translator = MLXStreamEventTranslator(
            thinkStartTag: nil,
            thinkEndTag: nil,
            maximumResponseTokens: 100
        )

        let events = [
            translator.consume(
                .init(
                    text: "",
                    toolCallDeltas: [
                        .init(
                            index: 0,
                            id: "call_1",
                            type: "function",
                            function: .init(name: "weather", arguments: "{\"city\":")
                        ),
                        .init(
                            index: 0,
                            id: nil,
                            type: nil,
                            function: .init(name: nil, arguments: "\"Toronto\"}")
                        )
                    ]
                )
            ),
            translator.consume(
                .init(
                    text: "",
                    toolCalls: [
                        .init(
                            index: 0,
                            id: "call_1",
                            type: "function",
                            function: .init(
                                name: "weather",
                                arguments: "{\"city\":\"Toronto\"}"
                            )
                        )
                    ]
                )
            ),
            translator.finish()
        ].flatMap { $0 }

        let argumentDeltas = events.compactMap { event -> String? in
            guard case .toolCall(_, .argumentsDelta(let delta)) = event else {
                return nil
            }
            return delta
        }
        XCTAssertEqual(argumentDeltas.joined(), "{\"city\":\"Toronto\"}")
        XCTAssertEqual(completedToolCall(from: events)?.arguments, "{\"city\":\"Toronto\"}")
        XCTAssertEqual(completionReason(from: events), .toolCalls)
    }

    func testMaximumTokenUsageProducesLengthCompletion() {
        var translator = MLXStreamEventTranslator(
            thinkStartTag: nil,
            thinkEndTag: nil,
            maximumResponseTokens: 2
        )
        var events = translator.consume(
            .init(text: "done", promptTokens: 3, completionTokens: 2)
        )
        events += translator.finish()

        XCTAssertEqual(completionReason(from: events), .length)
    }

    private func text(from events: [AFMGenerationEvent]) -> String {
        events.compactMap {
            guard case .responseText(_, let text, _) = $0 else { return nil }
            return text
        }.joined()
    }

    private func reasoning(from events: [AFMGenerationEvent]) -> String {
        events.compactMap {
            guard case .reasoningText(_, let text, _) = $0 else { return nil }
            return text
        }.joined()
    }

    private func completedToolCall(from events: [AFMGenerationEvent]) -> AFMToolCall? {
        events.compactMap {
            guard case .toolCall(let call, .completed) = $0 else { return nil }
            return call
        }.last
    }

    private func completionReason(from events: [AFMGenerationEvent]) -> AFMFinishReason? {
        events.compactMap {
            guard case .completed(let reason) = $0 else { return nil }
            return reason
        }.last
    }

    private func tokenCount(from events: [AFMGenerationEvent]) -> Int {
        events.reduce(into: 0) { total, event in
            switch event {
            case .responseText(_, _, let count), .reasoningText(_, _, let count):
                total += count
            default:
                break
            }
        }
    }
}
