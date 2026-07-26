import AFMKitCore
import XCTest

final class AFMReasoningOutputParserTests: XCTestCase {
    func testThinkTagsSeparateReasoningFromFinalContentAcrossChunks() {
        let parser = AFMReasoningOutputParser()

        let first = parser.processWithState(chunk: "<think>\nCheck facts")
        XCTAssertEqual(first.reasoningChunk, "\nCheck facts")
        XCTAssertNil(first.finalChunk)
        XCTAssertTrue(first.isInReasoningPhase)

        let middle = parser.processWithState(chunk: " before answering")
        XCTAssertNil(middle.finalChunk)
        XCTAssertTrue(middle.isInReasoningPhase)

        let last = parser.processWithState(chunk: "</think>\nThe sky is blue.")
        XCTAssertEqual(last.finalChunk, "\nThe sky is blue.")
        XCTAssertFalse(last.isInReasoningPhase)

        let result = parser.getResult()
        XCTAssertEqual(result.reasoning, "\nCheck facts before answering")
        XCTAssertEqual(result.finalContent, "\nThe sky is blue.")
        XCTAssertEqual(result.formatName, "thinkTags")
        XCTAssertNotNil(result.reasoningTokenCount)
    }

    func testImplicitThinkModeStreamsReasoningUntilClosingTag() {
        let parser = AFMReasoningOutputParser(allowImplicitReasoning: true)

        let first = parser.processWithState(chunk: "Plan")
        XCTAssertNil(first.finalChunk)
        XCTAssertTrue(first.isInReasoningPhase)

        let last = parser.processWithState(chunk: "</think>\nAnswer")
        XCTAssertEqual(last.reasoningChunk, "Plan")
        XCTAssertEqual(last.finalChunk, "Answer")
        XCTAssertFalse(last.isInReasoningPhase)

        let result = parser.getResult()
        XCTAssertEqual(result.reasoning, "Plan")
        XCTAssertEqual(result.finalContent, "\nAnswer")
        XCTAssertEqual(result.formatName, "thinkTagsImplicitStart")
    }

    func testGPTOSSMarkersDoNotStripBareEnglishWords() {
        let parser = AFMReasoningOutputParser()
        _ = parser.processWithState(
            chunk: "<|channel|>analysis<|message|>Use final and analysis as normal words."
        )
        let last = parser.processWithState(
            chunk: "<|channel|>final<|message|>Final answer."
        )

        XCTAssertEqual(last.finalChunk, "Final answer.")

        let result = parser.getResult()
        XCTAssertEqual(result.reasoning, "Use final and analysis as normal words.")
        XCTAssertEqual(result.finalContent, "Final answer.")
        XCTAssertEqual(result.formatName, "gptOSS")
    }

    func testSyncFilterMatchesParserForThinkTags() {
        let filter = AFMReasoningOutputFilterSync()

        let first = filter.process(chunk: "<think>reason")
        XCTAssertEqual(first.reasoning, "reason")
        XCTAssertNil(first.final)
        XCTAssertTrue(filter.isInReasoningPhase())

        let last = filter.process(chunk: "</think>done")
        XCTAssertNil(last.reasoning)
        XCTAssertEqual(last.final, "done")
        XCTAssertFalse(filter.isInReasoningPhase())

        let result = filter.getResult()
        XCTAssertEqual(result.reasoning, "reason")
        XCTAssertEqual(result.finalContent, "done")
    }

    func testFullTextExtractorSupportsCustomTags() {
        let extracted = AFMReasoningOutputExtractor.extractThinkContent(
            from: "<analysis>plan</analysis>\nAnswer",
            startTag: "<analysis>",
            endTag: "</analysis>"
        )

        XCTAssertEqual(extracted.reasoning, "plan")
        XCTAssertEqual(extracted.content, "Answer")
    }

    func testFullTextExtractorHandlesOrphanEndTag() {
        let extracted = AFMReasoningOutputExtractor.extractThinkContent(
            from: "reasoning trace</think>\nAnswer"
        )

        XCTAssertEqual(extracted.reasoning, "reasoning trace")
        XCTAssertEqual(extracted.content, "Answer")
    }
}
