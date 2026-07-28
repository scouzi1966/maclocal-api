import XCTest
@preconcurrency import MLXLMCommon
@testable import AFMKitMLX

final class AFMMLXGenerationParameterFactoryTests: XCTestCase {
    func testFactoryMapsVisibleAndHiddenGenerationParameters() {
        let parameters = AFMMLXGenerationParameterFactory.make(
            maxTokens: 128,
            maxKVSize: 4096,
            kvBits: 4,
            kvGroupSize: 32,
            quantizedKVStart: 16,
            temperature: 0.25,
            topP: 0.75,
            repetitionPenalty: 1.15,
            repetitionContextSize: 96,
            topK: 40,
            minP: 0.05,
            presencePenalty: 0.2,
            prefillStepSize: 1024
        )

        XCTAssertEqual(parameters.maxTokens, 128)
        XCTAssertEqual(parameters.maxKVSize, 4096)
        XCTAssertEqual(parameters.kvBits, 4)
        XCTAssertEqual(parameters.kvGroupSize, 32)
        XCTAssertEqual(parameters.quantizedKVStart, 16)
        XCTAssertEqual(parameters.temperature, Float(0.25))
        XCTAssertEqual(parameters.topP, Float(0.75))
        XCTAssertEqual(parameters.repetitionPenalty, Float(1.15))
        XCTAssertEqual(parameters.repetitionContextSize, 96)
        XCTAssertEqual(parameters.topK, 40)
        XCTAssertEqual(parameters.minP, Float(0.05))
        XCTAssertEqual(parameters.presencePenalty, Float(0.2))
        XCTAssertEqual(parameters.prefillStepSize, 1024)
    }

    func testFactoryDropsNeutralRepetitionPenalty() {
        let parameters = AFMMLXGenerationParameterFactory.make(
            maxTokens: 16,
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.0,
            prefillStepSize: 512
        )

        XCTAssertNil(parameters.repetitionPenalty)
        XCTAssertEqual(parameters.repetitionContextSize, 64)
        XCTAssertEqual(parameters.kvGroupSize, 64)
        XCTAssertEqual(parameters.quantizedKVStart, 0)
        XCTAssertEqual(parameters.topK, 0)
        XCTAssertEqual(parameters.minP, 0)
        XCTAssertEqual(parameters.presencePenalty, 0)
    }

    func testFactoryKeepsRequestValueObjectAsStableContract() {
        let request = AFMMLXGenerationParameterRequest(
            maxTokens: 32,
            maxKVSize: nil,
            kvBits: nil,
            kvGroupSize: 64,
            quantizedKVStart: 0,
            temperature: 0,
            topP: 1,
            repetitionPenalty: 1.05,
            repetitionContextSize: 128,
            topK: 0,
            minP: 0,
            presencePenalty: 0,
            prefillStepSize: 2048
        )

        let parameters = AFMMLXGenerationParameterFactory.make(request)

        XCTAssertEqual(parameters.maxTokens, request.maxTokens)
        XCTAssertEqual(parameters.temperature, Float(request.temperature))
        XCTAssertEqual(parameters.topP, Float(request.topP))
        XCTAssertEqual(parameters.repetitionPenalty, Float(request.repetitionPenalty))
        XCTAssertEqual(parameters.repetitionContextSize, request.repetitionContextSize)
        XCTAssertEqual(parameters.prefillStepSize, request.prefillStepSize)
    }
}
