#if canImport(FoundationModels)
import AFMKit
@testable import AFMKitFoundationModels27
@testable import AFMKitFoundationModels27DwarfStar
import FoundationModels
import XCTest

final class MLXFoundationLanguageModelTests: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        guard #available(macOS 27.0, *) else {
            throw XCTSkip("The Foundation Models LanguageModel API requires macOS 27")
        }
    }

    @available(macOS 27.0, *)
    func testCompatibilityFacadeReexportsMLXLanguageModel() {
        let model = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsVision: true,
            supportsReasoning: true,
            supportsToolCalling: true,
            supportsGuidedGeneration: true
        )

        XCTAssertEqual(model.modelID, "mlx-community/model-a")
        XCTAssertEqual(model.executorConfiguration.kvBits, 8)
        XCTAssertTrue(model.executorConfiguration.enablePrefixCaching)
        XCTAssertTrue(model.executorConfiguration.mtpEnabled)
        XCTAssertEqual(model.executorConfiguration.mtpDepth, 2)
        XCTAssertEqual(model.defaultMaximumResponseTokens, 4_096)
        XCTAssertTrue(model.capabilities.contains(.vision))
        XCTAssertTrue(model.capabilities.contains(.reasoning))
        XCTAssertTrue(model.capabilities.contains(.toolCalling))
        XCTAssertTrue(model.capabilities.contains(.guidedGeneration))
    }

    @available(macOS 27.0, *)
    func testCompatibilityFacadeReexportsLanguageModelPlan() {
        let descriptor = AFMModelDescriptor(
            providerID: "afmkit.mlx",
            modelID: "mlx-community/model-a",
            displayName: "Model A",
            capabilities: [.text, .vision, .reasoning, .toolCalling, .structuredOutput],
            contextWindow: 16_384,
            privacyBoundary: .device,
            requiresNetwork: false
        )

        let plan = AFMMLXFoundationLanguageModelPlan.make(
            modelID: "/cache/model-a",
            descriptor: descriptor,
            defaultMaximumResponseTokens: 768
        )

        XCTAssertEqual(plan.modelID, "/cache/model-a")
        XCTAssertEqual(plan.defaultMaximumResponseTokens, 768)
        XCTAssertTrue(plan.supportsVision)
        XCTAssertTrue(plan.supportsReasoning)
        XCTAssertTrue(plan.supportsToolCalling)
        XCTAssertTrue(plan.supportsGuidedGeneration)
    }

    @available(macOS 27.0, *)
    func testCompatibilityFacadeReexportsRequestAdapter() throws {
        let transcript = Transcript(entries: [
            .instructions(
                .init(
                    segments: [.text(.init(content: "Be concise."))],
                    toolDefinitions: []
                )
            ),
            .prompt(.init(segments: [.text(.init(content: "Question"))])),
            .response(
                .init(metadata: [:], segments: [.text(.init(content: "Answer"))])
            )
        ])

        let messages = try AFMFoundationModelsRequestAdapter.messages(from: transcript)

        XCTAssertEqual(messages.map(\.role), [.system, .user, .assistant])
        XCTAssertEqual(messages.map(Self.text), ["Be concise.", "Question", "Answer"])
    }

    @available(macOS 27.0, *)
    func testDwarfStarLanguageModelPublishesAppleToolCallingContract() {
        let model = DwarfStarLanguageModel(
            modelPath: "/models/deepseek-v4-flash.gguf",
            contextWindow: 65_536,
            enablePrefixCaching: true,
            maxConcurrent: 4,
            defaultMaximumResponseTokens: 1_024
        )

        XCTAssertTrue(model.capabilities.contains(.reasoning))
        XCTAssertTrue(model.capabilities.contains(.toolCalling))
        XCTAssertEqual(model.defaultMaximumResponseTokens, 1_024)
        XCTAssertEqual(model.executorConfiguration.contextWindow, 65_536)
        XCTAssertTrue(model.executorConfiguration.enablePrefixCaching)
        XCTAssertEqual(model.executorConfiguration.maxConcurrent, 4)
    }

    private static func text(_ message: AFMMessage) -> String {
        message.content.compactMap { part in
            guard case .text(let value) = part else { return nil }
            return value
        }.joined()
    }
}
#endif
