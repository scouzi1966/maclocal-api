@testable import AFMKit
import XCTest

final class AFMKitConsumerContractTests: XCTestCase {
    private struct EchoModel: AFMModel {
        let descriptor = AFMModelDescriptor(
            providerID: "test.echo",
            modelID: "echo-1",
            displayName: "Echo 1",
            capabilities: [.text]
        )

        func availability() async -> AFMModelAvailability { .available }

        func load(
            progress: (@Sendable (Double) -> Void)?
        ) async throws -> AFMModelDescriptor {
            progress?(1)
            return descriptor
        }

        func respond(to request: AFMRequest) async throws -> AFMModelResponse {
            let text = request.messages
                .flatMap(\.content)
                .compactMap { content -> String? in
                    guard case .text(let text) = content else { return nil }
                    return text
                }
                .joined(separator: " ")
            return AFMModelResponse(text: text)
        }

        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { continuation in
                continuation.finish()
            }
        }
    }

    func testCompatibilityEngineUsesRegisteredAFMKitProvider() async throws {
        let registry = AFMProviderRegistry()
        try registry.register(
            AnyAFMProviderFactory(
                descriptor: AFMProviderDescriptor(id: "test.echo", displayName: "Echo"),
                modelDescriptors: { [EchoModel().descriptor] },
                makeModel: { modelID, _ in
                    guard modelID == "echo-1" else {
                        throw AFMError.modelNotFound(provider: "test.echo", model: modelID)
                    }
                    return AnyAFMModel(EchoModel())
                }
            )
        )
        let engine = try AFMEngine(
            providerID: "test.echo",
            modelID: "echo-1",
            registry: registry
        )

        let response = try await engine.respond(
            to: [Message(role: "user", content: "through engine")]
        )

        XCTAssertEqual(response.content, "through engine")
    }

    func testOpenAIDeveloperRoleMapsToProviderSystemRole() throws {
        let request = try AFMRequest(
            openAIMessages: [Message(role: "developer", content: "Follow project policy.")],
            generationConfig: GenerationConfig()
        )

        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages[0].role, .system)
        XCTAssertEqual(request.messages[0].content, [.text("Follow project policy.")])
    }
}
