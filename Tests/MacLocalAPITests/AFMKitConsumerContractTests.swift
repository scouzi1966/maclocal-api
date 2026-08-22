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

    private struct ReplacementModel: AFMModel {
        let descriptor = AFMModelDescriptor(
            providerID: "test.replace",
            modelID: "replace-1",
            displayName: "Replace 1",
            capabilities: [.text, .streaming]
        )
        func availability() async -> AFMModelAvailability { .available }
        func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
            descriptor
        }
        func respond(to request: AFMRequest) async throws -> AFMModelResponse { .init() }
        func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { continuation in
                continuation.yield(.responseText(action: .replace, text: "H", tokenCount: 1))
                continuation.yield(.responseText(action: .replace, text: "Hello", tokenCount: 2))
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

    func testLegacyEnginePreservesProviderConstructionError() async {
        let engine = AFMEngine(
            backend: .provider(providerID: "missing", modelID: "model")
        )
        do {
            _ = try await engine.load()
            XCTFail("Expected provider registration failure")
        } catch let error as AFMError {
            XCTAssertEqual(error, .providerNotRegistered("missing"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testLegacyStreamConvertsCumulativeReplacementsToDeltas() async throws {
        let registry = AFMProviderRegistry()
        try registry.register(AnyAFMProviderFactory(
            descriptor: .init(id: "test.replace", displayName: "Replace"),
            modelDescriptors: { [ReplacementModel().descriptor] },
            makeModel: { _, _ in AnyAFMModel(ReplacementModel()) }
        ))
        let engine = try AFMEngine(
            providerID: "test.replace",
            modelID: "replace-1",
            registry: registry
        )
        var output = ""
        for try await delta in engine.streamRespond(to: [.init(role: "user", content: "hi")]) {
            output += delta
        }
        XCTAssertEqual(output, "Hello")
    }
}
