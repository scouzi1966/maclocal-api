@testable import AFMKit
import Foundation
import XCTest

final class AFMProviderRegistryTests: XCTestCase {
    private final class DescriptorState: @unchecked Sendable {
        private let lock = NSLock()
        private var value: AFMModelDescriptor

        init(_ value: AFMModelDescriptor) { self.value = value }

        func get() -> AFMModelDescriptor { lock.withLock { value } }
        func set(_ value: AFMModelDescriptor) { lock.withLock { self.value = value } }
    }

    private struct RuntimeQualifiedModel: AFMModel {
        let state: DescriptorState
        var descriptor: AFMModelDescriptor { state.get() }

        func availability() async -> AFMModelAvailability { .available }
        func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
            descriptor
        }
        func respond(to request: AFMRequest) async throws -> AFMModelResponse {
            AFMModelResponse(text: "ok")
        }
        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { $0.finish() }
        }
    }

    private struct EchoModel: AFMModel {
        let descriptor: AFMModelDescriptor

        func availability() async -> AFMModelAvailability {
            .available
        }

        func load(
            progress: (@Sendable (Double) -> Void)?
        ) async throws -> AFMModelDescriptor {
            progress?(1)
            return descriptor
        }

        func respond(to request: AFMRequest) async throws -> AFMModelResponse {
            let text = request.messages.compactMap { message in
                message.content.compactMap { part -> String? in
                    guard case .text(let value) = part else { return nil }
                    return value
                }
                .joined()
            }
            .joined(separator: " ")
            return AFMModelResponse(text: text)
        }

        func streamResponse(
            to request: AFMRequest
        ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { continuation in
                continuation.yield(
                    .responseText(action: .append, text: "echo", tokenCount: 1)
                )
                continuation.yield(.usage(.init(inputTokens: 1, outputTokens: 1)))
                continuation.yield(.completed(.stop))
                continuation.finish()
            }
        }
    }

    private func makeFactory(
        providerID: AFMProviderID = "test.echo"
    ) -> AnyAFMProviderFactory {
        let descriptor = AFMProviderDescriptor(
            id: providerID,
            displayName: "Echo"
        )
        return AnyAFMProviderFactory(
            descriptor: descriptor,
            modelDescriptors: {
                [
                    AFMModelDescriptor(
                        providerID: providerID,
                        modelID: "echo-1",
                        displayName: "Echo 1",
                        capabilities: [.text, .streaming],
                        privacyBoundary: .device,
                        requiresNetwork: false
                    )
                ]
            },
            makeModel: { modelID, _ in
                guard modelID == "echo-1" else {
                    throw AFMError.modelNotFound(provider: providerID, model: modelID)
                }
                return AnyAFMModel(
                    EchoModel(
                        descriptor: AFMModelDescriptor(
                            providerID: providerID,
                            modelID: modelID,
                            displayName: "Echo 1",
                            capabilities: [.text, .streaming]
                        )
                    )
                )
            }
        )
    }

    func testCustomProviderRegistersAndCreatesModelWithoutEngineSwitch() async throws {
        let registry = AFMProviderRegistry()
        try registry.register(makeFactory())

        let model = try registry.makeModel(
            providerID: "test.echo",
            modelID: "echo-1"
        )
        let response = try await model.respond(
            to: AFMRequest(messages: [.init(role: .user, text: "hello")])
        )

        XCTAssertEqual(response.text, "hello")
        XCTAssertEqual(registry.providerDescriptors().map(\.id), ["test.echo"])
    }

    func testTypeErasedModelUsesRuntimeQualifiedDescriptor() {
        let declared = AFMModelDescriptor(
            providerID: "mlx",
            modelID: "qualified",
            displayName: "Qualified",
            capabilities: [.text, .vision]
        )
        let qualified = AFMModelDescriptor(
            providerID: "mlx",
            modelID: "qualified",
            displayName: "Qualified",
            capabilities: [.text]
        )
        let state = DescriptorState(declared)
        let model = AnyAFMModel(RuntimeQualifiedModel(state: state))

        XCTAssertTrue(model.descriptor.capabilities.contains(.vision))
        state.set(qualified)
        XCTAssertFalse(model.descriptor.capabilities.contains(.vision))
    }

    func testDuplicateRegistrationFailsWithTypedError() throws {
        let registry = AFMProviderRegistry()
        try registry.register(makeFactory())

        XCTAssertThrowsError(try registry.register(makeFactory())) { error in
            XCTAssertEqual(
                error as? AFMError,
                .providerAlreadyRegistered("test.echo")
            )
        }
    }

    func testCompatibilityEngineUsesRegisteredProvider() async throws {
        let registry = AFMProviderRegistry()
        try registry.register(makeFactory())
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

    func testPortableEventStreamCarriesTextUsageAndCompletion() async throws {
        let registry = AFMProviderRegistry()
        try registry.register(makeFactory())
        let model = try registry.makeModel(
            providerID: "test.echo",
            modelID: "echo-1"
        )

        var events: [AFMGenerationEvent] = []
        for try await event in model.streamResponse(
            to: AFMRequest(messages: [.init(role: .user, text: "hello")])
        ) {
            events.append(event)
        }

        XCTAssertEqual(
            events,
            [
                .responseText(action: .append, text: "echo", tokenCount: 1),
                .usage(.init(inputTokens: 1, outputTokens: 1)),
                .completed(.stop)
            ]
        )
    }
}
