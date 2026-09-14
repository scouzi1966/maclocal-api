import AFMKit
@testable import AFMServer
import Vapor
import XCTVapor
import XCTest

/// Exercise the actual HTTP controller and generic provider adapter without
/// consuming PCC quota or requiring a development signing identity.
final class PCCHTTPTests: XCTestCase {
    private struct FixtureModel: AFMModel {
        let failure: AFMError?
        let descriptor = AFMModelDescriptor(
            providerID: "apple.foundation-models", modelID: "apple.private-cloud-compute",
            displayName: "PCC fixture", capabilities: [.text, .streaming])
        func availability() async -> AFMModelAvailability { .available }
        func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor { descriptor }
        func respond(to request: AFMRequest) async throws -> AFMModelResponse {
            if let failure { throw failure }
            return .init(text: "PCC fixture response")
        }
        func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
            AsyncThrowingStream { continuation in
                if let failure {
                    continuation.finish(throwing: failure)
                } else {
                    continuation.yield(.responseText(action: .append, text: "PCC fixture response", tokenCount: 3))
                    continuation.yield(.completed(.stop))
                    continuation.finish()
                }
            }
        }
    }

    private func request(streaming: Bool, failure: AFMError? = nil, status: HTTPStatus, contains: String) async throws {
        let app = try await Application.make(.testing)
        do {
            let model = FixtureModel(failure: failure)
            let adapter = AFMKitMLXChatServingAdapter(model: AnyAFMModel(model), modelID: model.descriptor.modelID.rawValue)
            try MLXChatCompletionsController(
                modelID: model.descriptor.modelID.rawValue, service: adapter,
                temperature: nil, repetitionPenalty: nil).boot(routes: app)
            let tester = try app.testable()
            var headers = HTTPHeaders()
            headers.contentType = .json
            let body = ByteBuffer(string: """
            {"model":"apple.private-cloud-compute","stream":\(streaming),"messages":[{"role":"user","content":"hello"}]}
            """)
            try await tester.test(.POST, "/v1/chat/completions", headers: headers, body: body) { response async in
                XCTAssertEqual(response.status, status)
                XCTAssertTrue(response.body.string.contains(contains), response.body.string)
            }
            try await app.asyncShutdown()
        } catch {
            try await app.asyncShutdown()
            throw error
        }
    }

    func testPCCProviderRespondsThroughChatCompletions() async throws {
        try await request(streaming: false, status: .ok, contains: "PCC fixture response")
    }

    func testPCCProviderStreamsThroughChatCompletions() async throws {
        try await request(streaming: true, status: .ok, contains: "PCC fixture response")
    }

    func testProviderUnavailableReturns503InsteadOfBadRequest() async throws {
        try await request(streaming: false, failure: .unavailable("PCC quota reached"),
                          status: .serviceUnavailable, contains: "provider_unavailable")
    }

    func testProviderGenerationFailureReturns502() async throws {
        try await request(streaming: false, failure: .generationFailed("PCC connection lost"),
                          status: .badGateway, contains: "provider_error")
    }
}
