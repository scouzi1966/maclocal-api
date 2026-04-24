import XCTest
import Vapor
import XCTVapor

@testable import MacLocalAPI

final class EmbeddingsControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testSingleStringInputReturnsOpenAIShape() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 3,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 2, 3]], tokenCounts: [4])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hello world","model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .ok)
            do {
                let decoded = try Self.decodeEmbeddingsResponse(response.body.string)
                XCTAssertEqual(decoded.object, "list")
                XCTAssertEqual(decoded.usage.promptTokens, 4)
                XCTAssertEqual(decoded.data.count, 1)
                let vector = decoded.data.first?.embedding ?? []
                Self.assertApproximatelyEqual(vector, [0.26726124, 0.5345225, 0.8017837])
            } catch {
                XCTFail("failed to decode response: \(error)\nbody: \(response.body.string)")
            }
        }
    }

    func testArrayInputReturnsOneEmbeddingPerInput() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0], [0, 1]], tokenCounts: [2, 3])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":["a","b"],"model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertContains(response.body.string, #""index":0"#)
            XCTAssertContains(response.body.string, #""index":1"#)
            XCTAssertContains(response.body.string, #""total_tokens":5"#)
        }
    }

    func testBase64EncodingReturnsStringPayload() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hi","model":"apple-nl-contextual-en","encoding_format":"base64"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertContains(response.body.string, #""embedding":"AACAPwAAAAA=""#)
        }
    }

    func testDimensionsTruncateAndRenormalize() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 3,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[3, 4, 12]], tokenCounts: [3])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hi","model":"apple-nl-contextual-en","dimensions":2}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .ok)
            do {
                let decoded = try Self.decodeEmbeddingsResponse(response.body.string)
                let vector = decoded.data.first?.embedding ?? []
                Self.assertApproximatelyEqual(vector, [0.6, 0.8])
            } catch {
                XCTFail("failed to decode response: \(error)\nbody: \(response.body.string)")
            }
        }
    }

    func testDimensionsAboveNativeReturn400() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hi","model":"apple-nl-contextual-en","dimensions":3}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .badRequest)
        }
    }

    func testUnknownModelReturns404() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hi","model":"other-model"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .notFound)
        }
    }

    func testMissingInputFieldReturns400() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .badRequest)
            XCTAssertContains(response.body.string, "input")
        }
    }

    func testUnknownEncodingFormatReturns400() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hi","model":"apple-nl-contextual-en","encoding_format":"bogus"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .badRequest)
        }
    }

    func testMalformedJSONBodyReturns400() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input": "hi", "#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .badRequest)
        }
    }

    func testEmptyInputReturns400() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"","model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .badRequest)
        }
    }

    func testListModelsReturnsOnlyLoadedModel() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        try await app.testable(method: .running(port: 0)).test(.GET, "/v1/models") { response async in
            XCTAssertEqual(response.status, .ok)
            struct ModelsPayload: Decodable {
                struct Entry: Decodable { let id: String }
                let object: String
                let data: [Entry]
            }
            do {
                let decoded = try JSONDecoder().decode(ModelsPayload.self, from: Data(response.body.readableBytesView))
                XCTAssertEqual(decoded.object, "list")
                XCTAssertEqual(decoded.data.map(\.id), ["apple-nl-contextual-en"])
            } catch {
                XCTFail("failed to decode /v1/models: \(error)\nbody: \(response.body.string)")
            }
        }
    }

    func testCORSPreflightReflectsRequestedHeaders() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.add(name: "Origin", value: "https://example.com")
        headers.add(name: "Access-Control-Request-Method", value: "POST")
        headers.add(name: "Access-Control-Request-Headers", value: "Content-Type, Authorization, x-stainless-arch, OpenAI-Organization")

        try await app.testable(method: .running(port: 0)).test(.OPTIONS, "/v1/embeddings", headers: headers) { response async in
            XCTAssertEqual(response.status, .ok)
            let allow = response.headers.first(name: .accessControlAllowHeaders) ?? ""
            XCTAssertContains(allow, "x-stainless-arch")
            XCTAssertContains(allow, "OpenAI-Organization")
        }
    }

    func testUnsupportedMediaTypePreservesStatus() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.add(name: .contentType, value: "application/x-not-a-real-type")
        let body = ByteBuffer(string: #"{"input":"hi","model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .unsupportedMediaType)
            struct ErrorPayload: Decodable {
                struct Inner: Decodable { let message: String; let type: String }
                let error: Inner
            }
            do {
                let decoded = try JSONDecoder().decode(ErrorPayload.self, from: Data(response.body.readableBytesView))
                XCTAssertEqual(decoded.error.type, "embedding_error")
                XCTAssertFalse(decoded.error.message.isEmpty)
            } catch {
                XCTFail("failed to decode 415 error body: \(error)\nbody: \(response.body.string)")
            }
        }
    }

    func testCORSPreflightOnModelsReturns200WithHeaders() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.add(name: "Origin", value: "https://example.com")
        headers.add(name: "Access-Control-Request-Method", value: "GET")
        headers.add(name: "Access-Control-Request-Headers", value: "Authorization")

        try await app.testable(method: .running(port: 0)).test(.OPTIONS, "/v1/models", headers: headers) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertEqual(response.headers.first(name: .accessControlAllowOrigin), "*")
            XCTAssertContains(response.headers.first(name: .accessControlAllowMethods) ?? "", "GET")
            XCTAssertContains(response.headers.first(name: .accessControlAllowHeaders) ?? "", "Authorization")
        }
    }

    func testOversizedPayloadReturnsEmbeddingsSpecificError() async throws {
        app.middleware.use(EmbeddingsPayloadTooLargeMiddleware())
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let oversized = String(repeating: "a", count: 2 * 1024 * 1024)
        let body = ByteBuffer(string: #"{"input":""# + oversized + #"","model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .payloadTooLarge)
            XCTAssertContains(response.body.string, "Embeddings request body")
            XCTAssertFalse(response.body.string.contains("conversation"))
        }
    }

    func testTruncationHeaderIsReturned() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [1], truncatedInputCount: 1)
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":"hi","model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertEqual(response.headers.first(name: "X-Embedding-Truncated"), "1")
        }
    }

    // Routing contract: a tokenized request reaches the backend's
    // `embedTokenIDs` path. Backends that do not support token-ID input
    // (e.g. the Apple NL backend) are expected to reject this at their own
    // layer; see `testTokenArrayInputRejectedByAppleLikeBackend` below.
    func testTokenArrayInputRoutesToTokenizedBackendPath() async throws {
        let backend = FakeEmbeddingBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128,
            result: EmbedResult(vectors: [[1, 0]], tokenCounts: [3])
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":[1,2,3],"model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .ok)
            let lastTokenIDs = await backend.lastTokenIDInputs
            XCTAssertEqual(lastTokenIDs, [[1, 2, 3]])
        }
    }

    func testTokenArrayInputRejectedByAppleLikeBackend() async throws {
        let backend = TokenIDsNotSupportedBackend(
            modelID: "apple-nl-contextual-en",
            nativeDimension: 2,
            maxInputTokens: 128
        )
        try EmbeddingsController(modelEntry: makeEntry(id: "apple-nl-contextual-en"), backend: backend).boot(routes: app)

        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"input":[1,2,3],"model":"apple-nl-contextual-en"}"#)

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/embeddings", headers: headers, body: body) { response async in
            XCTAssertEqual(response.status, .badRequest)
            XCTAssertContains(response.body.string, "Pre-tokenized input is not supported")
        }
    }

    private struct DecodedResponse: Decodable {
        struct Datum: Decodable {
            let index: Int
            let embedding: [Float]
        }
        struct Usage: Decodable {
            let promptTokens: Int
            let totalTokens: Int
            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case totalTokens = "total_tokens"
            }
        }
        let object: String
        let model: String
        let data: [Datum]
        let usage: Usage
    }

    private static func decodeEmbeddingsResponse(_ body: String) throws -> DecodedResponse {
        let data = Data(body.utf8)
        return try JSONDecoder().decode(DecodedResponse.self, from: data)
    }

    private static func assertApproximatelyEqual(
        _ actual: [Float],
        _ expected: [Float],
        tolerance: Float = 1e-5,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "vector length", file: file, line: line)
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a, e, accuracy: tolerance, file: file, line: line)
        }
    }

    private func makeEntry(id: String) -> EmbeddingModelEntry {
        EmbeddingModelEntry(
            id: id,
            backend: .nlContextual,
            nativeDimension: 3,
            supportsMatryoshka: false,
            pooling: .mean,
            normalized: true,
            maxInputTokens: 128,
            description: "test"
        )
    }
}

private actor TokenIDsNotSupportedBackend: EmbeddingBackend {
    let modelID: String
    let nativeDimension: Int
    let maxInputTokens: Int

    init(modelID: String, nativeDimension: Int, maxInputTokens: Int) {
        self.modelID = modelID
        self.nativeDimension = nativeDimension
        self.maxInputTokens = maxInputTokens
    }

    func embed(_ inputs: [String]) async throws -> EmbedResult {
        throw EmbeddingError.invalidInput("Text input not exercised in this test")
    }

    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult {
        throw EmbeddingError.invalidInput("Pre-tokenized input is not supported by Apple NL embeddings")
    }
}

private actor FakeEmbeddingBackend: EmbeddingBackend {
    let modelID: String
    let nativeDimension: Int
    let maxInputTokens: Int

    private let result: EmbedResult
    private(set) var lastStringInputs: [String] = []
    private(set) var lastTokenIDInputs: [[Int]] = []

    init(modelID: String, nativeDimension: Int, maxInputTokens: Int, result: EmbedResult) {
        self.modelID = modelID
        self.nativeDimension = nativeDimension
        self.maxInputTokens = maxInputTokens
        self.result = result
    }

    func embed(_ inputs: [String]) async throws -> EmbedResult {
        lastStringInputs = inputs
        lastTokenIDInputs = []
        return result
    }

    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult {
        lastStringInputs = []
        lastTokenIDInputs = inputs
        return result
    }
}
