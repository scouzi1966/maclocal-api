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
            XCTAssertContains(response.body.string, #""object":"list""#)
            XCTAssertContains(response.body.string, #""embedding":[0.26726124,0.5345225,0.8017837]"#)
            XCTAssertContains(response.body.string, #""prompt_tokens":4"#)
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
            XCTAssertContains(response.body.string, #""embedding":[0.6,0.8]"#)
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
