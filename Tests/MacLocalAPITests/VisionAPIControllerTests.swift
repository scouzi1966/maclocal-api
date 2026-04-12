import XCTest
import Vapor
import XCTVapor

@testable import MacLocalAPI

final class VisionAPIControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testOCRReturnsStructuredDocumentPayload() async throws {
        let service = FakeVisionService()
        service.verboseResult = makeVisionResult(filePath: "/tmp/doc.png")
        try VisionAPIController(makeVisionService: { service }).boot(routes: app)

        let body = ByteBuffer(string: #"{"file":"~/doc.png","recognition_level":"fast","languages":["en-US"]}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/vision/ocr", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, #""page_count":2"#)
            XCTAssertContains(res.body.string, #""document_hints""#)
            XCTAssertContains(res.body.string, #""invoice""#)
            XCTAssertContains(res.body.string, #""headers":["Item","Amount"]"#)
            XCTAssertContains(res.body.string, #""row_objects""#)
            XCTAssertContains(res.body.string, #""Widget""#)
            XCTAssertContains(res.body.string, #""10.00""#)
            XCTAssertContains(res.body.string, #""source_type":"file""#)
            XCTAssertContains(res.body.string, #""combined_text":"Page 1"#)
            XCTAssertEqual(service.lastOptions?.recognitionLevel, .fast)
            XCTAssertEqual(service.lastOptions?.recognitionLanguages, ["en-US"])
        }
    }

    func testOCRAcceptsBase64Payload() async throws {
        let service = FakeVisionService()
        service.verboseResult = makeVisionResult(filePath: "/tmp/upload.png")
        try VisionAPIController(makeVisionService: { service }).boot(routes: app)

        let payload = Data("fake image".utf8).base64EncodedString()
        let body = ByteBuffer(string: #"{"data":"\#(payload)","filename":"scan.png"}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/vision/ocr", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, #""source_type":"data""#)
            XCTAssertEqual(service.lastPath?.hasSuffix(".png"), true)
        }
    }

    func testOCRAcceptsOpenAIMessageImageURL() async throws {
        let service = FakeVisionService()
        service.verboseResult = makeVisionResult(filePath: "/tmp/upload.png")
        try VisionAPIController(makeVisionService: { service }).boot(routes: app)

        let payload = Data("fake pdf".utf8).base64EncodedString()
        let body = ByteBuffer(string: """
        {
          "messages": [{
            "role": "user",
            "content": [
              {"type":"text","text":"read this"},
              {"type":"image_url","image_url":{"url":"data:application/pdf;base64,\(payload)"}}
            ]
          }]
        }
        """)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/vision/ocr", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .ok)
            XCTAssertContains(res.body.string, #""source_type":"message_image""#)
            XCTAssertEqual(service.lastPath?.hasSuffix(".pdf"), true)
        }
    }

    func testOCRReturnsPayloadTooLargeForOversizeInput() async throws {
        try VisionAPIController(makeVisionService: { FakeVisionService() }).boot(routes: app)

        let data = Data(count: VisionRequestOptions.defaultMaxFileBytes + 1).base64EncodedString()
        let body = ByteBuffer(string: #"{"data":"\#(data)","filename":"big.png"}"#)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(.POST, "/v1/vision/ocr", headers: headers, body: body) { res async in
            XCTAssertEqual(res.status, .payloadTooLarge)
        }
    }

    func testVisionOCRAutoToolDetection() {
        let tool = RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: "apple_vision_ocr",
                description: nil,
                parameters: nil,
                strict: nil
            )
        )
        let request = ChatCompletionRequest(
            model: "foundation",
            messages: [
                Message(
                    role: "user",
                    content: .parts([
                        ContentPart(type: "text", text: "scan this", image_url: nil),
                        ContentPart(type: "image_url", text: nil, image_url: ImageURL(url: "data:image/png;base64,ZmFrZQ==", detail: nil))
                    ])
                )
            ],
            temperature: nil,
            maxTokens: nil,
            maxCompletionTokens: nil,
            topP: nil,
            repetitionPenalty: nil,
            repeatPenalty: nil,
            frequencyPenalty: nil,
            presencePenalty: nil,
            topK: nil,
            minP: nil,
            seed: nil,
            logprobs: nil,
            topLogprobs: nil,
            stop: nil,
            stream: nil,
            user: nil,
            tools: [tool],
            toolChoice: .mode("auto"),
            responseFormat: nil,
            chatTemplateKwargs: nil
        )

        XCTAssertTrue(VisionAPIController.shouldAutoRunVisionTool(request))
    }
}

private final class FakeVisionService: VisionServing {
    var lastPath: String?
    var lastOptions: VisionRequestOptions?
    var textResult = ""
    var verboseResult = VisionResult(fullText: "", textBlocks: [], filePath: "")
    var tablesResult: [TableResult] = []
    var debugResult = ""
    var textError: Error?
    var verboseError: Error?
    var tablesError: Error?
    var debugError: Error?

    func extractText(from filePath: String) async throws -> String {
        lastPath = filePath
        if let textError { throw textError }
        return textResult
    }

    func extractText(from filePath: String, options: VisionRequestOptions) async throws -> String {
        lastPath = filePath
        lastOptions = options
        if let textError { throw textError }
        return textResult
    }

    func extractTextWithDetails(from filePath: String) async throws -> VisionResult {
        lastPath = filePath
        if let verboseError { throw verboseError }
        return verboseResult
    }

    func extractTextWithDetails(from filePath: String, options: VisionRequestOptions) async throws -> VisionResult {
        lastPath = filePath
        lastOptions = options
        if let verboseError { throw verboseError }
        return verboseResult
    }

    func extractTables(from filePath: String) async throws -> [TableResult] {
        lastPath = filePath
        if let tablesError { throw tablesError }
        return tablesResult
    }

    func extractTables(from filePath: String, options: VisionRequestOptions) async throws -> [TableResult] {
        lastPath = filePath
        lastOptions = options
        if let tablesError { throw tablesError }
        return tablesResult
    }

    func debugRawDetection(from filePath: String) async throws -> String {
        lastPath = filePath
        if let debugError { throw debugError }
        return debugResult
    }

    func debugRawDetection(from filePath: String, options: VisionRequestOptions) async throws -> String {
        lastPath = filePath
        lastOptions = options
        if let debugError { throw debugError }
        return debugResult
    }
}

private func makeVisionResult(filePath: String) -> VisionResult {
    let table = TableResult(
        rows: [["Item", "Amount"], ["Widget", "10.00"]],
        columnCount: 2,
        averageConfidence: 0.93,
        boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.3),
        pageNumber: 2,
        headers: ["Item", "Amount"],
        rowObjects: [["Item": "Widget", "Amount": "10.00"]],
        mergedCellHints: ["row_1_contains_sparse_cells"]
    )
    let pageOne = VisionPageResult(
        pageNumber: 1,
        fullText: "Page 1",
        textBlocks: [
            TextBlock(text: "Page 1", confidence: 0.99, boundingBox: CGRect(x: 0, y: 0, width: 1, height: 0.2), pageNumber: 1)
        ],
        tables: [],
        width: 1024,
        height: 768
    )
    let pageTwo = VisionPageResult(
        pageNumber: 2,
        fullText: "Page 2",
        textBlocks: [
            TextBlock(text: "Widget", confidence: 0.96, boundingBox: CGRect(x: 0.2, y: 0.3, width: 0.2, height: 0.1), pageNumber: 2)
        ],
        tables: [table],
        width: 1024,
        height: 768
    )
    return VisionResult(
        fullText: "Page 1\n\nPage 2",
        textBlocks: pageOne.textBlocks + pageTwo.textBlocks,
        filePath: filePath,
        pages: [pageOne, pageTwo],
        documentHints: ["invoice", "multi_page", "table_like"]
    )
}
