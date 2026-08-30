import AFMKitCore
import AFMKitMLXImage
import XCTest
import Vapor
import XCTVapor

@testable import AFMServer

final class ImagesAPIControllerTests: XCTestCase {
    private var app: Application!

    override func setUp() async throws {
        app = try await Application.make(.testing)
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testGenerationReturnsBase64ImagePayload() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"model":"any-model","prompt":"a red square","n":1,"size":"256x256"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            let decoded = try! JSONDecoder().decode(OpenAIImagesResponse.self, from: Data(buffer: response.body))
            XCTAssertEqual(decoded.data.first?.b64Json, Data("png".utf8).base64EncodedString())
        }
        let request = await generator.generationRequest
        XCTAssertEqual(request?.prompt, "a red square")
        XCTAssertEqual(request?.width, 256)
        XCTAssertEqual(request?.height, 256)
    }

    func testEditAcceptsMultipartModelFieldAndImage() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        let boundary = "afm-image-boundary"
        let png = Data([0x89, 0x50, 0x4e, 0x47])
        var body = Data()
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nQwen3.8-27B\r\n".utf8))
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake it blue\r\n".utf8))
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\n".utf8))
        body.append(png)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var headers = HTTPHeaders()
        headers.contentType = HTTPMediaType(type: "multipart", subType: "form-data", parameters: ["boundary": boundary])

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/edits", headers: headers, body: ByteBuffer(data: body)
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertContains(response.body.string, "b64_json")
        }
        let request = await generator.editRequest
        XCTAssertEqual(request?.prompt, "make it blue")
        XCTAssertEqual(request?.images, [png])
    }

    func testInvalidSizeIsRejectedBeforeGeneration() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"test","size":"nope"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
        }
        let request = await generator.generationRequest
        XCTAssertNil(request)
    }

    func testFluxKleinGenerationAndEditIntegration() async throws {
        guard ProcessInfo.processInfo.environment["AFM_RUN_FLUX_INTEGRATION"] == "1" else {
            throw XCTSkip("Set AFM_RUN_FLUX_INTEGRATION=1 to run the 23.7 GB FLUX model smoke test")
        }
        let cachePath = ProcessInfo.processInfo.environment["MACAFM_MLX_MODEL_CACHE"]
            ?? "/Volumes/Crucial4TB/models/vesta-test-cache"
        let generator = FluxKleinImageService(configuration: FluxKleinImageConfiguration(
            cacheDirectory: URL(fileURLWithPath: cachePath, isDirectory: true),
            quantization: .int4
        ))
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"model":"Qwen3.8-27B","prompt":"a red square on a white background","n":1,"size":"256x256","response_format":"b64_json","seed":42}"#)

        var generatedPNG = Data()
        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async throws in
            XCTAssertEqual(response.status, .ok)
            let decoded = try JSONDecoder().decode(OpenAIImagesResponse.self, from: Data(buffer: response.body))
            let png = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(decoded.data.first?.b64Json)))
            XCTAssertEqual(Array(png.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])
            try png.write(to: URL(fileURLWithPath: "/tmp/afm-flux-klein-smoke.png"), options: .atomic)
            generatedPNG = png
        }

        let boundary = "afm-flux-integration-boundary"
        var editBody = Data()
        editBody.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nQwen3.8-27B\r\n".utf8))
        editBody.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nturn the red square into a blue circle on a white background\r\n".utf8))
        editBody.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n256x256\r\n".utf8))
        editBody.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"generated.png\"\r\nContent-Type: image/png\r\n\r\n".utf8))
        editBody.append(generatedPNG)
        editBody.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var editHeaders = HTTPHeaders()
        editHeaders.contentType = HTTPMediaType(type: "multipart", subType: "form-data", parameters: ["boundary": boundary])

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/edits", headers: editHeaders, body: ByteBuffer(data: editBody)
        ) { response async throws in
            XCTAssertEqual(response.status, .ok)
            let decoded = try JSONDecoder().decode(OpenAIImagesResponse.self, from: Data(buffer: response.body))
            let png = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(decoded.data.first?.b64Json)))
            XCTAssertEqual(Array(png.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])
            try png.write(to: URL(fileURLWithPath: "/tmp/afm-flux-klein-edit-smoke.png"), options: .atomic)
        }
    }
}

private actor FakeImageGenerator: AFMImageGenerating {
    var generationRequest: AFMImageGenerationRequest?
    var editRequest: AFMImageEditRequest?

    func generateImages(for request: AFMImageGenerationRequest) async throws -> [AFMGeneratedImage] {
        generationRequest = request
        return [AFMGeneratedImage(data: Data("png".utf8), width: request.width, height: request.height)]
    }

    func editImages(for request: AFMImageEditRequest) async throws -> [AFMGeneratedImage] {
        editRequest = request
        return [AFMGeneratedImage(data: Data("edited".utf8), width: request.width, height: request.height)]
    }
}
