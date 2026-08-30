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
        app.middleware.use(RequestIDMiddleware())
    }

    override func tearDown() async throws {
        try await app.asyncShutdown()
    }

    func testGenerationReturnsBase64ImagePayload() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator, modelAliases: ["any-model"]).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"model":"any-model","prompt":"a red square","n":1,"size":"256x256"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            let decoded = try! JSONDecoder().decode(OpenAIImagesResponse.self, from: Data(buffer: response.body))
            XCTAssertEqual(decoded.data.first?.b64Json, Data("png".utf8).base64EncodedString())
            XCTAssertEqual(response.headers.first(name: "X-AFM-Compatibility"), "emulated")
            XCTAssertEqual(response.headers.first(name: "X-AFM-Emulated-Parameters"), "model")
        }
        let request = await generator.generationRequest
        XCTAssertEqual(request?.prompt, "a red square")
        XCTAssertEqual(request?.width, 256)
        XCTAssertEqual(request?.height, 256)
    }

    func testEditAcceptsMultipartModelFieldAndImage() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator, modelAliases: ["Qwen3.8-27B"]).boot(routes: app)
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
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["type"] as? String, "invalid_request_error")
            XCTAssertEqual(error["code"] as? String, "invalid_request_error")
            XCTAssertEqual(error["param"] as? String, "size")
            XCTAssertNotNil(error["request_id"] as? String)
            XCTAssertNotNil(response.headers.first(name: "x-request-id"))
        }
        let request = await generator.generationRequest
        XCTAssertNil(request)
    }

    func testNonAlignedSizeIsRejectedInsteadOfSilentlyRounded() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/images/generations",
            headers: headers,
            body: ByteBuffer(string: #"{"prompt":"test","size":"257x256"}"#)
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["param"] as? String, "size")
        }
        let request = await generator.generationRequest
        XCTAssertNil(request)
    }

    func testUnknownImageModelIsRejected() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator, modelID: "local-image-model").boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/images/generations",
            headers: headers,
            body: ByteBuffer(string: #"{"model":"chat-model","prompt":"test"}"#)
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unsupported_model")
            XCTAssertEqual(error["param"] as? String, "model")
        }
    }

    func testPromptOverOpenAILimitIsRejected() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let data = try JSONSerialization.data(withJSONObject: ["prompt": String(repeating: "a", count: 32_001)])

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: ByteBuffer(data: data)
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["param"] as? String, "prompt")
        }
    }

    func testOversizedGenerationBodyReturnsRouteAppropriate413() async throws {
        app.middleware.use(PayloadTooLargeMiddleware())
        try ImagesAPIController(generator: FakeImageGenerator()).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"\#(String(repeating: "a", count: 1_100_000))"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .payloadTooLarge)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "payload_too_large")
            XCTAssertContains(error["message"] as? String ?? "", "Image request")
            XCTAssertEqual(response.headers.first(name: .cacheControl), "no-store")
            XCTAssertContains(response.headers.first(name: "Access-Control-Expose-Headers") ?? "", "X-Request-ID")
        }
    }

    func testExactImageModelDoesNotReportEmulation() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator, modelID: "local-image-model").boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/images/generations",
            headers: headers,
            body: ByteBuffer(string: #"{"model":"local-image-model","prompt":"test"}"#)
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertNil(response.headers.first(name: "X-AFM-Compatibility"))
        }
    }

    func testUnsupportedGenerationControlReturnsParameterSpecificOpenAIError() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"test","background":"transparent"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["type"] as? String, "invalid_request_error")
            XCTAssertEqual(error["code"] as? String, "unsupported_parameter")
            XCTAssertEqual(error["param"] as? String, "background")
            XCTAssertContains(error["message"] as? String ?? "", "not supported")
        }
        let request = await generator.generationRequest
        XCTAssertNil(request)
    }

    func testSupportedExplicitPNGAndNonStreamingControlsAreAccepted() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"test","output_format":"png","response_format":"b64_json","stream":false}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertNil(response.headers.first(name: "X-AFM-Compatibility"))
        }
    }

    func testStreamingRequestIsRejectedAsNonRetryableClientError() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"test","stream":true}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unsupported_parameter")
            XCTAssertEqual(error["param"] as? String, "stream")
        }
    }

    func testWrongGenerationContentTypeReturns415OpenAIError() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .plainText

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: ByteBuffer(string: "test")
        ) { response async in
            XCTAssertEqual(response.status, .unsupportedMediaType)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unsupported_media_type")
        }
    }

    func testUnknownGenerationParameterIsNotSilentlyIgnored() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"test","qualitty":"high"}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unknown_parameter")
            XCTAssertEqual(error["param"] as? String, "qualitty")
        }
    }

    func testUnsupportedMethodReturns405WithAllowHeader() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)

        try await app.testable(method: .running(port: 0)).test(
            .GET, "/v1/images/generations"
        ) { response async in
            XCTAssertEqual(response.status, .methodNotAllowed)
            XCTAssertEqual(response.headers.first(name: "Allow"), "POST, OPTIONS")
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "method_not_allowed")
        }
    }

    func testHeadReturns405InsteadOfFallingThroughTo404() async throws {
        try ImagesAPIController(generator: FakeImageGenerator()).boot(routes: app)

        try await app.testable(method: .running(port: 0)).test(
            .HEAD, "/v1/images/generations"
        ) { response async in
            XCTAssertEqual(response.status, .methodNotAllowed)
            XCTAssertEqual(response.headers.first(name: "Allow"), "POST, OPTIONS")
        }
    }

    func testCORSPreflightReflectsSDKHeadersAndVaries() async throws {
        try ImagesAPIController(generator: FakeImageGenerator()).boot(routes: app)
        var headers = HTTPHeaders()
        headers.add(name: "Access-Control-Request-Headers", value: "Content-Type, x-stainless-arch, OpenAI-Project")

        try await app.testable(method: .running(port: 0)).test(
            .OPTIONS, "/v1/images/generations", headers: headers
        ) { response async in
            XCTAssertEqual(response.status, .ok)
            XCTAssertEqual(
                response.headers.first(name: .accessControlAllowHeaders),
                "Content-Type, x-stainless-arch, OpenAI-Project"
            )
            XCTAssertEqual(response.headers.first(name: .vary), "Access-Control-Request-Headers")
        }
    }

    func testInvalidProviderImageReturnsParameterSpecific400() async throws {
        let generator = FakeImageGenerator(editFailure: .invalidImage)
        try ImagesAPIController(generator: generator).boot(routes: app)
        let request = Self.multipartEditBody(image: Data("not-an-image".utf8), filename: "input.png")

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/edits", headers: request.headers, body: request.body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "invalid_image")
            XCTAssertEqual(error["param"] as? String, "image")
        }
    }

    func testEditRejectsUnsupportedImageFileTypeBeforeProviderCall() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        let request = Self.multipartEditBody(image: Data("image".utf8), filename: "input.gif")

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/edits", headers: request.headers, body: request.body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["param"] as? String, "image")
        }
        let providerRequest = await generator.editRequest
        XCTAssertNil(providerRequest)
    }

    func testUnavailableImageSnapshotReturnsRetryable503() async throws {
        let generator = FakeImageGenerator(generationFailure: .unreadableSnapshot)
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/images/generations",
            headers: headers,
            body: ByteBuffer(string: #"{"prompt":"test"}"#)
        ) { response async in
            XCTAssertEqual(response.status, .serviceUnavailable)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["type"] as? String, "server_error")
            XCTAssertEqual(error["code"] as? String, "image_model_unavailable")
        }
    }

    func testUnexpectedProviderFailureReturnsShaped500() async throws {
        let generator = FakeImageGenerator(generationFailure: .unexpected)
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json

        try await app.testable(method: .running(port: 0)).test(
            .POST,
            "/v1/images/generations",
            headers: headers,
            body: ByteBuffer(string: #"{"prompt":"test"}"#)
        ) { response async in
            XCTAssertEqual(response.status, .internalServerError)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["type"] as? String, "server_error")
            XCTAssertEqual(error["code"] as? String, "image_generation_failed")
            XCTAssertNotNil(error["request_id"] as? String)
        }
    }

    func testStandardCountBeyondLocalLimitIsUnsupportedRatherThanInvalid() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        var headers = HTTPHeaders()
        headers.contentType = .json
        let body = ByteBuffer(string: #"{"prompt":"test","n":5}"#)

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/generations", headers: headers, body: body
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unsupported_parameter")
            XCTAssertEqual(error["param"] as? String, "n")
        }
    }

    func testUnsupportedEditMaskReturnsParameterSpecificOpenAIError() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        let boundary = "afm-image-mask-boundary"
        let png = Data([0x89, 0x50, 0x4e, 0x47])
        var body = Data()
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit it\r\n".utf8))
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\n".utf8))
        body.append(png)
        body.append(Data("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"\r\nContent-Type: image/png\r\n\r\n".utf8))
        body.append(png)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var headers = HTTPHeaders()
        headers.contentType = HTTPMediaType(type: "multipart", subType: "form-data", parameters: ["boundary": boundary])

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/edits", headers: headers, body: ByteBuffer(data: body)
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unsupported_parameter")
            XCTAssertEqual(error["param"] as? String, "mask")
        }
    }

    func testEditImageArraySyntaxIsRecognizedAndRejectedExplicitly() async throws {
        let generator = FakeImageGenerator()
        try ImagesAPIController(generator: generator).boot(routes: app)
        let boundary = "afm-image-array-boundary"
        var body = Data()
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit it\r\n".utf8))
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\nPNG\r\n--\(boundary)--\r\n".utf8))
        var headers = HTTPHeaders()
        headers.contentType = HTTPMediaType(type: "multipart", subType: "form-data", parameters: ["boundary": boundary])

        try await app.testable(method: .running(port: 0)).test(
            .POST, "/v1/images/edits", headers: headers, body: ByteBuffer(data: body)
        ) { response async in
            XCTAssertEqual(response.status, .badRequest)
            let error = try! XCTUnwrap(Self.errorDetail(response.body))
            XCTAssertEqual(error["code"] as? String, "unsupported_parameter")
            XCTAssertEqual(error["param"] as? String, "image")
        }
    }

    private static func errorDetail(_ body: ByteBuffer) -> [String: Any]? {
        guard let object = try? JSONSerialization.jsonObject(with: Data(buffer: body)) as? [String: Any] else {
            return nil
        }
        return object["error"] as? [String: Any]
    }

    private static func multipartEditBody(image: Data, filename: String) -> (headers: HTTPHeaders, body: ByteBuffer) {
        let boundary = "afm-image-test-boundary"
        var data = Data()
        data.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nedit it\r\n".utf8))
        data.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"\(filename)\"\r\nContent-Type: image/png\r\n\r\n".utf8))
        data.append(image)
        data.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var headers = HTTPHeaders()
        headers.contentType = HTTPMediaType(
            type: "multipart",
            subType: "form-data",
            parameters: ["boundary": boundary]
        )
        return (headers, ByteBuffer(data: data))
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
        try ImagesAPIController(generator: generator, modelAliases: ["Qwen3.8-27B"]).boot(routes: app)
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

private enum FakeImageFailure: Error, Sendable {
    case invalidImage
    case unreadableSnapshot
    case unexpected
}

private actor FakeImageGenerator: AFMImageGenerating {
    var generationRequest: AFMImageGenerationRequest?
    var editRequest: AFMImageEditRequest?
    let generationFailure: FakeImageFailure?
    let editFailure: FakeImageFailure?

    init(generationFailure: FakeImageFailure? = nil, editFailure: FakeImageFailure? = nil) {
        self.generationFailure = generationFailure
        self.editFailure = editFailure
    }

    func generateImages(for request: AFMImageGenerationRequest) async throws -> [AFMGeneratedImage] {
        try Self.throwFailure(generationFailure)
        generationRequest = request
        return [AFMGeneratedImage(data: Data("png".utf8), width: request.width, height: request.height)]
    }

    func editImages(for request: AFMImageEditRequest) async throws -> [AFMGeneratedImage] {
        try Self.throwFailure(editFailure)
        editRequest = request
        return [AFMGeneratedImage(data: Data("edited".utf8), width: request.width, height: request.height)]
    }

    private static func throwFailure(_ failure: FakeImageFailure?) throws {
        switch failure {
        case .invalidImage:
            throw FluxKleinImageError.invalidImage
        case .unreadableSnapshot:
            throw FluxKleinImageError.unreadableSnapshot("/missing")
        case .unexpected:
            throw FakeImageFailure.unexpected
        case nil:
            return
        }
    }
}
