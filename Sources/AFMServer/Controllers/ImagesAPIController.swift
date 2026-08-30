import AFMKit
import Foundation
import Vapor

public struct ImageGenerationRequestBody: Content {
    public var model: String?
    public var prompt: String
    public var background: String?
    public var moderation: String?
    public var n: Int?
    public var outputCompression: Int?
    public var outputFormat: String?
    public var partialImages: Int?
    public var quality: String?
    public var size: String?
    public var responseFormat: String?
    public var seed: UInt64?
    public var stream: Bool?
    public var style: String?
    public var user: String?

    enum CodingKeys: String, CodingKey {
        case model, prompt, background, moderation, n, quality, size, seed, stream, style, user
        case outputCompression = "output_compression"
        case outputFormat = "output_format"
        case partialImages = "partial_images"
        case responseFormat = "response_format"
    }
}

private struct ImageEditForm: Content {
    var model: String?
    var prompt: String
    var image: File
    var background: String?
    var inputFidelity: String?
    var mask: File?
    var moderation: String?
    var n: Int?
    var outputCompression: Int?
    var outputFormat: String?
    var partialImages: Int?
    var quality: String?
    var size: String?
    var responseFormat: String?
    var seed: UInt64?
    var stream: Bool?
    var user: String?

    enum CodingKeys: String, CodingKey {
        case model, prompt, image, background, mask, moderation, n, quality, size, seed, stream, user
        case inputFidelity = "input_fidelity"
        case outputCompression = "output_compression"
        case outputFormat = "output_format"
        case partialImages = "partial_images"
        case responseFormat = "response_format"
    }
}

public struct OpenAIImageData: Content {
    public var b64Json: String

    enum CodingKeys: String, CodingKey {
        case b64Json = "b64_json"
    }
}

public struct OpenAIImagesResponse: Content {
    public var created: Int64
    public var data: [OpenAIImageData]
}

public struct ImagesAPIController: RouteCollection, Sendable {
    private static let maximumBodySize: ByteCount = "100mb"
    private static let compatibilityHeader = HTTPHeaders.Name("X-AFM-Compatibility")
    private static let emulatedParametersHeader = HTTPHeaders.Name("X-AFM-Emulated-Parameters")
    private static let generationParameters: Set<String> = [
        "model", "prompt", "background", "moderation", "n", "output_compression",
        "output_format", "partial_images", "quality", "size", "response_format",
        "seed", "stream", "style", "user",
    ]
    private static let editParameters: Set<String> = [
        "model", "prompt", "image", "image[]", "background", "input_fidelity", "mask",
        "moderation", "n", "output_compression", "output_format", "partial_images",
        "quality", "size", "response_format", "seed", "stream", "user",
    ]
    private let generator: any AFMImageGenerating

    public init(generator: any AFMImageGenerating) {
        self.generator = generator
    }

    public func boot(routes: RoutesBuilder) throws {
        let images = routes.grouped("v1", "images")
        images.on(.POST, "generations", body: .collect(maxSize: Self.maximumBodySize), use: generate)
        images.on(.POST, "edits", body: .collect(maxSize: Self.maximumBodySize), use: edit)
        images.on(.OPTIONS, "generations", use: options)
        images.on(.OPTIONS, "edits", use: options)
        for method in [HTTPMethod.GET, .PUT, .PATCH, .DELETE] {
            images.on(method, "generations", use: unsupportedMethod)
            images.on(method, "edits", use: unsupportedMethod)
        }
    }

    private func generate(req: Request) async throws -> Response {
        guard Self.isJSON(req.headers.contentType) else {
            return try Self.errorResponse(
                request: req,
                status: .unsupportedMediaType,
                message: "Content-Type must be application/json",
                code: "unsupported_media_type"
            )
        }
        if let parameter = Self.unknownJSONParameter(in: req, allowed: Self.generationParameters) {
            return try Self.errorResponse(
                request: req,
                status: .badRequest,
                message: "Unknown parameter: \(parameter)",
                code: "unknown_parameter",
                param: parameter
            )
        }
        let body: ImageGenerationRequestBody
        do {
            body = try req.content.decode(ImageGenerationRequestBody.self)
        } catch {
            return try Self.errorResponse(
                request: req,
                status: .badRequest,
                message: "Invalid image generation request: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }
        do {
            try Self.validateGenerationControls(body)
            let dimensions = try Self.dimensions(body.size, defaultValue: (1024, 1024))
            let results = try await generator.generateImages(for: AFMImageGenerationRequest(
                prompt: try Self.validPrompt(body.prompt),
                width: dimensions.width,
                height: dimensions.height,
                count: try Self.validCount(body.n),
                seed: body.seed
            ))
            return try Self.response(
                for: results,
                emulatedParameters: body.model == nil ? [] : ["model"]
            )
        } catch let error as ImageAPIRequestError {
            return try Self.errorResponse(request: req, error: error)
        }
    }

    private func edit(req: Request) async throws -> Response {
        guard Self.isMultipartForm(req.headers.contentType) else {
            return try Self.errorResponse(
                request: req,
                status: .unsupportedMediaType,
                message: "Content-Type must be multipart/form-data",
                code: "unsupported_media_type"
            )
        }
        if let parameter = Self.unknownMultipartParameter(in: req, allowed: Self.editParameters) {
            return try Self.errorResponse(
                request: req,
                status: .badRequest,
                message: "Unknown parameter: \(parameter)",
                code: "unknown_parameter",
                param: parameter
            )
        }
        if Self.multipartParameterNames(in: req).contains("image[]") {
            return try Self.errorResponse(
                request: req,
                status: .badRequest,
                message: "Multiple image inputs are not supported; send one file using the image field",
                code: "unsupported_parameter",
                param: "image"
            )
        }
        let form: ImageEditForm
        do {
            form = try req.content.decode(ImageEditForm.self)
        } catch {
            return try Self.errorResponse(
                request: req,
                status: .badRequest,
                message: "Invalid multipart image edit request: \(error.localizedDescription)",
                code: "invalid_request_error"
            )
        }
        do {
            try Self.validateEditControls(form)
            let dimensions = try Self.dimensions(form.size, defaultValue: (1024, 1024))
            let results = try await generator.editImages(for: AFMImageEditRequest(
                prompt: try Self.validPrompt(form.prompt),
                images: [Data(buffer: form.image.data)],
                width: dimensions.width,
                height: dimensions.height,
                count: try Self.validCount(form.n),
                seed: form.seed
            ))
            return try Self.response(
                for: results,
                emulatedParameters: form.model == nil ? [] : ["model"]
            )
        } catch let error as ImageAPIRequestError {
            return try Self.errorResponse(request: req, error: error)
        }
    }

    private func options(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        Self.applyHeaders(to: response)
        return response
    }

    private func unsupportedMethod(req: Request) async throws -> Response {
        let response = try Self.errorResponse(
            request: req,
            status: .methodNotAllowed,
            message: "Method \(req.method.rawValue) is not allowed for this endpoint",
            code: "method_not_allowed"
        )
        response.headers.replaceOrAdd(name: "Allow", value: "POST, OPTIONS")
        return response
    }

    private static func validPrompt(_ prompt: String) throws -> String {
        let value = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            throw ImageAPIRequestError.invalid("prompt must not be empty", param: "prompt")
        }
        return value
    }

    private static func validCount(_ count: Int?) throws -> Int {
        let value = count ?? 1
        guard (1...10).contains(value) else {
            throw ImageAPIRequestError.invalid("n must be between 1 and 10", param: "n")
        }
        guard value <= 4 else {
            throw ImageAPIRequestError.unsupported("AFM image generation currently supports at most 4 images per request", param: "n")
        }
        return value
    }

    static func dimensions(_ size: String?, defaultValue: (Int, Int)) throws -> (width: Int, height: Int) {
        guard let size, size.lowercased() != "auto" else { return defaultValue }
        let parts = size.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]), let height = Int(parts[1]),
              (64...2048).contains(width), (64...2048).contains(height)
        else {
            throw ImageAPIRequestError.invalid(
                "size must be auto or WIDTHxHEIGHT between 64 and 2048",
                param: "size"
            )
        }
        return (width, height)
    }

    private static func validateGenerationControls(_ body: ImageGenerationRequestBody) throws {
        try validateResponseFormat(body.responseFormat)
        try validatePNGOutputFormat(body.outputFormat)
        try rejectIfPresent(body.background, param: "background")
        try rejectIfPresent(body.moderation, param: "moderation")
        try rejectIfPresent(body.outputCompression, param: "output_compression")
        try rejectIfPresent(body.partialImages, param: "partial_images")
        try rejectIfPresent(body.quality, param: "quality")
        try rejectIfTrue(body.stream, param: "stream")
        try rejectIfPresent(body.style, param: "style")
    }

    private static func validateEditControls(_ form: ImageEditForm) throws {
        try validateResponseFormat(form.responseFormat)
        try validatePNGOutputFormat(form.outputFormat)
        try rejectIfPresent(form.background, param: "background")
        try rejectIfPresent(form.inputFidelity, param: "input_fidelity")
        try rejectIfPresent(form.mask, param: "mask")
        try rejectIfPresent(form.moderation, param: "moderation")
        try rejectIfPresent(form.outputCompression, param: "output_compression")
        try rejectIfPresent(form.partialImages, param: "partial_images")
        try rejectIfPresent(form.quality, param: "quality")
        try rejectIfTrue(form.stream, param: "stream")
    }

    private static func validateResponseFormat(_ format: String?) throws {
        guard let format, format != "b64_json" else { return }
        throw ImageAPIRequestError.unsupported(
            "response_format=\(format) is not supported; use b64_json",
            param: "response_format"
        )
    }

    private static func validatePNGOutputFormat(_ format: String?) throws {
        guard let format, format != "png" else { return }
        throw ImageAPIRequestError.unsupported(
            "output_format=\(format) is not supported; use png",
            param: "output_format"
        )
    }

    private static func rejectIfPresent<T>(_ value: T?, param: String) throws {
        guard value != nil else { return }
        throw ImageAPIRequestError.unsupported("Parameter '\(param)' is not supported by the configured AFM image model", param: param)
    }

    private static func rejectIfTrue(_ value: Bool?, param: String) throws {
        guard value == true else { return }
        throw ImageAPIRequestError.unsupported("Parameter '\(param)=true' is not supported by the configured AFM image model", param: param)
    }

    private static func isJSON(_ mediaType: HTTPMediaType?) -> Bool {
        mediaType?.type == "application" && mediaType?.subType == "json"
    }

    private static func isMultipartForm(_ mediaType: HTTPMediaType?) -> Bool {
        mediaType?.type == "multipart" && mediaType?.subType == "form-data"
    }

    private static func unknownJSONParameter(in request: Request, allowed: Set<String>) -> String? {
        guard let buffer = request.body.data,
              let object = try? JSONSerialization.jsonObject(with: Data(buffer: buffer)) as? [String: Any]
        else { return nil }
        return object.keys.filter { !allowed.contains($0) }.sorted().first
    }

    private static func unknownMultipartParameter(in request: Request, allowed: Set<String>) -> String? {
        multipartParameterNames(in: request).filter { !allowed.contains($0) }.sorted().first
    }

    private static func multipartParameterNames(in request: Request) -> [String] {
        guard let buffer = request.body.data else { return [] }
        let text = String(decoding: Data(buffer: buffer), as: UTF8.self)
        let pattern = #"Content-Disposition:[^\r\n]*\bname="([^"]+)""#
        guard let expression = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return expression.matches(in: text, range: range).compactMap { match -> String? in
            guard let nameRange = Range(match.range(at: 1), in: text) else { return nil }
            return String(text[nameRange])
        }
    }

    private static func errorResponse(request: Request, error: ImageAPIRequestError) throws -> Response {
        try errorResponse(
            request: request,
            status: .badRequest,
            message: error.message,
            code: error.code,
            param: error.param
        )
    }

    private static func errorResponse(
        request: Request,
        status: HTTPResponseStatus,
        message: String,
        code: String,
        param: String? = nil
    ) throws -> Response {
        let response = Response(status: status)
        try response.content.encode(OpenAIError(
            message: message,
            type: "invalid_request_error",
            code: code,
            param: param,
            requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
        ))
        applyHeaders(to: response)
        return response
    }

    private static func response(for images: [AFMGeneratedImage], emulatedParameters: [String]) throws -> Response {
        let body = OpenAIImagesResponse(
            created: Int64(Date().timeIntervalSince1970),
            data: images.map { OpenAIImageData(b64Json: $0.data.base64EncodedString()) }
        )
        let response = Response(status: .ok)
        try response.content.encode(body)
        if !emulatedParameters.isEmpty {
            response.headers.replaceOrAdd(name: compatibilityHeader, value: "emulated")
            response.headers.replaceOrAdd(name: emulatedParametersHeader, value: emulatedParameters.joined(separator: ","))
        }
        applyHeaders(to: response)
        return response
    }

    private static func applyHeaders(to response: Response) {
        response.headers.replaceOrAdd(name: .accessControlAllowOrigin, value: "*")
        response.headers.replaceOrAdd(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.replaceOrAdd(name: .accessControlAllowHeaders, value: "Authorization, Content-Type")
        response.headers.replaceOrAdd(
            name: "Access-Control-Expose-Headers",
            value: "X-Request-ID, OpenAI-Request-ID, X-AFM-Compatibility, X-AFM-Emulated-Parameters"
        )
        response.headers.replaceOrAdd(name: .cacheControl, value: "no-store")
    }
}

private struct ImageAPIRequestError: Error {
    let message: String
    let code: String
    let param: String

    static func invalid(_ message: String, param: String) -> Self {
        Self(message: message, code: "invalid_request_error", param: param)
    }

    static func unsupported(_ message: String, param: String) -> Self {
        Self(message: message, code: "unsupported_parameter", param: param)
    }
}
