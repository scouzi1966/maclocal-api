import AFMKit
import AFMKitMLXImage
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
    var image: File?
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
    private static let maximumGenerationBodySize: ByteCount = "1mb"
    private static let maximumEditBodySize: ByteCount = "25mb"
    private static let maximumImageBytes = 20 * 1024 * 1024
    private static let maximumPromptCharacters = 32_000
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
    private let modelID: String
    private let modelAliases: Set<String>

    public init(
        generator: any AFMImageGenerating,
        modelID: String = "mlx-community/FLUX.2-klein-4B-bf16",
        modelAliases: Set<String> = []
    ) {
        self.generator = generator
        self.modelID = modelID
        self.modelAliases = modelAliases
    }

    public func boot(routes: RoutesBuilder) throws {
        let images = routes.grouped("v1", "images")
        images.on(.POST, "generations", body: .collect(maxSize: Self.maximumGenerationBodySize), use: generate)
        images.on(.POST, "edits", body: .collect(maxSize: Self.maximumEditBodySize), use: edit)
        images.on(.OPTIONS, "generations", use: options)
        images.on(.OPTIONS, "edits", use: options)
        for method in [HTTPMethod.GET, .HEAD, .PUT, .PATCH, .DELETE, .CONNECT, .TRACE] {
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
            let emulatesModel = try validateModel(body.model)
            let dimensions = try Self.dimensions(body.size, defaultValue: (1024, 1024))
            let results = try await generator.generateImages(for: AFMImageGenerationRequest(
                prompt: try Self.validPrompt(body.prompt),
                width: dimensions.width,
                height: dimensions.height,
                count: try Self.validCount(body.n),
                seed: body.seed
            ))
            return try Self.response(
                request: req,
                for: results,
                emulatedParameters: emulatesModel ? ["model"] : []
            )
        } catch let error as ImageAPIRequestError {
            return try Self.errorResponse(request: req, error: error)
        } catch {
            return try Self.providerErrorResponse(request: req, error: error, operation: "generation")
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
        do {
            let parameterNames = try Self.multipartParameterNames(in: req)
            if let parameter = parameterNames.filter({ !Self.editParameters.contains($0) }).sorted().first {
                return try Self.errorResponse(
                    request: req,
                    status: .badRequest,
                    message: "Unknown parameter: \(parameter)",
                    code: "unknown_parameter",
                    param: parameter
                )
            }
            if parameterNames.contains("image[]") {
                return try Self.errorResponse(
                    request: req,
                    status: .badRequest,
                    message: "Multiple image inputs are not supported; send one file using the image field",
                    code: "unsupported_parameter",
                    param: "image"
                )
            }
        } catch {
            return try Self.errorResponse(
                request: req,
                status: .badRequest,
                message: "Invalid multipart image edit request",
                code: "invalid_request_error"
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
            let emulatesModel = try validateModel(form.model)
            guard let image = form.image else {
                throw ImageAPIRequestError.invalid("image is required", param: "image")
            }
            try Self.validateImageFile(image)
            let dimensions = try Self.dimensions(form.size, defaultValue: (1024, 1024))
            let results = try await generator.editImages(for: AFMImageEditRequest(
                prompt: try Self.validPrompt(form.prompt),
                images: [Data(buffer: image.data)],
                width: dimensions.width,
                height: dimensions.height,
                count: try Self.validCount(form.n),
                seed: form.seed
            ))
            return try Self.response(
                request: req,
                for: results,
                emulatedParameters: emulatesModel ? ["model"] : []
            )
        } catch let error as ImageAPIRequestError {
            return try Self.errorResponse(request: req, error: error)
        } catch {
            return try Self.providerErrorResponse(request: req, error: error, operation: "edit")
        }
    }

    private func options(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        Self.applyHeaders(to: response, for: req)
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
        guard value.count <= maximumPromptCharacters else {
            throw ImageAPIRequestError.invalid(
                "prompt must not exceed \(maximumPromptCharacters) characters",
                param: "prompt"
            )
        }
        return value
    }

    private func validateModel(_ requestedModel: String?) throws -> Bool {
        guard let requestedModel else { return false }
        guard !requestedModel.isEmpty else {
            throw ImageAPIRequestError.invalid("model must not be empty", param: "model")
        }
        if requestedModel == modelID { return false }
        if modelAliases.contains(requestedModel) { return true }
        throw ImageAPIRequestError(
            message: "Model '\(requestedModel)' is not configured for image generation; use '\(modelID)'",
            code: "unsupported_model",
            param: "model"
        )
    }

    private static func validateImageFile(_ image: File) throws {
        guard image.data.readableBytes > 0 else {
            throw ImageAPIRequestError.invalid("image must not be empty", param: "image")
        }
        guard image.data.readableBytes <= maximumImageBytes else {
            throw ImageAPIRequestError.invalid("image must not exceed 20 MB", param: "image")
        }
        let supportedExtensions: Set<String> = ["png", "jpg", "jpeg", "webp"]
        guard let fileExtension = image.extension?.lowercased(), supportedExtensions.contains(fileExtension) else {
            throw ImageAPIRequestError.invalid("image must be a PNG, JPEG, or WebP file", param: "image")
        }
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
              (64...2048).contains(width), (64...2048).contains(height),
              width.isMultiple(of: 16), height.isMultiple(of: 16),
              width <= height * 3, height <= width * 3
        else {
            throw ImageAPIRequestError.invalid(
                "size must be auto or WIDTHxHEIGHT between 64 and 2048, divisible by 16, with an aspect ratio from 1:3 through 3:1",
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

    /// Parses only multipart boundaries and headers. Uploaded bytes are never decoded as text
    /// or copied into a second full-body allocation.
    private static func multipartParameterNames(in request: Request) throws -> Set<String> {
        guard let boundary = request.headers.contentType?.parameters["boundary"],
              let buffer = request.body.data
        else { return [] }
        let parser = MultipartParser(boundary: boundary)
        var headers = HTTPHeaders()
        var names: Set<String> = []
        parser.onHeader = { field, value in
            headers.replaceOrAdd(name: field, value: value)
        }
        parser.onBody = { _ in }
        parser.onPartComplete = {
            if let name = MultipartPart(headers: headers, body: ByteBuffer()).name {
                names.insert(name)
            }
            headers = HTTPHeaders()
        }
        try parser.execute(buffer)
        return names
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
        param: String? = nil,
        type: String = "invalid_request_error"
    ) throws -> Response {
        let response = Response(status: status)
        try response.content.encode(OpenAIError(
            message: message,
            type: type,
            code: code,
            param: param,
            requestId: request.afmRequestID.isEmpty ? nil : request.afmRequestID
        ))
        applyHeaders(to: response, for: request)
        return response
    }

    private static func providerErrorResponse(request: Request, error: Error, operation: String) throws -> Response {
        if error is CancellationError { throw error }
        if let error = error as? FluxKleinImageError {
            switch error {
            case .invalidImage:
                return try errorResponse(
                    request: request,
                    status: .badRequest,
                    message: "The input image could not be decoded",
                    code: "invalid_image",
                    param: "image"
                )
            case .unreadableSnapshot:
                request.logger.error("Image model snapshot is unavailable: \(error.localizedDescription)")
                return try errorResponse(
                    request: request,
                    status: .serviceUnavailable,
                    message: "The configured image model is unavailable",
                    code: "image_model_unavailable",
                    type: "server_error"
                )
            case .pngEncodingFailed:
                break
            }
        }
        request.logger.error("Image \(operation) failed: \(String(describing: error))")
        return try errorResponse(
            request: request,
            status: .internalServerError,
            message: "Image \(operation) failed",
            code: "image_generation_failed",
            type: "server_error"
        )
    }

    private static func response(request: Request, for images: [AFMGeneratedImage], emulatedParameters: [String]) throws -> Response {
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
        applyHeaders(to: response, for: request)
        return response
    }

    private static let defaultAllowHeaders = "Authorization, Content-Type, OpenAI-Organization, OpenAI-Project"

    private static func applyHeaders(to response: Response, for request: Request) {
        response.headers.replaceOrAdd(name: .accessControlAllowOrigin, value: "*")
        response.headers.replaceOrAdd(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        let requested = request.headers.first(name: "Access-Control-Request-Headers")
        response.headers.replaceOrAdd(
            name: .accessControlAllowHeaders,
            value: requested.flatMap { $0.isEmpty ? nil : $0 } ?? defaultAllowHeaders
        )
        response.headers.replaceOrAdd(
            name: "Access-Control-Expose-Headers",
            value: "X-Request-ID, OpenAI-Request-ID, X-AFM-Compatibility, X-AFM-Emulated-Parameters"
        )
        response.headers.replaceOrAdd(name: .cacheControl, value: "no-store")
        response.headers.replaceOrAdd(name: .vary, value: "Access-Control-Request-Headers")
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
