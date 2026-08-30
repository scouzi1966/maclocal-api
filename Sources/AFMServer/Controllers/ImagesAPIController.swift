import AFMKit
import Foundation
import Vapor

public struct ImageGenerationRequestBody: Content {
    public var model: String?
    public var prompt: String
    public var n: Int?
    public var size: String?
    public var responseFormat: String?
    public var seed: UInt64?

    enum CodingKeys: String, CodingKey {
        case model, prompt, n, size, seed
        case responseFormat = "response_format"
    }
}

private struct ImageEditForm: Content {
    var model: String?
    var prompt: String
    var image: File
    var n: Int?
    var size: String?
    var responseFormat: String?
    var seed: UInt64?

    enum CodingKeys: String, CodingKey {
        case model, prompt, image, n, size, seed
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
    }

    private func generate(req: Request) async throws -> Response {
        let body: ImageGenerationRequestBody
        do {
            body = try req.content.decode(ImageGenerationRequestBody.self)
        } catch {
            throw Abort(.badRequest, reason: "Invalid image generation request: \(error.localizedDescription)")
        }
        let dimensions = try Self.dimensions(body.size, defaultValue: (1024, 1024))
        try Self.validateResponseFormat(body.responseFormat)
        let results = try await generator.generateImages(for: AFMImageGenerationRequest(
            prompt: try Self.validPrompt(body.prompt),
            width: dimensions.width,
            height: dimensions.height,
            count: try Self.validCount(body.n),
            seed: body.seed
        ))
        return try Self.response(for: results)
    }

    private func edit(req: Request) async throws -> Response {
        let form: ImageEditForm
        do {
            form = try req.content.decode(ImageEditForm.self)
        } catch {
            throw Abort(.badRequest, reason: "Invalid multipart image edit request: \(error.localizedDescription)")
        }
        let dimensions = try Self.dimensions(form.size, defaultValue: (1024, 1024))
        try Self.validateResponseFormat(form.responseFormat)
        let results = try await generator.editImages(for: AFMImageEditRequest(
            prompt: try Self.validPrompt(form.prompt),
            images: [Data(buffer: form.image.data)],
            width: dimensions.width,
            height: dimensions.height,
            count: try Self.validCount(form.n),
            seed: form.seed
        ))
        return try Self.response(for: results)
    }

    private func options(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        Self.applyHeaders(to: response)
        return response
    }

    private static func validPrompt(_ prompt: String) throws -> String {
        let value = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { throw Abort(.badRequest, reason: "prompt must not be empty") }
        return value
    }

    private static func validCount(_ count: Int?) throws -> Int {
        let value = count ?? 1
        guard (1...4).contains(value) else { throw Abort(.badRequest, reason: "n must be between 1 and 4") }
        return value
    }

    static func dimensions(_ size: String?, defaultValue: (Int, Int)) throws -> (width: Int, height: Int) {
        guard let size, size.lowercased() != "auto" else { return defaultValue }
        let parts = size.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]), let height = Int(parts[1]),
              (64...2048).contains(width), (64...2048).contains(height)
        else {
            throw Abort(.badRequest, reason: "size must be auto or WIDTHxHEIGHT between 64 and 2048")
        }
        return (width, height)
    }

    private static func validateResponseFormat(_ format: String?) throws {
        guard let format else { return }
        guard format == "b64_json" else {
            throw Abort(.badRequest, reason: "Only response_format=b64_json is supported")
        }
    }

    private static func response(for images: [AFMGeneratedImage]) throws -> Response {
        let body = OpenAIImagesResponse(
            created: Int64(Date().timeIntervalSince1970),
            data: images.map { OpenAIImageData(b64Json: $0.data.base64EncodedString()) }
        )
        let response = Response(status: .ok)
        try response.content.encode(body)
        applyHeaders(to: response)
        return response
    }

    private static func applyHeaders(to response: Response) {
        response.headers.replaceOrAdd(name: .accessControlAllowOrigin, value: "*")
        response.headers.replaceOrAdd(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.replaceOrAdd(name: .accessControlAllowHeaders, value: "Authorization, Content-Type")
        response.headers.replaceOrAdd(name: .cacheControl, value: "no-store")
    }
}
