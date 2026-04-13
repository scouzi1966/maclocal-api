import Vapor
import Foundation

protocol VisionServing {
    func extractText(from filePath: String) async throws -> String
    func extractText(from filePath: String, options: VisionRequestOptions) async throws -> String
    func extractTextWithDetails(from filePath: String) async throws -> VisionResult
    func extractTextWithDetails(from filePath: String, options: VisionRequestOptions) async throws -> VisionResult
    func extractTables(from filePath: String) async throws -> [TableResult]
    func extractTables(from filePath: String, options: VisionRequestOptions) async throws -> [TableResult]
    func debugRawDetection(from filePath: String) async throws -> String
    func debugRawDetection(from filePath: String, options: VisionRequestOptions) async throws -> String
}

@available(macOS 26.0, *)
extension VisionService: VisionServing {}

struct VisionOCRRequest: Content {
    let file: String?
    let data: String?
    let filename: String?
    let mediaType: String?
    let imageURL: ImageURL?
    let messages: [Message]?
    let verbose: Bool?
    let table: Bool?
    let debug: Bool?
    let recognitionLevel: String?
    let usesLanguageCorrection: Bool?
    let languages: [String]?
    let maxPages: Int?

    enum CodingKeys: String, CodingKey {
        case file
        case data
        case filename
        case mediaType = "media_type"
        case imageURL = "image_url"
        case messages
        case verbose
        case table
        case debug
        case recognitionLevel = "recognition_level"
        case usesLanguageCorrection = "uses_language_correction"
        case languages
        case maxPages = "max_pages"
    }
}

struct VisionOCRResponse: Content {
    let object: String
    let mode: String
    let documents: [VisionOCRDocument]
    let combinedText: String
    let documentHints: [String]
    let debugOutput: String?

    enum CodingKeys: String, CodingKey {
        case object
        case mode
        case documents
        case combinedText = "combined_text"
        case documentHints = "document_hints"
        case debugOutput = "debug_output"
    }
}

struct VisionOCRDocument: Content {
    let file: String
    let sourceType: String
    let text: String
    let fullText: String
    let pageCount: Int
    let documentHints: [String]
    let pages: [VisionOCRPage]
    let textBlocks: [VisionOCRTextBlock]
    let tables: [VisionOCRTable]

    enum CodingKeys: String, CodingKey {
        case file
        case sourceType = "source_type"
        case text
        case fullText = "full_text"
        case pageCount = "page_count"
        case documentHints = "document_hints"
        case pages
        case textBlocks = "text_blocks"
        case tables
    }
}

struct VisionOCRPage: Content {
    let pageNumber: Int
    let text: String
    let width: Double
    let height: Double
    let textBlocks: [VisionOCRTextBlock]
    let tables: [VisionOCRTable]

    enum CodingKeys: String, CodingKey {
        case pageNumber = "page_number"
        case text
        case width
        case height
        case textBlocks = "text_blocks"
        case tables
    }
}

struct VisionOCRTextBlock: Content {
    let text: String
    let confidence: Float
    let pageNumber: Int
    let boundingBox: VisionOCRBoundingBox

    enum CodingKeys: String, CodingKey {
        case text
        case confidence
        case pageNumber = "page_number"
        case boundingBox = "bounding_box"
    }
}

struct VisionOCRTable: Content {
    let pageNumber: Int
    let rows: [[String]]
    let headers: [String]
    let rowObjects: [[String: String]]
    let columnCount: Int
    let averageConfidence: Float
    let mergedCellHints: [String]
    let boundingBox: VisionOCRBoundingBox
    let csvData: String

    enum CodingKeys: String, CodingKey {
        case pageNumber = "page_number"
        case rows
        case headers
        case rowObjects = "row_objects"
        case columnCount = "column_count"
        case averageConfidence = "average_confidence"
        case mergedCellHints = "merged_cell_hints"
        case boundingBox = "bounding_box"
        case csvData = "csv_data"
    }
}

struct VisionOCRBoundingBox: Content {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    init(_ rect: CGRect) {
        self.x = rect.origin.x
        self.y = rect.origin.y
        self.width = rect.size.width
        self.height = rect.size.height
    }
}

struct VisionResolvedInput {
    let path: String
    let sourceType: String
    let cleanupURLs: [URL]
}

struct VisionAPIController: RouteCollection {
    private static let maxRequestBodySize: ByteCount = "30mb"
    private static let supportedMediaTypes: [String: String] = [
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/heic": "heic",
        "application/pdf": "pdf"
    ]
    private static let builtInToolName = "apple_vision_ocr"

    private let makeVisionService: () -> (any VisionServing)?

    init(makeVisionService: @escaping () -> (any VisionServing)? = VisionAPIController.defaultVisionServiceFactory) {
        self.makeVisionService = makeVisionService
    }

    func boot(routes: RoutesBuilder) throws {
        let v1 = routes.grouped("v1", "vision")
        v1.on(.POST, "ocr", body: .collect(maxSize: Self.maxRequestBodySize), use: ocr)
        v1.on(.OPTIONS, "ocr", use: handleOptions)
    }

    func handleOptions(req: Request) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.headers.add(name: .accessControlAllowMethods, value: "POST, OPTIONS")
        response.headers.add(name: .accessControlAllowHeaders, value: "Content-Type, Authorization")
        return response
    }

    func ocr(req: Request) async throws -> Response {
        guard let visionService = makeVisionService() else {
            return try await createErrorResponse(
                message: VisionError.platformUnavailable.localizedDescription,
                status: .serviceUnavailable,
                type: "vision_unavailable"
            )
        }

        let parsed = try parseRequest(req)
        let options = try buildOptions(from: parsed.request)
        guard !parsed.inputs.isEmpty else {
            return try await createErrorResponse(
                message: VisionError.missingInput.localizedDescription,
                status: .badRequest
            )
        }

        defer { cleanup(parsed.cleanupURLs) }

        let wantsDebug = parsed.request.debug ?? false
        let wantsTable = parsed.request.table ?? false
        let mode = wantsDebug ? "debug" : (wantsTable ? "table" : "text")

        do {
            if wantsDebug {
                let outputs = try await parsed.inputs.asyncMap { input in
                    try await visionService.debugRawDetection(from: input.path, options: options)
                }
                return try await createSuccessResponse(
                    VisionOCRResponse(
                        object: "vision.ocr",
                        mode: mode,
                        documents: [],
                        combinedText: "",
                        documentHints: [],
                        debugOutput: outputs.joined(separator: "\n\n")
                    )
                )
            }

            let documents = try await parsed.inputs.asyncMap { input in
                let result = try await visionService.extractTextWithDetails(from: input.path, options: options)
                return Self.mapDocument(result, sourceType: input.sourceType)
            }

            let combinedText = documents.map(\.fullText).joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
            let hints = Array(Set(documents.flatMap(\.documentHints))).sorted()
            return try await createSuccessResponse(
                VisionOCRResponse(
                    object: "vision.ocr",
                    mode: mode,
                    documents: documents,
                    combinedText: combinedText,
                    documentHints: hints,
                    debugOutput: nil
                )
            )
        } catch let visionError as VisionError {
            return try await createErrorResponse(
                message: visionError.localizedDescription,
                status: Self.httpStatus(for: visionError)
            )
        } catch {
            return try await createErrorResponse(
                message: error.localizedDescription,
                status: .internalServerError,
                type: "internal_error"
            )
        }
    }

    private func parseRequest(_ req: Request) throws -> (request: VisionOCRRequest, inputs: [VisionResolvedInput], cleanupURLs: [URL]) {
        let multipartFile = try? req.content.get(Vapor.File.self, at: "file")
        let multipartFilename = multipartFile?.filename
        let multipartMediaType = multipartFile?.contentType?.description

        let request: VisionOCRRequest
        if multipartFile != nil {
            request = VisionOCRRequest(
                file: nil,
                data: nil,
                filename: multipartFilename,
                mediaType: multipartMediaType,
                imageURL: nil,
                messages: nil,
                verbose: boolField(req, "verbose"),
                table: boolField(req, "table"),
                debug: boolField(req, "debug"),
                recognitionLevel: stringField(req, "recognition_level"),
                usesLanguageCorrection: boolField(req, "uses_language_correction"),
                languages: arrayField(req, "languages"),
                maxPages: intField(req, "max_pages")
            )
        } else {
            request = try req.content.decode(VisionOCRRequest.self)
        }

        var inputs: [VisionResolvedInput] = []
        var cleanupURLs: [URL] = []

        if let multipartFile {
            let fileURL = try Self.writeTempFile(
                data: Data(buffer: multipartFile.data),
                filename: multipartFilename ?? "upload.bin",
                mediaType: multipartMediaType
            )
            inputs.append(VisionResolvedInput(path: fileURL.path, sourceType: "upload", cleanupURLs: [fileURL]))
            cleanupURLs.append(fileURL)
        }

        if let file = request.file?.trimmingCharacters(in: .whitespacesAndNewlines), !file.isEmpty {
            inputs.append(VisionResolvedInput(path: Self.resolvePath(file), sourceType: "file", cleanupURLs: []))
        }

        if let data = request.data?.trimmingCharacters(in: .whitespacesAndNewlines), !data.isEmpty {
            let fileURL = try Self.writeTempDataPayload(
                data,
                filename: request.filename ?? "upload",
                mediaType: request.mediaType
            )
            inputs.append(VisionResolvedInput(path: fileURL.path, sourceType: "data", cleanupURLs: [fileURL]))
            cleanupURLs.append(fileURL)
        }

        if let imageURL = request.imageURL {
            let resolved = try Self.resolveImageURL(imageURL)
            inputs.append(VisionResolvedInput(path: resolved.path, sourceType: "image_url", cleanupURLs: resolved.cleanupURLs))
            cleanupURLs.append(contentsOf: resolved.cleanupURLs)
        }

        if let messages = request.messages {
            for message in messages {
                guard let content = message.content, case .parts(let parts) = content else { continue }
                for part in parts where part.type == "image_url" {
                    guard let imageURL = part.image_url else { continue }
                    let resolved = try Self.resolveImageURL(imageURL)
                    inputs.append(VisionResolvedInput(path: resolved.path, sourceType: "message_image", cleanupURLs: resolved.cleanupURLs))
                    cleanupURLs.append(contentsOf: resolved.cleanupURLs)
                }
            }
        }

        return (request, inputs, cleanupURLs)
    }

    private func buildOptions(from request: VisionOCRRequest) throws -> VisionRequestOptions {
        let level: VisionRecognitionLevel
        if let requested = request.recognitionLevel?.lowercased(), !requested.isEmpty {
            guard let parsed = VisionRecognitionLevel(rawValue: requested) else {
                throw Abort(.badRequest, reason: "recognition_level must be 'accurate' or 'fast'")
            }
            level = parsed
        } else {
            level = .accurate
        }

        return VisionRequestOptions(
            recognitionLevel: level,
            usesLanguageCorrection: request.usesLanguageCorrection ?? true,
            recognitionLanguages: request.languages ?? [],
            maxPages: request.maxPages ?? VisionRequestOptions.defaultMaxPages
        )
    }

    private func createSuccessResponse(_ payload: VisionOCRResponse) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(payload)
        return response
    }

    private func createErrorResponse(message: String, status: HTTPStatus, type: String = "invalid_request_error") async throws -> Response {
        let response = Response(status: status)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(OpenAIError(message: message, type: type))
        return response
    }

    private func boolField(_ req: Request, _ field: String) -> Bool? {
        guard let value = try? req.content.get(String.self, at: field) else { return nil }
        switch value.lowercased() {
        case "true", "1", "yes", "y", "on":
            return true
        case "false", "0", "no", "n", "off":
            return false
        default:
            return nil
        }
    }

    private func intField(_ req: Request, _ field: String) -> Int? {
        guard let value = try? req.content.get(String.self, at: field) else { return nil }
        return Int(value)
    }

    private func stringField(_ req: Request, _ field: String) -> String? {
        try? req.content.get(String.self, at: field)
    }

    private func arrayField(_ req: Request, _ field: String) -> [String]? {
        if let items = try? req.content.get([String].self, at: field) {
            return items
        }
        guard let raw = try? req.content.get(String.self, at: field) else { return nil }
        return raw
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func cleanup(_ urls: [URL]) {
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private static func resolvePath(_ file: String) -> String {
        let expandedPath = NSString(string: file).expandingTildeInPath
        return URL(fileURLWithPath: expandedPath).standardized.path
    }

    static func resolveImageURL(_ imageURL: ImageURL) throws -> VisionResolvedInput {
        let raw = imageURL.url
        if raw.hasPrefix("data:") {
            let fileURL = try writeTempDataPayload(raw, filename: "image", mediaType: nil)
            return VisionResolvedInput(path: fileURL.path, sourceType: "data_url", cleanupURLs: [fileURL])
        }

        if let url = URL(string: raw), let scheme = url.scheme?.lowercased() {
            switch scheme {
            case "file":
                return VisionResolvedInput(path: url.path, sourceType: "file_url", cleanupURLs: [])
            case "http", "https":
                throw VisionError.remoteURLNotSupported
            default:
                throw VisionError.unsupportedURLScheme(scheme)
            }
        }

        return VisionResolvedInput(path: resolvePath(raw), sourceType: "file", cleanupURLs: [])
    }

    static func writeTempDataPayload(_ payload: String, filename: String, mediaType: String?) throws -> URL {
        if payload.hasPrefix("data:") {
            guard let commaIndex = payload.firstIndex(of: ",") else {
                throw VisionError.invalidDataURL
            }
            let header = String(payload[..<commaIndex])
            let base64 = String(payload[payload.index(after: commaIndex)...])
            guard let decoded = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else {
                throw VisionError.invalidDataURL
            }
            let media = header
                .replacingOccurrences(of: "data:", with: "")
                .split(separator: ";")
                .first
                .map(String.init)
            return try writeTempFile(data: decoded, filename: filename, mediaType: media)
        }

        guard let decoded = Data(base64Encoded: payload, options: .ignoreUnknownCharacters) else {
            throw VisionError.invalidDataURL
        }
        return try writeTempFile(data: decoded, filename: filename, mediaType: mediaType)
    }

    static func writeTempFile(data: Data, filename: String, mediaType: String?) throws -> URL {
        if data.count > VisionRequestOptions.defaultMaxFileBytes {
            throw VisionError.requestTooLarge(
                actualBytes: data.count,
                maxBytes: VisionRequestOptions.defaultMaxFileBytes
            )
        }

        let ext = inferExtension(filename: filename, mediaType: mediaType)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm_vision_\(UUID().uuidString).\(ext)")
        try data.write(to: tempURL)
        return tempURL
    }

    private static func inferExtension(filename: String, mediaType: String?) -> String {
        let filenameExt = URL(fileURLWithPath: filename).pathExtension.lowercased()
        if !filenameExt.isEmpty {
            return filenameExt
        }
        if let mediaType, let mapped = supportedMediaTypes[mediaType.lowercased()] {
            return mapped
        }
        return "png"
    }

    static func extractOCRTextFromMessages(_ messages: [Message], options: VisionRequestOptions) async throws -> (messages: [Message], cleanupURLs: [URL]) {
        guard #available(macOS 26.0, *) else {
            return (messages, [])
        }

        let service = VisionService()
        var updatedMessages: [Message] = []
        var cleanupURLs: [URL] = []

        for message in messages {
            guard let content = message.content, case .parts(let parts) = content else {
                updatedMessages.append(message)
                continue
            }

            var textChunks = parts.compactMap(\.text)
            var imageIndex = 0
            for part in parts where part.type == "image_url" {
                guard let imageURL = part.image_url else { continue }
                let resolved = try resolveImageURL(imageURL)
                cleanupURLs.append(contentsOf: resolved.cleanupURLs)
                let ocrText = try await service.extractText(from: resolved.path, options: options)
                imageIndex += 1
                textChunks.append("[Apple Vision OCR image \(imageIndex)]\n\(ocrText)")
            }

            updatedMessages.append(Message(role: message.role, content: textChunks.joined(separator: "\n\n")))
        }

        return (updatedMessages, cleanupURLs)
    }

    static func shouldAutoRunVisionTool(_ request: ChatCompletionRequest) -> Bool {
        guard let tools = request.tools, tools.contains(where: { $0.function.name == builtInToolName }) else {
            return false
        }

        let hasImageContent = request.messages.contains { message in
            guard let content = message.content, case .parts(let parts) = content else { return false }
            return parts.contains(where: { $0.type == "image_url" })
        }
        guard hasImageContent else { return false }

        switch request.toolChoice {
        case .mode(let mode):
            return mode == "required" || mode == "auto"
        case .function(let functionChoice):
            return functionChoice.function.name == builtInToolName
        case nil:
            return true
        }
    }

    private static func mapDocument(_ result: VisionResult, sourceType: String) -> VisionOCRDocument {
        let pages = result.pages.map { page in
            VisionOCRPage(
                pageNumber: page.pageNumber,
                text: page.fullText,
                width: page.width,
                height: page.height,
                textBlocks: page.textBlocks.map(mapTextBlock),
                tables: page.tables.map(mapTable)
            )
        }
        return VisionOCRDocument(
            file: result.filePath,
            sourceType: sourceType,
            text: result.fullText,
            fullText: result.fullText,
            pageCount: max(result.pages.count, 1),
            documentHints: result.documentHints,
            pages: pages,
            textBlocks: result.textBlocks.map(mapTextBlock),
            tables: result.pages.flatMap(\.tables).map(mapTable)
        )
    }

    private static func mapTextBlock(_ block: TextBlock) -> VisionOCRTextBlock {
        VisionOCRTextBlock(
            text: block.text,
            confidence: block.confidence,
            pageNumber: block.pageNumber,
            boundingBox: VisionOCRBoundingBox(block.boundingBox)
        )
    }

    private static func mapTable(_ table: TableResult) -> VisionOCRTable {
        VisionOCRTable(
            pageNumber: table.pageNumber,
            rows: table.rows,
            headers: table.headers,
            rowObjects: table.rowObjects,
            columnCount: table.columnCount,
            averageConfidence: table.averageConfidence,
            mergedCellHints: table.mergedCellHints,
            boundingBox: VisionOCRBoundingBox(table.boundingBox),
            csvData: table.csvData
        )
    }

    private static func httpStatus(for error: VisionError) -> HTTPStatus {
        switch error {
        case .missingInput, .unsupportedFormat, .remoteURLNotSupported, .unsupportedURLScheme, .invalidDataURL:
            return .badRequest
        case .fileNotFound:
            return .notFound
        case .requestTooLarge:
            return .payloadTooLarge
        case .pageLimitExceeded, .imageDimensionsExceeded, .imageLoadingFailed, .textRecognitionFailed, .noTextFound, .noTablesFound, .documentSegmentationFailed:
            return .unprocessableEntity
        case .platformUnavailable:
            return .serviceUnavailable
        }
    }

    private static func defaultVisionServiceFactory() -> (any VisionServing)? {
        if #available(macOS 26.0, *) {
            return VisionService()
        }
        return nil
    }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async throws -> [T] {
        var result: [T] = []
        result.reserveCapacity(count)
        for value in self {
            let transformed = try await transform(value)
            result.append(transformed)
        }
        return result
    }
}
