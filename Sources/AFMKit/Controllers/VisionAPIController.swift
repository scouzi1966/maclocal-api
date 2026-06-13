import Vapor
import Foundation

protocol VisionServing {
    @available(macOS 26.0, *)
    func extractText(from filePath: String) async throws -> String
    @available(macOS 26.0, *)
    func extractText(from filePath: String, options: VisionRequestOptions) async throws -> String
    @available(macOS 26.0, *)
    func extractTextWithDetails(from filePath: String) async throws -> VisionResult
    @available(macOS 26.0, *)
    func extractTextWithDetails(from filePath: String, options: VisionRequestOptions) async throws -> VisionResult
    @available(macOS 26.0, *)
    func extractTables(from filePath: String) async throws -> [TableResult]
    @available(macOS 26.0, *)
    func extractTables(from filePath: String, options: VisionRequestOptions) async throws -> [TableResult]
    @available(macOS 26.0, *)
    func debugRawDetection(from filePath: String) async throws -> String
    @available(macOS 26.0, *)
    func debugRawDetection(from filePath: String, options: VisionRequestOptions) async throws -> String

    // New modes (macOS 13+)
    func detectBarcodes(from filePath: String, options: VisionRequestOptions) throws -> [BarcodeResult]
    func classifyImage(from filePath: String, maxLabels: Int) throws -> ClassifyResult
    func detectSaliency(from filePath: String, type: String, includeHeatMap: Bool) throws -> SaliencyResult
    func autoCrop(imageData: Data) throws -> Data
}

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
    // New parameters
    let mode: String?
    let detail: String?
    let autoCrop: Bool?
    let responseFormat: String?
    let maxLabels: Int?
    let saliencyType: String?
    let includeHeatMap: Bool?

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
        case mode
        case detail
        case autoCrop = "auto_crop"
        case responseFormat = "response_format"
        case maxLabels = "max_labels"
        case saliencyType = "saliency_type"
        case includeHeatMap = "include_heat_map"
    }
}

// MARK: - New vision mode response types

struct VisionBarcodeResponse: Content {
    let object: String
    let mode: String
    let results: [VisionBarcodeItem]
}

struct VisionBarcodeItem: Content {
    let type: String
    let payload: String
    let boundingBox: VisionOCRBoundingBox
    let confidence: Float

    enum CodingKeys: String, CodingKey {
        case type, payload
        case boundingBox = "bounding_box"
        case confidence
    }
}

struct VisionClassifyResponse: Content {
    let object: String
    let mode: String
    let labels: [VisionClassifyLabel]
    let salientRegions: [VisionOCRBoundingBox]

    enum CodingKeys: String, CodingKey {
        case object, mode, labels
        case salientRegions = "salient_regions"
    }
}

struct VisionClassifyLabel: Content {
    let label: String
    let confidence: Float
}

struct VisionSaliencyResponse: Content {
    let object: String
    let mode: String
    let regions: [VisionSaliencyRegionItem]
    let heatMap: String?

    enum CodingKeys: String, CodingKey {
        case object, mode, regions
        case heatMap = "heat_map"
    }
}

struct VisionSaliencyRegionItem: Content {
    let type: String
    let boundingBox: VisionOCRBoundingBox

    enum CodingKeys: String, CodingKey {
        case type
        case boundingBox = "bounding_box"
    }
}

struct VisionAutoResponse: Content {
    let object: String
    let mode: String
    let modesRun: [String]
    let text: VisionOCRResponse?
    let barcodes: [VisionBarcodeItem]?
    let labels: [VisionClassifyLabel]?

    enum CodingKeys: String, CodingKey {
        case object, mode
        case modesRun = "modes_run"
        case text, barcodes, labels
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

        // Resolve mode: explicit mode > legacy flags > default "text"
        let wantsDebug = parsed.request.debug ?? false
        let wantsTable = parsed.request.table ?? false
        let resolvedMode: String
        if wantsDebug {
            resolvedMode = "debug"
        } else if let explicitMode = parsed.request.mode?.lowercased(), !explicitMode.isEmpty {
            resolvedMode = explicitMode
        } else if wantsTable {
            resolvedMode = "table"
        } else {
            resolvedMode = "text"
        }

        // Apply auto_crop preprocessing if requested
        var effectiveInputs = parsed.inputs
        var cropCleanupURLs: [URL] = []
        if parsed.request.autoCrop ?? false {
            effectiveInputs = try effectiveInputs.map { input in
                let fileURL = URL(fileURLWithPath: input.path)
                let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                if let fileSize = attrs[.size] as? Int, fileSize > VisionRequestOptions.defaultMaxFileBytes {
                    throw VisionError.fileTooLarge(bytes: fileSize, maxBytes: VisionRequestOptions.defaultMaxFileBytes)
                }
                let imageData: Data
                do {
                    imageData = try Data(contentsOf: fileURL)
                } catch {
                    throw VisionError.imageLoadingFailed
                }
                let croppedData = try visionService.autoCrop(imageData: imageData)
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("afm_vision_crop_\(UUID().uuidString).png")
                try croppedData.write(to: tempURL)
                cropCleanupURLs.append(tempURL)
                return VisionResolvedInput(path: tempURL.path, sourceType: input.sourceType, cleanupURLs: input.cleanupURLs)
            }
        }
        defer { cleanup(cropCleanupURLs) }

        do {
            switch resolvedMode {
            case "debug":
                if #available(macOS 26.0, *) {
                    let outputs = try await effectiveInputs.asyncMap { input in
                        try await visionService.debugRawDetection(from: input.path, options: options)
                    }
                    return try await createSuccessResponse(
                        VisionOCRResponse(
                            object: "vision.ocr",
                            mode: resolvedMode,
                            documents: [],
                            combinedText: "",
                            documentHints: [],
                            debugOutput: outputs.joined(separator: "\n\n")
                        )
                    )
                } else {
                    return try await createErrorResponse(
                        message: VisionError.modeRequiresMacOS26("debug").localizedDescription,
                        status: .notImplemented
                    )
                }

            case "text", "table":
                if #available(macOS 26.0, *) {
                    let documents = try await effectiveInputs.asyncMap { input in
                        let result = try await visionService.extractTextWithDetails(from: input.path, options: options)
                        return Self.mapDocument(result, sourceType: input.sourceType)
                    }
                    let combinedText = documents.map(\.fullText).joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
                    let hints = Array(Set(documents.flatMap(\.documentHints))).sorted()

                    // Handle response_format
                    let responseFormat = parsed.request.responseFormat?.lowercased() ?? "json"
                    if responseFormat == "text" {
                        let response = Response(status: .ok)
                        response.headers.add(name: .contentType, value: "text/plain")
                        response.headers.add(name: .accessControlAllowOrigin, value: "*")
                        response.body = .init(string: combinedText)
                        return response
                    }

                    return try await createSuccessResponse(
                        VisionOCRResponse(
                            object: "vision.ocr",
                            mode: resolvedMode,
                            documents: documents,
                            combinedText: combinedText,
                            documentHints: hints,
                            debugOutput: nil
                        )
                    )
                } else {
                    return try await createErrorResponse(
                        message: VisionError.modeRequiresMacOS26(resolvedMode).localizedDescription,
                        status: .notImplemented
                    )
                }

            case "barcode":
                let allBarcodes = try effectiveInputs.flatMap { input in
                    try visionService.detectBarcodes(from: input.path, options: options)
                }
                let items = allBarcodes.map { b in
                    VisionBarcodeItem(
                        type: b.type,
                        payload: b.payload,
                        boundingBox: VisionOCRBoundingBox(b.boundingBox),
                        confidence: b.confidence
                    )
                }
                return try await createJSONResponse(VisionBarcodeResponse(
                    object: "vision.ocr",
                    mode: "barcode",
                    results: items
                ))

            case "classify":
                let maxLabels = parsed.request.maxLabels ?? 5
                let allResults = try effectiveInputs.map { input in
                    try visionService.classifyImage(from: input.path, maxLabels: maxLabels)
                }
                let labels = allResults.flatMap { $0.labels }.map {
                    VisionClassifyLabel(label: $0.label, confidence: $0.confidence)
                }
                let regions = allResults.flatMap { $0.salientRegions }.map {
                    VisionOCRBoundingBox($0)
                }
                return try await createJSONResponse(VisionClassifyResponse(
                    object: "vision.ocr",
                    mode: "classify",
                    labels: labels,
                    salientRegions: regions
                ))

            case "saliency":
                let saliencyType = parsed.request.saliencyType ?? "attention"
                let includeHeatMap = parsed.request.includeHeatMap ?? false
                let allResults = try effectiveInputs.map { input in
                    try visionService.detectSaliency(from: input.path, type: saliencyType, includeHeatMap: includeHeatMap)
                }
                let regions = allResults.flatMap { $0.regions }.map {
                    VisionSaliencyRegionItem(type: $0.type, boundingBox: VisionOCRBoundingBox($0.boundingBox))
                }
                let heatMap = allResults.compactMap(\.heatMapPNG).first.map { $0.base64EncodedString() }
                return try await createJSONResponse(VisionSaliencyResponse(
                    object: "vision.ocr",
                    mode: "saliency",
                    regions: regions,
                    heatMap: heatMap
                ))

            case "auto":
                // Run barcode, classify (sync), then text (async)
                let maxLabels = parsed.request.maxLabels ?? 5

                // Filter to image-only inputs for barcode/classify (they reject PDFs)
                let imageInputs = effectiveInputs.filter { input in
                    let ext = URL(fileURLWithPath: input.path).pathExtension.lowercased()
                    return VisionService.imageOnlyExtensions.contains(ext)
                }

                var modesRun: [String] = []
                var barcodeItems: [VisionBarcodeItem]?
                var classifyLabels: [VisionClassifyLabel]?
                var textResponse: VisionOCRResponse?

                // Barcode and classify are synchronous Vision framework calls
                let barcodes = try imageInputs.flatMap { input in
                    try visionService.detectBarcodes(from: input.path, options: options)
                }
                let classified = try imageInputs.map { input in
                    try visionService.classifyImage(from: input.path, maxLabels: maxLabels)
                }

                if #available(macOS 26.0, *) {
                    // In auto mode, treat a per-input "no text found" as an empty OCR
                    // result rather than failing the whole request — other submode
                    // results (barcode, classify) should still come back.
                    var documents: [VisionOCRDocument] = []
                    for input in effectiveInputs {
                        do {
                            let result = try await visionService.extractTextWithDetails(from: input.path, options: options)
                            documents.append(Self.mapDocument(result, sourceType: input.sourceType))
                        } catch VisionError.noTextFound {
                            continue
                        }
                    }

                    if !documents.isEmpty {
                        let combinedText = documents.map(\.fullText).joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
                        let hints = Array(Set(documents.flatMap(\.documentHints))).sorted()
                        modesRun.append("text")
                        textResponse = VisionOCRResponse(
                            object: "vision.ocr",
                            mode: "text",
                            documents: documents,
                            combinedText: combinedText,
                            documentHints: hints,
                            debugOutput: nil
                        )
                    }
                }

                modesRun.append("barcode")
                if !barcodes.isEmpty {
                    barcodeItems = barcodes.map {
                        VisionBarcodeItem(type: $0.type, payload: $0.payload, boundingBox: VisionOCRBoundingBox($0.boundingBox), confidence: $0.confidence)
                    }
                }

                let labels = classified.flatMap(\.labels)
                modesRun.append("classify")
                if !labels.isEmpty {
                    classifyLabels = labels.map { VisionClassifyLabel(label: $0.label, confidence: $0.confidence) }
                }

                return try await createJSONResponse(VisionAutoResponse(
                    object: "vision.ocr",
                    mode: "auto",
                    modesRun: modesRun,
                    text: textResponse,
                    barcodes: barcodeItems,
                    labels: classifyLabels
                ))

            default:
                return try await createErrorResponse(
                    message: "Unknown mode '\(resolvedMode)'. Supported: text, table, barcode, classify, saliency, auto",
                    status: .badRequest
                )
            }
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
                maxPages: intField(req, "max_pages"),
                mode: stringField(req, "mode"),
                detail: stringField(req, "detail"),
                autoCrop: boolField(req, "auto_crop"),
                responseFormat: stringField(req, "response_format"),
                maxLabels: intField(req, "max_labels"),
                saliencyType: stringField(req, "saliency_type"),
                includeHeatMap: boolField(req, "include_heat_map")
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
        } else if let detail = request.detail?.lowercased(), !detail.isEmpty {
            // detail is an alias: high -> accurate, low -> fast
            switch detail {
            case "high": level = .accurate
            case "low": level = .fast
            default:
                throw Abort(.badRequest, reason: "detail must be 'high' or 'low'")
            }
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

    private func createJSONResponse<T: Content>(_ payload: T) async throws -> Response {
        let response = Response(status: .ok)
        response.headers.add(name: .contentType, value: "application/json")
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        try response.content.encode(payload)
        return response
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
                let sanitized = try sanitizeFilePath(url.path)
                return VisionResolvedInput(path: sanitized, sourceType: "file_url", cleanupURLs: [])
            case "http", "https":
                throw VisionError.remoteURLNotSupported
            default:
                throw VisionError.unsupportedURLScheme(scheme)
            }
        }

        let sanitized = try sanitizeFilePath(resolvePath(raw))
        return VisionResolvedInput(path: sanitized, sourceType: "file", cleanupURLs: [])
    }

    /// Validate that a file path points to an existing regular file with a
    /// supported image/document extension.  Resolves symlinks and rejects
    /// directories, non-image files, and path traversal.
    private static func sanitizeFilePath(_ path: String) throws -> String {
        let resolved = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
        let fm = FileManager.default

        // Must exist
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: resolved, isDirectory: &isDir) else {
            throw VisionError.fileNotFound
        }
        // Must be a regular file, not a directory
        guard !isDir.boolValue else {
            throw VisionError.unsupportedFormat
        }
        // Extension must be a supported image/document type
        let ext = URL(fileURLWithPath: resolved).pathExtension.lowercased()
        guard VisionRequestOptions.supportedExtensions.contains(ext) else {
            throw VisionError.unsupportedFormat
        }
        return resolved
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

    /// Run Apple Vision OCR on every `image_url` part in the request messages
    /// and return the extracted text in order, along with any temp files that
    /// need to be cleaned up by the caller.
    ///
    /// This helper intentionally does NOT return rebuilt messages: the OCR-only
    /// path in `ChatCompletionsController` bypasses the Foundation Model and
    /// streams the extracted text directly, so message reconstruction would be
    /// dead code.  If a future caller needs the OCR text spliced back into the
    /// conversation for downstream LLM input, add a separate helper (e.g.
    /// `injectOCRTextIntoMessages`) that layers on top of this one.
    static func extractOCRTextFromMessages(_ messages: [Message], options: VisionRequestOptions) async throws -> (ocrTexts: [String], cleanupURLs: [URL]) {
        guard #available(macOS 26.0, *) else {
            return ([], [])
        }

        let service = VisionService()
        var ocrTexts: [String] = []
        var cleanupURLs: [URL] = []
        var imageIndex = 0

        for message in messages {
            guard let content = message.content, case .parts(let parts) = content else {
                continue
            }

            for part in parts where part.type == "image_url" {
                guard let imageURL = part.image_url else { continue }
                let resolved = try resolveImageURL(imageURL)
                cleanupURLs.append(contentsOf: resolved.cleanupURLs)
                let ocrText = try await service.extractText(from: resolved.path, options: options)
                imageIndex += 1
                ocrTexts.append("[Apple Vision OCR image \(imageIndex)]\n\(ocrText)")
            }
        }

        return (ocrTexts, cleanupURLs)
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
        case .fileTooLarge:
            return .payloadTooLarge
        case .pageLimitExceeded, .imageDimensionsExceeded, .imageLoadingFailed, .textRecognitionFailed, .noTextFound, .noTablesFound, .documentSegmentationFailed:
            return .unprocessableEntity
        case .platformUnavailable:
            return .serviceUnavailable
        case .modeRequiresMacOS26:
            return .notImplemented
        }
    }

    private static func defaultVisionServiceFactory() -> (any VisionServing)? {
        return VisionService()
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
