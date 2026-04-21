import Foundation
import Vision
import CoreGraphics
import ImageIO
import PDFKit
import Quartz

enum VisionError: Error, LocalizedError {
    case platformUnavailable
    case missingInput
    case fileNotFound
    case unsupportedFormat
    case remoteURLNotSupported
    case unsupportedURLScheme(String)
    case invalidDataURL
    case pageLimitExceeded(actualPages: Int, maxPages: Int)
    case imageDimensionsExceeded(width: Int, height: Int, maxDimension: Int)
    case imageLoadingFailed
    case textRecognitionFailed(String)
    case noTextFound
    case noTablesFound
    case documentSegmentationFailed(String)
    case fileTooLarge(bytes: Int, maxBytes: Int)
    case modeRequiresMacOS26(String)

    var errorDescription: String? {
        switch self {
        case .platformUnavailable:
            return "Apple Vision OCR requires macOS 26.0 or later"
        case .missingInput:
            return "No OCR input was provided"
        case .fileNotFound:
            return "The specified file was not found"
        case .unsupportedFormat:
            let formats = VisionRequestOptions.supportedExtensions
                .map { $0.uppercased() }
                .sorted()
                .joined(separator: ", ")
            return "Unsupported file format. Supported formats: \(formats)"
        case .remoteURLNotSupported:
            return "Remote HTTP(S) image URLs are not supported for Apple Vision OCR"
        case .unsupportedURLScheme(let scheme):
            return "Unsupported image URL scheme: \(scheme)"
        case .invalidDataURL:
            return "Invalid data URL or base64 payload"
        case .pageLimitExceeded(let actualPages, let maxPages):
            return "Document has \(actualPages) pages which exceeds the limit of \(maxPages)"
        case .imageDimensionsExceeded(let width, let height, let maxDimension):
            return "Image dimensions \(width)x\(height) exceed the limit of \(maxDimension) pixels"
        case .imageLoadingFailed:
            return "Failed to load the image from the specified file"
        case .textRecognitionFailed(let message):
            return "Text recognition failed: \(message)"
        case .noTextFound:
            return "No text was found in the image"
        case .noTablesFound:
            return "No tables were found in the document"
        case .documentSegmentationFailed(let message):
            return "Document segmentation failed: \(message)"
        case .fileTooLarge(let bytes, let maxBytes):
            return "File size \(bytes / 1024 / 1024) MB exceeds the limit of \(maxBytes / 1024 / 1024) MB"
        case .modeRequiresMacOS26(let mode):
            return "Vision mode '\(mode)' requires macOS 26.0 or later"
        }
    }
}

enum VisionRecognitionLevel: String, Sendable {
    case accurate
    case fast

    var requestLevel: VNRequestTextRecognitionLevel {
        switch self {
        case .accurate:
            return .accurate
        case .fast:
            return .fast
        }
    }
}

struct VisionRequestOptions: Sendable {
    static let defaultMaxPages = 50
    static let defaultMaxImageDimension = 4096
    static let defaultMaxFileBytes: Int = 25 * 1024 * 1024  // 25 MB

    /// Single source of truth for file extensions accepted by the Vision OCR
    /// pipeline.  CGImageSource handles the image formats natively; PDF is
    /// rendered via PDFKit.  All Vision entry points (controller sanitization
    /// and service-level validation) must gate on this set.
    static let supportedExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "pdf",
        "tif", "tiff", "gif", "bmp", "webp"
    ]

    let recognitionLevel: VisionRecognitionLevel
    let usesLanguageCorrection: Bool
    let recognitionLanguages: [String]
    let maxPages: Int
    let maxImageDimension: Int

    init(
        recognitionLevel: VisionRecognitionLevel = .accurate,
        usesLanguageCorrection: Bool = true,
        recognitionLanguages: [String] = [],
        maxPages: Int = VisionRequestOptions.defaultMaxPages,
        maxImageDimension: Int = VisionRequestOptions.defaultMaxImageDimension
    ) {
        self.recognitionLevel = recognitionLevel
        self.usesLanguageCorrection = usesLanguageCorrection
        self.recognitionLanguages = recognitionLanguages
        self.maxPages = maxPages
        self.maxImageDimension = maxImageDimension
    }
}

// MARK: - New vision mode result types

struct BarcodeResult: Sendable {
    let type: String
    let payload: String
    let boundingBox: CGRect
    let confidence: Float
}

struct ClassificationLabel: Sendable {
    let label: String
    let confidence: Float
}

struct ClassifyResult: Sendable {
    let labels: [ClassificationLabel]
    let salientRegions: [CGRect]
}

struct SaliencyRegion: Sendable {
    let type: String
    let boundingBox: CGRect
}

struct SaliencyResult: Sendable {
    let regions: [SaliencyRegion]
    let heatMapPNG: Data?
}

// MARK: - VisionService (no class-level macOS restriction)

final class VisionService {
    static let imageOnlyExtensions: Set<String> = ["png", "jpg", "jpeg", "heic"]
    private static let pdfRenderScale: CGFloat = 2.0

    // MARK: - Text extraction (macOS 26+)

    @available(macOS 26.0, *)
    func extractText(from filePath: String) async throws -> String {
        try await extractText(from: filePath, options: VisionRequestOptions())
    }

    @available(macOS 26.0, *)
    func extractText(from filePath: String, options: VisionRequestOptions) async throws -> String {
        let result = try await extractTextWithDetails(from: filePath, options: options)
        return result.fullText
    }

    @available(macOS 26.0, *)
    func extractTextWithDetails(from filePath: String) async throws -> VisionResult {
        try await extractTextWithDetails(from: filePath, options: VisionRequestOptions())
    }

    @available(macOS 26.0, *)
    func extractTextWithDetails(from filePath: String, options: VisionRequestOptions) async throws -> VisionResult {
        let document = try analyzeDocument(at: filePath, options: options)
        let textBlocks = document.pages.flatMap(\.textBlocks)
        guard !textBlocks.isEmpty else {
            throw VisionError.noTextFound
        }
        return VisionResult(
            fullText: document.fullText,
            textBlocks: textBlocks,
            filePath: filePath,
            pages: document.pages,
            documentHints: document.documentHints
        )
    }

    @available(macOS 26.0, *)
    func extractTables(from filePath: String) async throws -> [TableResult] {
        try await extractTables(from: filePath, options: VisionRequestOptions())
    }

    @available(macOS 26.0, *)
    func extractTables(from filePath: String, options: VisionRequestOptions) async throws -> [TableResult] {
        let document = try analyzeDocument(at: filePath, options: options)
        let tables = document.pages.flatMap(\.tables)
        guard !tables.isEmpty else {
            throw VisionError.noTablesFound
        }
        return tables
    }

    @available(macOS 26.0, *)
    func debugRawDetection(from filePath: String) async throws -> String {
        try await debugRawDetection(from: filePath, options: VisionRequestOptions())
    }

    @available(macOS 26.0, *)
    func debugRawDetection(from filePath: String, options: VisionRequestOptions) async throws -> String {
        let (url, fileExtension) = try validateFile(at: filePath)
        let pageSources = try createPageSources(from: url, fileExtension: fileExtension, options: options)
        var sections: [String] = []

        for page in pageSources {
            let textRequest = makeTextRequest(options: options)
            do {
                try page.requestHandler.perform([textRequest])
            } catch {
                throw VisionError.textRecognitionFailed(error.localizedDescription)
            }

            guard let observations = textRequest.results else { continue }
            var debugOutput = "=== PAGE \(page.pageNumber) RAW VISION DETECTION ===\n"
            debugOutput += "Total text blocks detected: \(observations.count)\n\n"
            let sortedObservations = observations.sorted { first, second in
                if abs(first.boundingBox.origin.y - second.boundingBox.origin.y) < 0.01 {
                    return first.boundingBox.origin.x < second.boundingBox.origin.x
                }
                return first.boundingBox.origin.y > second.boundingBox.origin.y
            }

            for (index, observation) in sortedObservations.enumerated() {
                guard let candidate = observation.topCandidates(1).first else { continue }
                let box = observation.boundingBox
                debugOutput += "Block \(index + 1):\n"
                debugOutput += "  Text: \"\(candidate.string)\"\n"
                debugOutput += "  Confidence: \(String(format: "%.3f", candidate.confidence))\n"
                debugOutput += "  Position: x=\(String(format: "%.3f", box.origin.x)), y=\(String(format: "%.3f", box.origin.y))\n"
                debugOutput += "  Size: width=\(String(format: "%.3f", box.width)), height=\(String(format: "%.3f", box.height))\n\n"
            }
            sections.append(debugOutput)
        }

        let output = sections.joined(separator: "\n")
        guard !output.isEmpty else { throw VisionError.noTextFound }
        return output
    }

    // MARK: - Barcode detection (macOS 13+)

    func detectBarcodes(from filePath: String, options: VisionRequestOptions = VisionRequestOptions()) throws -> [BarcodeResult] {
        let (url, _) = try validateFile(at: filePath, allowedExtensions: Self.imageOnlyExtensions)
        guard let imageData = try? Data(contentsOf: url) else {
            throw VisionError.imageLoadingFailed
        }

        let handler = VNImageRequestHandler(data: imageData)
        let request = VNDetectBarcodesRequest()

        try handler.perform([request])

        guard let observations = request.results else { return [] }
        return observations.map { obs in
            BarcodeResult(
                type: obs.symbology.rawValue.replacingOccurrences(of: "VNBarcodeSymbology", with: ""),
                payload: obs.payloadStringValue ?? "",
                boundingBox: obs.boundingBox,
                confidence: obs.confidence
            )
        }
    }

    // MARK: - Image classification (macOS 13+)

    func classifyImage(from filePath: String, maxLabels: Int = 5) throws -> ClassifyResult {
        let (url, _) = try validateFile(at: filePath, allowedExtensions: Self.imageOnlyExtensions)
        guard let imageData = try? Data(contentsOf: url) else {
            throw VisionError.imageLoadingFailed
        }

        let handler = VNImageRequestHandler(data: imageData)
        let classifyRequest = VNClassifyImageRequest()
        let saliencyRequest = VNGenerateAttentionBasedSaliencyImageRequest()

        try handler.perform([classifyRequest, saliencyRequest])

        let labels: [ClassificationLabel]
        if let observations = classifyRequest.results {
            labels = observations
                .sorted { $0.confidence > $1.confidence }
                .prefix(maxLabels)
                .map { ClassificationLabel(label: $0.identifier, confidence: $0.confidence) }
        } else {
            labels = []
        }

        let salientRegions: [CGRect]
        if let saliencyObs = saliencyRequest.results?.first,
           let salientObjects = saliencyObs.salientObjects {
            salientRegions = salientObjects.map(\.boundingBox)
        } else {
            salientRegions = []
        }

        return ClassifyResult(labels: labels, salientRegions: salientRegions)
    }

    // MARK: - Saliency detection (macOS 13+)

    func detectSaliency(from filePath: String, type: String = "attention", includeHeatMap: Bool = false) throws -> SaliencyResult {
        let (url, _) = try validateFile(at: filePath, allowedExtensions: Self.imageOnlyExtensions)
        guard let imageData = try? Data(contentsOf: url) else {
            throw VisionError.imageLoadingFailed
        }

        let handler = VNImageRequestHandler(data: imageData)
        let request: VNImageBasedRequest
        if type == "objectness" {
            request = VNGenerateObjectnessBasedSaliencyImageRequest()
        } else {
            request = VNGenerateAttentionBasedSaliencyImageRequest()
        }

        try handler.perform([request])

        var regions: [SaliencyRegion] = []
        var heatMapPNG: Data? = nil

        if let saliencyObs = (request.results as? [VNSaliencyImageObservation])?.first {
            if let salientObjects = saliencyObs.salientObjects {
                regions = salientObjects.map { obj in
                    SaliencyRegion(type: type, boundingBox: obj.boundingBox)
                }
            }

            if includeHeatMap {
                let pixelBuffer = saliencyObs.pixelBuffer
                heatMapPNG = renderHeatMapPNG(from: pixelBuffer)
            }
        }

        return SaliencyResult(regions: regions, heatMapPNG: heatMapPNG)
    }

    // MARK: - Auto-crop via document segmentation (macOS 13+)

    func autoCrop(imageData: Data) throws -> Data {
        let handler = VNImageRequestHandler(data: imageData)
        let request = VNDetectDocumentSegmentationRequest()

        try handler.perform([request])

        guard let observation = request.results?.first else {
            // No document region found — return original
            return imageData
        }

        // Crop the image to the detected document region
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return imageData
        }

        let box = observation.boundingBox
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        let cropRect = CGRect(
            x: box.origin.x * imageWidth,
            y: (1.0 - box.origin.y - box.height) * imageHeight,
            width: box.width * imageWidth,
            height: box.height * imageHeight
        )

        guard let croppedImage = cgImage.cropping(to: cropRect) else {
            return imageData
        }

        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else {
            return imageData
        }
        CGImageDestinationAddImage(destination, croppedImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            return imageData
        }
        return mutableData as Data
    }

    // MARK: - Internal helpers

    func validateFile(at filePath: String, maxBytes: Int = VisionRequestOptions.defaultMaxFileBytes, allowedExtensions: Set<String>? = nil) throws -> (URL, String) {
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw VisionError.fileNotFound
        }

        let fileExtension = url.pathExtension.lowercased()
        let extensions = allowedExtensions ?? VisionRequestOptions.supportedExtensions
        guard extensions.contains(fileExtension) else {
            throw VisionError.unsupportedFormat
        }

        let attrs = try FileManager.default.attributesOfItem(atPath: filePath)
        if let fileSize = attrs[.size] as? Int, fileSize > maxBytes {
            throw VisionError.fileTooLarge(bytes: fileSize, maxBytes: maxBytes)
        }

        return (url, fileExtension)
    }

    // MARK: - Private helpers

    @available(macOS 26.0, *)
    private func analyzeDocument(at filePath: String, options: VisionRequestOptions) throws -> VisionDocumentResult {
        let (url, fileExtension) = try validateFile(at: filePath)
        let pageSources = try createPageSources(from: url, fileExtension: fileExtension, options: options)
        var pageResults: [VisionPageResult] = []
        let analyzer = TableAnalyzer()

        for page in pageSources {
            let textRequest = makeTextRequest(options: options)
            do {
                try page.requestHandler.perform([textRequest])
            } catch {
                throw VisionError.textRecognitionFailed(error.localizedDescription)
            }

            let observations = textRequest.results ?? []
            let textBlocks = observations.compactMap { observation -> TextBlock? in
                guard let candidate = observation.topCandidates(1).first else { return nil }
                return TextBlock(
                    text: candidate.string,
                    confidence: candidate.confidence,
                    boundingBox: observation.boundingBox,
                    pageNumber: page.pageNumber
                )
            }

            let positionedBlocks = observations.compactMap { observation -> PositionedTextBlock? in
                guard let candidate = observation.topCandidates(1).first else { return nil }
                return PositionedTextBlock(
                    text: candidate.string,
                    confidence: candidate.confidence,
                    boundingBox: observation.boundingBox,
                    pageNumber: page.pageNumber
                )
            }

            let tables = analyzer.detectTables(from: positionedBlocks, pageNumber: page.pageNumber)
            let fullText = textBlocks.map(\.text).joined(separator: "\n")
            pageResults.append(
                VisionPageResult(
                    pageNumber: page.pageNumber,
                    fullText: fullText,
                    textBlocks: textBlocks,
                    tables: tables,
                    width: page.size.width,
                    height: page.size.height
                )
            )
        }

        let allTextBlocks = pageResults.flatMap(\.textBlocks)
        guard !allTextBlocks.isEmpty else {
            throw VisionError.noTextFound
        }

        let fullText = pageResults
            .map { "Page \($0.pageNumber)\n\($0.fullText)" }
            .joined(separator: "\n\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return VisionDocumentResult(
            filePath: filePath,
            fullText: fullText,
            pages: pageResults,
            documentHints: deriveDocumentHints(from: pageResults)
        )
    }

    private func createPageSources(from url: URL, fileExtension: String, options: VisionRequestOptions) throws -> [VisionPageSource] {
        if fileExtension == "pdf" {
            return try createPDFPageSources(from: url, options: options)
        }
        return [try createImagePageSource(from: url, pageNumber: 1, options: options)]
    }

    private func createImagePageSource(from url: URL, pageNumber: Int, options: VisionRequestOptions) throws -> VisionPageSource {
        let imageData: Data
        do {
            imageData = try Data(contentsOf: url)
        } catch {
            throw VisionError.imageLoadingFailed
        }
        return try createImagePageSource(from: imageData, pageNumber: pageNumber, options: options)
    }

    private func createImagePageSource(from data: Data, pageNumber: Int, options: VisionRequestOptions) throws -> VisionPageSource {
        let metadata = try validateImageDimensions(data: data, maxDimension: options.maxImageDimension)
        return VisionPageSource(
            pageNumber: pageNumber,
            requestHandler: VNImageRequestHandler(data: data),
            size: metadata
        )
    }

    private func createPDFPageSources(from url: URL, options: VisionRequestOptions) throws -> [VisionPageSource] {
        guard let pdfDocument = PDFDocument(url: url), pdfDocument.pageCount > 0 else {
            throw VisionError.imageLoadingFailed
        }
        if pdfDocument.pageCount > options.maxPages {
            throw VisionError.pageLimitExceeded(actualPages: pdfDocument.pageCount, maxPages: options.maxPages)
        }

        var pageSources: [VisionPageSource] = []
        for index in 0..<pdfDocument.pageCount {
            guard let page = pdfDocument.page(at: index) else { continue }
            let pageRect = page.bounds(for: .mediaBox)
            let scaledSize = CGSize(
                width: pageRect.width * Self.pdfRenderScale,
                height: pageRect.height * Self.pdfRenderScale
            )
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                    data: nil,
                    width: Int(scaledSize.width),
                    height: Int(scaledSize.height),
                    bitsPerComponent: 8,
                    bytesPerRow: 0,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  ) else {
                throw VisionError.imageLoadingFailed
            }

            context.setFillColor(CGColor.white)
            context.fill(CGRect(origin: .zero, size: scaledSize))
            context.scaleBy(x: Self.pdfRenderScale, y: Self.pdfRenderScale)
            context.translateBy(x: 0, y: pageRect.height)
            context.scaleBy(x: 1.0, y: -1.0)
            page.draw(with: .mediaBox, to: context)

            guard let cgImage = context.makeImage() else {
                throw VisionError.imageLoadingFailed
            }

            if Int(max(cgImage.width, cgImage.height)) > options.maxImageDimension {
                throw VisionError.imageDimensionsExceeded(
                    width: cgImage.width,
                    height: cgImage.height,
                    maxDimension: options.maxImageDimension
                )
            }

            pageSources.append(
                VisionPageSource(
                    pageNumber: index + 1,
                    requestHandler: VNImageRequestHandler(cgImage: cgImage),
                    size: CGSize(width: cgImage.width, height: cgImage.height)
                )
            )
        }

        return pageSources
    }

    private func validateImageDimensions(data: Data, maxDimension: Int) throws -> CGSize {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw VisionError.imageLoadingFailed
        }

        if max(width, height) > maxDimension {
            throw VisionError.imageDimensionsExceeded(width: width, height: height, maxDimension: maxDimension)
        }
        return CGSize(width: width, height: height)
    }

    @available(macOS 26.0, *)
    private func makeTextRequest(options: VisionRequestOptions) -> VNRecognizeTextRequest {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = options.recognitionLevel.requestLevel
        request.usesLanguageCorrection = options.usesLanguageCorrection
        if !options.recognitionLanguages.isEmpty {
            request.recognitionLanguages = options.recognitionLanguages
        }
        return request
    }

    private func deriveDocumentHints(from pages: [VisionPageResult]) -> [String] {
        let fullText = pages.map(\.fullText).joined(separator: "\n").lowercased()
        var hints: [String] = []
        if pages.count > 1 {
            hints.append("multi_page")
        }
        if pages.contains(where: { !$0.tables.isEmpty }) {
            hints.append("table_like")
        }
        if fullText.contains("invoice") || fullText.contains("bill to") || fullText.contains("payment due") {
            hints.append("invoice")
        }
        if fullText.contains("receipt") || fullText.contains("subtotal") || fullText.contains("tax") {
            hints.append("receipt")
        }
        if fullText.contains("application") || fullText.contains("signature") || fullText.contains("date:") {
            hints.append("form")
        }
        if hints.isEmpty {
            hints.append("document")
        }
        return hints
    }

    private func renderHeatMapPNG(from pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: width, height: height)) else {
            return nil
        }
        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            return nil
        }
        return mutableData as Data
    }
}

private struct VisionPageSource {
    let pageNumber: Int
    let requestHandler: VNImageRequestHandler
    let size: CGSize
}

struct VisionDocumentResult {
    let filePath: String
    let fullText: String
    let pages: [VisionPageResult]
    let documentHints: [String]
}

struct VisionPageResult {
    let pageNumber: Int
    let fullText: String
    let textBlocks: [TextBlock]
    let tables: [TableResult]
    let width: Double
    let height: Double
}

struct TextBlock {
    let text: String
    let confidence: Float
    let boundingBox: CGRect
    let pageNumber: Int

    init(text: String, confidence: Float, boundingBox: CGRect, pageNumber: Int = 1) {
        self.text = text
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.pageNumber = pageNumber
    }
}

struct VisionResult {
    let fullText: String
    let textBlocks: [TextBlock]
    let filePath: String
    let pages: [VisionPageResult]
    let documentHints: [String]

    init(
        fullText: String,
        textBlocks: [TextBlock],
        filePath: String,
        pages: [VisionPageResult] = [],
        documentHints: [String] = []
    ) {
        self.fullText = fullText
        self.textBlocks = textBlocks
        self.filePath = filePath
        self.pages = pages
        self.documentHints = documentHints
    }
}

struct PositionedTextBlock {
    let text: String
    let confidence: Float
    let boundingBox: CGRect
    let pageNumber: Int

    init(text: String, confidence: Float, boundingBox: CGRect, pageNumber: Int = 1) {
        self.text = text
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.pageNumber = pageNumber
    }
}

struct TableResult {
    let rows: [[String]]
    let columnCount: Int
    let averageConfidence: Float
    let boundingBox: CGRect
    let pageNumber: Int
    let headers: [String]
    let rowObjects: [[String: String]]
    let mergedCellHints: [String]

    init(
        rows: [[String]],
        columnCount: Int,
        averageConfidence: Float,
        boundingBox: CGRect,
        pageNumber: Int = 1,
        headers: [String] = [],
        rowObjects: [[String: String]] = [],
        mergedCellHints: [String] = []
    ) {
        self.rows = rows
        self.columnCount = columnCount
        self.averageConfidence = averageConfidence
        self.boundingBox = boundingBox
        self.pageNumber = pageNumber
        self.headers = headers
        self.rowObjects = rowObjects
        self.mergedCellHints = mergedCellHints
    }

    var csvData: String {
        let cleanedRows = rows.compactMap { row -> String? in
            var cleanRow = row
            while cleanRow.last?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true {
                cleanRow.removeLast()
            }

            let hasContent = cleanRow.contains { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            guard hasContent else { return nil }

            return cleanRow.map { cell in
                let trimmedCell = cell.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedCell.contains(",") || trimmedCell.contains("\"") || trimmedCell.contains("\n") {
                    return "\"" + trimmedCell.replacingOccurrences(of: "\"", with: "\"\"") + "\""
                }
                return trimmedCell
            }.joined(separator: ",")
        }

        return cleanedRows.joined(separator: "\n")
    }
}

final class TableAnalyzer {
    func detectTables(from textBlocks: [PositionedTextBlock], pageNumber: Int) -> [TableResult] {
        let sortedBlocks = textBlocks.sorted { first, second in
            if abs(first.boundingBox.origin.y - second.boundingBox.origin.y) < 0.01 {
                return first.boundingBox.origin.x < second.boundingBox.origin.x
            }
            return first.boundingBox.origin.y > second.boundingBox.origin.y
        }

        let rowGroups = groupIntoRows(sortedBlocks)
        return detectTablesFromRows(rowGroups, pageNumber: pageNumber)
    }

    private func groupIntoRows(_ blocks: [PositionedTextBlock]) -> [[PositionedTextBlock]] {
        var rowGroups: [[PositionedTextBlock]] = []
        let yTolerance: Float = 0.02

        for block in blocks {
            let blockY = Float(block.boundingBox.origin.y)
            if let existingRowIndex = rowGroups.firstIndex(where: { rowGroup in
                let rowY = Float(rowGroup.first?.boundingBox.origin.y ?? 0)
                return abs(blockY - rowY) <= yTolerance
            }) {
                rowGroups[existingRowIndex].append(block)
            } else {
                rowGroups.append([block])
            }
        }

        for index in rowGroups.indices {
            rowGroups[index].sort { $0.boundingBox.origin.x < $1.boundingBox.origin.x }
        }

        return rowGroups
    }

    private func detectTablesFromRows(_ rowGroups: [[PositionedTextBlock]], pageNumber: Int) -> [TableResult] {
        var tables: [TableResult] = []
        var currentTableRows: [[PositionedTextBlock]] = []
        var maxColumnsSeen = 0

        for rowGroup in rowGroups {
            let currentColumnCount = rowGroup.count
            if currentColumnCount >= 1 {
                if currentTableRows.isEmpty {
                    currentTableRows = [rowGroup]
                    maxColumnsSeen = currentColumnCount
                } else {
                    let shouldContinue = isRowPartOfTable(
                        rowGroup,
                        comparedTo: currentTableRows,
                        maxColumns: maxColumnsSeen
                    )
                    if shouldContinue {
                        currentTableRows.append(rowGroup)
                        maxColumnsSeen = max(maxColumnsSeen, currentColumnCount)
                    } else {
                        if currentTableRows.count >= 2 && maxColumnsSeen >= 2,
                           let table = createTable(from: currentTableRows, pageNumber: pageNumber) {
                            tables.append(table)
                        }
                        currentTableRows = [rowGroup]
                        maxColumnsSeen = currentColumnCount
                    }
                }
            } else {
                if currentTableRows.count >= 2 && maxColumnsSeen >= 2,
                   let table = createTable(from: currentTableRows, pageNumber: pageNumber) {
                    tables.append(table)
                }
                currentTableRows = []
                maxColumnsSeen = 0
            }
        }

        if currentTableRows.count >= 2 && maxColumnsSeen >= 2,
           let table = createTable(from: currentTableRows, pageNumber: pageNumber) {
            tables.append(table)
        }

        return tables
    }

    private func isRowPartOfTable(
        _ newRow: [PositionedTextBlock],
        comparedTo existingRows: [[PositionedTextBlock]],
        maxColumns: Int
    ) -> Bool {
        guard !existingRows.isEmpty else { return true }

        let existingXPositions = existingRows.flatMap { row in
            row.map { Float($0.boundingBox.origin.x) }
        }.sorted()
        let newRowXPositions = newRow.map { Float($0.boundingBox.origin.x) }.sorted()

        let alignmentCount = newRowXPositions.filter { newX in
            existingXPositions.contains { existingX in
                abs(newX - existingX) <= 0.05
            }
        }.count

        let alignmentRatio = Float(alignmentCount) / Float(max(newRowXPositions.count, 1))
        let columnCountReasonable = newRow.count <= maxColumns * 2
        return alignmentRatio >= 0.3 || columnCountReasonable
    }

    private func createTable(from rowGroups: [[PositionedTextBlock]], pageNumber: Int) -> TableResult? {
        guard rowGroups.count >= 2 else { return nil }
        let headerRow = rowGroups.first!
        guard headerRow.count >= 2 else { return nil }

        let columnPositions = calculateColumnPositions(from: rowGroups, prioritizeRow: headerRow)
        let maxColumns = columnPositions.count
        var tableData: [[String]] = []
        var allConfidences: [Float] = []

        for rowGroup in rowGroups {
            var row = Array(repeating: "", count: maxColumns)
            for block in rowGroup {
                let columnIndex = findBestColumn(for: Float(block.boundingBox.origin.x), in: columnPositions)
                if columnIndex < maxColumns {
                    row[columnIndex] = block.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    allConfidences.append(block.confidence)
                }
            }
            tableData.append(row)
        }

        let allBlocks = rowGroups.flatMap { $0 }
        let minX = allBlocks.map { $0.boundingBox.origin.x }.min() ?? 0
        let maxX = allBlocks.map { $0.boundingBox.maxX }.max() ?? 0
        let minY = allBlocks.map { $0.boundingBox.origin.y }.min() ?? 0
        let maxY = allBlocks.map { $0.boundingBox.maxY }.max() ?? 0
        let tableBoundingBox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        let averageConfidence = allConfidences.isEmpty ? 0.0 : allConfidences.reduce(0, +) / Float(allConfidences.count)

        let headers = normalizedHeaders(from: tableData.first ?? [])
        let rowObjects = tableData.dropFirst().map { row in
            var object: [String: String] = [:]
            for index in 0..<min(headers.count, row.count) {
                object[headers[index]] = row[index]
            }
            return object
        }
        let mergedCellHints = deriveMergedCellHints(from: tableData)

        return TableResult(
            rows: tableData,
            columnCount: maxColumns,
            averageConfidence: averageConfidence,
            boundingBox: tableBoundingBox,
            pageNumber: pageNumber,
            headers: headers,
            rowObjects: rowObjects,
            mergedCellHints: mergedCellHints
        )
    }

    private func calculateColumnPositions(
        from rowGroups: [[PositionedTextBlock]],
        prioritizeRow: [PositionedTextBlock]
    ) -> [Float] {
        var positions = prioritizeRow.map { Float($0.boundingBox.origin.x) }

        for rowGroup in rowGroups {
            for block in rowGroup {
                let x = Float(block.boundingBox.origin.x)
                if !positions.contains(where: { abs($0 - x) <= 0.05 }) {
                    positions.append(x)
                }
            }
        }
        return positions.sorted()
    }

    private func findBestColumn(for blockX: Float, in columnPositions: [Float]) -> Int {
        guard !columnPositions.isEmpty else { return 0 }
        var bestIndex = 0
        var bestDistance = abs(blockX - columnPositions[0])
        for (index, position) in columnPositions.enumerated().dropFirst() {
            let distance = abs(blockX - position)
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = index
            }
        }
        return bestIndex
    }

    private func normalizedHeaders(from row: [String]) -> [String] {
        row.enumerated().map { index, value in
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? "column_\(index + 1)" : trimmed
        }
    }

    private func deriveMergedCellHints(from rows: [[String]]) -> [String] {
        var hints: [String] = []
        for (rowIndex, row) in rows.enumerated() where row.count >= 3 {
            let emptyRuns = row.enumerated().filter { $0.element.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            if !emptyRuns.isEmpty {
                hints.append("row_\(rowIndex + 1)_contains_sparse_cells")
            }
        }
        return hints
    }
}
