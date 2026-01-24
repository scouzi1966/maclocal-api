import Foundation
import Vision
import CoreImage
import PDFKit
import Quartz
import AppKit

enum VisionError: Error, LocalizedError {
    case fileNotFound
    case unsupportedFormat
    case imageLoadingFailed
    case textRecognitionFailed(String)
    case noTextFound
    case noTablesFound
    case documentSegmentationFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound:
            return "The specified file was not found"
        case .unsupportedFormat:
            return "Unsupported file format. Supported formats: PNG, JPG, JPEG, HEIC, PDF"
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
        }
    }
}

@available(macOS 26.0, *)
class VisionService {

    private let supportedExtensions = ["png", "jpg", "jpeg", "heic", "pdf"]

    /// Extract text from a base64-encoded data URL (supports images and PDFs)
    /// - Parameter dataURL: Data URL in format "data:image/png;base64,..." or "data:application/pdf;base64,..."
    /// - Returns: Extracted text from the file
    func extractTextFromBase64(_ dataURL: String) async throws -> String {
        // Parse data URL format: data:mime/type;base64,DATA...
        guard dataURL.hasPrefix("data:"),
              let semicolonIndex = dataURL.firstIndex(of: ";"),
              let commaIndex = dataURL.firstIndex(of: ",") else {
            throw VisionError.unsupportedFormat
        }

        let mimeType = String(dataURL[dataURL.index(dataURL.startIndex, offsetBy: 5)..<semicolonIndex])
        let base64String = String(dataURL[dataURL.index(after: commaIndex)...])

        guard let fileData = Data(base64Encoded: base64String) else {
            throw VisionError.imageLoadingFailed
        }

        // Handle PDFs specially - extract text from ALL pages
        if mimeType == "application/pdf" {
            return try await extractTextFromPDFData(fileData)
        }

        // Handle images
        let requestHandler = VNImageRequestHandler(data: fileData)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        try requestHandler.perform([request])

        guard let observations = request.results, !observations.isEmpty else {
            throw VisionError.noTextFound
        }

        return observations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }

    /// Extract text from PDF data (all pages)
    /// - Parameter pdfData: Raw PDF data
    /// - Returns: Combined text from all pages
    func extractTextFromPDFData(_ pdfData: Data) async throws -> String {
        guard let pdfDocument = PDFDocument(data: pdfData) else {
            throw VisionError.imageLoadingFailed
        }

        guard pdfDocument.pageCount > 0 else {
            throw VisionError.noTextFound
        }

        var allText: [String] = []

        // Process ALL pages
        for pageIndex in 0..<pdfDocument.pageCount {
            guard let pdfPage = pdfDocument.page(at: pageIndex) else { continue }

            // First try native text extraction (faster, works for text PDFs)
            if let pageText = pdfPage.string, !pageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                allText.append("--- Page \(pageIndex + 1) ---\n\(pageText)")
                continue
            }

            // Fallback to OCR for scanned PDFs
            let pageRect = pdfPage.bounds(for: .mediaBox)
            let scale: CGFloat = 2.0

            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                      data: nil,
                      width: Int(pageRect.width * scale),
                      height: Int(pageRect.height * scale),
                      bitsPerComponent: 8,
                      bytesPerRow: 0,
                      space: colorSpace,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  ) else { continue }

            context.setFillColor(CGColor.white)
            context.fill(CGRect(origin: .zero, size: CGSize(width: pageRect.width * scale, height: pageRect.height * scale)))
            context.scaleBy(x: scale, y: scale)
            context.translateBy(x: 0, y: pageRect.height)
            context.scaleBy(x: 1.0, y: -1.0)
            pdfPage.draw(with: .mediaBox, to: context)

            guard let cgImage = context.makeImage() else { continue }

            let requestHandler = VNImageRequestHandler(cgImage: cgImage)
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true

            try? requestHandler.perform([request])

            if let observations = request.results {
                let pageText = observations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
                if !pageText.isEmpty {
                    allText.append("--- Page \(pageIndex + 1) ---\n\(pageText)")
                }
            }
        }

        guard !allText.isEmpty else {
            throw VisionError.noTextFound
        }

        return allText.joined(separator: "\n\n")
    }

    /// Extract text from a remote URL (image or PDF)
    func extractTextFromURL(_ urlString: String) async throws -> String {
        guard let url = URL(string: urlString) else {
            throw VisionError.unsupportedFormat
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        // Check content type for PDFs
        let contentType = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") ?? ""
        if contentType.contains("pdf") || urlString.lowercased().hasSuffix(".pdf") {
            return try await extractTextFromPDFData(data)
        }

        // Process as image
        let requestHandler = VNImageRequestHandler(data: data)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        try requestHandler.perform([request])

        guard let observations = request.results, !observations.isEmpty else {
            throw VisionError.noTextFound
        }

        return observations.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }

    func extractText(from filePath: String) async throws -> String {
        // Validate file existence
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw VisionError.fileNotFound
        }
        
        // Validate file extension
        let fileExtension = url.pathExtension.lowercased()
        guard supportedExtensions.contains(fileExtension) else {
            throw VisionError.unsupportedFormat
        }
        
        // Create request handler based on file type
        let requestHandler: VNImageRequestHandler
        
        if fileExtension == "pdf" {
            // Handle PDF files
            requestHandler = try createPDFRequestHandler(from: url)
        } else {
            // Handle image files
            let imageData: Data
            do {
                imageData = try Data(contentsOf: url)
            } catch {
                throw VisionError.imageLoadingFailed
            }
            requestHandler = VNImageRequestHandler(data: imageData)
        }
        
        // Create vision request
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        // Perform text recognition
        do {
            try requestHandler.perform([request])
        } catch {
            throw VisionError.textRecognitionFailed(error.localizedDescription)
        }
        
        // Extract recognized text
        guard let observations = request.results else {
            throw VisionError.noTextFound
        }
        
        let recognizedStrings = observations.compactMap { observation in
            observation.topCandidates(1).first?.string
        }
        
        guard !recognizedStrings.isEmpty else {
            throw VisionError.noTextFound
        }
        
        return recognizedStrings.joined(separator: "\n")
    }
    
    func extractTextWithDetails(from filePath: String) async throws -> VisionResult {
        // Validate file existence
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw VisionError.fileNotFound
        }
        
        // Validate file extension
        let fileExtension = url.pathExtension.lowercased()
        guard supportedExtensions.contains(fileExtension) else {
            throw VisionError.unsupportedFormat
        }
        
        // Create request handler based on file type
        let requestHandler: VNImageRequestHandler
        
        if fileExtension == "pdf" {
            // Handle PDF files
            requestHandler = try createPDFRequestHandler(from: url)
        } else {
            // Handle image files
            let imageData: Data
            do {
                imageData = try Data(contentsOf: url)
            } catch {
                throw VisionError.imageLoadingFailed
            }
            requestHandler = VNImageRequestHandler(data: imageData)
        }
        
        // Create vision request
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        // Perform text recognition
        do {
            try requestHandler.perform([request])
        } catch {
            throw VisionError.textRecognitionFailed(error.localizedDescription)
        }
        
        // Extract recognized text with details
        guard let observations = request.results else {
            throw VisionError.noTextFound
        }
        
        let textBlocks = observations.compactMap { observation -> TextBlock? in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            
            return TextBlock(
                text: candidate.string,
                confidence: candidate.confidence,
                boundingBox: observation.boundingBox
            )
        }
        
        guard !textBlocks.isEmpty else {
            throw VisionError.noTextFound
        }
        
        let fullText = textBlocks.map { $0.text }.joined(separator: "\n")
        
        return VisionResult(
            fullText: fullText,
            textBlocks: textBlocks,
            filePath: filePath
        )
    }
    
    func extractTables(from filePath: String) async throws -> [TableResult] {
        // Validate file existence
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw VisionError.fileNotFound
        }
        
        // Validate file extension
        let fileExtension = url.pathExtension.lowercased()
        guard supportedExtensions.contains(fileExtension) else {
            throw VisionError.unsupportedFormat
        }
        
        // Create request handler based on file type
        let requestHandler: VNImageRequestHandler
        
        if fileExtension == "pdf" {
            // Handle PDF files
            requestHandler = try createPDFRequestHandler(from: url)
        } else {
            // Handle image files
            let imageData: Data
            do {
                imageData = try Data(contentsOf: url)
            } catch {
                throw VisionError.imageLoadingFailed
            }
            requestHandler = VNImageRequestHandler(data: imageData)
        }
        
        // First, try document segmentation to identify potential table regions
        let segmentationRequest = VNDetectDocumentSegmentationRequest()
        
        do {
            try requestHandler.perform([segmentationRequest])
        } catch {
            throw VisionError.documentSegmentationFailed(error.localizedDescription)
        }
        
        // Get text recognition for the entire document
        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = .accurate
        textRequest.usesLanguageCorrection = true
        
        do {
            try requestHandler.perform([textRequest])
        } catch {
            throw VisionError.textRecognitionFailed(error.localizedDescription)
        }
        
        guard let textObservations = textRequest.results else {
            throw VisionError.noTextFound
        }
        
        // Convert observations to text blocks with positions
        let textBlocks = textObservations.compactMap { observation -> PositionedTextBlock? in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            
            return PositionedTextBlock(
                text: candidate.string,
                confidence: candidate.confidence,
                boundingBox: observation.boundingBox
            )
        }
        
        // Analyze spatial relationships to detect tables
        let tableAnalyzer = TableAnalyzer()
        let detectedTables = tableAnalyzer.detectTables(from: textBlocks)
        
        guard !detectedTables.isEmpty else {
            throw VisionError.noTablesFound
        }
        
        return detectedTables
    }
    
    func debugRawDetection(from filePath: String) async throws -> String {
        // Validate file existence
        let url = URL(fileURLWithPath: filePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw VisionError.fileNotFound
        }
        
        // Validate file extension
        let fileExtension = url.pathExtension.lowercased()
        guard supportedExtensions.contains(fileExtension) else {
            throw VisionError.unsupportedFormat
        }
        
        // Create request handler based on file type
        let requestHandler: VNImageRequestHandler
        
        if fileExtension == "pdf" {
            // Handle PDF files
            requestHandler = try createPDFRequestHandler(from: url)
        } else {
            // Handle image files
            let imageData: Data
            do {
                imageData = try Data(contentsOf: url)
            } catch {
                throw VisionError.imageLoadingFailed
            }
            requestHandler = VNImageRequestHandler(data: imageData)
        }
        
        // Get text recognition for the entire document
        let textRequest = VNRecognizeTextRequest()
        textRequest.recognitionLevel = .accurate
        textRequest.usesLanguageCorrection = true
        
        do {
            try requestHandler.perform([textRequest])
        } catch {
            throw VisionError.textRecognitionFailed(error.localizedDescription)
        }
        
        guard let textObservations = textRequest.results else {
            throw VisionError.noTextFound
        }
        
        // Create debug output with all detected text blocks and their positions
        var debugOutput = "=== RAW VISION FRAMEWORK DETECTION ===\n"
        debugOutput += "Total text blocks detected: \(textObservations.count)\n\n"
        
        // Sort by Y position (top to bottom) for better readability
        let sortedObservations = textObservations.sorted { first, second in
            if abs(first.boundingBox.origin.y - second.boundingBox.origin.y) < 0.01 {
                return first.boundingBox.origin.x < second.boundingBox.origin.x
            }
            return first.boundingBox.origin.y > second.boundingBox.origin.y
        }
        
        for (index, observation) in sortedObservations.enumerated() {
            guard let candidate = observation.topCandidates(1).first else { continue }
            
            let text = candidate.string
            let confidence = candidate.confidence
            let boundingBox = observation.boundingBox
            
            debugOutput += "Block \(index + 1):\n"
            debugOutput += "  Text: \"\(text)\"\n"
            debugOutput += "  Confidence: \(String(format: "%.3f", confidence))\n"
            debugOutput += "  Position: x=\(String(format: "%.3f", boundingBox.origin.x)), y=\(String(format: "%.3f", boundingBox.origin.y))\n"
            debugOutput += "  Size: width=\(String(format: "%.3f", boundingBox.width)), height=\(String(format: "%.3f", boundingBox.height))\n"
            debugOutput += "\n"
        }
        
        return debugOutput
    }
    
    private func createPDFRequestHandler(from url: URL) throws -> VNImageRequestHandler {
        guard let pdfDocument = PDFDocument(url: url) else {
            throw VisionError.imageLoadingFailed
        }
        
        guard pdfDocument.pageCount > 0 else {
            throw VisionError.imageLoadingFailed
        }
        
        // Use the first page of the PDF
        guard let pdfPage = pdfDocument.page(at: 0) else {
            throw VisionError.imageLoadingFailed
        }
        
        // Convert PDF page to image using macOS APIs
        let pageRect = pdfPage.bounds(for: .mediaBox)
        let scale: CGFloat = 2.0 // Use 2x scale for better quality
        let scaledSize = CGSize(width: pageRect.width * scale, height: pageRect.height * scale)
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil,
                                    width: Int(scaledSize.width),
                                    height: Int(scaledSize.height),
                                    bitsPerComponent: 8,
                                    bytesPerRow: 0,
                                    space: colorSpace,
                                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw VisionError.imageLoadingFailed
        }
        
        // Set white background
        context.setFillColor(CGColor.white)
        context.fill(CGRect(origin: .zero, size: scaledSize))
        
        // Scale and render PDF page
        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: 0, y: pageRect.height)
        context.scaleBy(x: 1.0, y: -1.0)
        pdfPage.draw(with: .mediaBox, to: context)
        
        guard let cgImage = context.makeImage() else {
            throw VisionError.imageLoadingFailed
        }
        
        return VNImageRequestHandler(cgImage: cgImage)
    }
}

struct TextBlock {
    let text: String
    let confidence: Float
    let boundingBox: CGRect
}

struct VisionResult {
    let fullText: String
    let textBlocks: [TextBlock]
    let filePath: String
}

struct PositionedTextBlock {
    let text: String
    let confidence: Float
    let boundingBox: CGRect
}

struct TableResult {
    let rows: [[String]]
    let columnCount: Int
    let averageConfidence: Float
    let boundingBox: CGRect
    
    var csvData: String {
        let cleanedRows = rows.compactMap { row -> String? in
            // Remove trailing empty cells
            var cleanRow = row
            while cleanRow.last?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == true {
                cleanRow.removeLast()
            }
            
            // Skip rows that are completely empty or only contain empty strings
            let hasContent = cleanRow.contains { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            guard hasContent else { return nil }
            
            // Format each cell for CSV
            let csvRow = cleanRow.map { cell in
                let trimmedCell = cell.trimmingCharacters(in: .whitespacesAndNewlines)
                // Escape quotes and wrap in quotes if contains comma or quote
                if trimmedCell.contains(",") || trimmedCell.contains("\"") || trimmedCell.contains("\n") {
                    return "\"" + trimmedCell.replacingOccurrences(of: "\"", with: "\"\"") + "\""
                }
                return trimmedCell
            }.joined(separator: ",")
            
            return csvRow
        }
        
        return cleanedRows.joined(separator: "\n")
    }
}

class TableAnalyzer {
    func detectTables(from textBlocks: [PositionedTextBlock]) -> [TableResult] {
        // Sort text blocks by Y position (top to bottom), then X position (left to right)
        let sortedBlocks = textBlocks.sorted { first, second in
            if abs(first.boundingBox.origin.y - second.boundingBox.origin.y) < 0.01 {
                return first.boundingBox.origin.x < second.boundingBox.origin.x
            }
            return first.boundingBox.origin.y > second.boundingBox.origin.y // Note: Y coordinates are flipped in Vision
        }
        
        // Group text blocks into potential rows based on Y alignment
        let rowGroups = groupIntoRows(sortedBlocks)
        
        // Detect table structures from row groups
        return detectTablesFromRows(rowGroups)
    }
    
    private func groupIntoRows(_ blocks: [PositionedTextBlock]) -> [[PositionedTextBlock]] {
        var rowGroups: [[PositionedTextBlock]] = []
        let yTolerance: Float = 0.02 // 2% tolerance for Y alignment
        
        for block in blocks {
            let blockY = Float(block.boundingBox.origin.y)
            
            // Try to find an existing row with similar Y coordinate
            if let existingRowIndex = rowGroups.firstIndex(where: { rowGroup in
                let rowY = Float(rowGroup.first?.boundingBox.origin.y ?? 0)
                return abs(blockY - rowY) <= yTolerance
            }) {
                rowGroups[existingRowIndex].append(block)
            } else {
                // Create a new row
                rowGroups.append([block])
            }
        }
        
        // Sort blocks within each row by X position
        for i in 0..<rowGroups.count {
            rowGroups[i].sort { $0.boundingBox.origin.x < $1.boundingBox.origin.x }
        }
        
        return rowGroups
    }
    
    private func detectTablesFromRows(_ rowGroups: [[PositionedTextBlock]]) -> [TableResult] {
        var tables: [TableResult] = []
        
        // Look for sequences of rows that could form tables
        var currentTableRows: [[PositionedTextBlock]] = []
        var maxColumnsSeen = 0
        
        for rowGroup in rowGroups {
            let currentColumnCount = rowGroup.count
            
            // A table row should have at least 1 column (allowing for sparse rows)
            if currentColumnCount >= 1 {
                if currentTableRows.isEmpty {
                    // Start a new table
                    currentTableRows = [rowGroup]
                    maxColumnsSeen = currentColumnCount
                } else {
                    // Check if this row could belong to the current table
                    let shouldContinueTable = isRowPartOfTable(rowGroup, comparedTo: currentTableRows, maxColumns: maxColumnsSeen)
                    
                    if shouldContinueTable {
                        currentTableRows.append(rowGroup)
                        maxColumnsSeen = max(maxColumnsSeen, currentColumnCount)
                    } else {
                        // Finalize the current table if it has enough rows
                        if currentTableRows.count >= 2 && maxColumnsSeen >= 2 {
                            if let table = createTable(from: currentTableRows) {
                                tables.append(table)
                            }
                        }
                        
                        // Start a new potential table
                        currentTableRows = [rowGroup]
                        maxColumnsSeen = currentColumnCount
                    }
                }
            } else {
                // Empty row or single column - finalize current table
                if currentTableRows.count >= 2 && maxColumnsSeen >= 2 {
                    if let table = createTable(from: currentTableRows) {
                        tables.append(table)
                    }
                }
                currentTableRows = []
                maxColumnsSeen = 0
            }
        }
        
        // Don't forget the last table
        if currentTableRows.count >= 2 && maxColumnsSeen >= 2 {
            if let table = createTable(from: currentTableRows) {
                tables.append(table)
            }
        }
        
        return tables
    }
    
    private func isRowPartOfTable(_ newRow: [PositionedTextBlock], comparedTo existingRows: [[PositionedTextBlock]], maxColumns: Int) -> Bool {
        guard !existingRows.isEmpty else { return true }
        
        // Check if the new row's text blocks align spatially with existing rows
        let existingXPositions = existingRows.flatMap { row in 
            row.map { Float($0.boundingBox.origin.x) }
        }.sorted()
        
        let newRowXPositions = newRow.map { Float($0.boundingBox.origin.x) }.sorted()
        
        // If most of the new row's positions align with existing positions, it's likely part of the same table
        let alignmentCount = newRowXPositions.filter { newX in
            existingXPositions.contains { existingX in
                abs(newX - existingX) <= 0.05 // 5% tolerance for alignment
            }
        }.count
        
        // Consider it part of the table if at least 50% of positions align, or if column count is reasonable
        let alignmentRatio = Float(alignmentCount) / Float(newRowXPositions.count)
        let columnCountReasonable = newRow.count <= maxColumns * 2 // Allow up to 2x the max columns seen
        
        return alignmentRatio >= 0.3 || columnCountReasonable
    }
    
    private func createTable(from rowGroups: [[PositionedTextBlock]]) -> TableResult? {
        guard rowGroups.count >= 2 else { return nil }
        
        // Use the first row (usually the header) to establish column structure
        let headerRow = rowGroups.first!
        guard headerRow.count >= 2 else { return nil }
        
        // Calculate column positions based on all text blocks, but prioritize the header row structure
        let columnPositions = calculateColumnPositions(from: rowGroups, prioritizeRow: headerRow)
        let maxColumns = columnPositions.count
        
        // Convert to string matrix, aligning columns based on calculated positions
        var tableData: [[String]] = []
        var allConfidences: [Float] = []
        
        for rowGroup in rowGroups {
            var row: [String] = Array(repeating: "", count: maxColumns)
            
            for block in rowGroup {
                // Find the best column for this block based on its X position
                let blockX = Float(block.boundingBox.origin.x)
                let columnIndex = findBestColumn(for: blockX, in: columnPositions)
                
                if columnIndex < maxColumns {
                    row[columnIndex] = block.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    allConfidences.append(block.confidence)
                }
            }
            
            tableData.append(row)
        }
        
        // Calculate bounding box encompassing all table cells
        let allBlocks = rowGroups.flatMap { $0 }
        let minX = allBlocks.map { $0.boundingBox.origin.x }.min() ?? 0
        let maxX = allBlocks.map { $0.boundingBox.maxX }.max() ?? 0
        let minY = allBlocks.map { $0.boundingBox.origin.y }.min() ?? 0
        let maxY = allBlocks.map { $0.boundingBox.maxY }.max() ?? 0
        
        let tableBoundingBox = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        
        let averageConfidence = allConfidences.isEmpty ? 0.0 : allConfidences.reduce(0, +) / Float(allConfidences.count)
        
        return TableResult(
            rows: tableData,
            columnCount: maxColumns,
            averageConfidence: averageConfidence,
            boundingBox: tableBoundingBox
        )
    }
    
    private func calculateColumnPositions(from rowGroups: [[PositionedTextBlock]], prioritizeRow: [PositionedTextBlock]) -> [Float] {
        // Start with positions from the priority row (header)
        var columnPositions = prioritizeRow.map { Float($0.boundingBox.origin.x) }.sorted()
        
        // Collect all other X positions from remaining rows
        var allXPositions: [Float] = []
        for rowGroup in rowGroups {
            for block in rowGroup {
                allXPositions.append(Float(block.boundingBox.origin.x))
            }
        }
        allXPositions.sort()
        
        // Use a dynamic tolerance based on the average distance between header columns
        let dynamicTolerance = calculateDynamicTolerance(for: columnPositions)
        
        // Add additional column positions if they don't conflict with existing ones
        for position in allXPositions {
            let tooClose = columnPositions.contains { abs(position - $0) <= dynamicTolerance }
            if !tooClose {
                columnPositions.append(position)
            }
        }
        
        return columnPositions.sorted()
    }
    
    private func calculateDynamicTolerance(for positions: [Float]) -> Float {
        guard positions.count > 1 else { return 0.01 }
        
        var distances: [Float] = []
        for i in 1..<positions.count {
            distances.append(positions[i] - positions[i-1])
        }
        
        let avgDistance = distances.reduce(0, +) / Float(distances.count)
        // Use 25% of average distance as tolerance, with min/max bounds
        return max(0.005, min(0.03, avgDistance * 0.25))
    }
    
    private func findBestColumn(for xPosition: Float, in columnPositions: [Float]) -> Int {
        var bestColumn = 0
        var minDistance = Float.greatestFiniteMagnitude
        
        for (index, columnX) in columnPositions.enumerated() {
            let distance = abs(xPosition - columnX)
            if distance < minDistance {
                minDistance = distance
                bestColumn = index
            }
        }
        
        return bestColumn
    }
}