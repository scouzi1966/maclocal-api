import ArgumentParser
import Foundation

struct VisionCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Extract text, barcodes, and classifications from images using Apple's Vision framework",
        discussion: """
        ---
        name: afm-vision
        description: Extract text, tables, barcodes, classifications, and saliency from images and documents using Apple's Vision framework. Runs locally on-device with no network access required.
        tags: [vision, ocr, text-extraction, table-extraction, barcode, classify, saliency, pdf, image, csv, apple-vision, on-device]
        repository: https://github.com/scouzi1966/maclocal-api
        supported_formats: [PNG, JPG, JPEG, HEIC, PDF]
        modes: [text, table, barcode, classify, saliency, auto]
        triggers:
          - extract text from image
          - OCR document
          - extract table from image
          - convert image to text
          - PDF text extraction
          - scan barcode
          - classify image
          - detect saliency
        examples:
          - afm vision -f image.png
          - afm vision --file /path/to/document.pdf
          - afm vision -f report.png --table
          - afm vision -f invoice.pdf -t --verbose
          - afm vision -f photo.jpg --mode classify
          - afm vision -f photo.jpg --mode barcode
          - afm vision -f photo.jpg --mode saliency --heat-map
          - afm vision -f photo.jpg --mode auto
          - afm vision -f doc.png --auto-crop
          - afm vision -f doc.png --detail low
        ---

        Use Apple's Vision framework to perform OCR, barcode detection, image classification,
        and saliency analysis on images and documents.

        Supported formats: PNG, JPG, JPEG, HEIC, PDF

        Modes:
          text       Extract text (default, requires macOS 26)
          table      Extract tables as CSV (requires macOS 26)
          barcode    Detect barcodes and QR codes
          classify   Classify image content with labels
          saliency   Detect salient regions (attention or objectness)
          auto       Run text + barcode + classify together

        Examples:
          afm vision -f image.png                              # Extract text
          afm vision -f image.png --mode barcode               # Detect barcodes
          afm vision -f photo.jpg --mode classify              # Classify image
          afm vision -f photo.jpg --mode saliency --heat-map   # Saliency with heat map
          afm vision -f photo.jpg --mode auto                  # Run all modes
          afm vision -f doc.png --auto-crop                    # Auto-crop before OCR
          afm vision -f doc.png --detail low                   # Fast recognition
        """
    )

    @Option(name: [.short, .long], help: "Path to the image or document file")
    var file: String

    @Flag(name: .long, help: "Show detailed output with confidence scores and bounding boxes")
    var verbose: Bool = false

    @Flag(name: [.customShort("t"), .long], help: "Extract tables and output as CSV format")
    var table: Bool = false

    @Flag(name: [.customShort("D")], help: .hidden)
    var debug: Bool = false

    @Flag(name: .long, help: "Print machine-readable JSON capability card for AI agents and exit")
    var helpJson: Bool = false

    // New flags
    @Option(name: .long, help: "Vision mode: text, table, barcode, classify, saliency, auto (default: text)")
    var mode: String?

    @Option(name: .long, help: "Recognition detail: high or low (default: high)")
    var detail: String = "high"

    @Flag(name: .long, help: "Auto-crop document region before processing")
    var autoCrop: Bool = false

    @Option(name: .long, help: "Output format: json, text (default: text for CLI)")
    var format: String = "text"

    @Option(name: .long, help: "Max classification labels to return (default: 5)")
    var maxLabels: Int = 5

    @Option(name: .long, help: "Saliency type: attention or objectness (default: attention)")
    var saliencyType: String = "attention"

    @Flag(name: .long, help: "Include saliency heat map as base64 PNG")
    var heatMap: Bool = false

    func run() async throws {
        if helpJson {
            printHelpJson(command: "afm vision")
            return
        }

        guard !file.isEmpty else {
            print("Error: File path is required. Use -f or --file to specify the input file.")
            throw ExitCode.failure
        }

        let expandedPath = NSString(string: file).expandingTildeInPath
        let resolvedPath = URL(fileURLWithPath: expandedPath).standardized.path

        // Validate options
        let validModes: Set<String> = ["text", "table", "barcode", "classify", "saliency", "auto", "debug"]
        let validFormats: Set<String> = ["text", "json"]
        let validSaliencyTypes: Set<String> = ["attention", "objectness"]
        let validDetails: Set<String> = ["high", "low"]

        if let mode = mode, !validModes.contains(mode.lowercased()) {
            print("Error: Unknown mode '\(mode)'. Supported: \(validModes.sorted().joined(separator: ", "))")
            throw ExitCode.failure
        }
        if !validFormats.contains(format.lowercased()) {
            print("Error: Unknown format '\(format)'. Supported: \(validFormats.sorted().joined(separator: ", "))")
            throw ExitCode.failure
        }
        if !validSaliencyTypes.contains(saliencyType.lowercased()) {
            print("Error: Unknown saliency-type '\(saliencyType)'. Supported: \(validSaliencyTypes.sorted().joined(separator: ", "))")
            throw ExitCode.failure
        }
        if !validDetails.contains(detail.lowercased()) {
            print("Error: Unknown detail '\(detail)'. Supported: \(validDetails.sorted().joined(separator: ", "))")
            throw ExitCode.failure
        }

        // Resolve effective mode: --mode takes precedence; --table is legacy fallback
        let effectiveMode: String
        if let mode = mode {
            if table {
                fputs("Warning: --table ignored because --mode \(mode) was specified\n", stderr)
            }
            effectiveMode = mode.lowercased()
        } else if table {
            effectiveMode = "table"
        } else {
            effectiveMode = "text"
        }

        do {
            let visionService = VisionService()

            // Apply auto-crop preprocessing
            var processPath = resolvedPath
            var tempCropURL: URL?
            if autoCrop {
                let imageData = try Data(contentsOf: URL(fileURLWithPath: resolvedPath))
                if imageData.count > VisionRequestOptions.defaultMaxFileBytes {
                    throw VisionError.fileTooLarge(bytes: imageData.count, maxBytes: VisionRequestOptions.defaultMaxFileBytes)
                }
                let croppedData = try visionService.autoCrop(imageData: imageData)
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("afm_vision_crop_\(UUID().uuidString).png")
                try croppedData.write(to: tempURL)
                processPath = tempURL.path
                tempCropURL = tempURL
            }
            defer { if let url = tempCropURL { try? FileManager.default.removeItem(at: url) } }

            // Map detail to recognition level
            let recognitionLevel: VisionRecognitionLevel = detail.lowercased() == "low" ? .fast : .accurate
            let options = VisionRequestOptions(recognitionLevel: recognitionLevel)

            let output: String

            switch effectiveMode {
            case "barcode":
                let results = try visionService.detectBarcodes(from: processPath, options: options)
                if results.isEmpty {
                    output = "No barcodes found."
                } else if format == "json" {
                    let items = results.map { r in
                        ["type": r.type, "payload": r.payload, "confidence": String(format: "%.2f", r.confidence)]
                    }
                    let data = try JSONSerialization.data(withJSONObject: items, options: [.prettyPrinted])
                    output = String(data: data, encoding: .utf8) ?? "[]"
                } else {
                    output = results.map { r in
                        "\(r.type): \(r.payload) (confidence: \(String(format: "%.2f", r.confidence)))"
                    }.joined(separator: "\n")
                }

            case "classify":
                let result = try visionService.classifyImage(from: processPath, maxLabels: maxLabels)
                if result.labels.isEmpty {
                    output = "No classifications found."
                } else if format == "json" {
                    let items = result.labels.map { l in
                        ["label": l.label, "confidence": String(format: "%.4f", l.confidence)]
                    }
                    let data = try JSONSerialization.data(withJSONObject: items, options: [.prettyPrinted])
                    output = String(data: data, encoding: .utf8) ?? "[]"
                } else {
                    output = result.labels.map { l in
                        "\(l.label): \(String(format: "%.4f", l.confidence))"
                    }.joined(separator: "\n")
                }

            case "saliency":
                let result = try visionService.detectSaliency(from: processPath, type: saliencyType, includeHeatMap: heatMap)
                if result.regions.isEmpty {
                    output = "No salient regions found."
                } else {
                    var lines = result.regions.map { r in
                        let box = r.boundingBox
                        return "\(r.type): x=\(String(format: "%.3f", box.origin.x)), y=\(String(format: "%.3f", box.origin.y)), w=\(String(format: "%.3f", box.width)), h=\(String(format: "%.3f", box.height))"
                    }
                    if let heatMapData = result.heatMapPNG {
                        lines.append("heat_map_base64: \(heatMapData.base64EncodedString())")
                    }
                    output = lines.joined(separator: "\n")
                }

            case "auto":
                var sections: [String] = []

                // Barcode and classify are image-only. Skip them for PDFs so
                // `--mode auto` still runs OCR on documents.
                let fileExt = URL(fileURLWithPath: processPath).pathExtension.lowercased()
                let isImage = VisionService.imageOnlyExtensions.contains(fileExt)

                if isImage {
                    let barcodes = try visionService.detectBarcodes(from: processPath, options: options)
                    if !barcodes.isEmpty {
                        sections.append("=== Barcodes ===\n" + barcodes.map { "\($0.type): \($0.payload)" }.joined(separator: "\n"))
                    }

                    let classified = try visionService.classifyImage(from: processPath, maxLabels: maxLabels)
                    if !classified.labels.isEmpty {
                        sections.append("=== Classifications ===\n" + classified.labels.map { "\($0.label): \(String(format: "%.4f", $0.confidence))" }.joined(separator: "\n"))
                    }
                }

                if #available(macOS 26.0, *) {
                    do {
                        let text = try await visionService.extractText(from: processPath, options: options)
                        sections.append("=== Text ===\n" + text)
                    } catch VisionError.noTextFound {
                        // Auto mode: an image with no text is fine as long as
                        // another submode produced something.
                    }
                }

                output = sections.isEmpty ? "No results found." : sections.joined(separator: "\n\n")

            case "debug":
                if #available(macOS 26.0, *) {
                    output = try await visionService.debugRawDetection(from: processPath, options: options)
                } else {
                    throw VisionError.modeRequiresMacOS26("debug")
                }

            case "table":
                if #available(macOS 26.0, *) {
                    let tableResults = try await visionService.extractTables(from: processPath, options: options)
                    if verbose {
                        var verboseOutput = "File: \(resolvedPath)\n"
                        verboseOutput += "Found \(tableResults.count) table(s)\n\n"
                        for (index, tbl) in tableResults.enumerated() {
                            verboseOutput += "Table \(index + 1):\n"
                            verboseOutput += "Rows: \(tbl.rows.count), Columns: \(tbl.columnCount)\n"
                            verboseOutput += "Confidence: \(String(format: "%.2f", tbl.averageConfidence))\n\n"
                            verboseOutput += tbl.csvData
                            if index < tableResults.count - 1 {
                                verboseOutput += "\n" + String(repeating: "-", count: 50) + "\n\n"
                            }
                        }
                        output = verboseOutput
                    } else {
                        let csvSections = tableResults.enumerated().map { (index, tbl) in
                            "Detected Table \(index + 1)\n" + tbl.csvData
                        }
                        let csvOutput = csvSections.joined(separator: "\n")
                        output = csvOutput.isEmpty ? "No tables found in the document." : csvOutput
                    }
                } else {
                    throw VisionError.modeRequiresMacOS26("table")
                }

            default:
                // text mode (default)
                if #available(macOS 26.0, *) {
                    if verbose {
                        let visionResult = try await visionService.extractTextWithDetails(from: processPath, options: options)
                        var verboseOutput = "File: \(visionResult.filePath)\n"
                        verboseOutput += "Text Content:\n\n"
                        verboseOutput += visionResult.fullText
                        verboseOutput += "\n\nDetails:\n"
                        for (index, block) in visionResult.textBlocks.enumerated() {
                            verboseOutput += "Block \(index + 1): \"\(block.text)\" (confidence: \(String(format: "%.2f", block.confidence)))\n"
                        }
                        output = verboseOutput
                    } else {
                        output = try await visionService.extractText(from: processPath, options: options)
                    }
                } else {
                    throw VisionError.modeRequiresMacOS26("text")
                }
            }

            print(output)
        } catch {
            if let visionError = error as? VisionError {
                print("Error: \(visionError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
            throw ExitCode.failure
        }
    }
}
