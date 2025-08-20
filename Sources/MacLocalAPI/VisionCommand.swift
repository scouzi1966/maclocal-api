import ArgumentParser
import Foundation

struct VisionCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vision",
        abstract: "Extract text from images using Apple's Vision framework",
        discussion: """
        Use Apple's Vision framework to perform OCR (Optical Character Recognition) on images and documents.
        
        Supported formats: PNG, JPG, JPEG, HEIC, PDF
        
        Examples:
          afm vision -f image.png                    # Extract all text
          afm vision --file /path/to/document.pdf    # Extract text from PDF
          afm vision -f report.png --table           # Extract tables as CSV
          afm vision -f invoice.pdf -t --verbose     # Extract tables with details
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
    
    func run() async throws {
        // Validate that the file path was provided
        guard !file.isEmpty else {
            print("Error: File path is required. Use -f or --file to specify the input file.")
            throw ExitCode.failure
        }
        
        // Expand tilde and resolve relative paths
        let expandedPath = NSString(string: file).expandingTildeInPath
        let resolvedPath = URL(fileURLWithPath: expandedPath).standardized.path
        
        // Run vision processing
        do {
            if #available(macOS 26.0, *) {
                let visionService = VisionService()
                
                let output: String
                
                if debug {
                    // Debug mode: output raw Vision framework detection
                    output = try await visionService.debugRawDetection(from: resolvedPath)
                } else if table {
                    // Table extraction mode
                    let tableResults = try await visionService.extractTables(from: resolvedPath)
                    
                    if verbose {
                        var verboseOutput = "📄 File: \(resolvedPath)\n"
                        verboseOutput += "🗂️  Found \(tableResults.count) table(s)\n\n"
                        
                        for (index, table) in tableResults.enumerated() {
                            verboseOutput += "📊 Table \(index + 1):\n"
                            verboseOutput += "Rows: \(table.rows.count), Columns: \(table.columnCount)\n"
                            verboseOutput += "Confidence: \(String(format: "%.2f", table.averageConfidence))\n\n"
                            verboseOutput += table.csvData
                            if index < tableResults.count - 1 {
                                verboseOutput += "\n" + String(repeating: "-", count: 50) + "\n\n"
                            }
                        }
                        
                        output = verboseOutput
                    } else {
                        // Simple CSV output with table headers
                        let csvSections = tableResults.enumerated().map { (index, table) in
                            let tableHeader = "Detected Table \(index + 1)"
                            return tableHeader + "\n" + table.csvData
                        }
                        let csvOutput = csvSections.joined(separator: "\n")
                        output = csvOutput.isEmpty ? "No tables found in the document." : csvOutput
                    }
                } else if verbose {
                    // Get detailed results with confidence scores
                    let visionResult = try await visionService.extractTextWithDetails(from: resolvedPath)
                    var verboseOutput = "📄 File: \(visionResult.filePath)\n"
                    verboseOutput += "📝 Text Content:\n\n"
                    verboseOutput += visionResult.fullText
                    verboseOutput += "\n\n📊 Details:\n"
                    
                    for (index, block) in visionResult.textBlocks.enumerated() {
                        verboseOutput += "Block \(index + 1): \"\(block.text)\" (confidence: \(String(format: "%.2f", block.confidence)))\n"
                    }
                    
                    output = verboseOutput
                } else {
                    // Simple text extraction
                    output = try await visionService.extractText(from: resolvedPath)
                }
                
                print(output)
            } else {
                throw VisionError.textRecognitionFailed("Vision framework requires macOS 26.0 or later")
            }
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