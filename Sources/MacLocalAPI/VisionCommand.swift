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
          afm vision -f image.png
          afm vision --file /path/to/document.pdf
          afm vision -f screenshot.jpg
        """
    )
    
    @Option(name: [.short, .long], help: "Path to the image or document file")
    var file: String
    
    @Flag(name: .long, help: "Show detailed output with confidence scores and bounding boxes")
    var verbose: Bool = false
    
    func run() throws {
        // Validate that the file path was provided
        guard !file.isEmpty else {
            print("Error: File path is required. Use -f or --file to specify the input file.")
            throw ExitCode.failure
        }
        
        // Expand tilde and resolve relative paths
        let expandedPath = NSString(string: file).expandingTildeInPath
        let resolvedPath = URL(fileURLWithPath: expandedPath).standardized.path
        
        // Run vision processing
        let group = DispatchGroup()
        var result: Result<String, Error>?
        
        group.enter()
        Task {
            do {
                if #available(macOS 26.0, *) {
                    let visionService = VisionService()
                    
                    if verbose {
                        // Get detailed results with confidence scores
                        let visionResult = try await visionService.extractTextWithDetails(from: resolvedPath)
                        var output = "📄 File: \(visionResult.filePath)\n"
                        output += "📝 Text Content:\n\n"
                        output += visionResult.fullText
                        output += "\n\n📊 Details:\n"
                        
                        for (index, block) in visionResult.textBlocks.enumerated() {
                            output += "Block \(index + 1): \"\(block.text)\" (confidence: \(String(format: "%.2f", block.confidence)))\n"
                        }
                        
                        result = .success(output)
                    } else {
                        // Simple text extraction
                        let extractedText = try await visionService.extractText(from: resolvedPath)
                        result = .success(extractedText)
                    }
                } else {
                    result = .failure(VisionError.textRecognitionFailed("Vision framework requires macOS 26.0 or later"))
                }
            } catch {
                result = .failure(error)
            }
            group.leave()
        }
        
        group.wait()
        
        switch result {
        case .success(let text):
            print(text)
        case .failure(let error):
            if let visionError = error as? VisionError {
                print("Error: \(visionError.localizedDescription)")
            } else {
                print("Error: \(error.localizedDescription)")
            }
            throw ExitCode.failure
        case .none:
            print("Error: Unexpected error occurred during text extraction")
            throw ExitCode.failure
        }
    }
}