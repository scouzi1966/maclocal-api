import Foundation
import Vision
import CoreImage

enum VisionError: Error, LocalizedError {
    case fileNotFound
    case unsupportedFormat
    case imageLoadingFailed
    case textRecognitionFailed(String)
    case noTextFound
    
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
        }
    }
}

@available(macOS 26.0, *)
class VisionService {
    
    private let supportedExtensions = ["png", "jpg", "jpeg", "heic", "pdf"]
    
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
        
        // Load image data
        let imageData: Data
        do {
            imageData = try Data(contentsOf: url)
        } catch {
            throw VisionError.imageLoadingFailed
        }
        
        // Create vision request
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        // Create request handler
        let requestHandler = VNImageRequestHandler(data: imageData)
        
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
        
        // Load image data
        let imageData: Data
        do {
            imageData = try Data(contentsOf: url)
        } catch {
            throw VisionError.imageLoadingFailed
        }
        
        // Create vision request
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        // Create request handler
        let requestHandler = VNImageRequestHandler(data: imageData)
        
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