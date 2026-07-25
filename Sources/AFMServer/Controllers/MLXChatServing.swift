import Foundation
import AFMKit
import AFMOpenAICompat
import MLXLMCommon

typealias ChatGenerationResult = AFMMLXChatGenerationResult
typealias ChatStreamingResult = AFMMLXChatStreamingResult

protocol MLXChatServing:
    AFMMLXAPIProfiling,
    AFMMLXRequestScheduling,
    AFMMLXBatchControlling,
    AFMMLXServingConfigurationProviding,
    AFMMLXOpenAIChatGenerating
{
    var defaultGuidedJsonSchema: ResponseFormat? { get }

    /// Resolve effective response format: per-request format wins, falls back to server default.
    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat?
}

extension MLXChatServing {
    var defaultGuidedJsonSchema: ResponseFormat? { nil }

    func effectiveResponseFormat(requestFormat: ResponseFormat?) -> ResponseFormat? {
        requestFormat ?? defaultGuidedJsonSchema
    }
}

extension MLXModelService: MLXChatServing {}
