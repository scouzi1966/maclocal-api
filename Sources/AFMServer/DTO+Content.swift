// Retroactive Vapor `Content` conformance for AFMKit's OpenAI DTOs.
//
// AFMKit defines these as pure `Codable, Sendable` (no Vapor dependency). The HTTP layer
// lives here in AFMServer, so we re-add `Content` (Vapor's Codable+Request/ResponseCodable)
// retroactively. Vapor provides default `Content` implementations for any `Codable`, so these
// are empty conformances. Keeping them in AFMServer is what lets a consumer import AFMKit
// alone without pulling Vapor/NIO into their dependency graph.
import Vapor
import AFMKit

extension ChatCompletionRequest: Content {}
extension StreamOptions: Content {}
extension ResponseFormat: Content {}
extension ResponseJsonSchema: Content {}
extension RequestTool: Content {}
extension RequestToolFunction: Content {}
extension Message: Content {}
extension MessageToolCall: Content {}
extension MessageToolCallFunction: Content {}
extension BatchRequestItem: Content {}
extension BatchCompletionRequest: Content {}
extension BatchCreateRequest: Content {}
extension EmbeddingInput: Content {}
extension EmbeddingEncodingFormat: Content {}
extension EmbeddingsRequest: Content {}
extension AFMProfile: Content {}
extension AFMProfileSample: Content {}
extension AFMProfileExtended: Content {}
extension ChatCompletionResponse: Content {}
extension Choice: Content {}
extension ChoiceLogprobs: Content {}
extension TokenLogprobContent: Content {}
extension TopLogprobEntry: Content {}
extension ResponseMessage: Content {}
extension ResponseToolCall: Content {}
extension ResponseToolCallFunction: Content {}
extension PromptTokensDetails: Content {}
extension Usage: Content {}
extension ChatCompletionStreamResponse: Content {}
extension StreamTimings: Content {}
extension StreamChoice: Content {}
extension StreamDelta: Content {}
extension StreamDeltaToolCall: Content {}
extension StreamDeltaFunction: Content {}
extension StreamUsage: Content {}
extension OpenAIError: Content {}
extension OpenAIError.ErrorDetail: Content {}
extension FileObject: Content {}
extension FileDeleteResponse: Content {}
extension BatchRequestCounts: Content {}
extension BatchObject: Content {}
extension BatchError: Content {}
extension BatchListResponse: Content {}
extension EmbeddingVectorPayload: Content {}
extension EmbeddingDataItem: Content {}
extension EmbeddingsUsage: Content {}
extension EmbeddingsResponse: Content {}
