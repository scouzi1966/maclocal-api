#if canImport(FoundationModels)
import Foundation
import AFMKit
import AFMOpenAICompat
import FoundationModels
import ImageIO
import UniformTypeIdentifiers

@available(macOS 27.0, *)
enum MLXFoundationRequestAdapter {
    static func messages(from transcript: Transcript) throws -> [Message] {
        var messages: [Message] = []

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                if let content = try messageContent(from: instructions.segments) {
                    messages.append(Message(role: "system", content: content))
                }
            case .prompt(let prompt):
                if let content = try messageContent(from: prompt.segments) {
                    messages.append(Message(role: "user", content: content))
                }
            case .response(let response):
                if let content = try messageContent(from: response.segments) {
                    messages.append(Message(role: "assistant", content: content))
                }
            case .reasoning(let reasoning):
                if let content = try messageContent(from: reasoning.segments) {
                    messages.append(Message(role: "assistant", content: content))
                }
            case .toolCalls(let toolCalls):
                messages.append(
                    Message(
                        role: "assistant",
                        content: nil,
                        toolCalls: toolCalls.map { call in
                            MessageToolCall(
                                id: call.id,
                                type: "function",
                                function: MessageToolCallFunction(
                                    name: call.toolName,
                                    arguments: call.arguments.jsonString
                                )
                            )
                        }
                    )
                )
            case .toolOutput(let output):
                messages.append(
                    Message(
                        role: "tool",
                        content: try messageContent(from: output.segments),
                        toolCallId: output.id,
                        name: output.toolName
                    )
                )
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [entry],
                        debugDescription: "The transcript contains an unknown entry type."
                    )
                )
            }
        }

        return messages
    }

    static func generationConfig(
        from request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws -> GenerationConfig {
        var temperature = request.generationOptions.temperature
        var topP: Double?
        var topK: Int?
        var seed: Int?
        if let kind = request.generationOptions.samplingMode?.kind {
            switch kind {
            case .greedy:
                temperature = 0
            case .randomTopK(let value, let randomSeed):
                topK = value
                seed = randomSeed.flatMap { Int(exactly: $0) }
            case .randomProbabilityThreshold(let value, let randomSeed):
                topP = value
                seed = randomSeed.flatMap { Int(exactly: $0) }
            @unknown default:
                break
            }
        }

        let toolCallingMode: String?
        switch request.generationOptions.toolCallingMode?.kind {
        case .allowed: toolCallingMode = "allowed"
        case .required: toolCallingMode = "required"
        case .disallowed: toolCallingMode = "disallowed"
        case nil: toolCallingMode = nil
        @unknown default: toolCallingMode = nil
        }

        var metadata: [String: AFMJSONValue] = [
            "includeSchemaInPrompt": .bool(
                request.contextOptions.includeSchemaInPrompt ?? true
            )
        ]
        if let toolCallingMode {
            metadata["toolCallingMode"] = .string(toolCallingMode)
        }
        if let reasoningLevel = request.contextOptions.reasoningLevel {
            switch reasoningLevel {
            case .light: metadata["reasoningLevel"] = .string("light")
            case .moderate: metadata["reasoningLevel"] = .string("moderate")
            case .deep: metadata["reasoningLevel"] = .string("deep")
            case .custom(let value):
                metadata["reasoningLevel"] = .string(value)
            @unknown default:
                break
            }
        }
        metadata.merge(afmMetadata(request.metadata)) { _, new in new }

        let definitions = request.generationOptions.toolCallingMode?.kind == .disallowed
            ? []
            : request.enabledToolDefinitions
        return GenerationConfig(
            temperature: temperature,
            maxTokens: request.generationOptions.maximumResponseTokens
                ?? model.engineConfig.defaultMaximumResponseTokens,
            topP: topP,
            topK: topK,
            seed: seed,
            tools: try tools(from: definitions),
            responseFormat: try responseFormat(from: request.schema),
            metadata: metadata
        )
    }

    static func tools(
        from definitions: [Transcript.ToolDefinition]
    ) throws -> [RequestTool]? {
        guard !definitions.isEmpty else { return nil }
        return try definitions.map { definition in
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: definition.name,
                    description: definition.description,
                    parameters: try anyCodable(from: definition.parameters),
                    strict: true
                )
            )
        }
    }

    static func responseFormat(from schema: GenerationSchema?) throws -> ResponseFormat? {
        guard let schema else { return nil }
        return ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: schema.name,
                description: nil,
                schema: try anyCodable(from: schema),
                strict: true
            )
        )
    }

    static func foundationMetadata(
        _ values: [String: AFMJSONValue]
    ) -> [String: any Sendable & Codable & Equatable] {
        values.reduce(into: [:]) { result, item in
            switch item.value {
            case .null: result[item.key] = "null"
            case .bool(let value): result[item.key] = value
            case .integer(let value): result[item.key] = value
            case .number(let value): result[item.key] = value
            case .string(let value): result[item.key] = value
            case .array, .object:
                result[item.key] = String(describing: item.value)
            }
        }
    }

    private static func anyCodable(from schema: GenerationSchema) throws -> AnyCodable {
        let data = try JSONEncoder().encode(schema)
        let object = try JSONSerialization.jsonObject(with: data)
        return AnyCodable(object)
    }

    private static func messageContent(
        from segments: [Transcript.Segment]
    ) throws -> MessageContent? {
        let parts = try segments.flatMap { segment -> [ContentPart] in
            switch segment {
            case .text(let text):
                return [ContentPart(type: "text", text: text.content)]
            case .structure(let structure):
                return [ContentPart(type: "text", text: structure.content.jsonString)]
            case .attachment(let attachment):
                var result: [ContentPart] = []
                if let label = attachment.label, !label.isEmpty {
                    result.append(ContentPart(type: "text", text: label))
                }
                switch attachment.content {
                case .image(let image):
                    result.append(
                        ContentPart(
                            type: "image_url",
                            image_url: ImageURL(
                                url: try imageDataURL(image),
                                detail: nil
                            )
                        )
                    )
                @unknown default:
                    throw LanguageModelError.unsupportedTranscriptContent(
                        .init(
                            unsupportedContent: [],
                            debugDescription: "The transcript contains an unknown attachment type."
                        )
                    )
                }
                return result
            case .custom(let custom):
                return [
                    ContentPart(
                        type: "text",
                        text: customContent(custom)
                    )
                ]
            @unknown default:
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: [],
                        debugDescription: "The transcript contains an unknown segment type."
                    )
                )
            }
        }
        guard !parts.isEmpty else { return nil }
        if parts.allSatisfy({ $0.type == "text" }) {
            return .text(parts.compactMap(\.text).joined(separator: "\n"))
        }
        return .parts(parts)
    }

    private static func customContent<C: Transcript.CustomSegment>(
        _ custom: C
    ) -> String {
        if let data = try? JSONEncoder().encode(custom.content),
           let value = String(data: data, encoding: .utf8) {
            return value
        }
        return custom.description
    }

    private static func imageDataURL(
        _ image: Transcript.ImageAttachment
    ) throws -> String {
        if let url = image.url {
            return url.absoluteString
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: [],
                    debugDescription: "Could not create a PNG image destination."
                )
            )
        }
        CGImageDestinationAddImage(destination, image.cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: [],
                    debugDescription: "Could not encode the image attachment as PNG."
                )
            )
        }
        return "data:image/png;base64,\((data as Data).base64EncodedString())"
    }

    private static func afmMetadata(
        _ values: [String: any Sendable & Codable & Equatable]
    ) -> [String: AFMJSONValue] {
        values.reduce(into: [:]) { result, item in
            switch item.value {
            case let value as Bool: result[item.key] = .bool(value)
            case let value as Int: result[item.key] = .integer(value)
            case let value as Double: result[item.key] = .number(value)
            case let value as String: result[item.key] = .string(value)
            default: result[item.key] = .string(String(describing: item.value))
            }
        }
    }
}
#endif
