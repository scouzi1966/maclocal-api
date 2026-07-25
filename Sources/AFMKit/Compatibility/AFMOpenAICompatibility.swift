import Foundation

extension AFMRequest {
    init(openAIMessages: [Message], generationConfig: GenerationConfig) throws {
        messages = try openAIMessages.map(AFMMessage.init(openAIMessage:))
        tools = (generationConfig.tools ?? []).map { tool in
            AFMToolDefinition(
                name: tool.function.name,
                description: tool.function.description,
                inputSchema: tool.function.parameters?.value.afmJSONValue ?? .object([:])
            )
        }
        options = AFMGenerationOptions(
            temperature: generationConfig.temperature,
            maximumResponseTokens: generationConfig.maxTokens,
            topP: generationConfig.topP,
            topK: generationConfig.topK,
            minP: generationConfig.minP,
            repetitionPenalty: generationConfig.repetitionPenalty,
            presencePenalty: generationConfig.presencePenalty,
            seed: generationConfig.seed,
            stopSequences: generationConfig.stop ?? [],
            responseConstraint: generationConfig.responseFormat?.afmConstraint
        )
        metadata = [:]
    }
}

private extension AFMMessage {
    init(openAIMessage message: Message) throws {
        guard let role = AFMMessageRole(rawValue: message.role) else {
            throw AFMError.invalidRequest("Unsupported message role '\(message.role)'.")
        }

        let parts: [AFMContentPart]
        switch message.content {
        case nil:
            parts = []
        case .text(let text):
            parts = [.text(text)]
        case .parts(let contentParts):
            parts = try contentParts.map(AFMContentPart.init(openAIContentPart:))
        }

        self.init(
            role: role,
            content: parts,
            name: message.name,
            toolCallID: message.toolCallId,
            toolCalls: (message.toolCalls ?? []).map {
                AFMToolCall(
                    id: $0.id,
                    name: $0.function.name,
                    arguments: $0.function.arguments
                )
            }
        )
    }
}

private extension AFMContentPart {
    init(openAIContentPart part: ContentPart) throws {
        switch part.type {
        case "text":
            self = .text(part.text ?? "")
        case "image_url":
            guard let rawValue = part.image_url?.url else {
                throw AFMError.invalidRequest("An image_url content part has no URL.")
            }
            if let parsed = Self.parseDataURL(rawValue) {
                self = .data(mimeType: parsed.mimeType, value: parsed.data)
            } else if let url = URL(string: rawValue) {
                self = .reference(url)
            } else {
                throw AFMError.invalidRequest("Invalid image URL '\(rawValue)'.")
            }
        case "input_audio":
            guard let audio = part.input_audio,
                  let data = Data(base64Encoded: audio.data) else {
                throw AFMError.invalidRequest("An input_audio content part has invalid data.")
            }
            self = .data(mimeType: "audio/\(audio.format)", value: data)
        default:
            throw AFMError.invalidRequest("Unsupported content part '\(part.type)'.")
        }
    }

    static func parseDataURL(_ value: String) -> (mimeType: String, data: Data)? {
        guard value.hasPrefix("data:"),
              let comma = value.firstIndex(of: ",") else {
            return nil
        }
        let header = value[value.index(value.startIndex, offsetBy: 5)..<comma]
        let payload = value[value.index(after: comma)...]
        let fields = header.split(separator: ";")
        guard let mimeType = fields.first.map(String.init),
              fields.dropFirst().contains("base64"),
              let data = Data(base64Encoded: String(payload)) else {
            return nil
        }
        return (mimeType, data)
    }
}

private extension AnyCodableValue {
    var afmJSONValue: AFMJSONValue {
        switch self {
        case .null:
            return .null
        case .bool(let value):
            return .bool(value)
        case .int(let value):
            return .integer(value)
        case .double(let value):
            return .number(value)
        case .string(let value):
            return .string(value)
        case .array(let values):
            return .array(values.map(\.afmJSONValue))
        case .object(let values):
            return .object(values.mapValues(\.afmJSONValue))
        }
    }
}

private extension ResponseFormat {
    var afmConstraint: AFMResponseConstraint? {
        switch type {
        case "json_object":
            return .jsonObject
        case "json_schema":
            guard let schema = jsonSchema?.schema?.value.afmJSONValue else {
                return nil
            }
            return .jsonSchema(
                name: jsonSchema?.name,
                schema: schema,
                strict: jsonSchema?.strict ?? false
            )
        default:
            return nil
        }
    }
}

extension AFMResponse {
    init(modelResponse: AFMModelResponse) {
        self.init(
            content: modelResponse.text,
            reasoningContent: modelResponse.reasoning,
            toolCalls: modelResponse.toolCalls.enumerated().map { index, call in
                ResponseToolCall(
                    index: index,
                    id: call.id,
                    type: "function",
                    function: ResponseToolCallFunction(
                        name: call.name,
                        arguments: call.arguments
                    )
                )
            },
            promptTokens: modelResponse.usage.inputTokens,
            completionTokens: modelResponse.usage.outputTokens
        )
    }
}
