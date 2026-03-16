import Foundation

final class AFMLocalClient {
    private let baseURL: URL
    private let modelID: String
    private let instructions: String
    private let session: URLSession

    init(baseURL: String, modelID: String, instructions: String) {
        self.baseURL = URL(string: baseURL)!
        self.modelID = modelID
        self.instructions = instructions
        self.session = URLSession(configuration: .ephemeral)
    }

    func sendMessage(prompt: String, imageURLs: [URL], userTag: String) async throws -> String {
        let userMessage: Message
        if imageURLs.isEmpty {
            userMessage = Message(role: "user", content: prompt)
        } else {
            var parts = [ContentPart(type: "text", text: prompt, image_url: nil)]
            for imageURL in imageURLs {
                parts.append(ContentPart(type: "image_url", text: nil, image_url: ImageURL(url: imageURL.absoluteString, detail: nil)))
            }
            userMessage = Message(role: "user", content: .parts(parts))
        }

        return try await sendMessages([Message(role: "system", content: instructions), userMessage], userTag: userTag)
    }

    func sendMessages(_ messages: [Message], userTag: String) async throws -> String {
        let endpoint = baseURL.appendingPathComponent("v1/chat/completions")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let payload = ChatCompletionRequest(
            model: modelID,
            messages: messages,
            temperature: nil,
            maxTokens: nil,
            maxCompletionTokens: nil,
            topP: nil,
            repetitionPenalty: nil,
            repeatPenalty: nil,
            frequencyPenalty: nil,
            presencePenalty: nil,
            topK: nil,
            minP: nil,
            seed: nil,
            logprobs: nil,
            topLogprobs: nil,
            stop: nil,
            stream: false,
            user: userTag,
            tools: nil,
            toolChoice: nil,
            responseFormat: nil,
            chatTemplateKwargs: nil
        )

        request.httpBody = try JSONEncoder().encode(payload)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw TelegramBridgeError.localClientError("AFM request failed without HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw TelegramBridgeError.localClientError("AFM API error \(http.statusCode): \(body)")
        }

        let completion = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
        return completion.choices.first?.message.content ?? ""
    }
}
