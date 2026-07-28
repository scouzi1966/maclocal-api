import Foundation
@preconcurrency import MLXLMCommon

public struct AFMMLXPreparedChatHistory {
    public let history: [Chat.Message]
    public let snapshot: [Chat.Message]

    public init(history: [Chat.Message], snapshot: [Chat.Message]) {
        self.history = history
        self.snapshot = snapshot
    }
}

public enum AFMMLXChatHistoryPolicy {
    public static let defaultSystemPrompt = "You are a helpful assistant!"

    public static func reset(systemPrompt: String = defaultSystemPrompt) -> [Chat.Message] {
        [.system(systemPrompt)]
    }

    public static func applyingSystemPrompt(
        _ systemPrompt: String,
        to history: [Chat.Message]
    ) -> [Chat.Message] {
        guard history.count > 1 else {
            return reset(systemPrompt: systemPrompt)
        }

        var updatedHistory = history
        updatedHistory[0] = .system(systemPrompt)
        return updatedHistory
    }

    public static func preparedSnapshot(
        history: [Chat.Message],
        userMessage: String,
        images: [UserInput.Image],
        systemPrompt: String
    ) -> AFMMLXPreparedChatHistory {
        var updatedHistory = applyingSystemPrompt(systemPrompt, to: history)
        updatedHistory.append(.user(userMessage, images: images))
        return AFMMLXPreparedChatHistory(
            history: updatedHistory,
            snapshot: updatedHistory
        )
    }

    public static func appendingAssistantHistoryText(
        _ historyText: String,
        to history: [Chat.Message]
    ) -> [Chat.Message] {
        guard !historyText.isEmpty else { return history }
        return history + [.assistant(historyText)]
    }
}
