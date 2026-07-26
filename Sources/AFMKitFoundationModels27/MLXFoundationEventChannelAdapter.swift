#if canImport(FoundationModels)
import AFMKit
import Foundation
import FoundationModels

@available(macOS 27.0, *)
struct MLXFoundationEventChannelAdapter {
    enum ChannelPlan: Equatable {
        case responseText(String, tokenCount: Int)
        case reasoningText(String, tokenCount: Int)
        case usage(AFMUsage)
        case toolArguments(id: String, name: String, arguments: String)
        case metadata([String: AFMJSONValue])
        case customMetadata(key: String, value: String)
        case finishReason(String)
    }

    private var sentUsage = false
    private var streamedTokens = 0

    mutating func send(
        _ event: AFMStreamEvent,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        guard let plan = consume(event) else { return }
        await send(plan, into: channel)
    }

    mutating func consume(_ event: AFMStreamEvent) -> ChannelPlan? {
        switch event {
        case .text(let text, let tokenCount):
            streamedTokens += tokenCount
            return .responseText(text, tokenCount: tokenCount)
        case .reasoning(let text, let tokenCount):
            return .reasoningText(text, tokenCount: tokenCount)
        case .usage(let promptTokens, let completionTokens, let cachedTokens):
            sentUsage = true
            return .usage(
                AFMUsage(
                    inputTokens: promptTokens,
                    cachedInputTokens: cachedTokens,
                    outputTokens: completionTokens
                )
            )
        case .toolCall(let call, let stage):
            switch stage {
            case .started:
                return .toolArguments(id: call.id, name: call.name, arguments: "")
            case .argumentsDelta(let delta):
                return .toolArguments(id: call.id, name: call.name, arguments: delta)
            case .completed, .retracted:
                return nil
            }
        case .metadata(let values):
            return .metadata(values)
        case .custom(let type, let payload):
            return .customMetadata(
                key: "afm.custom.\(type)",
                value: payload.base64EncodedString()
            )
        case .completed(let reason):
            return .finishReason(reason.rawValue)
        case .tokenLogprobs:
            return nil
        }
    }

    func finish(into channel: LanguageModelExecutorGenerationChannel) async {
        guard let plan = finishPlan() else { return }
        await send(plan, into: channel)
    }

    func finishPlan() -> ChannelPlan? {
        guard !sentUsage else { return nil }
        return .usage(AFMUsage(outputTokens: streamedTokens))
    }

    private func send(
        _ plan: ChannelPlan,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        switch plan {
        case .responseText(let text, let tokenCount):
            await channel.send(
                .response(action: .appendText(text, tokenCount: tokenCount))
            )
        case .reasoningText(let text, let tokenCount):
            await channel.send(
                .reasoning(action: .appendText(text, tokenCount: tokenCount))
            )
        case .usage(let usage):
            let usage = Self.foundationUsage(from: usage)
            await channel.send(
                .response(action: .updateUsage(input: usage.input, output: usage.output))
            )
        case .toolArguments(let id, let name, let arguments):
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: id,
                        name: name,
                        action: .appendArguments(arguments, tokenCount: 0)
                    )
                )
            )
        case .metadata(let values):
            await channel.send(
                .response(
                    action: .updateMetadata(
                        MLXFoundationRequestAdapter.foundationMetadata(values)
                    )
                )
            )
        case .customMetadata(let key, let value):
            await channel.send(
                .response(
                    action: .updateMetadata([key: value])
                )
            )
        case .finishReason(let reason):
            await channel.send(
                .response(
                    action: .updateMetadata([
                        "afm.finishReason": reason
                    ])
                )
            )
        }
    }

    private static func foundationUsage(
        from usage: AFMUsage
    ) -> LanguageModelExecutorGenerationChannel.Usage {
        .init(
            input: .init(
                totalTokenCount: usage.inputTokens,
                cachedTokenCount: usage.cachedInputTokens
            ),
            output: .init(
                totalTokenCount: usage.outputTokens,
                reasoningTokenCount: usage.reasoningTokens
            )
        )
    }
}
#endif
