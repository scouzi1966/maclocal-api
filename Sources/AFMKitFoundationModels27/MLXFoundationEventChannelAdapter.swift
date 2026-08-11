#if canImport(FoundationModels)
import AFMKit
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationModelsEventChannelAdapter {
    public static let textBatchTokenLimit = 16
    public static let textBatchCharacterLimit = 256

    public enum ChannelPlan: Equatable, Sendable {
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
    private var pendingText: ChannelPlan?

    public init() {}

    public mutating func send(
        _ event: AFMStreamEvent,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        for readyPlan in plans(for: event) {
            await Self.send(readyPlan, into: channel)
        }
    }

    public mutating func plans(for event: AFMStreamEvent) -> [ChannelPlan] {
        guard let plan = consume(event) else { return [] }
        return enqueue(plan)
    }

    public mutating func consume(_ event: AFMStreamEvent) -> ChannelPlan? {
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

    public mutating func finish(into channel: LanguageModelExecutorGenerationChannel) async {
        for plan in completionPlans() {
            await Self.send(plan, into: channel)
        }
    }

    public mutating func completionPlans() -> [ChannelPlan] {
        var plans = flushPlans()
        if let finishPlan = finishPlan() {
            plans.append(finishPlan)
        }
        return plans
    }

    public func finishPlan() -> ChannelPlan? {
        guard !sentUsage else { return nil }
        return .usage(AFMUsage(outputTokens: streamedTokens))
    }

    public mutating func enqueue(_ plan: ChannelPlan) -> [ChannelPlan] {
        guard Self.isText(plan) else {
            return flushPlans() + [plan]
        }

        guard let pendingText else {
            self.pendingText = plan
            return Self.shouldFlush(plan) ? flushPlans() : []
        }

        guard let combined = Self.combine(pendingText, with: plan) else {
            self.pendingText = plan
            return [pendingText] + (Self.shouldFlush(plan) ? flushPlans() : [])
        }

        self.pendingText = combined
        return Self.shouldFlush(combined) ? flushPlans() : []
    }

    public mutating func flushPlans() -> [ChannelPlan] {
        guard let pendingText else { return [] }
        self.pendingText = nil
        return [pendingText]
    }

    private static func isText(_ plan: ChannelPlan) -> Bool {
        switch plan {
        case .responseText, .reasoningText:
            return true
        default:
            return false
        }
    }

    private static func combine(
        _ lhs: ChannelPlan,
        with rhs: ChannelPlan
    ) -> ChannelPlan? {
        switch (lhs, rhs) {
        case let (
            .responseText(lhsText, lhsTokenCount),
            .responseText(rhsText, rhsTokenCount)
        ):
            return .responseText(
                lhsText + rhsText,
                tokenCount: lhsTokenCount + rhsTokenCount
            )
        case let (
            .reasoningText(lhsText, lhsTokenCount),
            .reasoningText(rhsText, rhsTokenCount)
        ):
            return .reasoningText(
                lhsText + rhsText,
                tokenCount: lhsTokenCount + rhsTokenCount
            )
        default:
            return nil
        }
    }

    private static func shouldFlush(_ plan: ChannelPlan) -> Bool {
        switch plan {
        case .responseText(let text, let tokenCount),
             .reasoningText(let text, let tokenCount):
            return tokenCount >= textBatchTokenLimit
                || text.count >= textBatchCharacterLimit
        default:
            return true
        }
    }

    public static func send(
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
                        AFMFoundationModelsRequestAdapter.foundationMetadata(values)
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

@available(macOS 27.0, *)
typealias MLXFoundationEventChannelAdapter = AFMFoundationModelsEventChannelAdapter
#endif
