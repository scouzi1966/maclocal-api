#if canImport(FoundationModels)
import AFMKit
import FoundationModels

/// The provider settings AFMKit needs to translate a macOS 27 Foundation
/// Models request without knowing which inference engine executes it.
@available(macOS 27.0, *)
public protocol AFMFoundationModelsModelConfiguration: Sendable {
    var defaultMaximumResponseTokens: Int { get }
    var supportsReasoning: Bool { get }
}

/// Shared implementation for custom `LanguageModelExecutor` providers.
///
/// Provider packages keep ownership of model loading and inference. This bridge
/// owns the macOS 27 contract: transcript conversion, generation options,
/// tools, structured-output requests, and streaming channel events.
@available(macOS 27.0, *)
public enum AFMFoundationModelsExecutorBridge {
    public static func events(
        from model: AnyAFMModel,
        request: AFMRequest
    ) -> AsyncThrowingStream<AFMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await model.load()
                    for try await event in model.streamResponse(to: request) {
                        continuation.yield(streamEvent(from: event))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public static func respond(
        events: AsyncThrowingStream<AFMStreamEvent, Error>,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        var adapter = AFMFoundationModelsEventChannelAdapter()
        let (plans, continuation) = AsyncStream.makeStream(
            of: AFMFoundationModelsEventChannelAdapter.ChannelPlan.self
        )
        let sender = Task.detached(priority: .utility) {
            for await plan in plans {
                await AFMFoundationModelsEventChannelAdapter.send(plan, into: channel)
            }
        }

        do {
            for try await event in events {
                try Task.checkCancellation()
                for plan in adapter.plans(for: event) {
                    continuation.yield(plan)
                }
            }
            for plan in adapter.completionPlans() {
                continuation.yield(plan)
            }
            continuation.finish()
            await sender.value
        } catch {
            continuation.finish()
            sender.cancel()
            await sender.value
            throw error
        }
    }

    private static func streamEvent(from event: AFMGenerationEvent) -> AFMStreamEvent {
        switch event {
        case .responseText(_, let text, let tokenCount):
            return .text(text, tokenCount: tokenCount)
        case .reasoningText(_, let text, let tokenCount):
            return .reasoning(text, tokenCount: tokenCount)
        case .tokenLogprobs(let values):
            return .tokenLogprobs(values)
        case .toolCall(let call, let stage):
            return .toolCall(call, stage: stage)
        case .usage(let usage):
            return .usage(
                promptTokens: usage.inputTokens,
                completionTokens: usage.outputTokens,
                cachedTokens: usage.cachedInputTokens
            )
        case .metadata(let metadata):
            return .metadata(metadata)
        case .custom(let type, let payload):
            return .custom(type: type, payload: payload)
        case .completed(let reason):
            return .completed(reason)
        }
    }
}
#endif
