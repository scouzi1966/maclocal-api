import Foundation
import MLXLMCommon

/// Transfers a non-Sendable value across an isolation boundary exactly once.
/// Mirrors MLXLMCommon's internal `SendableBox`, which is not public.
private final class TransferBox<T>: @unchecked Sendable {
    private var value: T?

    init(_ value: consuming T) {
        self.value = consume value
    }

    func consume() -> T {
        let result = value!
        value = nil
        return result
    }
}

extension ModelContainer {
    /// Like ``ModelContainer/generate(input:parameters:)`` but also returns the
    /// producer `Task`, so abandoned generations (e.g. a non-streaming request
    /// whose client disconnected) can be cancelled instead of running to
    /// completion while holding GPU time.
    func generateTask(
        input: consuming sending LMInput,
        parameters: GenerateParameters
    ) async throws -> (AsyncStream<Generation>, Task<Void, Never>) {
        let box = TransferBox(input)
        return try await perform { context in
            let input = box.consume()
            let promptTokenCount = input.text.tokens.size
            let iterator = try TokenIterator(
                input: input, model: context.model, cache: nil, parameters: parameters)
            return MLXLMCommon.generateTask(
                promptTokenCount: promptTokenCount,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator,
                stopAfterToolCall: parameters.stopAfterToolCall)
        }
    }
}
