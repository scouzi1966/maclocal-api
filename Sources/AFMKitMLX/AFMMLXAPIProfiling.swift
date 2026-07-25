import AFMOpenAICompat

/// API-facing profiling hooks used by OpenAI-compatible MLX serving paths.
///
/// The concrete implementation still lives with the MLX runtime, but exposing
/// the contract from AFMKitMLX keeps server controllers from depending on the
/// concrete service type for profile lifecycle behavior.
public protocol AFMMLXAPIProfiling: Sendable {
    func startAPIProfile()

    func stopAPIProfile(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfile

    func stopAPIProfileExtended(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: Double,
        generateTime: Double
    ) -> AFMProfileExtended
}

extension MLXModelService: AFMMLXAPIProfiling {}
