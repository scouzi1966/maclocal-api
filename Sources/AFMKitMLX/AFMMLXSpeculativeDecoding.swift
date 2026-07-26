import Foundation

public enum AFMMLXSpeculativeDecodingMode: String, Codable, CaseIterable, Identifiable, Hashable, Sendable {
    case off
    case auto
    case mtp
    case eagle3

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .off: return "Off"
        case .auto: return "Auto"
        case .mtp: return "MTP"
        case .eagle3: return "EAGLE3"
        }
    }
}

public struct AFMMLXSpeculativeModeAvailability: Equatable, Sendable {
    public let mode: AFMMLXSpeculativeDecodingMode
    public let isSelectable: Bool
    public let reason: String

    public init(
        mode: AFMMLXSpeculativeDecodingMode,
        isSelectable: Bool,
        reason: String
    ) {
        self.mode = mode
        self.isSelectable = isSelectable
        self.reason = reason
    }

    public static func evaluate(
        modelLoaded: Bool,
        mtpCompatible: Bool,
        denseGemma4Verifier: Bool
    ) -> [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability] {
        let hasAccelerationPath = mtpCompatible || denseGemma4Verifier

        return [
            .off: AFMMLXSpeculativeModeAvailability(
                mode: .off,
                isSelectable: true,
                reason: "Use standard MLX generation."
            ),
            .auto: AFMMLXSpeculativeModeAvailability(
                mode: .auto,
                isSelectable: modelLoaded && hasAccelerationPath,
                reason: modelLoaded
                    ? "Auto requires a loaded model with MTP or dense Gemma4 EAGLE3 support."
                    : "Load a supported MLX model before enabling acceleration."
            ),
            .mtp: AFMMLXSpeculativeModeAvailability(
                mode: .mtp,
                isSelectable: modelLoaded && mtpCompatible,
                reason: modelLoaded
                    ? "MTP requires a compatible loaded model with mtp.safetensors."
                    : "Load a model with an MTP sidecar before selecting MTP."
            ),
            .eagle3: AFMMLXSpeculativeModeAvailability(
                mode: .eagle3,
                isSelectable: modelLoaded && denseGemma4Verifier,
                reason: modelLoaded
                    ? "EAGLE3 requires a loaded dense Gemma4 verifier model."
                    : "Load a dense Gemma4 model before selecting EAGLE3."
            )
        ]
    }

    public static let unloaded = evaluate(
        modelLoaded: false,
        mtpCompatible: false,
        denseGemma4Verifier: false
    )

    public static func pendingSelection(
        mtpCompatible: Bool,
        denseGemma4Verifier: Bool
    ) -> [AFMMLXSpeculativeDecodingMode: AFMMLXSpeculativeModeAvailability] {
        let hasAccelerationPath = mtpCompatible || denseGemma4Verifier

        return [
            .off: AFMMLXSpeculativeModeAvailability(
                mode: .off,
                isSelectable: true,
                reason: "Use standard MLX generation."
            ),
            .auto: AFMMLXSpeculativeModeAvailability(
                mode: .auto,
                isSelectable: hasAccelerationPath,
                reason: hasAccelerationPath
                    ? "Use the selected model's acceleration path after loading."
                    : "Select a model with MTP or dense Gemma4 EAGLE3 support."
            ),
            .mtp: AFMMLXSpeculativeModeAvailability(
                mode: .mtp,
                isSelectable: mtpCompatible,
                reason: mtpCompatible
                    ? "Use MTP after loading the selected model."
                    : "Select a model with mtp.safetensors before selecting MTP."
            ),
            .eagle3: AFMMLXSpeculativeModeAvailability(
                mode: .eagle3,
                isSelectable: denseGemma4Verifier,
                reason: denseGemma4Verifier
                    ? "Use EAGLE3 after loading the selected dense Gemma4 model."
                    : "Select a dense Gemma4 model before selecting EAGLE3."
            )
        ]
    }
}

public struct AFMMLXSpeculativeModelCompatibility: Equatable, Sendable {
    public let mtpCompatible: Bool
    public let denseGemma4Verifier: Bool

    public init(mtpCompatible: Bool, denseGemma4Verifier: Bool) {
        self.mtpCompatible = mtpCompatible
        self.denseGemma4Verifier = denseGemma4Verifier
    }

    public static let unavailable = AFMMLXSpeculativeModelCompatibility(
        mtpCompatible: false,
        denseGemma4Verifier: false
    )

    public static func evaluate(
        config: [String: Any],
        hasMTPSidecar: Bool
    ) -> AFMMLXSpeculativeModelCompatibility {
        AFMMLXSpeculativeModelCompatibility(
            mtpCompatible: hasMTPSidecar && isMTPCompatibleConfiguration(config),
            denseGemma4Verifier: isDenseGemma4VerifierConfiguration(config)
        )
    }

    public static func evaluate(modelDirectory: URL) -> AFMMLXSpeculativeModelCompatibility {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .unavailable
        }

        let hasMTPSidecar = FileManager.default.fileExists(
            atPath: modelDirectory.appendingPathComponent("mtp.safetensors").path
        )
        return evaluate(config: config, hasMTPSidecar: hasMTPSidecar)
    }

    private static func isMTPCompatibleConfiguration(_ config: [String: Any]) -> Bool {
        let topLevelType = AFMMLXModelArchitecture.canonicalModelType(config["model_type"] as? String ?? "")
        let textConfig = config["text_config"] as? [String: Any]
        let textType = AFMMLXModelArchitecture.canonicalModelType(textConfig?["model_type"] as? String ?? "")
        let architecture = ((config["architectures"] as? [String]) ?? []).joined(separator: " ").lowercased()

        return topLevelType.hasPrefix("qwen3_5")
            || topLevelType.hasPrefix("qwen3_6")
            || textType.hasPrefix("qwen3_5")
            || textType.hasPrefix("qwen3_6")
            || architecture.contains("qwen3_5")
            || architecture.contains("qwen3_6")
            || architecture.contains("qwen3.5")
            || architecture.contains("qwen3.6")
    }

    private static func isDenseGemma4VerifierConfiguration(_ config: [String: Any]) -> Bool {
        let modelType = AFMMLXModelArchitecture.canonicalModelType(config["model_type"] as? String ?? "")
        let architecture = ((config["architectures"] as? [String]) ?? []).joined(separator: " ").lowercased()
        return modelType == "gemma4" && !architecture.contains("moe")
    }
}

public enum AFMMLXSpeculativeRuntimeKind: Equatable, Sendable {
    case none
    case mtp
    case eagle3
}

public enum AFMMLXSpeculativeGenerationPath: String, Equatable, Sendable {
    case normal = "Normal MLX"
    case mtp = "MTP"
    case eagle3 = "EAGLE3"
    case fallback = "Fallback"
}

public enum AFMMLXSpeculativeFallbackReason: String, Equatable, Sendable {
    case modeOff = "Acceleration off"
    case runtimeUnavailable = "Runtime unavailable"
    case samplingEnabled = "Sampling enabled"
    case generationModifiers = "Generation modifiers enabled"
    case reasoningOutput = "Reasoning output enabled"
    case visionInput = "Vision input"
    case stopSequences = "Stop sequences enabled"
}

public struct AFMMLXSpeculativeGenerationDecision: Equatable, Sendable {
    public let path: AFMMLXSpeculativeGenerationPath
    public let reason: AFMMLXSpeculativeFallbackReason?

    public init(
        path: AFMMLXSpeculativeGenerationPath,
        reason: AFMMLXSpeculativeFallbackReason?
    ) {
        self.path = path
        self.reason = reason
    }

    public static func evaluate(
        mode: AFMMLXSpeculativeDecodingMode,
        installedRuntime: AFMMLXSpeculativeRuntimeKind,
        temperature: Double,
        hasUnsupportedGenerationModifiers: Bool,
        hasReasoningOutput: Bool,
        hasImages: Bool,
        hasStopSequences: Bool
    ) -> AFMMLXSpeculativeGenerationDecision {
        guard mode != .off else {
            return AFMMLXSpeculativeGenerationDecision(path: .normal, reason: .modeOff)
        }
        guard !hasImages else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .visionInput)
        }
        guard !hasStopSequences else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .stopSequences)
        }
        guard !hasUnsupportedGenerationModifiers else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .generationModifiers)
        }
        guard !hasReasoningOutput else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .reasoningOutput)
        }
        guard abs(temperature) < 0.000_001 else {
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .samplingEnabled)
        }

        switch (mode, installedRuntime) {
        case (.auto, .mtp), (.mtp, .mtp):
            return AFMMLXSpeculativeGenerationDecision(path: .mtp, reason: nil)
        case (.auto, .eagle3), (.eagle3, .eagle3):
            return AFMMLXSpeculativeGenerationDecision(path: .eagle3, reason: nil)
        case (.mtp, _), (.eagle3, _), (.auto, .none):
            return AFMMLXSpeculativeGenerationDecision(path: .fallback, reason: .runtimeUnavailable)
        case (.off, _):
            return AFMMLXSpeculativeGenerationDecision(path: .normal, reason: .modeOff)
        }
    }
}
