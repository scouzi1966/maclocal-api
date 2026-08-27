import AFMKitCore

enum AFMMLXCapabilityPresentation {
    static func modelCapabilityLabels(descriptor: AFMModelDescriptor?) -> [String] {
        guard let capabilities = descriptor?.capabilities else {
            return ["chat", "completion"]
        }

        var labels = ["chat", "completion"]
        let optionalLabels: [(AFMModelCapabilities, String)] = [
            (.vision, "vision"),
            (.reasoning, "reasoning"),
            (.toolCalling, "tools"),
            (.structuredOutput, "structured"),
            (.streaming, "streaming"),
            (.prefixCaching, "prefix_cache"),
            (.speculativeDecoding, "speculative_decoding"),
        ]
        labels.append(contentsOf: optionalLabels.compactMap { capability, label in
            capabilities.contains(capability) ? label : nil
        })
        if descriptor?.providerID.rawValue == "mlx" {
            labels.append(contentsOf: [
                "mlx_runtime",
                "batch",
                "context_window_override",
                "kv_quantization",
                "logprobs",
                "penalties",
                "prefill_tuning",
            ])
        } else if descriptor?.providerID.rawValue == "dwarfstar" {
            labels.append("dwarfstar_runtime")
        }
        return labels
    }

    static func supportsVision(descriptor: AFMModelDescriptor?) -> Bool {
        descriptor?.capabilities.contains(.vision) == true
    }
}
