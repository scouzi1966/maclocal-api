import AFMKitCore

enum AFMMLXCapabilityPresentation {
    static func modelCapabilityLabels(
        descriptor: AFMModelDescriptor?
    ) -> [String] {
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
        return labels
    }

    static func supportsVision(descriptor: AFMModelDescriptor?) -> Bool {
        descriptor?.capabilities.contains(.vision) == true
    }
}
