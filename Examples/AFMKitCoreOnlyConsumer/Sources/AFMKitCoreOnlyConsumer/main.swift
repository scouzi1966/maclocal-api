import AFMKitCore

let provider = AFMProviderDescriptor(
    id: "example.core-only",
    displayName: "Core-Only Example",
    privacyBoundary: .device,
    metadata: ["purpose": .string("dependency-boundary-smoke")]
)

let model = AFMModelDescriptor(
    providerID: provider.id,
    modelID: "example.core-only.echo",
    displayName: "Echo Contract",
    capabilities: [.text],
    contextWindow: 1024,
    privacyBoundary: .device
)

let request = AFMRequest(
    messages: [
        AFMMessage(role: .system, text: "Reply with plain text."),
        AFMMessage(role: .user, text: "Hello from AFMKitCore.")
    ],
    options: AFMGenerationOptions(maximumResponseTokens: 16)
)

let event = AFMGenerationEvent.responseText(action: .append, text: "ok", tokenCount: 1)

print("\(provider.id)|\(model.modelID)|\(request.messages.count)|\(event)")
