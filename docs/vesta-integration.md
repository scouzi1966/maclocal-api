# Integrating AFMKit into Vesta

Vesta and maclocal-api consume the same standalone AFMKit checkpoint. Vesta
imports provider products directly; it does not import maclocal-api's
`AFMServer`, CLI, WebUI, or release packaging.

## Package Dependency

Until AFMKit publishes its first tag, use the immutable checkpoint:

```swift
.package(
    url: "https://github.com/scouzi1966/AFMKit.git",
    revision: "dfeab23e95ea1979432958e3f9b002beb5685191"
)
```

AFMKit is private during this checkpoint. Vesta source builds require an
authenticated GitHub identity with read access; this is not a public downstream
installation path. Do not production-merge the private URL in either consumer
until AFMKit is public or an approved public immutable artifact replaces it.

Select only the products the app target needs:

```swift
.product(name: "AFMKitCore", package: "AFMKit")
.product(name: "AFMOpenAICompat", package: "AFMKit")
.product(name: "AFMKitMLX", package: "AFMKit")
.product(name: "AFMKitApple", package: "AFMKit")
```

`AFMKitCore` and `AFMOpenAICompat` have no server dependency. `AFMKitMLX` adds
the local MLX runtime. `AFMKitApple` adds the Foundation Models provider bridge.
None of these products pulls Vapor or maclocal-api into the app graph.

## Provider-Neutral Flow

Use `AFMProviderRegistry`, `AnyAFMModel`, `AFMRequest`, and typed
`AFMGenerationEvent` values as the app boundary:

```swift
import AFMKitCore
import AFMKitMLX

let registry = AFMProviderRegistry()
try registry.register(AFMMLXProviderFactory())
let model = try registry.makeModel(
    providerID: AFMMLXProviderFactory.providerID,
    modelID: AFMModelID(rawValue: selectedModelID),
    configuration: AFMProviderConfiguration(values: [
        "enablePrefixCaching": .bool(true)
    ])
)

_ = try await model.load()
let request = AFMRequest(messages: [AFMMessage(role: .user, text: prompt)])
for try await event in model.streamResponse(to: request) {
    // Reduce response text, reasoning, tools, usage, and completion by event type.
}
await model.unload()
```

The same generation loop works with `AFMFoundationProviderFactory` on macOS 27.
Keep app-specific provider selection, UI state, entitlements, provisioning
profiles, and chat persistence in Vesta.

## Compatibility Boundary

- The AFMKit package and MLX provider target macOS 26.
- Apple on-device and Private Cloud Compute provider registration is guarded by
  `@available(macOS 27.0, *)`.
- PCC requires the consuming app's signed
  `com.apple.developer.private-cloud-compute` entitlement. AFMKit cannot add or
  emulate it.
- AFMKit owns and resolves `AFMKit_AFMKitMLX.bundle`; app code should not copy a
  maclocal-api resource or assume the old `MacLocalAPI_AFMKit*.bundle` names.

The canonical runnable example is `Examples/AFMKitQuickstart` in the AFMKit
repository. maclocal-api's HTTP parity gate remains available for a cached MLX
model:

```bash
MACAFM_MLX_MODEL_CACHE=/path/to/model/cache \
  Scripts/test-afmkit-http-parity.sh --model <org/model>
```
