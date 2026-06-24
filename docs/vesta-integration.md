# Integrating AFMKit into vesta-mac

This guide shows how [`vesta-mac`](https://github.com/scouzi1966/vesta-mac) — a sandboxed
SwiftUI macOS app that today talks to Apple's `FoundationModels.LanguageModelSession`
directly — can adopt **AFMKit** to gain MLX inference (any Hugging Face MLX model, on-device)
*without* pulling Vapor/NIO into the app, and later opt into the HTTP server by simply adding
the `AFMServer` product.

AFMKit is the headless, Vapor-free inference core split out of the `afm` server (PR #136). It
ships:

- `AFMEngine` — an `actor` facade over both backends (`.mlx(...)` and `.foundationModels(...)`).
- `AFMLanguageModel` — a provider-agnostic protocol shaped like Apple's year-two
  `LanguageModel`, conformed by **both** `AFMEngine` and `FoundationModelService`.
- The Apple-native services (`VisionService`, `SpeechService`, `SpeechSynthesisService`,
  embeddings) surfaced through the engine.

> AFMKit's resource bundle (`MacLocalAPI_AFMKit.bundle/default.metallib`) is resolved relative
> to the executable, so it works inside an app sandbox. No `Bundle.module` is ever called.

## 1. Add the dependency (no Vapor in the graph)

```swift
// vesta-mac/Package.swift  (or the Xcode "Package Dependencies" panel)
dependencies: [
    .package(url: "https://github.com/scouzi1966/maclocal-api.git", branch: "feature/afmlib"),
],
targets: [
    .target(
        name: "VestaCore",
        dependencies: [
            .product(name: "AFMKit", package: "maclocal-api"),
            // NOTE: do NOT add AFMServer here — importing only AFMKit keeps Vapor/NIO
            // out of the sandboxed app's dependency closure entirely.
        ]
    )
]
```

To later expose vesta as a local OpenAI-compatible server ("select vapor and wire it"), add
`.product(name: "AFMServer", package: "maclocal-api")` to a *separate* (non-sandboxed) target
and stand up `Server`. The inference core does not change.

## 2. Before / after call sites

### a) Session with instructions

**Before (Apple FoundationModels, direct):**

```swift
import FoundationModels

let session = LanguageModelSession(instructions: "You are Vesta, a helpful assistant.")
```

**After (AFMKit — Apple backend, unchanged behavior):**

```swift
import AFMKit

// Apple on-device backend, same instructions (instructions live on EngineConfig):
let engine = AFMEngine(
    backend: .foundationModels,
    config: EngineConfig(instructions: "You are Vesta, a helpful assistant.")
)
_ = try await engine.load()      // prepares the session / loads weights

// …or switch to a local MLX model with ZERO call-site changes downstream,
// because both conform to `AFMLanguageModel`:
let engine = AFMEngine(
    backend: .mlx(modelID: "mlx-community/Qwen3.5-35B-A3B-4bit"),
    config: EngineConfig(enablePrefixCaching: true)
)
_ = try await engine.load { fraction in /* progress 0…1 */ }
```

> `AFMEngine.init` is synchronous; `load()` (async, throwing, with an optional progress
> callback) is what prepares the backend. Both backends are then driven through the same
> downstream call sites.

### b) Streaming a response

**Before:**

```swift
let stream = session.streamResponse(to: prompt)
for try await partial in stream {
    self.transcript = partial          // Apple yields cumulative text
}
```

**After:**

```swift
let messages = [Message(role: "user", content: prompt)]
for try await delta in engine.streamResponse(to: messages, options: GenerationConfig()) {
    self.transcript += delta           // AFMKit yields incremental deltas
}
```

> Behavioral note: Apple's `streamResponse` yields **cumulative** snapshots; AFMKit's
> `streamResponse` (and the server's SSE) yield **incremental deltas**. vesta's accumulator
> changes from `=` to `+=`. (AFMKit already de-cumulates Apple's stream internally in
> `FoundationModelService.generateNativeStreamingResponse`, so the delta contract holds for
> both backends.)

### c) Non-streaming response

**Before:**

```swift
let response = try await session.respond(to: prompt)
label.text = response.content
```

**After:**

```swift
let response = try await engine.respond(to: [Message(role: "user", content: prompt)])
label.text = response.content          // AFMResponse.content
```

### d) LoRA adapter (vesta ships fine-tuned adapters)

**Before:**

```swift
let adapter = try SystemLanguageModel.Adapter(fileURL: adapterURL)
let model   = SystemLanguageModel(adapter: adapter)
let session = LanguageModelSession(model: model, instructions: instructions)
```

**After:**

```swift
let engine = AFMEngine(
    backend: .foundationModels,
    config: EngineConfig(instructions: instructions, adapter: adapterURL.path)
)
_ = try await engine.load()
```

The MLX backend accepts a LoRA adapter the same way (`EngineConfig.adapter`), so a vesta
adapter trained for MLX loads through the identical path.

## 3. Treating the model as `any AFMLanguageModel`

Because both backends conform to `AFMLanguageModel`, vesta can store the backend behind the
protocol and choose it at runtime (user preference, availability, model size):

```swift
let engine = useLocalMLX
    ? AFMEngine(backend: .mlx(modelID: selectedModelID))
    : AFMEngine(backend: .foundationModels,
                config: EngineConfig(instructions: instructions))
_ = try await engine.load()
let model: any AFMLanguageModel = engine

guard model.isAvailable else { /* fall back / show UI */ return }
for try await delta in model.streamResponse(to: messages, options: opts) {
    self.transcript += delta
}
```

This is the seam that makes the **year-two migration** a drop-in: when macOS 27 ships Apple's
unified `LanguageModel`, an `@available(macOS 27, *)` adapter bridges a real Apple
`LanguageModel` to `AFMLanguageModel` without touching vesta's view models. See
[`wwdc26-migration.md`](./wwdc26-migration.md).

## 4. Auxiliary capabilities (all Apple-native, sandbox-clean)

```swift
let ocr        = engine.vision()             // VisionService — OCR / tables / barcodes
let transcript = engine.speech()             // SpeechService — audio → text
let tts        = engine.speechSynthesis()    // SpeechSynthesisService — text → audio
let embedder   = engine.embeddings()         // NaturalLanguage contextual embeddings
```

These are the same services the `afm` server exposes at `/v1/audio/transcriptions`,
`/v1/embeddings`, etc. — so vesta gets feature parity with the server without running one.

## 5. Sandbox checklist

- **No Vapor in the graph** — depend on `AFMKit` only (verified: `git grep import\ Vapor --
  Sources/AFMKit` is empty).
- **metallib** resolves next to the app executable / inside `MacLocalAPI_AFMKit.bundle` — no
  `Bundle.module`, so a relocated/sandboxed binary does not crash.
- **Model cache** — set the `MACAFM_MLX_MODEL_CACHE` environment variable to a
  sandbox-writable location so MLX weights are stored there and not re-downloaded.
- **Speech/mic** — if vesta uses `engine.speech()`, add `NSSpeechRecognitionUsageDescription`
  to vesta's Info.plist (the `afm` CLI embeds this; an app provides it via its own plist).

## 6. Proving parity

`Examples/AFMParityCheck` runs a fixed battery of prompts through **AFMKit-direct** and the
**spawned `afm` HTTP server** at temperature 0 and asserts identical output. Run it to confirm
the library vesta embeds produces byte-for-byte the same results as the server. See that
example's README.
