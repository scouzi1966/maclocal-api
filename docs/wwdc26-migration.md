# AFMKit ⇄ WWDC 26 Foundation Models — migration map

afm's `AFMKit` library is built so the Apple **WWDC 26 ("Year Two") Foundation Models**
direction can be adopted incrementally, **without breaking the macOS 26 build**. macOS 27 /
the year-two SDK is not yet available on the build machine, so this document records the
*seams* — where each year-two capability will plug into AFMKit — and nothing here references
an Apple symbol that doesn't exist in the macOS 26 SDK.

## Guiding principle

Apple year-two unifies every model provider behind one `LanguageModel` protocol that backs a
single `LanguageModelSession`. AFMKit mirrors that shape with **`AFMLanguageModel`** (see
`Sources/AFMKit/AFMLanguageModel.swift`) — an afm-native, source-stable protocol that
`AFMEngine` already conforms to. When the macOS 27 SDK lands, a thin
`@available(macOS 27, *)` adapter bridges a real Apple `LanguageModel` to `AFMLanguageModel`
(and exposes afm's MLX backend *as* an Apple `LanguageModel`), with no change to AFMKit's
public surface.

## Feature-by-feature seam map

| WWDC 26 capability | Apple year-two API (macOS 27) | AFMKit extension point (today) | Adoption note |
|---|---|---|---|
| **Provider abstraction** | `LanguageModel` protocol backing `LanguageModelSession` | `AFMLanguageModel` protocol; `AFMEngine` conforms | When 27 SDK is present, add `@available(macOS 27,*) struct AppleLanguageModelAdapter: AFMLanguageModel` wrapping a `LanguageModelSession`, and an `MLXLanguageModel: LanguageModel` exposing afm's MLX backend to Apple's session. |
| **Private Cloud Compute** | `PrivateCloudComputeLanguageModel`, `ContextOptions(reasoningLevel:)`, `model.availability`, `model.quotaUsage`; entitlement `com.apple.developer.private-cloud-compute` | New `AFMBackend` case `.privateCloudCompute` + an `AFMLanguageModel` conformer | Gate the conformer `@available(macOS 27,*)`. Surface availability/quota as **state** on `AFMEngine` (not thrown errors), matching Apple's guidance. Document the managed entitlement in the consuming app. |
| **Reasoning levels** | `ContextOptions(reasoningLevel: .light/.moderate/.deep)`, `response.usage.output.reasoningTokenCount` | Add `reasoningLevel` to `GenerationConfig` (no-op on macOS 26 / MLX) | Field can be added now as an afm-native enum; only the PCC conformer reads it. |
| **Multimodal image input** | `Attachment(UIImage/NSImage/CGImage/CIImage/CVPixelBuffer/URL)` in the prompt builder | `Message` + `MessageContent.parts([ContentPart])` with `ContentPart.image_url` already exist; the MLX **VLM** path already accepts `--media` | Map `ContentPart` image parts to Apple `Attachment`s inside the macOS-27 adapter. No new public surface needed. |
| **`@Generable` structured output / streaming `PartiallyGenerated<T>`** | `@Generable`, `session.streamResponse(generating:)` | `GenerationConfig.responseFormat` (`json_schema`) + `JSONSchemaConverter` → Apple `GenerationSchema` (already used by `FoundationModelService.generateGuidedResponse`) | Year-two streaming partials map to afm's existing guided-streaming; expose `PartiallyGenerated` only behind `@available(macOS 27,*)`. |
| **System tools** | `OCRTool`, `BarcodeReaderTool`, `SpotlightSearchTool` (vision-backed, on-device) | afm already ships `VisionService` (OCR, barcode, classification, saliency, tables) + a tool-calling runtime | When 27 SDK is present, register afm's Vision capabilities as Apple system-tool equivalents, or pass through Apple's built-ins via the adapter. |
| **Third-party providers** | `ClaudeLanguageModel`, `GeminiLanguageModel`, `MLXLanguageModel`, `CoreAILanguageModel` (all `LanguageModel`) | `AFMBackend` is an enum; add cases / conformers | Each becomes an `AFMLanguageModel` conformer or a new `AFMBackend` case; `AFMEngine` dispatch already switches on `backend`. |
| **Dynamic Profiles** | `LanguageModelSession.DynamicProfile` (swap instructions/tools/model/options mid-session, preserving history) | `EngineConfig` + a future `AFMEngine.reconfigure(_:)` | Year-two profiles map to reconfiguring an `AFMEngine` between turns; the transcript lives in the `[Message]` array the caller threads, so history preservation is already the caller's model. |
| **On-device LoRA training** | `LanguageModelAdapter.train(examples:configuration:)` | `EngineConfig.adapter` (path) already wires a LoRA adapter into the Foundation Models backend | Add a `@available(macOS 27,*)` training helper that produces an adapter `EngineConfig` consumes. |
| **`fm` CLI / Instruments / Evaluations** | `fm` tool, Xcode 27 Foundation Models Instrument | the `afm` CLI is the analogue; keep flag names aligned where sensible | Informational — no code seam. |

## What is intentionally NOT in this branch

- No reference to any macOS 27 Apple symbol (won't compile on macOS 26).
- No `PrivateCloudComputeLanguageModel` / `DynamicProfile` / `Attachment` implementation — only the seams above.
- The Vapor-free `AFMServer` split and full DTO de-Vapor are tracked separately (Stages B/C of the AFMKit work).

## When macOS 27 is installed — first adoption steps

1. Add `@available(macOS 27,*) struct AppleLanguageModelAdapter: AFMLanguageModel` wrapping `LanguageModelSession`.
2. Add `AFMBackend.privateCloudCompute` + a PCC conformer; surface `availability`/`quotaUsage` on `AFMEngine`.
3. Add `reasoningLevel` to `GenerationConfig`; thread it into the PCC path only.
4. Map `ContentPart` image parts → Apple `Attachment` in the adapter.
5. Keep `AFMLanguageModel`'s signature stable — everything above is additive.
