# Issue #191 Phase A: Qwen 3.8 VLM Planning Trace

Status: architecture gate 2 approved; implementation in progress.

Issue: <https://github.com/scouzi1966/maclocal-api/issues/191>

Branch: `codex/issue-191-qwen38-vlm`

## Scope and acceptance target

Issue #191 reports that `mlx-community/Qwen3.8-27B-4bit` serves text but an
image attached in the bundled WebUI does not reach a working vision path. The
requested result is config-driven Qwen 3.8 VLM detection, complete vision asset
loading, preservation of OpenAI image parts, multimodal inference for streaming
and non-streaming chat, a text-only fast path, accurate capability reporting,
and clear errors for incomplete vision checkpoints. Live acceptance requires
JPEG and PNG WebUI requests plus two distinguishable images that produce
grounded, observably different answers.

This trace is Phase A only. It does not authorize feature implementation or an
upstream pull request.

## Current-state evidence

### Qwen 3.8 is represented by Qwen 3.5 architecture metadata

- The published Qwen 3.8 fixture declares `model_type: qwen3_5`,
  `Qwen3_5ForConditionalGeneration`, image/video/vision token IDs, nested
  `text_config`, and `vision_config`; it does not expose a distinct `qwen3_8`
  architecture identifier
  (`Tests/MacLocalAPITests/Qwen38PublishedConfigFixture.swift:3`).
- The architecture aliases normalize Qwen 3.5 and 3.6 names, while the enum has
  language and vision cases for `qwen3_5`; adding a name-derived `qwen3_8` enum
  would not match the checkpoint contract
  (`Sources/AFMKitMLX/AFMMLXModelArchitecture.swift:118`,
  `Sources/AFMKitMLX/AFMMLXModelArchitecture.swift:147`,
  `Sources/AFMKitMLX/AFMMLXModelArchitecture.swift:208`).
- Current preflight derives `isVision` and `requiresVisionModelFactory` from the
  decoded descriptor, but the initial factory remains independently selected
  (`Sources/AFMKitMLX/AFMMLXModelArchitecture.swift:277`).
- Existing tests prove that the published Qwen 3.8 config maps to `qwen3_5`, is
  recognized as vision-capable and dual-mode, but intentionally selects the LLM
  factory initially
  (`Tests/MacLocalAPITests/AFMMLXModelArchitectureTests.swift:207`).
- Model-name heuristics mention `qwen3.8`, but only for pre-configuration name
  inference. They are not a sufficient capability or asset contract
  (`Sources/AFMKitMLX/AFMMLXModelArchitecture.swift:370`).

### Factory selection leaves a dual-mode model on the text container

- Initial factory policy uses VLM only for `--vlm`/forced mode or architectures
  that cannot load through the language factory. A dual-mode Qwen checkpoint
  therefore starts as LLM
  (`Sources/AFMKitMLX/AFMMLXLoadedModeSwitchPolicy.swift:43`).
- Request media validation checks whether the decoded architecture says that
  images or audio are supported. It does not check which factory produced the
  currently loaded container
  (`Sources/AFMKitMLX/AFMMLXLoadedModeSwitchPolicy.swift:61`).
- `MLXModelService` registers both MLX LLM and VLM load trampolines
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:218`), then applies the above
  selection and only falls back from LLM to VLM after an LLM load failure
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:1588`).
- Loaded service state records model ID, architecture, container, and the MTP
  binding, but not the resolved model directory, decoded configuration, or the
  actual factory that created the container
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:223`,
  `Sources/AFMKitMLX/Models/MLXModelService.swift:1673`).
- Request validation consequently accepts image input based on architecture and
  returns the existing container, even when that container is the LLM one
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:6010`). This is the principal
  routing gap behind the issue.

### Factory lifecycle history constrains the repair

- Commit `09b278f` (`Fix MLX multimodal routing for dual-mode models`) previously
  tracked the loaded directory/factory and reloaded a dual-mode LLM container
  through the VLM factory when a media request arrived.
- Commit `721d3ac` (`Fix MLX multimodal lifecycle review findings`) removed that
  request-time reload and changed vision configurations to load as VLM from
  startup. Its review finding was substantive: the old path began loading a new
  container before scheduler shutdown/quiescence and then swapped service state,
  so concurrent work could observe an unsafe lifecycle transition.
- Commit `6535ca1` (`Fix Gemma 4 cache stability`) deliberately restored LLM-first
  startup for dual-mode configurations to preserve text/cache behavior. The
  current Qwen 3.8 gap is therefore the consequence of two intentional
  constraints, not simply a missing architecture alias.
- The approved correction is not a generic lifecycle transition. It is a narrow
  startup rule for asset-usable Qwen conditional-generation configurations.
  Request-time promotion, demotion, container replacement, and reversible
  scheduler quiescence are explicitly outside issue #191.
- The current service only exposes terminal shutdown/release: it marks the
  service shutting down and cancels scheduler work. It has no reversible
  admission/quiescence transaction to reuse
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:3879`).

### The MLX compatibility patch already contains a Qwen vision implementation

- The patched VLM registry maps `qwen3_5` and `qwen3_5_moe` to
  `Qwen3_5MoEVL`, and the processor registry contains `Qwen3VLProcessor`
  (`Scripts/patches/VLMModelFactory.swift:103`,
  `Scripts/patches/VLMModelFactory.swift:109`).
- VLM loading reads model config, tokenizer/processor config, and weights, then
  explicitly chooses `Qwen3VLProcessor` for Qwen 3.5-family checkpoints
  (`Scripts/patches/VLMModelFactory.swift:282`,
  `Scripts/patches/VLMModelFactory.swift:356`).
- `Qwen3_5MoEVL` owns both `vision_tower` and `language_model`, invokes the
  vision tower only when media tensors are present, and remaps published
  `model.visual` weights into the expected vision-tower namespace
  (`Scripts/patches/Qwen3_5MoEVL.swift:1443`,
  `Scripts/patches/Qwen3_5MoEVL.swift:1548`,
  `Scripts/patches/Qwen3_5MoEVL.swift:1662`).
- The Qwen processor returns token IDs and masks before image preprocessing when
  there are no images or videos
  (`Scripts/patches/Qwen3VL.swift:72`). This provides a no-vision-preprocessing
  text path inside a VLM container, although it does not guarantee identical
  throughput or memory use to the LLM container.
- Compatibility files are maintained as repository patches and applied by
  `Scripts/apply-mlx-patches.sh`; vendored SwiftPM files must not be edited
  directly (`Scripts/apply-mlx-patches.sh:22`).
- The local cached revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
  of `mlx-community/Qwen3.8-27B-4bit` was inspected as static qualification
  evidence. Its config matches the fixture, both processor metadata files name
  `Qwen3VLProcessor`, all three shards referenced by its index are present, and
  the index contains 333 `vision_tower.*` and 1,847 `language_model.*` keys.
  This supports using the existing patched Qwen 3.5 VLM implementation, but no
  live model load or inference was performed during Phase A.
- A locally cached Qwen 3.8 MXFP8 snapshot is incomplete (four of six indexed
  shards), so it is not evidence that the MXFP8 variant can currently pass live
  qualification. The repository implementation log independently records the
  shared `qwen3_5` text/vision contract
  (`docs/qwen3.8-27b-mxfp8-implementation-log.md`).

### Image DTOs and request conversion already preserve multimodal content

- OpenAI-compatible message content supports either a string or ordered content
  parts, including `image_url` with URL and detail fields
  (`Sources/AFMOpenAICompat/OpenAIRequest.swift:222`,
  `Sources/AFMOpenAICompat/OpenAIRequest.swift:252`).
- The MLX chat controller decodes the request with a 100 MB body limit and
  passes the original message array to the serving abstraction for both
  non-streaming and streaming execution
  (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:88`,
  `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:244`,
  `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:593`).
- `MLXModelService` extracts data URLs, remote/local image URLs, and videos, then
  builds user chat messages containing the decoded images/videos
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:6089`,
  `Sources/AFMKitMLX/Models/MLXModelService.swift:6421`). Multimodal input also
  bypasses prompt-cache reuse
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:2574`).
- Non-streaming service errors become HTTP 400 responses, but streaming starts
  the HTTP body before generation and later serializes failures into stream
  content. Vision preflight must therefore happen before the streaming response
  is committed if missing assets are to produce a clear protocol-level error
  (`Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:498`,
  `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:514`,
  `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift:1514`).

### WebUI attachment transmission is capability-gated

- The bundled llama.cpp WebUI converts stored image attachments into OpenAI
  `image_url` parts with a base64 data URL
  (`vendor/llama.cpp/tools/server/webui/src/lib/services/chat.service.ts:639`).
- Before that conversion, it removes image attachments when the model store says
  the selected model lacks vision support
  (`vendor/llama.cpp/tools/server/webui/src/lib/services/chat.service.ts:138`).
- The model store derives vision support from model properties/modalities
  (`vendor/llama.cpp/tools/server/webui/src/lib/stores/models.svelte.ts:127`) and
  fetches those properties from `/props`
  (`vendor/llama.cpp/tools/server/webui/src/lib/stores/models.svelte.ts:289`).
- maclocal-api currently reports `vision` for every MLX model in `/v1/models`
  and `vision: true` for every MLX model in `/props`, regardless of the actual
  descriptor or local asset completeness
  (`Sources/AFMServer/Server.swift:333`, `Sources/AFMServer/Server.swift:539`).
  The attachment reaches the API for this checkpoint, but these endpoints are
  not accurate and cannot protect other models or incomplete caches.

### Model descriptors are config-aware but are not the server source of truth

- `AFMMLXProvider` adds `.vision` when its decoded configuration is vision
  capable (`Sources/AFMKitMLX/AFMMLXProvider.swift:492`). Vision detection checks
  architecture/model type and the presence of text/vision or visual config
  structures (`Sources/AFMKitMLX/AFMMLXProvider.swift:554`).
- Its `requiresVisionModelFactory` heuristic is deliberately narrower: a nested
  text+vision checkpoint is forced to VLM only when language attention metadata
  is absent (`Sources/AFMKitMLX/AFMMLXProvider.swift:604`). Qwen 3.8 has nested
  attention metadata, so capability detection and factory selection diverge.
- Curated Qwen 3.8 4-bit and MXFP8 entries are listed as vision models and receive
  text, vision, and streaming capabilities
  (`Sources/AFMKitMLX/AFMMLXModelCatalog.swift:360`,
  `Sources/AFMKitMLX/AFMMLXModelCatalog.swift:482`).
- Cache completeness currently requires configuration and weight files/shards,
  but does not validate processor metadata or vision weight namespaces
  (`Sources/AFMKitMLX/Models/MLXCacheResolver.swift:141`).

### Existing tests cover recognition and generic VLM plumbing, not this route

- Architecture tests cover the published Qwen 3.8 config and decoding into the
  patched Qwen VLM configuration, but do not load an image
  (`Tests/MacLocalAPITests/AFMMLXModelArchitectureTests.swift:207`).
- Mode-switch policy tests establish that dual-mode models default to LLM and
  that media support is checked from configuration
  (`Tests/MacLocalAPITests/AFMMLXLoadedModeSwitchPolicyTests.swift:54`,
  `Tests/MacLocalAPITests/AFMMLXLoadedModeSwitchPolicyTests.swift:77`).
- Existing model tests demonstrate direct image preparation and generic
  multimodal adapter conversion, while concurrent tests include Gemma VLM
  cohorts
  (`Tests/MacLocalAPITests/MuseGlimmerModelTests.swift:269`,
  `Tests/MacLocalAPITests/MLXFoundationLanguageModelTests.swift:233`,
  `Tests/MacLocalAPITests/ConcurrentBatchTests.swift:166`). There is no current
  end-to-end Qwen/Gemma image-grounding test through the OpenAI chat route.

## Revised architecture

### 1. Build immutable vision qualification from configuration and assets

Add an immutable AFMKitMLX value such as `AFMMLXVisionAssetQualification` that
contains only evidence derived from a resolved snapshot:

- canonical architecture and conditional-generation architecture evidence;
- declared image/video capability from structured model configuration;
- required vision configuration and image/vision token IDs;
- processor metadata presence and the processor class the VLM factory will use;
- vision weight evidence from a safetensor index or standalone safetensor
  headers; and
- an asset-usable result with stable missing categories.

It must not contain `currentFactory`, a container, scheduler state, or any other
mutable runtime property. Qwen 3.8 remains configuration-resolved as
`qwen3_5`/`qwen3_5_moe`; repository-name heuristics cannot make a checkpoint
vision-usable.

Run this qualification once during model startup after
`MLXCacheResolver.localModelDirectory` has resolved the language-usable
snapshot. Cache the result by snapshot identity. Use structured JSON for model,
processor, and index metadata and safetensor headers for unindexed weights; do
not scan tensor payloads and do not repeat filesystem qualification per request.

The base cache resolver's completeness contract remains unchanged. Missing
processor metadata, vision token/config fields, or vision-tower weights are
optional-vision failures, not base-cache failures. A Qwen snapshot with usable
language assets but incomplete vision assets must still resolve and start for
text. Diagnostics expose missing categories without machine-specific paths.

### 2. Select the Qwen VLM factory narrowly at startup

Extend AFMKitMLX startup factory policy with this ordered decision:

1. Preserve the existing explicit `--vlm` behavior.
2. Preserve existing factory requirements for vision-only architectures.
3. Select `VLMModelFactory` for a configuration-resolved `qwen3_5` or
   `qwen3_5_moe` conditional-generation checkpoint only when its required vision
   contract is present and its local vision qualification is asset-usable.
4. For that same Qwen conditional-generation shape with incomplete optional
   vision assets, select the LLM factory so text startup remains available.
5. Leave Gemma, language-only Qwen, and every other existing architecture on
   their current policy.

`MLXModelService` records the selected/actual factory beside its model ID,
architecture, qualification, container, and MTP binding under the existing
state lock. The factory is mutable runtime state; it is not part of the
qualification value or reusable static provider descriptor.

The selected container is static for that model load. No request promotes,
demotes, reloads, or replaces it. It changes only through the existing explicit
model-load/switch or terminal shutdown lifecycle. This avoids a second 27B
container, rollback state, scheduler quiescence, cache transfer, and all
request-time factory mutation.

For a complete Qwen checkpoint the VLM container is also the text path. The
existing `Qwen3VLProcessor` returns before image preprocessing when media is
absent, and `Qwen3_5MoEVL` skips vision-tower execution. That is the required
text-only fast path. Live qualification must measure startup time, peak memory,
and text throughput rather than requiring LLM-container residency.

### 3. Keep request preflight provider-owned and side-effect-free

Add a media preflight operation at the `AFMMLXOpenAIChatServing` boundary
(`Sources/AFMKitMLX/AFMMLXOpenAIChatGenerating.swift:93`) with a no-op default
for source-compatible fakes/adapters. `MLXModelService` classifies ordered media
parts and reads one coherent runtime snapshot: immutable qualification plus the
loaded factory/container state. It performs no file inspection, factory choice,
load, reload, scheduler shutdown, or container mutation.

For a direct image request:

- an active asset-usable VLM runtime proceeds;
- a declared Qwen vision model with incomplete assets throws a typed
  `visionAssetsUnavailable` failure carrying stable missing categories; and
- a model that does not declare the requested media retains the distinct
  unsupported-media error.

AFMServer invokes this provider operation before scheduler reservation and
before constructing an SSE response body. The server only maps the typed asset
failure to HTTP `400`, `type: invalid_request_error`, and
`code: vision_assets_unavailable`; it never reads model files or selects a
factory. The `AFMKitMLXChatServingAdapter` forwards the operation. The generation
entry points retain an internal validation guard so non-HTTP AFMKit callers
cannot bypass runtime capability checks.

### 4. Publish honest, explicitly scoped capabilities

Use two named capability surfaces and never use the catalog surface for request
admission:

- **Declared/catalog capability:** immutable metadata for curated entries that
  may not be downloaded. It can advertise that a known checkpoint family is
  designed for vision, but says nothing about local usability.
- **Runtime-usable capability:** the loaded provider descriptor computed from
  the immutable qualification and coherent active runtime state. Vision is true
  only when assets qualified successfully and the active container is VLM-backed.

`/props` reports the loaded descriptor's runtime-usable capability. The loaded
entry in `/v1/models` is derived from that same descriptor, replacing the
unconditional server flag. Other unavailable catalog entries may retain declared
capability, but that distinction must be explicit in naming/tests and cannot be
fed into media preflight. Keep existing JSON shapes so the bundled WebUI needs
no source change: complete loaded Qwen enables attachments; incomplete or
text-only loaded models report vision false.

Concretely, keep pre-load discovery/catalog description separate from the
post-load descriptor. After `MLXModelService` publishes its startup state it
synthesizes the runtime descriptor; `AFMMLXRuntime.load` returns that descriptor,
and the serving abstraction exposes the same read-only value to AFMServer. This
avoids reconstructing runtime capability in `Server.swift` and avoids treating
the pre-load `AFMMLXModelDescriptor.describe` result as request-admission truth.
The descriptor returned by provider load is the authoritative loaded descriptor;
the pre-load descriptor property remains discovery/catalog information.

### 5. Qualify the existing mlx-swift-lm compatibility patches

The static path already maps Qwen 3.8's published `qwen3_5` metadata to
`Qwen3_5MoEVL` and `Qwen3VLProcessor`, including vision weight remapping. Do not
add a speculative `qwen3_8` implementation. First run focused live qualification
through the startup-selected VLM factory.

Only a demonstrated loader, processor, or weight-mapping incompatibility may
change the AFM-owned files under `Scripts/patches/`, reproducibly applied through
`Scripts/apply-mlx-patches.sh`. No upstream mlx-swift-lm, llama.cpp, or separate
AFMKit pull request is part of this issue.

## Ownership by layer

| Layer | Responsibility |
| --- | --- |
| AFMKitMLX | Owns immutable vision qualification, snapshot-scoped asset validation, startup factory policy, synchronized actual-factory state, runtime descriptors, side-effect-free media preflight, and typed vision failures. All model-file and factory decisions stay here. |
| maclocal-api host/server (`AFMServer`, CLI) | Calls provider preflight before reservation/response commitment, maps typed errors, and exposes the provider descriptor through `/props` and the loaded `/v1/models` entry. It does not inspect assets or choose/reload factories. |
| AFMKitCore / AFMOpenAICompat | Existing capability and multimodal/error DTO shapes remain source-compatible. Add shared code only if implementation proves an existing shape is insufficient. |
| Bundled WebUI (`vendor/llama.cpp`) | No feature change. Existing attachment conversion works with honest `/props`; do not edit the submodule. |
| AFM-owned mlx compatibility patch (`Scripts/patches`) | Qualification-first and conditional. Change only for a reproduced direct-VLM incompatibility, then apply through the repository patch workflow. |
| External/upstream repositories | No ownership and no PR. AFMKit products are in this repository (`Package.swift:20`). |

## Exact implementation files

Required or strongly likely:

- `Sources/AFMKitMLX/AFMMLXModelArchitecture.swift`
- `Sources/AFMKitMLX/AFMMLXLoadedModeSwitchPolicy.swift` (startup policy only;
  no mode switch is added)
- `Sources/AFMKitMLX/AFMMLXProvider.swift`
- `Sources/AFMKitMLX/AFMMLXRuntime.swift`
- `Sources/AFMKitMLX/AFMMLXModelStore.swift` (make declared/catalog semantics
  explicit; no runtime admission from discovery descriptors)
- `Sources/AFMKitMLX/Models/MLXModelService.swift`
- `Sources/AFMKitMLX/AFMMLXOpenAIChatGenerating.swift`
- `Sources/AFMServer/Controllers/AFMKitMLXChatServingAdapter.swift`
- `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift`
- `Sources/AFMServer/Server.swift`
- `Tests/MacLocalAPITests/Qwen38PublishedConfigFixture.swift`
- `Tests/MacLocalAPITests/AFMMLXModelArchitectureTests.swift`
- `Tests/MacLocalAPITests/AFMMLXLoadedModeSwitchPolicyTests.swift`
- `Tests/MacLocalAPITests/AFMMLXProviderTests.swift`
- `Tests/MacLocalAPITests/AFMMLXRuntimeTests.swift`
- `Tests/MacLocalAPITests/AFMMLXModelStoreTests.swift`
- `Tests/MacLocalAPITests/MLXChatCompletionsControllerStreamingTests.swift`

Preferred new AFMKitMLX files:

- `Sources/AFMKitMLX/AFMMLXVisionAssetQualification.swift`
- `Sources/AFMKitMLX/Models/AFMMLXVisionAssetValidator.swift`
- `Tests/MacLocalAPITests/AFMMLXVisionAssetQualificationTests.swift`
- `Tests/MacLocalAPITests/AFMMLXStartupFactoryPolicyTests.swift`
- `Tests/MacLocalAPITests/MLXMediaPreflightTests.swift`
- `Tests/MacLocalAPITests/MLXCapabilityEndpointTests.swift`

Reviewed but unchanged unless a narrow implementation need is demonstrated:

- `Sources/AFMKitMLX/Models/MLXCacheResolver.swift`: base language snapshot
  completeness must not absorb optional vision validation; an accessor for
  stable snapshot identity is the maximum expected change.
- `Sources/AFMKitCore/AFMCoreTypes.swift`: existing capability vocabulary.
- `Sources/AFMOpenAICompat/OpenAIRequest.swift`: ordered image parts already work.
- `Sources/AFMOpenAICompat/OpenAIResponse.swift`: typed error `code` already works.
- bundled WebUI files: no change.

Conditional only after a demonstrated live compatibility failure:

- `Scripts/patches/VLMModelFactory.swift`
- `Scripts/patches/Qwen3_5MoEVL.swift`
- `Scripts/patches/Qwen3VL.swift`
- `Scripts/apply-mlx-patches.sh` only if the mapped patch set changes.

## Dependency and patch implications

- Keep the pinned mlx-swift-lm dependency revision unless live qualification
  proves the fix cannot be represented in the existing AFM-owned patch set.
- Never edit `.build/checkouts`, SwiftPM-managed sources, or the llama.cpp
  submodule directly.
- Reuse MLXVLM, `Qwen3VLProcessor`, and the existing `UserInput` media pipeline;
  do not add an image-processing dependency.
- The preflight protocol method has a default implementation for source
  compatibility, while the concrete AFMKitMLX service remains authoritative.
- Asset validation uses structured metadata and caches by snapshot identity. It
  does not alter base resolution and does not scan tensor payloads per request.

## Backward compatibility

- OpenAI request/response JSON and ordered `image_url` content remain unchanged.
- Complete Qwen `qwen3_5`/`qwen3_5_moe` conditional-generation snapshots now
  start once through VLM; this is the intentionally narrow behavior change.
- Text on that VLM container avoids media decoding and vision execution. Its
  throughput and memory are qualification criteria, not reasons to add reloads.
- Incomplete optional Qwen vision assets select LLM, preserve text startup, omit
  runtime vision capability, and reject direct image requests clearly.
- Explicit `--vlm` remains authoritative with its current startup semantics.
- Gemma, language-only Qwen, and other architectures retain their current
  startup factory policy.
- No request changes container identity, so prompt/radix cache and scheduler
  lifetimes remain tied to the single startup container.
- Existing Qwen MTP integration is text-model-oriented. Issue #191 does not add
  VLM MTP support: when the narrow rule selects VLM, an incompatible explicit
  `--mtp` combination fails clearly at startup and never silently routes media
  to an LLM.

## Risks and mitigations

- **Over-broad factory rule:** a generic "dual-mode means VLM" would regress
  Gemma/cache behavior. Require canonical Qwen family, conditional-generation
  shape, complete contract, and asset-usable qualification in tests.
- **Text performance and memory:** a 27B VLM container may cost more than its LLM
  path even when the vision tower is skipped. Measure startup, peak resident
  memory, and fixed-prompt throughput; do not solve a failed budget with mutable
  request-time containers inside this issue.
- **False asset completeness:** metadata can declare vision while shards or keys
  are absent. Validate indexed shard presence and vision namespaces; support
  unindexed safetensor headers without reading tensor payloads.
- **Text startup regression:** optional vision checks must not feed base cache
  completeness. Test missing processor/config/token/vision-weight cases through
  resolver plus LLM text startup.
- **Mutable/immutable state mixing:** putting `currentFactory` in qualification
  can stale provider descriptors. Store evidence immutably and publish one
  coherent runtime snapshot under the service lock.
- **Capability disagreement:** `/props`, the loaded `/v1/models` entry, and media
  preflight must derive from the same loaded descriptor; catalog declarations
  remain separately named and never authorize requests.
- **Processor variants:** support `preprocessor_config.json`,
  `processor_config.json`, and factory precedence without filename-only guesses.
- **Streaming errors:** preflight after SSE commitment yields HTTP 200. Invoke it
  before body creation and test exact status/type/code.
- **Remote images and WebUI caching:** preserve current body/network behavior;
  use deterministic data URLs in automation and refresh model properties during
  WebUI acceptance.
- **Large-model cost:** keep CI fixture/fake coverage deterministic and gate live
  Qwen runs separately with revision, timing, and memory evidence.

## Revised test matrix

### Unit tests

| Area | Cases and assertions |
| --- | --- |
| Architecture | Published Qwen 3.8 resolves to canonical `qwen3_5` conditional generation from config; arbitrary repository names do not matter; name-only Qwen3.8 and language-only Qwen are not classified as asset-usable vision. |
| Immutable qualification | Complete processor + config/token IDs + indexed vision keys is asset-usable; missing processor, token IDs, vision config, shard, vision namespace, malformed index, and standalone weights are deterministic. The value has no loaded-factory/runtime state. |
| Base cache independence | Every language-usable fixture with missing optional vision evidence still resolves through `localModelDirectory`; qualification reports unusable separately and is cached by snapshot identity. |
| Narrow startup factory | Complete Qwen conditional generation selects VLM; incomplete optional vision selects LLM; similarly shaped non-Qwen and non-conditional Qwen do not trigger the rule. Gemma retains current policy. |
| Explicit factory behavior | `--vlm` retains existing authority; already vision-only models retain current selection; direct VLM load failure follows startup failure semantics and never creates a request-time fallback/swap. |
| Static runtime state | Service publishes qualification and actual factory separately under its lock; text/image preflight never invokes either factory and never changes container identity. |
| Processor/model fast path | Text-only Qwen VLM preparation performs no image decode and no vision-tower execution; JPEG/PNG reach vision; multiple images preserve order. |
| DTO conversion | String and ordered mixed text/image parts, data/HTTP URLs, `detail`, and histories decode without dropping or reordering images. |
| Provider preflight | Complete VLM runtime accepts image; incomplete declared vision returns typed missing categories; unsupported media is distinct; text always remains admissible for the language-usable fallback. |
| Capability surfaces | Loaded descriptor, `/props`, and loaded `/v1/models` entry agree for complete VLM, incomplete LLM fallback, and text-only models. Catalog-only declared capability cannot pass request admission. |
| Error mapping | Missing assets map to `400 invalid_request_error` / `vision_assets_unavailable`; messages omit local paths. Stream and non-stream mappings match. |
| Regression | Gemma factory/cache tests, language-only Qwen, explicit `--vlm`, Qwen text, MTP compatibility behavior, and generic OpenAI chat remain green. |

### Integration tests

| Flow | Cases and assertions |
| --- | --- |
| Static complete-Qwen lifecycle | Resolve complete Qwen, select and load VLM exactly once, send text then image then text, and assert one unchanged container/factory for the full model lifetime. |
| Incomplete-assets fallback | Resolve the same language snapshot with each optional vision category absent, load LLM, complete text generation, advertise vision false, and reject image without any VLM factory call. |
| Concurrent static container | Concurrent text/image and two image requests use one already-loaded VLM container; there is no promotion task, duplicate load, container swap, or scheduler shutdown. |
| Non-streaming controller | Provider preflight precedes reservation/generation; usable image returns assistant JSON; incomplete assets return the stable HTTP 400. |
| Streaming controller | Provider preflight finishes before headers/body; usable image streams normally; incomplete assets return HTTP 400 rather than an HTTP 200 error token. |
| WebUI contract | `/props` enables vision only for the usable loaded VLM; WebUI-style base64 `image_url` is preserved; incomplete/text-only models disable attachment submission. |
| Grounding and media order | JPEG, PNG, mixed history, and two ordered image parts reach the processor and produce image-conditioned output. |
| Other models | Gemma, language-only Qwen, unsupported-media models, and explicit `--vlm` preserve existing startup and error behavior. |

### Live qualification and acceptance

Run only after implementation approval and deterministic tests:

1. Build with the repository wrapper and launch without `--vlm` against recorded
   Qwen 3.8 revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
   Assert startup qualification selected one VLM container before any request.
2. Send a fixed text prompt first. Record startup time, tokens/second, peak
   resident memory, selected factory/container identity, and instrumentation
   proving no image decode or vision-tower execution.
3. In a fresh WebUI session, attach a JPEG and then a PNG and ask facts grounded
   in visible content. Verify `/props` enabled attachment submission and both
   answers are correct.
4. Send equivalent data-URL API requests in streaming and non-streaming modes;
   record statuses, chunks, finish reasons, and confirmation that the original
   startup container remained unchanged.
5. Use two deliberately different images with the same predeclared grounded
   prompt. Assert each response matches its own image and the answers differ;
   repeat once to reduce chance success.
6. Send text again and verify no image processing or vision execution. Compare
   text throughput/memory with the pre-image text request and a recorded
   pre-change LLM baseline; no factory transition is expected.
7. Run concurrent text/image and two-image requests. Verify one static VLM
   container, no reload, no scheduler shutdown, bounded memory, and correct
   outputs.
8. Use disposable snapshots missing processor metadata and vision weight
   evidence. Verify base resolution and LLM text startup succeed, `/props` and
   the loaded model entry report vision false, and both stream modes reject
   direct images with the stable HTTP 400 before response commitment.
9. Run a separate explicit `--vlm` launch as a control and verify its behavior is
   unchanged. Repeat the core grounding case on MXFP8 only after all indexed
   shards are locally available; otherwise record it as unexecuted.
10. Run repository smoke assertions and Gemma/Qwen regressions. Save revision,
    request hashes, redacted logs, outputs, timings, memory, and WebUI screenshots
    under the normal test-report location without committing model artifacts or
    machine-specific cache paths.

## Fixed decisions and remaining qualification gates

Architecture review fixed the following decisions: Qwen remains `qwen3_5`; the
narrow complete-Qwen rule uses VLM at startup; containers are static per load;
missing optional vision assets preserve LLM text startup; direct unusable-vision
requests use the approved HTTP 400 contract; image acceptance is in scope while
video behavior is only preserved; patch changes require a reproduced direct-VLM
failure; and WebUI/upstream changes remain out of scope.

The remaining gates are empirical, not alternative architecture proposals:

- quantify text throughput, startup latency, and peak memory on the VLM
  container;
- directly qualify the existing Qwen processor/model/weight patch before
  changing it; and
- defer MXFP8 live coverage until its local snapshot is complete.

## Architecture review verdict and resolution trace

The durable review dated 2026-08-17 returned **REQUEST CHANGES** for commit
`87b06c8`. Its blocking finding was that request-time promotion could not both
avoid dual 27B residency and retain the LLM container for rollback, while the
service has no reversible quiescence lifecycle. Implementation remains blocked
until this revised plan is re-gated.

| Review requirement | Resolution in this revision |
| --- | --- |
| 1. Narrow startup Qwen VLM rule | Section 2 selects VLM only for configuration-resolved `qwen3_5`/`qwen3_5_moe` conditional generation with a complete vision contract/assets, keeps other architectures unchanged, and defines the VLM no-media text path plus live performance qualification. |
| 2. Remove request-time promotion | Sections 2, Backward compatibility, Risks, and all test matrices define one static startup container and remove promotion, rollback, quiescence, dual residency, and cache-transfer work. |
| 3. Separate capability evidence from runtime state | Section 1 makes qualification immutable and asset-only; Section 2 stores actual factory/container state solely under `MLXModelService` synchronization. |
| 4. Keep optional vision separate from base cache completeness | Sections 1 and Exact implementation files preserve `localModelDirectory` language completeness, add a sibling snapshot-cached validator, and require LLM text startup for incomplete optional vision assets. |
| 5. Keep decisions in AFMKitMLX | Section 3 and the ownership table assign files, factory choice, runtime admission, and typed errors to AFMKitMLX; AFMServer only invokes side-effect-free preflight and maps the result before SSE commitment. |
| 6. Define capability surfaces | Section 4 separates declared/catalog from runtime-usable capability, derives `/props` and the loaded `/v1/models` entry from the loaded descriptor, and prohibits catalog capability from request admission. |
| 7. Replace promotion tests | The revised unit, integration, and live matrices cover narrow startup selection, incomplete-assets LLM fallback, one static VLM container under concurrency, text no-media execution, and unchanged Gemma/language-Qwen/`--vlm` behavior while retaining DTO, controller, JPEG/PNG, grounding, and regression coverage. |

Architecture gate 2 approved this revision at commit `a1915e2`; implementation
may proceed within the approved scope.

## Implementation decisions and verification

### Checkpoint 1: immutable qualification and startup policy

- Added `AFMMLXVisionAssetQualification` as immutable snapshot evidence. It
  contains architecture/config/processor/weight results and stable missing
  categories, but no loaded factory, container, or scheduler state.
- Added a sibling `AFMMLXVisionAssetValidator`. Its cache key fingerprints the
  resolved directory plus relevant config, processor, index, and safetensor file
  metadata. Indexed checkpoints use structured `weight_map` keys and shard
  presence; standalone checkpoints read only the bounded safetensor JSON header.
- Matched VLM factory processor precedence: `preprocessor_config.json` wins over
  `processor_config.json`, and valid Qwen 3.5-family metadata resolves through
  the existing `Qwen3VLProcessor` factory override.
- Kept `MLXCacheResolver.hasRequiredFiles` unchanged. Missing processor metadata,
  vision config/token IDs, or vision weights makes optional vision unusable but
  does not make the language snapshot incomplete.
- Extended startup factory policy only for an asset-usable, config-resolved
  Qwen `qwen3_5`/`qwen3_5_moe` conditional-generation qualification. Explicit
  `--vlm`, vision-only behavior, Gemma, and incomplete-Qwen LLM selection retain
  their approved ordering.

Verification on 2026-08-17:

- `./Scripts/swiftpm-reliable.sh test --filter
  'AFMMLX(VisionAssetQualification|StartupFactoryPolicy)Tests'`
- Result: 9 tests executed, 0 failures.
- The first run exposed and fixed a Swift 6 optional-flattening compile error in
  the bounded safetensor reader; the rerun passed.

### Checkpoint 2: static runtime capability state

- `MLXModelService.ensureLoaded` now runs snapshot qualification before factory
  selection and passes it into the narrow startup policy. The service publishes
  immutable qualification, actual factory, runtime descriptor, architecture,
  container, and MTP binding together under `stateLock`; terminal shutdown
  clears all of those runtime fields together.
- Added `AFMMLXRuntimeVisionPolicy` as a pure decision layer. A loaded descriptor
  advertises vision only when the active factory is VLM and, for Qwen
  conditional generation, the immutable qualification is asset-usable.
  Complete qualification cannot authorize an LLM-backed container.
- Added side-effect-free media preflight to `AFMMLXOpenAIChatServing` with
  source-compatible defaults. `MLXModelService` classifies ordered message
  parts from one locked runtime snapshot; it performs no file read, factory
  invocation, reload, scheduler operation, or state mutation.
- Added typed `visionAssetsUnavailable` and distinct `unsupportedMediaInput`
  service failures. Missing-asset diagnostics contain sorted category names and
  no snapshot paths. Direct generation retains the same internal media guard,
  so non-HTTP callers cannot bypass runtime capability validation.
- `AFMMLXRuntime.load` now returns the authoritative post-load descriptor
  synthesized by the service instead of returning its pre-load catalog
  descriptor.

Verification on 2026-08-17:

- `./Scripts/swiftpm-reliable.sh test --filter
  'MLXMediaPreflightTests|AFMMLXStartupFactoryPolicyTests|AFMMLXVisionAssetQualificationTests|AFMMLXRuntimeTests'`
- Result: 20 tests executed, 0 failures. Log:
  `.build-reliable-logs/test-20260817-171828.log`.
- The first compile exposed an ambiguous `flatMap` result for messages without
  multipart content; the array result was made explicit and the identical
  focused command then passed.
- No compatibility patch source changed. The repository wrapper only reapplied
  the existing AFM-owned patch set to the local vendor worktree.

### Checkpoint 3: server preflight and capability surfaces

- `AFMKitMLXChatServingAdapter` now forwards media preflight and the loaded
  descriptor for concrete MLX services. Its fixed-model path validates media
  against that model's descriptor without reading files or changing runtime
  state.
- `MLXChatCompletionsController` invokes provider preflight after request/model
  validation but before slot reservation and before streaming response
  construction. The typed missing-assets failure maps to HTTP `400`,
  `invalid_request_error`, and `vision_assets_unavailable` for streaming and
  non-streaming requests; no generation or slot accounting starts first.
- `/props` and the loaded `/v1/models` details entry now read the same
  runtime-usable descriptor through the serving abstraction. Both fail closed
  for vision when no loaded descriptor is available. The existing response
  shapes and WebUI contract are unchanged.
- Added explicit `declaredDescriptor` and `isDeclaredVisionModel` discovery APIs
  to `AFMMLXModelStore`, retaining the old spellings as compatibility aliases.
  Their documentation prohibits using catalog declarations for runtime media
  admission.

Verification on 2026-08-17:

- `./Scripts/swiftpm-reliable.sh test --filter
  'MLX(ChatCompletionsControllerStreaming|CapabilityEndpoint)Tests|AFMKitMLXReasoningPropagationTests|AFMMLXProviderTests'`
- Result: 59 tests executed, 0 failures. Log:
  `.build-reliable-logs/test-20260817-172309.log`.
- `./Scripts/swiftpm-reliable.sh test --filter
  'AFMMLXModelStoreTests|MLXCapabilityEndpointTests|MLXChatCompletionsControllerStreamingTests'`
- Result: 59 tests executed, 0 failures. Log:
  `.build-reliable-logs/test-20260817-172415.log`.
- Controller tests assert both stream modes return JSON before response
  commitment and observe one preflight call, zero slot reservations, and zero
  generation calls for missing vision assets.
