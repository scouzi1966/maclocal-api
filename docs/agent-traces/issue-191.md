# Issue #191 Phase A: Qwen 3.8 VLM Planning Trace

Status: planning complete; implementation requires reviewer approval.

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
- Loaded service state records model ID, architecture, configuration, and
  container but not the actual factory that created the container
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:223`,
  `Sources/AFMKitMLX/Models/MLXModelService.swift:1673`).
- Request validation consequently accepts image input based on architecture and
  returns the existing container, even when that container is the LLM one
  (`Sources/AFMKitMLX/Models/MLXModelService.swift:6010`). This is the principal
  routing gap behind the issue.

### The MLX compatibility patch already contains a Qwen vision implementation

- The patched VLM registry maps `qwen3_5` and `qwen3_5_moe` to
  `Qwen3_5MoEVL`, and the processor registry contains `Qwen3VLProcessor`
  (`Scripts/patches/VLMModelFactory.swift:84`,
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
- The local cached revision of `mlx-community/Qwen3.8-27B-4bit` was inspected as
  qualification evidence: its config matches the fixture, its processor config
  names `Qwen3VLProcessor`, and its safetensor index contains both
  `vision_tower` and `language_model` keys. This supports using the existing
  patched Qwen 3.5 VLM implementation, subject to a live load test.

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

## Proposed architecture

### 1. Define a config- and asset-driven vision contract

Add an AFMKitMLX value type representing:

- declared media capability from decoded architecture/configuration;
- required image token IDs and vision configuration;
- processor metadata availability and selected processor class;
- presence of vision-tower weights in either a safetensor index or standalone
  safetensor files; and
- the current load factory (`llm` or `vlm`).

Detection must use published configuration and weight metadata, not the model
repository name. Qwen 3.8 should remain a `qwen3_5` conditional-generation
checkpoint. The contract should distinguish "the architecture supports images"
from "the local snapshot has all assets needed to serve images."

Vision asset checks should be lazy or separately scoped so an incomplete vision
snapshot can continue to serve text. An image request against it should fail
before generation with a typed error listing the missing category (processor
configuration, token IDs/vision config, or vision weights), without printing
machine-specific cache paths.

### 2. Track the loaded factory and promote dual-mode models atomically

Keep the current LLM-first startup for dual-mode Qwen checkpoints. Before slot
reservation or response commitment, classify the request. For an image request
whose model is image-capable and asset-complete but currently loaded through the
LLM factory, perform a single-flight, one-way promotion to a VLM container:

1. block new generation admission for that model;
2. wait for active scheduler work to quiesce, or return an explicit busy/retry
   error if safe quiescence cannot be guaranteed;
3. load the same snapshot through `VLMModelFactory`;
4. atomically replace the container and record the actual factory;
5. invalidate prefix/radix caches and any factory-specific MTP state; and
6. resume admission and build the multimodal `UserInput`.

Concurrent image requests must share the same promotion task. Do not retain both
27B containers, and do not demote after each request. After promotion, text-only
requests remain free of image decoding and vision-tower execution because the
Qwen processor/model already branch on absent media; live tests must quantify
the remaining VLM-container throughput and memory cost.

This one-way lazy promotion is the recommended compromise: it preserves the
existing LLM fast path until vision is actually requested, avoids duplicate
residency, and avoids repeated reload thrash. It needs reviewer approval because
text requests after the first image no longer use the original LLM container.

### 3. Add pre-response request preparation to the server boundary

Extend the AFMKitMLX chat-serving protocol with an async preparation operation
that defaults to a no-op for test doubles and non-MLX adapters. The concrete MLX
service will use it for media classification, vision asset validation, and any
factory promotion. Call it from the controller before non-streaming generation
and before streaming headers/body are committed.

Return a stable HTTP 400 OpenAI-style error for missing vision assets, proposed
as `type: invalid_request_error` and `code: vision_assets_unavailable`. Preserve
the existing DTO and ordered content-part representation; no request schema
fork is needed.

### 4. Make capability endpoints descriptor-backed

Replace unconditional MLX vision flags in `/v1/models` and `/props` with the
same resolved capability contract used by request preparation. For a locally
loaded/downloaded model, advertise vision only when the configuration declares
it and required local vision assets are complete. Text-only models must report
vision false. Keep the response shape expected by the bundled WebUI so its
existing attachment path needs no source change.

If model listing must describe a curated but not-yet-downloaded model, use its
catalog descriptor and treat that as advertised/catalog capability, while
`/props` for the selected local model should report runtime-usable capability.
This distinction must be documented in code and tested.

### 5. Qualify the existing mlx-swift-lm compatibility patches

The static path already maps Qwen 3.8's published `qwen3_5` metadata to
`Qwen3_5MoEVL` and `Qwen3VLProcessor`, including vision weight remapping. Do not
add a speculative `qwen3_8` model implementation. First run focused live
qualification through the VLM factory.

Only if that qualification exposes a loader/processor incompatibility should
the repository-owned patch files be updated. Any such change belongs under
`Scripts/patches/` and must be reproducible through the patch application
script. No upstream mlx-swift-lm PR is part of this issue.

## Ownership by layer

| Layer | Required? | Responsibility |
| --- | --- | --- |
| maclocal-api / AFMKitMLX | Yes | Config/asset contract, actual-factory state, lazy promotion, cache/MTP invalidation, typed vision errors. |
| maclocal-api / AFMServer | Yes | Pre-response preparation, consistent stream/non-stream errors, descriptor-backed `/v1/models` and `/props`. |
| maclocal-api / AFMOpenAICompat | Test-only expected | Existing multimodal DTO is sufficient; add code only if tests expose a decoding incompatibility. |
| Bundled WebUI (`vendor/llama.cpp`) | No feature change expected | Existing attachment conversion is correct once capability and backend routing are correct. Do not edit the submodule for this issue unless a reproducible WebUI contract defect is found. |
| mlx-swift-lm compatibility patches | Qualification required; code change conditional | Existing Qwen VLM model/factory should be sufficient. Patch only a demonstrated compatibility gap in repository patch files. |
| External AFMKit or upstream repositories | No | AFMKit is in this repository. Do not create upstream PRs. |

## Likely implementation files

Required or strongly likely:

- `Sources/AFMKitMLX/AFMMLXModelArchitecture.swift`
- `Sources/AFMKitMLX/AFMMLXLoadedModeSwitchPolicy.swift`
- `Sources/AFMKitMLX/Models/MLXModelService.swift`
- `Sources/AFMKitMLX/Models/MLXCacheResolver.swift`
- `Sources/AFMKitMLX/AFMMLXOpenAIChatGenerating.swift`
- `Sources/AFMServer/Controllers/MLXChatCompletionsController.swift`
- `Sources/AFMServer/Server.swift`
- `Tests/MacLocalAPITests/Qwen38PublishedConfigFixture.swift`
- `Tests/MacLocalAPITests/AFMMLXModelArchitectureTests.swift`
- `Tests/MacLocalAPITests/AFMMLXLoadedModeSwitchPolicyTests.swift`
- New focused asset-validation, promotion, capability-endpoint, and controller
  preflight tests under `Tests/MacLocalAPITests/`.

Possible new production file to keep the contract testable and out of the model
service actor: `Sources/AFMKitMLX/AFMMLXVisionCapability.swift`.

Conditional only after a demonstrated live compatibility failure:

- `Scripts/patches/VLMModelFactory.swift`
- `Scripts/patches/Qwen3_5MoEVL.swift`
- `Scripts/patches/Qwen3VL.swift`
- `Scripts/apply-mlx-patches.sh` only if the mapped patch set changes.

No change is expected in the OpenAI DTO files or bundled WebUI source.

## Dependency and patch implications

- Keep the pinned mlx-swift-lm dependency revision unless live qualification
  proves the required fix cannot be represented in the existing patch set.
- Never edit `.build/checkouts`, SwiftPM-managed sources, or the llama.cpp
  submodule directly. Regenerate/apply compatibility changes through the
  repository patch workflow.
- Do not introduce a second image-processing dependency. Reuse MLXVLM,
  `Qwen3VLProcessor`, and the existing `UserInput` media pipeline.
- A new protocol preparation method should have a default implementation to
  avoid breaking fake services and alternate AFMKit adapters.
- If safetensor key validation requires reading an index, use structured JSON
  decoding and support both indexed shards and unindexed safetensor layouts.

## Backward compatibility

- OpenAI request and response JSON remain unchanged; existing text and
  `image_url` clients need no migration.
- Explicit `--vlm` remains authoritative and skips lazy promotion because the
  container is already VLM-backed.
- Text-only Qwen requests before promotion continue on the existing LLM path.
- Missing optional vision assets do not prevent text-only startup or generation;
  they affect capability advertisement and image requests only.
- Non-vision models stop falsely advertising vision. Clients that relied on the
  current unconditional flags will see a correction, not a schema change.
- Promotion invalidates model-specific prompt/radix cache and incompatible MTP
  state. This may reduce cache hit rate once, but prevents cross-factory state
  reuse.
- Existing Gemma and other models retain their current factory policy unless
  they meet the same dual-mode contract and are explicitly covered by tests.

## Risks and mitigations

- **Concurrent promotion and generation:** replacing a container while scheduler
  work is active can crash or corrupt state. Use one admission gate and a
  single-flight transition; add stress tests before enabling it.
- **Text fast-path ambiguity:** the VLM model skips visual computation for text,
  but historical behavior indicates extra memory and lower throughput than the
  LLM container. Preserve LLM until first media and publish live measurements.
- **Memory pressure:** two 27B containers are not acceptable. Promotion must
  release the LLM container before or as the VLM container becomes active,
  without leaving the service in a half-loaded state on failure.
- **Failed promotion rollback:** retain or restore the usable LLM container when
  VLM loading fails, then return a typed image error. Verify subsequent text.
- **Cache and MTP coupling:** current Qwen MTP paths contain text-model casts and
  factory-specific state. Disable/invalidate MTP during promotion until a VLM
  path is explicitly verified; never reuse prompt caches for media.
- **Processor variants:** repositories may provide `preprocessor_config.json`,
  `processor_config.json`, or both. Match existing VLM factory precedence and
  report absence without relying on filenames alone.
- **Weight validation:** indexed and standalone safetensors require different
  inspection. Avoid scanning full tensor payloads during each request; cache a
  snapshot fingerprint and validation result.
- **Streaming error semantics:** validation after body commitment yields HTTP 200
  with an error token. Complete validation/promotion before response commitment.
- **Remote image behavior:** current URL decoding has network, size, and latency
  implications. This issue should preserve current behavior and body limits,
  while tests primarily use deterministic data URLs.
- **WebUI caching:** capability data may be cached in the browser. Integration
  instructions must force model-property refresh/reload before judging a fix.
- **Large-model test cost:** full Qwen 3.8 cannot be a routine unit test. Keep
  deterministic fixtures/fakes for CI and a separately gated live qualification.

## Test matrix

### Unit tests

| Area | Cases and assertions |
| --- | --- |
| Architecture | Published Qwen 3.8 fixture resolves to `qwen3_5`; arbitrary repository name still works; name-only `Qwen3.8` without vision config is not vision-capable; language-only Qwen remains unchanged. |
| Asset contract | Complete processor + vision config/token IDs + indexed `vision_tower` keys is usable; missing processor config, missing token IDs, missing vision config, missing vision keys, malformed index, and standalone weights each produce deterministic results. |
| Factory policy | Dual-mode complete Qwen starts LLM; `--vlm` starts VLM; text request causes no transition; first image causes one promotion; already-VLM request does not reload. |
| Promotion concurrency | Multiple simultaneous image preparations await one load; text admission during transition follows the defined gate; failed promotion restores text service; cancellation does not strand transition state. |
| Cache/MTP state | Promotion invalidates prefix/radix cache and MTP binding; media never uses prompt cache; subsequent text can establish cache only against the new container. |
| Processor/model | Text-only Qwen VLM preparation does not decode images or execute the vision tower; JPEG and PNG tensors reach the vision path; multiple images preserve order. |
| DTO conversion | String content, mixed text/image parts, data URLs, HTTP URLs, `detail`, and multi-message histories decode without dropping image parts. |
| Error mapping | Unsupported model and incomplete vision assets produce distinct stable errors; error text names missing asset categories and omits local paths. |
| Capabilities | Provider descriptor, `/v1/models`, and `/props` agree for complete vision, incomplete vision, and text-only models; curated unavailable models retain catalog semantics. |
| Regression | Existing Gemma VLM factory tests, Qwen text tests, forced-VLM tests, and generic OpenAI chat tests remain green. |

### Integration tests

| Flow | Cases and assertions |
| --- | --- |
| Non-streaming controller | OpenAI request with an image invokes preparation before generation and returns normal assistant JSON; incomplete assets return HTTP 400 with the stable code. |
| Streaming controller | Preparation finishes before headers/body; image request streams normal chunks; incomplete assets return protocol-level HTTP 400 rather than an HTTP 200 error token. |
| WebUI contract | `/props` reports usable vision, WebUI-style base64 `image_url` payload is preserved, and a text-only/incomplete model reports false so attachments are not falsely enabled. |
| Lazy promotion | Load Qwen through fake LLM factory, send text, send image, verify one VLM load and container swap, then send text again without image preprocessing. |
| Multiple media | Mixed history and two image parts retain ordering and are passed once to the processor. |
| Concurrency | Two image requests arriving together cannot double-load; active text plus image transition follows the documented admission behavior without deadlock. |
| Failure recovery | Inject VLM load failure, confirm image error, then confirm text generation still succeeds through the retained/restored LLM container. |
| Other models | Gemma VLM, language-only Qwen, and a model without media support preserve existing routing and errors. |

### Live qualification and acceptance

Run this matrix only after implementation approval and focused automated tests:

1. Build using the repository's reliable SwiftPM/build wrapper and start the
   server with the exact cached Qwen 3.8 revision recorded in the test report.
2. Before any image request, run a fixed text prompt and record load time,
   tokens/second, peak resident memory, selected factory, and whether image
   preprocessing/vision execution occurred.
3. In a fresh WebUI session, attach a JPEG, ask a fact grounded in visible
   content, and verify the response. Repeat with PNG.
4. Send equivalent `/v1/chat/completions` data-URL requests through `curl` in
   non-streaming and streaming modes; record status, chunks, finish reason, and
   server factory-transition logs.
5. Use two deliberately different images with the same grounded prompt. Define
   expected facts before running, assert each response matches its own image,
   and assert the answers differ. Repeat once to guard against a lucky response.
6. Send text after promotion and verify no image decoding or vision-tower
   execution. Record throughput/memory versus the pre-promotion baseline.
7. Use a disposable local snapshot with processor metadata absent, then one with
   vision weight metadata absent. Verify text still works, capability is false,
   and image requests return the stable clear error in both stream modes.
8. Run concurrent text/image and two-image-request cases while watching for
   deadlock, duplicate loading, memory spikes, and stale cache use.
9. Repeat the core API grounding case with the curated MXFP8 Qwen 3.8 variant if
   locally available; otherwise record it as an unexecuted coverage item.
10. Run the repository smoke assertions and relevant Gemma/Qwen regression
    suites. Save model revision, request hashes, redacted logs, outputs, timings,
    and WebUI screenshots under the normal test-report location; do not commit
    model artifacts or machine-specific cache paths.

## Unresolved architectural questions for reviewer approval

1. **Factory lifecycle:** approve the recommended one-way lazy promotion, or
   require VLM-at-startup? Dual residency is not recommended for a 27B model.
2. **Meaning of "text-only fast path":** is preserving LLM performance until
   the first image sufficient, with VLM text-only execution afterward, or must
   every later text request return to the LLM factory? The latter implies costly
   demotion/reload or duplicate residency.
3. **Admission during promotion:** should active requests be drained while new
   requests wait, or should image requests receive a retryable busy error when
   the scheduler is active? Draining is preferable if the scheduler exposes a
   reliable quiescence primitive.
4. **Capability semantics:** should `/v1/models` describe catalog capability
   while `/props` describes locally usable capability, as proposed, or must both
   be false whenever the selected local snapshot is incomplete?
5. **MTP behavior:** is disabling MTP across the first vision promotion and
   using autoregressive generation acceptable until Qwen VLM MTP is separately
   qualified? Current service integration is text-model-oriented.
6. **Error contract:** approve HTTP 400 with
   `invalid_request_error`/`vision_assets_unavailable`, or preserve the existing
   generic `mlx_error` type for compatibility.
7. **Video scope:** Qwen metadata and current routing advertise video, but issue
   acceptance is image-only. The recommendation is to preserve existing video
   behavior without expanding this implementation or its live acceptance gate.
8. **Patch gate:** static evidence says the existing mlx-swift-lm compatibility
   patches should load Qwen 3.8. Any patch change should wait for the direct VLM
   live qualification rather than be assumed during implementation.

Implementation must not begin until these planning decisions receive reviewer
approval.
