# Claude Code Reference for MacLocal API

## Project Overview

AFM (Apple Foundation Models) is an OpenAI-compatible API server for local LLM inference on Apple Silicon. It supports two backends:
- **MLX** (`afm mlx`): Uses mlx-swift-lm for MLX-format models from Hugging Face
- **Apple Foundation Models** (`afm` without subcommand): Uses Apple's on-device models (macOS 26+)

The server exposes `/v1/chat/completions` and `/v1/models` endpoints compatible with OpenAI API clients.

## Project Structure

The package vends **two SPM products**: a `.library(AFMKit)` (headless, importable
`import AFMKit`) and a `.executable(afm)` thin CLI. Library import is proven by
`Examples/AFMKitConsumer`. (The clean Vapor-free `AFMServer` split is a tracked follow-up —
today AFMKit also contains the server.)

```
Sources/
├── AFMKit/                             # LIBRARY target (importable via SPM)
│   ├── AFMEngine.swift                 # Public facade: AFMEngine actor + EngineConfig/GenerationConfig/AFMResponse
│   ├── AFMLanguageModel.swift          # WWDC26-shaped provider protocol (AFMEngine conforms)
│   ├── Server.swift                    # Vapor HTTP server (currently in AFMKit)
│   ├── Controllers/
│   │   ├── MLXChatCompletionsController.swift  # Streaming/non-streaming SSE handler
│   │   └── ...
│   ├── Models/
│   │   ├── MLXModelService.swift       # Model loading, generation, prompt caching
│   │   ├── OpenAIRequest.swift         # Request types (ChatCompletionRequest, etc.)
│   │   ├── OpenAIResponse.swift        # Response types
│   │   └── ...
│   ├── BuildInfo.swift                 # version string (SHA injected by build.sh)
│   └── Resources/default.metallib      # MLX Metal kernels → bundle MacLocalAPI_AFMKit.bundle
├── AFMCLI/                             # EXECUTABLE target (product name: afm)
│   ├── main.swift                      # CLI entry point (ArgumentParser)
│   ├── {Mlx,Serve,Vision,Speech,Embeddings}Command.swift
│   └── Info.plist                      # embedded into the binary (privacy usage descriptions)
Examples/AFMKitConsumer/                # standalone SPM package proving `import AFMKit`
docs/wwdc26-migration.md                # WWDC26 Foundation Models adoption seam map
vendor/
├── mlx-swift-lm/                       # Git submodule — DO NOT modify directly
├── llama.cpp/                          # Git submodule
Scripts/
├── patches/                            # Our patches to vendor code (copied over originals)
├── apply-mlx-patches.sh                # Applies patches from Scripts/patches/ to vendor/
├── build-from-scratch.sh               # Full build: submodules + patches + webui + build
```

## Vendor Patch System

**NEVER modify files in `vendor/` directly.** All changes go through `Scripts/patches/`.

The patch script (`Scripts/apply-mlx-patches.sh`) copies complete Swift files from `Scripts/patches/` to vendor targets. Three arrays define the mapping:
- `PATCH_FILES=()` — filenames in `Scripts/patches/`
- `TARGET_PATHS=()` — relative paths under `vendor/mlx-swift-lm/`
- `NEW_FILES=()` — files that don't exist upstream

Commands: `--check` (verify), `--revert` (restore originals), no flag (apply).

### MLX C++ / Metal-kernel patches (separate from the Swift patch set)

The Swift patches above target the `vendor/mlx-swift-lm` submodule. The low-level MLX **C++/Metal**
code lives in a *different* tree — the `mlx-swift` remote SwiftPM dependency at
`.build/checkouts/mlx-swift` (ephemeral; wiped by `swift package clean`/re-resolve). Two scripts
patch it, applied by `build.sh` after `swift package resolve` and before the metallib rebuild:

- `Scripts/apply-mlx-cpp-patches.sh` — `qmv_fast_wide` quantized matvec kernels.
- `Scripts/apply-mlx-sdpa-backport.sh` — backports mlx-swift **0.31.3's adaptive-block 2-pass SDPA**
  into the pinned 0.30.3 tree. 0.30.3 hardcodes the split-K count `blocks=32`; 0.31.3 makes it a
  runtime function-constant scaled by sequence length (up to 1024), giving **decode@16k ~+10%
  (≈13.0→14.4 tok/s on Qwen3.6-27B-4bit/M4 Pro), correct at all depths**. Source files live in
  `Scripts/patches/mlx-cpp-sdpa/`; it also inserts the `check_kernel_threadgroup_size` helper into
  `utils.h`. Both scripts support `--check`/`--revert`. **After applying, the metallib MUST be
  rebuilt** (`Scripts/rebuild-metallib.sh`) so the kernel change takes effect — see the Build section.

## Build

**IMPORTANT:** Always run the full build with ALL steps (submodules, patches, webui, metallib) unless the user explicitly asks to skip a step. Never add `--skip-webui`, `--skip-patches`, `--skip-submodules`, or `--skip-metallib` on your own.

```bash
swift build                              # Debug build
swift build -c release                   # Release build
./Scripts/build-from-scratch.sh          # Full build (submodules + patches + webui + clean + metallib + build)
```

### MLX Metal shader library (`default.metallib`)

`swift build` does **NOT** compile any Metal. The MLX kernels ship as a prebuilt
`Sources/AFMKit/Resources/default.metallib` (committed to git) that `swift build` only
copies into the app bundle. The kernel *sources* live in the resolved `mlx-swift` dependency
(`.build/checkouts/mlx-swift/.../kernels/*.metal`), so editing a kernel (e.g. `sdpa_vector.h`)
has **zero effect** until the metallib is regenerated. (Editing the dispatch C++ in
`scaled_dot_product_attention.cpp` *does* recompile — so a kernel/dispatch mismatch silently
produces garbage at every context length.)

`./build.sh` regenerates the metallib from source as step 4b via `Scripts/rebuild-metallib.sh`
(compiles the pinned kernel set, links with `metal -o`, verifies kernel-symbol parity, installs).
This needs the **Metal Toolchain**, which Xcode 26 ships as a separate downloadable component:

```bash
xcodebuild -showComponent MetalToolchain        # status (installed/uninstalled)
xcodebuild -downloadComponent MetalToolchain    # one-time ~688 MB install
./Scripts/rebuild-metallib.sh                    # rebuild + parity-check + install
./Scripts/rebuild-metallib.sh --check            # just verify the toolchain is available
./Scripts/rebuild-metallib.sh --no-install        # build to /tmp + parity check, don't replace committed
```

If the toolchain is absent, `build.sh` falls back to the committed prebuilt metallib (after
offering to download). There is no separate `metallib` tool on Xcode 26 — link via `metal -o lib.metallib *.air`.

## Running the Server

**IMPORTANT:** Always set the model cache directory to avoid re-downloading models:

```bash
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model-id> --port 9999
```

Debug logging:
```bash
AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache afm mlx -m <model-id> --port 9999
```

Debug logging shows `[KVCache]` hit/miss stats, tool call detection, and timing info.

### GPU Shader Profiling

Four profiling modes, from lightweight to deep:

```bash
# 1. Per-request stats: device info, memory breakdown, bandwidth estimate (no overhead)
afm mlx -m <model> --gpu-profile -s "Hello"

# 2. + measured DRAM bandwidth via mactop (adds ~5s, requires brew install mactop)
afm mlx -m <model> --gpu-profile-bw -s "Hello"

# 3. xctrace with per-kernel shader names (~100-300 MB, opens in Instruments)
afm mlx -m <model> --gpu-trace 10 -s "Hello"
# Then: open /tmp/afm-metal.trace

# 4. Full Metal GPU capture in Xcode (WARNING: multi-GB traces, small models only)
afm mlx -m <small-model> --gpu-capture /tmp/afm-trace.gputrace -s "Hello"
```

One-time setup for per-kernel shader names in `--gpu-trace`:
```bash
python3 Scripts/create-shader-template.py   # patches Metal System Trace template
```

Live bandwidth monitoring (separate terminal, no sudo):
```bash
./Scripts/gpu-profile.sh bandwidth          # visual bar chart via mactop
```

Helper script: `./Scripts/gpu-profile.sh` wraps all profiling workflows.

**Tradeoffs:**
- `--gpu-profile`: Zero overhead. Device info, memory split (weights vs KV), timing, calculated bandwidth with chip detection.
- `--gpu-profile-bw`: Adds mactop DRAM bandwidth sampling (~5s post-inference). Requires `brew install mactop`.
- `--gpu-trace N`: Lightweight (~100-300 MB for 10-15s). With shader template: captures 60+ MLX Metal kernel names (`affine_qmv_fast`, `steel_gemm_fused`, `sdpa_vector`, etc.). Without: command-buffer timing only.
- `--gpu-capture`: Full Xcode shader debugger with per-line costs. Multi-GB traces, auto-limited to 5 tokens — only practical for small models.

**Measured on Qwen3.5-35B-A3B-4bit (M3 Ultra 512GB):** 100% GPU utilization, ~28W GPU power, 171 GB/s sustained DRAM bandwidth (21.4% of 800 GB/s theoretical) during 4096-token decode at 95.7 tok/s.

**Key kernels (from shader trace):** `affine_qmv_fast` (quantized MatVec, decode bottleneck), `affine_gather_qmv_fast` (MoE expert dispatch), `steel_gemm_fused` (prefill GEMM), `sdpa_vector` (attention), `rmsbfloat16` (normalization), `custom_kernel_gated_delta_step_fused` (Mamba/hybrid layers).

## Key Features

### Sampling Parameters
All functional end-to-end: `temperature`, `top_p`, `repetition_penalty`, `top_k`, `min_p`, `presence_penalty`, `seed`.

Added via vendor patch in `Scripts/patches/Evaluate.swift`: `TopKProcessor`, `MinPProcessor`, `PresenceContext`, `CompositeLogitProcessor`.

Sampler chain order (following llama.cpp): penalties → top_k → min_p → temperature+sampling.

`frequency_penalty`: parsed but silently ignored (not implemented).

### Tool Calling
OpenAI-compatible `tools`, `tool_choice`, `tool_calls` implemented.

**Streaming tool call detection** uses token-level start/end tag matching (mlx-lm Python style):
- Tags derived from model's `ToolCallFormat` (e.g., `<tool_call>`/`</tool_call>`)
- Content outside tool calls streams normally; only tool call body is buffered
- Fallback regex parser (`extractToolCallsFallback`) handles edge cases
- `finish_reason: "tool_calls"` when tool calls are present

**Qwen3-Coder XML format**: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`. Vendor's `ToolCallProcessor` fails on this (regex without `dotMatchesLineSeparators`). Fixed with:
1. `inferToolCallFormat()` reads `model_type` from config.json
2. `extractToolCallsFallback()` post-generation regex parsing
3. Duplicate parameter workaround (keep first non-empty value)

### Logprobs
OpenAI-compatible `logprobs` and `top_logprobs` (0-20) implemented.

Uses `log(softmax(logits/temp))` after processor chain. `logSoftmax` not available in MLX Swift — use `log(softmax(x))`. Use `Swift.min()` to avoid MLX namespace collision.

### Prompt Caching
Server-level single-slot token-level prefix matching (llama.cpp style). `PromptCacheBox` stores KV cache + prompt token array. Reuses matching prefix, only processes suffix tokens. Multimodal inputs skip caching.

### Think/Reasoning Extraction
`<think>...</think>` tags extracted into `reasoning_content` field for streaming and non-streaming responses. Buffer holds 7-8 chars for tag boundary detection.

**Note:** Not all models support thinking. Check the model's `chat_template.jinja` for `<think>` / `enable_thinking` logic. For example, `Qwen3-Coder-Next` does NOT have thinking support (confirmed by Hugging Face model card and chat template).

### Stop Sequences
`stop` field implemented end-to-end. Buffer-based approach handles stop strings spanning chunk boundaries.

### Response Format
`response_format` supports `text`, `json_object`, `json_schema`. Current implementation uses prompt injection (not guaranteed valid JSON).

## Integrations

- **OpenCode** (https://github.com/anomalyco/opencode): Uses OpenAI-compatible API as local provider
- **OpenClaw** (https://github.com/openclaw/openclaw): Uses `openai-completions` API mode. `afm mlx -m <model> --openclaw-config` generates provider config.
- `max_completion_tokens` accepted alongside `max_tokens`
- `developer` role mapped to `system`

## Model-Specific Notes

### Qwen3-Coder-Next
- Tool call format: `xmlFunction` (auto-detected via `model_type` in config.json)
- Chat template wraps tool calls in `<tool_call>`/`</tool_call>` tags
- `<tool_call>` (id 151657) and `</tool_call>` (id 151658) are added tokens in the vocabulary
- **No thinking/reasoning support** — chat template has no `<think>` logic
- Known issue: sometimes emits duplicate `<parameter=key>` tags or JSON objects instead of strings for tool parameters

### Supported Tool Call Formats
Defined in `vendor/.../ToolCallFormat.swift`: `json`, `lfm2`, `xmlFunction`, `glm4`, `gemma`, `kimiK2`, `minimaxM2`. Auto-detected from `model_type` in config.json via `ToolCallFormat.infer()`.

## Testing (Autonomous Agent Protocol)

When running tests autonomously (Claude Code, Codex, or other AI agents):

1. **Inform the user of intent first** — state model, port, mode, expected duration before proceeding
2. **End-to-end control** — start server, run tests, parse results, diagnose failures, fix, re-test autonomously
3. **End-to-end visibility** — log intermediate results, show pass/fail, surface actual errors
4. **Port 9999** — ToolCall-15 browser GUI is hardcoded to port 9999. Always use this port for tool-call testing.
5. **Full model names for batch endpoint** — `POST /v1/batch/completions` requires the full HuggingFace model ID (e.g., `mlx-community/gemma-4-31b-it-4bit`), not short aliases. The regular `/v1/chat/completions` accepts aliases but batch does not.
6. **Batch request format** — requires `custom_id` per request: `{"requests":[{"custom_id":"tc-01","body":{...}}]}`
7. **Memory budget (#115)** — the spawned AFM server holds the model weights in unified memory. On 32 GB machines, default to a model under 10 B parameters (e.g. `Meta-Llama-3.1-8B-Instruct-4bit`) for autonomous test runs. A 35 B-class 4-bit model needs ~22 GB resident, which combined with Claude Code's own footprint can OOM the host and crash the server mid-run, producing empty responses and cascading test failures. Only spawn 30 B+ models when the user explicitly opts in or runs on ≥64 GB hardware. Always kill the server in a `trap` or `finally` so it's released on test failure or interrupt.
8. **Never claim a failure is "pre-existing", "model quality", or "not a regression" without proof.** Acceptable proof is exactly one of: (a) a linked GitHub issue that names the exact failing assertion or interaction (e.g. #86 for concurrent + grammar empty responses); (b) a baseline run of the same suite against `main` (or the parent commit) showing the same failures, with the report file path captured; (c) a documented vendor / model-card limitation cited inline; (d) a `git blame` showing the assertion was already failing before this branch existed. If none apply, the failure is **unattributed** — surface it as such, run the baseline before claiming attribution, or treat it as a candidate regression and investigate. Do not invent categories like "model non-determinism" or "build-cache quirk" without reproducing them in a clean run.

### Test Suites Available

| Suite | Command | Description |
|-------|---------|-------------|
| Assertions | `./Scripts/test-assertions-multi.sh --models MODEL --tier standard` | 65+ deterministic pass/fail tests |
| ToolCall Matrix | `./Scripts/test-toolcall-matrix.sh --models MODEL --port 9999` | Tool call accuracy across parser configs |
| Promptfoo Agentic | `./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh` | 137 cases across 16 configs |
| Batch Validation | `./Scripts/feature-mlx-concurrent-batch/validate_responses.py` | Known-answer correctness at B=1,2,4,8 |
| Smart Analysis | `./Scripts/mlx-model-test.sh` | AI-judge quality scoring |

## Coding Rules

- **No magic numbers.** Define numeric constants (timeouts, buffer sizes, thresholds, retry counts, port numbers, etc.) as named constants at the top of the file or in a shared `Constants` enum. Never hard-code literal values in logic. Example: use `static let slotQueueTimeout: TimeInterval = 240` not `timeout: 240` inline.

## Architecture Notes

### Metal Buffer Lifecycle (cache materialization every 512 steps)

MLX uses lazy graph evaluation — operations build a compute graph that is only materialized when explicitly requested. In the BatchScheduler decode loop, each step's `cache.update()` does a slice assignment (`self.keys![..., idx, ...] = newKeys`) that creates a new MLXArray under copy-on-write semantics. The old array becomes garbage only if no other reference holds it, but the lazy graph retains intermediate arrays until the next materialization.

With 60 layers x 2 arrays (K+V) = 120 new Metal buffer allocations per decode step, the OS Metal allocation limit (499,000 buffers) is reached after ~4,000 steps — crashing the server with `[metal::malloc] Resource limit exceeded`.

**Fix**: `BatchScheduler.generationLoop()` calls `MLX.eval()` on all cache arrays every 512 decode steps. This materializes all cache state, collapsing the lazy graph and releasing intermediate Metal buffers. The 512-step interval balances buffer pressure against the materialization overhead (~2ms on M3 Ultra).

This is invisible to the model — the arrays hold the same values before and after. It only affects when the GPU physically executes the accumulated operations.

### MLXModelService
- `ModelContainer` uses `SerialAccessContainer` (async mutex) for thread-safe model access
- `container.perform {}` holds lock for entire generation — ensures single-sequence access
- `generateStreaming()` returns `(modelID, stream, promptTokens, toolCallStartTag, toolCallEndTag)`
- The inner `generateTask` (vendor code) runs token generation in a `Task` with synchronous `iterator.next()` loop

### MLXChatCompletionsController
- Handles both streaming (SSE) and non-streaming responses
- Streaming uses Vapor's `Response.Body.init(asyncStream:)` with `NIOAsyncWriter`
- Tool call detection state machine: `inToolCall` / `madeToolCall` / `currentToolText`
- Think extraction runs in the same streaming loop via `extractThinkTags()`
- `X-Accel-Buffering: no` header set for nginx proxy compatibility

### Vendor Code (mlx-swift-lm)
- `ToolCallProcessor`: processes chunks through format-specific parsers (inline mode for xmlFunction)
- `NaiveStreamingDetokenizer`: decodes accumulated tokens, returns suffix diff per token
- `TokenIterator.next()`: synchronous GPU computation per token
- `GenerateParameters`: sampling config passed to the model
