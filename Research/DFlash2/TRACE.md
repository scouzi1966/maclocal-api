# DFlash 2 Research Trace

Date: 2026-08-19, takeover audit updated 2026-08-20

## Workspace Verification

```text
pwd
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git branch --show-current
```

Finding: worktree is
`/Volumes/edata/dev/git/CODEX/vesta-mac/Worktrees/maclocal-api-dflash2`, clean on
`codex/dflash2-afmkit-20260819`; `HEAD` and `origin/main` were both
`224125f684881017b2c79f7c4cd457cf4e701061` before research edits.

## Primary Source Commands

```text
curl -L https://inco.ai/blog/dflash2/ | textutil -convert txt -stdin -stdout
curl -L https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/config.json
curl -L https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2/raw/main/config.json
curl -L https://huggingface.co/api/models/<repo>
curl -L -I https://huggingface.co/<repo>/resolve/main/model.safetensors
curl -L -r 0-1048575 https://huggingface.co/<repo>/resolve/main/model.safetensors
curl -L 'https://export.arxiv.org/api/query?id_list=2602.06036'
curl -L https://arxiv.org/html/2602.06036 | textutil -convert txt -stdin -stdout
curl -L https://api.github.com/repos/z-lab/omlx-fork/releases/tags/0.6.2-dflash2
curl -L https://api.github.com/repos/z-lab/omlx-fork/git/ref/tags/0.6.2-dflash2
```

Findings:

- Article publication date is 2026-08-18.
- DFlash 2 keeps one-pass parallel drafting.
- Selector: top 16, rank 256, parallel adjacent-pair scoring, sequential walk
  only over already computed scores, lossless rejection sampling.
- Convolution: two-tap dynamic depthwise, block-local/stateless, before and
  after attention/MLP, first position reads last verified token.
- Qwen config block size 8; Muse block size 16; both five draft layers.
- Qwen weight payload 3,848,817,896 bytes; Muse 5,544,328,424 bytes.
- Safetensors headers were decoded from range responses without downloading
  model payloads. Both have 81 tensors and expected selector/conv weights.
- Original DFlash paper inspected at arXiv v2 (2026-05-28).
- oMLX release tag resolves to commit
  `46225aebee34967d4f4bbb669bf02fc4e2de696a`.

## Reference Runtime Inspection

```text
curl -L 'https://api.github.com/repos/z-lab/omlx-fork/git/trees/<commit>?recursive=1'
curl -L https://raw.githubusercontent.com/z-lab/omlx-fork/<commit>/pyproject.toml
curl -L https://raw.githubusercontent.com/z-lab/omlx-fork/<commit>/omlx/engine/dflash.py
curl -L https://api.github.com/repos/z-lab/omlx-fork/commits/8cf8e75b591f
curl -L https://api.github.com/repos/z-lab/omlx-fork/commits/1fca1d105445
git ls-remote https://github.com/z-lab/dflash-mlx.git <pin>
git ls-remote https://github.com/jianc99/dflash-mlx.git <intermediate-pin>
```

Findings:

- oMLX adds `dflash_block_size` and forwards temperature/top-p/top-k plus the
  runtime block setting to DFlash.
- Its UI states DFlash/DFlash2 is single-stream.
- Long-context/unsupported flows use a separate AR batched/VLM fallback.
- Its DFlash prefix cache is separate because snapshots include draft model
  state and target hidden chunks.
- Final dependency pin is
  `z-lab/dflash-mlx@415cc48d83846cfcd0d5b9da3c83e4f1478acda6`.

Failure: the final and intermediate DFlash MLX dependency repositories returned
GitHub 404 / `Repository not found`.

The official release artifact resolved that blocker:

```text
curl -L <release DMG URL> -o .build/dflash2-reference/oMLX-0.6.2-dflash2.dmg
shasum -a 256 .build/dflash2-reference/oMLX-0.6.2-dflash2.dmg
hdiutil attach -readonly .build/dflash2-reference/oMLX-0.6.2-dflash2.dmg
rg -n 'candidate_selector|base_kernel|accepted|rollback' \
  /Volumes/oMLX/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages/dflash_mlx
```

Finding: SHA-256 matched
`94f56e14bfa8188d47e187f571bc61244a65010fb23d3601a28a2e22d5e5bd21`.
The mounted signed app contains `dflash_mlx 0.1.10+omlx.5`, including complete
Apache-2.0 Python source. Inspection confirmed:

- stateless grouped dynamic causal convolution with base shape
  `[2, kernel, hidden]` and `hidden -> 2*kernel*groups` projection;
- selector top-k, hidden/predecessor/successor rank embeddings, and path walk;
- the staged verifier token is included in each target block but excluded from
  the proposal count;
- longest target-matching prefix is committed, then the target verifier token
  at the first mismatch becomes the next staged output;
- the optimized reference maintains draft context caches and complete prefix
  snapshots, and implements target-distribution-preserving rejection sampling.

## Current Repository Inspection

```text
rg --files
rg -n -i 'dflash|dspark|speculat|draft_model|mtp' Sources Tests Scripts
nl -ba Sources/AFMKitCore/AFMCoreTypes.swift
nl -ba Sources/AFMKit/AFMEngine.swift
nl -ba Sources/AFMKitMLX/AFMMLXRuntime.swift
nl -ba Sources/AFMKitMLX/AFMMLXRuntimeAdapter.swift
nl -ba Sources/AFMKitMLX/AFMMLXSpeculativeDecoding.swift
nl -ba Sources/AFMKitMLX/AFMMLXSpeculativeRuntimeResourceResolver.swift
nl -ba Sources/AFMKitMLX/Models/MLXModelService.swift
nl -ba Sources/AFMKitMLX/Models/BatchScheduler.swift
nl -ba Sources/AFMKitMLX/Models/StatsAggregator.swift
nl -ba Sources/AFMOpenAICompat/OpenAIRequest.swift
nl -ba Sources/AFMCLI/main.swift
nl -ba Scripts/apply-mlx-patches.sh
git submodule status
```

Findings:

- No DFlash runtime exists in the checkout.
- AFMKitCore already has neutral speculative capability and metadata channels.
- AFMKitMLX has MTP/EAGLE3 policy and runtime enums, while the main service also
  has DSpARK, MTP, and EAGLE3 fast paths.
- Speculative fast paths are serial; concurrent scheduler uses AR.
- Current MTP/EAGLE3 eligibility is greedy/text/no modifiers/no stops; reasoning
  is parsed downstream but policy currently excludes reasoning output in the
  AFMMLX abstraction.
- Hub model download progress/stages and an auxiliary MTP resolver are reusable.
- OpenAI request schema has no speculative extension.
- Metrics lack draft/accepted/cycle/phase counters.
- Vendor submodules are uninitialized in this worktree. `Package.swift` falls
  back to pinned pre-patched `scouzi1966/mlx-swift-lm` revision
  `6bab4f5ac55e81903dd74090244c25feb3233338` when absent.

## Rejected Approaches

- Repository-name matching: conflicts with renamed/imported models and the
  released config types.
- Treating DFlash 2 as MTP/EAGLE3/DSpARK: incompatible algorithm/cache/runtime.
- Adding DFlash tensor details to AFMKitCore: violates dependency and ownership
  boundaries.
- Silent fallback after explicit startup enable or after token emission: hides
  configuration errors and risks divergent duplicate output.
- Enabling prefix restore or batching before full speculative-state support:
  current cache/scheduler contracts are insufficient.
- Repeating upstream speed claims as local results: no same-model local evidence.

## Implementation Commands and Findings

```text
git submodule update --init vendor/mlx-swift-lm vendor/ds4
./Scripts/apply-mlx-patches.sh
./Scripts/apply-mlx-deepseek-v4-kernels.sh
./Scripts/apply-mlx-official-fp8-loader.sh
./Scripts/check-dflash2-vendor-patch.sh
swift build --target AFMKitMLX
swift build --target AFMCLI
swift test --filter AFMMLXDFlash2ConfigurationTests
swift test --filter AFMMLXSpeculativeDecodingTests
```

Build failures encountered and resolved:

1. The worktree's declared vendor submodules were initially uninitialized.
   They were initialized at their pinned commits; no upstream worktree or
   repository was modified.
2. The first AFMKitMLX build failed because the existing DeepSeek V4 source
   overlay requires the repository's declared MLXFast kernel patch. Running the
   supported `apply-mlx-deepseek-v4-kernels.sh` and official FP8 loader scripts
   restored the expected pinned build state.
3. The first focused test compile found two missing `try` markers in new test
   assertions; corrected before the test checkpoint.
4. Weight-tree inspection found dotted selector `@ModuleInfo` keys were not an
   established MLX Swift pattern. Replaced with a nested candidate-selector
   module and added an exact parameter-key test.

Implemented runtime findings:

- DFlash 2 needs its own MLX primitive. Existing DFlash/DSpark concepts are
  reusable only for orchestration, fallback, streaming, cancellation, model
  download, and neutral telemetry.
- Greedy verification is target-lossless in the tiny deterministic fixture.
- The correctness-first loop restores and replays committed target state. It
  does not yet implement the reference draft KV cache, prefix snapshots,
  rejection sampling, or batched row-aligned verification.
- Existing 22-case speculative policy suite remains green.
- No heavy Qwen/Muse inference was started. Compile/unit work is ready for the
  requested coordination point.

Pushed checkpoints:

```text
00c6191 Document DFlash 2 integration plan
bb44f0b Add DFlash 2 MLX vendor primitive
be00de3 Fix DFlash 2 selector weight hierarchy
d15cd5d Integrate opt-in DFlash 2 runtime
a07a795 Use monotonic DFlash 2 phase timings
75f9bb0 Expose DFlash 2 request policy to tests
158e3f5 Test DFlash 2 contracts and losslessness
5bdd5e8 Record DFlash 2 implementation evidence
6b28dc8 Complete DFlash2 runtime compatibility checks
```

## 2026-08-20 Takeover Audit

The eight implementation commits through `5bdd5e8` did not fully satisfy the
released-checkpoint or AFMKit execution contracts. The corrective checkpoint
`6b28dc8` addressed the following production gaps:

- released Qwen and Muse configs place `rope_theta` under `rope_parameters`;
  the original parser required a top-level value and rejected both drafters;
- compatibility checks now cover architecture, target dimensions, tokenizer
  IDs, context/RoPE, target layers/taps, sliding-window metadata, and the exact
  81-tensor shape contract before MLX reads the weight payload;
- draft attention now follows released `layer_types` and `sliding_window`
  metadata instead of using unrestricted causal attention;
- AFMKit HTTP execution now forwards request speculative controls through the
  adapter and provider rather than dropping preferred/required/off semantics;
- request-level `off` overrides a loaded required-by-default runtime, while
  request-level required DFlash2 cannot be consumed by MTP, EAGLE3, or the
  concurrent AR scheduler;
- DFlash2 is selected ahead of other speculative runtimes when explicitly
  requested, and request-time drafter switching is rejected.

The follow-up working checkpoint adds versioned
`afm.speculative_decoding.v1` metadata to AFMKit responses and stream events,
preserves it through the HTTP adapter, reports explicit pre-emission fallback
reasons, and exports emitted-token totals alongside draft/accept/cycle/timing
metrics. It also makes runtime draft limits fail closed instead of silently
clamping values above the loaded checkpoint limit.

Official config snapshots were retained under
`.build/qualification/dflash2-official-configs-20260820` with these SHA-256
values:

```text
873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980  qwen-draft-config.json
14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85  qwen-target-config.json
cb684d6f688a22619a63ea1debe7d30c139c195bf3141fd86a763763ab34b5d9  muse-draft-config.json
c7f48468db2ef9c3de4cb912be24ecc9fbed36d83f3b8386a0b224ee7ba876ca  muse-target-config.json
```

### Vendor materialization proof

A clean local clone was created at
`.build/qualification/dflash2-vendor-clean-6b28dc8`, reset to pinned vendor
commit `6bab4f5ac55e81903dd74090244c25feb3233338`, then processed with the normal
`Scripts/apply-mlx-patches.sh` workflow. Application and `--check` both passed,
and `Scripts/patches/DFlash2.swift` was byte-identical to the clean clone's
materialized `Libraries/MLXLMCommon/DFlash2.swift`.

The clean clone and working vendor submodule produced the same status: modified
`NemotronH.swift`, `Qwen3_5MoE.swift`, `ToolCallFormat.swift`, and
`VLMModelFactory.swift`; new `DFlash2.swift`, `ATEMToolCallParser.swift`,
`ToolCallFormat.swift.original`, `MuseGlimmer.swift`,
`VLMModelFactory.swift.original`, and `Package.swift.original`. The dirty
`vendor/mlx-swift-lm` state is therefore expected materialized patch output,
not an unrecorded dependency edit. No dependency repository was pushed.

After the static gate passed, both official draft payloads were downloaded to
`/Volumes/edata2/models/huggingface-cache` and tested serially under the shared
`.agent-live-test-lock`. Each run acquired the lock with atomic `mkdir`, wrote
the owner record, installed a cleanup trap, and released the lock before the
next model started. No live suite overlapped another agent's Release run.

Retained artifacts are under
`.build/qualification/dflash2-final-20260820`:

- Qwen target revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`
  with draft revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`;
- Muse target revision `3e7677d7a40d348a3daba263a2b1c0aa41910710`
  with draft revision `8336acb8dc9b8bf9c25f12d7785ee6df26703119`;
- both started with `--dflash2-block 5 --dflash2-required`;
- request-level `off` and required non-streaming output matched for the same
  greedy prompt on both pairs, comparing visible plus reasoning channels;
- required streaming completed with usage and `[DONE]` for both pairs;
- Qwen totals after prewarm plus two required requests were 11 drafted, 5
  accepted, 8 emitted, and 3 cycles; Muse totals were 21 drafted, 12 accepted,
  20 emitted, and 7 cycles.

The CLI defaults to a four-token prewarm, and that work intentionally appears
in process-lifetime Prometheus counters. A separate Qwen debug probe captured
baseline prewarm totals of 3 drafted, 3 accepted, 4 emitted, and 1 cycle; the
single subsequent request added 3 drafted, 1 accepted, 2 emitted, and 1 cycle.
This ruled out telemetry double counting.

At that checkpoint these were bounded smoke results, not the full matrix. Long
generation, reasoning-level comparisons, tools, cancellation, memory limits,
statistical sampling, prefix snapshots, batch verification, and performance
were still unqualified on the released checkpoints.

## 2026-08-20 to 2026-08-21 Independent Review Remediation

The independent review of PR #196 identified ten release blockers. Checkpoints
`3b9b7c0`, `6135f0c`, `cf7c0bc`, and `24c6ce9` close them as follows:

1. DFlash2 eligibility now treats omitted temperature as the normal sampling
   default (`0.6`), accepts only explicit greedy temperature, and gates top-p,
   top-k, min-p, repetition, and presence modifiers. Preferred uses AR;
   required fails before emission.
2. DFlash2 builds one EOS set from configured, tokenizer, and extra token IDs.
   The shared generator checks EOS before callback/append, so secondary
   terminators are neither exposed nor counted in streaming or nonstreaming.
3. Request mode `off` and explicit DFlash2 preferred/required govern DFlash2,
   DSpARK, MTP, and EAGLE3 selection rather than only the DFlash fast-path flag.
4. Process-global DFlash2 state was removed. Each `MLXModelService` owns runtime
   state keyed to its service instance, model ID, drafter, loaded container, and
   validated target metadata.
5. Vendor callbacks and AFMKit provider/service paths propagate
   `CancellationError`. Partial output is not converted to a successful stop,
   usage completion, or speculative metric commit.
6. Nonstreaming DFlash2 seeds the response parser from the prompt-opened
   reasoning boundary, keeping private reasoning in `reasoning_content`.
7. DFlash2 now publishes prompt/completion token counts and prompt/decode
   timings through the normal response and process metrics paths.
8. Both batch controllers force preferred work to deterministic AR and reject
   required DFlash2 as a pre-emission batch conflict; Muse is not run as serial
   DFlash inside the batch endpoint.
9. Public `AFMMLXRuntimeAdapter` construction uses the same compatibility
   preflight as production, covering target metadata bytes, architecture,
   context/RoPE/token IDs, feature taps, block limit, and safetensor shapes.
10. Service-level regressions cover nil temperature, independent sampling
    controls and penalties (including nil companion fields), secondary EOS,
    partial cancellation, prompt-opened reasoning, independent service scopes,
    prefix/concurrency/batch reasons, forced-AR batch behavior, and base plus
    speculative telemetry.

The patch definition in `Scripts/patches/DFlash2.swift` remains the source of
truth. `Scripts/check-dflash2-vendor-patch.sh` passes, and the dirty
`vendor/mlx-swift-lm` submodule remains expected materialized output from the
repository patch workflow described above. No dependency repository was
pushed.

Focused validation after the review fixes:

```text
Debug AFMMLXDFlash2ConfigurationTests: 23/23 pass
Debug BatchCompletionsControllerTests: 17/17 pass
Debug TokenizeAndOpenAPITests: 11/11 pass
Release DFlash2 + batch + stream cancellation selection: 58/58 pass
Scripts/check-dflash2-vendor-patch.sh: pass
git diff --check: pass
```

Two complete Release attempts reached 437/437 passing Swift Testing cases, but
the XCTest process terminated with signal 11 in
`MLXFoundationLanguageModelTests.testGenerationConfigDisablesThinkingByDefaultForReasoningModels`.
Repeated crash reports locate the fault in
`AFMFoundationModelsRequestAdapter.generationConfig` under macOS 27.0 and
Xcode 27 beta 3. Isolated empty- and nonempty-metadata tests reproduce the
same unrelated compiler/framework code-generation fault. Experimental guards
were removed; no unrelated workaround is included in this branch. Logs are in
`.build/qualification/dflash2-review-fixes-20260820`.

Production-path qualification used Release `afm` binaries built from `cf7c0bc`
for the Qwen sampling/reasoning/cancellation run and `24c6ce9` for the Qwen
prefix and Muse batch runs. It used the same released Qwen/Muse snapshots listed
above, atomic shared lock acquisition, an owner record, and trap-based cleanup.
No run overlapped another agent's live or Release suite. Evidence is retained
under `.build/qualification/dflash2-review-fixes-20260820/live`:

- `qwen-final/summary.json`: preferred omitted-temperature and penalty requests
  returned 200 through AR; required variants returned 400; explicit greedy
  required generation returned content and nonzero normal usage/timings;
  reasoning was split into nonempty `reasoning_content` and visible `content`
  without leaked think tags;
- the Qwen SSE cancellation probe cancelled only after a nonempty partial token,
  returned `finish_reason=cancelled` without stop or usage, held speculative
  cycles at 22 after cancellation, and passed a subsequent health check;
- `qwen-prefix/summary.json`: preferred prefix-cache conflict used AR, required
  returned 400 with `prefix_cache`, and speculative cycles stayed zero;
- `muse-batch/summary.json`: two preferred batch rows completed through AR, the
  required row emitted `batch.error` with stable reason `batch`, and speculative
  cycles stayed `0 -> 0`.

The focused Release artifact is
`.build/qualification/dflash2-review-fixes-20260820/release/focused-final.log`.
The two broad attempt logs are `broad-release.log` and
`broad-release-rerun.log`. No comprehensive, Promptfoo, UI, performance, or AI
judge suite was run; those are not needed to claim the bounded production paths
above and remain outside this qualification.

## Exact Performance Methodology

The live methodology is normative in `TEST_MATRIX.md`. Raw run records must
include command, git revision, target/draft Hub revisions, config hash, hardware,
OS/power/thermal state, warmups, prompt bytes, seed/sampling, cache state,
concurrency, token outputs, acceptance/cycle counters, timings, and memory.
Compare AR and DFlash 2 only with the same target checkpoint and request matrix.
