# DFlash 2 Research Trace

Date: 2026-08-19

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
```

## Exact Performance Methodology

The live methodology is normative in `TEST_MATRIX.md`. Raw run records must
include command, git revision, target/draft Hub revisions, config hash, hardware,
OS/power/thermal state, warmups, prompt bytes, seed/sampling, cache state,
concurrency, token outputs, acceptance/cycle counters, timings, and memory.
Compare AR and DFlash 2 only with the same target checkpoint and request matrix.
