# Beta release-candidate integration audit

Scope: branch-tip activity and PR updates from 2026-08-26 through 2026-09-04,
including local-only branches. Git patch equivalence was checked against fetched
`origin/main`; old branch names alone are not evidence of missing implementation.
This document tracks a candidate, not a completed release qualification.

## Inventory and disposition

| Repository / PR | Work | Disposition |
| --- | --- | --- |
| AFMKit #85 | GLM cached-source conversion, bounded checksum memory, FP32 router preservation and repair | Merged after focused review; 29 converter tests, full conversion and live load passed |
| AFMKit #77 | Qwen Next measured parity documentation | Merged; preserved newer causal-prefill medians separately from historical peak-of-three results |
| AFMKit #86 | DeepSeek embedded speculation and graph optimizations | New non-draft PR; review and qualify MTP on/off, cache, concurrency and exact-checkpoint performance |
| AFMKit #87 | MLX Swift LM independence feasibility study | Merged after review; not authorization to restructure dependencies |
| AFMKit #88 | Older long-generation lifetime experiment | New non-draft PR; reconcile newer host-token scheduling and measure cache-evaluation/allocator tradeoffs before inclusion |
| AFMKit #89 | OpenAI request compatibility plus older prefill error handling | New non-draft PR; separate missing API behavior from potentially superseded execution changes |
| AFMKit #90 | Older recurrent replay/error-handler branch | New non-draft PR; high overlap with merged scheduler and cache changes; reconcile before inclusion |
| AFMKit #91 | Earlier GLM compilation experiment | New non-draft PR; compare with merged #81 and avoid restoring superseded graph code |
| maclocal-api #251 | Messages assistant continuation disables thinking | Merged after semantic review and 15 passing request-mapping tests, including ordinary-thinking preservation |
| AFMKit #74 | Experimental MLX GGUF loader | Existing draft; separate prototype, not presumed RC-ready |
| maclocal-api #89–91, #196 | TurboQuant / DFlash | Previously deferred by user; leave open and untouched |
| AFMKit issue #76 | Per-slot continuous batching | Previously assigned to next release; do not silently include |

Recent AFMKit #83, #82, #81, #80, #78, #75 and their consumer dependency bumps
are already merged. GLM converter fixes were extracted onto current main rather
than merging the entire `perf/glm53-graph-capture` branch, which also inherits
DeepSeek work. The uncommitted GLM attention-preparation experiment is not part
of #85 and is not qualified for this RC.

The old `fix/glm53-kda-throughput` history includes an explicit optimization
revert; do not restore the reverted change merely because its commit is unique.
Old dependency-bump branches, the closed Qwen sidecar consumer PR #244, and the
superseded consumer-owned Qwen patch PR #214 are not new merge work.

`fix/v0.9.18.1-foundationmodels-release` contains an older linker-only guard and
version changes. Current main already invokes the stronger
`check-foundation-models-build.sh` capability gate. Do not revert versions or
replace that gate with the older branch wholesale.

The original maclocal-api workspace contains uncommitted build-script changes,
`Scripts/codex-spark`, and `voxel-art/`. They are preserved and not implicitly
included in this release. Other worktrees and uncommitted performance probes
must likewise remain separate until their ownership and validation are clear.

## Integration order

Progress: DeepSeek #86 has been integrated with current main at `4b601745` and
53 targeted Release tests passed (speculation policy, architecture numerics,
prefix replay). Full live MTP equivalence and performance remain pending.
The API-only portion of #89 is isolated on `fix/rc-request-decoding`; its custom
decoder preserves newer fields such as `ignore_eos`, with all-field round-trip
regression coverage. It does not import the older asynchronous prefill changes.

1. Land isolated correctness/converter fixes and reconcile documentation.
2. Review API compatibility separately from old runtime changes. For overlapping
   branches, preserve only demonstrably missing behavior in a current-main PR;
   link a superseding PR instead of merging obsolete implementations.
3. Qualify DeepSeek and any retained graph/lifetime changes with both correctness
   and same-checkpoint performance tests. Consult the user before accepting a
   performance tradeoff. Do not make an unconditional merge-all claim.
4. Tag a qualified AFMKit version, then update maclocal-api's single exact package
   dependency and lockfile. Provider sources stay exclusively in AFMKit.
5. Build the beta in an isolated consumer worktree through
   `Scripts/swiftpm-reliable.sh`; test the immutable dependency graph, not a
   development override. Do not depend on GitHub Actions.
6. Stage only on the beta/staging distribution channel after package validation;
   no stable-channel promotion is implied.

## Release-candidate gates

- Build provenance: exact source commits, AFMKit pin, toolchain, executable hash,
  canonical beta/nightly identifier, packaged MLX resource verification.
- Installation: Foundation Models compiled capability and live smoke request;
  MLX model load and live request from the installed artifact.
- Unit/API suites: request validation, stops, streaming, tool calls, structured
  output, cancellation, model-switch behavior, prefix/radix cache and concurrency.
- Model qualification: GLM 5.3 Flash, Qwen Next, Qwen 3.8 27B and DeepSeek 0731,
  sequential GPU workloads using pinned, documented local checkpoint paths.
- Performance: separate prefill and decode; first four context sizes; same
  checkpoint, prompts, token budget, speculation, cache policy and concurrency.
  Test MTP on/off only where weights and runtime support it. Preserve both raw
  output and timing. No environment-only tuning may be hidden in a default claim.
- Quality: inspect actual outputs; use Codex-only judging for comprehensive
  evaluations as previously requested. Keep protocol conformance, model/agent
  behavior and forced-parser experiments separate. Unexplained failures remain
  unattributed until reproduced or evidenced.
- Preserve the earlier 0.9.16/0.9.17 baseline artifacts; do not recreate them
  unnecessarily. Qwen Next uses its supported, separately recorded baseline.
- Store bulky reports outside Git on persistent storage. Final release-report
  bundles belong on the matching release as optional assets, not in the package.

## GLM evidence carried forward

`scouzi1966/GLM-5.3-Flash-AFM-MLX-4bit` was uploaded successfully at HF revision
`6739b0ce6b97d4d854fd89806413d37ee943c6d1`; all 197 weight hashes match the local
conversion manifest. The official source revision is
`04c4e9e95c5da8862dced7e5056455116f83a7e0`.

Initial repaired-checkpoint inference passed arithmetic/logic checks and produced
roughly 34 tok/s for the first three synthetic context sizes, 27.4 tok/s for the
largest. The paired oQ4e reference run stopped on reasoning-only output at a
256-token budget, so this is **not a completed quality/performance parity result**.
The AFM conversion excludes MTP; do not qualify it with `--mtp`.
