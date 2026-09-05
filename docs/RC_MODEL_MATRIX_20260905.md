# Release-candidate model qualification matrix

Requested September 4: package a candidate first, then run the full suite for
the three AFM-generated models, Qwen 27B, and DeepSeek DS4, with speculation
requested both off and on. Run sequentially to avoid GPU contention. Use
Codex-only judging for comprehensive reports and preserve all raw results.

| Checkpoint | Exact local path | Off | On |
| --- | --- | --- | --- |
| AFM DeepSeek 0731 | `/Volumes/edata2/models/afm/DeepSeek-V4-Flash-0731-AFM-MLX-native-v11` | Native target | `--mtp`, requires qualification of AFMKit #86 |
| AFM GLM 5.3 Flash | `/Volumes/edata2/models/afm/GLM-5.3-Flash-AFM-MLX-4bit` | Native target | Request `--mtp`; checkpoint has zero MTP tensors, record unavailable/fallback, not speculative coverage; AFMKit #94 tracks conversion support |
| AFM Qwen Next | `/Volumes/edata2/models/afm/Qwen3.8-Flash-Next-AFM-MLX-4bit` | Native target | `--mtp`; embedded weights present |
| Qwen 3.8 27B 4-bit | `/Volumes/edata2/models/huggingface-cache/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff` | Native target | `--mtp` with cached matching 4-bit head, revision `b643c01b6d3b094e325edb6ebd832e16c486c575` |
| DeepSeek 0731 DS4 | `/Volumes/edata/models/ds4/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf` | DwarfStar target | DwarfStar `--dspark-support /Volumes/edata/models/ds4/DeepSeek-V4-Flash-DSpark-support-0731.gguf` |

## Required evidence

- Candidate source commits, exact AFMKit dependency, packaged binary hash,
  displayed version, installed resource checks, Foundation/MLX installation smoke.
- Full assertion suite, comprehensive prompts with Codex judge, and applicable
  agentic/structured-output coverage. Keep native conformance, model behavior,
  forced-parser experiments, and unavailable capabilities separate.
- Record requested versus actually active speculation. Temperature, penalties,
  batching or unsupported assets may force target fallback; such runs do not
  establish speculative performance or correctness.
- Performance: prefill and decode, first four contexts, raw visible/reasoning
  output, prompt budget and completion count. Same checkpoint across on/off.
- Prefix cache, concurrency/batching, streaming, cancellation, stop handling,
  tool calling and structured output must retain their actual test outcomes.
- File confirmed engine/harness defects as issues, fix through PRs, rebuild
  changed code, and rerun affected coverage. Do not erase failed-run evidence.
- The final report must distinguish completed tests from missing MTP coverage
  and unresolved release blockers. No blanket release-ready claim from smokes.

Default requested thinking mode is `--no-think`. The comprehensive prompt file
uses a 20,000-token default, with deliberate per-test limits retained. Do not
substitute unrelated checkpoints to improve numbers or hide missing assets.

Current native greedy-speculation eligibility requires temperature zero, no
repetition/presence penalties, top-k/min-p disabled, normal EOS handling, and no
tools, response format, logprobs, stop sequences, or media. The full suite must
still exercise these feature combinations with MTP requested, but their target
fallback is intentional compatibility coverage rather than a speculative speed
measurement. Preserve separate focused greedy runs for actual on/off throughput.

## Pre-package progress (September 5)

These are qualification probes, not completed full-suite RC results:

- AFMKit main at `03fd2b44` passed full release qualification: all 16 public
  API gates, root tests, and all 16 downstream consumers. It is published as
  prerelease `v0.1.18-rc.1` and pinned in the consumer lockfile.
- DeepSeek candidate `54ed9ea4` passed nine focused architecture/rollback
  tests. Its paired Release binary measured median counting throughput of
  30.12 target / 51.04 speculative tok/s and prose throughput of
  29.78 / 27.32 tok/s. Prose was coherent but not byte-identical. API timing
  in the speculative path includes prefill, so these are not pure kernel rates.
- A bounded two-client, repeated two-turn isolated-code probe passed 24/24
  checks. Target mode recorded B=2 execution and radix hits; speculation
  remained serialized with cache misses. Do not interpret this as speculative
  batching/cache support. AFMKit issue #98 records that limitation; issue #97
  tracks the rollback fix on PR #86, now merged in `03fd2b44`.
- Promptfoo DS4 support-path and positive load-timeout argument regression
  tests passed, including a checkpoint path containing spaces. Real DS4
  inference qualification is still pending. Changes are on this PR (#252),
  associated with maclocal-api issue #253.

Raw pre-package evidence is preserved under
`/Volumes/edata2/afm-benchmarks/rc-20260904/`.

The earlier full validator built all sixteen downstream consumers but failed
its stale terminal expectation of fifteen. AFMKit PR #100 fixes that count by
using the generated inventory; regression tests accept sixteen and reject an
incomplete fifteen. The local publisher then completed full qualification
of `03fd2b44` and published `v0.1.18-rc.1` only after that success.

Consumer release-tooling regressions, explicit MTP-head runner arguments,
categorized Promptfoo summaries, and external judge-work paths pass their
focused checks. Codex execution preflight succeeded. The immutable consumer
Release build and Swift tests are running; packaged model runs have not begun.
