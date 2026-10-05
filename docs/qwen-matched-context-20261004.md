# Matched Qwen Next qualification — 2026-10-04

This is diagnostic evidence, not release approval. The native community
checkpoint's historical performance gap remains a separate open gate.

## Identities and method

- Checkpoint: `/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001`.
- Config SHA256: `fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59`.
- Control AFM SHA256: `e05ab5c66696fd7d56b444a77e9f0bf8c7cd19eded6c2a0878c724d018100e23`.
- Experimental AFM SHA256: `415f1a31a8dfb6727488bb1a3fa0704a97f16390f53b0154983a0716bde846e0`.
- Reference: mlx-serve 26.10.1, SHA256 `46d017b5f6e49890e654dbefd2b750eda44b627a20f4ec7a01015a4d0d8ea1e0`.
- MTP depth 3, temperature 0, top-p 1, reasoning off, prefix cache off,
  one active request, AFM capacity 2, 8192 prefill step, throughput-v2 CLI profile.
- Three trials per context, deterministic prompt markers, 128 generated tokens.
  No context-shape warmup; the first trial is retained. Tiny arithmetic preflight
  precedes measurement. No tuning environment variables, concurrent inference,
  or overlapping builds.
- Prefill is prompt tokens / client TTFT for both engines. These are means,
  not selected CSV peaks. Files retain every raw trial and generated response.

## First matched pass

| Context | Control AFM prefill/decode | Experimental AFM prefill/decode | Reference prefill/decode |
|---|---:|---:|---:|
| 0.5K | 915.12 / 112.82 | 909.80 / 112.67 | 1022.18 / 100.81 |
| 1K | 1085.07 / 107.26 | 1109.30 / 109.87 | 1187.96 / 107.58 |
| 2K | 1268.24 / 96.62 | 1277.14 / 99.09 | 1287.42 / 99.29 |
| 4K | 1302.85 / 98.74 | 1325.27 / 98.55 | 1352.01 / 98.90 |

All eight experimental metrics pass the 3% regression threshold against the
control. All twelve AFM output texts are exactly unchanged. This does not
establish broader semantic quality parity with the reference.
Cross-engine checks confirm all twelve prompt hashes and prompt-token counts
match. Only two generated texts are byte-identical across engines; semantic
quality must therefore be assessed independently rather than inferred from
the timing match.

The earlier roughly 30% 2K decode shortfall does not reproduce with this exact
prompt set. Earlier full runs used different cold markers; do not attribute
the difference solely to the new kernel. Even the unchanged control reaches
96.62 tok/s here. Prompt-dependent MTP acceptance remains relevant.

The experiment replaces HC mix's row-count Metal template specialization with
a runtime scalar; arithmetic is unchanged. Eight focused provider tests pass
both with and without the throughput profile. Reverse-order repetition and
native-community testing are recorded below. Small gains may be noise.

## Reverse-order repeat and native checkpoint

The reverse-order overlay repeat passed all eight 3% performance gates and
preserved all twelve output texts. Decode changes versus control were
-0.17%, -0.01%, +5.22%, +0.13% at 0.5/1/2/4K respectively. Other than 2K,
the small initial gains did not repeat consistently.

Native checkpoint:
`/Volumes/edata/models/vesta-test-cache/mlx-community/Qwen3.8-Flash-Next-4bit-mtp`,
revision `a53d7aa384247a095068485ac75f4383cd25e3fd`.
Same MTP/profile/sampling/context settings and binaries as above.

| Context | Control AFM prefill/decode | Experimental AFM prefill/decode | Same-checkpoint reference |
|---|---:|---:|---|
| 0.5K | 888.62 / 84.32 | 960.96 / 87.59 | Not available |
| 1K | 1111.11 / 86.01 | 1111.92 / 85.89 | Not available |
| 2K | 1288.62 / 79.46 | 1286.68 / 79.35 | Not available |
| 4K | 1325.21 / 86.31 | 1337.79 / 84.99 | Not available |

All eight native metrics pass the 3% gate; all twelve texts are identical.
The reference did not load this native checkpoint in the tested configuration.
Do not substitute its overlay numbers as native engine parity. This change
does not recover the native checkpoint's historical decode high-water mark.
API qualification completed on the experimental binary: **116/116 passed,
two capability skips**. Coverage includes prefix cache, concurrent/batch
dispatch, grammar, tool parsing, stop/streaming, and sampling interactions.
Evidence: `dynamic-rows-native-api/assertions/`. This is not a rerun of source
XCTest. The comprehensive suite with `codex-glm` as judge is running from the
frozen `candidate-runtime-hc-20261004` binary. Broader release qualification
and historical-performance recovery are still pending.

## Rejected submission-cadence screen

Same native checkpoint, frozen experimental binary, matched prompts, three
trials, MTP depth 3. Only the diagnostic scheduling override differs; ordinary
qualification uses the CLI profile without overrides. These are decode means:

| Verifier cadence | 0.5K | 1K | 2K | 4K |
|---|---:|---:|---:|---:|
| Existing 2 | 87.59 | 85.89 | 79.35 | 84.99 |
| 0 | 81.28 | 78.41 | 69.17 | 75.21 |
| 1 | 89.67 | 86.00 | 79.48 | 85.84 |
| 4 | 88.91 | 85.06 | 76.34 | 84.89 |
| 8 | 88.16 | 84.37 | 77.80 | 84.10 |

All outputs are identical to the existing cadence. No setting is promoted:
cadence 0 is materially worse, 4 fails the 3% gate at 2K, 8 is not a win,
and 1's small gains do not recover the historical deficit or justify another
release setting. Same-checkpoint reference remains unavailable for native
community; the overlay reference is not substituted here.

Separate host diagnostics show the current verifier's build/submission phase
absorbs work that the retained binary waits for in its decision phase. Summing
both is necessary: calling the difference a 6x graph-build regression would
be false. For measured 128-token requests, current build+decision is about
28.9–31.8 ms/cycle versus retained 26.5–29.1 ms/cycle. Draft acceptance varies
by prompt/output trajectory. These profiled runs are attribution evidence,
not clean speed gates. The regression checker now explicitly rejects them.

## Rejected HC-only projection routing

A temporary `AFM_QWEN_VERIFY_HC_QMM=0` screen bypassed the custom batched
projection only for the HyperConnection role; other projections retained the
profile. Seven projection-policy tests passed before benchmarking. Same-binary
three-trial decode means (on/off) were 89.14/80.24, 85.72/86.61, 79.18/77.19,
86.42/87.87 tok/s at 0.5/1/2/4K. Outputs differed. There is no consistent
benefit and the 0.5K loss is 9.98%, so the switch was removed rather than added
to the public tuning surface. Evidence: `native-hc-qmm-control/` and
`native-hc-qmm-off/`. No model weights were changed. The comprehensive judge
continues to use the earlier frozen `415f1a...` binary, not this experiment.

## Evidence root

## Native reference compatibility check

The latest reference release remains mlx-serve 26.10.1. A fresh direct native
community load fails with `FileNotFound` after loading the weights. Its loader
requires `ngram_table.bin`; the native checkpoint stores quantized embedding
shards instead. However, a sidecar repack alone is **not** a valid remedy:
`src/model.zig` also explicitly assumes the reference converter has folded
every zero-centered norm's `1 + weight`. AFM detects the native shard layout
and preserves its unfurled norms. No checkpoint weights were modified and no
native reference performance number is claimed. Evidence:
`native-reference-recheck/`; reference source revision
`02bee553f48cd3bc7d82aba0f8073820bd924738`.

After removing the rejected HC routing switch, the clean rebuilt binary
`9e92db0b2b265dac5e19066d933219e63a369968985c000cbc0668e9d25c8487`
passed both native and overlay eight-cell regression gates, preserving all
twelve outputs per checkpoint. Overlay decode was 113.04/109.61/98.85/98.48
tok/s versus reference 100.81/107.58/99.29/98.90 at 0.5/1/2/4K. Native decode
was 89.59/85.95/77.87/85.94, with no native reference counterpart.

The comprehensive per-case judge lacked paired responses for seeded
streaming/non-streaming comparisons. The saved outputs are identical. The
harness now supplies the matching checkpoint/prompt peer, without changing
the measured results or manufacturing success; three pairing tests pass.
The original judge scores are retained, with a separate paired-evidence
rescore rather than silently overwriting the original report.

`/Volumes/edata/afm-release-artifacts/nightly-qualification-20261004`

- `fixed-prompts-control/`
- `fixed-prompts-dynamic-rows/`
- `fixed-prompts-reference/`
- `run-fixed-context-arm.py`

Each run includes binary/config identities, server command, test command,
raw trial results, transcripts, usage, logs, and completion status.
