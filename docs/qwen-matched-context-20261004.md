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
XCTest. The comprehensive suite with `codex-glm` as judge subsequently completed
on the frozen `candidate-runtime-hc-20261004` binary; results are recorded below.
Broader release qualification and historical-performance recovery remain pending.

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

## Rejected HC activation lookup

A BF16 lookup replacing only `silu(down/4)` and `2*sigmoid(inject/4)`
passed exhaustive 65,536-pattern equivalence and all seven HC policy tests.
Native decode was 89.15/86.02/79.33/85.74 tok/s, versus the clean control's
89.59/85.95/77.87/85.94. All twelve responses were unchanged, but the gain
was not material or consistent. Overlay decode was 113.21/111.14/92.24/99.20,
including a 6.69% 2K regression versus control; reference there is 99.29.
This isolated slowdown needs a restored control repeat before assigning its
cause. The experiment is not promoted regardless: its code and test were
removed, while patch files and raw measurements remain in the evidence root.
The rebuilt restored control returned 113.51/111.12/99.79/99.36 tok/s;
all eight performance cells passed and all twelve responses were unchanged.
Do not promote the discarded experiment using its unit equivalence alone.

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

The completed comprehensive run contains 91/91 successful harness executions
(not 91 independent protocol assertions). Original `codex-glm` scores are
82 × 5, 7 × 4, 2 × 3. Paired-evidence rescoring gives both seeded transport
variants 4/5: exact determinism, minor poem grammar. The other 3/5 is degraded
long-form fluency with repetition penalty; its cause is not assigned to the
engine without further isolation. There are no scores of 1 or 2.

Editing the running shell harness interrupted its final HTML-report step after
all 91 judge scores were saved. HTML was regenerated from those saved results;
no inference rerun or score replacement was performed. Future harness edits
must wait for the active shell process to exit.

## Independent review and provenance gate

Independent review found no concrete defect in the retained runtime-row HC
change or paired-judge evidence handling. It did identify that missing tuning,
checkpoint revision, and warmup metadata could be accepted when absent in both
runs. The corrected checker requires explicit tuning and schedule fields plus
a checkpoint revision or manifest digest; 17 tests pass. The reviewer verified
the fix. Earlier numerical/output checks above remain recorded results, but
legacy records missing this provenance cannot independently qualify a release
under the stricter gate. Raw metadata is not silently backfilled to obtain a pass.

## Completed Promptfoo regression qualification

The runtime-row candidate completed all 406 cases on the native community
checkpoint, with MTP depth 3, throughput-v2, vision enabled, and no provider
tuning environment overrides. Case identity and actual request evidence match
the qualified AFM control: 357 shared passes, 49 shared failures, and zero
candidate-only failures. Native conformance is 76/76, model/agent behavior
82/99, and forced-parser experiments 199/231. The 17 behavioral failures remain
unresolved rather than automatically attributed to model quality. This is an
AFM regression comparison, not a same-checkpoint reference-engine comparison.
Evidence: `runtime-promptfoo-comparison.json` and `runtime-hc-promptfoo/`.

## Fresh provenance-complete rebuild checks

The clean rebuilt binary is
`1f50eb823d9584fbc1ff15334a5dc938e64470f6154d3ad179808ea26c1d044b`.
Both native-community and shared-overlay comparisons against frozen `415f1a31`
pass all eight 3% performance checks and preserve all twelve responses per
checkpoint. Metadata now records explicit launch/sampling/schedule settings,
HF revision and a weight-identity manifest. Weight provenance uses HF download
metadata, same-file checks and before/after mutation checks, not a fresh full
payload rehash. Overlay weights are hardlinks to the ddalcu source revision
`6d3eaa041ef4a8c081c3a987a57afc21258c492a`.

Fresh shared-overlay comparison, same settings as above, three-trial means:

| Context | AFM prefill/decode | mlx-serve prefill/decode | AFM difference prefill/decode |
|---|---:|---:|---:|
| 0.5K | 951.69 / 113.43 | 1032.91 / 105.26 | -7.9% / +7.8% |
| 1K | 1110.42 / 109.84 | 1171.02 / 103.96 | -5.2% / +5.7% |
| 2K | 1282.66 / 98.90 | 1288.43 / 97.49 | -0.4% / +1.4% |
| 4K | 1334.20 / 98.40 | 1361.73 / 100.66 | -2.0% / -2.2% |

All twelve cross-engine prompt hashes match. This is single-request MTP-on,
cache-off performance, not a refreshed concurrency, MTP-off or semantic quality
comparison. Native-community rebuilt decode means are 89.41/85.81/79.34/83.97
tok/s; no same-checkpoint reference is available. Its historical speed deficit
remains open. Evidence: `provenance-native-{control,restored}/` and
`provenance-overlay-{control,restored,reference}/`.

## Tiny-image coverage regression and fix

Full rebuilt-native llmprobe exposed a coverage loss masked by its headline:
280/280 conformance versus the previous 283/283. Chat, Responses and Messages
vision probes returned HTTP 500 for the probe's 1x1 image and were classified
as unsupported, while the 64x64 image smoke passed. Agentic outcomes remain
6/8, with the same two invalid `list_files(path=...)` tasks; capability and
fidelity scores are 100%. The subsequent 0.5K–32K Context sweep completed.

Cause: removing blanket 1024x1024 resizing exposed the shared processor's
rejection of images smaller than its 32-pixel patch group. Provider `aba2fe7f`
lets the Qwen3VL processor used by Qwen Next upscale valid tiny inputs using
its existing minimum pixel budget. Ordinary geometry and text execution are
unchanged; other shared-helper callers retain their prior default policy.
Review also corrected fractional aspect-ratio validation. Six focused tests
pass. Live rebuilt tests correctly identify red 1x1 and blue 2x4 images,
alongside the 64x64 controls. Full llmprobe restores all three vision probes:
283/283 conformance, 100% capability/fidelity and unchanged 6/8 agentic tasks.
The sampled Messages delimiter case changes from HTTP 500 to a protocol pass
but still has 0/3 exact-content warnings; do not attribute that to this image fix.
All seven Context sizes through 32K complete, with saved responses.

The external qualification gate now rejects missing/unsupported expected vision
probes even with a perfect reduced conformance headline. Its 22 offline tests
pass, and it rejects the saved 280/280 report. Independent review found no
remaining issue in these fixes. Do not count that report as release-qualified.

## Corrected artifact: MTP-on/off matrix

Binary SHA256:
`84675472a44a7b604d0305e212c2800470680a0681ac11a96b6b588e8d9db2d2`.
Provider `aba2fe7f`; coverage gate consumer `ba323e7`. No tuning environment
overrides. Same fixed 12 prompts and prompt-token counts per arm. Values are
three-trial means, single request, cache off, reasoning off. MTP-on uses depth
3 and throughput-v2; MTP-off omits the MTP profile. Both use 8192 prefill chunks.

Shared ddalcu vision-overlay, same weights on AFM and mlx-serve 26.10.1:

| MTP | Context | AFM prefill | Reference prefill | AFM decode | Reference decode |
|---|---|---:|---:|---:|---:|
| On | 0.5K | 953.56 | 1026.81 | 112.45 | 104.24 |
| On | 1K | 1108.03 | 1175.69 | 107.81 | 104.19 |
| On | 2K | 1284.39 | 1284.75 | 97.98 | 97.54 |
| On | 4K | 1332.46 | 1343.58 | 98.28 | 100.14 |
| Off | 0.5K | 905.15 | 1014.62 | 68.90 | 72.01 |
| Off | 1K | 1051.84 | 1171.51 | 68.32 | 69.68 |
| Off | 2K | 1199.40 | 1285.04 | 62.06 | 63.28 |
| Off | 4K | 1237.98 | 1356.80 | 62.66 | 64.78 |

MTP-on is within 10% for every cell. MTP-off decode is 1.9–4.3% behind;
prefill is 6.7–10.8% behind. The 0.5K and 1K prefill cells do not meet the 10%
target. No claim of universally achieved parity is made.

Native-community checkpoint, same native revision as above:

| Context | MTP-on prefill/decode | MTP-off prefill/decode | Same-checkpoint reference |
|---|---:|---:|---|
| 0.5K | 949.85 / 89.41 | 918.86 / 35.53 | Cannot load native layout |
| 1K | 1114.53 / 85.80 | 1048.75 / 35.49 | Cannot load native layout |
| 2K | 1293.85 / 79.63 | 1199.76 / 34.29 | Cannot load native layout |
| 4K | 1340.51 / 86.59 | 1238.94 / 34.47 | Cannot load native layout |

Native and overlay MTP-on/off each pass all eight 3% rebuild regression
checks against the frozen control, with identical outputs. The native MTP-off
control differs by under 0.9% in every cell: its lower speed is not a rebuild
regression. All 48 output texts match their respective controls. The native historical
fast-binary gap remains open. Concurrent aggregate throughput, cache-enabled
performance and the separate AFM-converted checkpoint are not refreshed by
this matrix. Evidence: `tiny-fixed-*` and `tiny-image-fixed-native-full/`.

## MTP-off shared-profile diagnostic

The existing throughput-v2 CLI profile also affects shared decode components
when `--mtp` is omitted. Server logs confirm the embedded predictor remains
disabled. This is not a default change or a claim of complete qualification.
Same native checkpoint, three fixed trials per context:

| Configuration | 0.5K decode | 1K decode | 2K decode | 4K decode |
|---|---:|---:|---:|---:|
| No profile, MTP off | 35.53 | 35.49 | 34.29 | 34.47 |
| throughput-v2, MTP off | 40.13 | 40.37 | 38.92 | 38.98 |
| Profile, CPU n-gram disabled | 39.66 | 39.60 | 38.26 | 38.25 |
| Profile, quantized HC disabled | 36.42 | 36.36 | 34.63 | 35.03 |

The full profile improves decode about 13–14% with effectively unchanged
prefill. All twelve outputs in each arm match the ordinary MTP-off control.
The one-switch ablations point primarily to corrected HC fusion, with a smaller
CPU n-gram contribution. They are diagnostic-only and cannot pass the clean
release gate. Seventeen existing fusion/lookup/cache-history tests pass under
throughput-v2 without skips. Full no-MTP/profile llmprobe and Context
qualification completed in `native-off-profile-full-qualification/`:
283/283 conformance, 100% capability and fidelity, and 6/8 agentic tasks.
The two unsuccessful tasks are the same as the no-profile baseline:
`coding-follow-test-output` and `coding-already-done`, both involving an
unexpected `path` argument to `list_files`; the first also edits before tests.
These are not attributed to the model or engine solely from this comparison.
All three vision API probes pass, including the tiny-image regression case.
The seven-size Context sweep completed through 32K without a crash; the saved
32K response is coherent but cut at the requested 128-token limit. This is not
a substitute for broad semantic quality evaluation. llmprobe still reports
SHOULD warnings for the Responses error message and Messages token-count
estimate. Full Promptfoo and comprehensive AI-judge qualification of this
additional MTP-off/profile combination remain outstanding.
No same-checkpoint native reference result exists; the overlay is not substituted.

## Evidence root

`/Volumes/edata/afm-release-artifacts/nightly-qualification-20261004`

- `fixed-prompts-control/`
- `fixed-prompts-dynamic-rows/`
- `fixed-prompts-reference/`
- `run-fixed-context-arm.py`

Each run includes binary/config identities, server command, test command,
raw trial results, transcripts, usage, logs, and completion status.
