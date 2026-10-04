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
API qualification is running on the experimental binary; broader release
qualification is still pending.

## Evidence root

`/Volumes/edata/afm-release-artifacts/nightly-qualification-20261004`

- `fixed-prompts-control/`
- `fixed-prompts-dynamic-rows/`
- `fixed-prompts-reference/`
- `run-fixed-context-arm.py`

Each run includes binary/config identities, server command, test command,
raw trial results, transcripts, usage, logs, and completion status.
