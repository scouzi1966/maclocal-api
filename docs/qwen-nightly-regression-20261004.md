# Qwen Next candidate regression: release blocked

The candidate `v0.9.20-next.20261004.f688d14` is not release-qualified.
Successful conformance tests do not override performance or quality failures.

## Reproduction

Same native community checkpoint:
`/Volumes/edata/models/vesta-test-cache/mlx-community/Qwen3.8-Flash-Next-4bit-mtp`.
One client, server capacity 2, VLM text route, MTP depth 3, greedy/top-p 1,
128 output tokens, three measured trials per context after a disjoint warmup.
The archived September 27 recipe and deterministic prompts were reused.

| Context | Retained binary mean decode | Candidate, same recipe | Candidate plus explicit corrected HC |
|---|---:|---:|---:|
| 0.5K | 96.53 | 59.04 | 75.23 |
| 1K | 98.13 | 57.19 | 72.76 |
| 2K | 88.86 | 53.25 | 66.18 |
| 4K | 96.85 | 58.98 | 73.33 |

All rates are tokens/second, means of all three raw trials, not best trials.
The first matched pair's prefill differs by less than 0.5% at every context.
This is a measured decode regression; enabling corrected HC alone does not
recover the old recipe. Output trajectories differ, requiring quality review.
Do not reinstate old arithmetic solely to recover throughput: later fixes
addressed native tool-quality failures.

Retained binary SHA256:
`dda5c6abb344676a4ce90c22d7874e7aa8dcf1c199b8eaf32719d57c2204883e`.
Candidate SHA256:
`e69f5750447866b2902a568252be9b4ab9e69b2e9171f71c8356036f61cd91c8`.

Raw evidence root:
`/Volumes/edata/afm-benchmarks/qwen-community-lookup-20260924/`.
Arms: `cache-only-head-api-regression-old-a-20261004`,
`cache-only-head-api-regression-new-a-20261004`, and
`cache-only-head-api-regression-new-hc-20261004`.
Historical means and peaks remain in `MATCHED-WARM-PARITY-20260927.md`.

## Confirmed profile wiring defect

The `throughput-v2` dictionary selects quantized HC and sparse verification,
but both kernel owners read raw `ProcessInfo` instead of the profile resolver.
Consequently selecting the CLI profile did not activate those two paths.
The old unit tests checked dictionary equivalence, not kernel activation.
The provider fix routes both owners through the resolver and adds fresh-process
activation tests. The corrected arithmetic remains unchanged. Performance and
quality requalification are still required; this fix is not proof of parity.

## Profile repair validation (interim)

Kernel-owner activation passes in three fresh test processes: profile absent,
`throughput-v2`, and v2 with both affected flags explicitly set to zero.
Release rebuild passed. The repaired diagnostic binary SHA256 is
`7418ee0092d3598d8f72275414731c4d91e80b570726902b09bb7fc27cd531b2`.
The provider change is `f69eae4a`; it changes configuration lookup only,
not projection arithmetic.

Same-harness CLI-only comparison (first repaired arm versus unpatched control):

| Context | Unpatched CLI v2 | Repaired CLI v2 | Decode improvement |
|---|---:|---:|---:|
| 0.5K | 81.11 | 87.90 | +8.4% |
| 1K | 77.68 | 85.07 | +9.5% |
| 2K | 70.85 | 79.74 | +12.6% |
| 4K | 78.89 | 85.74 | +8.7% |

All twelve measured outputs match exactly. All prefill cells pass the 3%
regression gate. These numbers do not replace the faster historical baseline.
Subsequent repeats and long-context measurements are recorded below.

The repeated repaired arm produced 89.24 / 85.34 / 80.21 / 86.26 tok/s;
all twelve outputs again match the unpatched CLI-profile control exactly.
The final retained-binary control produced 96.13 / 97.54 / 88.64 / 97.15.
Thus the 9–13% historical short-context deficit remains reproducible; do not
mark the overall regression resolved. A draft-shortlist-off diagnostic produced
87.26 / 83.69 / 78.79 / 84.56, so it is not promoted.

Disabling verifier QMM produced 84.09 / 87.08 / 78.94 / 82.73, with no
across-context win, so it is also not promoted. No experimental overrides are
added to the CLI profile from these screens.

The repaired CLI-only long sweep (two measured trials per context after a
separate warmup) reports means of 81.83 at 2K, 71.80 at 8K, 76.07 at 16K, and
69.65 tok/s at 32K. This uses different prompt-marker scheduling from the
initial qualification; do not claim a causal delta against its independent
CSV phase peaks. Evidence: `cache-only-head-api-profile-fixed-long-20261004`.

## Regression gate usage

`Scripts/check-context-regression.py BASELINE_CASE CANDIDATE_CASE` checks raw
trial completeness, matching checkpoint metadata, prompt hashes, token counts,
context order, warmup schedule, all-trial prefill/decode means, and exact outputs.
Default tolerance is 3%; changing it is an explicit argument, not a hidden
baseline adjustment. Peaks are reported separately. Output changes fail the
combined gate and need independent quality review; exact equality alone is not
semantic qualification. The gate must pass on the final packaged binary before
promotion. Reversed-order repeats are required before attributing small deltas.
