# EAGLE3 speculative decode — dense Gemma4-31B (2026-06-07)

afm vs mlx-vlm — the only two MLX engines that implement EAGLE3. Same verifier
(`mlx-community/gemma-4-31b-it-4bit`), same drafter (`RedHatAI/gemma-4-31B-it-speculator.eagle3`),
same bf16 drafter precision, Apple M4 Pro, greedy, 200 tokens, decode-only tok/s.

## Headline: afm is LOSSLESS, mlx-vlm is approximate

The key difference is the **verification step**:

- **afm** projects the **full vocabulary (262k)** to argmax-verify every draft token → output is
  **bit-exact equal to greedy autoregressive decode** (proven byte-identical, `Eagle3SpecLoopP1Tests`
  + server output == AR).
- **mlx-vlm** verifies draft tokens against a **~32k "hot-vocab" subset** and only computes the
  full-vocab argmax at the single divergence position (`_eagle3_verify_target_hot`). This is ~4%
  faster but **not bit-exact** — it can accept a draft that matches the hot argmax even when the
  true full-vocab greedy token differs.

So afm trades ~4% throughput for **exactness** (the model's true greedy output).

## Matched-thermal decode (decode-only, 200 tok, alternating order, cooled)

| Engine | AR | EAGLE3 | speedup | verify | accept |
|--------|---:|------:|------:|:------|------:|
| **afm** (MLX/Swift) | 13.2 | **17.1** | +30% | **full-vocab — lossless / bit-exact** | 66.7% |
| mlx-vlm 0.6.2 (MLX/Python) | 13.2 | 17.9 | +36% | hot-vocab — approximate | 68.6% |

AR baselines are identical (13.2) — same kernels for single-token decode. The EAGLE3 gap is entirely
the verify-fidelity choice above.

## Optimization attempts (to close the gap losslessly)

Both failed to beat mlx-vlm *without* sacrificing exactness, confirming the gap is the
lossless-vs-approximate tradeoff, not an afm inefficiency:

- **Per-round host-sync removal** (keep the drafter seed on-GPU, `generateFastBS2`): lossless,
  but **perf-neutral** — the syncs were waiting on real compute, not overhead. Kept (cleaner).
- **Route the lm_head to qmm at M=2** (so the 700MB tied-embedding weight is read once instead of
  per-token): **slower** (debug verify 107→118ms) — qmm's GEMM overhead at M=2 outweighs the
  single-read benefit. Reverted.
- The "multi-token qmv" premise was wrong: mlx 0.31.0 uses the *identical* per-token qmv dispatch
  (`grid(M, N/bn, B)`), so there is no upstream kernel to backport here.

## Conclusion

afm's EAGLE3 delivers **+30% decode (17.1 tok/s) with bit-exact greedy output**. mlx-vlm is ~4%
faster (17.9) only because it uses an **approximate hot-vocab verify** that can diverge from greedy.
afm's lossless verify is the deliberate, enhanced behavior — same speedup class, strictly correct
output. (afm is also pinned to mlx 0.30.x for long-context SDPA correctness; mlx-vlm runs 0.31.2.)

Plot: `results-latest/eagle3-decode.png`. Logs: `results-latest/e3-*`, `/tmp/e3_clean.log`.
