# EAGLE3 on afm — dense Gemma4-31B (P2 service wiring + P3 A/B)

Verifier: `mlx-community/gemma-4-31b-it-4bit` (dense, 60 layers).
Drafter: `RedHatAI/gemma-4-31B-it-speculator.eagle3` (1 EAGLE layer, hot vocab 32000, capture @ [2,30,57]).
Hardware: M4 Pro, 64 GB. Debug build. Greedy (temp=0), max_tokens=200, non-streaming.

## Prerequisite fix

EAGLE3 capture fidelity exposed a real production bug: afm's Gemma4 full-attention layers used
stock `RoPE` instead of the `proportional` variant the config specifies (wrong frequency
denominator 128 vs 512 + wrong rotate-half geometry). Fixed in `Gemma4ProportionalRoPE`
(commit on afm-opt). Layer-57 hidden-capture cosine 0.975 → 0.9988; this also corrects normal
Gemma4 generation on full-attention layers.

## CLI

```
afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 <drafter-dir> --port 9999
```

Greedy, text-only, no tools/grammar/logprobs requests take the speculative fast path; everything
else falls back to AR. `AFM_EAGLE3_BLOCK` overrides the block size (default 2).

## Block-size sweep (single prompt, decode tok/s)

| blockSize | accept | tok/round | decode tok/s | vs AR |
|-----------|--------|-----------|--------------|-------|
| 2 (default) | 68% | 1.68 | 15.2 | **+21%** |
| 3 | 57% | 2.13 | 14.3 | +14% |
| 4 | 48% | 2.42 | 11.8 | −6% |

blockSize 2 wins: each round drafts only the carried seed (zero in-block drafter forwards), so the
2-wide verify amortizes best. Larger blocks add sequential drafter forwards that erase the savings
(matches mlx-vlm's no-arg default of capping to 2).

## P3 matched A/B (blockSize=2, 3 prompts each)

| prompt | EAGLE3 tok/s | AR tok/s | speedup |
|--------|-------------|----------|---------|
| CPU cache (prose) | 15.53 | 12.6 | +23% |
| Fibonacci (code) | 16.75 | 12.7 | +32% |
| WWI causes (prose) | 13.94 | 12.5 | +11% |
| **average** | **15.41** | **12.6** | **+22%** |

AR baseline is stable (12.5–12.7 tok/s across runs). Confirms the validated +25% target
(prompt-dependent; predictable/code text accepts more and goes faster).

## Correctness

- P0: drafter bit-exact vs Python reference (cos=1.0).
- P0b: verifier hidden capture → drafter fused input cos=0.9993.
- P1: greedy spec loop output == greedy AR (token-identical) in unit test — the authoritative
  lossless-greedy proof. The server fast path calls the same `generateSpeculative`.
- P2/P3: server `--eagle3` generates coherent output at +22%.

Note: `gemma-4-31b-it` is a thinking model — at low `max_tokens` (≤200) all tokens land in
`reasoning_content` (the `<think>` block) and `content` is empty (`finish_reason: length`). That is
correct behavior, not an EAGLE3 defect; the generated reasoning is coherent.
