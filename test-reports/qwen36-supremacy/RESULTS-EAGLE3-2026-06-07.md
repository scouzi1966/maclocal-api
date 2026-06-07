# EAGLE3 speculative decode — dense Gemma4-31B (2026-06-07)

Feature-fair head-to-head on the **only two MLX engines that implement EAGLE3**: afm (`--eagle3`)
and mlx-vlm (`--draft-kind eagle3`). The other MLX servers (oMLX, LM Studio, rapid-mlx, ollama)
do not expose EAGLE3, so they are not in this comparison.

- **Verifier:** `mlx-community/gemma-4-31b-it-4bit` (dense, 60 layers).
- **Drafter:** `RedHatAI/gemma-4-31B-it-speculator.eagle3` (same for both engines).
- **Hardware:** Apple M4 Pro, 64 GB. Greedy, 200 tokens, decode-only tok/s (excludes prefill;
  matches mlx-vlm's "Generation" metric and afm's in-server decode loop).

| Engine | AR | EAGLE3 | speedup | accept | tok/round |
|--------|---:|------:|------:|------:|------:|
| afm (MLX/Swift) | 13.2 | 17.2 | **+31%** | 66.7% | 1.67 |
| mlx-vlm 0.6.2 (MLX/Python) | 13.2 | **17.9** | **+36%** | 68.6% | 1.69 |

## Reading it

- **AR baselines are identical** (13.2 tok/s) — same weights, same hardware.
- **Acceptance is identical** (~67–69%, ~1.68 tok/round) — same drafter, so draft quality matches.
- **mlx-vlm extracts slightly more** from the same accepted tokens (17.9 vs 17.2): per-round
  execution is a bit leaner, most likely its newer **mlx 0.31.2** kernels vs afm's pinned 0.30.3,
  plus a tighter loop. afm's in-loop decode is ~17.2 tok/s; the end-to-end server number incl.
  prefill is ~16.2 tok/s.
- Net: afm's first-cut EAGLE3 (+31%) is within ~4% of the reference mlx-vlm implementation (+36%).
  Headroom for afm is in the per-round loop (fewer GPU syncs) and the MLX version gap.

Both confirm EAGLE3 is a real **~+30–36% decode win** on the dense Gemma4-31B (it is *negative* on
the MoE 26B-A4B — see `gemma4-specdecode`).

Plot: `results-latest/eagle3-decode.png`. Logs: `results-latest/e3-*`.
