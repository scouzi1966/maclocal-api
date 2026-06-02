# Gemma 4 speculative decoding — validation result (NEGATIVE)

**Date:** 2026-06-01 · **Machine:** Apple M4 Pro · 64 GB
**Verifier:** `mlx-community/gemma-4-26b-a4b-it-4bit` (Gemma 4 26B-A4B **MoE**, 128 experts, top-8, ~4B active)
**Method:** mlx-vlm 0.6.0 `generate --draft-kind {mtp,dflash,eagle3}`, 200-tok greedy, temp 0, same prompt.

## Result — all three are SLOWER than plain AR on this MoE at batch 1

| config | decode tok/s | vs AR | accept | tok/round |
|---|---|---|---|---|
| **AR baseline** | **70.0** | — | — | — |
| EAGLE3 (`RedHatAI/...speculator.eagle3`) | 64.3 | **0.92×** | 45.3% | 1.45 |
| MTP (`mlx-community/gemma-4-26B-A4B-it-assistant-bf16`) | 49.7 | **0.71×** | 42.0% | 2.25 |
| DFlash (`z-lab/gemma-4-26B-A4B-it-DFlash`) | 44.9 | **0.64×** | 32.9% | 2.04 |

## Why (fundamental, not an implementation gap)

The 26B-A4B MoE activates only ~4B params/token, so **AR decode is already cheap (70 tok/s)**.
Speculation's cost is a verify forward over K+1 tokens, which on an MoE means **gathering K+1
distinct expert sets** — cost scales ~linearly with K and erases the savings. Acceptance is also
low (33–45%) because these are *separate* drafters, unlike Qwen3.6's in-model MTP head (81%).

The Google blog's ~2.2× and the mlx-vlm PR's 1.49× are at **batch 4–8** (continuous batching),
where the MoE expert-gather amortizes across the batch. **afm does single-stream batch-1 decode**,
where every one of these is a net loss.

Contrast: Qwen3.6 MTP gave **+52%** in afm — because its hybrid GatedDeltaNet trunk makes the
verify cheap relative to the speculative win, and the in-model MTP head hits 81% acceptance.

## Recommendation

**Do NOT port EAGLE3 / DFlash / Gemma4-MTP to afm.** It would be multi-day work (incl.
safetensors→MLX converters for the PyTorch EAGLE3/DFlash drafters) to ship something **slower**
than afm's existing AR decode on this model. Revisit ONLY if afm gains continuous batching
(batch 4–8 unlocks the ~1.5–2.2× regime), or for a **dense** Gemma 4 target (31B-it), where the
verify is expensive enough for speculation to pay — not measured here.

Drafters validated (downloaded, run, all functional in mlx-vlm — the negative result is real,
not a setup failure): MTP MLX-ready; DFlash + EAGLE3 are PyTorch, mlx-vlm converts at load.
