# Gemma 4 speculative decoding — validation result

**Verdict: target-dependent.** MoE 26B-A4B → all 3 methods SLOWER than AR (batch-1). Dense 31B →
**EAGLE3 +25%** (worth porting), MTP +5%, DFlash −19%.


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

## Dense 31B (gemma-4-31b-it-4bit) — EAGLE3 WINS (+25%)

Same harness, **dense** verifier (60 layers, 5376 hidden, no experts; AR baseline 13.0 tok/s —
slow because it reads all 31B/token):

| config | tok/s | vs AR | accept | tok/round |
|---|---|---|---|---|
| AR baseline | 13.0 | — | — | — |
| **EAGLE3** (`RedHatAI/gemma-4-31B-it-speculator.eagle3`) | **16.2** | **+25%** | 53.1% | 1.53 |
| MTP (`mlx-community/gemma-4-31B-it-assistant-bf16`) | 13.6 | +5% | 44.5% | 2.33 |
| DFlash (`z-lab/gemma-4-31B-it-DFlash`) | 10.5 | −19% | 32.1% | 2.04 |

**The dense model flips the result positive** (vs all-negative on the MoE): reading all 31B/token
makes AR expensive (13 tok/s), so the speculative verify amortizes. **EAGLE3 is the winner**
(+25%) — its 1-layer feature-reuse draft adds the least overhead per speculative step. MTP barely
breaks even (+5%); DFlash still loses (low 32% acceptance — the diffusion drafter mismatches this
model at batch-1).

So: **EAGLE3 on dense Gemma4-31B is worth porting to afm** (~+25%); the MoE 26B-A4B is not (all
negative). DFlash is not worth it on either target at batch-1.

## Recommendation

**MoE 26B-A4B:** do NOT port any of the three — all slower than AR at batch-1.

**Dense 31B:** **EAGLE3 is worth porting (+25%, measured).** MTP marginal (+5%), DFlash negative.
The dense verify is expensive enough that EAGLE3's cheap 1-layer feature-reuse draft pays off.
Porting cost: EAGLE3 drafter is PyTorch (`RedHatAI/gemma-4-31B-it-speculator.eagle3`, 1.x B, dense
Llama-style 1-layer + d2t hot-vocab map), needs a safetensors→MLX `sanitize`; the afm verifier
must expose captured hidden states at layers [from config] + `rollback_speculative_cache`. mlx-vlm
`speculative/eagle3.py` is the reference. Still revisit MoE only if afm gains continuous batching.

Drafters validated (downloaded, run, all functional in mlx-vlm — the negative result is real,
not a setup failure): MTP MLX-ready; DFlash + EAGLE3 are PyTorch, mlx-vlm converts at load.
