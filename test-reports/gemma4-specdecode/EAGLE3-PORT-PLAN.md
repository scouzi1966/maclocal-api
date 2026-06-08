# EAGLE3 Speculative Decoding — afm/Swift Port Plan (dense Gemma4-31B)

**Goal:** port mlx-vlm v0.6.0's EAGLE3 speculative decoding into afm for the **dense** Gemma4-31B
verifier. **Validated upside on this M4 Pro: +25% decode** (13.0 → 16.2 tok/s, 53% accept, greedy).
Reference: `mlx_vlm/speculative/eagle3.py` + `drafters/eagle3/eagle3.py` (studied from installed
source). Only the dense target wins — the MoE 26B-A4B was all-negative (see RESULTS.md).

> Scope: multi-day. Structurally HARDER than the Qwen MTP port (which we shipped at +52%): EAGLE3
> needs the verifier to expose **captured intermediate hidden states** and ships a **separate
> PyTorch drafter** with a reduced "hot" vocab. Reuses the MTP scaffolding (snapshot/restore,
> generator structure, sidecar-load pattern). See [[gemma4-specdecode-negative]], [[mtplx-setup]].

## 0. The model + drafter (verified)

- **Verifier:** `mlx-community/gemma-4-31b-it-4bit` — dense, 60 layers, hidden 5376, vocab 262144,
  head_dim 256. afm already loads it as `Gemma4Model` (model_type `gemma4_text`, LLM factory).
- **Drafter:** `RedHatAI/gemma-4-31B-it-speculator.eagle3` — **PyTorch safetensors, bf16, 16 tensors**:
  `embed_tokens` (262144×5376), `fc` (5376 × 3*5376), one `Eagle3FirstLayer` (`layers.0`:
  input_layernorm, hidden_norm, self_attn q/k/v/o, post_attention_layernorm, mlp gate/up/down),
  `norm`, `lm_head` (32000×5376), and `d2t`/`t2d` (hot↔full vocab maps). `transformer_layer_config`
  = llama, 1 layer. `draft_vocab_size 32000`, `eagle_aux_hidden_state_layer_ids [2,30,57]`,
  `norm_before_residual true`, `norm_before_fc false`, `speculative_tokens 3`, `verifier_accept_k 1`.

## 1. Drafter architecture (from drafters/eagle3/eagle3.py)

`Eagle3FirstLayer.__call__(embeds, hidden, mask, cache, pos)`:
```
embeds        = input_layernorm(embeds)              # embed of current token
hidden_normed = hidden_norm(hidden)                  # the fused 3-layer target hidden (post fc)
residual      = hidden_normed   (norm_before_residual=true)
h = self_attn(concat([embeds, hidden_normed], -1), mask, cache, pos)   # attn input_size = 2*hidden
h = residual + h
h = h + mlp(post_attention_layernorm(h))
```
Fusion: `fc(concat(hidden_sink))` where hidden_sink = [target_h@2, target_h@30, target_h@57]
(each 5376 → fc: 3*5376 → 5376). `norm_before_fc=false` so no input_norm.

`draft_block(last_bonus, hidden, block_size, sampler, greedy)` — autoregressive chain:
```
tok = last_bonus; h_prev = hidden
while len(tokens) < block_size-1:        # block_size = speculative_tokens+1 = 4 -> drafts 3
    h_prev = _forward_tokens(tok, h_prev)    # embed(tok) + first-layer with h_prev as the hidden
    tok    = sample(_logits(h_prev))         # _logits = lm_head -> 32000 hot
    tokens.append(tok)
```
Hot→full vocab: `full_id = draft_id + d2t[draft_id]` (applied before feeding to verifier/emit).

## 2. Verifier hidden-state capture — THE NEW HARD PART

EAGLE3 needs the target's hidden states at layers **[2,30,57]** captured DURING the verify forward,
concatenated, and fed to the drafter for the next round. afm's `Gemma4TextModelInner.callAsFunction`
(layer loop at Gemma4Text.swift:735/752) returns only the final norm output — must add an optional
`captureLayerIds: [Int]` that collects `h` after those layers into a sink and returns it alongside.

- Gemma4 has per-layer input gating / scaled embeddings / softcapping — capture the residual-stream
  `h` *after* each requested decoder layer (matching mlx-vlm's `hidden_sink` semantics; verify the
  exact capture point — pre or post the layer — against the reference numerically in P0).
- New verifier hooks (mirror the MTP `forwardHidden`): `forwardCapture(inputIds, cache, captureLayerIds)
  -> (logits, [captured hidden])`, plus `embedTokens`, `projectLMHead`, and `logitsFromHidden`
  (for the optional hot-vocab path — can skip initially, project full vocab).

## 3. Draft / verify / accept / rollback loop (from eagle3.py _eagle3_rounds + _eagle3_walk)

```
prefill verifier; capture hidden@[2,30,57] at last pos -> fused via drafter.fc -> seed hidden
primary = argmax(prefill logits)
loop:
  drafts = drafter.draft_block(primary, seed_hidden, block=4)      # 3 draft tokens (hot->full)
  snapshot GDN/recurrent (here Gemma4 dense full-attn => KV trim only; NO GDN)
  verify = verifier.forwardCapture([primary]+drafts, cache, [2,30,57])   # 1 forward, 4 positions
  target = argmax(verify.logits)         # per position
  accepted = first-mismatch walk(drafts, target[:-1])  (verifier_accept_k=1)
  emit drafts[:accepted] + target[accepted]  (the bonus/correction)
  if accepted < len(drafts): rollback verifier KV to accepted+1
  drafter.accept_verified_tokens(...)   # advance drafter cache to accepted prefix, set seed
  seed_hidden = fused(verify.captured @ accepted)  ; primary = last emitted
```
Note Gemma4 dense is full-attention only (no GatedDeltaNet) → rollback is pure KVCacheSimple.trim,
SIMPLER than the Qwen MTP case (no recurrent state to snapshot). The Qwen `MTPCacheSnapshot` zero-copy
trick isn't even needed — just trim.

## 4. Loading the EAGLE3 drafter (safetensors→MLX)

The drafter is PyTorch safetensors. mlx-vlm loads + sanitizes at runtime; afm needs a Swift loader:
new `Gemma4Eagle3Drafter` module (embed/fc/1 first-layer/norm/lm_head + d2t/t2d as buffers), load the
16 bf16 tensors directly (no quantization — they're bf16), map names (mostly 1:1). `bind()` shares the
target's `embed_tokens` if shapes match (they do: 262144×5376) to save 1.4GB.

## 5. CLI / integration

`afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 <drafter-path-or-id>` (drafter is a separate
repo, not a sidecar in the model dir — so it needs its own path arg, unlike --mtp). Gate the
eligible path like MTP (greedy/text/no-tools). Auto-disable if drafter absent or model isn't dense
Gemma4.

## 6. Phased delivery + gates (rule #8)

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | capture mlx-vlm reference: (a) verifier hidden@[2,30,57] for a fixed prompt, (b) one drafter `_forward_tokens`+`_logits` step → draft token. Build Swift `Gemma4Eagle3Drafter`, load weights, match the draft logits/token | drafter draft token matches reference for one step |
| **P0b** | Swift verifier `forwardCapture` reproduces the captured hidden states bit/near-exactly vs reference | captured hidden max\|Δ\| ~0 at layers [2,30,57] |
| **P1** | greedy draft→verify→accept→rollback loop; output == greedy AR | tokens identical to AR (like MTP P2) |
| **P2** | `--eagle3` CLI + service routing; end-to-end | runnable; decode > AR (target ~+25%) |
| **P3** | matched A/B, polish | confirmed +25%, no AR regression |

## 7. Risk register

1. **Hidden-capture point correctness** (HIGH) — must capture the exact residual-stream tensor
   mlx-vlm uses (pre/post layer, before/after Gemma4's per-layer scaling). Wrong point → low
   acceptance, not a crash. Mitigate: P0b numerical match vs reference before trusting.
2. **Drafter attention details** (MED) — the first layer's attention has input_size 2*hidden and
   its own RoPE/position; the llama-style config differs from Gemma4's attention (no q/k norm, no
   gating). Port the drafter's OWN attention, don't reuse Gemma4's.
3. **Hot vocab d2t** (LOW) — just an index add; verify off-by-one (`id + d2t[id]`).
4. **Effort** — bigger than MTP (new drafter arch + capture hook). But NO GatedDeltaNet rollback
   (dense model) and NO custom Metal kernel.

**Bottom line:** validated +25% target, afm has the verifier, reference is fully readable. The novel
work is the hidden-state capture hook (P0b) and the EAGLE3 drafter module (P0). Start there.
