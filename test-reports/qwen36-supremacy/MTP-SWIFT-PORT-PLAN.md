# MTP Self-Speculative Decoding — afm/Swift Port Plan

**Goal:** port `mtplx`'s built-in-MTP self-speculative decoding to afm (Swift / mlx-swift), for
Qwen3.6-27B. **Validated upside on this M4 Pro: 1.47× decode** (21.78 vs 14.78 tok/s, temp 0.6),
quality-preserving (exact Leviathan–Chen acceptance). Reference: `github.com/youssofal/mtplx`
(Python/MLX), studied in depth — this plan is grounded in its source, not guesswork.

> Status: PLAN ONLY (no code yet). Scope is multi-day. The single biggest decode lever available
> (1.47× vs SDPA's +10% / KV-quant single digits). See [[mtplx-setup]], [[decode-speed-levers]].

---

## 0. What we're building

A speculative loop where the model's **built-in MTP head** drafts K tokens, the **full 64-layer
trunk verifies all K in one forward pass**, and exact speculative sampling accepts a prefix +1
bonus. No separate draft model. Per cycle you emit 1 (worst case) to K+1 (best case) tokens for the
cost of ~one trunk forward → the speedup.

**Model:** `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` — afm's current 16 GB 4-bit base
(byte-identical shards) **+ `mtp.safetensors`** (338 MB, 29 tensors INT4+BF16). Already downloaded
to `/Volumes/Crucial4TB/models/mtplx/`.

## 1. Architecture mapping (mtplx Python → afm Swift)

afm ALREADY has every component the MTP head needs (`Scripts/patches/Qwen3_5MoEVL.swift`):

| MTP head piece (29 tensors) | afm Swift equivalent | status |
|---|---|---|
| `mtp.pre_fc_norm_embedding`, `pre_fc_norm_hidden` | `RMSNorm` (used throughout) | reuse |
| `mtp.fc` (5120×10240, BF16) | `Linear`/quantized linear | new module, trivial |
| `mtp.layers.0` (full-attn block: q/k/v/o + q_norm/k_norm + gated-query, gate/up/down MLP, 2 layernorms) | **`Qwen3_5VLAttention` + `Qwen3_5VLDecoderLayer`** — already implements the gated-query split, q_norm/k_norm, partial-rotary | **reuse directly** |
| `mtp.norm` | `RMSNorm` | reuse |
| final projection | reuse main `lm_head`/`embedTokens.asLinear` | reuse |

The MTP head is `_mtp_core` (mtplx `mtp_patch.py:545-593`):
`e=norm_emb(embed(tok)); h=norm_hid(hidden); x=fc(concat[e,h]); x=attn_block(x, mtp_cache, pos_offset); logits=lm_head(norm(x))`.
→ A **new `Qwen3_5MTPHead` Swift module** wrapping one decoder layer + 3 norms + fc, ~150 LOC,
built from existing afm classes.

## 2. The decode loop (mtplx `generate_mtpk`, `generation.py:4211-5208`)

One cycle (greedy/temp≤0 path first — simplest, still gets the speedup):
```
1. primary = committed token for this cycle (or bonus carried from last cycle)
2. DRAFT K (autoregressive through the MTP head; cache owns RoPE offset):
     draft_hidden = trunk_last_hidden; next = primary
     for k in 0..<K:
        logits, draft_hidden = mtpHead(draft_hidden, next, mtpCache)   // single-layer KVCache
        draft_token = argmax(logits[-1]); record; next = draft_token
3. SNAPSHOT recurrent (GDN) state of the 64-layer trunk cache    // §3 — the hard part
4. VERIFY: trunkLogits, trunkHidden = trunk([primary]+drafts, trunkCache)   // ONE forward, K+1 tokens
5. ACCEPT/REJECT (greedy: draft_k == argmax(trunkLogits[:,k]); first mismatch stops)   // §4
6. COMMIT:
     all accepted  -> emit drafts; bonus = argmax(trunkLogits[:,K]); carry as next primary
     partial       -> emit accepted prefix + correction; ROLLBACK trunk cache to accepted len; re-forward committed prefix to repair live logits/hidden
7. append accepted (hidden,token) to MTP history cache; loop
```
Tokens stream out after each commit. afm's controller lives in `MLXModelService.generateStreaming`
+ the vendor `TokenIterator`; MTP replaces the inner `iterator.next()` single-token step with a
cycle. Likely cleanest as a **new `MTPGenerator`** alongside the existing path, gated by a flag.

## 3. GDN state rollback — THE HARD PART (mtplx `gdn_capture.py`, `cache_state.py`)

Qwen3.6 = 48 GatedDeltaNet (linear-attn) layers + 16 full-attn layers. The verify pass advances all
64 layers by K tokens; on partial accept the **48 GDN recurrent states must roll back to the
accepted length.** Full-attn KV just `trim()`s (afm's `KVCacheSimple` already supports this). GDN is
the risk.

- **GDN state per layer** = `[conv_state (B, kernel-1, conv_dim) bf16, gdn_state (B, Hv, Dv, Dk) fp32]`.
  afm builds these in `newCache` as `MambaCache` (the linear layers) — need to confirm afm's
  MambaCache exposes/stores both arrays and allows replace.
- **Two rollback strategies** (mtplx supports both):
  - **(A) Full-state capture** (`_from_conv_v1`): verify kernel emits per-position states
    `(B, K, Hv, Dv, Dk)`; rollback = **index** `states[:, M-1]`. Simple, O(K·Dk) memory.
  - **(B) Tape replay** (`_from_conv_tape_v1` + `_replay_v1`): verify stores only cheap per-token
    `delta` tape `(B,K,Hv,Dv)`; rollback = **replay** M steps from the pre-verify state. Memory-lean,
    but needs the replay Metal kernel.
- **Decision for the port: start with strategy (A)** — index a per-position state stack. It avoids a
  second custom kernel and the bit-exact replay round-trip (mtplx's #1 risk: per-step
  `static_cast<StT>` rounding must bit-match or the recurrent state silently corrupts). Memory cost
  at K=3 is modest. Move to tape replay only if memory pressure demands it.
- **What afm must implement:**
  - A GDN forward that **also emits the per-position state stack** during the K-token verify (afm's
    existing GatedDelta kernel `vlGatedDeltaKernel` computes the recurrence — extend it to write
    `states[:, t]` each step, or run K single-token steps and stack). **This is the main new Metal/kernel work.**
  - Snapshot the 48 GDN states before verify (deep copy that breaks MLX COW aliasing — mtplx uses
    `value + zeros`; Swift must force a distinct buffer, e.g. `MLX.contiguous` + `eval`).
  - `replace` GDN cache state with the indexed M-prefix; `eval`/materialize the committed leaf so it
    doesn't retain the big capture buffer (mtplx `detach_array_leaf` / mlx-lm #1077 ownership fix —
    **a Swift port that skips this leaks the K-token capture graph across every decode step**).
- **Riskiest items to validate (build a Swift-vs-Python bit-equality harness on reconstructed state
  for M=0..K before trusting):** (1) GDN state dtype/rounding round-trip; (2) the in-kernel q/k
  RMS-norm + scaling (`1/Dk`, `1/sqrt(Dk)` with bf16 intermediate cast); (3) SIMD reduction layout.

## 4. Acceptance sampling (mtplx `sampling.py`, `fast_sampling.py`) — LOW risk, no custom kernel

- **Greedy (temp≤0), do first:** accept draft_k iff `draft_k == argmax(trunkLogits[:,k])`; first
  mismatch → emit argmax correction, stop. Trivial; already gets the 1.47×-class win at temp 0.
- **Exact speculative sampling (temp>0), phase 2:** build tempered+top-p+top-k distributions for
  BOTH target and draft with **identical** transform (top-p THEN top-k, fp32 softmax / full-vocab
  `logsumexp`), then per token: `a = min(1, p/q)`; draw `u~U[0,1)`; **accept if u ≤ a** (inclusive);
  on reject resample residual `(p−q)+`. All stock MLX ops (`argpartition`, `argsort`, `logsumexp`,
  `cumsum`, `random.uniform`/`categorical`) — **no custom kernel**. Support ≤ top_k=20 so the
  correction math can run on host for exact parity. Draft sampler temp 0.7, target 0.6 (per contract).
- **Adaptive depth (phase 3, optional):** EV-gated controller (`adaptive.py`) scales K=1..3 by
  per-depth acceptance EWMA. Defaults: depth 3, measured accept-by-depth [1.0, 0.98, 0.94].

## 5. Loading the MTP sidecar

- afm loads weights via the patched `Load.swift`/`LLMModelFactory`. The `mtp.safetensors` is a
  separate file with `mtp.*`-prefixed INT4 tensors. Need: (a) load it alongside the main shards,
  (b) instantiate `Qwen3_5MTPHead` and assign quantized weights, (c) read the runtime contract
  (`mtplx_runtime.json`: depth_max=3, samplers, exactness gate) for config.
- The base trunk is byte-identical to afm's current model, so **no change to the 64-layer model load**;
  MTP is purely additive. A model lacking `mtp.safetensors` (plain mlx-community) just disables MTP.

## 6. Phased delivery + gates

| Phase | Deliverable | Gate (rule #8: prove it) |
|---|---|---|
| **P0 Spike** | Load `mtp.safetensors` into a `Qwen3_5MTPHead`; run the head once; compare its draft logits vs mtplx Python for the same (hidden, token) input | bit/near-bit logit match on 1 step |
| **P1 GDN rollback** | Snapshot + per-position state capture + index-restore for the 48 GDN layers; Swift-vs-Python state bit-equality harness for M=0..K | reconstructed GDN state matches a fresh M-token forward |
| **P2 Greedy MTP loop** | Full draft→verify→accept(greedy)→commit→rollback loop, temp=0, behind a `--mtp` flag | output **identical** to AR temp=0 (same tokens); decode tok/s > AR (target ~1.4×) |
| **P3 Exact sampling** | temp>0 speculative-sampling acceptance + residual | distribution-exactness check vs AR (TV≈0 over many samples); correct text |
| **P4 Adaptive + polish** | EV depth controller, bonus-token, MTP history cache, perf tuning | matched A/B decode@various depths; no regression to non-MTP path |

## 7. Risk register

1. **GDN rollback correctness** (HIGH) — silent recurrent-state drift degrades output many tokens
   later. Mitigation: P1 bit-equality harness before any loop work; start with index-capture (A) not
   tape-replay (B).
2. **Custom kernel work** (MED) — extending afm's `vlGatedDeltaKernel` to emit per-position states is
   the main new GPU code. afm's metallib-rebuild pipeline ([[metallib-is-prebuilt]]) already supports
   shipping kernel changes.
3. **mlx-swift vs mlx-python API gaps** (MED) — `argpartition`/`logsumexp`/`random.categorical`,
   COW-breaking deep copy, KVCache `trim`/`offset`, MambaCache state replace. Verify each exists in
   afm's pinned mlx-swift 0.30.3 before relying on it.
4. **Memory** (LOW-MED) — capture buffer + MTP head + draft caches on top of 20 GB resident. K=3
   index-capture adds bounded state; must materialize committed leaf to avoid retaining the capture graph.
5. **Maintenance** — MTP path is substantial new surface; gate behind a flag, keep the AR path as the
   default/fallback so a model without the sidecar (or a bug) cleanly degrades.

## 8. Effort estimate

P0–P2 (greedy, the bulk of the win): **multi-day**, dominated by P1 (GDN rollback) + the
per-position-state kernel. P3–P4: additional days. The validated 1.47× makes it the highest-value
decode work available; the GDN bit-exactness harness (P1) is the make-or-break gate — if Swift can't
reproduce the recurrent state exactly, stop there.
