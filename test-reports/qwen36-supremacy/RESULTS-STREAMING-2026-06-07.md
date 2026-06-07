# Streaming speculative decode — MTP & EAGLE3 (2026-06-07)

`feature/speculative-decoding-enhanced` adds streaming (SSE) support to the MTP and EAGLE3 fast
paths (previously serial non-streaming only). This is the result of retesting over the real
HTTP streaming path.

## Correctness (the gate)

Both validated **byte-identical to non-streaming output** (= greedy AR) — streaming preserves the
lossless guarantee:

| Path | streamed == non-streamed | output len |
|------|:--:|--|
| EAGLE3 (Gemma4-31B) | ✅ True | 572 chars |
| MTP (Qwen3.6) | ✅ True | 569 chars |

Mechanism: the generators emit per verify round via an `onToken` callback; `generateStreaming()`
yields a `StreamChunk` text delta per token (incremental detokenize); the controller does think-tag
extraction from the deltas. Reasoning (`<think>`) streams correctly.

## Speed preserved (streamed vs non-streamed, same model/prompt)

| Path | streamed | non-streamed |
|------|---:|---:|
| EAGLE3 | 14.0 tok/s | 14.0 tok/s |
| MTP | 17.5 tok/s | 17.6 tok/s |

Streaming adds no measurable overhead — the spec speedup is fully delivered through the SSE path.

## Streaming head-to-head over HTTP SSE (matched-thermal)

The canonical `probe.py` streams over HTTP; it now triggers afm's spec fast path (it measured
AR-only before this feature).

**EAGLE3 (Gemma4-31B):**
| round | afm (stream, HTTP) | mlx-vlm (gen) |
|---|---:|---:|
| 1 | 16.36 | 16.44 |
| 2 | 16.37 | 16.37 |

→ **Tie.** afm's streaming EAGLE3 matches mlx-vlm.

**MTP (Qwen3.6):** the streaming path delivers the speedup on favorable prompts (17.5 tok/s,
validated == non-streaming). On the probe's *creative* prompt MTP acceptance is low → ~14.3 tok/s
(≈ AR). MTP is **prompt-dependent**: high speedup on structured/predictable text (+52% on the
photosynthesis prompt), ≈ AR on low-acceptance creative text.

## Scope / limitations

- **Serial mode only.** Concurrent mode (`--concurrent N≥2`) routes through the BatchScheduler,
  which doesn't implement spec-decode → falls back to batched AR.
- **No stop sequences** in the fast path (falls back to AR when `stop` is requested).
- The spec fast path always engages for eligible (greedy) requests, so MTP can be marginally
  slower than AR on low-acceptance prompts (an adaptive accept-rate guard is a possible follow-up).

## Bottom line

Streaming spec-decode now works for both MTP and EAGLE3, **lossless and with no speed penalty vs
non-streaming**. afm's streaming EAGLE3 ties mlx-vlm over HTTP SSE; MTP's streaming speedup tracks
prompt predictability.
