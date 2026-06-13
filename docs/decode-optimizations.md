# Decode Optimizations — Feature Guide

New decode-performance features on the MLX backend. All are **opt-in** (default behavior is
unchanged) and **output-validated** (each produces correct output before any speedup is claimed).

Common env var for every example (avoids re-downloading models):

```bash
export MACAFM_MLX_MODEL_CACHE=/Volumes/Crucial4TB/models/vesta-test-cache
```

---

## Which flag for which model? (start here)

afm has exactly **two** speculative-decoding options — one per model family. Both are lossless
(output identical to greedy) and need a specific drafter/checkpoint.

| Running… | Use | Speedup | Needs |
|----------|-----|---------|-------|
| **Qwen3.6-27B** | `--mtp` | **~+52%** | checkpoint with the MTP head: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` (has `mtp.safetensors`) |
| **Gemma4-31B (dense)** | `--eagle3 <drafter-dir>` | **~+30%** | the `RedHatAI/gemma-4-31B-it-speculator.eagle3` drafter |
| anything else (incl. **Gemma4 MoE 26B-A4B**) | — | none | no speculative option — use normal decode |

```bash
# Qwen3.6  → MTP
afm mlx -m Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --mtp --port 9999
# Gemma4-31B dense → EAGLE3
afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 <eagle3-drafter-dir> --port 9999
```

Both engage only for greedy (`temperature: 0`), text-only requests (streaming or not); anything
else silently uses normal autoregressive decode. Full details per feature below.

**Evaluated but NOT shipped** (so you don't go looking): Gemma4 MoE spec-decode (all methods slower
than AR), the Gemma4 "assistant"-MTP that llama.cpp uses (~+9% in MLX — not worth it; llama.cpp's
GGUF/Metal version is a different engine, ~+39% on M4 Pro), DFlash (negative). On M4 Pro the practical
spec-decode ceiling is ~+40% in any engine — afm's lossless EAGLE3/MTP are at or near it.

---

## 1. EAGLE3 speculative decoding — dense Gemma4-31B

**Flag:** `--eagle3 <drafter-dir>`

Lossless speculative decoding for the **dense Gemma4-31B** verifier using an EAGLE3 drafter.
Output is **bit-exact to greedy autoregressive decode**. ~**+30% decode** on M4 Pro.

```bash
# drafter dir = the EAGLE3 speculator (config.json + safetensors)
DRAFTER=~/.cache/huggingface/hub/models--RedHatAI--gemma-4-31B-it-speculator.eagle3/snapshots/<hash>

afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 "$DRAFTER" --port 9999
```

Then call it like any OpenAI endpoint:

```bash
curl -s http://127.0.0.1:9999/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mlx-community/gemma-4-31b-it-4bit",
  "messages": [{"role":"user","content":"Write a Python function for the nth Fibonacci number."}],
  "temperature": 0, "max_tokens": 200, "stream": false
}'
```

**When the fast path engages** (otherwise it silently falls back to plain AR):
- greedy (`temperature: 0`), text-only, no `tools` / `response_format` / `logprobs` / `stop`. **Streaming (`stream: true`) is supported** — tokens are emitted per verify round.
- verifier is a dense Gemma4 text model (else logs `verifier is not a dense Gemma4 text model` and uses AR).

**Tuning:** block size (drafts per round) defaults to 2 (the sweet spot). Override:

```bash
AFM_EAGLE3_BLOCK=3 afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 "$DRAFTER" --port 9999
```

**Notes**
- afm's verify is **full-vocab → lossless** (bit-exact greedy). It does **not** use the approximate
  hot-vocab verify some engines use, so output exactly matches the model's greedy decode.
- MoE Gemma4 (26B-A4B) is **not** accelerated by spec-decode (validated negative) — `--eagle3` only
  helps the dense 31B.

---

## 2. MTP self-speculative decoding — Qwen3.6

**Flag:** `--mtp`

Self-speculative decoding using Qwen3.6's **in-model MTP head** (no separate draft model).
Output identical to greedy AR. ~**+52% decode** vs AR; ~**+47%** end-to-end.

Requires a model checkpoint that ships the `mtp.safetensors` sidecar (the plain
`mlx-community/Qwen3.6-27B-4bit` conversion **strips** the MTP head, so `--mtp` no-ops there):

```bash
# model dir must contain mtp.safetensors next to the base weights
afm mlx -m Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --mtp --port 9999
```

```bash
curl -s http://127.0.0.1:9999/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
  "messages": [{"role":"user","content":"Explain how a CPU cache works in 4 sentences."}],
  "temperature": 0, "max_tokens": 200, "stream": false
}'
```

**When the fast path engages:** same eligibility as EAGLE3 (greedy, text-only, streaming or
non-streaming, no tools/grammar/logprobs/stop). No MTP head present → silently falls back to AR.

`--mtp-depth N` is accepted for compatibility but **not used** (the loop uses the fixed
depth-2-bonus structure from mlx-lm PR #990).

---

## 3. Long-context SDPA (automatic — no flag)

The pinned mlx-swift 0.30.3 tree is patched with **0.31.3's adaptive-block 2-pass SDPA**
(backported in `Scripts/patches/mlx-cpp-sdpa/`). This is applied at build time and needs no flag —
it just makes long-context decode faster: **~+10% decode@16k** (≈13.0→14.4 tok/s on
Qwen3.6-27B-4bit / M4 Pro), correct at all depths.

Applied automatically by the full build:

```bash
./build.sh                  # applies all patches + rebuilds default.metallib + builds
```

> The metallib **must** be rebuilt after the SDPA patch (`./build.sh` does this via
> `Scripts/rebuild-metallib.sh`); a kernel/dispatch mismatch silently produces garbage.

---

## 4. Faster reasoning TTFT (automatic)

Streaming responses now emit the `<think>` open tag eagerly, cutting reasoning **time-to-first-token
~610ms → ~346ms**. No flag — it just applies to streaming chat completions on thinking models.

## 5. Metal-kernel prewarm (automatic)

Metal kernels are prewarmed on server startup, so the **cold first token** is faster. No flag.

---

## Debugging / profiling

```bash
# [MTP]/[EAGLE3] decode tok/s, [KVCache] hit/miss, tool-call + timing logs
AFM_DEBUG=1 afm mlx -m <model> --mtp --port 9999

# EAGLE3 per-round phase breakdown (verify vs draft ms)
AFM_DEBUG=1 AFM_EAGLE3_PROFILE=1 afm mlx -m <gemma4-31b> --eagle3 <drafter> --port 9999
```

---

## Quick reference

| Feature | Flag | Model | Speedup | Output |
|---------|------|-------|---------|--------|
| EAGLE3 | `--eagle3 <dir>` | dense Gemma4-31B | ~+30% decode | lossless (== greedy AR) |
| MTP | `--mtp` | Qwen3.6 w/ `mtp.safetensors` | ~+52% decode | identical to greedy AR |
| SDPA backport | (build-time) | any | ~+10% @16k | correct at all depths |
| Eager think-tag | (auto) | thinking models | TTFT 610→346ms | unchanged |
| Kernel prewarm | (auto) | any | faster cold token | unchanged |

| Env var | Default | Purpose |
|---------|---------|---------|
| `MACAFM_MLX_MODEL_CACHE` | — | model cache root (avoids re-download) |
| `AFM_EAGLE3_BLOCK` | `2` | EAGLE3 drafts per round |
| `AFM_DEBUG` | off | decode tok/s, cache, timing logs |
| `AFM_EAGLE3_PROFILE` | off | EAGLE3 per-round verify/draft timing |

All fast paths require: **greedy** (`temperature: 0`), text-only, no `tools` / `response_format` /
`logprobs` / `stop` sequences. **Streaming and non-streaming are both supported.** Concurrent mode
(`--concurrent N≥2`) routes through the batch scheduler and does **not** use MTP/EAGLE3 (serial only).
Anything else uses normal autoregressive decode.
EOF
