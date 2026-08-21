# Decode Optimizations — Feature Guide

New decode-performance features on the MLX backend. All are **opt-in** (default behavior is
unchanged) and **output-validated** (each produces correct output before any speedup is claimed).

Common env var for every example (avoids re-downloading models):

```bash
export MACAFM_MLX_MODEL_CACHE=/Volumes/Crucial4TB/models/vesta-test-cache
```

---

## Which flag for which model? (start here)

AFM has three MLX speculative-decoding options plus DwarfStar DSpark. Each
requires a compatible target and drafter or sidecar.

| Running… | Use | Speedup | Needs |
|----------|-----|---------|-------|
| **Qwen3.6-27B** | `--mtp` | **~+52%** | checkpoint with the MTP head: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` (has `mtp.safetensors`) |
| **Gemma4-31B (dense)** | `--eagle3 <drafter-dir>` | **~+30%** | the `RedHatAI/gemma-4-31B-it-speculator.eagle3` drafter |
| **Qwen3.8-27B** | `--dflash2 <repo-or-dir>` | no local speed result | `incoai/Qwen3.8-27B-DFlash2` |
| **Muse-Glimmer-30B** | `--dflash2 <repo-or-dir>` | no local speed result | `incoai/Muse-Glimmer-30B-DFlash2` |
| anything else (incl. **Gemma4 MoE 26B-A4B**) | — | none | use normal decode |

```bash
# Qwen3.6  → MTP
afm mlx -m Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --mtp --port 9999
# Gemma4-31B dense → EAGLE3
afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 <eagle3-drafter-dir> --port 9999
# Qwen3.8-27B → DFlash 2
afm mlx -m mlx-community/Qwen3.8-27B-4bit \
  --dflash2 incoai/Qwen3.8-27B-DFlash2 --port 9999
```

These paths engage only for supported greedy, serial requests. DFlash 2 has an
explicit preferred/required policy described below.

Legacy DFlash and DFlash 2 are distinct checkpoint contracts. Only checkpoints
declaring `DFlash2DraftModel` are accepted by `--dflash2`.

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

## 2. MTP self-speculative decoding — Qwen3.6 and Qwen3.8

**Flag:** `--mtp`

Self-speculative decoding using a small MTP head published for the base model.
Quality-preserving: bit-exact to greedy AR on short generations, near-greedy on long ones (the
depth-2 "bonus" token comes from a batched verify forward, so longer outputs may differ
token-for-token while staying greedy-quality). ~**+52% decode** vs AR; ~**+47%** end-to-end.

Qwen3.6 requires a checkpoint that ships `mtp.safetensors`. For Qwen3.8, AFM detects the
architecture and quantization from `config.json`, then downloads the matching standalone MTP
repository (`4bit`, `8bit`, `bf16`, `mxfp4`, `mxfp8`, or `nvfp4`). The head is prefetched even
without `--mtp`, and remains separate from the base weights so the normal model loader cannot
ingest it accidentally.

```bash
# model dir must contain mtp.safetensors next to the base weights
afm mlx -m Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --mtp --port 9999

# Automatically uses mlx-community/Qwen3.8-27B-MTP-4bit
afm mlx -m mlx-community/Qwen3.8-27B-4bit --mtp --port 9999

# Explicit repository, local directory, or .safetensors override
afm mlx -m mlx-community/Qwen3.8-27B-4bit --mtp \
  --mtp-model mlx-community/Qwen3.8-27B-MTP-4bit --port 9999
```

```bash
curl -s http://127.0.0.1:9999/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
  "messages": [{"role":"user","content":"Explain how a CPU cache works in 4 sentences."}],
  "temperature": 0, "max_tokens": 200, "stream": false
}'
```

**When the fast path engages:** same eligibility as EAGLE3 (greedy, text-only, serial streaming
or non-streaming, no tools/grammar/logprobs/stop). Concurrent, continuous-batch, prefix-cache,
and ineligible requests remain fully functional through autoregressive decoding. If `--mtp` is
explicit and its head cannot be resolved, startup fails instead of silently running without the
requested acceleration. Without `--mtp`, a head-prefetch failure is non-fatal so offline AR usage
continues to work.

`--mtp-depth N` is accepted for compatibility but **not used** (the loop uses the fixed
depth-2-bonus structure from mlx-lm PR #990).

---

## 3. DFlash 2 — Qwen 3.8 and Muse Glimmer

**Flags:** `--dflash2 <repo-or-directory>`, `--dflash2-block <N>`, and
`--dflash2-required`

DFlash 2 performs one-pass parallel drafting with metadata-validated Qwen 3.8
and Muse Glimmer target adapters. AFM validates the draft architecture, target
dimensions, tokenizer/context/RoPE contract, feature taps, and all safetensor
shapes before loading draft weights. Repository names are not used as proof of
compatibility.

```bash
afm mlx -m mlx-community/Qwen3.8-27B-4bit \
  --dflash2 incoai/Qwen3.8-27B-DFlash2 --port 9999

afm mlx -m mlx-community/Muse-Glimmer-30B-4bit \
  --dflash2 incoai/Muse-Glimmer-30B-DFlash2 --port 9999
```

The supported fast path is greedy, serial, text-only generation without tools,
grammar, logprobs, or string stop sequences. Streaming and non-streaming use
the same draft/verify generator. Prefix-cache and concurrent/batch execution
remain autoregressive in preferred mode. `--dflash2-required` converts any
unsupported request or runtime conflict into an error before output begins.

Requests may explicitly disable a loaded runtime or require it:

```json
{
  "model": "mlx-community/Qwen3.8-27B-4bit",
  "messages": [{"role": "user", "content": "Explain speculative decoding."}],
  "temperature": 0,
  "speculative_decoding": {
    "mode": "dflash2",
    "requirement": "required",
    "max_draft_tokens": 4
  }
}
```

Use `"mode": "off"` for per-request autoregressive decoding. A request cannot
load or switch drafters; an optional `drafter` value must match the resource
selected at startup. AFM exports neutral draft, accepted, emitted, cycle, and
phase-time counters under `/metrics`, and AFMKit receives the same summary in
`afm.speculative_decoding.v1` response metadata/events.

The current generator is correctness-first: greedy output is target-equivalent
in synthetic qualification and bounded live smoke tests on both released pairs,
but draft KV caching, speculative prefix snapshots, sampling rejection, and
batched verification are deferred. No local Qwen/Muse speedup claim has been
qualified.

---

## 4. Long-context SDPA (automatic — no flag)

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

## 5. Faster reasoning TTFT (automatic)

Streaming responses now emit the `<think>` open tag eagerly, cutting reasoning **time-to-first-token
~610ms → ~346ms**. No flag — it just applies to streaming chat completions on thinking models.

## 6. Metal-kernel prewarm (automatic)

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
| MTP | `--mtp` | Qwen3.6 sidecar or Qwen3.8 matching head | serial decode acceleration | near-greedy (bit-exact on short gens) |
| DFlash 2 | `--dflash2 <repo-or-dir>` | Qwen3.8-27B or Muse-Glimmer-30B | no local speed result | lossless greedy |
| SDPA backport | (build-time) | any | ~+10% @16k | correct at all depths |
| Eager think-tag | (auto) | thinking models | TTFT 610→346ms | unchanged |
| Kernel prewarm | (auto) | any | faster cold token | unchanged |

| Env var | Default | Purpose |
|---------|---------|---------|
| `MACAFM_MLX_MODEL_CACHE` | — | model cache root (avoids re-download) |
| `AFM_EAGLE3_BLOCK` | `2` | EAGLE3 drafts per round |
| `AFM_DEBUG` | off | decode tok/s, cache, timing logs |
| `AFM_EAGLE3_PROFILE` | off | EAGLE3 per-round verify/draft timing |

All fast paths require supported greedy requests. **Streaming and non-streaming
are both supported.** Concurrent mode routes through the batch scheduler and
does not run serial MTP/EAGLE3/DFlash 2. DFlash 2 preferred mode falls back to
AR; required mode rejects the conflict before generation.
