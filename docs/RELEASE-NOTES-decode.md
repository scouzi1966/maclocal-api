# Release notes & social copy — Decode optimizations

Source of truth for the next release README/CHANGELOG and social posts. Numbers are M4 Pro, 64 GB,
4-bit, matched-thermal, decode-only unless noted. Charts: `eagle3-decode.png`, `social-decode.png`
(in `test-reports/qwen36-supremacy/results-latest/`).

---

## A. Release notes (README / CHANGELOG ready)

### ⚡ Speculative & long-context decode (opt-in)

This release adds speculative decoding and long-context attention speedups to the MLX backend.
All are opt-in flags; default behavior is unchanged and every optimization is **output-validated**.

- **MTP self-speculative decoding for Qwen3.6** (`--mtp`) — **~+52% decode**, output identical to
  greedy. Uses the model's in-model MTP head (needs the `mtp.safetensors` sidecar). On par with the
  reference mlx-vlm MTP (~23 tok/s on Qwen3.6-27B / M4 Pro).

- **EAGLE3 speculative decoding for dense Gemma4-31B** (`--eagle3 <drafter-dir>`) — **~+30% decode**,
  and **lossless**: output is bit-exact to greedy autoregressive decode (verified token-for-token).
  Unlike engines that use an approximate hot-vocab verify, afm verifies against the full vocabulary,
  so you get the model's true greedy output at speculative speed.

- **Long-context attention** — backported MLX 0.31.3's adaptive-block 2-pass SDPA into the pinned
  0.30.3 tree: **~+10% decode at 16k context**, correct at all depths. Automatic (build-time).

- **Faster reasoning TTFT** — eager `<think>` tag emission cuts reasoning time-to-first-token
  **~610 ms → ~346 ms** on thinking models.

- **Faster cold start** — Metal kernels are prewarmed on server startup.

**Correctness fix:** Gemma4 full-attention layers now use the config's `proportional` RoPE (they
previously used stock RoPE — wrong frequency base + rotation geometry), improving Gemma4 output
quality on every full-attention layer.

**Benchmarks:** a reproducible 6-engine MLX suite (afm vs mlx-vlm, oMLX, LM Studio, rapid-mlx,
ollama) on Qwen3.6-27B-4bit ships in `test-reports/qwen36-supremacy/`. afm is tied for the fastest
MLX decode, leads at long context, and is the fastest configuration measured (MTP, ~23 tok/s).

#### Quick start
```bash
export MACAFM_MLX_MODEL_CACHE=/path/to/models

# MTP (Qwen3.6, needs mtp.safetensors sidecar)
afm mlx -m Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --mtp --port 9999

# EAGLE3 (dense Gemma4-31B + drafter dir)
afm mlx -m mlx-community/gemma-4-31b-it-4bit --eagle3 /path/to/gemma-4-31B-speculator.eagle3 --port 9999
```
Fast paths engage for greedy, non-streaming, text-only requests (else normal AR). Full guide:
`docs/decode-optimizations.md`.

---

## B. Social media copy

### X / Twitter (thread)

**1/**
afm (local MLX LLM server for Apple Silicon) just got speculative decoding ⚡

Qwen3.6-27B with MTP: **+52% decode** — ~23 tok/s on an M4 Pro laptop.
Gemma4-31B with EAGLE3: **+30%, and lossless**.
Same output as greedy. All opt-in flags.

**2/**
We benchmarked it against every MLX server — mlx-vlm, oMLX, LM Studio, rapid-mlx, ollama — same
weights, matched-thermal.

afm is tied for the fastest plain decode, leads at long context, and MTP is the fastest config of
the lot (~23 tok/s, +47%). 📊 [social-decode.png]

**3/**
The EAGLE3 detail we're proud of: afm's verify is **lossless** — full-vocab argmax, bit-exact to
greedy decode. Most engines use an approximate ~32k hot-vocab verify that's ~4% faster but can
diverge from the model's true output.

We chose correctness. 📊 [eagle3-decode.png]

**4/**
Also in this release:
• +10% decode @16k context (backported adaptive 2-pass SDPA)
• reasoning TTFT 610ms → 346ms
• Metal kernel prewarm for faster cold start
• a real Gemma4 RoPE correctness fix found along the way

`afm mlx -m <model> --mtp` / `--eagle3 <drafter>`. That's it.

### LinkedIn / longer post

We added speculative decoding to afm, our OpenAI-compatible MLX inference server for Apple Silicon —
and benchmarked it honestly against every other MLX engine.

Highlights on an M4 Pro (64 GB), 4-bit models:
• **Qwen3.6-27B + MTP: +52% decode** (~23 tok/s), output identical to greedy. On par with the
  reference implementation.
• **Gemma4-31B + EAGLE3: +30% decode, and lossless** — bit-exact to greedy autoregressive output.
• **+10% decode at 16k context** from a backported adaptive 2-pass SDPA kernel.

Two things we care about: every speedup is validated to produce correct output before we report it,
and our EAGLE3 verify is lossless — where other engines trade a few % of exactness for speed via an
approximate hot-vocab verify, afm verifies against the full vocabulary and gives you the model's
true greedy output at speculative speed.

All opt-in: `afm mlx -m <model> --mtp` or `--eagle3 <drafter>`.

### One-liner (release tagline)
> Speculative decoding for Apple Silicon: Qwen3.6 +52%, Gemma4-31B +30% — and lossless.

---

## Assets
- `test-reports/qwen36-supremacy/results-latest/social-decode.png` — 6-engine decode + MTP (+47%) bar chart
- `test-reports/qwen36-supremacy/results-latest/eagle3-decode.png` — EAGLE3 lossless vs approximate
- Full data: `test-reports/qwen36-supremacy/RESULTS-2026-06-06.md`, `RESULTS-EAGLE3-2026-06-07.md`
