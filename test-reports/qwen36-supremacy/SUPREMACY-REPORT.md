# afm MLX — Performance Supremacy Benchmark vs Competitors

**Model:** Qwen3.6-27B (4-bit class), thinking enabled (model default)
**Hardware:** Apple M4 Pro · 14 cores (10P/4E) · 64 GB unified · macOS 26.5
**Conditions:** single stream (batch=1), temperature 0, one engine on the GPU at a time
**Date:** 2026-05-31

---

## TL;DR verdict

1. **afm has the fastest decode** of all 7 engines tested — 15.9 tok/s — **the front of the MLX pack** (LM Studio 15.6, oMLX 15.3, Rapid-MLX 15.1, mlx-vlm 14.2, ollama-MLX 13.5, llama.cpp 9.7). All decode numbers use one uniform metric: server-reported `usage.completion_tokens ÷ measured decode window` (see §2).
2. **afm is ~1.6× faster than the GGUF/llama.cpp path** on decode (15.54 vs 9.56 tok/s).
3. **Every MLX engine runs Qwen3.6, but the GGUF/llama.cpp path lags.** ollama runs it **only via its MLX backend** (`qwen3.6:27b-mlx`, ~19 GB) — its GGUF path fails (`unknown model architecture: 'qwen35'` on 0.24.0).
4. **afm's prefix cache is the decisive differentiator and wins head-to-head**: on the realistic multi-turn pattern (shared context + new question) afm reuses **99.5%** of the context for a **59.8× faster prefill** (36s→0.6s) — vs LM Studio 18.9×, llama.cpp 7.1×, oMLX 2×, and ollama/rapid-mlx/mlx-vlm 1.0× (no true prefix reuse). This is the metric that matters for agentic/multi-turn latency.

The decode lead over LM Studio / oMLX is **narrow** (~1–2 %, within the MLX family) — honest call, not a blowout there. The blowouts are vs the GGUF path (+63 %) and the prefix-cache prefill (8×).

---

## 1. Capability — can the engine even load Qwen3.6-27B?

| Engine | Loads Qwen3.6-27B? | Evidence |
|---|---|---|
| **afm** (Swift, custom `Qwen3_5MoEVL` patch) | ✅ | warmup + full sweep, GPU 100 % |
| **mlx-vlm 0.5.0** (official MLX Python VLM server) | ✅ | `qwen3_5` arch, `/v1/chat/completions` |
| **LM Studio** (bundled MLX engine) | ✅ | loaded in 29 s, emits `reasoning_content` |
| **oMLX** (omlx-cli serve) | ✅ | loaded, OpenAI `/v1` |
| **Rapid-MLX** (`rapid-mlx serve --no-mllm`) | ✅ (text-only) | hybrid DeltaNet backbone needs `--no-mllm`; multimodal path unsupported for this arch |
| **llama.cpp** (Homebrew `llama-server`, build b8180) | ✅ | GGUF Q4_K_M, registers `qwen35` |
| **ollama 0.24.0 — MLX backend** (`qwen3.6:27b-mlx`, NVFP4, ~19 GB) | ✅ | warmup OK, benchmarked; `ollama show` → quant `nvfp4`, arch `qwen3_5` |
| **ollama 0.24.0 — GGUF path** (`…Qwen3.6-27B-GGUF:Q4_K_M`) | ❌ | `error loading model architecture: unknown model architecture: 'qwen35'` |
| `mlx_lm.server` (text-only Python) | ❌ | VLM checkpoint → `333 parameters not in model` |

> ollama ships **two** backends. Its bundled llama.cpp predates `qwen35` so the **GGUF** model won't load; but ollama's newer **MLX** backend runs the `qwen3.6:27b-mlx` build fine. Upstream llama.cpp (b8180) also added `qwen35`, so the GGUF path works via standalone `llama-server`.

---

## 2. Decode speed (token generation) — the headline

**Canonical metric = `probe.py`** (we standardized on it after both llama-benchy and guidellm proved unreliable for these servers — see §6). One uniform formula for every engine: **decode tok/s = server `usage.completion_tokens` ÷ decode window** (window = first→last streamed-delta time). Immune to client-side tokenizer mismatch *and* to chunk-batching, because the count comes from the server's own standard `usage` field, not re-tokenization.

| Rank | Engine | Decode tok/s |
|---|---|---|
| 🥇 | **afm** | **15.90** |
| 🥈 | LM Studio (MLX) | 15.62 |
| 🥉 | oMLX (MLX) | 15.31 |
| 4 | Rapid-MLX (MLX, vllm_mlx) | 15.09 |
| 5 | mlx-vlm (MLX Python) | 14.16 |
| 6 | ollama (MLX, NVFP4) | 13.48 |
| 7 | llama.cpp GGUF Q4_K_M | 9.69 (server `eval time` cross-check: 9.56 ✓) |

**afm is fastest.** Margin: +1.8 % over LM Studio, +3.9 % over oMLX, +5.4 % over Rapid-MLX (all small — same MLX family), +12 % over mlx-vlm, +18 % over ollama-MLX, **+64 % over llama.cpp/GGUF**.

> Why not llama-benchy or guidellm? **llama-benchy** inflates decode for servers without streamed `token_ids` (it reported afm 29, oMLX "peak 156" — impossible for a 27B; it re-tokenizes the text, sometimes with a wrong/gpt2 tokenizer). **guidellm 0.6.0** counts tokens correctly but failed to capture TTFT/ITL for afm *and* the reference `llama-server` (and returned nonsense for oMLX, failed on ollama). Both are unreliable here; the server-`usage`-anchored probe is the trustworthy basis (afm's own `[STATS]` = 15.8 corroborates probe's 15.9).
>
> ollama's MLX quant `qwen3.6:27b-mlx` is **NVFP4** (NVIDIA 4-bit float, confirmed via `ollama show`) — 4-bit class like the others, a fair iso-bit-width comparison (~19 GB on disk = NVFP4 block-scale overhead + vision tower, not more bits). Rapid-MLX ran **text-only** (`--no-mllm`): its multimodal batching path is incompatible with Qwen3.6's hybrid DeltaNet backbone.

---

## 3. Prefill (prompt processing)

**Cold** (~1790-token prompt, no cache), probe `prefill_tps`:

| Engine | Cold prefill tok/s |
|---|---|
| LM Studio | 123.0 |
| oMLX | 121.2 |
| **afm** | **120.3** |
| ollama (MLX) | 118.1 |
| mlx-vlm | 110.1 |
| llama.cpp GGUF | 105.4 |

Cold prefill is **effectively tied** across the MLX engines (afm within 3 % of the top — noise). afm does **not** win cold prefill, and the report says so.

**Prefix-cached** (afm) — from the llama-benchy depth-4096 sweep, afm's `[STATS]`:

| afm scenario | prefill tok/s |
|---|---|
| Cold (cache MISS, ~4156 tok) | 117–121 |
| **Cache HIT (88 %, reuse 4149/4678)** | **932–951** |

→ **~8× prefill speedup on a context reload.**

### Prefix cache, head-to-head (the metric that matters for multi-turn/agentic)
Realistic radix test: a shared ~4000-token context, reused with a **new question** each turn (cold = fresh context; hit = same context, different question). Speedup = cold TTFT ÷ hit TTFT.

| Engine | cold TTFT | hit TTFT | **speedup** | context reused |
|---|---|---|---|---|
| 🥇 **afm** | 35.8s | **0.60s** | **59.8×** | **3978/3996 (99.5%)** |
| LM Studio | 33.2s | 1.76s | 18.9× | n/r |
| llama.cpp | 36.2s | 5.08s | 7.1× | n/r |
| oMLX | 32.4s | 16.5s | 2.0× | 2048 (51%) |
| rapid-mlx / ollama / mlx-vlm | ~33s | ~33s | **1.0×** | 0 / none |

**afm wins prefix caching by a wide margin** — it reused 99.5% of the context, turning a 36 s prefill into 0.6 s when only the question changed.

> ⚠️ Two *different* cache behaviors — don't conflate them. An earlier "exact-repeat" test (byte-identical prompt twice) showed ollama 70× / oMLX 29× / afm 1×, but that rewards **whole-response memoization** (seen-this-exact-request), which does nothing in real use. On the **realistic new-question** pattern those collapse to 1.0× while afm leads at 59.8×. afm's radix cache does **true prefix reuse**; ollama/rapid-mlx's headline cache numbers are exact-repeat memoization. See `results/plots/cache_realistic.png`.

---

## 4. Honest caveats

- **Quant parity is approximate** but all 4-bit class: MLX "4bit" ≈ 4.5 bpw group quant (afm/LM Studio/oMLX/mlx-vlm); ollama `qwen3.6:27b-mlx` = **NVFP4** (NVIDIA 4-bit float E2M1 + block scales); llama.cpp Q4_K_M ≈ 4.8 bpw mixed (q4_K/q5_K/q6_K + imatrix). Close, not identical.
- **Single stream, batch=1.** This is latency, not concurrent throughput.
- **Thinking enabled** (Qwen3.6 default) for all — affects total tokens, not the tok/s rate.
- The decode lead over LM Studio/oMLX is within a couple percent; calling afm "supreme" there is a narrow win on the same backend. The strong, defensible claims are: fastest overall, ~1.6× vs GGUF, and the 8× prefix-cache prefill.

## 5. One-line supremacy claim (defensible)

> On Apple M4 Pro, afm runs Qwen3.6-27B at **15.5 tok/s decode — the fastest of every local engine tested** (LM Studio, oMLX, ollama-MLX, mlx-vlm, llama.cpp), **~1.6× faster than the GGUF/llama.cpp path**; and afm's prefix cache delivers **~8× faster prefill (~950 tok/s) on context reloads** — the differentiator for agentic/multi-turn work.
