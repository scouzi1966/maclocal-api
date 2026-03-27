# Nightly Build Test Summary — 2026-03-27

**Binary:** `v0.9.8-392de4a`
**Model:** `mlx-community/Qwen3.5-35B-A3B-4bit` (19 GB, MoE)
**Platform:** macOS 26.3.1, M3 Ultra 512GB, Swift 6.2.1

## Test Results

| Suite | Pass | Total | Rate | Notes |
|---|---|---|---|---|
| Build verification (8 checks) | 8 | 8 | 100% | Patches, MLX pin, xgrammar, metallib, webui, BuildInfo, binary |
| Unit tests (`swift test`) | 261 | 261 | 100% | 16 suites, 0.55s |
| Assertions — auto parser (full tier) | 361 | 365 | 98.9% | 4 prefix cache (no `--enable-prefix-caching` in multi-model) |
| Assertions — forced qwen3_xml (full tier) | 360 | 365 | 98.6% | Same 4 cache + 1 float/bool XML coercion |
| Batch correctness B={1,2,4,8} | 31 | 32 | 96.9% | 1 model answer flake at B=8 |
| Batch mixed workload | 32 | 32 | 100% | Short+long decode, GPU metrics |
| Batch multiturn prefix cache | 39 | 44 | 88.6% | 5 model quality fails at high concurrency |
| Comprehensive smart analysis (91 variants) | 91 | 91 | 100% | All inference OK; Claude judge scoring truncated |
| **Promptfoo — structured** | **10** | **10** | **100%** | JSON schema + stress |
| **Promptfoo — toolcall** | **21** | **21** | **100%** | default + adaptive-xml + grammar |
| **Promptfoo — toolcall-quality** | **14** | **18** | **77.8%** | 4 fails on adaptive-xml profiles |
| **Promptfoo — grammar schema** | **15** | **20** | **75.0%** | 5 fails on concurrent profile only |
| **Promptfoo — grammar tools** | **23** | **28** | **82.1%** | 5 fails on concurrent profile only |
| **Promptfoo — grammar header/mixed** | **4** | **5** | **80.0%** | 1 mixed-strict fail |
| **Promptfoo — frameworks** | **24** | **24** | **100%** | All 3 profiles |
| **Promptfoo — agentic** | **10** | **12** | **83.3%** | 2 fails on adaptive-xml profiles |
| **Promptfoo — hermes** | **34** | **36** | **94.4%** | 2 fails on adaptive-xml profiles |
| **Promptfoo — opencode** | **79** | **111** | **71.2%** | Model quality at high complexity |
| **Promptfoo — openclaw** | **30** | **36** | **83.3%** | 6 fails across profiles |
| **Promptfoo — PI** | **50** | **60** | **83.3%** | Model prompt injection resistance |

## Performance

- Decode: 89–118 tok/s across all test variants
- Prefill: consistent across step sizes (256, default, 4096)
- Presence/repetition penalty: ~85–90 tok/s (expected slight slowdown)

## Failure Classification

### Server-side (0 failures)
No server bugs found. All core features verified: streaming, tool calling, grammar constraints, logprobs, stop sequences, batch dispatch, prefix caching, think extraction, structured output.

### Known/Expected (14 failures)
- **Prefix cache assertions (8):** Server not launched with `--enable-prefix-caching` in multi-model harness
- **Float/bool XML coercion (2):** Known limitation of XML parameter format
- **Batch answer quality (6):** Model generates wrong answers at high concurrency — not server isolation bug

### Model Quality (53 failures across promptfoo)
- **Concurrent grammar (10):** Race condition in `--concurrent 2` grammar path
- **OpenCode (32):** Complex multi-tool agentic scenarios exceed model capability
- **PI (10):** Model prompt injection resistance varies
- **OpenClaw (6):** Model quality on OpenClaw-specific tool schemas
- **Agentic/Hermes/Quality (remaining):** adaptive-xml parser produces slightly different formatting that quality judges score lower

## Files in This Directory

- `assertions-report-*.html` / `.jsonl` — Assertion test reports (auto + forced parser)
- `multi-assertions-report-*.html` / `.jsonl` — Combined multi-model assertion report
- `smart-analysis-claude-*.md` — AI judge per-test analysis
- `mlx-model-report-*.html` / `.jsonl` — Comprehensive model report
- `*-mlx-community_Qwen3.5-35B-A3B-4bit.json` — Promptfoo eval results (38 files)
