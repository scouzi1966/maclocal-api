# AgentBlaster × afm benchmark harness

Drives the [AgentBlaster](https://github.com/scouzi1966/AgentBlaster) local agentic
benchmark suite against a running **afm** server and reports pass/fail per suite plus
afm's native `/metrics` telemetry (decode tok/s, TTFT, e2e latency, radix prefix‑cache hits).

It exercises the engine-level capabilities AgentBlaster probes — tool calling,
structured/grammar output, prompt-cache reuse, fan-out concurrency, cancellation — and the
larger generated agent-workflow / harness suites (opencode, hermes, codex, cache-replay, …).

## Prerequisites

```bash
# 1. AgentBlaster (provides the `agentblaster` CLI)
git clone https://github.com/scouzi1966/AgentBlaster && pip install -e ./AgentBlaster

# 2. afm — uses the installed Homebrew binary by default (/opt/homebrew/bin/afm), or pass --bin
afm --version

# 3. The model must be in your MLX cache
export MACAFM_MLX_MODEL_CACHE=/path/to/vesta-test-cache
```

## Usage

```bash
cd Tests/agentblaster

# (once, or to refresh) build the larger generated suite set into ./suites/  (~94 cases)
./generate-suites.sh

# run the full generated benchmark against the installed afm (MoE default)
./run-benchmark.sh

# fast capability smoke (the 13 built-in probes, ~19 cases)
./run-benchmark.sh --mode probe --model mlx-community/Qwen3.6-27B-4bit

# both, against a specific binary
./run-benchmark.sh --mode both --bin .build/arm64-apple-macosx/release/afm
```

`run-benchmark.sh` launches afm with the agentic flag set
(`--no-think --enable-prefix-caching --enable-grammar-constraints --tool-call-parser afm_adaptive_xml --concurrent 4`),
registers an `openai`-contract provider pointed at it (with `/metrics`), runs the suites, prints a
summary, and **kills the server on exit** (trap). Per-run artifacts land under
`/tmp/agentblaster-runs/<timestamp>/` (`results.jsonl`, `raw/`, `metrics/prometheus-summary.json`).

Generate an HTML report for any run:

```bash
agentblaster report /tmp/agentblaster-runs/<timestamp>/<run_id> --format html,json
```

### Flags

| flag | default | meaning |
|------|---------|---------|
| `--model` | `mlx-community/Qwen3.6-35B-A3B-4bit` | model id (must be cached) |
| `--mode` | `generated` | `probe` (built-ins) · `generated` (suites/) · `both` |
| `--bin` | `/opt/homebrew/bin/afm` | afm binary (installed stable by default) |
| `--port` | `9999` | afm port |
| `--concurrency` | `1` | client-side concurrent cases (fan-out/concurrency suites use 4) |

## Why `--no-think`

Qwen3.6 is a reasoning model. Without `--no-think` it spends the token budget on
`reasoning_content` and the exact-output benchmark cases fail with empty content
(`finish_reason: length`) — classified `model_quality` but really a config issue. `--no-think`
(fixed in afm v0.9.13 to actually disable thinking server-side) makes the suite runnable.

> ⚠️ Don't pair `--no-think` with very high server `--concurrent` on the short known-answer
> suites — that combination can corrupt batched output. See
> [maclocal-api#140](https://github.com/scouzi1966/maclocal-api/issues/140).

## Interpreting results

AgentBlaster tags every failure with a **`failure_class`** (`model_quality` | engine bug |
feature gap | runtime). **Check it in `results.jsonl` before blaming the engine** — most
failures on hard agent workflows and the structured/json-object path are `model_quality`
(the model's tool/structured decision), not afm bugs.

## Reference results (afm v0.9.13, 2026‑06‑23, M4 Pro)

Generated set, **76/94 (81%)** on `Qwen3.6-35B-A3B-4bit`:

- **harness-cachereplay 48/48** — prompt/prefix caching flawless across warmup/replay/mutation/invalidation
- **harness-concurrency 12/12** (conc=4) — clean fan-out
- weak spots, all `model_quality`: `structured`/`contract-fuzz` json-object output (fails on both 27B and 35B), and the hardest multi-tool agent profiles (codex/cline/hermes)

Throughput contrast (agentic workload): **MoE 35B‑A3B ≈ 85 tok/s decode vs dense 27B ≈ 15.5 tok/s (~5.5×)**.
