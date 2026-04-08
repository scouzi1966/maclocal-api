# AFM Concurrent-Throughput Demo

A marketing-ready dual-Y-axis video showing AFM's concurrent inference
performance in real time: number of live connections (left axis, amber)
and aggregate token generation rate (right axis, cyan) as virtual users
ramp from 0 to N and hold.

## Layout

```
Scripts/
├── demo-concurrent-throughput.sh        # entry point
└── demo/
    ├── concurrent_load_driver.py        # async virtual-user ramp + metrics
    ├── render_demo_video.py             # matplotlib → MP4 renderer
    ├── prompts/
    │   ├── general.json                 # 100 varied short-form prompts
    │   └── prefix_cache.json            # shared ~1800-token system prompt
    │                                    #   + 25 short user variations
    └── out/                             # trace.jsonl, summary, MP4 land here
```

## Quick start

```bash
# Defaults: Qwen3.5-35B-A3B-4bit, 200 concurrent, 45s ramp, 75s hold,
#           general prompts, prefix cache enabled, port 9998.
./Scripts/demo-concurrent-throughput.sh
```

The script will:

1. Start `afm mlx -m <model> --concurrent N --enable-prefix-caching --port 9998`
2. Wait for the model to load + run a single tiny warmup request
3. Ramp virtual users 0 → N over the ramp window, hold at N, then stop
4. Sample metrics every 250ms into `Scripts/demo/out/trace.jsonl`
5. Render `Scripts/demo/out/concurrent_demo.mp4` (16:9, 1920×1080, 30 fps)
6. Stop the server (unless `--keep-server` is passed)

Expected wall time: `ramp + hold + ~60s` (model load + render).

## Typical variations

**Larger model, same shape:**
```bash
./Scripts/demo-concurrent-throughput.sh \
    --model mlx-community/gemma-4-31b-it-8bit
```

**Prefix-cache stress profile** — every request shares an ~1800-token system
prompt, so the radix tree should reuse the prefix KV across every request
after the first. Compare the aggregate tok/s vs the general profile to
visualize the caching win.

```bash
./Scripts/demo-concurrent-throughput.sh \
    --mode prefix-cache \
    --title "Prefix cache at 200 connections"
```

**Without prefix cache** — baseline comparison for the same workload.

```bash
./Scripts/demo-concurrent-throughput.sh \
    --no-prefix-cache \
    --title "No prefix cache baseline"
```

**Longer ramp, lighter load** — smoother visual, easier to read on a phone.

```bash
./Scripts/demo-concurrent-throughput.sh \
    --concurrent 100 \
    --ramp 60 --hold 60
```

**Re-render an existing trace** — tweak titles or duration cap without
re-running the server.

```bash
./Scripts/demo-concurrent-throughput.sh --skip-load \
    --title "v0.9.10 · 200 concurrent users"
```

**Run load but skip the render** — useful for pure benchmark data.

```bash
./Scripts/demo-concurrent-throughput.sh --skip-render
```

## All options

```
--model MODEL              MLX model ID or local path
                           (default: mlx-community/Qwen3.5-35B-A3B-4bit)
--concurrent N             Peak virtual users AND --concurrent slots   (200)
--ramp S                   Seconds to ramp from 0 to N                 (45)
--hold S                   Seconds to hold at N after ramp             (75)
--mode {general,prefix-cache,mixed}                                    (general)
--max-tokens N             Per-request max_tokens                      (192)
--temperature F            Sampling temperature                        (0.7)
--top-p F                  Top-p; 1.0 avoids the TopPSampler 1D path   (1.0)
--port N                   Server port                                 (9998)
--title STR                Video title (headline)
--subtitle STR             Video subtitle (otherwise auto-built)
--output PATH              Output mp4 path
--cache PATH               Model cache (MACAFM_MLX_MODEL_CACHE)
--no-prefix-cache          Disable --enable-prefix-caching on the server
--skip-render              Run load only, skip video
--skip-load                Skip load, re-render existing trace
--keep-server              Leave the server running after the demo
```

## What each output file contains

- `out/trace.jsonl` — one JSON line per 250 ms sample:
  ```json
  {"t": 12.25, "active": 55, "inflight": 54, "agg_tps": 412, "completed": 83, "errors": 0}
  ```
  - `t`         — seconds since ramp start
  - `active`    — virtual users the ramp controller currently wants running
  - `inflight`  — requests actively streaming (gap vs `active` = queued in AFM slot scheduler)
  - `agg_tps`   — aggregate completion tokens observed in the last 1 s window
  - `completed` — cumulative completed requests
  - `errors`    — cumulative errored requests

- `out/trace.summary.json` — wall time, totals, and the driver cfg used.
- `out/concurrent_demo.mp4` — 16:9 1920×1080 30 fps video, ~2 minutes,
  ~3 MB with H.264 + yuv420p + `+faststart` (optimized for Twitter/X, LinkedIn, Bluesky).

## Prompts

- **`prompts/general.json`** — 100 varied short-form user prompts across
  science, code, math, language, creative, trivia, and practical topics.
  Each virtual user picks one at random per request.

- **`prompts/prefix_cache.json`** — one shared `system` field (~1800 tokens)
  describing a fictional customer-support product ("OrbitDesk") plus 25
  short `user_prompts`. With `--enable-prefix-caching`, every request after
  the first should hit the radix cache on the system portion, bringing
  prefill down dramatically. Designed to visualize the caching benefit.

Edit either JSON file to change the prompt pool — no script changes needed.

## Visual design

Dark theme (`#0B0F17` background), two contrasting accent lines
(amber `#F59E0B` = connections, cyan `#22D3EE` = throughput), both
with soft translucent fills, live large-type numeric readouts in the
upper-right, grid aligned to the left (connections) axis. Title block
top-left, footer with branding bottom-left/right. Exported at 1920×1080
so the same MP4 works cleanly for Twitter/X, LinkedIn, Bluesky, YouTube
Shorts (no re-encoding needed).

## Prerequisites

- `afm` binary built (`swift build -c release`) or on PATH
- Python 3: `aiohttp`, `matplotlib`, `numpy`
- `ffmpeg` on PATH (otherwise renders to animated GIF)
- Model weights in `MACAFM_MLX_MODEL_CACHE` (default:
  `/Volumes/edata/models/vesta-test-cache`)

```bash
pip install aiohttp matplotlib numpy
brew install ffmpeg
```

## Repeatability

Every run is fully parameterized — change `--model`, `--concurrent`,
`--ramp`, `--hold`, `--mode`, `--max-tokens`, `--temperature`, `--top-p`
on the command line and the driver uses those exact values. Prompt pools
are edited in the JSON files (no code changes). Traces and renders are
reproducible given the same cfg, modulo sampling noise from `temperature > 0`
and real wall-clock timing variance under load.
