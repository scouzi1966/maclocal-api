# feature/mlx-concurrent-batch

Scripts and reports for the concurrent batching feature (Phase 1: interleaving, Phase 2: dense batched decoding).

## Stress Tests

Send N concurrent streaming requests and measure aggregate throughput, per-request tok/s, TTFT, and GPU metrics.

Reports from test runs go in `reports/` (mirrored to private maclocal-api-mlx repo).

## Scripts

### `batch_stress_mactop.py` (recommended)

Uses `mactop --headless --format json` for accurate GPU metrics: active %, power (W), frequency (MHz), temperature, DRAM power, system power.

```bash
python3 batch_stress_mactop.py 1 2 4 6 8
```

### `batch_stress_ioreg.py` (legacy)

Uses `ioreg -r -c AGXAccelerator` for GPU stats. The `Device Utilization %` metric is coarse and unreliable (shows 71-88% while actual GPU active is 97-99%). Kept for reference; prefer `batch_stress_mactop.py`.

```bash
python3 batch_stress_ioreg.py 1 2 4 8
```

## Prerequisites

```bash
pip install aiohttp
brew install mactop        # for batch_stress_mactop.py
```

## Server Setup

```bash
# Release mode, concurrent batching enabled
MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
  afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999 --concurrent
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Agg t/s | Total tokens generated / wall time (all requests combined) |
| Per-req | Average per-request tok/s |
| ms/step | 1000 / per-req tok/s — time per decode step per sequence |
| GPU % | GPU active percentage (mactop) |
| GPU W | GPU power draw in watts |
| DRAM W | DRAM power draw |
| Sys W | Total system power |
| Freq | GPU frequency in MHz |
| Temp | GPU temperature in Celsius |
| TTFT | Time to first token (averaged across requests) |

## Reference Results (M3 Ultra, Qwen3.5-35B-A3B-4bit, 500 max_tokens)

### Current (lazy eval + pipelined dispatch, d070fad)

```
    B   Agg t/s   Per-req   ms/step   GPU %   GPU W   DRAM W   Sys W   Temp    TTFT
    1      90        90      11.1      97%    21.6    17.0    46.8    38°    0.32s
    2     162        81      12.4      99%    34.0    20.5    52.1    39°    0.15s
    4     253        63      15.8      99%    50.4    24.6    60.2    42°    0.24s
    6     299        50      20.0      98%    60.5    25.8    64.3    43°    0.60s
    8     359        45      22.3      97%    62.4    23.4    70.0    45°    0.43s
```

4.0x aggregate throughput at B=8. Serial baseline: ~103 tok/s (B=1 batch overhead ~13%).

### Initial Phase 2 (20fb2e1)

```
    B   Agg t/s   Per-req   ms/step   GPU %   GPU W   DRAM W   Sys W   Temp    TTFT
    1      85        85      11.8      97%    17.7    16.5    47.0    36°    0.31s
    2     152        76      13.1      98%    32.2    19.6    50.6    38°    0.15s
    4     225        56      17.8     100%    45.3    22.9    54.8    40°    0.45s
    6     279        47      21.5     100%    56.0    24.0    62.8    42°    0.35s
    8     315        39      25.4      98%    56.5    21.9    65.2    44°    0.63s
```

GPU power plateaus around 60W at B=6-8 (memory bandwidth limited). Memory constant at ~80 GB regardless of batch size.
