#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  AFM vs mlx-lm  —  Concurrency Benchmark & Graph           ║
║  Fair comparison: both servers with batch concurrency       ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python3 Scripts/benchmark_afm_vs_mlxlm.py              # Full benchmark + graph
    python3 Scripts/benchmark_afm_vs_mlxlm.py --graph       # Re-generate graph from last results
    python3 Scripts/benchmark_afm_vs_mlxlm.py --graph FILE  # Re-generate graph from specific JSON
"""

import asyncio
import aiohttp
import json
import time
import subprocess
import sys
import os
import threading
from pathlib import Path
from datetime import datetime

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"
MODEL_CACHE = "/Volumes/edata/models/vesta-test-cache"
MODEL_LOCAL_PATH = f"{MODEL_CACHE}/{MODEL_ID}"
MAX_TOKENS = 4096
AFM_PORT = 9999
MLX_PORT = 8080
MAX_CONCURRENT = 28          # AFM --concurrent / mlx-lm --decode-concurrency (headroom)
TIMEOUT_PER_REQ = 600        # seconds

# Same concurrency levels for BOTH servers — fair comparison
LEVELS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]

RESULTS_DIR = Path("Scripts/benchmark-results")

# Unique prompts per slot to defeat prompt caching
PROMPTS = [
    "Write an extremely detailed essay about the history of computing from 1800 to 1950, covering every mechanical calculator, early electrical computer, and the people who built them. Include deep technical detail on architecture and design. Continue until stopped.",
    "Write an extremely detailed essay about the history of programming languages from FORTRAN to Rust, covering every major language, its design philosophy, type system, and impact on industry. Continue until stopped.",
    "Write an extremely detailed essay about the entire history of artificial intelligence, from Turing's 1950 paper through modern transformer architectures and LLMs. Include math and architecture details. Continue until stopped.",
    "Write an extremely detailed essay about computer networking from ARPANET through modern cloud and edge computing, covering every protocol, routing algorithm, and architectural pattern. Continue until stopped.",
    "Write an extremely detailed essay about the history of computer graphics from early vector displays to modern real-time ray tracing, covering rendering algorithms, GPU evolution, and game engines. Continue until stopped.",
    "Write an extremely detailed essay about the history of cryptography from Caesar ciphers to post-quantum algorithms, covering RSA, elliptic curves, lattice-based schemes, and real-world deployments. Continue until stopped.",
    "Write an extremely detailed essay about the history of database systems from early ISAM files to modern distributed NewSQL, covering relational theory, indexing, query optimization, and consensus protocols. Continue until stopped.",
    "Write an extremely detailed essay about the history of operating systems from batch monitors to modern microkernels, covering process scheduling, memory management, file systems, and containerization. Continue until stopped.",
    "Write an extremely detailed essay about the history of mobile computing from early PDAs through modern smartphone SoCs, covering ARM architecture, cellular protocols, and mobile OS evolution. Continue until stopped.",
    "Write an extremely detailed essay about web development from Tim Berners-Lee's first browser to modern edge-rendered frameworks, covering HTML, CSS, JavaScript engines, and WebAssembly. Continue until stopped.",
    "Write an extremely detailed essay about robotics from early automatons to modern autonomous vehicles, covering kinematics, SLAM, reinforcement learning, and manufacturing robots. Continue until stopped.",
    "Write an extremely detailed essay about computer hardware from vacuum tubes to modern chiplet architectures, covering transistor scaling, cache hierarchies, and heterogeneous computing. Continue until stopped.",
    "Write an extremely detailed essay about cybersecurity from early computer viruses to modern APTs, covering exploit techniques, defense-in-depth, zero trust, and incident response. Continue until stopped.",
    "Write an extremely detailed essay about digital media from early digital audio to modern streaming platforms, covering compression codecs, content delivery networks, and recommendation algorithms. Continue until stopped.",
    "Write an extremely detailed essay about scientific computing from early numerical methods to modern GPU-accelerated HPC, covering finite elements, Monte Carlo methods, and quantum simulation. Continue until stopped.",
    "Write an extremely detailed essay about search engines from early web crawlers to modern LLM-powered semantic search, covering inverted indices, PageRank, and retrieval-augmented generation. Continue until stopped.",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU Power Monitor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GPUPowerMonitor:
    """Sample GPU power via mactop --headless during benchmark runs."""

    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.results()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["mactop", "--headless", "--format", "json",
                     "--count", "1", "-i", str(self.interval_ms)],
                    timeout=self.interval_ms / 1000 + 5,
                    text=True,
                )
                data = json.loads(out)
                if isinstance(data, list):
                    data = data[0]
                soc = data.get("soc_metrics", {})
                self.samples.append({
                    "gpu_power": soc.get("gpu_power", 0),
                    "system_power": soc.get("system_power", 0),
                    "gpu_freq_mhz": soc.get("gpu_freq_mhz", 0),
                    "gpu_temp": soc.get("gpu_temp", 0),
                    "gpu_usage": data.get("gpu_usage", 0),
                })
            except Exception:
                pass

    def results(self):
        if not self.samples:
            return {"avg_gpu_power": 0, "peak_gpu_power": 0,
                    "avg_system_power": 0, "avg_gpu_usage": 0,
                    "avg_gpu_temp": 0, "n_samples": 0}
        return {
            "avg_gpu_power": round(sum(s["gpu_power"] for s in self.samples) / len(self.samples), 2),
            "peak_gpu_power": round(max(s["gpu_power"] for s in self.samples), 2),
            "avg_system_power": round(sum(s["system_power"] for s in self.samples) / len(self.samples), 2),
            "avg_gpu_usage": round(sum(s["gpu_usage"] for s in self.samples) / len(self.samples), 1),
            "avg_gpu_temp": round(sum(s["gpu_temp"] for s in self.samples) / len(self.samples), 1),
            "n_samples": len(self.samples),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hardware Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_hardware_info():
    chip = "Apple Silicon"
    mem_gb = 0
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
        for line in sp.split("\n"):
            if "Chip" in line and ":" in line:
                chip = line.split(":")[1].strip()
            if "Memory" in line and ":" in line:
                val = line.split(":")[1].strip()
                if "GB" in val:
                    mem_gb = int(val.replace("GB", "").strip())
                elif "TB" in val:
                    mem_gb = int(float(val.replace("TB", "").strip()) * 1024)
    except Exception:
        pass
    return chip, mem_gb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streaming Request
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def send_one(session, base_url, model, max_tokens, prompt, rid):
    """Send one streaming request. Uses server-reported usage for token counts."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    start = time.perf_counter()
    completion_tokens = 0
    prompt_tokens = 0
    n_chunks = 0  # fallback counter for servers that don't send usage in streaming
    content_parts = []    # capture response text
    reasoning_parts = []  # capture reasoning text

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"error": f"HTTP {resp.status}: {body[:200]}", "rid": rid}

            buf = ""
            async for raw in resp.content:
                buf += raw.decode()
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage:
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            c = delta.get("content", "")
                            r = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                            if c:
                                content_parts.append(c)
                                n_chunks += 1
                            if r:
                                reasoning_parts.append(r)
                                n_chunks += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        return {"error": str(e), "rid": rid}
    end = time.perf_counter()

    # Prefer server-reported usage; fall back to chunk count
    tokens = completion_tokens if completion_tokens > 0 else n_chunks
    wall = end - start
    tps = tokens / wall if wall > 0 and tokens > 0 else 0

    return {
        "rid": rid,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": tokens,
        "wall": round(wall, 3),
        "tps": round(tps, 1),
        "token_source": "usage" if completion_tokens > 0 else "chunks",
        "response_text": "".join(content_parts),
        "reasoning_text": "".join(reasoning_parts),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concurrent Benchmark
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_benchmark(base_url, model, n_concurrent, max_tokens):
    """Fire n_concurrent simultaneous requests, return aggregate metrics."""
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_PER_REQ)
    conn = aiohttp.TCPConnector(limit=n_concurrent + 4)

    gpu_mon = GPUPowerMonitor(interval_ms=1000)
    gpu_mon.start()

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_concurrent)]

        wall_start = time.perf_counter()
        tasks = [
            send_one(session, base_url, model, max_tokens, prompts[i], i)
            for i in range(n_concurrent)
        ]
        results = await asyncio.gather(*tasks)
        wall_end = time.perf_counter()

    gpu_stats = gpu_mon.stop()

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    wall_time = wall_end - wall_start
    total_tokens = sum(r["completion_tokens"] for r in valid)
    agg_tps = total_tokens / wall_time if wall_time > 0 else 0
    avg_tps = sum(r["tps"] for r in valid) / len(valid) if valid else 0

    out = {
        "concurrent": n_concurrent,
        "wall_time": round(wall_time, 2),
        "total_tokens": total_tokens,
        "aggregate_tps": round(agg_tps, 1),
        "avg_per_request_tps": round(avg_tps, 1),
        "gpu": gpu_stats,
        "n_valid": len(valid),
        "n_errors": len(errors),
        "per_request": valid,
    }
    if errors:
        out["errors"] = [e.get("error", "unknown") for e in errors]
    return out


async def warmup(base_url, model):
    """Multi-round warmup: triggers MLX kernel JIT compilation for various seq lengths."""
    warmup_specs = [
        ("Say hello.", 16),
        ("Write a paragraph about the weather today.", 128),
        ("Write a detailed paragraph about the history of computers.", 512),
        ("Write a long essay about artificial intelligence and its impact on society.", 1024),
        ("Explain quantum computing in detail.", 512),
    ]
    n = len(warmup_specs)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, (prompt, mt) in enumerate(warmup_specs):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": mt,
                "stream": False,
                "temperature": 0.7,
            }
            try:
                async with session.post(
                    f"{base_url}/v1/chat/completions", json=payload
                ) as resp:
                    data = await resp.json()
                    tokens = data.get("usage", {}).get("completion_tokens", "?")
                    print(f"    warmup {i+1}/{n}: {tokens} tokens (max {mt})")
            except Exception as e:
                print(f"    warmup {i+1}/{n} failed: {e}")
        await asyncio.sleep(3)


async def get_model_id(base_url):
    """Find our model in the /v1/models list. Match by MODEL_ID or MODEL_LOCAL_PATH."""
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base_url}/v1/models") as resp:
                data = await resp.json()
                models = data.get("data", [])
                for m in models:
                    mid = m.get("id", "")
                    if MODEL_ID in mid or MODEL_LOCAL_PATH in mid:
                        return mid
                # If no match, return last one (most likely the one we loaded)
                if models:
                    return models[-1].get("id", MODEL_ID)
    except Exception:
        pass
    return MODEL_ID


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Server Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def wait_for_server(url, timeout=300):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


def start_afm(port, concurrent):
    """Start AFM with --concurrent for batch mode."""
    env = {**os.environ, "MACAFM_MLX_MODEL_CACHE": MODEL_CACHE}
    cmd = [
        "afm", "mlx",
        "-m", MODEL_ID,
        "--port", str(port),
        "--concurrent", str(concurrent),
    ]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def start_mlx_lm(port, concurrency):
    """Start mlx-lm with local model path and both concurrency flags matching AFM."""
    env = {**os.environ}
    cmd = [
        "mlx_lm.server",
        "--model", MODEL_LOCAL_PATH,
        "--port", str(port),
        "--decode-concurrency", str(concurrency),
        "--prompt-concurrency", str(concurrency),
        "--max-tokens", str(MAX_TOKENS),
    ]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def stop_server(proc, name="server"):
    if proc and proc.poll() is None:
        print(f"  Stopping {name}...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # Wait for GPU memory release
        time.sleep(10)
        print(f"  {name} stopped (10s GPU cooldown)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_graph(afm_data, mlx_data, hw_info, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np

    # ── Data ──
    afm_x = [r["concurrent"] for r in afm_data]
    afm_y = [r["aggregate_tps"] for r in afm_data]
    afm_gpu = [r.get("gpu", {}).get("avg_gpu_power", 0) for r in afm_data]
    mlx_x = [r["concurrent"] for r in mlx_data]
    mlx_y = [r["aggregate_tps"] for r in mlx_data]
    mlx_gpu = [r.get("gpu", {}).get("avg_gpu_power", 0) for r in mlx_data]

    chip, mem_gb = hw_info

    # Best points
    afm_opt_idx = int(np.argmax(afm_y))
    afm_peak = afm_y[afm_opt_idx]
    afm_peak_x = afm_x[afm_opt_idx]

    mlx_peak = max(mlx_y) if mlx_y else 0
    mlx_peak_x = mlx_x[int(np.argmax(mlx_y))] if mlx_y else 1

    # Speedup at matching concurrency levels
    speedup_at_peak = afm_peak / mlx_peak if mlx_peak > 0 else 0

    # ── Theme ──
    BG      = "#0d1117"
    FG      = "#c9d1d9"
    GRID    = "#21262d"
    AFM_C   = "#58a6ff"
    AFM_F   = "#1f6feb"
    MLX_C   = "#f97583"
    GREEN   = "#7ee787"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "grid.color": GRID,
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "SF Pro Display", "DejaVu Sans"],
    })

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.92, top=0.84, bottom=0.22)

    # ── mlx-lm line ──
    ax.plot(mlx_x, mlx_y, color=MLX_C, lw=3, ls="-",
            marker="D", ms=9, markerfacecolor="white",
            markeredgecolor=MLX_C, markeredgewidth=2, zorder=4,
            label=f"mlx-lm v0.31.1  (concurrency {MAX_CONCURRENT})")
    ax.plot(mlx_x, mlx_y, color=MLX_C, lw=8, alpha=0.10, zorder=3)
    ax.fill_between(mlx_x, 0, mlx_y, color=MLX_C, alpha=0.06, zorder=1)

    # mlx-lm data labels (with GPU watts)
    for x, y, w in zip(mlx_x, mlx_y, mlx_gpu):
        label = f"{y:.0f}" if w == 0 else f"{y:.0f} (@{w:.0f}W)"
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(0, -20),
            ha="center", fontsize=9, color=MLX_C, alpha=0.85,
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
        )

    # ── AFM line ──
    ax.plot(afm_x, afm_y, color=AFM_C, lw=3.5,
            marker="o", ms=10, markerfacecolor="white",
            markeredgecolor=AFM_C, markeredgewidth=2.5, zorder=5,
            label=f"AFM v0.9.7  (--concurrent {MAX_CONCURRENT})")
    ax.plot(afm_x, afm_y, color=AFM_C, lw=10, alpha=0.12, zorder=3)
    ax.fill_between(afm_x, 0, afm_y, color=AFM_F, alpha=0.12, zorder=1)

    # AFM data labels (with GPU watts)
    for x, y, w in zip(afm_x, afm_y, afm_gpu):
        if x == afm_peak_x:
            continue
        label = f"{y:.0f}" if w == 0 else f"{y:.0f} (@{w:.0f}W)"
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(0, 16),
            ha="center", fontsize=9, color=AFM_C, alpha=0.85,
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
        )

    # ── AFM optimal highlight ──
    afm_peak_gpu = afm_gpu[afm_opt_idx] if afm_opt_idx < len(afm_gpu) else 0
    peak_label = f"{afm_peak:.0f} tok/s" if afm_peak_gpu == 0 else f"{afm_peak:.0f} tok/s (@{afm_peak_gpu:.0f}W)"
    ax.plot(afm_peak_x, afm_peak, "o", color=GREEN, ms=18,
            markeredgecolor="white", markeredgewidth=2.5, zorder=6)
    ax.annotate(
        peak_label, (afm_peak_x, afm_peak),
        textcoords="offset points", xytext=(0, 22),
        ha="center", fontsize=13, fontweight="bold", color=GREEN,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
    )



    # ── Axes ──
    all_x = sorted(set(afm_x + mlx_x))
    ax.set_xlabel("Concurrent Requests", fontsize=15, fontweight="bold", labelpad=12, color=FG)
    ax.set_ylabel("Aggregate Throughput  (tok/s)", fontsize=15, fontweight="bold", labelpad=12, color=FG)
    ax.set_xticks(all_x)
    ax.grid(True, alpha=0.3, ls="-")
    ax.set_xlim(0, max(all_x) + 4)
    all_y = afm_y + mlx_y
    ax.set_ylim(0, max(all_y) * 1.20)
    ax.tick_params(labelsize=13, colors=FG)

    # ── Title (above axes border) ──
    model_short = MODEL_ID.split("/")[-1]
    ax.set_title(
        f"Batched Concurrency  ·  {model_short}  ·  {MAX_TOKENS} max tokens  ·  {chip}  ·  {mem_gb} GB",
        fontsize=12, color=FG, alpha=0.55, pad=10, loc="left",
    )

    # ── Header (top of figure) ──
    fig.text(0.07, 0.94, "AFM v0.9.7", fontsize=30, fontweight="bold",
             color=AFM_C, transform=fig.transFigure,
             path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    fig.text(0.25, 0.95, "vs", fontsize=22, color=FG,
             transform=fig.transFigure)
    fig.text(0.30, 0.94, "mlx-lm v0.31.1", fontsize=30, fontweight="bold",
             color=MLX_C, transform=fig.transFigure)

    # ── Legend ──
    leg = ax.legend(loc="lower right", fontsize=12,
                    frameon=True, fancybox=True, framealpha=0.4,
                    edgecolor=GRID, facecolor=BG)
    for t in leg.get_texts():
        t.set_color(FG)

    # ── Footer: CLI commands, prompt, date ──
    afm_cmd = f"MACAFM_MLX_MODEL_CACHE={MODEL_CACHE} afm mlx -m {MODEL_ID} --concurrent {MAX_CONCURRENT}"
    mlx_cmd = f"mlx_lm.server --model {MODEL_LOCAL_PATH} --decode-concurrency {MAX_CONCURRENT} --prompt-concurrency {MAX_CONCURRENT} --max-tokens {MAX_TOKENS}"
    prompt_preview = PROMPTS[0][:120] + "..."
    date_str = datetime.now().strftime("%Y-%m-%d")

    footer_lines = [
        f"AFM:      $ {afm_cmd}",
        f"mlx-lm:  $ {mlx_cmd}",
        f"Prompt:   \"{prompt_preview}\"",
        f"Date: {date_str}  |  github.com/scouzi1966/maclocal-api",
    ]
    for i, line in enumerate(footer_lines):
        fig.text(0.07, 0.12 - i * 0.028, line,
                 fontsize=7.5, color=FG, alpha=0.4, family="monospace",
                 transform=fig.transFigure)

    # ── Save ──
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"\n  Graph saved: {output_path}")
    plt.close()

    return {
        "afm_peak_concurrent": afm_peak_x,
        "afm_peak_tps": afm_peak,
        "mlx_peak_tps": mlx_peak,
        "mlx_peak_concurrent": mlx_peak_x,
        "speedup": speedup_at_peak,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmark Phase
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_phase(name, base_url, model, levels, store_key, store, results_file):
    """Run benchmark at all concurrency levels for a server."""
    model_name = await get_model_id(base_url)
    print(f"  Model ready: {model_name}")

    # Wait 60s for model to fully settle in GPU memory
    print("  Waiting 60s for model to fully load and settle...")
    await asyncio.sleep(60)

    print("  Warming up (3 rounds — triggering kernel JIT)...")
    await warmup(base_url, model_name)

    for n in levels:
        print(f"\n  >> {n} concurrent request{'s' if n != 1 else ''} ...")
        r = await run_benchmark(base_url, model_name, n, MAX_TOKENS)
        store[store_key].append(r)

        gpu = r.get('gpu', {})
        print(f"     agg {r['aggregate_tps']:>7.1f} tok/s  "
              f"| per-req {r['avg_per_request_tps']:>6.1f} tok/s  "
              f"| wall {r['wall_time']:>6.1f}s  "
              f"| tokens {r['total_tokens']}  "
              f"| GPU {gpu.get('avg_gpu_power', 0):.1f}W (peak {gpu.get('peak_gpu_power', 0):.1f}W)")
        if r["n_errors"]:
            print(f"     ⚠ {r['n_errors']} error(s): {r.get('errors', [])}")

        with open(results_file, "w") as f:
            json.dump(store, f, indent=2)

        await asyncio.sleep(3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"concurrency-benchmark-{ts}.json"

    # ── --graph mode ──
    if len(sys.argv) > 1 and sys.argv[1] == "--graph":
        src = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        if src is None:
            files = sorted(RESULTS_DIR.glob("concurrency-benchmark-*.json"))
            if not files:
                print("No results found. Run the benchmark first.")
                sys.exit(1)
            src = files[-1]
        print(f"Loading: {src}")
        with open(src) as f:
            data = json.load(f)
        hw = (data.get("hardware", {}).get("chip", "Apple Silicon"),
              data.get("hardware", {}).get("memory_gb", 0))
        gpath = RESULTS_DIR / f"concurrency-benchmark-{ts}.png"
        stats = generate_graph(data["afm"], data.get("mlx_lm", []), hw, gpath)
        print(f"  AFM peak:  {stats['afm_peak_concurrent']} concurrent "
              f"→ {stats['afm_peak_tps']:.0f} tok/s")
        print(f"  mlx-lm peak: {stats['mlx_peak_concurrent']} concurrent "
              f"→ {stats['mlx_peak_tps']:.0f} tok/s")
        print(f"  Speedup: {stats['speedup']:.1f}x")
        return

    # ── Hardware ──
    chip, mem_gb = get_hardware_info()
    print(f"\n{'═' * 64}")
    print(f"  AFM vs mlx-lm  —  Fair Concurrency Benchmark")
    print(f"  {chip} · {mem_gb} GB · {MODEL_ID.split('/')[-1]}")
    print(f"  {MAX_TOKENS} max tokens · {len(LEVELS)} levels · both with batch decode")
    print(f"{'═' * 64}\n")

    store = {
        "timestamp": ts,
        "model": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "hardware": {"chip": chip, "memory_gb": mem_gb},
        "afm_version": "0.9.7",
        "mlx_lm_version": "0.31.1",
        "afm_flags": f"--concurrent {MAX_CONCURRENT}",
        "mlx_lm_flags": f"--decode-concurrency {MAX_CONCURRENT} --prompt-concurrency {MAX_CONCURRENT} --max-tokens {MAX_TOKENS}",
        "afm": [],
        "mlx_lm": [],
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — mlx-lm FIRST (fresh GPU, no residual memory from AFM)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{'─' * 64}")
    print(f"  Phase 1: mlx-lm v0.31.1  (--decode-concurrency {MAX_CONCURRENT})")
    print(f"  Running FIRST for fresh GPU — no residual memory pressure")
    print(f"{'─' * 64}")

    mlx_proc = start_mlx_lm(MLX_PORT, MAX_CONCURRENT)
    mlx_url = f"http://127.0.0.1:{MLX_PORT}"

    try:
        print("  Waiting for model load...")
        if not wait_for_server(mlx_url, timeout=300):
            print("  ERROR: mlx-lm did not start in time.")
            stop_server(mlx_proc, "mlx-lm")
            sys.exit(1)

        await run_phase("mlx-lm", mlx_url, MODEL_ID, LEVELS,
                       "mlx_lm", store, results_file)
    finally:
        stop_server(mlx_proc, "mlx-lm")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — AFM (after mlx-lm released GPU)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 64}")
    print(f"  Phase 2: AFM v0.9.7  (--concurrent {MAX_CONCURRENT})")
    print(f"{'─' * 64}")

    afm_proc = start_afm(AFM_PORT, MAX_CONCURRENT)
    afm_url = f"http://127.0.0.1:{AFM_PORT}"

    try:
        print("  Waiting for model load...")
        if not wait_for_server(afm_url, timeout=300):
            print("  ERROR: AFM did not start in time.")
            stop_server(afm_proc, "AFM")
            sys.exit(1)

        await run_phase("AFM", afm_url, MODEL_ID, LEVELS,
                       "afm", store, results_file)
    finally:
        stop_server(afm_proc, "AFM")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Graph
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 64}")
    print(f"  Phase 3: Generating Graph")
    print(f"{'─' * 64}")

    graph_path = RESULTS_DIR / f"concurrency-benchmark-{ts}.png"
    stats = generate_graph(store["afm"], store["mlx_lm"],
                           (chip, mem_gb), graph_path)

    print(f"\n{'═' * 64}")
    print(f"  RESULTS")
    print(f"{'═' * 64}")
    print(f"  AFM peak:     {stats['afm_peak_concurrent']} concurrent "
          f"→ {stats['afm_peak_tps']:.0f} tok/s")
    print(f"  mlx-lm peak:  {stats['mlx_peak_concurrent']} concurrent "
          f"→ {stats['mlx_peak_tps']:.0f} tok/s")
    print(f"  Speedup:      {stats['speedup']:.1f}x")
    print(f"  Results:      {results_file}")
    print(f"  Graph:        {graph_path}")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    asyncio.run(main())
