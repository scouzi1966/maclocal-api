#!/usr/bin/env python3
"""
Live monitor for the concurrent-throughput demo.

Tails `trace.jsonl` as the driver writes to it and shows a live-updating
matplotlib window using the same dark dual-Y-axis theme as the final MP4.
Run this in a SECOND TERMINAL while the driver is running — or launch it
via the bash wrapper's --watch flag and it will background itself.

Usage:
    python3 Scripts/demo/watch_live.py
    python3 Scripts/demo/watch_live.py --trace /path/to/trace.jsonl

Ctrl-C to quit. The window closes on its own when the driver stops writing
new samples for > STALL_SECONDS seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib

# Prefer the native macOS interactive backend. Fall back to TkAgg if not
# available (e.g. Linux). On a headless machine there is nothing this
# script can do — it needs a display.
_BACKENDS = ["MacOSX", "TkAgg", "Qt5Agg", "Qt4Agg"]
for _backend in _BACKENDS:
    try:
        matplotlib.use(_backend, force=True)
        import matplotlib.pyplot as plt  # noqa: F401 — triggers backend init
        break
    except Exception:
        continue
else:
    print("ERROR: no interactive matplotlib backend available. This script "
          "requires a display — use render_demo_video.py for headless rendering.",
          file=sys.stderr)
    sys.exit(1)

import matplotlib.pyplot as plt


THEME = {
    "bg":             "#0B0F17",
    "panel":          "#131926",
    "grid":           "#1F2937",
    "axis":           "#4B5563",
    "text_primary":   "#F9FAFB",
    "text_secondary": "#9CA3AF",
    "accent_warm":    "#F59E0B",
    "accent_cool":    "#22D3EE",
    "accent_warm_fill": "#F59E0B22",
    "accent_cool_fill": "#22D3EE22",
}

STALL_SECONDS = 8.0  # close window if no new samples for this long
REDRAW_INTERVAL = 0.25


def _get_machine_info() -> str:
    """Return a one-line machine description, e.g. 'Apple M3 Ultra · 512 GB'."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        chip = "Apple Silicon"
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        mem_gb = mem_bytes // (1024**3)
    except Exception:
        mem_gb = 0
    mem_str = f" {mem_gb} GB" if mem_gb else ""
    return f"{chip} {mem_str}".strip()


def setup_figure(model: str = "", machine: str = "", run_params: str = "") -> tuple:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor":   THEME["axis"],
        "axes.labelcolor":  THEME["text_secondary"],
        "xtick.color":      THEME["text_secondary"],
        "ytick.color":      THEME["text_secondary"],
        "text.color":       THEME["text_primary"],
        "figure.facecolor": THEME["bg"],
        "axes.facecolor":   THEME["panel"],
    })

    fig = plt.figure(figsize=(14, 8), dpi=100)
    fig.canvas.manager.set_window_title("AFM · live concurrent throughput")
    fig.patch.set_facecolor(THEME["bg"])

    ax_left = fig.add_axes([0.08, 0.13, 0.84, 0.67])
    ax_right = ax_left.twinx()

    for ax in (ax_left, ax_right):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(THEME["axis"])
            ax.spines[spine].set_linewidth(1.0)

    ax_left.grid(True, which="major", color=THEME["grid"], linestyle="-", linewidth=0.6, alpha=0.8)
    ax_left.set_axisbelow(True)

    fig.text(0.08, 0.92, "AFM · live concurrent throughput",
             fontsize=22, fontweight="bold", color=THEME["text_primary"],
             ha="left", va="center")
    # Subtitle: model, machine, date
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    subtitle_parts = []
    if model:
        subtitle_parts.append(model)
    if machine:
        subtitle_parts.append(machine)
    subtitle_parts.append(date_str)
    fig.text(0.08, 0.878, " · ".join(subtitle_parts),
             fontsize=11, color=THEME["text_secondary"],
             ha="left", va="center")
    if run_params:
        fig.text(0.08, 0.848, run_params,
                 fontsize=8, color=THEME["text_secondary"], family="monospace",
                 ha="left", va="center", alpha=0.7)

    ax_left.set_xlabel("Time (s)", fontsize=11, labelpad=8)
    ax_left.set_ylabel("Concurrent connections", fontsize=11, color=THEME["accent_warm"], labelpad=10)
    ax_left.tick_params(axis="y", colors=THEME["accent_warm"])
    ax_right.set_ylabel("Aggregate tokens / sec", fontsize=11, color=THEME["accent_cool"], labelpad=12)
    ax_right.tick_params(axis="y", colors=THEME["accent_cool"])

    return fig, ax_left, ax_right


def tail_jsonl(path: Path, stop_after_stall: float = STALL_SECONDS):
    """
    Generator yielding parsed JSON samples as they're appended to `path`.
    Waits for the file to exist. Stops (raises StopIteration) when no new
    data for `stop_after_stall` seconds AND at least one sample was seen.
    """
    while not path.exists():
        time.sleep(0.2)

    last_new = time.monotonic()
    saw_any = False
    with path.open() as f:
        f.seek(0, os.SEEK_SET)
        while True:
            line = f.readline()
            if line:
                last_new = time.monotonic()
                saw_any = True
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            else:
                if saw_any and time.monotonic() - last_new > stop_after_stall:
                    return
                time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace",
                    default=str(Path(__file__).resolve().parent / "out" / "trace.jsonl"))
    ap.add_argument("--target-users", type=int, default=200,
                    help="Fixed Y1 upper bound (connections axis). Should match --concurrent.")
    ap.add_argument("--initial-tps-max", type=float, default=1300.0,
                    help="Initial Y2 upper bound (aggregate tok/s). Axis expands if exceeded.")
    ap.add_argument("--total-seconds", type=float, default=120.0,
                    help="Fixed X-axis upper bound in seconds (ramp_s + hold_s). "
                         "Pre-setting this lets the lines sweep left-to-right across the "
                         "entire canvas rather than the axis rescaling as data arrives.")
    ap.add_argument("--smoothing-window", type=int, default=40,
                    help="Moving-average window for the agg_tps line, in samples. "
                         "Default 40 samples = 10s at 250ms cadence. Set 1 to disable.")
    ap.add_argument("--model", type=str, default="",
                    help="Model name to display in the chart subtitle.")
    ap.add_argument("--run-params", type=str, default="",
                    help="Test parameters string to display on the chart (e.g. the CLI invocation).")
    args = ap.parse_args()
    smoothing_w = max(1, args.smoothing_window)
    trace_path = Path(args.trace)
    y1_max = max(1, int(round(args.target_users * 1.05)))
    y2_max = max(1.0, args.initial_tps_max)
    x_max = max(1.0, args.total_seconds)
    model_name = args.model
    run_params = args.run_params

    print(f"[watch] tailing {trace_path}")
    print("[watch] waiting for driver to start writing...")

    # Background mactop sampler for GPU power, memory, and temperature.
    # Runs in a daemon thread; latest sample is read from the main loop.
    # Requires `brew install mactop` — gracefully no-ops if unavailable.
    _mactop_latest: dict = {}
    _mactop_peak_gpu_w: float = 0.0
    _mactop_peak_mem_gb: float = 0.0
    _mactop_lock = threading.Lock()

    def _mactop_thread():
        nonlocal _mactop_peak_gpu_w, _mactop_peak_mem_gb
        try:
            proc = subprocess.Popen(
                ["mactop", "--headless", "--format", "json", "-i", "500", "--count", "0"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            sys.stderr.write("[watch] mactop not found — GPU stats unavailable (brew install mactop)\n")
            return
        try:
            for raw_line in proc.stdout:
                try:
                    data = json.loads(raw_line.decode())
                    soc = data.get("soc_metrics", {})
                    mem = data.get("memory", {})
                    gpu_w = soc.get("gpu_power", 0.0)
                    mem_gb = mem.get("used", 0) / (1024**3) if "used" in mem else 0.0
                    with _mactop_lock:
                        _mactop_latest.update({
                            "gpu_power": gpu_w,
                            "gpu_usage": data.get("gpu_usage", 0),
                            "mem_gb": mem_gb,
                            "gpu_temp": soc.get("gpu_temp", 0),
                            "sys_power": soc.get("system_power", 0),
                        })
                        if gpu_w > _mactop_peak_gpu_w:
                            _mactop_peak_gpu_w = gpu_w
                        if mem_gb > _mactop_peak_mem_gb:
                            _mactop_peak_mem_gb = mem_gb
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass
        finally:
            proc.terminate()

    mactop_t = threading.Thread(target=_mactop_thread, daemon=True)
    mactop_t.start()

    machine_info = _get_machine_info()
    fig, ax_left, ax_right = setup_figure(model=model_name, machine=machine_info, run_params=run_params)
    # Fixed axes from the start — do NOT autoscale.
    # X is pinned to the full run duration so the lines sweep left to right
    # across the entire canvas for visual effect, rather than the axis
    # rescaling under them as new data arrives.
    ax_left.set_xlim(0, x_max)
    ax_left.set_ylim(0, y1_max)
    ax_right.set_ylim(0, y2_max)
    plt.ion()
    plt.show(block=False)

    xs: list[float] = []
    ys_conn: list[float] = []
    ys_tps: list[float] = []

    (line_conn,) = ax_left.plot(
        [], [], color=THEME["accent_warm"], linewidth=2.8,
        solid_capstyle="round", solid_joinstyle="round",
    )
    (line_tps,) = ax_right.plot(
        [], [], color=THEME["accent_cool"], linewidth=2.8,
        solid_capstyle="round", solid_joinstyle="round",
    )

    # Readouts — big numbers
    readout_conn = fig.text(
        0.88, 0.825, "0", ha="right", va="center",
        fontsize=24, fontweight="bold", color=THEME["accent_warm"],
    )
    fig.text(0.88, 0.795, "connections", ha="right", va="center",
             fontsize=10, color=THEME["text_secondary"])
    readout_tps = fig.text(
        0.73, 0.825, "0", ha="right", va="center",
        fontsize=24, fontweight="bold", color=THEME["accent_cool"],
    )
    fig.text(0.73, 0.795, "tokens / sec", ha="right", va="center",
             fontsize=10, color=THEME["text_secondary"])

    # Readouts — counters row (smaller, below the big numbers)
    counter_y = 0.755
    counter_fs = 9
    counter_color = THEME["text_secondary"]
    counter_val_color = "#FFFFFF"

    readout_peak_tps = fig.text(
        0.20, counter_y, "peak: 0", ha="left", va="center",
        fontsize=counter_fs, color=counter_val_color,
    )
    readout_gen_total = fig.text(
        0.36, counter_y, "gen: 0", ha="left", va="center",
        fontsize=counter_fs, color=counter_val_color,
    )
    readout_prompt_total = fig.text(
        0.52, counter_y, "prompt: 0", ha="left", va="center",
        fontsize=counter_fs, color=counter_val_color,
    )
    readout_cache = fig.text(
        0.70, counter_y, "cache: 0/0", ha="left", va="center",
        fontsize=counter_fs, color=counter_val_color,
    )
    readout_mem = fig.text(
        0.88, counter_y, "mem: --", ha="right", va="center",
        fontsize=counter_fs, color=counter_val_color,
    )

    # Tracking variables for counters
    peak_tps_seen = 0.0

    fill_conn = None
    fill_tps = None
    last_redraw = 0.0

    try:
        for sample in tail_jsonl(trace_path):
            xs.append(sample["t"])
            ys_conn.append(sample["active"])
            ys_tps.append(sample["agg_tps"])

            now = time.monotonic()
            if now - last_redraw < REDRAW_INTERVAL:
                continue
            last_redraw = now

            # Causal moving average on tok/sec: each plotted point is the
            # mean of the last `smoothing_w` samples. Smooths per-sample
            # noise (default 20 samples = 5s at 250ms cadence) without
            # leaking future data backwards. Connection line stays raw —
            # it's already a step function by construction.
            if smoothing_w > 1 and len(ys_tps) >= 1:
                w = smoothing_w
                ys_tps_smooth: list[float] = []
                for i in range(len(ys_tps)):
                    lo = max(0, i - w + 1)
                    window = ys_tps[lo:i + 1]
                    ys_tps_smooth.append(sum(window) / len(window))
            else:
                ys_tps_smooth = ys_tps

            line_conn.set_data(xs, ys_conn)
            line_tps.set_data(xs, ys_tps_smooth)

            if fill_conn is not None:
                try: fill_conn.remove()
                except Exception: pass
            if fill_tps is not None:
                try: fill_tps.remove()
                except Exception: pass
            fill_conn = ax_left.fill_between(xs, 0, ys_conn,
                                             color=THEME["accent_warm_fill"], linewidth=0)
            fill_tps = ax_right.fill_between(xs, 0, ys_tps_smooth,
                                             color=THEME["accent_cool_fill"], linewidth=0)

            # Axes stay pinned to their pre-set limits — do not rescale as
            # data arrives. This keeps the lines sweeping across the canvas
            # for visual effect. Only expand if observed data truly exceeds
            # the caps (safety net, not the intended path).
            obs_y1 = max(ys_conn) if ys_conn else 0
            if obs_y1 > y1_max:
                y1_max = obs_y1 * 1.05
                ax_left.set_ylim(0, y1_max)
            obs_y2 = max(ys_tps_smooth) if ys_tps_smooth else 0
            if obs_y2 > y2_max:
                y2_max = obs_y2 * 1.05
                ax_right.set_ylim(0, y2_max)
            if xs and xs[-1] > x_max:
                x_max = xs[-1] * 1.02
                ax_left.set_xlim(0, x_max)

            readout_conn.set_text(f"{int(round(ys_conn[-1]))}")
            readout_tps.set_text(f"{int(round(ys_tps_smooth[-1]))}")

            # Update counters from the latest sample
            cur_tps = ys_tps_smooth[-1] if ys_tps_smooth else 0
            if cur_tps > peak_tps_seen:
                peak_tps_seen = cur_tps
            readout_peak_tps.set_text(f"peak: {int(round(peak_tps_seen))} tok/s")

            gen_t = sample.get("gen_total", 0)
            prompt_t = sample.get("prompt_total", 0)
            chits = sample.get("cache_hits", 0)
            cmiss = sample.get("cache_misses", 0)

            def _fmt_k(n: int) -> str:
                return f"{n/1000:.1f}k" if n >= 1000 else str(n)

            readout_gen_total.set_text(f"gen: {_fmt_k(gen_t)}")
            readout_prompt_total.set_text(f"prompt: {_fmt_k(prompt_t)}")
            hit_pct = f" ({100*chits/(chits+cmiss):.0f}%)" if (chits + cmiss) > 0 else ""
            readout_cache.set_text(f"cache: {chits} hit / {cmiss} miss{hit_pct}")

            # GPU power + system memory from mactop (background thread)
            with _mactop_lock:
                gpu_w = _mactop_latest.get("gpu_power", 0)
                mem_gb = _mactop_latest.get("mem_gb", 0)
            if gpu_w > 0 or mem_gb > 0:
                readout_mem.set_text(
                    f"GPU: {gpu_w:.0f}W (peak {_mactop_peak_gpu_w:.0f}W)  "
                    f"mem: {mem_gb:.0f} GB (peak {_mactop_peak_mem_gb:.0f} GB)"
                )

            try:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except Exception:
                # Window closed — exit gracefully
                return

        print("[watch] trace finished (stall timeout reached)")
        print("[watch] close the window to exit, or press Ctrl-C")
        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        print("\n[watch] interrupted")


if __name__ == "__main__":
    main()
