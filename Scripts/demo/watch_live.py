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
import sys
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


def setup_figure() -> tuple:
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
    fig.text(0.08, 0.875, "tail -f Scripts/demo/out/trace.jsonl",
             fontsize=11, color=THEME["text_secondary"], family="monospace",
             ha="left", va="center")

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
    ap.add_argument("--smoothing-window", type=int, default=20,
                    help="Moving-average window for the agg_tps line, in samples. "
                         "Default 20 samples = 5s at 250ms cadence. Set 1 to disable.")
    args = ap.parse_args()
    smoothing_w = max(1, args.smoothing_window)
    trace_path = Path(args.trace)
    y1_max = max(1, int(round(args.target_users * 1.05)))
    y2_max = max(1.0, args.initial_tps_max)
    x_max = max(1.0, args.total_seconds)

    print(f"[watch] tailing {trace_path}")
    print("[watch] waiting for driver to start writing...")

    fig, ax_left, ax_right = setup_figure()
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

    # Readouts
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
