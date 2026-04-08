#!/usr/bin/env python3
"""
Render the concurrent-throughput demo trace into an MP4 suitable for
social posts. Dark theme, dual Y-axis, smooth animated line draw,
AFM-branded title block.

Input : Scripts/demo/out/trace.jsonl  (from concurrent_load_driver.py)
Output: Scripts/demo/out/concurrent_demo.mp4

Dependencies: matplotlib + ffmpeg on PATH.

Usage:
    python3 Scripts/demo/render_demo_video.py \
        --trace Scripts/demo/out/trace.jsonl \
        --summary Scripts/demo/out/trace.summary.json \
        --output Scripts/demo/out/concurrent_demo.mp4 \
        --fps 30 \
        --duration-cap 150
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
except ImportError:
    print("ERROR: matplotlib/numpy not installed. Run: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Visual theme — dark, refined, marketing-grade
# ---------------------------------------------------------------------------

THEME = {
    "bg":            "#0B0F17",   # near-black with a slight blue tint
    "panel":         "#131926",   # card background
    "grid":          "#1F2937",
    "axis":          "#4B5563",
    "text_primary":  "#F9FAFB",
    "text_secondary":"#9CA3AF",
    "accent_warm":   "#F59E0B",   # connections line — amber
    "accent_cool":   "#22D3EE",   # throughput line — cyan
    "accent_warm_fill": "#F59E0B22",
    "accent_cool_fill": "#22D3EE22",
}


def setup_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": THEME["axis"],
        "axes.labelcolor": THEME["text_secondary"],
        "xtick.color": THEME["text_secondary"],
        "ytick.color": THEME["text_secondary"],
        "text.color": THEME["text_primary"],
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["panel"],
        "savefig.facecolor": THEME["bg"],
        "savefig.edgecolor": THEME["bg"],
    })

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(THEME["bg"])

    # Leave room at the top for a title block and at the bottom for a footer
    ax_left = fig.add_axes([0.08, 0.15, 0.84, 0.62])
    ax_right = ax_left.twinx()

    for ax in (ax_left, ax_right):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(THEME["axis"])
            ax.spines[spine].set_linewidth(1.0)

    ax_left.grid(True, which="major", color=THEME["grid"], linestyle="-", linewidth=0.6, alpha=0.8)
    ax_left.set_axisbelow(True)

    return fig, ax_left, ax_right


def draw_static_chrome(
    fig: plt.Figure,
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    title_line1: str,
    title_line2: str,
    footer: str,
    max_t: float,
    y1_max: float,
    y2_max: float,
) -> None:
    # Title
    fig.text(
        0.08, 0.90,
        title_line1,
        fontsize=26, fontweight="bold", color=THEME["text_primary"],
        ha="left", va="center",
    )
    fig.text(
        0.08, 0.845,
        title_line2,
        fontsize=14, color=THEME["text_secondary"],
        ha="left", va="center",
    )

    # Footer
    fig.text(
        0.08, 0.05,
        footer,
        fontsize=11, color=THEME["text_secondary"],
        ha="left", va="center",
    )
    fig.text(
        0.92, 0.05,
        "afm · Swift + MLX on Apple Silicon",
        fontsize=11, color=THEME["text_secondary"],
        ha="right", va="center",
    )

    # Left axis — concurrent connections (warm). Fixed-range from 0 to y1_max.
    ax_left.set_xlabel("Time (s)", fontsize=12, labelpad=10)
    ax_left.set_ylabel("Concurrent connections", fontsize=12, color=THEME["accent_warm"], labelpad=10)
    ax_left.tick_params(axis="y", colors=THEME["accent_warm"])
    ax_left.set_xlim(0, max(1.0, max_t))
    ax_left.set_ylim(0, max(1.0, y1_max))

    # Right axis — throughput (cool). Fixed-range from 0 to y2_max.
    ax_right.set_ylabel("Aggregate tokens / sec", fontsize=12, color=THEME["accent_cool"], labelpad=12)
    ax_right.tick_params(axis="y", colors=THEME["accent_cool"])
    ax_right.set_ylim(0, max(1.0, y2_max))

    # Legend as two swatches inside the plot top-left
    lg_y = 0.73
    ax_left_bbox = ax_left.get_position()
    legend_x0 = ax_left_bbox.x0 + 0.012
    # Connections swatch
    fig.add_artist(
        plt.Line2D(
            [legend_x0, legend_x0 + 0.03],
            [lg_y, lg_y],
            color=THEME["accent_warm"],
            linewidth=3,
            solid_capstyle="round",
            transform=fig.transFigure,
        )
    )
    fig.text(
        legend_x0 + 0.037, lg_y, "Concurrent connections",
        fontsize=11, color=THEME["text_primary"], va="center",
    )
    # Throughput swatch
    fig.add_artist(
        plt.Line2D(
            [legend_x0 + 0.22, legend_x0 + 0.25],
            [lg_y, lg_y],
            color=THEME["accent_cool"],
            linewidth=3,
            solid_capstyle="round",
            transform=fig.transFigure,
        )
    )
    fig.text(
        legend_x0 + 0.257, lg_y, "Aggregate tokens / sec",
        fontsize=11, color=THEME["text_primary"], va="center",
    )


def render_animation(
    trace: list[dict],
    output: Path,
    fps: int,
    title_line1: str,
    title_line2: str,
    footer: str,
    duration_cap: float | None,
    target_users_for_axis: int | None,
    initial_tps_max_for_axis: float,
    total_seconds_for_axis: float | None,
) -> None:
    # Normalize trace to arrays
    if not trace:
        print("ERROR: trace is empty", file=sys.stderr)
        sys.exit(2)

    ts = np.array([s["t"] for s in trace], dtype=float)
    active = np.array([s["active"] for s in trace], dtype=float)
    agg_tps = np.array([s["agg_tps"] for s in trace], dtype=float)

    # Smooth the throughput line with a short moving average to make it
    # look refined on the video (token arrivals are bursty at millisecond
    # scale but the graph shows ~250ms buckets — one smoothing pass at
    # window=3 removes visual jitter without masking real trends).
    smoothed = np.convolve(agg_tps, np.ones(3) / 3.0, mode="same")
    # Keep the first and last points unsmoothed so the chart starts/ends at 0.
    smoothed[0] = agg_tps[0]
    smoothed[-1] = agg_tps[-1]

    # Duration cap
    if duration_cap is not None and ts[-1] > duration_cap:
        mask = ts <= duration_cap
        ts = ts[mask]
        active = active[mask]
        smoothed = smoothed[mask]

    data_max_t = float(ts[-1]) if len(ts) else 1.0
    observed_max_conn = int(active.max()) if len(active) else 1
    observed_max_tps = float(smoothed.max()) if len(smoothed) else 1.0

    # Fixed-axis policy: all three axes are pinned from frame 1.
    #   X: total_seconds (ramp_s + hold_s) — so the lines sweep left-to-right
    #   Y1 (connections): target_users * 1.05
    #   Y2 (tok/s): max(initial_tps_max, observed * 1.05) — expand if needed
    # Only Y axes expand if observed data exceeds them. X is hard-capped to
    # the total run duration for visual effect.
    max_t = total_seconds_for_axis if total_seconds_for_axis else data_max_t
    y1_max = max(observed_max_conn, int(round(target_users_for_axis * 1.05)) if target_users_for_axis else observed_max_conn)
    y2_max = max(observed_max_tps, initial_tps_max_for_axis)

    fig, ax_left, ax_right = setup_figure()
    draw_static_chrome(
        fig, ax_left, ax_right,
        title_line1, title_line2, footer,
        max_t, y1_max, y2_max,
    )

    # Connection line (left axis) — solid, thick, amber, soft glow via low-alpha fill
    (line_conn,) = ax_left.plot(
        [], [], color=THEME["accent_warm"], linewidth=2.8,
        solid_capstyle="round", solid_joinstyle="round",
    )
    fill_conn = ax_left.fill_between(
        ts, 0, 0, color=THEME["accent_warm_fill"], linewidth=0,
    )

    # Throughput line (right axis) — solid, thick, cyan
    (line_tps,) = ax_right.plot(
        [], [], color=THEME["accent_cool"], linewidth=2.8,
        solid_capstyle="round", solid_joinstyle="round",
    )
    fill_tps = ax_right.fill_between(
        ts, 0, 0, color=THEME["accent_cool_fill"], linewidth=0,
    )

    # Live numeric readouts (top-right area of the chart)
    ax_left_bbox = ax_left.get_position()
    readout_x = ax_left_bbox.x0 + ax_left_bbox.width - 0.01
    readout_conn = fig.text(
        readout_x, 0.715,
        "", ha="right", va="center",
        fontsize=22, fontweight="bold", color=THEME["accent_warm"],
    )
    readout_conn_label = fig.text(
        readout_x, 0.686,
        "connections", ha="right", va="center",
        fontsize=10, color=THEME["text_secondary"],
    )
    readout_tps = fig.text(
        readout_x - 0.13, 0.715,
        "", ha="right", va="center",
        fontsize=22, fontweight="bold", color=THEME["accent_cool"],
    )
    readout_tps_label = fig.text(
        readout_x - 0.13, 0.686,
        "tokens / sec", ha="right", va="center",
        fontsize=10, color=THEME["text_secondary"],
    )

    # Frame plan: one frame per rendered video frame.
    # Map video time → data-index via interpolation.
    total_video_seconds = max_t
    total_frames = max(2, int(round(total_video_seconds * fps)))

    def frame_data_index(frame: int) -> int:
        if total_frames <= 1:
            return len(ts) - 1
        v_time = (frame / (total_frames - 1)) * max_t
        # Index of the last sample with t <= v_time
        idx = int(np.searchsorted(ts, v_time, side="right")) - 1
        return max(0, min(len(ts) - 1, idx))

    def init():
        line_conn.set_data([], [])
        line_tps.set_data([], [])
        readout_conn.set_text("0")
        readout_tps.set_text("0")
        return line_conn, line_tps, readout_conn, readout_tps

    def update(frame: int):
        nonlocal fill_conn, fill_tps
        idx = frame_data_index(frame)
        xs = ts[: idx + 1]
        ys_conn = active[: idx + 1]
        ys_tps = smoothed[: idx + 1]
        line_conn.set_data(xs, ys_conn)
        line_tps.set_data(xs, ys_tps)

        # Refresh fill polygons — remove and re-add each frame
        # (matplotlib doesn't support mutating fill_between in place).
        try:
            fill_conn.remove()
        except Exception:
            pass
        try:
            fill_tps.remove()
        except Exception:
            pass
        fill_conn = ax_left.fill_between(
            xs, 0, ys_conn, color=THEME["accent_warm_fill"], linewidth=0,
        )
        fill_tps = ax_right.fill_between(
            xs, 0, ys_tps, color=THEME["accent_cool_fill"], linewidth=0,
        )

        readout_conn.set_text(f"{int(round(ys_conn[-1]))}")
        readout_tps.set_text(f"{int(round(ys_tps[-1]))}")
        return line_conn, line_tps, fill_conn, fill_tps, readout_conn, readout_tps

    print(f"[render] frames: {total_frames} @ {fps} fps ({total_video_seconds:.1f}s video)")
    print(f"[render] peak connections: {max_conn}")
    print(f"[render] peak agg tok/s:   {max_tps:.1f}")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    # Progress callback — matplotlib passes (current_frame, total_frames).
    import time as _time
    render_start = _time.monotonic()
    is_tty = sys.stderr.isatty()
    bar_width = 28

    def _progress(current: int, total: int) -> None:
        pct = ((current + 1) / total) * 100.0
        filled = int(round(bar_width * pct / 100.0))
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = _time.monotonic() - render_start
        if current > 0:
            eta = elapsed * (total - current - 1) / (current + 1)
        else:
            eta = 0.0
        line = (
            f"\r[render] [{bar}] {pct:5.1f}%  "
            f"frame {current + 1:4d}/{total:4d}  "
            f"elapsed={elapsed:5.1f}s  eta={eta:5.1f}s"
        )
        if is_tty:
            sys.stderr.write(line)
        else:
            sys.stderr.write(line.lstrip("\r") + "\n")
        sys.stderr.flush()

    # Export via ffmpeg
    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found on PATH. Saving as animated GIF instead.", file=sys.stderr)
        gif_path = output.with_suffix(".gif")
        anim.save(
            str(gif_path),
            writer=animation.PillowWriter(fps=fps),
            progress_callback=_progress,
        )
        if is_tty:
            sys.stderr.write("\n")
        print(f"[render] wrote {gif_path}")
        return

    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=6500,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    anim.save(str(output), writer=writer, progress_callback=_progress)
    if is_tty:
        sys.stderr.write("\n")
    total_elapsed = _time.monotonic() - render_start
    print(f"[render] wrote {output} in {total_elapsed:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render concurrent demo trace to MP4.")
    ap.add_argument("--trace", default="Scripts/demo/out/trace.jsonl")
    ap.add_argument("--summary", default="Scripts/demo/out/trace.summary.json")
    ap.add_argument("--output", default="Scripts/demo/out/concurrent_demo.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--duration-cap", type=float, default=None,
        help="Max seconds of trace to render (trims to fit a video length).",
    )
    ap.add_argument(
        "--title", default="AFM: Concurrent Inference at Scale",
        help="Top headline text.",
    )
    ap.add_argument(
        "--subtitle", default=None,
        help="Subtitle override. If omitted, built from summary.json (model + config).",
    )
    ap.add_argument(
        "--footer", default="Live single-node benchmark",
        help="Bottom-left footer text.",
    )
    ap.add_argument(
        "--initial-tps-max", type=float, default=None,
        help="Override the initial Y2 maximum for aggregate tok/s. "
             "Defaults to the value in summary.json, or 1300 if absent.",
    )
    ap.add_argument(
        "--target-users", type=int, default=None,
        help="Override the connection-axis upper bound. "
             "Defaults to the target_users value in summary.json.",
    )
    ap.add_argument(
        "--total-seconds", type=float, default=None,
        help="Pin the X-axis to this total duration (ramp_s + hold_s). "
             "Defaults to ramp_s + hold_s from summary.json, or the trace's last t if absent.",
    )
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"ERROR: trace not found at {trace_path}", file=sys.stderr)
        sys.exit(1)

    trace = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                trace.append(json.loads(line))

    summary_path = Path(args.summary)
    summary: dict = {}
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    cfg = summary.get("cfg", {})
    model = cfg.get("model", "(unknown model)")
    target = cfg.get("target_users", None)
    prefix_cache = cfg.get("mode", "") or ""
    summary_initial_tps = cfg.get("initial_tps_max", 1300.0)

    # Clean up model path for display
    display_model = model.split("/", 1)[-1] if "/" in model else model

    if args.subtitle is None:
        bits = [display_model]
        if target:
            bits.append(f"{target} concurrent connections")
        if prefix_cache and prefix_cache != "general":
            bits.append(f"mode: {prefix_cache}")
        subtitle = "  ·  ".join(bits)
    else:
        subtitle = args.subtitle

    # Resolve axis overrides: CLI > summary > default.
    target_users_for_axis = args.target_users if args.target_users else target
    initial_tps_max_for_axis = (
        args.initial_tps_max if args.initial_tps_max is not None else summary_initial_tps
    )
    if args.total_seconds is not None:
        total_seconds_for_axis: float | None = args.total_seconds
    elif cfg.get("total_seconds") is not None:
        total_seconds_for_axis = float(cfg["total_seconds"])
    else:
        ramp_s = cfg.get("ramp_s")
        hold_s = cfg.get("hold_s")
        if ramp_s is not None and hold_s is not None:
            total_seconds_for_axis = float(ramp_s) + float(hold_s)
        else:
            total_seconds_for_axis = None  # fall back to data-driven

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_animation(
        trace=trace,
        output=output_path,
        fps=args.fps,
        title_line1=args.title,
        title_line2=subtitle,
        footer=args.footer,
        duration_cap=args.duration_cap,
        target_users_for_axis=target_users_for_axis,
        initial_tps_max_for_axis=initial_tps_max_for_axis,
        total_seconds_for_axis=total_seconds_for_axis,
    )


if __name__ == "__main__":
    main()
