"""Shared caption + winner-star helpers for all benchmark plots."""
import os
MODEL   = "Qwen3.6-27B · 4-bit class (MLX 4bit · ollama NVFP4 · llama.cpp Q4_K_M)"
MACHINE = "Apple M4 Pro · 10P+4E · 64 GB · macOS 26.5 (25F71)"
STAMP   = os.environ.get("BENCH_STAMP", "")
CAPTION = f"{MODEL}\n{MACHINE}   |   {STAMP}   |   single-stream · probe.py (server-usage anchored)"

def caption(fig):
    fig.text(0.5, 0.01, CAPTION, ha="center", va="bottom", fontsize=7, color="#555")
    fig.subplots_adjust(bottom=0.22)

def star_bar(ax, vals, lower_is_better=False):
    """Put a ★ above the winning bar; returns winner index."""
    pairs = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not pairs:
        return None
    idx = (min if lower_is_better else max)(pairs, key=lambda p: p[1])[0]
    top = max(v for _, v in pairs)
    ax.annotate("★", (idx, vals[idx]), textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=15, color="#f1a700")
    return idx

def star_label(names, vals, lower_is_better=False):
    """Return legend labels with ★ on the winner (for line charts)."""
    pairs = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not pairs:
        return list(names)
    idx = (min if lower_is_better else max)(pairs, key=lambda p: p[1])[0]
    return [f"{n} ★" if i == idx else n for i, n in enumerate(names)]
