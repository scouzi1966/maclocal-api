#!/bin/bash
# gpu-profile.sh — GPU shader profiling helpers for AFM
#
# Usage:
#   ./Scripts/gpu-profile.sh capture <model> [port]    # Capture Metal GPU trace
#   ./Scripts/gpu-profile.sh profile <model> [port]    # Run with GPU profiling stats
#   ./Scripts/gpu-profile.sh bandwidth [interval_ms]   # Monitor memory bandwidth (requires sudo)
#   ./Scripts/gpu-profile.sh trace <pid>               # Attach Instruments Metal System Trace
#   ./Scripts/gpu-profile.sh power [interval_ms]       # Monitor GPU power/frequency (requires sudo)
#
# Examples:
#   ./Scripts/gpu-profile.sh capture mlx-community/Qwen3.5-35B-A3B-4bit
#   ./Scripts/gpu-profile.sh bandwidth 200
#   ./Scripts/gpu-profile.sh trace $(pgrep afm)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
TRACE_DIR="/tmp"

usage() {
    cat <<'EOF'
GPU Shader Profiling for AFM

Commands:
  profile <model>            Per-request GPU stats: device, memory, bandwidth estimate.
                             Zero overhead — runs at full speed. Best for iterative work.

  trace <model> [seconds]    Record Metal System Trace during inference (default: 10s).
                             Command-buffer timing, GPU scheduling, pipeline bubbles.
                             ~100 MB output. No per-kernel shader names.

  capture <model>            Full Metal GPU capture (.gputrace) for Xcode shader analysis.
                             Per-kernel names and costs. WARNING: multi-GB traces.
                             Auto-limits to 5 tokens. Only practical for small models (<3B).

  bandwidth [interval_ms]    Monitor DRAM bandwidth saturation during inference.
                             Run in a separate terminal while AFM is serving.
                             Requires sudo. Default interval: 200ms.

  trace-pid <pid>            Attach Instruments Metal System Trace to a running process.
                             Output: /tmp/afm-metal.trace (open in Instruments)

  power [interval_ms]        Monitor GPU power, frequency, and active residency
                             Requires sudo. Default interval: 500ms.

Workflow:
  1. Start with 'profile' to get macro-level stats and bandwidth estimate
  2. If bandwidth-bound (>80%), kernel optimization won't help — need algorithmic changes
  3. If low utilization (<30%), use 'trace' to find CPU-GPU pipeline bubbles
  4. Use 'capture' for per-kernel Xcode analysis of specific bottlenecks
  5. Use 'bandwidth' in a separate terminal to verify DRAM saturation during inference
EOF
    exit 0
}

cmd_capture() {
    local model="${1:?Error: model required (e.g. mlx-community/Qwen3.5-35B-A3B-4bit)}"
    local port="${2:-9999}"
    local trace_path="${TRACE_DIR}/afm-trace.gputrace"

    echo "=== GPU Capture ==="
    echo "Model:  $model"
    echo "Trace:  $trace_path"
    echo ""
    echo "Starting AFM with GPU capture (single-prompt mode)..."
    echo "After completion, open in Xcode: open $trace_path"
    echo ""

    MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" \
        "$PROJECT_DIR/.build/release/afm" mlx \
        -m "$model" \
        --gpu-capture "$trace_path" \
        -s "Write a short haiku about metal shaders." \
        --temperature 0.7

    echo ""
    if [ -e "$trace_path" ]; then
        echo "Trace captured: $trace_path"
        echo "Open in Xcode: open $trace_path"
        echo ""
        echo "In Xcode Metal Debugger:"
        echo "  1. Dependencies view → sort by GPU Time"
        echo "  2. Expected: affine_qmv_fast dominates decode, steel_gemm_fused dominates prefill"
        echo "  3. Shader Cost Graph → per-line costs within kernels"
        echo "  4. Performance Heat Map → bottleneck instructions"
    else
        echo "Warning: No trace file generated."
        echo "Make sure MTL_CAPTURE_ENABLED=1 is set (afm sets this automatically with --gpu-capture)."
    fi
}

cmd_profile() {
    local model="${1:?Error: model required}"
    local port="${2:-9999}"

    echo "=== GPU Profile ==="
    echo "Model: $model"
    echo ""

    MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" \
        "$PROJECT_DIR/.build/release/afm" mlx \
        -m "$model" \
        --gpu-profile \
        -s "Explain the difference between compute-bound and memory-bandwidth-bound GPU kernels in 2 sentences." \
        --temperature 0.7
}

cmd_trace() {
    local model="${1:?Error: model required}"
    local duration="${2:-10}"

    echo "=== Metal System Trace ==="
    echo "Model:    $model"
    echo "Duration: ${duration}s"
    echo ""

    MACAFM_MLX_MODEL_CACHE="$CACHE_DIR" \
        "$PROJECT_DIR/.build/release/afm" mlx \
        -m "$model" \
        --gpu-trace "$duration" \
        -s "Explain the difference between compute-bound and memory-bandwidth-bound GPU kernels in 2 sentences." \
        --temperature 0.7

    echo ""
    if [ -e "/tmp/afm-metal.trace" ]; then
        echo "Open in Instruments: open /tmp/afm-metal.trace"
    fi
}

cmd_bandwidth() {
    local interval="${1:-500}"

    if ! command -v mactop &>/dev/null; then
        echo "mactop not found, falling back to powermetrics (requires sudo)"
        echo "Install mactop: brew install mactop"
        echo ""
        sudo powermetrics --samplers gpu_power,bandwidth -i "$interval"
        return
    fi

    # Use PTY to force mactop line-buffered output (Go binary ignores stdbuf)
    python3 -u - "$interval" << 'PYEOF'
import pty, os, select, sys, signal, json, time

interval = sys.argv[1] if len(sys.argv) > 1 else "500"
THEORETICAL_BW = 800.0  # GB/s — adjust for your chip
BAR_WIDTH = 50

def make_bar(value, max_val, width):
    filled = min(int(value / max_val * width), width)
    return "█" * filled + "░" * (width - filled)

def color(bw):
    if bw > 200: return "\033[31m"    # red
    if bw > 100: return "\033[33m"    # yellow
    if bw > 30:  return "\033[32m"    # green
    return "\033[90m"                  # gray

RESET = "\033[0m"

print()
print("╔═══════════════════════════════════════════════════════════════════════════════════╗")
print("║  AFM DRAM Bandwidth Monitor (mactop)                     Ctrl+C to stop          ║")
print("╚═══════════════════════════════════════════════════════════════════════════════════╝")
print()
print(f"{'TIME':>8}  {'GPU%':>4}  {'POWER':>6}  {'BW':>7}  {'UTILIZATION':^{BAR_WIDTH+6}}")
print(f"{'─'*8}  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*(BAR_WIDTH+6)}")

master, slave = pty.openpty()
pid = os.fork()
if pid == 0:
    os.setsid()
    os.dup2(slave, 1)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    os.close(master); os.close(slave)
    os.execvp("mactop", ["mactop", "--headless", "--format", "json", "-i", interval, "--count", "0"])
os.close(slave)

peak_bw = 0.0
samples = 0

def cleanup(*_):
    os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)
    os.close(master)
    print(f"\n{'─'*8}  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*(BAR_WIDTH+6)}")
    print(f"Peak: {peak_bw:.1f} GB/s ({peak_bw/THEORETICAL_BW*100:.1f}% of {THEORETICAL_BW:.0f} GB/s)  |  {samples} samples")
    print(f"Each █ = {THEORETICAL_BW/BAR_WIDTH:.0f} GB/s  |  \033[31m■\033[0m >200  \033[33m■\033[0m >100  \033[32m■\033[0m >30  \033[90m■\033[0m idle")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

buf = b""
while True:
    r, _, _ = select.select([master], [], [], 2.0)
    if not r: continue
    try:
        chunk = os.read(master, 131072)
    except OSError:
        break
    if not chunk: break
    buf += chunk
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        decoded = line.decode("utf-8", errors="replace").strip()
        if decoded.startswith("[{"): decoded = decoded[1:]
        if decoded.startswith(",{"): decoded = decoded[1:]
        if decoded.endswith(","): decoded = decoded[:-1]
        if not decoded.startswith("{"): continue
        try:
            d = json.loads(decoded)
            soc = d["soc_metrics"]
            bw = soc.get("dram_bw_combined_gbs", 0)
            gpu = d.get("gpu_usage", 0)
            pw = soc.get("gpu_power", 0)
            rd = soc.get("dram_read_bw_gbs", 0)
            wr = soc.get("dram_write_bw_gbs", 0)
            if bw > peak_bw: peak_bw = bw
            samples += 1
            c = color(bw)
            bar = make_bar(bw, THEORETICAL_BW, BAR_WIDTH)
            pct = bw / THEORETICAL_BW * 100
            ts = time.strftime("%H:%M:%S")
            print(f"{ts:>8}  {gpu:3.0f}%  {pw:5.1f}W  {bw:5.1f}  {c}{bar}{RESET} {pct:4.1f}%", flush=True)
        except (json.JSONDecodeError, KeyError):
            pass
PYEOF
}

cmd_trace_pid() {
    local pid="${1:?Error: PID required (e.g. \$(pgrep afm))}"
    local output="${TRACE_DIR}/afm-metal.trace"

    echo "=== Metal System Trace (attach) ==="
    echo "PID:    $pid"
    echo "Output: $output"
    echo ""
    echo "Recording for 10 seconds..."
    echo "Send requests to AFM during this window."
    echo ""

    xcrun xctrace record \
        --template "Metal System Trace" \
        --attach "$pid" \
        --time-limit 10s \
        --output "$output"

    echo ""
    echo "Trace recorded: $output"
    echo "Open in Instruments: open $output"
}

cmd_power() {
    local interval="${1:-500}"

    echo "=== GPU Power Monitor ==="
    echo "Interval: ${interval}ms"
    echo "Press Ctrl+C to stop"
    echo ""

    sudo powermetrics --samplers gpu_power -i "$interval" --show-process-gpu
}

# --- Main ---
case "${1:-help}" in
    profile)   shift; cmd_profile "$@" ;;
    trace)     shift; cmd_trace "$@" ;;
    capture)   shift; cmd_capture "$@" ;;
    bandwidth) shift; cmd_bandwidth "$@" ;;
    trace-pid) shift; cmd_trace_pid "$@" ;;
    power)     shift; cmd_power "$@" ;;
    help|--help|-h) usage ;;
    *) echo "Unknown command: $1"; usage ;;
esac
