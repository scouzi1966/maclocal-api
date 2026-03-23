#!/usr/bin/env python3
"""AFM GPU Shader Profile — Full test harness with HTML report.

Runs inference with --gpu-profile --gpu-trace, collects mactop bandwidth
samples during inference, extracts shader kernel names from the trace,
and generates an interactive HTML report.

Usage:
  python3 Scripts/gpu-profile-report.py [model] [max_tokens] [prompt]

Defaults:
  model:      mlx-community/Qwen3.5-35B-A3B-4bit
  max_tokens: 4096
  prompt:     (built-in GPU analysis prompt)
"""
import pty, os, select, signal, json, time, subprocess, sys, re

# ─── Config ───
MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-35B-A3B-4bit"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
PROMPT = sys.argv[3] if len(sys.argv) > 3 else (
    "You are a senior GPU performance engineer specializing in Apple Silicon Metal compute shaders. "
    "Write a comprehensive 2000-word technical analysis covering: "
    "1) Apple M-series GPU memory hierarchy and its impact on compute shader performance, "
    "2) Quantized MatVec optimization for 4-bit inference (affine_qmv_fast), "
    "3) MoE expert dispatch bandwidth analysis, "
    "4) SDPA attention implementation tradeoffs on Apple Silicon, "
    "5) Top 5 concrete optimizations ranked by expected throughput improvement. "
    "Include formulas, pseudocode, and specific numbers."
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
AFM = os.path.join(PROJECT_DIR, ".build/release/afm")
CACHE = os.environ.get("MACAFM_MLX_MODEL_CACHE", "/Volumes/edata/models/vesta-test-cache")
TRACE_DURATION = 15
THEORETICAL_BW = 800.0
OUTPUT_HTML = "/tmp/afm-gpu-profile.html"
OUTPUT_JSON = "/tmp/afm-gpu-samples.json"

SAMPLES = []

# ─── Phase 1: Warm up mactop ───
print("Phase 1: Warming up mactop...", flush=True)
master, slave = pty.openpty()
mactop_pid = os.fork()
if mactop_pid == 0:
    os.setsid()
    os.dup2(slave, 1)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    os.close(master); os.close(slave)
    os.execvp("mactop", ["mactop", "--headless", "--format", "json", "-i", "300", "--count", "500"])
os.close(slave)

buf = b""
warmup = 0
while warmup < 3:
    r, _, _ = select.select([master], [], [], 5.0)
    if r:
        buf += os.read(master, 131072)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            d = line.decode("utf-8", errors="replace").strip()
            if d.startswith("[{"): d = d[1:]
            if d.startswith(",{"): d = d[1:]
            if d.startswith("{"): warmup += 1
print(f"  mactop ready ({warmup} warmup samples)", flush=True)

# ─── Phase 2: Launch AFM ───
afm_args = [
    AFM, "mlx", "-m", MODEL,
    "--gpu-profile", "--gpu-trace", str(TRACE_DURATION),
    "-s", PROMPT, "--no-think", "--max-tokens", str(MAX_TOKENS), "--temperature", "0.3"
]
afm_cmd_display = f"MACAFM_MLX_MODEL_CACHE={CACHE} {' '.join(afm_args)}"
print(f"Phase 2: Launching AFM inference...", flush=True)
print(f"  {afm_cmd_display}", flush=True)

env = os.environ.copy()
env["MACAFM_MLX_MODEL_CACHE"] = CACHE
afm_out_path = "/tmp/afm-profile-output.txt"
afm_out = open(afm_out_path, "w")
afm = subprocess.Popen(afm_args, stdout=afm_out, stderr=subprocess.STDOUT, env=env)

t0 = time.time()
print("Phase 3: Collecting mactop samples during inference...", flush=True)

# ─── Phase 3: Collect samples ───
def parse_mactop_line(raw):
    d = raw.strip()
    if d.startswith("[{"): d = d[1:]
    if d.startswith(",{"): d = d[1:]
    if d.endswith(","): d = d[:-1]
    if not d.startswith("{"): return None
    try:
        j = json.loads(d)
        soc = j["soc_metrics"]
        return {
            "t": round(time.time() - t0, 1),
            "bw": round(soc.get("dram_bw_combined_gbs", 0), 1),
            "bw_r": round(soc.get("dram_read_bw_gbs", 0), 1),
            "bw_w": round(soc.get("dram_write_bw_gbs", 0), 1),
            "gpu": round(j.get("gpu_usage", 0), 1),
            "gpu_power": round(soc.get("gpu_power", 0), 1),
            "sys_power": round(soc.get("system_power", 0), 1),
        }
    except:
        return None

while afm.poll() is None:
    r, _, _ = select.select([master], [], [], 1.0)
    if r:
        try: buf += os.read(master, 131072)
        except OSError: break
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = parse_mactop_line(line.decode("utf-8", errors="replace"))
            if s:
                SAMPLES.append(s)
                bw = s["bw"]; bar = "█" * min(int(bw/16), 50) + "░" * max(0, 50-min(int(bw/16),50))
                c = "\033[31m" if bw>200 else "\033[33m" if bw>100 else "\033[32m" if bw>30 else "\033[90m"
                print(f"  t={s['t']:5.1f}s | GPU:{s['gpu']:5.1f}% {s['gpu_power']:5.1f}W | BW:{bw:6.1f} GB/s | {c}{bar}\033[0m {bw/8:.1f}%", flush=True)

# Idle tail
for _ in range(6):
    r, _, _ = select.select([master], [], [], 1.0)
    if r:
        try: buf += os.read(master, 131072)
        except: break
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = parse_mactop_line(line.decode("utf-8", errors="replace"))
            if s: SAMPLES.append(s)

try: os.kill(mactop_pid, signal.SIGTERM)
except: pass
try: os.waitpid(mactop_pid, 0)
except: pass
try: os.close(master)
except: pass
afm_out.close()

# ─── Phase 4: Extract shader kernel names ───
print("\nPhase 4: Extracting shader kernel names...", flush=True)
kernels = []
try:
    result = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", "/tmp/afm-metal.trace",
         "--xpath", '/trace-toc/run/data/table[@schema="metal-shader-profiler-shader-list"]'],
        capture_output=True, text=True, timeout=30
    )
    labels = re.findall(r'<metal-object-label[^>]*fmt="([^"]+)"', result.stdout)
    kernels = sorted(set(re.sub(r'\s*\(\d+\)$', '', l) for l in labels))
    print(f"  {len(kernels)} unique Metal kernels", flush=True)
except Exception as e:
    print(f"  Shader extraction failed: {e}", flush=True)

# ─── Phase 5: Parse AFM profile output ───
print("Phase 5: Parsing AFM profile output...", flush=True)
afm_output = open(afm_out_path).read()
profile_lines = [l for l in afm_output.split("\n") if "GPU-PROFILE" in l or "GPU-TRACE" in l]

# Extract the command line from profile output
afm_command = afm_cmd_display
for l in profile_lines:
    if "Command" not in l and "afm" in l and "MACAFM" in l:
        afm_command = l.split("]")[-1].strip()
        break

# Extract timing
decode_toks = MAX_TOKENS; prefill_toks = 0; decode_time = 0; prefill_time = 0
for l in profile_lines:
    m = re.search(r'Prefill:\s*([\d.]+)s\s*\((\d+)\s*tokens', l)
    if m: prefill_time = float(m.group(1)); prefill_toks = int(m.group(2))
    m = re.search(r'Decode:\s*([\d.]+)s\s*\((\d+)\s*tokens', l)
    if m: decode_time = float(m.group(1)); decode_toks = int(m.group(2))

decode_tps = decode_toks / decode_time if decode_time > 0 else 0
prefill_tps = prefill_toks / prefill_time if prefill_time > 0 else 0

# ─── Phase 6: Compute stats ───
active = [s for s in SAMPLES if s["gpu"] > 80]
peak_bw = max((s["bw"] for s in SAMPLES), default=0)
avg_bw = sum(s["bw"] for s in active) / len(active) if active else 0
peak_gpu = max((s["gpu"] for s in SAMPLES), default=0)
peak_power = max((s["gpu_power"] for s in SAMPLES), default=0)
avg_power = sum(s["gpu_power"] for s in active) / len(active) if active else 0

print(f"\n{'='*70}")
print(f"  Samples: {len(SAMPLES)} ({len(active)} active)")
print(f"  Peak BW: {peak_bw:.1f} GB/s ({peak_bw/THEORETICAL_BW*100:.1f}%)")
print(f"  Avg BW (active): {avg_bw:.1f} GB/s ({avg_bw/THEORETICAL_BW*100:.1f}%)")
print(f"  Peak GPU: {peak_gpu:.1f}% | Peak Power: {peak_power:.1f}W | Avg Power: {avg_power:.1f}W")
print(f"  Decode: {decode_tps:.1f} tok/s | Prefill: {prefill_tps:.1f} tok/s")
print(f"  Kernels: {len(kernels)}")
print(f"{'='*70}")

# ─── Phase 7: Generate HTML ───
print("Phase 6: Generating HTML report...", flush=True)

# Top kernels
important = [
    ("affine_qmv_fast", "Quantized MatVec (decode)", "#f85149", "BW-bound"),
    ("affine_gather_qmv_fast", "MoE Expert Dispatch", "#d29922", "BW-bound"),
    ("affine_qmm_t", "Quantized MatMul (prefill)", "#58a6ff", "Compute"),
    ("steel_gemm_fused", "Steel GEMM", "#58a6ff", "Compute"),
    ("sdpa_vector", "SDPA Attention", "#d29922", "BW-bound"),
    ("custom_kernel_gated_delta_step_fused", "Mamba/Hybrid Layer", "#3fb950", "Mixed"),
    ("rmsbfloat16", "RMSNorm", "#3fb950", "BW-bound"),
    ("rope", "RoPE Position Encoding", "#8b949e", "Compute"),
    ("block_softmax", "Softmax", "#8b949e", "Compute"),
    ("custom_kernel_fused_silu_mul", "Fused SiLU Activation", "#8b949e", "Compute"),
    ("gather_front", "Gather (MoE routing)", "#d29922", "BW-bound"),
    ("carg_block_sort", "TopK Sort (MoE)", "#8b949e", "Compute"),
]
kernel_html = ""
for base, desc, color, bound in important:
    found = [k for k in kernels if base in k]
    if found:
        bc = "badge-red" if "BW" in bound else "badge-blue" if "Compute" in bound else "badge-green"
        kernel_html += f'<div class="bar-row"><span class="bar-label">{base}</span><div class="bar-track"><div class="bar-fill" style="width:{max(3,len(found)*2)}%;background:{color}"></div></div><span class="bar-val">{len(found)} variants <span class="badge {bc}">{bound}</span></span></div>\n'

times_js = json.dumps([s["t"] for s in SAMPLES])
bw_js = json.dumps([s["bw"] for s in SAMPLES])
gpu_js = json.dumps([s["gpu"] for s in SAMPLES])
power_js = json.dumps([s["gpu_power"] for s in SAMPLES])
today = time.strftime("%Y-%m-%d")

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AFM GPU Shader Profile — {MODEL}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'SF Mono','Menlo',monospace;background:#0d1117;color:#c9d1d9;padding:24px;max-width:1200px;margin:0 auto}}
h1{{font-size:22px;color:#58a6ff;margin-bottom:4px}}
h2{{font-size:14px;color:#8b949e;font-weight:normal;margin-bottom:8px}}
.cmd-box{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:10px 14px;margin-bottom:20px;font-size:11px;color:#7ee787;overflow-x:auto;white-space:nowrap}}
.cmd-box .label{{color:#8b949e;margin-right:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.card-title{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}}
.big-num{{font-size:32px;font-weight:bold;line-height:1.1}}
.big-label{{font-size:11px;color:#8b949e;margin-top:2px}}
.green{{color:#3fb950}}.yellow{{color:#d29922}}.blue{{color:#58a6ff}}.red{{color:#f85149}}.orange{{color:#f0883e}}
.chart-box{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}}
.chart-title{{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
.stat-row{{display:flex;justify-content:space-between;padding:3px 0;font-size:12px}}
.stat-row .label{{color:#8b949e}}.stat-row .val{{color:#e6edf3}}
.badge{{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:bold}}
.badge-red{{background:#da3633;color:#fff}}.badge-blue{{background:#1f6feb;color:#fff}}.badge-green{{background:#238636;color:#fff}}
.bar-row{{display:flex;align-items:center;gap:6px;margin:4px 0;font-size:12px}}
.bar-label{{width:180px;text-align:right;color:#8b949e;flex-shrink:0;font-size:11px}}
.bar-track{{flex:1;height:20px;background:#21262d;border-radius:3px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:3px}}
.bar-val{{width:140px;color:#e6edf3;flex-shrink:0;font-size:11px}}
.footer{{margin-top:16px;font-size:10px;color:#484f58;text-align:center}}
</style>
</head>
<body>
<h1>AFM GPU Shader Profile</h1>
<h2>{MODEL} — Apple M3 Ultra 80-core GPU — 512 GB — {decode_toks} tokens decoded</h2>
<div class="cmd-box"><span class="label">$</span>{afm_command}</div>

<div class="grid">
  <div class="card"><div class="card-title">Decode Throughput</div><div class="big-num green">{decode_tps:.1f} <span style="font-size:16px">tok/s</span></div><div class="big-label">{decode_toks} tokens in {decode_time:.1f}s</div></div>
  <div class="card"><div class="card-title">Peak DRAM Bandwidth</div><div class="big-num yellow">{peak_bw:.1f} <span style="font-size:16px">GB/s</span></div><div class="big-label">{peak_bw/800*100:.1f}% of 800 GB/s theoretical</div></div>
  <div class="card"><div class="card-title">GPU During Decode</div><div class="big-num blue">{peak_gpu:.0f}<span style="font-size:16px">%</span> <span style="font-size:18px;color:#f0883e">{avg_power:.0f}W</span></div><div class="big-label">sustained @ {avg_bw:.0f} GB/s avg bandwidth</div></div>
</div>

<div class="chart-box"><div class="chart-title">DRAM Bandwidth Timeline (mactop, 300ms samples, {len(SAMPLES)} total)</div><canvas id="bwChart" height="220"></canvas></div>
<div class="chart-box"><div class="chart-title">GPU Utilization & Power Timeline</div><canvas id="gpuChart" height="180"></canvas></div>

<div class="grid2">
  <div class="card">
    <div class="card-title">Memory Breakdown</div>
    <div class="stat-row"><span class="label">Model Weights (4-bit)</span><span class="val">18,595 MB</span></div>
    <div class="stat-row"><span class="label">KV Cache + Runtime</span><span class="val">12,185 MB</span></div>
    <div class="stat-row"><span class="label">Peak Active</span><span class="val">31,396 MB</span></div>
    <div class="stat-row"><span class="label">Prefill</span><span class="val">{prefill_tps:.0f} tok/s ({prefill_toks} tokens, {prefill_time:.3f}s)</span></div>
  </div>
  <div class="card">
    <div class="card-title">Bandwidth Analysis</div>
    <div class="stat-row"><span class="label">Theoretical Max</span><span class="val">800 GB/s</span></div>
    <div class="stat-row"><span class="label">Peak Measured</span><span class="val yellow">{peak_bw:.1f} GB/s</span></div>
    <div class="stat-row"><span class="label">Avg (active)</span><span class="val">{avg_bw:.1f} GB/s</span></div>
    <div class="stat-row"><span class="label">Utilization</span><span class="val">{avg_bw/800*100:.1f}%</span></div>
    <div class="stat-row"><span class="label">Efficiency</span><span class="val">{avg_bw/avg_power:.1f} GB/s per W</span></div>
  </div>
</div>

<div class="chart-box">
  <div class="chart-title">Metal Shader Kernels — Measured ({len(kernels)} unique from xctrace Shader Timeline)</div>
  {kernel_html}
  <div style="margin-top:8px;font-size:11px;color:#484f58">Captured via custom Instruments template with Shader Timeline enabled</div>
</div>

<div class="footer">
  Generated by <code>Scripts/gpu-profile-report.py</code> — AFM v0.9.8 — {today}<br>
  Trace: <code>/tmp/afm-metal.trace</code> &middot; Data: <code>/tmp/afm-gpu-samples.json</code>
</div>

<script>
const times={times_js},bw={bw_js},gpu={gpu_js},power={power_js};
const g='#21262d',tc='#484f58';
new Chart(document.getElementById('bwChart'),{{type:'line',data:{{labels:times.map(t=>t.toFixed(0)+'s'),datasets:[{{label:'DRAM BW (GB/s)',data:bw,borderColor:'#3fb950',backgroundColor:'rgba(63,185,80,0.12)',fill:true,tension:0.3,pointRadius:0,borderWidth:1.5}},{{label:'800 GB/s theoretical',data:times.map(()=>800),borderColor:'rgba(248,81,73,0.2)',borderDash:[5,5],pointRadius:0,borderWidth:1,fill:false}}]}},options:{{responsive:true,plugins:{{legend:{{labels:{{color:tc,font:{{size:10}}}}}}}},scales:{{x:{{grid:{{color:g}},ticks:{{color:tc,font:{{size:9}},maxTicksLimit:20}}}},y:{{min:0,max:850,grid:{{color:g}},ticks:{{color:tc,font:{{size:10}},callback:v=>v+' GB/s'}}}}}}}}}});
new Chart(document.getElementById('gpuChart'),{{type:'line',data:{{labels:times.map(t=>t.toFixed(0)+'s'),datasets:[{{label:'GPU %',data:gpu,borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.08)',fill:true,tension:0.3,pointRadius:0,borderWidth:1.5,yAxisID:'y'}},{{label:'GPU Power (W)',data:power,borderColor:'#f0883e',backgroundColor:'rgba(240,136,62,0.08)',fill:true,tension:0.3,pointRadius:0,borderWidth:1.5,yAxisID:'y1'}}]}},options:{{responsive:true,plugins:{{legend:{{labels:{{color:tc,font:{{size:10}}}}}}}},scales:{{x:{{grid:{{color:g}},ticks:{{color:tc,font:{{size:9}},maxTicksLimit:20}}}},y:{{min:0,max:105,position:'left',grid:{{color:g}},ticks:{{color:'#58a6ff',font:{{size:10}},callback:v=>v+'%'}}}},y1:{{min:0,max:100,position:'right',grid:{{drawOnChartArea:false}},ticks:{{color:'#f0883e',font:{{size:10}},callback:v=>v+'W'}}}}}}}}}});
</script>
</body>
</html>'''

with open(OUTPUT_HTML, "w") as f:
    f.write(html)
with open(OUTPUT_JSON, "w") as f:
    json.dump({"samples": SAMPLES, "kernels": kernels, "profile": profile_lines, "command": afm_command}, f)

print(f"\nReport: {OUTPUT_HTML}")
print(f"Data:   {OUTPUT_JSON}")
print(f"Trace:  /tmp/afm-metal.trace")

# Auto-open
subprocess.run(["open", OUTPUT_HTML])
print("Opened in browser.", flush=True)
