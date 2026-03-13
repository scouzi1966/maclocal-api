#!/usr/bin/env python3
"""Stress test: N concurrent requests with GPU utilization sampling via ioreg.

Uses ioreg AGXAccelerator for GPU stats (coarse/unreliable — prefer batch_stress_mactop.py).

Usage:
    python3 batch_stress_ioreg.py [concurrency_levels...]
    python3 batch_stress_ioreg.py 1 2 4 8       # test specific batch sizes
    python3 batch_stress_ioreg.py                # defaults to 1 2 4 8

Prerequisites:
    pip install aiohttp
    Server running: afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999 --concurrent

Note: ioreg "Device Utilization %" is unreliable/coarse (shows 71-88% while actual GPU
active is 97-99%). Use batch_stress_mactop.py with mactop --headless for accurate stats.
"""
import asyncio, aiohttp, json, time, sys, subprocess, re

URL = "http://localhost:9999/v1/chat/completions"
MAX_TOKENS = 500

PROMPTS = [
    "Explain the Pythagorean theorem in detail with examples.",
    "What is the capital of Japan and why is it historically significant?",
    "Write a poem about the ocean, at least 4 stanzas.",
    "Describe how a combustion engine works step by step.",
    "What are the main differences between Python and JavaScript? Give code examples.",
    "Explain photosynthesis in detail.",
    "What causes thunder and lightning? Explain the physics.",
    "How does Wi-Fi work? Explain the radio protocol.",
]

def get_gpu_stats():
    """Read GPU stats from ioreg (AGXAccelerator)."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-c", "AGXAccelerator"],
            timeout=2, text=True
        )
        for line in out.split("\n"):
            if "Device Utilization %" in line:
                dev = re.search(r'"Device Utilization %"=(\d+)', line)
                ren = re.search(r'"Renderer Utilization %"=(\d+)', line)
                mem = re.search(r'"In use system memory"=(\d+)', line)
                alloc = re.search(r'"Alloc system memory"=(\d+)', line)
                return {
                    "device": int(dev.group(1)) if dev else 0,
                    "renderer": int(ren.group(1)) if ren else 0,
                    "in_use_gb": int(mem.group(1)) / (1024**3) if mem else 0,
                    "alloc_gb": int(alloc.group(1)) / (1024**3) if alloc else 0,
                }
    except:
        pass
    return None

async def gpu_sampler(interval=0.3, stop_event=None):
    """Sample GPU utilization in background."""
    samples = []
    while not stop_event.is_set():
        stats = await asyncio.get_event_loop().run_in_executor(None, get_gpu_stats)
        if stats:
            samples.append(stats)
        await asyncio.sleep(interval)
    return samples

async def send_request(session, idx, prompt):
    payload = {
        "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "temperature": 0.7,
    }
    start = time.monotonic()
    first_token_time = None
    token_count = 0
    text = ""

    async with session.post(URL, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or delta.get("reasoning_content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.monotonic() - start
                    token_count += 1
                    text += content
            except:
                pass

    elapsed = time.monotonic() - start
    tps = token_count / elapsed if elapsed > 0 else 0
    ttft = first_token_time or 0
    return {
        "idx": idx,
        "tokens": token_count,
        "elapsed": elapsed,
        "tps": tps,
        "ttft": ttft,
        "text_preview": text[:60].replace("\n", " "),
    }

async def run_test(N):
    stop_event = asyncio.Event()
    gpu_task = asyncio.create_task(gpu_sampler(interval=0.3, stop_event=stop_event))

    wall_start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i, PROMPTS[i % len(PROMPTS)]) for i in range(N)]
        results = await asyncio.gather(*tasks)
    wall_elapsed = time.monotonic() - wall_start

    stop_event.set()
    gpu_samples = await gpu_task

    total_tokens = sum(r["tokens"] for r in results)
    agg_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results)

    if gpu_samples:
        gpu_dev_avg = sum(s["device"] for s in gpu_samples) / len(gpu_samples)
        gpu_dev_max = max(s["device"] for s in gpu_samples)
        gpu_ren_avg = sum(s["renderer"] for s in gpu_samples) / len(gpu_samples)
        gpu_ren_max = max(s["renderer"] for s in gpu_samples)
        gpu_mem_avg = sum(s["in_use_gb"] for s in gpu_samples) / len(gpu_samples)
        gpu_mem_max = max(s["in_use_gb"] for s in gpu_samples)
        gpu_alloc = gpu_samples[-1]["alloc_gb"]
    else:
        gpu_dev_avg = gpu_dev_max = gpu_ren_avg = gpu_ren_max = -1
        gpu_mem_avg = gpu_mem_max = gpu_alloc = -1

    print(f"\n{'='*82}")
    print(f"  B={N:2d} | {MAX_TOKENS} max_tok | {total_tokens} total tok | wall {wall_elapsed:.1f}s")
    print(f"{'='*82}")
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"  Req {r['idx']:2d}: {r['tokens']:3d} tok {r['elapsed']:5.1f}s = {r['tps']:5.1f} t/s  TTFT={r['ttft']:.2f}s  | {r['text_preview']}...")
    print(f"{'='*82}")
    print(f"  Aggregate tok/s:   {agg_tps:>7.1f}")
    print(f"  Avg per-req tok/s: {avg_tps:>7.1f}")
    print(f"  Avg TTFT:          {avg_ttft:>7.2f}s")
    print(f"  GPU Device:        avg={gpu_dev_avg:4.0f}%  max={gpu_dev_max}%")
    print(f"  GPU Renderer:      avg={gpu_ren_avg:4.0f}%  max={gpu_ren_max}%")
    print(f"  GPU Memory:        avg={gpu_mem_avg:5.1f}GB  max={gpu_mem_max:.1f}GB  alloc={gpu_alloc:.1f}GB")
    print(f"  ({len(gpu_samples)} GPU samples)")
    print(f"{'='*82}")

    return {
        "N": N, "agg_tps": agg_tps, "avg_tps": avg_tps,
        "wall": wall_elapsed, "total_tokens": total_tokens,
        "gpu_dev": gpu_dev_avg, "gpu_dev_max": gpu_dev_max,
        "gpu_ren": gpu_ren_avg, "gpu_ren_max": gpu_ren_max,
        "gpu_mem": gpu_mem_max, "gpu_alloc": gpu_alloc,
        "avg_ttft": avg_ttft,
    }

async def main():
    concurrency_levels = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 4, 8]

    all_results = []
    for N in concurrency_levels:
        r = await run_test(N)
        all_results.append(r)
        await asyncio.sleep(1)  # brief cooldown

    print(f"\n{'='*82}")
    print(f"  SUMMARY — Qwen3.5-35B-A3B-4bit | max_tokens={MAX_TOKENS}")
    print(f"{'='*82}")
    print(f"  {'B':>3s}  {'Agg t/s':>8s}  {'Per-req':>8s}  {'Wall':>6s}  {'Tokens':>7s}  {'GPU Dev':>8s}  {'GPU Ren':>8s}  {'Mem GB':>7s}  {'TTFT':>6s}")
    print(f"  {'---':>3s}  {'-------':>8s}  {'-------':>8s}  {'----':>6s}  {'------':>7s}  {'-------':>8s}  {'-------':>8s}  {'------':>7s}  {'----':>6s}")
    for r in all_results:
        print(f"  {r['N']:3d}  {r['agg_tps']:8.1f}  {r['avg_tps']:8.1f}  {r['wall']:5.1f}s  {r['total_tokens']:7d}  {r['gpu_dev']:6.0f}%  {r['gpu_ren']:6.0f}%  {r['gpu_mem']:6.1f}  {r['avg_ttft']:5.2f}s")
    print(f"{'='*82}")

asyncio.run(main())
