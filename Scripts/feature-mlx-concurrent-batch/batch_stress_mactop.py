#!/usr/bin/env python3
"""Stress test: N concurrent requests with mactop --headless GPU stats.

Uses mactop for accurate GPU metrics (active %, power, frequency, temperature,
DRAM power, system power). This is the recommended stress test script.

Usage:
    python3 batch_stress_mactop.py [concurrency_levels...]
    python3 batch_stress_mactop.py 1 2 4 6 8    # test specific batch sizes
    python3 batch_stress_mactop.py               # defaults to 1 2 4 8

Prerequisites:
    pip install aiohttp
    brew install mactop (or cargo install mactop)
    Server running: afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999 --concurrent

Example output (M3 Ultra, Qwen3.5-35B-A3B-4bit, 500 max_tokens):
    B=1:  85 agg tok/s, GPU 97%, 18W, 36C
    B=2: 152 agg tok/s, GPU 98%, 32W, 38C
    B=4: 225 agg tok/s, GPU 100%, 45W, 40C
    B=8: 315 agg tok/s, GPU 98%, 57W, 44C
"""
import asyncio, aiohttp, json, time, sys, subprocess

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

async def mactop_sampler(interval_ms=300, stop_event=None):
    """Run mactop --headless and collect GPU samples."""
    samples = []
    proc = await asyncio.create_subprocess_exec(
        "mactop", "--headless", "--format", "json",
        "-i", str(interval_ms), "--count", "0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while not stop_event.is_set():
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=2.0)
            if not line:
                break
            try:
                data = json.loads(line.decode())
                samples.append(data)
            except:
                pass
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except:
            proc.kill()
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
        "idx": idx, "tokens": token_count, "elapsed": elapsed,
        "tps": tps, "ttft": ttft,
        "text_preview": text[:50].replace("\n", " "),
    }

def extract_gpu_stats(samples):
    """Extract GPU metrics from mactop JSON samples."""
    if not samples:
        return {}

    gpu_usages = []
    gpu_powers = []
    gpu_freqs = []
    dram_powers = []
    sys_powers = []
    mem_used_gb = []
    gpu_temps = []

    for s in samples:
        if "gpu_usage" in s:
            gpu_usages.append(s["gpu_usage"])
        soc = s.get("soc_metrics", {})
        if "gpu_power" in soc:
            gpu_powers.append(soc["gpu_power"])
        if "gpu_freq_mhz" in soc:
            gpu_freqs.append(soc["gpu_freq_mhz"])
        if "dram_power" in soc:
            dram_powers.append(soc["dram_power"])
        if "system_power" in soc:
            sys_powers.append(soc["system_power"])
        if "gpu_temp" in soc:
            gpu_temps.append(soc["gpu_temp"])
        mem = s.get("memory", {})
        if "used" in mem:
            mem_used_gb.append(mem["used"] / (1024**3))

    def avg(lst): return sum(lst)/len(lst) if lst else 0
    def mx(lst): return max(lst) if lst else 0

    return {
        "gpu_usage_avg": avg(gpu_usages), "gpu_usage_max": mx(gpu_usages),
        "gpu_power_avg": avg(gpu_powers), "gpu_power_max": mx(gpu_powers),
        "gpu_freq_avg": avg(gpu_freqs), "gpu_freq_max": mx(gpu_freqs),
        "dram_power_avg": avg(dram_powers), "dram_power_max": mx(dram_powers),
        "sys_power_avg": avg(sys_powers), "sys_power_max": mx(sys_powers),
        "gpu_temp_avg": avg(gpu_temps), "gpu_temp_max": mx(gpu_temps),
        "mem_used_gb": avg(mem_used_gb),
        "n_samples": len(samples),
    }

async def run_test(N):
    stop_event = asyncio.Event()
    gpu_task = asyncio.create_task(mactop_sampler(interval_ms=300, stop_event=stop_event))

    wall_start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i, PROMPTS[i % len(PROMPTS)]) for i in range(N)]
        results = await asyncio.gather(*tasks)
    wall_elapsed = time.monotonic() - wall_start

    stop_event.set()
    await asyncio.sleep(0.5)
    gpu_samples = await gpu_task

    total_tokens = sum(r["tokens"] for r in results)
    agg_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    steps_per_sec = avg_tps  # Each slot does one step per decode iteration

    stats = extract_gpu_stats(gpu_samples)

    print(f"\n{'='*90}")
    print(f"  B={N:2d} | {MAX_TOKENS} max_tok | {total_tokens} total tok | wall {wall_elapsed:.1f}s")
    print(f"{'='*90}")
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"  Req {r['idx']:2d}: {r['tokens']:3d} tok {r['elapsed']:5.1f}s = {r['tps']:5.1f} t/s  TTFT={r['ttft']:.2f}s  | {r['text_preview']}...")
    print(f"{'='*90}")
    print(f"  Aggregate tok/s:   {agg_tps:>7.1f}    Per-req tok/s: {avg_tps:.1f}")
    print(f"  Steps/sec:         {steps_per_sec:>7.1f}    ms/step: {1000/steps_per_sec:.1f}")
    print(f"  Avg TTFT:          {avg_ttft:>7.2f}s")

    if stats.get("n_samples", 0) > 0:
        print(f"  ---  mactop ({stats['n_samples']} samples)  ---")
        print(f"  GPU Active:        avg={stats['gpu_usage_avg']:5.1f}%   max={stats['gpu_usage_max']:.1f}%")
        print(f"  GPU Power:         avg={stats['gpu_power_avg']:5.1f}W   max={stats['gpu_power_max']:.1f}W")
        print(f"  GPU Freq:          avg={stats['gpu_freq_avg']:5.0f} MHz max={stats['gpu_freq_max']:.0f} MHz")
        print(f"  GPU Temp:          avg={stats['gpu_temp_avg']:5.1f}°C  max={stats['gpu_temp_max']:.1f}°C")
        print(f"  DRAM Power:        avg={stats['dram_power_avg']:5.1f}W   max={stats['dram_power_max']:.1f}W")
        print(f"  System Power:      avg={stats['sys_power_avg']:5.1f}W   max={stats['sys_power_max']:.1f}W")
        print(f"  Memory Used:       {stats['mem_used_gb']:.1f} GB")
    else:
        print(f"  (no mactop samples collected)")

    print(f"{'='*90}")

    return {
        "N": N, "agg_tps": agg_tps, "avg_tps": avg_tps,
        "wall": wall_elapsed, "total_tokens": total_tokens,
        "steps_sec": steps_per_sec, "ms_step": 1000/steps_per_sec if steps_per_sec > 0 else 0,
        "avg_ttft": avg_ttft,
        **{k: v for k, v in stats.items() if k != "raw_keys"},
    }

async def main():
    concurrency_levels = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 4, 8]

    all_results = []
    for N in concurrency_levels:
        r = await run_test(N)
        all_results.append(r)
        await asyncio.sleep(1)

    print(f"\n{'='*100}")
    print(f"  SUMMARY — Qwen3.5-35B-A3B-4bit | max_tokens={MAX_TOKENS}")
    print(f"{'='*100}")
    print(f"  {'B':>3s}  {'Agg t/s':>8s}  {'Per-req':>8s}  {'ms/step':>8s}  {'GPU %':>6s}  {'GPU W':>6s}  {'DRAM W':>7s}  {'Sys W':>6s}  {'Freq':>5s}  {'Temp':>5s}  {'TTFT':>6s}")
    print(f"  {'---':>3s}  {'-------':>8s}  {'-------':>8s}  {'-------':>8s}  {'-----':>6s}  {'-----':>6s}  {'------':>7s}  {'-----':>6s}  {'----':>5s}  {'----':>5s}  {'----':>6s}")
    for r in all_results:
        print(f"  {r['N']:3d}  {r['agg_tps']:8.1f}  {r['avg_tps']:8.1f}  {r['ms_step']:8.1f}"
              f"  {r.get('gpu_usage_avg',0):5.0f}%"
              f"  {r.get('gpu_power_avg',0):5.1f}"
              f"  {r.get('dram_power_avg',0):6.1f}"
              f"  {r.get('sys_power_avg',0):5.1f}"
              f"  {r.get('gpu_freq_avg',0):5.0f}"
              f"  {r.get('gpu_temp_avg',0):4.0f}°"
              f"  {r['avg_ttft']:5.2f}s")
    print(f"{'='*100}")

asyncio.run(main())
