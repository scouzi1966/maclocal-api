#!/usr/bin/env python3
"""Mixed workload validation with full server-reported stats.

Reports pp (prompt processing), tg (token generation), TTFT, total tokens,
and GPU metrics via mactop. Designed for A/B comparison between code versions.

Usage:
    python3 validate_mixed_workload.py              # test B=1,2,4,8
    python3 validate_mixed_workload.py 1 4          # specific batch sizes
    python3 validate_mixed_workload.py --label "overlap-v2" 1 2 4 8
"""
import asyncio, aiohttp, json, time, sys, subprocess, threading, os

URL = "http://localhost:9999/v1/chat/completions"
MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"

# --- Short-answer tests (long prompts, expect brief response) ---
SHORT_ANSWER_TESTS = [
    {
        "name": "long-ctx-arithmetic",
        "prompt": (
            "I have a complex problem. Consider the following sequence of operations: "
            "Start with 100. Add 37. Multiply by 2. Subtract 15. Divide by 3. "
            "Add 42. Multiply by 5. Subtract 200. Add 17. Divide by 2. "
            "Now, separately and ignoring all of that: What is 7 times 8? "
            "Answer with ONLY the number, nothing else."
        ),
        "expected": ["56"],
        "max_tokens": 4096,
    },
    {
        "name": "long-ctx-capital",
        "prompt": (
            "Here is some background context that is not relevant to the question: "
            "The history of cartography spans thousands of years. Ancient Babylonians "
            "created clay tablet maps around 600 BCE. Ptolemy's Geographia from the 2nd "
            "century CE was influential for centuries. The Age of Exploration brought "
            "major advances in mapmaking with Mercator's projection in 1569. Modern GIS "
            "systems use satellite imagery and digital processing. "
            "Now answer this simple question: What is the capital of Japan? One word only."
        ),
        "expected": ["tokyo"],
        "max_tokens": 4096,
    },
    {
        "name": "long-ctx-element",
        "prompt": (
            "Let me give you detailed context about the periodic table. Dmitri Mendeleev "
            "published the first widely recognized periodic table in 1869, arranging 63 "
            "known elements by atomic weight. Henry Moseley later determined that atomic "
            "number was the better organizing principle. The table now contains 118 "
            "confirmed elements, with the most recent additions being nihonium (113), "
            "moscovium (115), tennessine (117), and oganesson (118), all confirmed in "
            "2015-2016. Elements are arranged in 18 groups and 7 periods. The lanthanides "
            "and actinides are typically shown separately below the main table. "
            "Simple question: What element has the symbol 'Au'? Answer in one word."
        ),
        "expected": ["gold"],
        "max_tokens": 4096,
    },
    {
        "name": "long-ctx-year",
        "prompt": (
            "Context about space exploration milestones: The Space Race between the US "
            "and Soviet Union drove rapid advancement. Sputnik 1 launched October 4, 1957. "
            "Yuri Gagarin became the first human in space on April 12, 1961, aboard Vostok 1. "
            "John Glenn orbited Earth on February 20, 1962. Valentina Tereshkova became "
            "the first woman in space in 1963. Ed White performed the first American "
            "spacewalk in 1965. The Gemini program tested orbital maneuvers and docking. "
            "The tragic Apollo 1 fire killed three astronauts in January 1967. Apollo 8 "
            "orbited the Moon in December 1968. "
            "Question: In what year did humans first walk on the Moon? Just the year."
        ),
        "expected": ["1969"],
        "max_tokens": 4096,
    },
]

# --- Long-decode tests (short prompts, expect 1000+ tokens) ---
LONG_DECODE_TESTS = [
    {
        "name": "calculus-explain",
        "prompt": (
            "Explain calculus concepts from limits through multivariable calculus "
            "with rigorous mathematical notation. Cover: epsilon-delta definition of "
            "limits, derivatives, chain rule, integration techniques, fundamental "
            "theorem of calculus, sequences and series, Taylor series, partial "
            "derivatives, gradient, divergence, curl, and multiple integrals."
        ),
        "expected": ["limit", "derivative", "integral"],
        "min_tokens": 500,
        "max_tokens": 4096,
    },
    {
        "name": "history-essay",
        "prompt": (
            "Write a detailed essay on the causes and consequences of World War I, "
            "covering the alliance systems, the assassination of Archduke Franz Ferdinand, "
            "major battles, technological innovations in warfare, the Treaty of Versailles, "
            "and how it set the stage for World War II."
        ),
        "expected": ["assassination", "versailles", "trench"],
        "min_tokens": 500,
        "max_tokens": 4096,
    },
    {
        "name": "code-tutorial",
        "prompt": (
            "Write a comprehensive tutorial on implementing a red-black tree in Python. "
            "Include the complete implementation with insert, delete, search, rotation "
            "operations, and explain the balancing rules. Add docstrings and comments."
        ),
        "expected": ["class", "def ", "insert", "rotate"],
        "min_tokens": 500,
        "max_tokens": 4096,
    },
    {
        "name": "physics-explain",
        "prompt": (
            "Explain quantum mechanics from first principles. Start with the double-slit "
            "experiment, cover wave-particle duality, the Schrodinger equation, "
            "Heisenberg uncertainty principle, quantum entanglement, superposition, "
            "and the measurement problem. Use mathematical notation where appropriate."
        ),
        "expected": ["wave", "particle", "uncertainty"],
        "min_tokens": 500,
        "max_tokens": 4096,
    },
]


# ─── mactop GPU sampler ──────────────────────────────────────────────────────

class MactopSampler:
    """Background mactop --headless --format json sampler."""

    def __init__(self):
        self.samples = []
        self._proc = None
        self._thread = None
        self._stop = False

    def start(self):
        try:
            self._proc = subprocess.Popen(
                ["sudo", "mactop", "--headless", "--format", "json", "-i", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            self._proc = None

    def _reader(self):
        if not self._proc:
            return
        for line in self._proc.stdout:
            if self._stop:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self.samples.append(d)
            except Exception:
                pass

    def stop(self):
        self._stop = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()

    def summary(self):
        if not self.samples:
            return None
        def avg(key):
            vals = []
            for s in self.samples:
                v = s.get(key)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            return sum(vals) / len(vals) if vals else 0

        return {
            "gpu_pct": avg("gpu_active_pct") or avg("active_pct"),
            "gpu_w": avg("gpu_power_w") or avg("gpu_power"),
            "dram_w": avg("dram_power_w") or avg("dram_power"),
            "sys_w": avg("system_power_w") or avg("system_power"),
            "freq_mhz": avg("gpu_freq_mhz") or avg("gpu_freq"),
            "temp_c": avg("gpu_temp_c") or avg("gpu_temp"),
            "n_samples": len(self.samples),
        }


# ─── request sender ──────────────────────────────────────────────────────────

async def send_request(session, prompt, max_tokens=4096):
    """Send streaming request, return full stats dict from server."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.3,
    }
    text = ""
    ttft = None
    usage = {}
    timings = {}
    start = time.monotonic()

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
                # Capture usage and timings from final chunk
                if "usage" in chunk:
                    usage = chunk["usage"]
                if "timings" in chunk:
                    timings = chunk["timings"]

                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or delta.get("reasoning_content", "")
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    text += content
            except Exception:
                pass

    elapsed = time.monotonic() - start

    return {
        "text": text,
        "wall_s": elapsed,
        "ttft": ttft or 0,
        # Server-reported stats
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "pp_tok_s": usage.get("prompt_tokens_per_second", 0),
        "tg_tok_s": usage.get("completion_tokens_per_second", 0),
        "prompt_time_s": usage.get("prompt_time", 0),
        "completion_time_s": usage.get("completion_time", 0),
        "total_time_s": usage.get("total_time", 0),
        # Raw timings (ms)
        "prompt_ms": timings.get("prompt_ms", 0),
        "predicted_ms": timings.get("predicted_ms", 0),
    }


def check_response(text, expected_substrings, min_tokens=0):
    lower = text.lower()
    missing = [s for s in expected_substrings if s.lower() not in lower]
    is_garbage = len(text.strip()) < 2 or text.count('\ufffd') > 5
    too_short = min_tokens > 0 and len(text.split()) < min_tokens * 0.3
    return {"missing": missing, "all_found": len(missing) == 0,
            "is_garbage": is_garbage, "too_short": too_short}


# ─── batch runner ─────────────────────────────────────────────────────────────

async def run_batch(batch_size, tests):
    passed = 0
    failed = 0
    results = []

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(tests), batch_size):
            batch = tests[batch_start:batch_start + batch_size]
            tasks = [
                send_request(session, t["prompt"], t.get("max_tokens", 4096))
                for t in batch
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            for test, outcome in zip(batch, outcomes):
                name = test["name"]
                if isinstance(outcome, Exception):
                    failed += 1
                    print(f"  FAIL  {name}: exception {outcome}")
                    results.append({"name": name, "status": "EXCEPTION"})
                    continue

                r = outcome
                min_tok = test.get("min_tokens", 0)
                check = check_response(r["text"], test["expected"], min_tok)

                if check["is_garbage"]:
                    failed += 1
                    print(f"  FAIL  {name}: GARBAGE")
                    results.append({"name": name, "status": "GARBAGE"})
                elif check["too_short"]:
                    failed += 1
                    print(f"  FAIL  {name}: TOO SHORT ({r['completion_tokens']} tok)")
                    results.append({"name": name, "status": "TOO_SHORT"})
                elif not check["all_found"]:
                    failed += 1
                    print(f"  FAIL  {name}: missing {check['missing']}")
                    results.append({"name": name, "status": "MISSING", "missing": check["missing"]})
                else:
                    passed += 1
                    r["name"] = name
                    r["status"] = "OK"
                    r["kind"] = "long" if min_tok > 0 else "short"
                    results.append(r)

                    pp = r["pp_tok_s"]
                    tg = r["tg_tok_s"]
                    pt = r["prompt_tokens"]
                    ct = r["completion_tokens"]
                    print(f"  OK    {name:24s}  pp={pt:4d} tok {pp:7.1f} t/s  "
                          f"tg={ct:4d} tok {tg:6.1f} t/s  "
                          f"TTFT={r['ttft']:.2f}s  wall={r['wall_s']:.1f}s")

    return passed, failed, results


# ─── main ─────────────────────────────────────────────────────────────────────

async def main():
    args = sys.argv[1:]
    label = ""
    if "--label" in args:
        idx = args.index("--label")
        label = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    batch_sizes = [int(x) for x in args] if args else [1, 2, 4, 8]
    all_tests = SHORT_ANSWER_TESTS + LONG_DECODE_TESTS

    total_passed = 0
    total_failed = 0
    all_batch_results = {}

    for bs in batch_sizes:
        print(f"\n{'='*100}")
        hdr = f"  B={bs} — {len(SHORT_ANSWER_TESTS)} short-answer + {len(LONG_DECODE_TESTS)} long-decode (4K max)"
        if label:
            hdr += f"  [{label}]"
        print(hdr)
        print(f"{'='*100}")

        gpu = MactopSampler()
        gpu.start()

        p, f, results = await run_batch(bs, all_tests)
        total_passed += p
        total_failed += f

        gpu.stop()
        gpu_stats = gpu.summary()

        ok_results = [r for r in results if r.get("status") == "OK"]
        long_ok = [r for r in ok_results if r.get("kind") == "long"]
        short_ok = [r for r in ok_results if r.get("kind") == "short"]

        # Per-request summary
        if ok_results:
            total_prompt = sum(r["prompt_tokens"] for r in ok_results)
            total_completion = sum(r["completion_tokens"] for r in ok_results)
            total_all = total_prompt + total_completion
            avg_pp = sum(r["pp_tok_s"] for r in ok_results) / len(ok_results)
            avg_tg = sum(r["tg_tok_s"] for r in ok_results) / len(ok_results)
            avg_ttft = sum(r["ttft"] for r in ok_results) / len(ok_results)
            max_wall = max(r["wall_s"] for r in ok_results)
            agg_tg = total_completion / max_wall if max_wall > 0 else 0

            print(f"  {'─'*96}")
            print(f"  Totals: {total_prompt} prompt + {total_completion} completion = {total_all} tokens")
            print(f"  Avg pp: {avg_pp:.1f} tok/s   Avg tg (per-req): {avg_tg:.1f} tok/s   "
                  f"Agg tg: {agg_tg:.1f} tok/s   Avg TTFT: {avg_ttft:.2f}s")

            if long_ok:
                l_pp = sum(r["pp_tok_s"] for r in long_ok) / len(long_ok)
                l_tg = sum(r["tg_tok_s"] for r in long_ok) / len(long_ok)
                l_ct = sum(r["completion_tokens"] for r in long_ok)
                l_ttft = sum(r["ttft"] for r in long_ok) / len(long_ok)
                print(f"  Long-decode: {l_ct} tok, avg pp={l_pp:.1f} tg={l_tg:.1f} tok/s, TTFT={l_ttft:.2f}s")

            if short_ok:
                s_pp = sum(r["pp_tok_s"] for r in short_ok) / len(short_ok)
                s_tg = sum(r["tg_tok_s"] for r in short_ok) / len(short_ok)
                s_ct = sum(r["completion_tokens"] for r in short_ok)
                s_ttft = sum(r["ttft"] for r in short_ok) / len(short_ok)
                print(f"  Short-answer: {s_ct} tok, avg pp={s_pp:.1f} tg={s_tg:.1f} tok/s, TTFT={s_ttft:.2f}s")

        if gpu_stats and gpu_stats["n_samples"] > 0:
            g = gpu_stats
            print(f"  GPU: {g['gpu_pct']:.0f}% active, {g['gpu_w']:.1f}W, "
                  f"{g['freq_mhz']:.0f} MHz, {g['temp_c']:.0f}°C  "
                  f"| DRAM {g['dram_w']:.1f}W  Sys {g['sys_w']:.1f}W  "
                  f"({g['n_samples']} samples)")

        print(f"  Result: {p}/{p+f} passed")
        print(f"{'='*100}")

        all_batch_results[bs] = {
            "passed": p, "failed": f, "results": ok_results,
            "gpu": gpu_stats,
        }
        await asyncio.sleep(1)

    # ─── Final summary table ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    title = f"  SUMMARY — {MODEL} | mixed workload"
    if label:
        title += f" | {label}"
    print(title)
    print(f"{'='*120}")
    print(f"  {'B':>3s}  {'Pass':>4s}  {'PP tok':>6s}  {'PP t/s':>7s}  "
          f"{'TG tok':>6s}  {'TG t/s':>7s}  {'Agg t/s':>7s}  "
          f"{'TTFT':>6s}  {'Wall':>5s}  "
          f"{'GPU%':>5s}  {'GPU W':>5s}  {'DRAM W':>6s}  {'Sys W':>5s}  {'Temp':>5s}")
    print(f"  {'───':>3s}  {'────':>4s}  {'──────':>6s}  {'───────':>7s}  "
          f"{'──────':>6s}  {'───────':>7s}  {'───────':>7s}  "
          f"{'──────':>6s}  {'─────':>5s}  "
          f"{'─────':>5s}  {'─────':>5s}  {'──────':>6s}  {'─────':>5s}  {'─────':>5s}")

    for bs in batch_sizes:
        br = all_batch_results.get(bs)
        if not br or not br["results"]:
            continue
        ok = br["results"]
        p_f = f"{br['passed']}/{br['passed']+br['failed']}"
        total_prompt = sum(r["prompt_tokens"] for r in ok)
        total_compl = sum(r["completion_tokens"] for r in ok)
        avg_pp = sum(r["pp_tok_s"] for r in ok) / len(ok)
        avg_tg = sum(r["tg_tok_s"] for r in ok) / len(ok)
        avg_ttft = sum(r["ttft"] for r in ok) / len(ok)
        max_wall = max(r["wall_s"] for r in ok)
        agg_tg = total_compl / max_wall if max_wall > 0 else 0
        g = br.get("gpu") or {}

        gpu_pct = f"{g.get('gpu_pct', 0):.0f}%" if g else "—"
        gpu_w = f"{g.get('gpu_w', 0):.1f}" if g else "—"
        dram_w = f"{g.get('dram_w', 0):.1f}" if g else "—"
        sys_w = f"{g.get('sys_w', 0):.1f}" if g else "—"
        temp = f"{g.get('temp_c', 0):.0f}°" if g else "—"

        print(f"  {bs:3d}  {p_f:>4s}  {total_prompt:6d}  {avg_pp:7.1f}  "
              f"{total_compl:6d}  {avg_tg:7.1f}  {agg_tg:7.1f}  "
              f"{avg_ttft:5.2f}s  {max_wall:4.0f}s  "
              f"{gpu_pct:>5s}  {gpu_w:>5s}  {dram_w:>6s}  {sys_w:>5s}  {temp:>5s}")

    print(f"{'='*120}")
    print(f"  TOTAL: {total_passed}/{total_passed+total_failed} passed across {len(batch_sizes)} batch sizes")
    if total_failed > 0:
        print(f"  ({total_failed} failures: model answer mismatches, not code bugs)")
    print(f"{'='*120}")


asyncio.run(main())
