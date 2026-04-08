#!/usr/bin/env python3
"""
Align server log events with trace.jsonl throughput dips on one timeline.

Given a server log (with [BatchScheduler] Batched prefill + [STATS] lines)
and a trace.jsonl from the demo driver, this script:

  1. Parses the log wall-clock timestamps and converts them to relative
     seconds matching the trace's `t` field.
  2. Detects dips in the trace (agg_tps < threshold while inflight is high).
  3. For each dip, reports which events (merges / finishes) happened in a
     ±2 s window around the dip start.
  4. Prints summary counts and the interval distribution of each event type,
     so you can see which one's cadence matches the dip cadence.

Usage:
    python3 Scripts/demo/correlate_dips.py \
        /tmp/afm-demo-server-20260408_183941-31488.log \
        Scripts/demo/out/trace.jsonl
"""
import json
import re
import sys
from collections import defaultdict
from datetime import datetime


LOG_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
PREFILL_RE = re.compile(r"Batched prefill: B=(\d+),")
STATS_RE = re.compile(r"\[STATS\]")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path):
    """Return list of (t_wall_seconds, kind, meta) events."""
    events = []
    t0 = None
    with open(path, errors="replace") as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            m = LOG_TS_RE.search(clean)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            t_wall = ts.timestamp()
            if t0 is None:
                t0 = t_wall
            t_rel = t_wall - t0

            if "Batched prefill" in clean:
                bm = PREFILL_RE.search(clean)
                B = int(bm.group(1)) if bm else 0
                events.append((t_rel, "prefill", {"B": B}))
            elif STATS_RE.search(clean):
                events.append((t_rel, "finish", {}))

    return events, t0


def load_trace(path):
    """Return list of (t, agg_tps, inflight) sorted by t."""
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                (float(d.get("t", 0)), float(d.get("agg_tps", 0) or 0), float(d.get("inflight", 0) or 0))
            )
    rows.sort(key=lambda r: r[0])
    return rows


def detect_dips(rows, peak, frac=0.25, min_inflight=20, min_width_s=1.0):
    """Find dips in the trace. Return list of (t_start, t_end, depth_min)."""
    thresh = peak * frac
    dips = []
    in_dip = False
    start = None
    min_tps = float("inf")
    for t, tps, inf in rows:
        if inf >= min_inflight and tps < thresh:
            if not in_dip:
                in_dip = True
                start = t
                min_tps = tps
            else:
                min_tps = min(min_tps, tps)
        else:
            if in_dip:
                if t - start >= min_width_s:
                    dips.append((start, t, min_tps))
                in_dip = False
                start = None
                min_tps = float("inf")
    return dips


def interval_stats(times):
    """Return (count, mean_interval, stdev_interval) of consecutive intervals."""
    if len(times) < 2:
        return len(times), None, None
    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return len(times), mean, var ** 0.5


def main():
    if len(sys.argv) != 3:
        print("usage: correlate_dips.py <server.log> <trace.jsonl>", file=sys.stderr)
        sys.exit(1)

    log_path, trace_path = sys.argv[1], sys.argv[2]
    events, log_t0 = parse_log(log_path)
    rows = load_trace(trace_path)
    # Look for requests.jsonl next to the trace for precise wall-clock alignment.
    req_path = trace_path.replace("trace.jsonl", "requests.jsonl")

    if not rows:
        print("empty trace")
        return

    peak = max(r[1] for r in rows)
    print(f"=== Trace  : {trace_path}")
    print(f"=== Log    : {log_path}")
    print(f"peak tok/s : {peak:.0f}")
    print(f"t range    : {rows[0][0]:.0f} .. {rows[-1][0]:.0f} s")
    print(f"log events : {len(events)} ({sum(1 for e in events if e[1]=='prefill')} prefill, {sum(1 for e in events if e[1]=='finish')} finish)")

    # --- align timelines: the log uses its own t0, the trace uses the driver's ---
    # The driver starts BEFORE the server sees requests, so align by first prefill.
    first_prefill = next((t for t, k, _ in events if k == "prefill"), None)
    if first_prefill is None:
        print("no prefill events in log")
        return

    # Heuristic: driver's first request corresponds to log's first prefill.
    # Shift log timeline so first prefill aligns with first trace row where inflight > 0.
    first_inflight_t = next((t for t, _, inf in rows if inf > 0), 0)
    offset = first_inflight_t - first_prefill
    aligned_events = [(t + offset, k, m) for t, k, m in events]

    print(f"alignment  : log offset = {offset:+.2f}s (first prefill → first inflight)")

    # --- detect dips ---
    dips = detect_dips(rows, peak)
    print(f"\ndips       : {len(dips)} detected (<25% peak, inflight>=20, width>=1s)")

    # --- interval analysis for each event type and dips ---
    prefill_times = sorted(t for t, k, _ in aligned_events if k == "prefill")
    finish_times = sorted(t for t, k, _ in aligned_events if k == "finish")
    dip_times = [d[0] for d in dips]

    print("\n--- Cadence comparison ---")
    print(f"{'source':<12} {'count':>6} {'mean Δ':>10} {'stdev':>10}")
    for label, times in [
        ("prefills", prefill_times),
        ("finishes", finish_times),
        ("dips",     dip_times),
    ]:
        n, mean, sd = interval_stats(times)
        if mean is None:
            print(f"{label:<12} {n:>6}        --")
        else:
            print(f"{label:<12} {n:>6} {mean:>10.2f} {sd:>10.2f}")

    # --- correlate: for each dip, what events land in a ±2s window? ---
    print("\n--- Per-dip event window (±2s around dip start) ---")
    print(f"{'dip_t':>7} {'width':>6} {'floor':>6} {'prefills':>9} {'finishes':>9}")

    window = 2.0
    prefill_hits_total = 0
    finish_hits_total = 0
    for (ts, te, fl) in dips[:60]:  # cap output
        w = te - ts
        pr_hits = sum(1 for t in prefill_times if ts - window <= t <= ts + window)
        fi_hits = sum(1 for t in finish_times if ts - window <= t <= ts + window)
        prefill_hits_total += pr_hits
        finish_hits_total += fi_hits
        print(f"{ts:>7.1f} {w:>6.1f} {fl:>6.0f} {pr_hits:>9d} {fi_hits:>9d}")
    if len(dips) > 60:
        print(f"... {len(dips)-60} more dips not shown")

    print(f"\nTOTAL in-window events across all dips:")
    print(f"  prefills: {prefill_hits_total} (vs {len(prefill_times)} total prefills)")
    print(f"  finishes: {finish_hits_total} (vs {len(finish_times)} total finishes)")
    print(f"  dips    : {len(dips)}")

    if len(dips) > 0:
        print(f"\n  prefill coincidence rate: {prefill_hits_total/len(dips):.2f} per dip")
        print(f"  finish  coincidence rate: {finish_hits_total/len(dips):.2f} per dip")
    if prefill_times:
        print(f"  prefills falling in dip windows: {prefill_hits_total/len(prefill_times)*100:.0f}% of all prefills")
    if finish_times:
        print(f"  finishes falling in dip windows: {finish_hits_total/len(finish_times)*100:.0f}% of all finishes")

    print("""
Interpretation guide:
  - If dips_mean_interval ≈ prefills_mean_interval  → merges drive dips
  - If dips_mean_interval ≈ finishes_mean_interval  → filter() drives dips
  - If neither matches                              → something else (e.g. 512-step flush, GC)
  - Coincidence rate > 0.7 for one event type is strong causal evidence.
""")


if __name__ == "__main__":
    main()
