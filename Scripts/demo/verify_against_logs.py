#!/usr/bin/env python3
"""
Cross-check the driver's observed aggregate tok/s (from trace.jsonl)
against the server's ground-truth per-request [STATS] lines from the
afm mlx log. Detects measurement drift, false peaks, and real dips.

Reads two inputs:
    --trace       driver's per-250ms JSONL trace
    --server-log  the afm mlx stdout log captured by the bash wrapper

Produces:
    - stdout table comparing per-second driver vs server tok/s
    - alignment score (Pearson correlation)
    - flagged dips (server-side) with timestamp and surrounding context
    - cache hit rate + per-request rate distribution from STATS
    - optional CSV output (--csv PATH)

Usage:
    python3 Scripts/demo/verify_against_logs.py \\
        --trace Scripts/demo/out/trace.jsonl \\
        --server-log /tmp/afm-demo-server-YYYYMMDD_HHMMSS-PID.log
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STATS_RX = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\] \[STATS\] "
    r"pp:\s*(\d+) tok,\s*([\d.]+)s\s*\(([\d.]+) tok/s\)\s*\|\s*"
    r"tg:\s*(\d+) tok,\s*([\d.]+)s\s*\(([\d.]+) tok/s\)\s*\|\s*"
    r"cache:\s*(MISS|HIT [\d/]+ \(\d+%\))"
)
POST_RX = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\] \[INFO\] "
    r"POST /v1/chat/completions \[request-id:\s*([^\]]+)\]"
)


@dataclass
class StatRecord:
    end: datetime
    pp_tok: int
    pp_sec: float
    pp_tps: float
    tg_tok: int
    tg_sec: float
    tg_tps: float
    cache: str

    @property
    def decode_rate(self) -> float:
        return self.tg_tok / self.tg_sec if self.tg_sec > 0 else 0.0


def parse_server_log(path: Path) -> tuple[list[StatRecord], list[str]]:
    """Returns (STATS records, list of request-ids seen in POST lines)."""
    text = path.read_text()
    text = ANSI.sub("", text)
    records: list[StatRecord] = []
    for m in STATS_RX.finditer(text):
        records.append(
            StatRecord(
                end=datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f"),
                pp_tok=int(m.group(2)),
                pp_sec=float(m.group(3)),
                pp_tps=float(m.group(4)),
                tg_tok=int(m.group(5)),
                tg_sec=float(m.group(6)),
                tg_tps=float(m.group(7)),
                cache=m.group(8),
            )
        )
    post_ids: list[str] = []
    for m in POST_RX.finditer(text):
        post_ids.append(m.group(2).strip())
    return records, post_ids


def parse_trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def server_per_second_decode(records: list[StatRecord]) -> tuple[float, list[float]]:
    """
    Given completed requests, distribute each request's decode tokens
    across its 1-second decode window. Returns (t0_epoch, buckets).
    The bucket index corresponds to seconds since the first record's
    earliest decode start time.
    """
    if not records:
        return 0.0, []
    # Work in epoch seconds for simplicity
    t0_epoch = min((r.end.timestamp() - r.tg_sec) for r in records)
    t_end_epoch = max(r.end.timestamp() for r in records)
    total_sec = int(math.ceil(t_end_epoch - t0_epoch)) + 1
    buckets = [0.0] * (total_sec + 2)

    for r in records:
        rate = r.decode_rate
        decode_start = r.end.timestamp() - r.tg_sec - t0_epoch
        decode_end = r.end.timestamp() - t0_epoch
        b_lo = max(0, int(math.floor(decode_start)))
        b_hi = min(len(buckets) - 1, int(math.floor(decode_end)))
        for b in range(b_lo, b_hi + 1):
            overlap = min(b + 1, decode_end) - max(b, decode_start)
            if overlap > 0:
                buckets[b] += rate * overlap
    return t0_epoch, buckets


def driver_per_second(trace: list[dict]) -> list[float]:
    """Bucket the driver's sampled agg_tps into 1-second buckets (mean over samples)."""
    if not trace:
        return []
    t_last = max(s["t"] for s in trace)
    total = int(math.ceil(t_last)) + 1
    sums = [0.0] * (total + 2)
    counts = [0] * (total + 2)
    for s in trace:
        b = int(math.floor(s["t"]))
        if 0 <= b < len(sums):
            sums[b] += s["agg_tps"]
            counts[b] += 1
    return [sums[i] / counts[i] if counts[i] > 0 else 0.0 for i in range(len(sums))]


def pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den_a = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def find_dips(series: list[float], threshold_frac: float = 0.2) -> list[tuple[int, int, float]]:
    """
    Identify dip regions where the series falls below threshold_frac *
    peak for at least 2 consecutive seconds after the peak has been
    reached. Returns list of (start_s, end_s, min_val_in_dip).
    """
    if not series:
        return []
    peak = max(series)
    if peak <= 0:
        return []
    cutoff = peak * threshold_frac
    # Only look for dips after reaching half the peak (avoid the initial ramp).
    started = False
    dips: list[tuple[int, int, float]] = []
    in_dip = False
    dip_start = 0
    dip_min = float("inf")
    for i, v in enumerate(series):
        if not started and v >= peak * 0.5:
            started = True
        if not started:
            continue
        if v < cutoff:
            if not in_dip:
                in_dip = True
                dip_start = i
                dip_min = v
            else:
                dip_min = min(dip_min, v)
        else:
            if in_dip and i - dip_start >= 2:
                dips.append((dip_start, i - 1, dip_min))
            in_dip = False
    if in_dip:
        dips.append((dip_start, len(series) - 1, dip_min))
    return dips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="Scripts/demo/out/trace.jsonl")
    ap.add_argument("--requests", default="Scripts/demo/out/requests.jsonl",
                    help="Per-request driver capture (for corr_id integrity check).")
    ap.add_argument("--server-log", required=True,
                    help="Path to the afm mlx stdout log captured during the run.")
    ap.add_argument("--csv", default=None,
                    help="Optional CSV output with per-second driver vs server numbers.")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    log_path = Path(args.server_log)
    if not trace_path.exists():
        print(f"ERROR: trace not found at {trace_path}", file=sys.stderr)
        sys.exit(1)
    if not log_path.exists():
        print(f"ERROR: server log not found at {log_path}", file=sys.stderr)
        sys.exit(1)

    trace = parse_trace(trace_path)
    stats, post_ids = parse_server_log(log_path)

    print(f"driver trace  : {trace_path} ({len(trace)} samples)")
    print(f"server log    : {log_path}")
    print(f"server STATS  : {len(stats)} completed requests")
    print(f"server POSTs  : {len(post_ids)} /v1/chat/completions requests received")
    print()

    # --- Integrity: driver corr_id vs server request-id ---------------------
    req_path = Path(args.requests)
    if req_path.exists():
        driver_records: list[dict] = []
        with req_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    driver_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        driver_ids = {r.get("corr_id") for r in driver_records if r.get("corr_id")}
        driver_errors = [r for r in driver_records if r.get("error")]

        # Skip the tiny warmup request that the bash wrapper sends before the
        # driver starts — those POSTs have UUID request-ids assigned by Vapor
        # (not our u###-r####-###### pattern).
        server_corr_matches = [p for p in post_ids if p in driver_ids]
        server_unmatched = [p for p in post_ids if p not in driver_ids]

        print("=== Integrity: corr_id cross-check ===")
        print(f"  driver records (out/requests.jsonl) : {len(driver_records)}")
        print(f"  server POST lines with matching id  : {len(server_corr_matches)}")
        print(f"  server POST lines without match     : {len(server_unmatched)} (warmup / probe requests)")
        print(f"  driver errors                       : {len(driver_errors)}")
        if len(driver_records) == len(server_corr_matches):
            print("  STATUS: PASS — every driver request landed on the server")
        else:
            missing = driver_ids - set(post_ids)
            print(f"  STATUS: FAIL — {len(missing)} driver request(s) have no server POST line")
            for m in list(missing)[:5]:
                print(f"    missing: {m}")
        # Server should also have produced one STATS line per successful POST
        # that made it out of slot-queueing. Count comparison:
        expected_stats = len(driver_records) - len(driver_errors)
        print(f"  expected STATS lines                : {expected_stats}")
        print(f"  actual STATS lines                  : {len(stats)}")
        print()
    else:
        print(f"(no requests.jsonl at {req_path} — skipping corr_id integrity check)")
        print()

    if not stats:
        print("ERROR: no STATS lines in server log — nothing to verify against.")
        sys.exit(2)

    # --- Per-request rate distribution --------------------------------------
    rates = [s.decode_rate for s in stats]
    print("=== Per-request decode rate (server truth) ===")
    print(f"  min={min(rates):.1f}  p50={statistics.median(rates):.1f}  "
          f"p95={statistics.quantiles(rates, n=20)[-1]:.1f}  "
          f"max={max(rates):.1f}  mean={statistics.mean(rates):.1f}")
    print()

    # --- Cache hit rate -----------------------------------------------------
    hits = sum(1 for s in stats if s.cache.startswith("HIT"))
    misses = sum(1 for s in stats if s.cache == "MISS")
    print("=== Prefix cache hit rate ===")
    print(f"  HIT: {hits}/{hits+misses}  ({100*hits/max(1, hits+misses):.1f}%)")
    if hits < 0.1 * (hits + misses):
        print("  NOTE: low hit rate suggests the prompt pool has no shared prefix.")
        print("        Run with --mode prefix-cache to exercise the radix tree.")
    print()

    # --- Per-second aggregate decode: driver vs server ----------------------
    t0, server_bucket = server_per_second_decode(stats)
    driver_bucket = driver_per_second(trace)

    # Driver trace starts around when the driver posted its first request.
    # Server log first STATS line ends AFTER the first request completes,
    # so the two bucket series have different origins. To compare, we align
    # them by trimming leading zeros from both and using the first non-zero
    # bucket as t=0.
    def first_nonzero(xs: list[float]) -> int:
        for i, v in enumerate(xs):
            if v > 0:
                return i
        return 0

    d_off = first_nonzero(driver_bucket)
    s_off = first_nonzero(server_bucket)
    driver_aligned = driver_bucket[d_off:]
    server_aligned = server_bucket[s_off:]
    n = min(len(driver_aligned), len(server_aligned))
    driver_aligned = driver_aligned[:n]
    server_aligned = server_aligned[:n]

    corr = pearson(driver_aligned, server_aligned)
    print("=== Per-second agg tok/s: driver observation vs server truth ===")
    print(f"  aligned window: {n} seconds")
    print(f"  Pearson correlation: {corr:.3f}  "
          f"{'(strong agreement)' if corr > 0.9 else '(check for drift)' if corr > 0.7 else '(DIVERGENT — investigate)'}")
    print()
    print(" t(s) | driver | server | ratio | bar (server)")
    print("-" * 70)
    for i in range(n):
        d = driver_aligned[i]
        s = server_aligned[i]
        ratio = (d / s) if s > 0 else 0
        bar = "#" * int(s / 40)
        print(f"{i:4d}  | {d:6.0f} | {s:6.0f} | {ratio:4.2f}  | {bar}")

    # --- Dip detection (server-side) ----------------------------------------
    dips = find_dips(server_aligned, threshold_frac=0.25)
    print()
    print("=== Server-side dips (<25% of peak, ≥2s wide) ===")
    if not dips:
        print("  none — throughput is clean.")
    else:
        peak = max(server_aligned) if server_aligned else 0
        for start, end, mn in dips:
            width = end - start + 1
            print(f"  t=[{start:3d}..{end:3d}] ({width}s wide), min={mn:.0f} tok/s  "
                  f"(peak={peak:.0f} tok/s, drop={100*(1-mn/peak):.0f}%)")
    print()

    # --- Peak sustained throughput ------------------------------------------
    if len(server_aligned) >= 5:
        # Use 5-second rolling window to compute sustained peak.
        rolling = [
            sum(server_aligned[i:i+5]) / 5
            for i in range(len(server_aligned) - 4)
        ]
        print(f"=== Server-side sustained peak (5s rolling) ===")
        print(f"  peak_5s = {max(rolling):.0f} tok/s")
        print(f"  overall peak (1s) = {max(server_aligned):.0f} tok/s")
        print()

    # --- Optional CSV -------------------------------------------------------
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w") as f:
            f.write("t_s,driver_tps,server_tps,ratio\n")
            for i in range(n):
                d = driver_aligned[i]
                s = server_aligned[i]
                ratio = (d / s) if s > 0 else 0
                f.write(f"{i},{d:.1f},{s:.1f},{ratio:.3f}\n")
        print(f"wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
