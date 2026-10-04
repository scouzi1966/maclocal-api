#!/usr/bin/env python3
"""Fail closed on missing/mismatched paired Context evidence or throughput loss.

Uses every raw trial, never the independently selected CSV phase peaks.
This is a performance/exact-output gate, not a semantic quality qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_TOLERANCE_PERCENT = 3.0
METRICS = ("prompt_tps_e2e", "generation_tps")


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    if json.loads((directory / "result.json").read_text())["status"] != "completed":
        raise ValueError("Run did not complete")
    command = json.loads((directory / "executed-test-command.json").read_text())
    metadata["benchmark_arguments"] = command[3:]
    trials = records(directory / "raw-trial-results.jsonl")
    transcripts = records(directory / "paired-transcripts.jsonl")
    usage = records(directory / "stream-usage.jsonl")
    if not trials or len(trials) != len(transcripts) or len(trials) != len(usage):
        raise ValueError("Missing trials or transcript count mismatch")
    expected = metadata["experiment"]["trials_per_context"]
    if expected < 2:
        raise ValueError("At least two trials per context required")
    contexts = command[command.index("--contexts") + 1].split(",")
    observed = {float(row["context_size"].removesuffix("k")) for row in trials}
    if observed != {float(value) for value in contexts}:
        raise ValueError("Missing or unexpected contexts")
    for trial, transcript, response in zip(trials, transcripts, usage):
        if hashlib.sha256(transcript["prompt"].encode()).hexdigest() != transcript["prompt_sha256"]:
            raise ValueError("Prompt digest mismatch")
        if trial["generated_text"] != transcript["generated_text"]:
            raise ValueError("Trial and transcript output mismatch")
        if response["prompt_sha256"] != transcript["prompt_sha256"]:
            raise ValueError("Usage belongs to a different prompt")
        if response["usage"]["prompt_tokens_details"]["cached_tokens"] != 0:
            raise ValueError("Cold comparison contains cached prompt tokens")
        if response["usage"]["prompt_tokens"] != trial["prompt_tokens"]:
            raise ValueError("Prompt accounting mismatch")
        if trial["generation_tokens"] != int(command[command.index("--max-tokens") + 1]):
            raise ValueError("Incomplete generation")
        for metric in METRICS:
            if not math.isfinite(trial[metric]) or trial[metric] <= 0:
                raise ValueError("Invalid throughput")
    for context in {row["context_size"] for row in trials}:
        if sum(row["context_size"] == context for row in trials) != expected:
            raise ValueError("Incomplete context trial count")
    return metadata, trials, transcripts


def compare(baseline, candidate, tolerance):
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 100:
        raise ValueError("Invalid tolerance")
    old_meta, old, old_text = load_run(baseline)
    new_meta, new, new_text = load_run(candidate)
    if old_meta["checkpoint"] != new_meta["checkpoint"]:
        raise ValueError("Checkpoint paths differ")
    if old_meta.get("community_revision") != new_meta.get("community_revision"):
        raise ValueError("Checkpoint revisions differ")
    if old_meta.get("benchmark_arguments") != new_meta.get("benchmark_arguments"):
        raise ValueError("Client workload or sampling arguments differ")
    for key in ("trials_per_context", "in_process_warm_context_pass", "warm_marker_epoch"):
        if old_meta["experiment"].get(key) != new_meta["experiment"].get(key):
            raise ValueError("Workload schedule differs: " + key)
    def workload(rows, texts):
        return [(r["context_size"], r["prompt_tokens"], r["generation_tokens"], t["prompt_sha256"])
                for r, t in zip(rows, texts)]
    if workload(old, old_text) != workload(new, new_text):
        raise ValueError("Prompts, tokens, trial order or context grid differ")
    cells = []
    for context in dict.fromkeys(row["context_size"] for row in old):
        for metric in METRICS:
            before = [r[metric] for r in old if r["context_size"] == context]
            after = [r[metric] for r in new if r["context_size"] == context]
            delta = 100 * (mean(after) / mean(before) - 1)
            cells.append(dict(context=context, metric=metric, baseline_mean=mean(before),
                              candidate_mean=mean(after), delta_percent=delta,
                              baseline_peak=max(before), candidate_peak=max(after),
                              passed=delta >= -tolerance))
    exact = all(a["generated_text"] == b["generated_text"] for a, b in zip(old, new))
    performance = all(cell["passed"] for cell in cells)
    return dict(passed=performance and exact, performance_passed=performance,
                exact_outputs=exact, semantic_quality_qualified=False,
                tolerance_percent=tolerance, cells=cells,
                baseline_binary_sha256=old_meta["binary_sha256"],
                candidate_binary_sha256=new_meta["binary_sha256"],
                note="Failing output equivalence needs separate quality review; never a release approval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tolerance-percent", type=float, default=DEFAULT_TOLERANCE_PERCENT)
    args = parser.parse_args()
    try:
        result = compare(args.baseline, args.candidate, args.tolerance_percent)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = dict(passed=False, error=str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
