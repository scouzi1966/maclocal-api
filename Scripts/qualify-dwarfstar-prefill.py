#!/usr/bin/env python3
"""Compare accelerated DwarfStar prefill logits with its quality reference path."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys


PROMPT_SENTENCE = (
    "The quick brown fox studies accelerated sparse attention and routed expert "
    "computation on Apple silicon. "
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run DwarfStar prefill with accelerated kernels and --quality, then "
            "compare full vocabulary logits at 32 and 4,128 tokens."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=root / "vendor" / "ds4" / "ds4-bench"
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / ".build"
        / "test-work"
        / "dwarfstar-prefill"
        / dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--max-logit-diff", type=float, default=3.0)
    parser.add_argument("--max-rms-diff", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-top-k-overlap", type=int, default=7)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not 1 <= args.min_top_k_overlap <= args.top_k:
        parser.error("--min-top-k-overlap must be between 1 and --top-k")
    return args


def require_path(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} not found: {resolved}")
    if executable and not resolved.stat().st_mode & 0o111:
        raise RuntimeError(f"{label} is not executable: {resolved}")
    return resolved


def run_mode(
    *, binary: Path, model: Path, prompt: Path, output: Path, quality: bool
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    logits = output / "logits"
    logits.mkdir()
    command = [
        str(binary),
        "-m",
        str(model),
        "--metal",
    ]
    if quality:
        command.append("--quality")
    command.extend(
        [
            "--prompt-file",
            str(prompt),
            "--ctx-start",
            "32",
            "--ctx-max",
            "4128",
            "--ctx-alloc",
            "4160",
            "--step-incr",
            "4096",
            "--gen-tokens",
            "0",
            "--dump-frontier-logits-dir",
            str(logits),
            "--csv",
            str(output / "results.csv"),
        ]
    )
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=binary.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{'quality' if quality else 'accelerated'} run failed; "
            f"see {output / 'run.log'}"
        )


def top_indices(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count]


def compare_frontier(
    accelerated_path: Path,
    quality_path: Path,
    *,
    max_logit_diff: float,
    max_rms_diff: float,
    top_k: int,
    min_top_k_overlap: int,
) -> dict[str, object]:
    accelerated = json.loads(accelerated_path.read_text())
    quality = json.loads(quality_path.read_text())
    accelerated_logits = [float(value) for value in accelerated["logits"]]
    quality_logits = [float(value) for value in quality["logits"]]
    if len(accelerated_logits) != len(quality_logits):
        raise RuntimeError(f"vocabulary size differs at {accelerated_path.name}")

    differences = [
        abs(lhs - rhs) for lhs, rhs in zip(accelerated_logits, quality_logits)
    ]
    maximum = max(differences)
    rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    accelerated_top = top_indices(accelerated_logits, top_k)
    quality_top = top_indices(quality_logits, top_k)
    overlap = len(set(accelerated_top).intersection(quality_top))
    argmax_matches = accelerated["argmax_id"] == quality["argmax_id"]
    passed = (
        argmax_matches
        and maximum <= max_logit_diff
        and rms <= max_rms_diff
        and overlap >= min_top_k_overlap
    )
    return {
        "frontier_tokens": accelerated["frontier_tokens"],
        "argmax_matches": argmax_matches,
        "accelerated_argmax_id": accelerated["argmax_id"],
        "quality_argmax_id": quality["argmax_id"],
        "max_absolute_logit_diff": maximum,
        "rms_logit_diff": rms,
        "top_k": top_k,
        "top_k_overlap": overlap,
        "passed": passed,
    }


def main() -> int:
    args = parse_args()
    model = require_path(args.model, "model")
    binary = require_path(args.binary, "ds4-bench", executable=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    if args.prompt_file:
        prompt = require_path(args.prompt_file, "prompt")
    else:
        prompt = output / "prompt.txt"
        prompt.write_text((PROMPT_SENTENCE * 900).strip() + "\n")

    accelerated_dir = output / "accelerated"
    quality_dir = output / "quality"
    print("Running accelerated DwarfStar prefill...", flush=True)
    run_mode(
        binary=binary,
        model=model,
        prompt=prompt,
        output=accelerated_dir,
        quality=False,
    )
    print("Running DwarfStar --quality reference prefill...", flush=True)
    run_mode(
        binary=binary,
        model=model,
        prompt=prompt,
        output=quality_dir,
        quality=True,
    )

    accelerated_logits = accelerated_dir / "logits"
    quality_logits = quality_dir / "logits"
    accelerated_files = sorted(accelerated_logits.glob("*.json"))
    if {path.name for path in accelerated_files} != {
        "frontier_000032.logits.json",
        "frontier_004128.logits.json",
    }:
        raise RuntimeError("expected 32- and 4,128-token frontier logits")

    comparisons = [
        compare_frontier(
            path,
            quality_logits / path.name,
            max_logit_diff=args.max_logit_diff,
            max_rms_diff=args.max_rms_diff,
            top_k=args.top_k,
            min_top_k_overlap=args.min_top_k_overlap,
        )
        for path in accelerated_files
    ]
    report = {
        "model": str(model),
        "binary": str(binary),
        "thresholds": {
            "max_absolute_logit_diff": args.max_logit_diff,
            "max_rms_logit_diff": args.max_rms_diff,
            "top_k": args.top_k,
            "min_top_k_overlap": args.min_top_k_overlap,
            "argmax_must_match": True,
        },
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons),
    }
    report_path = output / "qualification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for item in comparisons:
        print(
            f"{item['frontier_tokens']:>5} tokens: "
            f"argmax={'match' if item['argmax_matches'] else 'DIFF'} "
            f"max={item['max_absolute_logit_diff']:.4f} "
            f"rms={item['rms_logit_diff']:.4f} "
            f"top-{item['top_k']}={item['top_k_overlap']} "
            f"{'PASS' if item['passed'] else 'FAIL'}"
        )
    print(f"Report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
