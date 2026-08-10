#!/usr/bin/env python3
"""Compare accelerated DwarfStar prefill logits with its quality reference path."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
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
            "Compare accelerated DwarfStar prefill against the isolated Metal 4 "
            "reference path, verify repeatability, and smoke-test AFM integration."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=root / "vendor" / "ds4" / "ds4-bench"
    )
    parser.add_argument(
        "--afm-binary",
        type=Path,
        default=root / ".build" / "arm64-apple-macosx" / "release" / "afm",
    )
    parser.add_argument(
        "--skip-afm-integration",
        action="store_true",
        help="Skip the shipped AFM DwarfStar bridge smoke test.",
    )
    parser.add_argument(
        "--require-metal4",
        action="store_true",
        help="Fail instead of reporting a skip when Metal 4 Tensor kernels are unavailable.",
    )
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--afm-smoke-tokens", type=int, default=8)
    parser.add_argument(
        "--skip-model-hash",
        action="store_true",
        help="Do not hash the GGUF (faster, but the report is less reproducible).",
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
    if args.repeat_runs < 2:
        parser.error("--repeat-runs must be at least 2")
    if args.afm_smoke_tokens < 1:
        parser.error("--afm-smoke-tokens must be positive")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_mode(
    *, binary: Path, model: Path, prompt: Path, output: Path, disable_metal4: bool
) -> str:
    output.mkdir(parents=True, exist_ok=False)
    logits = output / "logits"
    logits.mkdir()
    command = [
        str(binary),
        "-m",
        str(model),
        "--metal",
    ]
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
    environment = os.environ.copy()
    if disable_metal4:
        environment["DS4_METAL_DISABLE_METAL4"] = "1"
    else:
        environment.pop("DS4_METAL_DISABLE_METAL4", None)
    log_path = output / "run.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=binary.parent,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{'reference' if disable_metal4 else 'accelerated'} run failed; "
            f"see {log_path}"
        )
    return log_path.read_text(errors="replace")


def run_afm_integration(
    *, binary: Path, model: Path, output: Path, tokens: int
) -> dict[str, object]:
    result_path = output / "afm-dwarfstar-smoke.json"
    log_path = output / "afm-dwarfstar-smoke.log"
    command = [
        str(binary),
        "dwarfstar-bench",
        "-m",
        str(model),
        "--prompt",
        "Count upward from 1, separated only by commas. Continue until stopped.",
        "--tokens",
        str(tokens),
        "--runs",
        "2",
        "--warmup-tokens",
        "1",
        "--output",
        str(result_path),
    ]
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"AFM DwarfStar integration failed; see {log_path}")
    report = json.loads(result_path.read_text())
    if report.get("runtime") != "in-process-dwarfstar" or len(report.get("runs", [])) != 2:
        raise RuntimeError("AFM DwarfStar integration produced an invalid report")
    return report


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
    afm_binary = None
    if not args.skip_afm_integration:
        afm_binary = require_path(args.afm_binary, "AFM release binary", executable=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    if args.prompt_file:
        prompt = require_path(args.prompt_file, "prompt")
    else:
        prompt = output / "prompt.txt"
        prompt.write_text((PROMPT_SENTENCE * 900).strip() + "\n")

    accelerated_dirs = []
    accelerated_logs = []
    for run in range(1, args.repeat_runs + 1):
        accelerated_dir = output / f"accelerated-{run}"
        print(f"Running accelerated DwarfStar prefill ({run}/{args.repeat_runs})...", flush=True)
        accelerated_logs.append(
            run_mode(
                binary=binary,
                model=model,
                prompt=prompt,
                output=accelerated_dir,
                disable_metal4=False,
            )
        )
        accelerated_dirs.append(accelerated_dir)

    reference_dir = output / "metal4-disabled-reference"
    print("Running DwarfStar with Metal 4 Tensor kernels disabled...", flush=True)
    run_mode(
        binary=binary,
        model=model,
        prompt=prompt,
        output=reference_dir,
        disable_metal4=True,
    )

    accelerated_logits = accelerated_dirs[0] / "logits"
    reference_logits = reference_dir / "logits"
    accelerated_files = sorted(accelerated_logits.glob("*.json"))
    if {path.name for path in accelerated_files} != {
        "frontier_000032.logits.json",
        "frontier_004128.logits.json",
    }:
        raise RuntimeError("expected 32- and 4,128-token frontier logits")

    comparisons = [
        compare_frontier(
            path,
            reference_logits / path.name,
            max_logit_diff=args.max_logit_diff,
            max_rms_diff=args.max_rms_diff,
            top_k=args.top_k,
            min_top_k_overlap=args.min_top_k_overlap,
        )
        for path in accelerated_files
    ]
    repeatability = []
    for frontier in accelerated_files:
        digests = [
            sha256_file(directory / "logits" / frontier.name)
            for directory in accelerated_dirs
        ]
        repeatability.append(
            {
                "frontier": frontier.name,
                "sha256": digests,
                "deterministic": len(set(digests)) == 1,
            }
        )

    metal4_enabled = any(
        "Metal 4 tensor API enabled for Tensor kernels" in log
        for log in accelerated_logs
    )
    if args.require_metal4 and not metal4_enabled:
        raise RuntimeError(
            "Metal 4 Tensor kernels were not enabled; qualification cannot exercise that path"
        )

    afm_report = None
    if afm_binary is not None:
        print("Running AFM in-process DwarfStar integration...", flush=True)
        afm_report = run_afm_integration(
            binary=afm_binary,
            model=model,
            output=output,
            tokens=args.afm_smoke_tokens,
        )

    root = Path(__file__).resolve().parents[1]
    ds4_revision = subprocess.check_output(
        ["git", "-C", str(root / "vendor" / "ds4"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    result_passed = (
        all(item["passed"] for item in comparisons)
        and all(item["deterministic"] for item in repeatability)
        and (afm_report is not None or args.skip_afm_integration)
    )
    qualification_status = (
        "passed" if metal4_enabled and result_passed
        else "failed" if metal4_enabled
        else "skipped-metal4-unavailable"
    )
    report = {
        "model": str(model),
        "model_sha256": None if args.skip_model_hash else sha256_file(model),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "afm_binary": None if afm_binary is None else str(afm_binary),
        "afm_binary_sha256": None if afm_binary is None else sha256_file(afm_binary),
        "dwarfstar_revision": ds4_revision,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_version": platform.mac_ver()[0],
        },
        # DwarfStar does not expose per-dispatch counters. This records the
        # upstream runtime's explicit Metal 4 enablement evidence without
        # claiming that a particular private kernel symbol was dispatched.
        "metal4_tensor_api": "enabled" if metal4_enabled else "unavailable",
        "qualification_status": qualification_status,
        "thresholds": {
            "max_absolute_logit_diff": args.max_logit_diff,
            "max_rms_logit_diff": args.max_rms_diff,
            "top_k": args.top_k,
            "min_top_k_overlap": args.min_top_k_overlap,
            "argmax_must_match": True,
        },
        "comparisons": comparisons,
        "repeatability": repeatability,
        "afm_integration": afm_report,
        "passed": result_passed if metal4_enabled else None,
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
    if qualification_status == "skipped-metal4-unavailable":
        print("SKIP: Metal 4 Tensor kernels are unavailable on this host")
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
