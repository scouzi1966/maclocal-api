#!/usr/bin/env python3
"""Export benchmark cost priors from local ``./runs`` summaries.

Writes ``harness/data/cost_priors.json`` for cold-start cost estimation.
Regenerate before releases when reference benchmark runs are available.

    python scripts/export_cost_priors.py
    python scripts/export_cost_priors.py --suite v1-compare --runs-dir ./runs
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from harness.leaderboard import load_summaries
from harness.suite import load_suite

_SKIP_MODEL_PREFIXES = ("mock:",)
_DEFAULT_OUT = Path("harness/data/cost_priors.json")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] * (c - k) + ordered[c] * (k - f)


def _bucket_stats(costs: list[float]) -> dict[str, float | int]:
    return {
        "low": round(_percentile(costs, 0.25), 4),
        "mid": round(statistics.median(costs), 4),
        "high": round(_percentile(costs, 0.90), 4),
        "n": len(costs),
    }


def _judges_on(summary: dict) -> bool:
    human = (summary.get("scores") or {}).get("metric_details", {}).get("human_like", {})
    jp = int(human.get("judge_prompt_tokens", 0) or 0)
    jc = int(human.get("judge_completion_tokens", 0) or 0)
    return jp > 0 or jc > 0


def export_priors(
    runs_dir: Path,
    *,
    suite: str | None = None,
    judges_only: bool = False,
    models: list[str] | None = None,
) -> dict:
    allowed_tasks: set[str] | None = None
    if suite is not None:
        allowed_tasks = set(load_suite(suite).task_ids)

    by_model_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_model_scale: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for summary in load_summaries(runs_dir):
        model = summary.get("model") or ""
        task_id = summary.get("task_id") or ""
        cost = summary.get("cost_usd")
        if not model or not task_id or cost is None:
            continue
        if any(model.startswith(p) for p in _SKIP_MODEL_PREFIXES):
            continue
        if models is not None and model not in models:
            continue
        if allowed_tasks is not None and task_id not in allowed_tasks:
            continue
        if judges_only and not _judges_on(summary):
            continue

        c = float(cost)
        by_model_task[model][task_id].append(c)
        manifest_task = (summary.get("manifest") or {}).get("task", {})
        scale = manifest_task.get("repo_scale") or "unknown"
        by_model_scale[model][scale].append(c)

    out_mt: dict[str, dict[str, dict[str, float | int]]] = {}
    for model in sorted(by_model_task):
        out_mt[model] = {
            task_id: _bucket_stats(costs)
            for task_id, costs in sorted(by_model_task[model].items())
        }

    out_ms: dict[str, dict[str, dict[str, float | int]]] = {}
    for model in sorted(by_model_scale):
        out_ms[model] = {
            scale: _bucket_stats(costs)
            for scale, costs in sorted(by_model_scale[model].items())
        }

    return {
        "version": 1,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "judges": True,
        "by_model_task": out_mt,
        "by_model_scale": out_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export cost priors from local runs")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("./runs"),
        help="Directory containing run summaries (default: ./runs)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Only include tasks from this suite (e.g. v1-compare)",
    )
    parser.add_argument(
        "--judges-only",
        action="store_true",
        help="Only include runs with judge token usage",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="Only include this model (repeat for multiple)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output JSON path (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    if not args.runs_dir.exists():
        print(f"error: runs dir not found: {args.runs_dir}", file=sys.stderr)
        return 2

    data = export_priors(
        args.runs_dir,
        suite=args.suite,
        judges_only=args.judges_only,
        models=args.models,
    )
    n_task_buckets = sum(len(v) for v in data["by_model_task"].values())
    n_scale_buckets = sum(len(v) for v in data["by_model_scale"].values())
    if n_task_buckets == 0 and n_scale_buckets == 0:
        print("warning: no qualifying runs found; writing empty priors", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({n_task_buckets} model×task, {n_scale_buckets} model×scale buckets)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
