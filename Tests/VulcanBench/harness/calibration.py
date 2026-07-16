"""Empirical difficulty calibration from recorded runs.

``calibrate_tasks`` derives each task's empirical difficulty from observed solve
rates across models, compares it against the hand-labeled ``difficulty``, and
returns a JSON-serializable result. Below the evidence gate the status is
``insufficient_data`` — never a fabricated label.

Sort order: disagreements first, then calibrated, then insufficient; ties by
``task_id``.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.leaderboard import mark_stale
from harness.tasks import list_task_ids, load_task


def _is_solved(row: dict[str, Any]) -> bool:
    return (row.get("functional") or 0) >= 1.0


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(float(statistics.stdev(values)) / math.sqrt(len(values)), 4)


def _band(solve_rate: float, easy_min: float, medium_min: float) -> str:
    if solve_rate >= easy_min:
        return "easy"
    if solve_rate >= medium_min:
        return "medium"
    return "hard"


def _normalize_label(label: str | None) -> str | None:
    if label == "trivial":
        return "easy"
    return label


def _build_entry(
    task_id: str,
    attempts: list[dict[str, Any]],
    labeled_difficulty: str | None,
    min_attempts: int,
    min_models: int,
    easy_min: float,
    medium_min: float,
) -> dict[str, Any]:
    """Build one calibration entry for a single task."""
    n_attempts = len(attempts)

    if n_attempts == 0:
        return {
            "task_id": task_id,
            "labeled_difficulty": labeled_difficulty,
            "n_attempts": 0,
            "n_models": 0,
            "models": [],
            "per_model_solve_rate": {},
            "solve_rate": None,
            "solve_rate_stderr": None,
            "empirical_difficulty": None,
            "status": "insufficient_data",
            "agreement": None,
        }

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in attempts:
        by_model[a.get("model") or "?"].append(a)

    per_model_rate: dict[str, float] = {}
    for model, model_attempts in by_model.items():
        solved = sum(1 for a in model_attempts if _is_solved(a))
        per_model_rate[model] = round(solved / len(model_attempts), 4)

    model_rates = list(per_model_rate.values())
    solve_rate = round(sum(model_rates) / len(model_rates), 4)
    solve_rate_stderr = _stderr(model_rates)
    n_models = len(by_model)
    models_list = sorted(by_model.keys())

    if n_attempts >= min_attempts and n_models >= min_models:
        empirical = _band(solve_rate, easy_min, medium_min)
        norm_label = _normalize_label(labeled_difficulty)
        agreement = empirical == norm_label if norm_label is not None else None
        status = "ok"
    else:
        empirical = None
        agreement = None
        status = "insufficient_data"

    return {
        "task_id": task_id,
        "labeled_difficulty": labeled_difficulty,
        "n_attempts": n_attempts,
        "n_models": n_models,
        "models": models_list,
        "per_model_solve_rate": per_model_rate,
        "solve_rate": solve_rate,
        "solve_rate_stderr": solve_rate_stderr,
        "empirical_difficulty": empirical,
        "status": status,
        "agreement": agreement,
    }


def _validate_thresholds(
    easy_min: float, medium_min: float, min_attempts: int, min_models: int
) -> None:
    if not (math.isfinite(easy_min) and math.isfinite(medium_min)):
        raise ValueError("easy_min and medium_min must be finite")
    if not (0 <= medium_min < easy_min <= 1):
        raise ValueError(f"require 0 <= medium_min < easy_min <= 1, got {medium_min}, {easy_min}")
    if min_attempts < 1:
        raise ValueError(f"min_attempts must be >= 1, got {min_attempts}")
    if min_models < 1:
        raise ValueError(f"min_models must be >= 1, got {min_models}")


def _filter_rows(
    rows: list[dict[str, Any]],
    valid_task_ids: set[str],
    include_mock: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter out mock, stale, and unknown-task rows; return filtered list + excluded counts."""
    mock_runs = 0
    stale_runs = 0
    unknown_task_runs = 0
    filtered: list[dict[str, Any]] = []
    for r in rows:
        model = r.get("model") or ""
        if not include_mock and model.startswith("mock:"):
            mock_runs += 1
            continue
        if r.get("task_stale") is True:
            stale_runs += 1
            continue
        tid = r.get("task_id")
        if tid not in valid_task_ids:
            unknown_task_runs += 1
            continue
        filtered.append(r)
    return filtered, {
        "mock_runs": mock_runs,
        "stale_runs": stale_runs,
        "unknown_task_runs": unknown_task_runs,
    }


def calibrate_tasks(
    rows: list[dict[str, Any]],
    tasks_root: Path = Path("tasks/v1"),
    min_attempts: int = 5,
    min_models: int = 2,
    easy_min: float = 0.85,
    medium_min: float = 0.40,
    include_mock: bool = False,
) -> dict[str, Any]:
    """Derive empirical difficulty per task from observed solve rates.

    Filters out mock-model runs and stale runs, macro-averages per-model solve
    rates so one heavily-repeated model can't dominate, and applies banding
    thresholds only when evidence suffices.
    """
    _validate_thresholds(easy_min, medium_min, min_attempts, min_models)

    mark_stale(rows, tasks_root)
    valid_task_ids = set(list_task_ids(tasks_root))
    filtered, excluded = _filter_rows(rows, valid_task_ids, include_mock)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in filtered:
        tid = r.get("task_id")
        if isinstance(tid, str):
            by_task[tid].append(r)

    entries: list[dict[str, Any]] = []
    n_calibrated = 0
    n_disagree = 0
    n_insufficient = 0

    for task_id in sorted(valid_task_ids):
        attempts = by_task.get(task_id, [])
        labeled_difficulty: str | None = None
        try:
            meta = load_task(task_id, tasks_root).metadata
            labeled_difficulty = meta.get("difficulty")
        except (OSError, ValueError):
            pass

        entry = _build_entry(
            task_id, attempts, labeled_difficulty, min_attempts, min_models, easy_min, medium_min
        )
        entries.append(entry)
        if entry["status"] == "ok":
            n_calibrated += 1
            if entry["agreement"] is False:
                n_disagree += 1
        else:
            n_insufficient += 1

    entries.sort(
        key=lambda e: (
            0 if e["agreement"] is False else 1 if e["status"] == "ok" else 2,
            e["task_id"],
        )
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "thresholds": {"easy_min": easy_min, "medium_min": medium_min},
        "criteria": {"min_attempts": min_attempts, "min_models": min_models},
        "excluded": excluded,
        "tasks": entries,
        "summary": {
            "n_tasks": len(entries),
            "n_calibrated": n_calibrated,
            "n_disagree": n_disagree,
            "n_insufficient": n_insufficient,
        },
    }


def calibration_to_markdown(cal: dict[str, Any]) -> str:
    """Render a calibration result as Markdown."""
    lines: list[str] = ["## Calibration", ""]
    summary = cal["summary"]
    thresholds = cal["thresholds"]
    criteria = cal["criteria"]
    excluded = cal["excluded"]

    calibrated = [e for e in cal["tasks"] if e["status"] == "ok"]
    disagreements = [e for e in calibrated if e["agreement"] is False]

    if disagreements:
        names = ", ".join(e["task_id"] for e in disagreements)
        lines.append(
            f"> ⚠️ **{len(disagreements)} task(s) have empirical difficulty that "
            f"disagrees with the hand label**: {names}"
        )
        lines.append("")

    lines.append(
        f"Thresholds: easy ≥ {thresholds['easy_min']}, medium ≥ {thresholds['medium_min']}. "
        f"Criteria: ≥ {criteria['min_attempts']} attempts, ≥ {criteria['min_models']} models. "
        f"Excluded: {excluded['mock_runs']} mock, {excluded['stale_runs']} stale, "
        f"{excluded['unknown_task_runs']} unknown-task runs."
    )
    lines.append("")

    if calibrated:
        lines.append("| Task | Label | Empirical | Solve rate ± se | Attempts | Models |")
        lines.append("|---|---|---|---|---|---|")
        for e in calibrated:
            sr = e["solve_rate"]
            se = e["solve_rate_stderr"]
            sr_str = f"{sr:.4f} ± {se:.4f}" if sr is not None else "—"
            label = e["labeled_difficulty"] or "—"
            lines.append(
                f"| {e['task_id']} | {label} | {e['empirical_difficulty']} | "
                f"{sr_str} | {e['n_attempts']} | {e['n_models']} |"
            )
    else:
        lines.append("_No tasks have sufficient data for calibration._")

    if summary["n_insufficient"]:
        lines.append(
            f"\n{summary['n_insufficient']} task(s) lack sufficient data "
            f"({criteria['min_attempts']}+ attempts from {criteria['min_models']}+ models)."
        )

    lines.append("")
    return "\n".join(lines)
