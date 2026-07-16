"""Build a shareable results report from recorded runs.

``build_report`` produces a structured dict (JSON-serializable); ``to_markdown``
renders it for humans. A report has:

- ``models``      — the per-model ranking (pass@1 ± stderr, pass@k, cost, latency)
- ``tasks``       — per-task breakdown (per-model attempts / solve rate)
- ``environment`` — distinct models / Python / tool versions seen in run manifests
- ``integrity``   — runs scored against a now-stale task version, and runs
  scored against tasks not known to be decontaminated (``decontaminated: false``)
- ``totals``      — run / model / task counts and total known cost
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.calibration import calibrate_tasks, calibration_to_markdown
from harness.leaderboard import (
    aggregate_by_model,
    load_summaries,
    mark_stale,
    summary_to_row,
)
from harness.task_metadata import repo_scale, task_complexity
from harness.tasks import list_task_ids, load_task


def build_report(
    runs_dir: Path = Path("./runs"),
    tasks_root: Path = Path("tasks/v1"),
    suite: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a report dict from the run summaries under ``runs_dir``."""
    summaries = load_summaries(runs_dir)
    if suite is not None:
        summaries = [s for s in summaries if s.get("suite") == suite]
    rows = [summary_to_row(s, s.get("run_id", "")) for s in summaries]
    mark_stale(rows, tasks_root)

    models = aggregate_by_model(rows)
    tasks = _per_task(rows)
    environment = _environment(summaries)
    integrity = _integrity(rows, tasks_root)
    calibration = calibrate_tasks(rows, tasks_root)
    effort_sensitivity = build_effort_sensitivity(rows, tasks_root)
    known_costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]

    return {
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "suite": suite,
        "totals": {
            "n_runs": len(rows),
            "n_models": len(models),
            "n_tasks": len({r.get("task_id") for r in rows}),
            "total_cost_usd": round(sum(known_costs), 6) if known_costs else None,
        },
        "models": models,
        "tasks": tasks,
        "environment": environment,
        "integrity": integrity,
        "calibration": calibration,
        "effort_sensitivity": effort_sensitivity,
    }


def _per_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_id")].append(r)
    out: list[dict[str, Any]] = []
    for task_id in sorted(by_task, key=str):
        attempts = by_task[task_id]
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in attempts:
            by_model[a.get("model") or "?"].append(a)
        model_stats = []
        for model in sorted(by_model):
            ms = by_model[model]
            solved = sum(1 for a in ms if (a.get("functional") or 0) >= 1.0)
            model_stats.append(
                {
                    "model": model,
                    "attempts": len(ms),
                    "solved": solved,
                    "solve_rate": round(solved / len(ms), 4),
                }
            )
        out.append({"task_id": task_id, "n_runs": len(attempts), "models": model_stats})
    return out


def _environment(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    models: set[str] = set()
    pythons: set[str] = set()
    tools: dict[str, set[str]] = defaultdict(set)
    for s in summaries:
        manifest = s.get("manifest") or {}
        if manifest.get("model"):
            models.add(str(manifest["model"]))
        runtime = manifest.get("runtime") or {}
        if runtime.get("python"):
            pythons.add(str(runtime["python"]))
        for tool, version in (manifest.get("tools") or {}).items():
            if version:
                tools[tool].add(str(version))
    return {
        "models": sorted(models),
        "python": sorted(pythons),
        "tools": {t: sorted(v) for t, v in sorted(tools.items())},
    }


def _not_decontaminated_tasks(tasks_root: Path, task_ids: set[Any]) -> set[str]:
    """Task ids whose metadata explicitly marks them not decontaminated.

    Only ``decontaminated: false`` counts — a missing field is treated as
    unknown here (the validator is what *requires* the field at authoring time).
    """
    flagged: set[str] = set()
    for tid in task_ids:
        if not tid:
            continue
        meta_path = tasks_root / str(tid) / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("decontaminated") is False:
            flagged.add(str(tid))
    return flagged


def _integrity(rows: list[dict[str, Any]], tasks_root: Path) -> dict[str, Any]:
    stale = [r["run_id"] for r in rows if r.get("task_stale") is True]
    unknown = sum(1 for r in rows if r.get("task_stale") is None)
    flagged = _not_decontaminated_tasks(tasks_root, {r.get("task_id") for r in rows})
    nd_runs = [r["run_id"] for r in rows if str(r.get("task_id")) in flagged]
    return {
        "stale": len(stale),
        "stale_run_ids": stale,
        "unknown": unknown,
        "not_decontaminated": len(nd_runs),
        "not_decontaminated_run_ids": nd_runs,
        "not_decontaminated_tasks": sorted(flagged),
    }


def _task_meta(task_id: Any, tasks_root: Path) -> dict[str, Any]:
    if not isinstance(task_id, str):
        return {}
    try:
        return load_task(task_id, tasks_root).metadata
    except (OSError, ValueError):
        return {}


def _row_languages(row: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    raw = row.get("languages") or meta.get("languages") or []
    if isinstance(raw, list):
        values = [str(v) for v in raw if v]
        return values or ["unknown"]
    return [str(raw)] if raw else ["unknown"]


def _row_scale(row: dict[str, Any], meta: dict[str, Any]) -> str:
    raw = row.get("repo_scale")
    return str(raw) if raw else repo_scale(meta)


def _row_complexity(row: dict[str, Any], meta: dict[str, Any]) -> str:
    raw = row.get("task_complexity")
    return str(raw) if raw else task_complexity(meta)


def _row_difficulty(row: dict[str, Any], meta: dict[str, Any]) -> str:
    raw = row.get("difficulty") or meta.get("difficulty")
    return str(raw) if raw else "unknown"


def build_effort_sensitivity(rows: list[dict[str, Any]], tasks_root: Path) -> dict[str, Any]:
    tagged = [r for r in rows if isinstance(r.get("effort"), dict) and r["effort"].get("requested")]
    if not tagged:
        return {"available": False, "strata": [], "warnings": []}

    meta_cache: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in tagged:
        task_id = row.get("task_id")
        meta = meta_cache.setdefault(str(task_id), _task_meta(task_id, tasks_root))
        for language in _row_languages(row, meta):
            key = (
                str(row.get("model") or "unknown"),
                language,
                _row_scale(row, meta),
                _row_difficulty(row, meta),
                _row_complexity(row, meta),
            )
            groups[key][str(row["effort"]["requested"])].append(row)

    strata: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(part) for part in k)):
        model, language, scale, difficulty, complexity = key
        by_effort = {
            effort: _effort_metrics(effort_rows)
            for effort, effort_rows in sorted(groups[key].items())
        }
        low = by_effort.get("low")
        high = by_effort.get("high")
        deltas = _effort_deltas(low, high)
        strata.append(
            {
                "model": model,
                "language": language,
                "repo_scale": scale,
                "difficulty": difficulty,
                "task_complexity": complexity,
                "efforts": by_effort,
                **deltas,
            }
        )

    return {
        "available": True,
        "strata": strata,
        "warnings": _effort_coverage_warnings(tagged, tasks_root),
    }


def _effort_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task_id")].append(row)
    per_task_rate = []
    for attempts in by_task.values():
        solved = sum(1 for row in attempts if (row.get("functional") or 0) >= 1.0)
        per_task_rate.append(solved / len(attempts))

    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    durations = [r["duration_s"] for r in rows if r.get("duration_s") is not None]
    return {
        "n_runs": len(rows),
        "n_tasks": len(by_task),
        "pass_at_1": _mean(per_task_rate),
        "avg_total": _mean([r.get("total") for r in rows]),
        "total_cost": round(sum(costs), 6) if costs else None,
        "cost_known": len(costs) == len(rows),
        "avg_duration_s": _mean(durations),
        "total_tokens": sum(r.get("total_tokens") or 0 for r in rows),
    }


def _effort_deltas(low: dict[str, Any] | None, high: dict[str, Any] | None) -> dict[str, Any]:
    if low is None or high is None:
        return {
            "high_minus_low_pass_at_1": None,
            "high_minus_low_avg_total": None,
            "high_cost_ratio": None,
            "high_latency_ratio": None,
            "classification": "insufficient",
        }
    pass_delta = _delta(high.get("pass_at_1"), low.get("pass_at_1"))
    total_delta = _delta(high.get("avg_total"), low.get("avg_total"))
    status = "insufficient"
    if pass_delta is not None and total_delta is not None:
        status = (
            "low sufficient" if pass_delta <= 0.05 and total_delta <= 0.03 else "effort sensitive"
        )
    return {
        "high_minus_low_pass_at_1": pass_delta,
        "high_minus_low_avg_total": total_delta,
        "high_cost_ratio": _ratio(high.get("total_cost"), low.get("total_cost")),
        "high_latency_ratio": _ratio(high.get("avg_duration_s"), low.get("avg_duration_s")),
        "classification": status,
    }


def _effort_coverage_warnings(rows: list[dict[str, Any]], tasks_root: Path) -> list[str]:
    warnings: list[str] = []
    corpus_large: dict[str, int] = defaultdict(int)
    run_large: dict[str, int] = defaultdict(int)

    for task_id in list_task_ids(tasks_root):
        meta = _task_meta(task_id, tasks_root)
        scale = repo_scale(meta)
        if scale not in {"medium", "large"}:
            continue
        for language in _row_languages({}, meta):
            if language in {"python", "rust"}:
                corpus_large[language] += 1

    meta_cache: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_task_id = row.get("task_id")
        meta = meta_cache.setdefault(str(row_task_id), _task_meta(row_task_id, tasks_root))
        scale = _row_scale(row, meta)
        if scale not in {"medium", "large"}:
            continue
        for language in _row_languages(row, meta):
            if language in {"python", "rust"}:
                run_large[language] += 1

    for language in ("python", "rust"):
        if corpus_large[language] == 0:
            warnings.append(
                f"No medium/large {language} tasks exist in the corpus; "
                "large-codebase effort sensitivity cannot be evaluated for this language."
            )
        elif run_large[language] == 0:
            warnings.append(f"No effort-tagged runs cover medium/large {language} tasks yet.")
    return warnings


def _delta(high: Any, low: Any) -> float | None:
    if not isinstance(high, (int, float)) or not isinstance(low, (int, float)):
        return None
    return round(float(high) - float(low), 4)


def _ratio(high: Any, low: Any) -> float | None:
    if not isinstance(high, (int, float)) or not isinstance(low, (int, float)) or float(low) <= 0:
        return None
    return round(float(high) / float(low), 4)


def _mean(values: Any) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def _fmt(n: Any) -> str:
    return "—" if n is None else (f"{n:.4f}" if isinstance(n, float) else str(n))


def to_markdown(report: dict[str, Any]) -> str:
    suite = report.get("suite") or "all runs"
    totals = report["totals"]
    lines: list[str] = [
        f"# VulcanBench report — {suite}",
        "",
        f"_Generated {report['generated_at']}_",
        "",
        f"- **{totals['n_runs']}** runs · **{totals['n_models']}** models · "
        f"**{totals['n_tasks']}** tasks · total cost "
        f"{('$' + str(totals['total_cost_usd'])) if totals['total_cost_usd'] is not None else 'n/a'}",
    ]

    integ = report["integrity"]
    if integ["stale"]:
        lines += [
            "",
            f"> ⚠️ **{integ['stale']} run(s) scored against a now-stale task version** "
            f"(task definition changed since): {', '.join(integ['stale_run_ids'])}",
        ]
    if integ.get("not_decontaminated"):
        lines += [
            "",
            f"> ⚠️ **{integ['not_decontaminated']} run(s) scored against "
            f"non-decontaminated task(s)** "
            f"({', '.join(integ['not_decontaminated_tasks'])}) — these tasks derive from "
            "public sources that predate model training cutoffs; treat their scores with care.",
        ]

    lines += [
        "",
        "## Models",
        "",
        "| Model | Tasks | Runs | pass@1 ± se | pass@k | Avg total | Cost $ | Avg time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in report["models"]:
        cost = "?" if not m.get("cost_known") else _fmt(m.get("total_cost"))
        lines.append(
            f"| {m['model']} | {m['n_tasks']} | {m['n_runs']} | "
            f"{_fmt(m['pass_at_1'])} ± {_fmt(m['pass_at_1_stderr'])} | {_fmt(m['pass_at_k'])} | "
            f"{_fmt(m['avg_total'])} | {cost} | {_fmt(m['avg_duration_s'])} |"
        )

    effort = report.get("effort_sensitivity") or {}
    if effort.get("available"):
        lines += [
            "",
            "## Effort Sensitivity",
            "",
        ]
        for warning in effort.get("warnings") or []:
            lines.append(f"> ⚠️ {warning}")
        if effort.get("warnings"):
            lines.append("")
        lines += [
            "| Model | Language | Scale | Difficulty | Complexity | low pass@1 | high pass@1 | Δ pass@1 | Cost ratio | Latency ratio | Classification |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in effort["strata"]:
            low = row["efforts"].get("low", {})
            high = row["efforts"].get("high", {})
            lines.append(
                f"| {row['model']} | {row['language']} | {row['repo_scale']} | "
                f"{row['difficulty']} | {row['task_complexity']} | "
                f"{_fmt(low.get('pass_at_1'))} | {_fmt(high.get('pass_at_1'))} | "
                f"{_fmt(row.get('high_minus_low_pass_at_1'))} | "
                f"{_fmt(row.get('high_cost_ratio'))} | "
                f"{_fmt(row.get('high_latency_ratio'))} | {row['classification']} |"
            )

    lines += ["", "## Per-task", "", "| Task | Runs | Model | Solve rate |", "|---|---|---|---|"]
    for t in report["tasks"]:
        for ms in t["models"]:
            lines.append(
                f"| {t['task_id']} | {ms['attempts']} | {ms['model']} | "
                f"{ms['solved']}/{ms['attempts']} ({_fmt(ms['solve_rate'])}) |"
            )

    env = report["environment"]
    lines += ["", "## Environment", ""]
    lines.append(f"- Models: {', '.join(env['models']) or '—'}")
    lines.append(f"- Python: {', '.join(env['python']) or '—'}")
    for tool, versions in env["tools"].items():
        lines.append(f"- {tool}: {', '.join(versions)}")

    cal = report.get("calibration")
    if cal and cal["summary"]["n_calibrated"]:
        lines.append("")
        lines.append(calibration_to_markdown(cal))

    return "\n".join(lines) + "\n"
