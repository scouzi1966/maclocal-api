"""Suite definition and runner.

A *suite* is a named set of tasks (e.g. ``v1`` → the tasks under ``tasks/v1/``).
``run_suite`` runs a model against every task in the suite, tags each run so the
leaderboard can group it, and writes an aggregate ``suite.json``.
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.agent.loop import run_agent
from harness.leaderboard import aggregate_by_model, scan_leaderboard
from harness.task_metadata import repo_scale
from harness.tasks import list_task_ids, load_task

DEFAULT_TASKS_BASE = Path("tasks")
SUITE_ALIASES = {"v1-micro": "v1", "v1-large": "v1", "v1-rust": "v1", "v1-compare": "v1"}


@dataclass
class Suite:
    name: str
    tasks_root: Path
    task_ids: list[str]


def _scale_filter(tasks_root: Path, tier: str) -> list[str]:
    """Filter task ids by ``repo_scale`` tier (micro | large)."""
    out: list[str] = []
    for task_id in list_task_ids(tasks_root):
        meta = load_task(task_id, tasks_root).metadata
        scale = repo_scale(meta)
        if (tier == "micro" and scale in ("micro", "small")) or (
            tier == "large" and scale in ("medium", "large")
        ):
            out.append(task_id)
    return out


def _language_filter(tasks_root: Path, language: str) -> list[str]:
    """Filter task ids by presence of ``language`` in metadata.languages."""
    out: list[str] = []
    for task_id in list_task_ids(tasks_root):
        meta = load_task(task_id, tasks_root).metadata
        langs = meta.get("languages", [])
        if language in langs:
            out.append(task_id)
    return out


def load_suite(name: str, tasks_base: Path = DEFAULT_TASKS_BASE) -> Suite:
    """Load a suite by name.

    The task set is every task directory under ``tasks/<name>/``, unless a
    ``tasks/<name>/suite.json`` manifest pins a subset/order. Aliases ``v1-micro``
    and ``v1-large`` read ``tasks/v1/suite.json`` keys ``micro`` / ``large``.
    ``v1-rust`` filters by language ``rust`` in metadata.languages.
    """
    if name in SUITE_ALIASES:
        tasks_root = tasks_base / SUITE_ALIASES[name]
        if name == "v1-rust":
            task_ids = list(_language_filter(tasks_root, "rust"))
            return Suite(name=name, tasks_root=tasks_root, task_ids=task_ids)
        manifest = tasks_root / "suite.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if name == "v1-compare":
                task_ids = list(data.get("compare") or [])
            else:
                key = "micro" if name == "v1-micro" else "large"
                task_ids = list(data.get(key) or _scale_filter(tasks_root, key))
        else:
            key = "micro" if name == "v1-micro" else "large"
            task_ids = _scale_filter(tasks_root, key)
        return Suite(name=name, tasks_root=tasks_root, task_ids=task_ids)

    tasks_root = tasks_base / name
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"suite {name!r} not found under {tasks_base}")
    manifest = tasks_root / "suite.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        task_ids = list(data.get("full") or data.get("tasks") or list_task_ids(tasks_root))
    else:
        task_ids = list_task_ids(tasks_root)
    return Suite(name=name, tasks_root=tasks_root, task_ids=task_ids)


def run_suite(  # noqa: PLR0915 — linear scheduler: validation + budget loop + summary assembly
    name: str,
    model: str,
    output_dir: Path = Path("./runs"),
    tasks_base: Path = DEFAULT_TASKS_BASE,
    repeat: int = 1,
    max_concurrency: int = 1,
    max_cost: float | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run ``model`` against every task in the suite, ``repeat`` times each.

    With ``max_concurrency > 1`` the (task x repeat) units run on a thread pool.
    Each unit is a fully isolated :func:`run_agent` call (its own run dir, git
    workspace, collector and provider), so results are identical to sequential
    execution — only wall-clock differs. One unit raising does not sink the
    suite; its failure is recorded in ``errors``.

    ``max_cost`` is a USD spend cap: once the accumulated cost of completed runs
    reaches it, no *new* runs are launched and the remainder are recorded in
    ``skipped``. It is not a hard ceiling — runs already in flight finish, so the
    total can exceed ``max_cost`` by up to ``max_concurrency`` runs' worth. It
    **fails closed**: if a completed run reports no cost (e.g. an unpriced model),
    the budget can't be guaranteed, so launching stops (``cost_unknown=True`` in
    the summary) rather than silently running the whole suite.

    Returns the suite summary (also written to ``<output_dir>/<suite_id>/suite.json``).
    Extra keyword args (judges, sandbox, image, network, max_steps, …) pass
    through to :func:`run_agent`.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    if max_cost is not None and max_cost <= 0:
        raise ValueError("max_cost must be > 0")
    suite = load_suite(name, tasks_base)
    suite_id = f"suite-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(UTC)

    units = [task_id for task_id in suite.task_ids for _ in range(repeat)]

    def _run_one(task_id: str) -> dict[str, Any]:
        res = run_agent(
            task_id=task_id,
            model=model,
            output_dir=output_dir,
            tasks_root=suite.tasks_root,
            suite=name,
            suite_id=suite_id,
            **run_kwargs,
        )
        summary = res["summary"]
        return {
            "task_id": task_id,
            "run_id": res["run_id"],
            "total": summary.get("scores", {}).get("total"),
            "functional": summary.get("scores", {}).get("functional"),
            "cost_usd": summary.get("cost_usd"),
            "duration_s": summary.get("duration_s"),
            "effort": summary.get("effort"),
            "experiment_id": summary.get("experiment_id"),
        }

    task_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped: list[str] = []
    spent = 0.0
    cost_unknown = False  # a completed run reported no cost while a budget is set

    def _budget_reached() -> bool:
        # Fail CLOSED: if we can't price a completed run, we can't guarantee the
        # cap, so stop launching rather than silently running the whole suite.
        return max_cost is not None and (cost_unknown or spent >= max_cost)

    # One scheduler for both sequential (max_concurrency=1) and parallel: keep up
    # to max_concurrency runs in flight, and stop launching new ones once the
    # spend cap is reached. In-flight runs are always allowed to finish.
    pending = list(units)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        in_flight: dict[Any, str] = {}
        while pending or in_flight:
            while pending and len(in_flight) < max_concurrency and not _budget_reached():
                task_id = pending.pop(0)
                in_flight[pool.submit(_run_one, task_id)] = task_id
            if pending and _budget_reached():
                skipped.extend(pending)
                pending = []
            if not in_flight:
                break
            done, _ = wait(set(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                task_id = in_flight.pop(fut)
                try:
                    result = fut.result()
                    task_results.append(result)
                    cost = result.get("cost_usd")
                    if cost is None:
                        if max_cost is not None:
                            cost_unknown = True  # can't account this run -> fail closed
                    else:
                        spent += cost
                except Exception as e:  # contain one unit's failure; keep the suite going
                    errors.append({"task_id": task_id, "error": str(e)})

    # Deterministic order regardless of completion order under concurrency.
    task_results.sort(key=lambda r: (r["task_id"], r["run_id"]))

    # Aggregate only this invocation's runs (match suite_id, not just the name).
    suite_rows = [r for r in scan_leaderboard(output_dir) if r.get("suite_id") == suite_id]
    aggregate = aggregate_by_model(suite_rows, suite=name)

    suite_summary = {
        "suite_id": suite_id,
        "suite": name,
        "model": model,
        "effort": run_kwargs.get("effort"),
        "experiment_id": run_kwargs.get("experiment_id"),
        "repeat": repeat,
        "max_concurrency": max_concurrency,
        "max_cost": max_cost,
        "spent_usd": round(spent, 6),
        "cost_unknown": cost_unknown,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "n_tasks": len(suite.task_ids),
        "n_runs": len(task_results),
        "n_skipped": len(skipped),
        "tasks": task_results,
        "errors": errors,
        "skipped": skipped,
        "aggregate": aggregate,
    }
    suite_dir = output_dir / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "suite.json").write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")
    return suite_summary
