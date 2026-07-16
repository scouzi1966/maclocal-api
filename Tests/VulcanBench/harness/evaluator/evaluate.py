"""Orchestrates the full multi-metric evaluation of a finished run.

Runs quality + security analyzers and (optionally) the judge ensemble, records a
transparent trace event per metric, and returns the combined score dict from
:func:`harness.evaluator.scorer.score_run`. The per-metric breakdowns are
attached under ``metric_details`` for the run summary / dashboard.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.evaluator.judges import assess_human_like
from harness.evaluator.langs import MetricResult
from harness.evaluator.quality import assess_quality
from harness.evaluator.scorer import score_run
from harness.evaluator.security import assess_security

if TYPE_CHECKING:
    from harness.agent.providers import LLMProvider
    from harness.tracer.collector import TraceCollector

RemainingSeconds = Callable[[], float | None]
BudgetExceeded = Callable[[str], None]


def evaluate_run(
    *,
    functional: float,
    total_tokens: int,
    steps: int,
    workspace: Path,
    patch: str,
    changed_files: list[str],
    issue: str,
    verifier_payload: dict[str, Any],
    judges_enabled: bool = True,
    judge_provider: LLMProvider | None = None,
    collector: TraceCollector | None = None,
    remaining_s: RemainingSeconds | None = None,
    budget_exceeded: BudgetExceeded | None = None,
) -> dict[str, Any]:
    """Compute all metrics for a run and return the combined score dict."""
    if _budget_is_exhausted(remaining_s):
        _mark_budget_exceeded(budget_exceeded, "quality")
        quality = MetricResult(score=None, details={"reason": "run budget exceeded"})
    else:
        quality = assess_quality(workspace, changed_files, remaining_s=remaining_s)
    _record(collector, "quality_detail", quality.details)

    if _budget_is_exhausted(remaining_s):
        _mark_budget_exceeded(budget_exceeded, "security")
        security = MetricResult(score=None, details={"reason": "run budget exceeded"})
    else:
        security = assess_security(workspace, changed_files, remaining_s=remaining_s)
    _record(collector, "security_detail", security.details)

    _warn_if_unscored("quality", quality)
    _warn_if_unscored("security", security)

    human_like_score: float | None = None
    human_like_details: dict[str, Any]
    if _budget_is_exhausted(remaining_s):
        _mark_budget_exceeded(budget_exceeded, "human_like")
        human_like_details = {"reason": "run budget exceeded"}
    elif not judges_enabled:
        human_like_details = {"reason": "judges disabled"}
    elif judge_provider is None:
        human_like_details = {
            "reason": "no judge provider available (missing API key or invalid model)"
        }
    else:
        verifier_summary = _summarize_verifier(verifier_payload, functional)
        human = assess_human_like(
            issue, patch, verifier_summary, judge_provider, remaining_s=remaining_s
        )
        human_like_score = human.score
        human_like_details = human.details
        if _budget_is_exhausted(remaining_s):
            _mark_budget_exceeded(budget_exceeded, "human_like")
        for vote in human.details.get("votes", []):
            _record(collector, "judge_vote", vote)
    _record(collector, "human_like_detail", human_like_details)

    scores = score_run(
        functional=functional,
        total_tokens=total_tokens,
        steps=steps,
        quality=quality.score,
        security=security.score,
        human_like=human_like_score,
        extra={
            "metric_details": {
                "quality": quality.details,
                "security": security.details,
                "human_like": human_like_details,
            }
        },
    )
    return scores


def _summarize_verifier(verifier_payload: dict[str, Any], functional: float) -> str:
    passed = "passed" if functional >= 1.0 else f"partial/failed (functional={functional})"
    err = verifier_payload.get("scores", {}).get("error")
    return f"Tests {passed}." + (f" Verifier note: {err}" if err else "")


def _record(collector: TraceCollector | None, event: str, data: dict[str, Any]) -> None:
    if collector is not None:
        collector.record(event, data)


def _warn_if_unscored(metric: str, result: MetricResult) -> None:
    """Make degraded scoring loud: a None score (e.g. analyzer missing from PATH)
    is recorded in the trace, but operators reading the console should see it too."""
    if result.score is not None:
        return
    reasons = {
        str(detail["reason"])
        for detail in (result.details, *result.details.values())
        if isinstance(detail, dict) and detail.get("reason")
    }
    note = "; ".join(sorted(reasons)) or "no reason recorded"
    print(f"[vulcanbench] WARNING: {metric} score is None ({note})", file=sys.stderr)


def _budget_is_exhausted(remaining_s: RemainingSeconds | None) -> bool:
    remaining = remaining_s() if remaining_s is not None else None
    return remaining is not None and remaining <= 0


def _mark_budget_exceeded(callback: BudgetExceeded | None, stage: str) -> None:
    if callback is not None:
        callback(stage)
