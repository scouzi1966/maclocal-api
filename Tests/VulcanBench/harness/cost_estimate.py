"""Pre-run cost estimation from local run history and bundled priors.

Estimates are honest ranges built from completed runs in ``runs_dir`` (excluding
``mock:*``). When a (model, task) pair has no history, we fall back to bundled
cost priors, then the same task on other models (scaled by relative list price),
then the model's median run cost, then a conservative default derived from
observed data.

These are planning numbers — actual spend varies with model behavior, retries,
and judges. Use ``recommended_usd`` as a minimum credit buffer, not a hard cap.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.cost_priors import PriorBuckets, PriorRange, load_cost_priors
from harness.leaderboard import load_summaries
from harness.pricing import _rate, is_priced
from harness.task_metadata import repo_scale
from harness.tasks import load_task

# Ignore free/offline runs when building the index.
_SKIP_MODEL_PREFIXES = ("mock:",)

# When we have no history at all for a priced model, assume this per run (USD).
_DEFAULT_PER_RUN: dict[str, float] = {
    "openai:": 0.06,
    "anthropic:": 0.055,
    "zai:": 0.055,
}
_DEFAULT_FALLBACK = 0.08

_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zai": "ZAI_API_KEY",
}

_PROVIDER_LABEL = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "zai": "Z.ai (Zhipu)",
}

_KNOWN_TASK_SOURCES = frozenset({"exact", "prior_exact"})
_PRIOR_SOURCES = frozenset(
    {"prior_exact", "prior_task_scaled", "prior_model_median"}
)


@dataclass
class RunCostEstimate:
    """Estimated USD for one (model, task) run."""

    task_id: str
    low_usd: float
    mid_usd: float
    high_usd: float
    source: str


@dataclass
class ModelCostEstimate:
    model: str
    provider: str
    env_var: str
    n_runs: int
    low_usd: float
    mid_usd: float
    high_usd: float
    recommended_usd: float
    confidence: str
    # high | medium | low
    per_task: list[RunCostEstimate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanCostEstimate:
    """Roll-up for a benchmark plan (one or more models)."""

    task_ids: list[str]
    repeat: int
    judges: bool
    models: list[ModelCostEstimate]
    low_usd: float
    mid_usd: float
    high_usd: float
    recommended_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_ids": self.task_ids,
            "repeat": self.repeat,
            "judges": self.judges,
            "low_usd": self.low_usd,
            "mid_usd": self.mid_usd,
            "high_usd": self.high_usd,
            "recommended_usd": self.recommended_usd,
            "models": [
                {
                    "model": m.model,
                    "provider": m.provider,
                    "env_var": m.env_var,
                    "n_runs": m.n_runs,
                    "low_usd": m.low_usd,
                    "mid_usd": m.mid_usd,
                    "high_usd": m.high_usd,
                    "recommended_usd": m.recommended_usd,
                    "confidence": m.confidence,
                    "notes": m.notes,
                    "per_task": [
                        {
                            "task_id": t.task_id,
                            "low_usd": t.low_usd,
                            "mid_usd": t.mid_usd,
                            "high_usd": t.high_usd,
                            "source": t.source,
                        }
                        for t in m.per_task
                    ],
                }
                for m in self.models
            ],
        }


def _provider(model: str) -> str:
    return model.split(":", 1)[0] if ":" in model else model


def _price_index(model: str) -> float | None:
    """Single scalar for cross-model scaling (input + output $/1M)."""
    rate = _rate(model)
    if rate is None:
        return None
    return rate["input"] + rate["output"]


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


def _estimate_from_range(
    task_id: str, low: float, mid: float, high: float, source: str
) -> RunCostEstimate:
    return RunCostEstimate(
        task_id=task_id,
        low_usd=round(low, 4),
        mid_usd=round(mid, 4),
        high_usd=round(high, 4),
        source=source,
    )


def _estimate_from_prior(task_id: str, prior: PriorRange, source: str) -> RunCostEstimate:
    return _estimate_from_range(task_id, prior.low, prior.mid, prior.high, source)


def _task_scaled_estimate(
    task_id: str,
    model: str,
    entries: list[tuple[str, float]],
    source: str,
) -> RunCostEstimate | None:
    target_idx = _price_index(model)
    if not entries or not target_idx:
        return None
    scaled: list[float] = []
    for src_model, c in entries:
        src_idx = _price_index(src_model)
        if src_idx and src_idx > 0:
            scaled.append(c * (target_idx / src_idx))
    if not scaled:
        return None
    return RunCostEstimate(
        task_id=task_id,
        low_usd=round(_percentile(scaled, 0.25), 4),
        mid_usd=round(statistics.median(scaled), 4),
        high_usd=round(_percentile(scaled, 0.90), 4),
        source=source,
    )


@dataclass
class _CostIndex:
    by_model_task: dict[tuple[str, str], list[float]]
    by_task: dict[str, list[float]]
    by_model: dict[str, list[float]]
    by_model_scale: dict[tuple[str, str], list[float]]
    priors: PriorBuckets = field(default_factory=PriorBuckets.empty)

    @classmethod
    def from_runs(
        cls,
        runs_dir: Path,
        priors: PriorBuckets | None = None,
    ) -> _CostIndex:
        by_model_task: dict[tuple[str, str], list[float]] = defaultdict(list)
        by_task: dict[str, list[float]] = defaultdict(list)
        by_model: dict[str, list[float]] = defaultdict(list)
        by_model_scale: dict[tuple[str, str], list[float]] = defaultdict(list)

        for summary in load_summaries(runs_dir):
            model = summary.get("model") or ""
            task_id = summary.get("task_id") or ""
            cost = summary.get("cost_usd")
            if not model or not task_id or cost is None:
                continue
            if any(model.startswith(p) for p in _SKIP_MODEL_PREFIXES):
                continue
            c = float(cost)
            by_model_task[(model, task_id)].append(c)
            by_task[task_id].append(c)
            by_model[model].append(c)
            scale = (summary.get("manifest") or {}).get("task", {}).get("repo_scale") or "unknown"
            by_model_scale[(model, scale)].append(c)

        return cls(
            by_model_task=dict(by_model_task),
            by_task=dict(by_task),
            by_model=dict(by_model),
            by_model_scale=dict(by_model_scale),
            priors=priors or PriorBuckets.empty(),
        )

    def _default_for_model(self, model: str) -> float:
        costs = self.by_model.get(model)
        if costs:
            return statistics.median(costs)
        prefix = _provider(model) + ":"
        if prefix in _DEFAULT_PER_RUN:
            return _DEFAULT_PER_RUN[prefix]
        return _DEFAULT_FALLBACK

    def estimate_one(self, model: str, task_id: str, tasks_root: Path) -> RunCostEstimate:
        direct = self.by_model_task.get((model, task_id))
        if direct:
            return RunCostEstimate(
                task_id=task_id,
                low_usd=round(_percentile(direct, 0.25), 4),
                mid_usd=round(statistics.median(direct), 4),
                high_usd=round(_percentile(direct, 0.90), 4),
                source="exact",
            )

        prior_direct = self.priors.by_model_task.get((model, task_id))
        if prior_direct is not None:
            return _estimate_from_prior(task_id, prior_direct, "prior_exact")

        task_entries = [
            (m, c) for (m, t), costs in self.by_model_task.items() if t == task_id for c in costs
        ]
        scaled_local = _task_scaled_estimate(task_id, model, task_entries, "task_scaled")
        if scaled_local is not None:
            return scaled_local

        prior_task_entries = [
            (m, p.mid)
            for (m, t), p in self.priors.by_model_task.items()
            if t == task_id
        ]
        scaled_prior = _task_scaled_estimate(
            task_id, model, prior_task_entries, "prior_task_scaled"
        )
        if scaled_prior is not None:
            return scaled_prior

        try:
            meta = load_task(task_id, tasks_root).metadata
            scale = repo_scale(meta)
        except FileNotFoundError:
            scale = "unknown"

        scale_costs = self.by_model_scale.get((model, scale), [])
        if scale_costs:
            return RunCostEstimate(
                task_id=task_id,
                low_usd=round(_percentile(scale_costs, 0.25), 4),
                mid_usd=round(statistics.median(scale_costs), 4),
                high_usd=round(_percentile(scale_costs, 0.90), 4),
                source="model_median",
            )

        prior_scale = self.priors.by_model_scale.get((model, scale))
        if prior_scale is not None:
            return _estimate_from_prior(task_id, prior_scale, "prior_model_median")

        base = self._default_for_model(model)
        return RunCostEstimate(
            task_id=task_id,
            low_usd=round(base * 0.6, 4),
            mid_usd=round(base, 4),
            high_usd=round(base * 1.8, 4),
            source="default",
        )


def _task_judge_mult(judges: bool, source: str, priors: PriorBuckets) -> float:
    """Scale per-task USD to match requested ``--judges`` setting."""
    if source.startswith("prior"):
        if judges:
            return 1.0 if priors.judges else 3.0
        return 1.0 / 3.0 if priors.judges else 1.0
    return 3.0 if judges else 1.0


def estimate_plan(
    *,
    models: list[str],
    task_ids: list[str],
    repeat: int = 1,
    judges: bool = False,
    runs_dir: Path = Path("./runs"),
    tasks_root: Path | None = None,
    credit_buffer: float = 1.25,
    use_priors: bool = True,
) -> PlanCostEstimate:
    """Estimate USD for running ``task_ids`` with each model, ``repeat`` times each."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if not models:
        raise ValueError("at least one model required")
    if not task_ids:
        raise ValueError("at least one task required")

    tasks_root = tasks_root or Path("tasks/v1")
    priors = load_cost_priors() if use_priors else PriorBuckets.empty()
    index = _CostIndex.from_runs(runs_dir, priors=priors)

    model_estimates: list[ModelCostEstimate] = []
    for model in models:
        if not is_priced(model):
            raise ValueError(f"no built-in pricing for model {model!r}; cannot estimate cost")

        per_task = [index.estimate_one(model, tid, tasks_root) for tid in task_ids]
        n_known = sum(1 for t in per_task if t.source in _KNOWN_TASK_SOURCES)
        n_local_exact = sum(1 for t in per_task if t.source == "exact")
        n_prior_only = sum(
            1 for t in per_task if t.source in _PRIOR_SOURCES and t.source != "prior_exact"
        )
        n_prior_exact = sum(1 for t in per_task if t.source == "prior_exact")

        if n_known >= len(task_ids) * 0.75:
            confidence = "high"
        elif n_known >= len(task_ids) * 0.25:
            confidence = "medium"
        else:
            confidence = "low"

        if n_local_exact == 0 and n_prior_exact > 0 and confidence == "high":
            confidence = "medium"

        low = round(
            sum(
                t.low_usd * _task_judge_mult(judges, t.source, priors) for t in per_task
            )
            * repeat,
            4,
        )
        mid = round(
            sum(
                t.mid_usd * _task_judge_mult(judges, t.source, priors) for t in per_task
            )
            * repeat,
            4,
        )
        high = round(
            sum(
                t.high_usd * _task_judge_mult(judges, t.source, priors) for t in per_task
            )
            * repeat,
            4,
        )
        recommended = round(high * credit_buffer, 2)

        notes: list[str] = []
        if judges and any(_task_judge_mult(True, t.source, priors) > 1.0 for t in per_task):
            notes.append("Includes ~3x agent-token judge ensemble (--judges).")
        if n_prior_exact > 0 and n_local_exact == 0:
            notes.append(
                f"Using bundled cost priors (no local ./runs history for "
                f"{n_prior_exact} task(s))."
            )
        elif n_prior_only > 0:
            notes.append(f"Partial bundled priors for {n_prior_only} task(s).")
        if confidence == "low":
            notes.append("Limited history; defaults are conservative — load extra credit.")
        unknown = [t.task_id for t in per_task if t.source == "default"]
        if unknown:
            notes.append(f"No history for {len(unknown)} task(s); using defaults.")

        prov = _provider(model)
        model_estimates.append(
            ModelCostEstimate(
                model=model,
                provider=_PROVIDER_LABEL.get(prov, prov),
                env_var=_PROVIDER_ENV.get(prov, f"{prov.upper()}_API_KEY"),
                n_runs=len(task_ids) * repeat,
                low_usd=low,
                mid_usd=mid,
                high_usd=high,
                recommended_usd=recommended,
                confidence=confidence,
                per_task=per_task,
                notes=notes,
            )
        )

    return PlanCostEstimate(
        task_ids=list(task_ids),
        repeat=repeat,
        judges=judges,
        models=model_estimates,
        low_usd=round(sum(m.low_usd for m in model_estimates), 4),
        mid_usd=round(sum(m.mid_usd for m in model_estimates), 4),
        high_usd=round(sum(m.high_usd for m in model_estimates), 4),
        recommended_usd=round(sum(m.recommended_usd for m in model_estimates), 2),
    )
