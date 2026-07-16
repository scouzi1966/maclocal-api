"""Bundled benchmark cost priors for cold-start estimation.

When local ``./runs`` history is absent, ``harness.cost_estimate`` falls back to
shipped priors before hardcoded provider defaults. Override with
``VULCANBENCH_COST_PRIORS`` (path to JSON) or pass an explicit path to
``load_cost_priors``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PriorRange:
    """USD range for one bucket (p25 / median / p90 per run)."""

    low: float
    mid: float
    high: float
    n: int = 0


@dataclass(frozen=True)
class PriorBuckets:
    """Indexed priors from ``cost_priors.json``."""

    by_model_task: dict[tuple[str, str], PriorRange]
    by_model_scale: dict[tuple[str, str], PriorRange]
    judges: bool = True

    @classmethod
    def empty(cls) -> PriorBuckets:
        return cls(by_model_task={}, by_model_scale={})


def _parse_range(raw: Any) -> PriorRange | None:
    if not isinstance(raw, dict):
        return None
    low = raw.get("low")
    mid = raw.get("mid")
    high = raw.get("high")
    if not all(isinstance(v, (int, float)) for v in (low, mid, high)):
        return None
    n = raw.get("n", 0)
    n_int = int(n) if isinstance(n, (int, float)) else 0
    return PriorRange(low=float(low), mid=float(mid), high=float(high), n=n_int)


def _parse_priors(data: Any) -> PriorBuckets:
    if not isinstance(data, dict):
        return PriorBuckets.empty()
    if data.get("version") != 1:
        return PriorBuckets.empty()

    by_model_task: dict[tuple[str, str], PriorRange] = {}
    raw_mt = data.get("by_model_task")
    if isinstance(raw_mt, dict):
        for model, tasks in raw_mt.items():
            if not isinstance(model, str) or not isinstance(tasks, dict):
                continue
            for task_id, entry in tasks.items():
                if not isinstance(task_id, str):
                    continue
                parsed = _parse_range(entry)
                if parsed is not None:
                    by_model_task[(model, task_id)] = parsed

    by_model_scale: dict[tuple[str, str], PriorRange] = {}
    raw_ms = data.get("by_model_scale")
    if isinstance(raw_ms, dict):
        for model, scales in raw_ms.items():
            if not isinstance(model, str) or not isinstance(scales, dict):
                continue
            for scale, entry in scales.items():
                if not isinstance(scale, str):
                    continue
                parsed = _parse_range(entry)
                if parsed is not None:
                    by_model_scale[(model, scale)] = parsed

    judges = data.get("judges", True)
    return PriorBuckets(
        by_model_task=by_model_task,
        by_model_scale=by_model_scale,
        judges=bool(judges),
    )


def _read_json(path: Path) -> PriorBuckets:
    try:
        with path.open(encoding="utf-8") as f:
            return _parse_priors(json.load(f))
    except (OSError, json.JSONDecodeError):
        return PriorBuckets.empty()


def _bundled_path() -> Path:
    return Path(str(files("harness").joinpath("data/cost_priors.json")))


@lru_cache(maxsize=1)
def load_cost_priors(path: str | None = None) -> PriorBuckets:
    """Load cost priors from explicit path, env override, or bundled data."""
    if path is not None:
        return _read_json(Path(path))

    override = os.environ.get("VULCANBENCH_COST_PRIORS")
    if override and os.path.exists(override):
        return _read_json(Path(override))

    bundled = _bundled_path()
    if bundled.is_file():
        return _read_json(bundled)
    return PriorBuckets.empty()


def reset_cache() -> None:
    """Clear memoized priors (for tests after env changes)."""
    load_cost_priors.cache_clear()
