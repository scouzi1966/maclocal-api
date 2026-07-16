"""Shared evaluator value types and language detection.

This is the lowest-level evaluator module (no intra-package imports) so the
per-metric analyzers and the orchestrator can all depend on it without cycles.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result of one metric assessment.

    ``score`` is on a 0.0-1.0 scale, or ``None`` when the metric could not be
    computed (e.g. no analyzer available for the changed languages). ``details``
    carries the transparent breakdown -- tools used, per-language sub-scores,
    counts, and a ``reason`` whenever something was skipped.
    """

    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# File extension -> canonical language name.
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}


def detect_language(path: str) -> str | None:
    """Return the canonical language for a file path, or ``None`` if unknown."""
    return EXT_TO_LANG.get(Path(path).suffix.lower())


def group_by_language(files: list[str]) -> dict[str, list[str]]:
    """Group file paths by detected language, dropping unrecognized extensions."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for f in files:
        lang = detect_language(f)
        if lang is not None:
            grouped[lang].append(f)
    return dict(grouped)
