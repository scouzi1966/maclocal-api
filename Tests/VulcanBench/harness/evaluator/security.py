"""Security metric: static-analysis scanners on changed files, severity-weighted.

Python uses bandit natively. JS/TS use ``npm audit`` (needs a manifest), Go uses
gosec, Java uses spotbugs, Rust uses ``cargo audit`` -- each only when the tool
is present, otherwise the language reports ``None`` with a reason. Score per
language is ``1 - (0.4*high + 0.15*med + 0.05*low)`` clamped to [0, 1]; the
overall score averages whichever languages were scanned.

For Rust, an additional "unsafe delta" penalty applies: 0.05 is subtracted per
new ``unsafe`` block keyword found in the changed files (clamped to [0, 1]).
This is reported in details as ``unsafe_delta`` count and ``unsafe_penalty``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.evaluator.langs import MetricResult, group_by_language

_SEVERITY_PENALTY = {"high": 0.4, "medium": 0.15, "low": 0.05}

RemainingSeconds = Callable[[], float | None]


def assess_security(
    workspace: Path, changed_files: list[str], remaining_s: RemainingSeconds | None = None
) -> MetricResult:
    """Assess security of the agent's changed files."""
    by_lang = group_by_language(changed_files)
    if not by_lang:
        return MetricResult(score=None, details={"reason": "no recognized source files changed"})

    per_lang: dict[str, Any] = {}
    scores: list[float] = []
    for lang, files in by_lang.items():
        if _budget_exhausted(remaining_s):
            result = MetricResult(score=None, details={"reason": "run budget exceeded"})
        else:
            result = _SCANNERS.get(lang, _unsupported)(workspace, files, remaining_s)
        per_lang[lang] = result.details | {"score": result.score}
        if result.score is not None:
            scores.append(result.score)

    overall = round(sum(scores) / len(scores), 4) if scores else None
    reason = None if scores else "no security scanner available for changed languages"
    return MetricResult(
        score=overall,
        details={"languages": per_lang, **({"reason": reason} if reason else {})},
    )


def score_from_counts(high: int, medium: int, low: int) -> float:
    """Severity-weighted score in [0, 1] from issue counts."""
    penalty = (
        _SEVERITY_PENALTY["high"] * high
        + _SEVERITY_PENALTY["medium"] * medium
        + _SEVERITY_PENALTY["low"] * low
    )
    return round(max(0.0, min(1.0, 1.0 - penalty)), 4)


def _timeout(default: int, remaining_s: RemainingSeconds | None) -> int | None:
    if remaining_s is None:
        return default
    remaining = remaining_s()
    if remaining is None:
        return default
    if remaining <= 0:
        return None
    return max(1, min(default, math.ceil(remaining)))


def _budget_exhausted(remaining_s: RemainingSeconds | None) -> bool:
    remaining = remaining_s() if remaining_s is not None else None
    return remaining is not None and remaining <= 0


def _python(
    workspace: Path, files: list[str], remaining_s: RemainingSeconds | None
) -> MetricResult:
    if shutil.which("bandit") is None:
        return MetricResult(score=None, details={"tool": "bandit", "reason": "bandit not on PATH"})
    abs_files = [str((workspace / f).resolve()) for f in files if (workspace / f).exists()]
    if not abs_files:
        return MetricResult(score=None, details={"reason": "changed files no longer exist"})
    timeout = _timeout(180, remaining_s)
    if timeout is None:
        return MetricResult(score=None, details={"tool": "bandit", "reason": "run budget exceeded"})
    try:
        proc = subprocess.run(
            ["bandit", "-f", "json", "-q", *abs_files],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MetricResult(score=None, details={"tool": "bandit", "reason": "timed out"})
    try:
        totals = json.loads(proc.stdout or "{}").get("metrics", {}).get("_totals", {})
    except json.JSONDecodeError:
        return MetricResult(
            score=None, details={"tool": "bandit", "reason": "could not parse bandit output"}
        )
    high = int(totals.get("SEVERITY.HIGH", 0))
    medium = int(totals.get("SEVERITY.MEDIUM", 0))
    low = int(totals.get("SEVERITY.LOW", 0))
    return MetricResult(
        score=score_from_counts(high, medium, low),
        details={"tool": "bandit", "high": high, "medium": medium, "low": low},
    )


def _js_ts(workspace: Path, files: list[str], remaining_s: RemainingSeconds | None) -> MetricResult:
    if shutil.which("npm") is None:
        return MetricResult(score=None, details={"tool": "npm audit", "reason": "npm not on PATH"})
    if not (workspace / "package.json").exists():
        return MetricResult(
            score=None, details={"tool": "npm audit", "reason": "no package.json in workspace"}
        )
    timeout = _timeout(180, remaining_s)
    if timeout is None:
        return MetricResult(
            score=None, details={"tool": "npm audit", "reason": "run budget exceeded"}
        )
    try:
        proc = subprocess.run(
            ["npm", "audit", "--json"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MetricResult(score=None, details={"tool": "npm audit", "reason": "timed out"})
    try:
        vulns = json.loads(proc.stdout or "{}").get("metadata", {}).get("vulnerabilities", {})
    except json.JSONDecodeError:
        return MetricResult(
            score=None, details={"tool": "npm audit", "reason": "could not parse npm audit output"}
        )
    high = int(vulns.get("high", 0)) + int(vulns.get("critical", 0))
    medium = int(vulns.get("moderate", 0))
    low = int(vulns.get("low", 0)) + int(vulns.get("info", 0))
    return MetricResult(
        score=score_from_counts(high, medium, low),
        details={"tool": "npm audit", "high": high, "medium": medium, "low": low},
    )


def _go(workspace: Path, files: list[str], remaining_s: RemainingSeconds | None) -> MetricResult:
    if shutil.which("gosec") is None:
        return MetricResult(score=None, details={"tool": "gosec", "reason": "gosec not on PATH"})
    timeout = _timeout(180, remaining_s)
    if timeout is None:
        return MetricResult(score=None, details={"tool": "gosec", "reason": "run budget exceeded"})
    try:
        proc = subprocess.run(
            ["gosec", "-fmt=json", "./..."],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MetricResult(score=None, details={"tool": "gosec", "reason": "timed out"})
    try:
        issues = json.loads(proc.stdout or "{}").get("Issues", [])
    except json.JSONDecodeError:
        return MetricResult(
            score=None, details={"tool": "gosec", "reason": "could not parse gosec output"}
        )
    high = sum(1 for i in issues if i.get("severity") == "HIGH")
    medium = sum(1 for i in issues if i.get("severity") == "MEDIUM")
    low = sum(1 for i in issues if i.get("severity") == "LOW")
    return MetricResult(
        score=score_from_counts(high, medium, low),
        details={"tool": "gosec", "high": high, "medium": medium, "low": low},
    )


def _java(workspace: Path, files: list[str], remaining_s: RemainingSeconds | None) -> MetricResult:
    if shutil.which("spotbugs") is None:
        return MetricResult(
            score=None, details={"tool": "spotbugs", "reason": "spotbugs not on PATH"}
        )
    timeout = _timeout(240, remaining_s)
    if timeout is None:
        return MetricResult(
            score=None, details={"tool": "spotbugs", "reason": "run budget exceeded"}
        )
    try:
        proc = subprocess.run(
            ["spotbugs", "-textui", "-xml:withMessages", *files],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MetricResult(score=None, details={"tool": "spotbugs", "reason": "timed out"})
    # SpotBugs ranks 1-20; treat as one bucket of "medium" findings for v1.
    findings = proc.stdout.count("<BugInstance ")
    return MetricResult(
        score=score_from_counts(0, findings, 0),
        details={"tool": "spotbugs", "findings": findings},
    )


def _rust(workspace: Path, files: list[str], remaining_s: RemainingSeconds | None) -> MetricResult:  # noqa: PLR0911
    if shutil.which("cargo") is None:
        return MetricResult(
            score=None, details={"tool": "cargo audit", "reason": "cargo not on PATH"}
        )
    if not (workspace / "Cargo.lock").exists():
        return MetricResult(
            score=None, details={"tool": "cargo audit", "reason": "no Cargo.lock in workspace"}
        )
    if _budget_exhausted(remaining_s):
        return MetricResult(
            score=None, details={"tool": "cargo audit", "reason": "run budget exceeded"}
        )
    timeout = _timeout(180, remaining_s)
    if timeout is None:
        return MetricResult(
            score=None, details={"tool": "cargo audit", "reason": "run budget exceeded"}
        )
    try:
        proc = subprocess.run(
            ["cargo", "audit", "--json"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MetricResult(score=None, details={"tool": "cargo audit", "reason": "timed out"})

    # cargo audit exits non-zero when vulnerabilities are found; JSON is on stdout.
    try:
        report = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return MetricResult(
            score=None,
            details={"tool": "cargo audit", "reason": "could not parse cargo audit output"},
        )

    vulnerabilities = report.get("vulnerabilities", {})
    counts = vulnerabilities.get("count", 0) if isinstance(vulnerabilities, dict) else 0
    # Iterate individual vulnerability entries to map severities.
    vuln_list = vulnerabilities.get("list", []) if isinstance(vulnerabilities, dict) else []
    high = 0
    medium = 0
    low = 0
    for v in vuln_list:
        severity = str(v.get("severity", "")).lower()
        advisory = v.get("advisory", {})
        if not severity and isinstance(advisory, dict):
            severity = str(advisory.get("severity", "")).lower()
        if severity in ("critical", "high"):
            high += 1
        elif severity == "medium":
            medium += 1
        else:
            low += 1

    base_score = score_from_counts(high, medium, low)

    # Unsafe delta: count "unsafe" keywords in changed .rs files.
    unsafe_delta = _count_unsafe_delta(workspace, files)
    unsafe_penalty = round(min(1.0, 0.05 * unsafe_delta), 4)
    final_score = (
        round(max(0.0, base_score - unsafe_penalty), 4) if base_score is not None else None
    )

    return MetricResult(
        score=final_score,
        details={
            "tool": "cargo audit",
            "vulnerabilities": counts,
            "high": high,
            "medium": medium,
            "low": low,
            "unsafe_delta": unsafe_delta,
            "unsafe_penalty": unsafe_penalty,
        },
    )


_UNSAFE_RE = re.compile(r"\bunsafe\b")


def _count_unsafe_delta(workspace: Path, files: list[str]) -> int:
    """Count ``unsafe`` keyword occurrences in the changed Rust files."""
    count = 0
    for f in files:
        if not f.endswith(".rs"):
            continue
        path = workspace / f
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        count += len(_UNSAFE_RE.findall(src))
    return count


def _unsupported(
    workspace: Path, files: list[str], remaining_s: RemainingSeconds | None
) -> MetricResult:
    del workspace, files, remaining_s
    return MetricResult(score=None, details={"reason": "no security scanner for this language"})


# Per-language scanner registry.
_SCANNERS = {
    "python": _python,
    "typescript": _js_ts,
    "javascript": _js_ts,
    "go": _go,
    "java": _java,
    "rust": _rust,
}
