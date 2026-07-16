"""Tests for the security metric."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.evaluator import security
from harness.evaluator.security import assess_security, score_from_counts


def test_score_from_counts_monotonic() -> None:
    assert score_from_counts(0, 0, 0) == 1.0
    assert score_from_counts(0, 0, 1) < 1.0
    assert score_from_counts(1, 0, 0) < score_from_counts(0, 1, 0)
    assert score_from_counts(10, 0, 0) == 0.0  # clamped, never negative


@pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not installed")
def test_security_flags_shell_injection(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "danger.py").write_text(
        "import subprocess\ndef run(cmd):\n    subprocess.call(cmd, shell=True)\n"
    )
    safe = assess_security(tmp_path, ["safe.py"])
    danger = assess_security(tmp_path, ["danger.py"])
    assert safe.score == 1.0
    assert danger.score is not None and danger.score < 1.0
    assert danger.details["languages"]["python"]["high"] >= 1


def test_security_no_source_files(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("# hi")
    result = assess_security(tmp_path, ["readme.md"])
    assert result.score is None
    assert "reason" in result.details


def test_security_python_missing_bandit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(security.shutil, "which", lambda name: None)
    result = assess_security(tmp_path, ["a.py"])
    assert result.details["languages"]["python"]["reason"] == "bandit not on PATH"
