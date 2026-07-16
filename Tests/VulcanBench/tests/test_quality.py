"""Tests for the quality metric."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.evaluator import quality
from harness.evaluator.quality import assess_quality

pytestmark = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")

_CLEAN = "def add(a: int, b: int) -> int:\n    return a + b\n"
_MESSY = (
    "import subprocess\n"
    "def f(x):\n"
    "    unused = 5\n"
    "    if x > 0:\n"
    "        if x > 1:\n"
    "            if x > 2:\n"
    "                if x > 3:\n"
    "                    return 1\n"
    "    return 0\n"
)


def test_quality_clean_beats_messy(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text(_CLEAN)
    (tmp_path / "messy.py").write_text(_MESSY)
    clean = assess_quality(tmp_path, ["clean.py"])
    messy = assess_quality(tmp_path, ["messy.py"])
    assert clean.score is not None and messy.score is not None
    assert clean.score > messy.score
    assert clean.details["languages"]["python"]["tool"] == "ruff+radon"


def test_quality_no_source_files(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("hello")
    result = assess_quality(tmp_path, ["data.txt", "notes.md"])
    assert result.score is None
    assert "reason" in result.details


def test_quality_unsupported_language_reports_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "main.go").write_text("package main\n")
    # Force the go toolchain to look unavailable.
    monkeypatch.setattr(quality.shutil, "which", lambda name: None)
    result = assess_quality(tmp_path, ["main.go"])
    assert result.score is None
    assert result.details["languages"]["go"]["reason"]


def test_quality_python_missing_ruff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.py").write_text(_CLEAN)
    monkeypatch.setattr(quality.shutil, "which", lambda name: None)
    result = assess_quality(tmp_path, ["a.py"])
    assert result.details["languages"]["python"]["reason"] == "ruff not on PATH"
