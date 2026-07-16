"""Verify per-language analyzers degrade honestly (null + reason, never a fake score).

These exercise the Go/Java/JS branches that otherwise need external toolchains,
by simulating tool presence/absence. The honesty rule under test: a missing or
unusable tool yields ``score=None`` with a recorded ``reason`` for that language.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.evaluator import quality, security


def _which_none(_name: str) -> None:
    return None


# --- quality: missing toolchains -> reason, no score ---------------------------


def test_quality_js_missing_npx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "app.ts").write_text("export const x = 1;\n")
    monkeypatch.setattr(quality.shutil, "which", _which_none)
    result = quality.assess_quality(tmp_path, ["app.ts"])
    assert result.score is None
    assert "npx" in result.details["languages"]["typescript"]["reason"]


def test_quality_java_missing_checkstyle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "A.java").write_text("class A {}\n")
    monkeypatch.setattr(quality.shutil, "which", _which_none)
    result = quality.assess_quality(tmp_path, ["A.java"])
    assert result.details["languages"]["java"]["reason"] == "checkstyle not on PATH"


# --- security: missing toolchains / preconditions -> reason, no score ----------


def test_security_js_no_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "app.js").write_text("console.log(1)\n")
    # npm "present", but no package.json in the workspace.
    monkeypatch.setattr(security.shutil, "which", lambda name: "/usr/bin/npm")
    result = security.assess_security(tmp_path, ["app.js"])
    assert result.score is None
    assert "package.json" in result.details["languages"]["javascript"]["reason"]


def test_security_go_missing_gosec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.go").write_text("package main\n")
    monkeypatch.setattr(security.shutil, "which", _which_none)
    result = security.assess_security(tmp_path, ["main.go"])
    assert result.details["languages"]["go"]["reason"] == "gosec not on PATH"


def test_security_java_missing_spotbugs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "A.java").write_text("class A {}\n")
    monkeypatch.setattr(security.shutil, "which", _which_none)
    result = security.assess_security(tmp_path, ["A.java"])
    assert result.details["languages"]["java"]["reason"] == "spotbugs not on PATH"


# --- mixed languages: analyzed languages still aggregate ------------------------


def test_mixed_languages_average_over_analyzed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python scans; Go has no tool -> overall reflects only the analyzed language."""
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "main.go").write_text("package main\n")

    real_which = security.shutil.which

    def which(name: str) -> str | None:
        return None if name == "gosec" else real_which(name)

    monkeypatch.setattr(security.shutil, "which", which)
    result = security.assess_security(tmp_path, ["a.py", "main.go"])
    langs = result.details["languages"]
    assert langs["go"]["score"] is None
    # Overall is not None as long as at least one language (python) was analyzed
    # when bandit is available; if bandit is absent both are None.
    if langs["python"]["score"] is not None:
        assert result.score == langs["python"]["score"]
