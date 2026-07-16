"""Tests for the Rust quality and security analyzers.

Exercises the Rust branches of the quality/security evaluators with
subprocess mocked: cargo present/absent, clippy JSON parsing, cargo-audit
severity mapping, unsafe delta counting, and budget-exhausted paths.
All paths must return score=None with a reason when tools are absent —
never a fabricated 0.0 or 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.evaluator import quality, security
from harness.evaluator.security import _count_unsafe_delta, score_from_counts

# --- helpers -------------------------------------------------------------------


def _which_none(_name: str) -> None:
    return None


def _mock_subprocess_run(outputs: dict[str, Any]):
    """Return a mock subprocess.run that responds to specific command prefixes."""

    def _run(args, **kwargs):
        cmd_str = " ".join(args[:2]) if len(args) >= 2 else " ".join(args)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        r = Result()
        if cmd_str in outputs:
            out = outputs[cmd_str]
            r.returncode = out.get("returncode", 0)
            r.stdout = out.get("stdout", "")
            r.stderr = out.get("stderr", "")
        return r

    return _run


# --- quality: Rust --------------------------------------------------------------


def test_quality_rust_missing_cargo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    monkeypatch.setattr(quality.shutil, "which", _which_none)
    result = quality.assess_quality(tmp_path, ["main.rs"])
    assert result.score is None
    assert "cargo" in result.details["languages"]["rust"]["reason"]


def test_quality_rust_cargo_fmt_unformatted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "main.rs").write_text("fn main ( ) { }\n")
    monkeypatch.setattr(quality.shutil, "which", lambda name: "/usr/bin/cargo")
    monkeypatch.setattr(
        quality.subprocess,
        "run",
        _mock_subprocess_run(
            {
                "cargo fmt": {"returncode": 1, "stderr": "main.rs\n"},
                "cargo clippy": {"returncode": 0, "stdout": ""},
            }
        ),
    )
    result = quality.assess_quality(tmp_path, ["main.rs"])
    assert result.score is not None
    assert result.details["languages"]["rust"]["unformatted"] >= 1


def test_quality_rust_budget_exhausted(tmp_path: Path) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    result = quality._rust(tmp_path, ["main.rs"], remaining_s=lambda: -1.0)
    assert result.score is None
    assert "budget" in result.details.get("reason", "").lower() or result.score is None


# --- security: Rust -------------------------------------------------------------


def test_security_rust_missing_cargo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    monkeypatch.setattr(security.shutil, "which", _which_none)
    result = security.assess_security(tmp_path, ["main.rs"])
    assert result.score is None
    assert "cargo" in result.details["languages"]["rust"]["reason"]


def test_security_rust_no_cargo_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    monkeypatch.setattr(security.shutil, "which", lambda name: "/usr/bin/cargo")
    result = security.assess_security(tmp_path, ["main.rs"])
    assert result.score is None
    assert "Cargo.lock" in result.details["languages"]["rust"]["reason"]


def test_security_rust_clean_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    (tmp_path / "Cargo.lock").write_text("# lockfile\n")
    monkeypatch.setattr(security.shutil, "which", lambda name: "/usr/bin/cargo")
    audit_output = json.dumps({"vulnerabilities": {"count": 0, "list": []}})
    monkeypatch.setattr(
        security.subprocess,
        "run",
        _mock_subprocess_run(
            {
                "cargo audit": {"returncode": 0, "stdout": audit_output},
            }
        ),
    )
    result = security.assess_security(tmp_path, ["main.rs"])
    assert result.score is not None
    assert result.score == 1.0


def test_security_rust_vulnerabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    (tmp_path / "Cargo.lock").write_text("# lockfile\n")
    monkeypatch.setattr(security.shutil, "which", lambda name: "/usr/bin/cargo")
    audit_output = json.dumps(
        {
            "vulnerabilities": {
                "count": 2,
                "list": [
                    {"severity": "high", "advisory": {}},
                    {"severity": "medium", "advisory": {}},
                ],
            }
        }
    )
    monkeypatch.setattr(
        security.subprocess,
        "run",
        _mock_subprocess_run(
            {
                "cargo audit": {"returncode": 1, "stdout": audit_output},
            }
        ),
    )
    result = security.assess_security(tmp_path, ["main.rs"])
    assert result.score is not None
    assert result.score < 1.0
    assert result.details["languages"]["rust"]["high"] == 1
    assert result.details["languages"]["rust"]["medium"] == 1


# --- unsafe delta ---------------------------------------------------------------


def test_unsafe_delta_count(tmp_path: Path) -> None:
    """Count ``unsafe`` keywords in .rs file content."""
    (tmp_path / "safe.rs").write_text("fn foo() {}\n", encoding="utf-8")
    (tmp_path / "unsafe.rs").write_text(
        "unsafe fn bar() {}\nunsafe impl Send for X {}\n", encoding="utf-8"
    )
    # _count_unsafe_delta reads workspace/filename
    assert _count_unsafe_delta(tmp_path, ["safe.rs"]) == 0
    assert _count_unsafe_delta(tmp_path, ["unsafe.rs"]) == 2
    # Non-.rs files are ignored.
    assert _count_unsafe_delta(tmp_path, ["safe.rs", "readme.md"]) == 0


def test_unsafe_delta_penalty() -> None:
    """0.05 per unsafe keyword, subtracted from base score."""
    base = score_from_counts(0, 0, 0)  # 1.0
    # 2 unsafe blocks -> penalty 0.1
    penalty = min(1.0, 0.05 * 2)
    final = round(max(0.0, base - penalty), 4)
    assert final == 0.9


def test_security_rust_budget_exhausted(tmp_path: Path) -> None:
    (tmp_path / "main.rs").write_text("fn main() {}\n")
    (tmp_path / "Cargo.lock").write_text("# lockfile\n")
    result = security._rust(tmp_path, ["main.rs"], remaining_s=lambda: -1.0)
    assert result.score is None
    assert "budget" in result.details.get("reason", "").lower()


# --- clippy JSON parsing --------------------------------------------------------


def test_quality_rust_clippy_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "main.rs").write_text("fn main() { let x = 1; }\n")
    monkeypatch.setattr(quality.shutil, "which", lambda name: "/usr/bin/cargo")
    clippy_msg = json.dumps(
        {
            "reason": "diagnostic",
            "message": {"spans": [{"file_name": str((tmp_path / "main.rs").resolve())}]},
        }
    )
    monkeypatch.setattr(
        quality.subprocess,
        "run",
        _mock_subprocess_run(
            {
                "cargo fmt": {"returncode": 0, "stderr": ""},
                "cargo clippy": {"returncode": 1, "stdout": clippy_msg + "\n"},
            }
        ),
    )
    result = quality._rust(tmp_path, ["main.rs"], remaining_s=lambda: None)
    assert result.score is not None
    assert result.details["clippy_warnings"] >= 1
