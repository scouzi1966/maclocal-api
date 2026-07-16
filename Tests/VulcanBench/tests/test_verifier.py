"""Tests for the declarative verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from harness.tasks import load_task, prepare_workspace
from harness.verifier import run_declarative_verifier

pytestmark = pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")


def _make_task(root: Path, f_returns: int, p2p_ok: bool = True) -> None:
    """A tiny Python task: m.f() should return 2; starting code returns f_returns."""
    task = root / "fix-f"
    (task / "repo").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    (task / "repo" / "m.py").write_text(f"def f():\n    return {f_returns}\n")
    # fail_to_pass: f() must equal 2
    (task / "tests" / "t_f2p.py").write_text(
        "from m import f\n\ndef test_two():\n    assert f() == 2\n"
    )
    # pass_to_pass: f() returns an int (or, when p2p_ok=False, a deliberately failing check)
    p2p_assert = "isinstance(f(), int)" if p2p_ok else "f() == 999"
    (task / "tests" / "t_p2p.py").write_text(
        f"from m import f\n\ndef test_int():\n    assert {p2p_assert}\n"
    )
    (task / "metadata.json").write_text(
        json.dumps(
            {
                "id": "fix-f",
                "category": "bug_fix",
                "languages": ["python"],
                "difficulty": "trivial",
                "created": "2026-05-30",
                "source": "hand-authored",
                "decontamination_notes": "fixture",
                "tests": {
                    "fail_to_pass": [{"name": "two", "cmd": "python -m pytest t_f2p.py -q"}],
                    "pass_to_pass": [{"name": "int", "cmd": "python -m pytest t_p2p.py -q"}],
                },
            }
        )
    )
    (task / "issue.md").write_text("make f return 2")


def _functional(root: Path, tmp: Path) -> float:
    task = load_task("fix-f", root)
    ws = prepare_workspace(task, tmp / "ws")
    payload = run_declarative_verifier(task, ws)
    return float(payload["scores"]["functional"])


def test_passes_when_correct(tmp_path: Path) -> None:
    _make_task(tmp_path / "tasks", f_returns=2)
    assert _functional(tmp_path / "tasks", tmp_path) == 1.0


def test_fails_when_buggy(tmp_path: Path) -> None:
    _make_task(tmp_path / "tasks", f_returns=1)
    assert _functional(tmp_path / "tasks", tmp_path) == 0.0


def test_pass_to_pass_regression_gates_to_zero(tmp_path: Path) -> None:
    # f is "correct" (returns 2) so fail_to_pass would pass, but pass_to_pass fails.
    _make_task(tmp_path / "tasks", f_returns=2, p2p_ok=False)
    task = load_task("fix-f", tmp_path / "tasks")
    ws = prepare_workspace(task, tmp_path / "ws")
    payload = run_declarative_verifier(task, ws)
    assert payload["scores"]["functional"] == 0.0
    assert payload["pass_to_pass_ok"] is False


def test_custom_runner_is_used(tmp_path: Path) -> None:
    """The verifier dispatches every test command through the provided runner."""
    _make_task(tmp_path / "tasks", f_returns=1)  # buggy code, but runner is faked
    task = load_task("fix-f", tmp_path / "tasks")
    ws = prepare_workspace(task, tmp_path / "ws")

    seen: list[str] = []

    def fake_runner(cmd: str, workspace: Path, timeout: int) -> int:
        seen.append(cmd)
        return 0  # pretend everything passes, regardless of the (buggy) code

    payload = run_declarative_verifier(task, ws, runner=fake_runner)
    assert payload["scores"]["functional"] == 1.0  # runner said pass
    assert any("t_f2p" in c for c in seen)
    assert any("t_p2p" in c for c in seen)


def test_hidden_tests_not_in_prepared_workspace(tmp_path: Path) -> None:
    """The agent's workspace must not contain hidden tests until verification."""
    _make_task(tmp_path / "tasks", f_returns=1)
    task = load_task("fix-f", tmp_path / "tasks")
    ws = prepare_workspace(task, tmp_path / "ws")
    assert (ws / "m.py").exists()  # repo copied
    assert not (ws / "t_f2p.py").exists()  # hidden test NOT present yet
