"""Tests for the task validator (harness.validate)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from harness import validate
from harness.tasks import Task


def _full_task(root: Path, task_id: str = "demo", gold: str = "") -> Path:
    """A schema-complete Python task; ``gold`` is the gold_patch.diff contents."""
    task = root / task_id
    (task / "repo").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    (task / "repo" / "m.py").write_text("def f():\n    return 1\n")
    (task / "tests" / "t.py").write_text(
        "from m import f\n\ndef test_two():\n    assert f() == 2\n"
    )
    (task / "metadata.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "category": "bug_fix",
                "languages": ["python"],
                "difficulty": "trivial",
                "created": "2026-05-30",
                "source": "hand-authored",
                "decontaminated": True,
                "decontamination_notes": "fixture",
                "tests": {
                    "fail_to_pass": [{"name": "two", "cmd": "python -m pytest t.py -q"}],
                    "pass_to_pass": [],
                },
            }
        )
    )
    (task / "issue.md").write_text("make f return 2")
    (task / "gold_patch.diff").write_text(gold)
    return task


def test_legacy_task_skipped(tmp_path: Path) -> None:
    # metadata only, no tests/gold -> legacy/demo skip.
    task = tmp_path / "legacy"
    task.mkdir()
    (task / "metadata.json").write_text(json.dumps({"id": "legacy", "category": "synthetic"}))
    res = validate.validate_task(task)
    assert res.status == validate.SKIP


def test_missing_toolchain_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _full_task(tmp_path)
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    res = validate.validate_task(task)
    assert res.status == validate.SKIP
    assert "toolchain" in res.reasons[0]


def test_schema_failure(tmp_path: Path) -> None:
    task = _full_task(tmp_path)
    # Remove a required field.
    meta = json.loads((task / "metadata.json").read_text())
    del meta["decontamination_notes"]
    (task / "metadata.json").write_text(json.dumps(meta))
    res = validate.validate_task(task)
    assert res.status == validate.FAIL


_NON_SOLVING_GOLD = (
    "diff --git a/m.py b/m.py\n"
    "--- a/m.py\n"
    "+++ b/m.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def f():\n"
    "-    return 1\n"
    "+    return 3\n"
)

_SOLVING_GOLD = (
    "diff --git a/m.py b/m.py\n"
    "--- a/m.py\n"
    "+++ b/m.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def f():\n"
    "-    return 1\n"
    "+    return 2\n"
)


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_gold_must_solve(tmp_path: Path) -> None:
    # Gold applies cleanly but f returns 3 (not 2) -> fail_to_pass still fails -> FAIL.
    task = _full_task(tmp_path, gold=_NON_SOLVING_GOLD)
    res = validate.validate_task(task)
    assert res.status == validate.FAIL
    assert "gold patch scored" in res.reasons[0]


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_gold_must_apply(tmp_path: Path) -> None:
    # A malformed/empty gold patch -> git apply fails -> FAIL.
    task = _full_task(tmp_path, gold="")
    res = validate.validate_task(task)
    assert res.status == validate.FAIL


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_real_python_tasks_pass() -> None:
    for task_id in ("py-ttl-cache-expiry", "py-csv-export-feature"):
        res = validate.validate_task(Path("tasks/v1") / task_id)
        assert res.status == validate.PASS, f"{task_id}: {res.reasons}"


def _oss_task(root: Path, *, notes: str, with_license: bool = True) -> Path:
    """A schema-complete task labeled source:oss / decontaminated:false."""
    task = _full_task(root, task_id="oss-demo")
    meta = json.loads((task / "metadata.json").read_text())
    meta["source"] = "oss"
    meta["decontaminated"] = False
    meta["decontamination_notes"] = notes
    meta["base_commit"] = "deadbeef1234567890deadbeef1234567890deadbeef"
    meta["upstream"] = {
        "url": "https://github.com/x/y",
        "issue": "https://github.com/x/y/issues/33",
        "pr": "",
        "fix_commit": "cafebabe1234567890cafebabe1234567890cafebabe",
    }
    (task / "metadata.json").write_text(json.dumps(meta))
    if with_license:
        (task / "repo" / "LICENSE").write_text("MIT License\n\nCopyright ...\n")
    return task


def test_missing_decontaminated_field_fails(tmp_path: Path) -> None:
    task = _full_task(tmp_path)
    meta = json.loads((task / "metadata.json").read_text())
    del meta["decontaminated"]
    (task / "metadata.json").write_text(json.dumps(meta))
    res = validate.validate_task(task)
    assert res.status == validate.FAIL
    assert "decontaminated" in res.reasons[0]


def test_handauthored_must_be_decontaminated_true(tmp_path: Path) -> None:
    task = _full_task(tmp_path)
    meta = json.loads((task / "metadata.json").read_text())
    meta["decontaminated"] = False
    (task / "metadata.json").write_text(json.dumps(meta))
    res = validate.validate_task(task)
    assert res.status == validate.FAIL
    assert "must be true" in res.reasons[0]


def test_oss_requires_source_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _oss_task(tmp_path, notes="from some lib, fixed in commit abc1234def")
    meta = json.loads((task / "metadata.json").read_text())
    meta["upstream"] = {"url": "", "issue": "", "pr": "", "fix_commit": ""}
    (task / "metadata.json").write_text(json.dumps(meta))
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    res = validate.validate_task(task)
    assert res.status == validate.FAIL
    assert "url" in res.reasons[0].lower()


def test_oss_requires_commit_or_issue_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _oss_task(tmp_path, notes="see https://example.com/somewhere for details")
    meta = json.loads((task / "metadata.json").read_text())
    meta["upstream"] = {
        "url": "https://example.com/somewhere",
        "issue": "",
        "pr": "",
        "fix_commit": "",
    }
    (task / "metadata.json").write_text(json.dumps(meta))
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    res = validate.validate_task(task)
    assert res.status == validate.FAIL
    assert "commit" in res.reasons[0].lower() or "issue" in res.reasons[0].lower()


def test_oss_requires_preserved_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _oss_task(
        tmp_path,
        notes="https://github.com/x/y/issues/33 fixed in commit deadbeef1234",
        with_license=False,
    )
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    res = validate.validate_task(task)
    assert res.status == validate.FAIL
    assert "license" in res.reasons[0].lower()


def test_oss_valid_provenance_passes_decontamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # URL + issue ref + commit hash + preserved LICENSE -> decontamination OK.
    # With no toolchain present it falls through to a SKIP, not a decontamination FAIL.
    task = _oss_task(
        tmp_path,
        notes="https://github.com/x/y/issues/33 fixed in commit deadbeef1234",
    )
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)
    res = validate.validate_task(task)
    assert res.status == validate.SKIP


def test_real_oss_task_passes() -> None:
    res = validate.validate_task(Path("tasks/v1") / "oss-inflection-titleize")
    # PASS where pytest is available; SKIP otherwise — never FAIL.
    assert res.status in (validate.PASS, validate.SKIP), res.reasons


def test_main_returns_zero_on_skips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task = tmp_path / "legacy"
    task.mkdir()
    (task / "metadata.json").write_text(json.dumps({"id": "legacy"}))
    assert validate.main([str(tmp_path)]) == 0


def test_main_docker_requires_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harness.sandbox.docker_executor._docker_available",
        lambda: False,
    )
    assert validate.main(["tasks/v1", "--sandbox", "docker"]) == 2


@pytest.mark.skipif(shutil.which("pytest") is None, reason="pytest not on PATH")
def test_validate_docker_ignores_missing_host_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker mode runs verifiers in the sandbox image, not on the host PATH."""
    from harness.agent.loop import _executor_runner
    from harness.verifier import host_runner

    task = _full_task(tmp_path, gold=_SOLVING_GOLD)

    class FakeExecutor:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def run_command(self, args: object) -> dict[str, int]:
            cmd = getattr(args, "cmd", "")
            timeout = getattr(args, "timeout", None) or 120
            return {"exit_code": host_runner(cmd, self.workspace, timeout)}

        def close(self) -> None:
            pass

    def fake_docker_runner(task: Task, workspace: Path, image: str | None) -> tuple[object, FakeExecutor]:
        ex = FakeExecutor(workspace)
        return _executor_runner(ex), ex

    monkeypatch.setattr(validate, "_docker_runner", fake_docker_runner)
    monkeypatch.setattr(validate.shutil, "which", lambda name: None)

    res = validate.validate_task(task, validate.ValidateOptions(sandbox="docker"))
    assert res.status == validate.PASS
    assert "(docker)" in res.reasons[0]
