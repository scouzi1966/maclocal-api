"""Tests for the local tool executor."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.agent.local_executor import LocalToolExecutor
from harness.agent.protocol import (
    EditFileArgs,
    ListFilesArgs,
    ReadFileArgs,
    RunCommandArgs,
    SearchCodeArgs,
    ToolCall,
)


def _ws(tmp_path: Path) -> LocalToolExecutor:
    return LocalToolExecutor(tmp_path)


def test_edit_creates_then_reads(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ex.edit_file(EditFileArgs(path="a.txt", old_string="", new_string="hello\n"))
    assert ex.read_file(ReadFileArgs(path="a.txt")) == "hello\n"


def test_edit_replaces_substring(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ex.edit_file(EditFileArgs(path="a.txt", old_string="", new_string="foo bar\n"))
    diff = ex.edit_file(EditFileArgs(path="a.txt", old_string="foo", new_string="baz"))
    assert ex.read_file(ReadFileArgs(path="a.txt")) == "baz bar\n"
    assert "baz" in diff


def test_edit_missing_old_string_raises(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ex.edit_file(EditFileArgs(path="a.txt", old_string="", new_string="x\n"))
    with pytest.raises(ValueError, match="old_string not found"):
        ex.edit_file(EditFileArgs(path="a.txt", old_string="nope", new_string="y"))


def test_read_file_line_range(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ex.edit_file(EditFileArgs(path="a.txt", old_string="", new_string="l1\nl2\nl3\nl4\n"))
    out = ex.read_file(ReadFileArgs(path="a.txt", start_line=2, limit=2))
    assert out == "l2\nl3\n"


def test_list_files(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ex.edit_file(EditFileArgs(path="a.txt", old_string="", new_string="x"))
    ex.edit_file(EditFileArgs(path="sub/b.txt", old_string="", new_string="y"))
    top = ex.list_files(ListFilesArgs(dir="."))
    assert "a.txt" in top
    recursive = ex.list_files(ListFilesArgs(dir=".", recursive=True))
    assert "sub/b.txt" in recursive


def test_list_files_missing_dir(tmp_path: Path) -> None:
    assert _ws(tmp_path).list_files(ListFilesArgs(dir="nope")) == []


def test_path_escape_blocked(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    with pytest.raises(PermissionError):
        ex.read_file(ReadFileArgs(path="../../etc/passwd"))


def test_sibling_prefix_not_treated_as_inside(tmp_path: Path) -> None:
    # A sibling dir whose name shares the workspace's prefix (ws / ws-evil)
    # must NOT be considered inside the workspace.
    ex = LocalToolExecutor(tmp_path / "ws")
    (tmp_path / "ws-evil").mkdir()
    (tmp_path / "ws-evil" / "secret.txt").write_text("nope")
    with pytest.raises(PermissionError):
        ex.read_file(ReadFileArgs(path="../ws-evil/secret.txt"))


def test_search_code_finds_match(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ex.edit_file(EditFileArgs(path="m.py", old_string="", new_string="def needle():\n    pass\n"))
    results = ex.search_code(SearchCodeArgs(query="needle"))
    assert results  # found via rg or python fallback


def test_run_command_and_builtins(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    out = ex.run_command(RunCommandArgs(cmd="echo hi"))
    assert out["exit_code"] == 0
    assert "hi" in out["stdout"]
    assert ex.run_build()["ok"] is True
    assert "score" in ex.security_scan()  # delegates to the evaluator's scanner
    assert "exit_code" in ex.run_lint()


def test_security_scan_flags_real_issue(tmp_path: Path) -> None:
    if shutil.which("bandit") is None:
        pytest.skip("bandit not installed")
    ex = _ws(tmp_path)
    ex.edit_file(
        EditFileArgs(
            path="danger.py",
            old_string="",
            new_string="import subprocess\nsubprocess.call(c, shell=True)\n",
        )
    )
    result = ex.security_scan()
    assert result["score"] is not None and result["score"] < 1.0


def test_execute_dispatch_and_error(tmp_path: Path) -> None:
    ex = _ws(tmp_path)
    ok = ex.execute(ToolCall(tool="run_build"))
    assert ok.error is None
    bad = ex.execute(ToolCall(tool="read_file", args={"path": "missing.txt"}))
    assert bad.error is not None
