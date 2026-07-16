"""Tool output caps for large workspaces."""

from __future__ import annotations

from pathlib import Path

from harness.agent.local_executor import LocalToolExecutor
from harness.agent.protocol import ListFilesArgs, SearchCodeArgs
from harness.task_metadata import LIST_FILES_CAP


def test_list_files_truncates_recursive(tmp_path: Path) -> None:
    for i in range(LIST_FILES_CAP + 10):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    ex = LocalToolExecutor(tmp_path)
    out = ex.list_files(ListFilesArgs(dir=".", recursive=True))
    assert isinstance(out, dict)
    assert out.get("truncated") is True
    assert out.get("total") == LIST_FILES_CAP + 10


def test_search_code_respects_vulcanbenchignore(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("needle here", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "skip.py").write_text("needle here", encoding="utf-8")
    (tmp_path / ".vulcanbenchignore").write_text("dist\n", encoding="utf-8")
    ex = LocalToolExecutor(tmp_path)
    out = ex.search_code(SearchCodeArgs(query="needle"))

    def _match_path(m: dict) -> str:
        return m.get("data", {}).get("path", {}).get("text", m.get("path", ""))

    if isinstance(out, dict):
        paths = [_match_path(m) for m in out.get("matches", [])]
    else:
        paths = [_match_path(m) for m in out]
    assert any("keep.py" in p for p in paths)
    assert not any("dist" in p for p in paths)
