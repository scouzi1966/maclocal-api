from __future__ import annotations

import contextlib
import difflib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from harness.agent.protocol import (
    EditFileArgs,
    ListFilesArgs,
    ReadFileArgs,
    RunCommandArgs,
    SearchCodeArgs,
    ToolProtocol,
)
from harness.agent.test_commands import default_test_command
from harness.evaluator.langs import detect_language
from harness.evaluator.security import assess_security
from harness.task_metadata import LIST_FILES_CAP, SEARCH_CODE_CAP
from harness.tasks import _should_ignore, vulcanbenchignore_paths

SearchResult = list[dict[str, Any]] | dict[str, Any]


def _cap_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]] | dict[str, Any]:
    if len(results) <= SEARCH_CODE_CAP:
        return results
    return {
        "matches": results[:SEARCH_CODE_CAP],
        "truncated": True,
        "total": len(results),
        "message": f"search capped at {SEARCH_CODE_CAP} matches; refine query or glob",
    }


class LocalToolExecutor(ToolProtocol):
    def __init__(self, workspace: Path | str = ".") -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve(self, p: str) -> Path:
        # Use is_relative_to, not str.startswith: the latter would let a sibling
        # like /ws-evil slip past a /ws workspace.
        path = (self.workspace / p).resolve()
        if not path.is_relative_to(self.workspace):
            raise PermissionError("path escapes workspace")
        return path

    def list_files(self, args: ListFilesArgs) -> list[str] | dict[str, Any]:
        root = self._resolve(args.dir)
        if not root.exists():
            return []
        ignored = vulcanbenchignore_paths(self.workspace)
        if args.recursive:
            paths = [
                p.relative_to(self.workspace)
                for p in root.rglob("*")
                if p.is_file() and not _should_ignore(p.relative_to(self.workspace), ignored)
            ]
        else:
            paths = [
                p.relative_to(self.workspace)
                for p in root.iterdir()
                if p.is_file() and not _should_ignore(p.relative_to(self.workspace), ignored)
            ]
        rels = sorted(str(p) for p in paths)
        if len(rels) <= LIST_FILES_CAP:
            return rels
        return {
            "files": rels[:LIST_FILES_CAP],
            "truncated": True,
            "total": len(rels),
            "message": f"listing capped at {LIST_FILES_CAP} files; use search_code or narrower dir",
        }

    def read_file(self, args: ReadFileArgs) -> str:
        path = self._resolve(args.path)
        text = path.read_text(encoding="utf-8")
        if args.start_line is not None:
            lines = text.splitlines(keepends=True)
            start = max(0, args.start_line - 1)
            end = start + (args.limit or len(lines))
            text = "".join(lines[start:end])
        return text

    def _ignored_rel_paths(self) -> set[Path]:
        return vulcanbenchignore_paths(self.workspace)

    def _filter_search_match(self, rel: Path, ignored: set[Path]) -> bool:
        return not _should_ignore(rel, ignored)

    def search_code(self, args: SearchCodeArgs) -> SearchResult:  # noqa: PLR0912
        ignored = self._ignored_rel_paths()
        if shutil.which("rg"):
            cmd = ["rg", "--json", "-n", args.query, str(self.workspace)]
            if args.glob:
                cmd += ["--glob", args.glob]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout or 30,
                check=False,
            )
            results: list[dict[str, Any]] = []
            for line in proc.stdout.splitlines():
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if obj.get("type") != "match":
                            continue
                        path_text = obj.get("data", {}).get("path", {}).get("text", "")
                        if not path_text:
                            continue
                        rel = Path(path_text)
                        if rel.is_absolute():
                            with contextlib.suppress(ValueError):
                                rel = rel.relative_to(self.workspace)
                        if not self._filter_search_match(rel, ignored):
                            continue
                        results.append(obj)
                    except Exception:
                        pass
            return _cap_search_results(results)
        matches = []
        for p in self.workspace.rglob("*.py"):
            rel = p.relative_to(self.workspace)
            if not self._filter_search_match(rel, ignored):
                continue
            try:
                if args.query.lower() in p.read_text(encoding="utf-8").lower():
                    matches.append({"path": str(rel)})
            except Exception:
                pass
        return _cap_search_results(matches)

    def edit_file(self, args: EditFileArgs) -> str:
        path = self._resolve(args.path)
        if not path.exists() and args.old_string == "":
            original = ""
            new_text = args.new_string
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_text, encoding="utf-8")
            return f"+ {args.new_string}"
        original = path.read_text(encoding="utf-8")
        if args.old_string not in original:
            raise ValueError("old_string not found exactly")
        new_text = original.replace(args.old_string, args.new_string, 1)
        path.write_text(new_text, encoding="utf-8")
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
        return "".join(diff)

    def run_command(self, args: RunCommandArgs) -> dict[str, Any]:
        cwd = self._resolve(args.cwd) if args.cwd else self.workspace
        proc = subprocess.run(
            args.cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=args.timeout or 120,
            check=False,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }

    def run_tests(self) -> dict[str, Any]:
        cmd = default_test_command(self.workspace)
        return self.run_command(RunCommandArgs(cmd=cmd))

    def run_lint(self) -> dict[str, Any]:
        return self.run_command(RunCommandArgs(cmd="ruff check . || true"))

    def run_build(self) -> dict[str, Any]:
        return {"ok": True}

    def security_scan(self, timeout_s: float | None = None) -> dict[str, Any]:
        # Reuse the evaluator's scanner over all recognized source files so the
        # agent's security_scan tool returns real findings.
        files = [
            str(p.relative_to(self.workspace))
            for p in self.workspace.rglob("*")
            if p.is_file() and detect_language(p.name) is not None
        ]
        remaining_s = (lambda: timeout_s) if timeout_s is not None else None
        result = assess_security(self.workspace, files, remaining_s=remaining_s)
        return {"score": result.score, **result.details}
