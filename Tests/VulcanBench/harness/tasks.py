"""Task loading and workspace preparation.

A task lives at ``tasks/v1/<id>/`` and contains at minimum:

- ``metadata.json``  — id, category, languages, difficulty, source, and a
  declarative ``tests`` block (``fail_to_pass`` / ``pass_to_pass``).
- ``issue.md``       — the natural-language problem statement given to the agent.

Starting state, supplied as either:

- ``repo/``                 — plain files (preferred for hand-authored tasks; git-reviewable), or
- ``repo_snapshot.tar.gz``  — a tarball (for large imported repos).

And, used only at verification time (never shown to the agent):

- ``tests/``          — HIDDEN tests, copied into the workspace by the verifier.
- ``gold_patch.diff`` — the reference solution (for the validator, not the run).
- ``verifier.py``     — legacy per-task verifier (declarative ``tests`` preferred).
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TASKS_ROOT = Path("tasks/v1")


@dataclass
class Task:
    task_id: str
    root: Path
    metadata: dict[str, Any]
    issue: str
    verifier: Path | None

    @property
    def snapshot(self) -> Path | None:
        snap = self.root / "repo_snapshot.tar.gz"
        return snap if snap.exists() else None

    @property
    def repo_dir(self) -> Path | None:
        """Plain starting-repo directory, if the task uses one."""
        d = self.root / "repo"
        return d if d.is_dir() else None

    @property
    def hidden_tests_dir(self) -> Path | None:
        """Hidden tests, added to the workspace only at verification time."""
        d = self.root / "tests"
        return d if d.is_dir() else None

    @property
    def tests_spec(self) -> dict[str, Any] | None:
        """Declarative test spec (``fail_to_pass``/``pass_to_pass``), if present."""
        spec = self.metadata.get("tests")
        return spec if isinstance(spec, dict) else None

    @property
    def gold_patch(self) -> Path | None:
        p = self.root / "gold_patch.diff"
        return p if p.exists() else None

    @property
    def setup_commands(self) -> list[dict[str, str]]:
        """Optional setup commands (``metadata.setup``): ``[{"name": str, "cmd": str}]``.

        These run through the same Runner abstraction before the agent starts
        (e.g. ``cargo build --tests`` warm-up for Rust tasks).
        """
        setup = self.metadata.get("setup")
        if isinstance(setup, list) and all(
            isinstance(s, dict) and "name" in s and "cmd" in s for s in setup
        ):
            return setup
        return []

    @property
    def setup_timeout_s(self) -> int:
        """Per-setup-command timeout from ``metadata.setup_timeout_s`` (default 600)."""
        raw = self.metadata.get("setup_timeout_s")
        if isinstance(raw, (int, float)) and raw > 0:
            return int(raw)
        return 600


def load_task(task_id: str, tasks_root: Path = DEFAULT_TASKS_ROOT) -> Task:
    """Load a task definition from disk."""
    root = tasks_root / task_id
    if not root.is_dir():
        raise FileNotFoundError(f"task {task_id!r} not found under {tasks_root}")
    metadata: dict[str, Any] = {}
    meta_path = root / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    issue_path = root / "issue.md"
    issue = issue_path.read_text(encoding="utf-8") if issue_path.exists() else ""
    verifier = root / "verifier.py"
    return Task(
        task_id=task_id,
        root=root,
        metadata=metadata,
        issue=issue,
        verifier=verifier if verifier.exists() else None,
    )


def list_task_ids(tasks_root: Path = DEFAULT_TASKS_ROOT) -> list[str]:
    if not tasks_root.exists():
        return []
    return sorted(d.name for d in tasks_root.iterdir() if d.is_dir())


def _hash_dir(h: hashlib._Hash, label: str, root: Path) -> None:
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        h.update(label.encode())
        h.update(b"\0")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")


def task_hash(task: Task) -> str:
    """A deterministic sha256 of a task's *scoring-relevant* definition.

    Covers everything that determines what a run is asked to do and scored
    against: the starting repo (or snapshot), the issue/prompt (``issue.md``),
    the hidden tests, the declarative ``tests`` spec, the legacy ``verifier.py``
    (if any), and the gold patch. Cosmetic metadata (id, created, source,
    decontamination_notes, difficulty, task_complexity) is intentionally
    excluded, so editing a note does not register as task drift, while changing the prompt, any
    test/source file, or the scoring logic does.
    """
    h = hashlib.sha256()
    if task.repo_dir is not None:
        _hash_dir(h, "repo", task.repo_dir)
    elif task.snapshot is not None:
        h.update(b"snapshot\0")
        h.update(task.snapshot.read_bytes())
        h.update(b"\0")
    h.update(b"issue\0")
    h.update(task.issue.encode())
    if task.hidden_tests_dir is not None:
        _hash_dir(h, "tests", task.hidden_tests_dir)
    h.update(b"\0tests_spec\0")
    h.update(json.dumps(task.tests_spec or {}, sort_keys=True).encode())
    h.update(b"\0verifier\0")
    if task.verifier is not None:
        h.update(task.verifier.read_bytes())
    h.update(b"\0gold\0")
    if task.gold_patch is not None:
        h.update(task.gold_patch.read_bytes())
    return h.hexdigest()


def vulcanbenchignore_paths(workspace: Path) -> set[Path]:
    """Paths (relative to workspace) excluded from list/search per ``.vulcanbenchignore``."""
    ignore_file = workspace / ".vulcanbenchignore"
    if not ignore_file.is_file():
        return set()
    ignored: set[Path] = set()
    for raw_line in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ignored.add(Path(line))
    return ignored


def _should_ignore(rel: Path, ignored: set[Path]) -> bool:
    for pattern in ignored:
        if rel == pattern or rel.as_posix().startswith(pattern.as_posix() + "/"):
            return True
        if pattern.name == pattern and rel.parts and rel.parts[0] == pattern.name:  # type: ignore[comparison-overlap]
            return True
    return False


def prepare_workspace(task: Task, workspace: Path) -> Path:
    """Materialize the task's starting state into ``workspace``.

    Copies the ``repo/`` directory (or extracts the snapshot), then writes
    ``issue.md``. The hidden ``tests/`` are deliberately NOT copied here — they
    are added by the verifier at scoring time so the agent never sees them.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    if task.repo_dir is not None:
        _copytree(task.repo_dir, workspace)
    elif task.snapshot is not None:
        with tarfile.open(task.snapshot, "r:gz") as tar:
            _safe_extract(tar, workspace)

    (workspace / "issue.md").write_text(task.issue, encoding="utf-8")

    ignore_src = task.root / ".vulcanbenchignore"
    if ignore_src.is_file():
        (workspace / ".vulcanbenchignore").write_bytes(ignore_src.read_bytes())

    return workspace


def install_hidden_tests(task: Task, workspace: Path) -> None:
    """Overlay a task's hidden ``tests/`` onto the workspace (verification only).

    Contents are merged at the workspace root preserving their relative paths, so
    a task places each test file where it must live: a Python test at
    ``tests/test_x.py`` lands at ``<ws>/test_x.py``; a Go test at
    ``tests/pkg/x_test.go`` lands inside the package at ``<ws>/pkg/x_test.go``.
    """
    if task.hidden_tests_dir is not None:
        _copytree(task.hidden_tests_dir, workspace)


def run_setup(
    task: Task,
    workspace: Path,
    runner: Callable[[str, Path, int], int] | None = None,
    timeout: int | None = None,
    collector: Any | None = None,
) -> list[dict[str, Any]]:
    """Run the task's optional ``setup`` commands before the agent starts.

    Uses the same ``Runner`` abstraction as the verifier ``(cmd, workspace,
    timeout) -> exit_code``. When ``runner`` is ``None`` a host subprocess is
    used. Each command is time-boxed by ``timeout`` (default from
    ``task.setup_timeout_s``). Results are recorded as ``"setup"`` events on
    the collector (if provided). A non-zero exit aborts subsequent setup
    commands and raises ``RuntimeError``.
    """
    from harness.verifier import host_runner  # noqa: PLC0415 — avoid circular import

    runner = runner or host_runner
    timeout = timeout or task.setup_timeout_s
    results: list[dict[str, Any]] = []

    for entry in task.setup_commands:
        name = entry["name"]
        cmd = entry["cmd"]
        started = time.monotonic()
        exit_code = runner(cmd, workspace, timeout)
        duration = round(time.monotonic() - started, 3)
        result = {"name": name, "cmd": cmd, "exit_code": exit_code, "duration_s": duration}
        results.append(result)
        if collector is not None:
            collector.record("setup", result)
        if exit_code != 0:
            raise RuntimeError(f"setup command {name!r} failed (exit code {exit_code}): {cmd}")

    return results


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract a tarball, refusing members that escape ``dest`` (path traversal)."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"unsafe path in snapshot: {member.name}")
    tar.extractall(dest)


def _copytree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
