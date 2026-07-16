"""Validate VulcanBench task definitions.

For each task, in fresh temp workspaces, this checks:

1. schema            — required metadata, declarative tests, repo + gold patch.
2. toolchain         — SKIP (not fail) when a language's tool is absent on the host
                       (``--sandbox local`` only; Docker mode uses the sandbox image).
3. gold solves       — apply gold_patch.diff -> functional == 1.0.
4. fail-to-pass real — without the patch -> functional < 1.0 (not pre-solved).
5. determinism       — the gold verifier scores identically across runs.
6. provenance        — source + created present; an explicit `decontaminated`
                       bool (hand-authored => true; oss => notes with a source
                       URL + commit/issue ref and a preserved LICENSE).

Legacy/demo tasks (no declarative tests and no gold patch) are skipped, not failed.
``main()`` returns 1 if any task FAILs (SKIPs do not fail the run).

Use ``--sandbox docker`` to run setup and verifiers inside the same container
environment as ``vulcanbench run`` (recommended before large benchmark spend).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.sandbox.docker_executor import DockerToolExecutor
from harness.sandbox.images import resolve_sandbox_image
from harness.task_metadata import resolve_verifier_timeout_s, validate_scale_fields
from harness.tasks import Task, _safe_extract, load_task, prepare_workspace, run_setup
from harness.verifier import DEFAULT_TIMEOUT, Runner, run_declarative_verifier

if TYPE_CHECKING:
    from harness.sandbox.docker_executor import DockerToolExecutor

REQUIRED_META = ["id", "category", "languages", "difficulty", "source", "decontamination_notes"]
LANG_TOOL = {
    "python": "pytest",
    "go": "go",
    "typescript": "node",
    "javascript": "node",
    "rust": "cargo",
}
DETERMINISM_RUNS = 3
# Files that count as preserving an upstream license in a vendored OSS task.
_LICENSE_NAMES = ("license", "licence", "copying", "notice")
# A commit hash, an issue/PR ref (#123 or .../issues/9), or the word commit/issue/pull.
_OSS_REF = re.compile(
    r"\b[0-9a-f]{7,40}\b|#\d+|/(issues|pull|commit)/|\b(commit|issue|pull)\b", re.I
)

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"


@dataclass
class Result:
    task_id: str
    status: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidateOptions:
    """Runtime options for :func:`validate_task`."""

    sandbox: str = "local"  # local | docker
    image: str | None = None


def _apply_gold(workspace: Path, gold_patch: Path) -> bool:
    proc = subprocess.run(
        ["git", "apply", str(gold_patch.resolve())],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _has_license(repo_dir: Path) -> bool:
    """True if the vendored repo preserves an upstream license/notice file."""
    for p in repo_dir.rglob("*"):
        if p.is_file() and any(name in p.name.lower() for name in _LICENSE_NAMES):
            return True
    return False


def _check_decontamination(task: Task) -> str | None:  # noqa: PLR0911
    """Return a failure reason if the task's provenance labeling is dishonest/incomplete.

    ``decontaminated`` must be an explicit boolean. Hand-authored tasks are
    decontaminated by construction (written now), so they must assert ``true``.
    An ``oss`` task may honestly be ``false``, but it must then *prove* its
    provenance: notes naming a source URL + a commit/issue ref, and a preserved
    upstream license in ``repo/``.
    """
    decon = task.metadata.get("decontaminated")
    if not isinstance(decon, bool):
        return "metadata.decontaminated must be present and a boolean (true/false)"
    source = task.metadata.get("source")
    notes = str(task.metadata.get("decontamination_notes") or "")
    if source == "hand-authored" and decon is not True:
        return "hand-authored tasks are written now; metadata.decontaminated must be true"
    if source == "oss":
        upstream = task.metadata.get("upstream")
        upstream_notes = ""
        if isinstance(upstream, dict):
            upstream_notes = " ".join(
                str(upstream.get(k) or "") for k in ("url", "issue", "pr", "fix_commit")
            )
        provenance_text = f"{notes} {upstream_notes}"
        if "http" not in provenance_text:
            return (
                "oss task must include upstream URL in decontamination_notes or metadata.upstream"
            )
        if not _OSS_REF.search(provenance_text):
            return "oss task must reference upstream commit/issue/PR in notes or metadata.upstream"
        if not task.metadata.get("base_commit"):
            return "oss tasks require metadata.base_commit"
        upstream = task.metadata.get("upstream")
        if not isinstance(upstream, dict) or not upstream.get("url"):
            return "oss tasks require metadata.upstream.url"
        repo = task.repo_dir
        if repo is None and task.snapshot is not None:
            with tempfile.TemporaryDirectory() as td:
                extract_root = Path(td)
                with tarfile.open(task.snapshot, "r:gz") as tar:
                    _safe_extract(tar, extract_root)
                repo = extract_root
        if repo is not None and not _has_license(repo):
            return "oss task must preserve the upstream LICENSE/NOTICE file"
    return None


def _functional(task: Task, workspace: Path, runner: Runner | None = None) -> float:
    timeout = resolve_verifier_timeout_s(task.metadata, DEFAULT_TIMEOUT)
    payload = run_declarative_verifier(task, workspace, runner=runner, timeout=timeout)
    return float(payload.get("scores", {}).get("functional", 0.0))


def _docker_runner(task: Task, workspace: Path, image: str | None) -> tuple[Runner, Any]:
    """Open a Docker sandbox for ``workspace`` and return a verifier runner + executor."""
    from harness.agent.loop import _executor_runner  # noqa: PLC0415 — shared with agent loop

    resolved = resolve_sandbox_image(task, image)
    executor: DockerToolExecutor = DockerToolExecutor(workspace, image=resolved)
    return _executor_runner(executor), executor


def _fresh(
    task: Task,
    tmp: Path,
    name: str,
    apply_gold: bool,
    opts: ValidateOptions,
) -> float:
    ws = tmp / name
    prepare_workspace(task, ws)
    executor: Any | None = None
    runner: Runner | None = None
    if opts.sandbox == "docker":
        runner, executor = _docker_runner(task, ws, opts.image)
    try:
        if task.setup_commands:
            run_setup(task, ws, runner=runner)
        if apply_gold:
            assert task.gold_patch is not None
            if not _apply_gold(ws, task.gold_patch):
                raise RuntimeError("gold patch did not apply cleanly (git apply failed)")
        return _functional(task, ws, runner=runner)
    finally:
        if executor is not None:
            executor.close()


def validate_task(task_root: Path, opts: ValidateOptions | None = None) -> Result:  # noqa: PLR0911, PLR0912
    """Validate a single task directory and return a :class:`Result`."""
    opts = opts or ValidateOptions()
    task_id = task_root.name
    task = load_task(task_id, task_root.parent)
    r = Result(task_id=task_id, status=PASS)

    # Legacy/demo tasks (no declarative tests and no gold patch) are not validated.
    if task.tests_spec is None and task.gold_patch is None:
        return Result(task_id, SKIP, ["legacy/demo task (no declarative tests)"])

    # 1. schema
    missing = [k for k in REQUIRED_META if not task.metadata.get(k)]
    if missing:
        return Result(task_id, FAIL, [f"metadata missing/empty: {missing}"])
    spec = task.tests_spec
    if not spec or not spec.get("fail_to_pass"):
        return Result(task_id, FAIL, ["metadata.tests.fail_to_pass is empty"])
    if task.repo_dir is None and task.snapshot is None:
        return Result(task_id, FAIL, ["no repo/ directory or repo_snapshot.tar.gz"])
    if task.gold_patch is None:
        return Result(task_id, FAIL, ["no gold_patch.diff"])
    if not task.metadata.get("created"):
        return Result(task_id, FAIL, ["metadata.created is required (contamination audit)"])

    # 1b. decontamination honesty — provenance labeling must be explicit and provable.
    decon_reason = _check_decontamination(task)
    if decon_reason is not None:
        return Result(task_id, FAIL, [decon_reason])

    scale_reasons = validate_scale_fields(task_root, task.metadata)
    if scale_reasons:
        return Result(task_id, FAIL, scale_reasons)

    # 2. toolchain — skip (do not fail) when a host language tool is absent (local mode only).
    if opts.sandbox == "local":
        langs = task.metadata.get("languages", [])
        absent = sorted(
            {
                LANG_TOOL[lng]
                for lng in langs
                if lng in LANG_TOOL and shutil.which(LANG_TOOL[lng]) is None
            }
        )
        if absent:
            return Result(task_id, SKIP, [f"toolchain not installed: {absent}"])

    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 3. gold solves
            gold = _fresh(task, tmp, "gold", apply_gold=True, opts=opts)
            if gold != 1.0:
                return Result(task_id, FAIL, [f"gold patch scored functional={gold}, expected 1.0"])
            # 4. fail-to-pass is real (not already solved)
            base = _fresh(task, tmp, "base", apply_gold=False, opts=opts)
            if base >= 1.0:
                return Result(task_id, FAIL, [f"task already solved pre-patch (functional={base})"])
            # 5. determinism
            scores = {gold}
            for i in range(DETERMINISM_RUNS - 1):
                scores.add(_fresh(task, tmp, f"det{i}", apply_gold=True, opts=opts))
            if len(scores) != 1:
                return Result(task_id, FAIL, [f"nondeterministic gold scores: {sorted(scores)}"])
            mode = "docker" if opts.sandbox == "docker" else "local"
            r.reasons.append(
                f"gold=1.0, pre-patch={base}, deterministic over {DETERMINISM_RUNS} runs "
                f"({mode})"
            )
    except (RuntimeError, AssertionError) as e:
        return Result(task_id, FAIL, [str(e)])

    return r


def iter_task_roots(target: Path) -> list[Path]:
    if (target / "metadata.json").exists():
        return [target]
    return [d for d in sorted(target.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]


def _filter_roots_by_scale(roots: list[Path], tier: str, tasks_root: Path) -> list[Path]:
    """Keep only tasks listed in ``suite.json`` micro/large tier."""
    manifest = tasks_root / "suite.json"
    if not manifest.exists():
        return roots
    data = json.loads(manifest.read_text(encoding="utf-8"))
    allowed = set(data.get(tier) or [])
    if not allowed:
        return roots
    return [r for r in roots if r.name in allowed]


def _docker_preflight(image: str | None) -> int | None:
    """Return an exit code when Docker mode cannot run, else ``None``."""
    from harness.sandbox.docker_executor import _docker_available  # noqa: PLC0415

    if not _docker_available():
        print("error: --sandbox docker requires a running Docker daemon", file=sys.stderr)
        return 2
    if image:
        print(f"Validating in Docker sandbox ({image}) ...")
    else:
        print("Validating in Docker sandbox (per-task image selection) ...")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate VulcanBench tasks")
    parser.add_argument("target", nargs="?", default="tasks/v1", help="tasks dir or one task dir")
    parser.add_argument(
        "--filter-scale",
        choices=("micro", "large", "all"),
        default="all",
        help="validate only tasks in suite.json micro/large tier (default: all)",
    )
    parser.add_argument(
        "--sandbox",
        choices=("local", "docker"),
        default="local",
        help="where to run setup and verifiers (docker matches vulcanbench run)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Docker sandbox image when --sandbox docker (default: vulcanbench/sandbox:base)",
    )
    args = parser.parse_args(argv)

    if args.sandbox == "docker":
        preflight = _docker_preflight(args.image)
        if preflight is not None:
            return preflight

    target = Path(args.target)
    if not target.exists():
        print(f"error: {target} does not exist", file=sys.stderr)
        return 2

    opts = ValidateOptions(sandbox=args.sandbox, image=args.image)

    roots = iter_task_roots(target)
    if args.filter_scale != "all":
        suite_root = target
        if not (suite_root / "suite.json").exists() and target.parent.name == "v1":
            suite_root = target.parent
        roots = _filter_roots_by_scale(roots, args.filter_scale, suite_root)
    if not roots:
        print(f"no tasks found under {target}")
        return 0

    results = [validate_task(r, opts) for r in roots]
    icon = {PASS: "✓", SKIP: "○", FAIL: "✗"}
    for res in results:
        suffix = f" — {'; '.join(res.reasons)}" if res.reasons else ""
        print(f"  {icon[res.status]} {res.status:4} {res.task_id}{suffix}")

    n_pass = sum(r.status == PASS for r in results)
    n_skip = sum(r.status == SKIP for r in results)
    n_fail = sum(r.status == FAIL for r in results)
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0
