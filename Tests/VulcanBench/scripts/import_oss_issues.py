#!/usr/bin/env python3
"""Scaffold a VulcanBench task from a local repo + an issue.

This creates the *structure* of a task — it never fabricates provenance or
decontamination. After running it, a human must: trim ``repo/`` to a minimal
starting state, write the hidden ``tests/``, fill in the declarative
``tests`` block and honest ``decontamination_notes`` in ``metadata.json``, and
generate ``gold_patch.diff`` from a working fix. Then ``validate_tasks.py`` must
pass before the task is trusted.

Usage:
    python scripts/import_oss_issues.py \\
        --id <task-id> --repo <path-to-local-repo> \\
        [--issue <path-to-issue.md>] [--languages python,go] \\
        [--base-commit SHA] [--upstream-url URL] \\
        [--upstream-issue URL] [--repo-scale medium] \\
        [--output repo|snapshot] [--tasks-root tasks/v1]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

_IGNORE = shutil.ignore_patterns(
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "*.pyc",
    "dist",
    "build",
    ".tox",
    "vendor",
)

_METADATA_TEMPLATE: dict = {
    "id": "",
    "category": "bug_fix",
    "languages": [],
    "difficulty": "medium",
    "task_complexity": "system",
    "created": "",
    "source": "oss",
    "decontaminated": False,
    "decontamination_notes": (
        "TODO: record the upstream repo + commit + issue URL and verify the fix "
        "post-dates the evaluated models' training cutoffs."
    ),
    "repo_scale": "medium",
    "base_commit": "",
    "upstream": {"url": "", "issue": "", "pr": "", "fix_commit": ""},
    "test_timeout_s": 180,
    "agent_hints": {"suggested_max_steps": 100, "entry_paths": []},
    "tests": {
        "fail_to_pass": [
            {"name": "TODO", "cmd": "TODO: command that fails before the fix, passes after"}
        ],
        "pass_to_pass": [
            {"name": "TODO", "cmd": "TODO: command that passes before and after (regression guard)"}
        ],
    },
}


def _write_snapshot(repo: Path, dest: Path) -> None:
    archive = dest / "repo_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in repo.rglob("*"):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(repo).as_posix())


def scaffold(  # noqa: PLR0912
    task_id: str,
    repo: Path,
    issue: Path | None,
    languages: list[str],
    tasks_root: Path,
    *,
    base_commit: str,
    upstream_url: str,
    upstream_issue: str,
    upstream_pr: str,
    fix_commit: str,
    repo_scale: str,
    output_mode: str,
    exclude: list[str],
    paths: list[str],
) -> Path:
    dest = tasks_root / task_id
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    (dest / "tests").mkdir(parents=True)

    ignore = _IGNORE
    if exclude:
        ignore = shutil.ignore_patterns(
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "*.pyc",
            "dist",
            "build",
            ".tox",
            "vendor",
            *exclude,
        )

    staging = dest / "_staging_repo"
    if paths:
        staging.mkdir(parents=True)
        for raw_rel in paths:
            rel = raw_rel.strip().strip("/")
            src = repo / rel
            if not src.exists():
                raise FileNotFoundError(f"--paths entry not found: {rel}")
            tgt = staging / rel
            if src.is_dir():
                shutil.copytree(src, tgt, ignore=ignore, dirs_exist_ok=True)
            else:
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tgt)
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "NOTICE"):
            lic = repo / name
            if lic.is_file():
                shutil.copy2(lic, staging / name)
        repo_src = staging
    else:
        repo_src = repo

    if output_mode == "snapshot":
        tmp_repo = dest / "repo"
        shutil.copytree(repo_src, tmp_repo, ignore=ignore)
        _write_snapshot(tmp_repo, dest)
        shutil.rmtree(tmp_repo)
    else:
        shutil.copytree(repo_src, dest / "repo", ignore=ignore)
        if staging.exists():
            shutil.rmtree(staging)

    meta = dict(_METADATA_TEMPLATE)
    meta["id"] = task_id
    meta["languages"] = languages
    meta["created"] = datetime.now(UTC).strftime("%Y-%m-%d")
    meta["repo_scale"] = repo_scale
    meta["base_commit"] = base_commit
    meta["upstream"] = {
        "url": upstream_url,
        "issue": upstream_issue,
        "pr": upstream_pr,
        "fix_commit": fix_commit,
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    issue_text = (
        issue.read_text(encoding="utf-8")
        if issue
        else "TODO: describe the problem (no solution).\n"
    )
    (dest / "issue.md").write_text(issue_text, encoding="utf-8")

    (dest / "gold_patch.diff").write_text(
        "# TODO: git diff of the reference fix (applied to repo/)\n", encoding="utf-8"
    )
    (dest / "expected_metrics.json").write_text(
        json.dumps({"functional": 1.0}, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "calibration.json").write_text(
        json.dumps({"pass_at_1": None, "notes": "fill after reference model sweep"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scaffold a VulcanBench task from a repo + issue")
    p.add_argument("--id", required=True, help="task id, e.g. py-some-fix")
    p.add_argument(
        "--repo", required=True, type=Path, help="path to a local repo to seed repo/ from"
    )
    p.add_argument("--issue", type=Path, help="path to a markdown file with the issue text")
    p.add_argument("--languages", default="", help="comma-separated, e.g. python,go")
    p.add_argument("--tasks-root", type=Path, default=Path("tasks/v1"))
    p.add_argument("--base-commit", default="", help="upstream commit SHA for the slice")
    p.add_argument("--upstream-url", default="", help="https://github.com/org/repo")
    p.add_argument("--upstream-issue", default="", help="issue URL")
    p.add_argument("--upstream-pr", default="", help="PR URL")
    p.add_argument("--fix-commit", default="", help="commit that fixes the issue")
    p.add_argument(
        "--repo-scale",
        default="medium",
        choices=["micro", "small", "medium", "large"],
    )
    p.add_argument(
        "--output",
        default="repo",
        choices=["repo", "snapshot"],
        help="repo/ directory or repo_snapshot.tar.gz",
    )
    p.add_argument("--paths", default="", help="comma-separated paths to copy from --repo")
    p.add_argument("--exclude", default="", help="extra ignore dir names")
    args = p.parse_args(argv)

    if not args.repo.is_dir():
        print(f"error: --repo {args.repo} is not a directory", file=sys.stderr)
        return 2
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    paths = [s.strip() for s in args.paths.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
    try:
        dest = scaffold(
            args.id,
            args.repo,
            args.issue,
            languages,
            args.tasks_root,
            base_commit=args.base_commit,
            upstream_url=args.upstream_url,
            upstream_issue=args.upstream_issue,
            upstream_pr=args.upstream_pr,
            fix_commit=args.fix_commit,
            repo_scale=args.repo_scale,
            output_mode=args.output,
            exclude=exclude,
            paths=paths,
        )
    except (FileExistsError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"scaffolded {dest}")
    print("Next: trim repo/ to a minimal start state, write hidden tests/, fill in")
    print("metadata.json (tests + decontamination_notes), and generate gold_patch.diff,")
    print(f"then: python scripts/validate_tasks.py {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
