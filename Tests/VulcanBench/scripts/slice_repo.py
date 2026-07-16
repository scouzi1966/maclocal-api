#!/usr/bin/env python3
"""Slice a git repository at a pinned commit into a minimal directory or tarball.

Produces a reproducible workspace slice plus ``SLICE_MANIFEST.json`` (file list, LOC,
license files).

Usage::

    python scripts/slice_repo.py \\
        --url https://github.com/org/repo.git \\
        --commit abc123 \\
        --paths src/pkg,tests \\
        --output /tmp/slice

    python scripts/slice_repo.py --repo /path/to/local.git --commit HEAD --output ./out

Cargo-aware pruning (for Rust workspace repos)::

    python scripts/slice_repo.py \\
        --url https://github.com/org/rust-repo.git \\
        --commit abc123 \\
        --cargo-prune --crates my-crate,my-other-crate \\
        --output /tmp/slice

When ``--cargo-prune`` is set, after the path-based slice the script computes
the transitive workspace-member dependency closure of the target crates (via
``cargo metadata``), drops non-member crates, and rewrites the root
``Cargo.toml`` ``[workspace] members`` array so the slice still compiles.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.task_metadata import measure_repo_path

_DEFAULT_EXCLUDE = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "vendor",
)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")


def _checkout_repo(url: str | None, repo: Path | None, commit: str, work: Path) -> Path:
    if repo is not None:
        if not (repo / ".git").exists():
            raise ValueError(f"--repo {repo} is not a git directory")
        shutil.copytree(repo, work, dirs_exist_ok=True, symlinks=True)
        _run(["git", "checkout", "--force", commit], cwd=work)
        return work
    if url is None:
        raise ValueError("either --url or --repo is required")
    _run(["git", "clone", "--filter=blob:none", url, str(work)])
    try:
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=work)
    except RuntimeError:
        _run(["git", "fetch", "origin", commit], cwd=work)
    _run(["git", "checkout", "--force", commit], cwd=work)
    return work


def _copy_paths(
    src: Path,
    dest: Path,
    paths: list[str],
    exclude: set[str],
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for raw_rel in paths:
        rel = raw_rel.strip().strip("/")
        if not rel:
            continue
        source = src / rel
        if not source.exists():
            raise FileNotFoundError(f"path not in repo: {rel}")
        target = dest / rel
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns(*exclude),
                dirs_exist_ok=True,
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    # Always preserve license files at repo root when present.
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "NOTICE"):
        lic = src / name
        if lic.is_file():
            shutil.copy2(lic, dest / name)


def _find_license_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in root.rglob("*"):
        if p.is_file() and "license" in p.name.lower():
            out.append(str(p.relative_to(root)))
    return sorted(out)


def _cargo_metadata(workspace: Path) -> dict[str, Any]:
    """Run ``cargo metadata --format-version 1 --no-deps`` and return parsed JSON."""
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cargo metadata failed: {proc.stderr}")
    return json.loads(proc.stdout)


def _workspace_dependency_closure(metadata: dict[str, Any], target_crates: list[str]) -> set[str]:
    """Compute the transitive workspace-member dependency closure of ``target_crates``.

    Returns a set of package names (workspace member names) that must be kept.
    """
    packages = metadata.get("packages", [])
    pkg_by_name: dict[str, dict[str, Any]] = {}
    for pkg in packages:
        name = pkg.get("name", "")
        pkg_by_name[name] = pkg

    # Build adjacency: pkg_name -> set of dependency names (workspace members only).
    workspace_names = {pkg.get("name", "") for pkg in packages}
    adj: dict[str, set[str]] = {name: set() for name in workspace_names}
    for pkg in packages:
        name = pkg.get("name", "")
        for dep in pkg.get("dependencies", []):
            dep_name = dep.get("name", "")
            if dep_name in workspace_names:
                adj[name].add(dep_name)

    # BFS from each target crate.
    closure: set[str] = set()
    queue = list(target_crates)
    while queue:
        current = queue.pop(0)
        if current in closure:
            continue
        closure.add(current)
        for dep in adj.get(current, set()):
            if dep not in closure:
                queue.append(dep)
    return closure


def _rewrite_workspace_members(cargo_toml: Path, kept_members: set[str]) -> None:  # noqa: PLR0912
    """Rewrite ``[workspace] members`` in the root Cargo.toml to only ``kept_members``.

    Uses a minimal targeted text edit to avoid reformatting the file. Reads with
    ``tomllib`` for understanding, writes with text substitution.
    """
    content = cargo_toml.read_text(encoding="utf-8")
    try:
        parsed = tomllib.loads(content)
    except Exception as e:
        raise RuntimeError(f"cannot parse {cargo_toml}: {e}") from e

    original_members = parsed.get("workspace", {}).get("members", [])
    if not original_members:
        return  # No workspace members to rewrite.

    kept_sorted = sorted(kept_members)

    # Build replacement: replace the members array in the text.
    # Find the [workspace] section and the members = [...] line(s).
    lines = content.splitlines(True)
    new_lines: list[str] = []
    in_workspace = False
    members_started = False
    members_done = False
    bracket_depth = 0

    for line in lines:
        stripped = line.strip()

        # Track workspace section.
        if stripped == "[workspace]":
            in_workspace = True
            new_lines.append(line)
            continue
        if stripped.startswith("[") and stripped != "[workspace]":
            in_workspace = False

        if members_done:
            new_lines.append(line)
            continue

        # Detect the start of the members array.
        if in_workspace and not members_started and stripped.startswith("members"):
            members_started = True
            bracket_depth += stripped.count("[") - stripped.count("]")
            if bracket_depth <= 0:
                # Single-line array like members = ["a", "b"]
                members_done = True
            # Emit our replacement regardless.
            if len(kept_sorted) == 1:
                members_line = f'members = ["{kept_sorted[0]}"]\n'
            else:
                members_line = "members = [\n"
                for m in kept_sorted:
                    members_line += f'  "{m}",\n'
                members_line += "]\n"
            new_lines.append(members_line)
            continue

        # Skip continuation of old members array until brackets balance.
        if members_started and not members_done:
            bracket_depth += stripped.count("[") - stripped.count("]")
            if bracket_depth <= 0:
                members_done = True
            continue  # Skip this line — it's part of the old array.

        new_lines.append(line)

    cargo_toml.write_text("".join(new_lines), encoding="utf-8")


def _cargo_prune(slice_root: Path, target_crates: list[str]) -> None:
    """Prune non-target workspace members from a Cargo workspace slice.

    Computes the transitive dependency closure of ``target_crates``, removes
    directories for unneeded workspace members, and rewrites the root
    ``Cargo.toml`` ``[workspace] members``.

    Raises ``RuntimeError`` if the pruned workspace fails ``cargo metadata``.
    """
    if not (slice_root / "Cargo.toml").exists():
        raise RuntimeError("cargo-prune requires a Cargo.toml at the repo root")

    metadata = _cargo_metadata(slice_root)
    packages = metadata.get("packages", [])
    workspace_members = {pkg.get("name", ""): pkg.get("manifest_path", "") for pkg in packages}

    closure = _workspace_dependency_closure(metadata, target_crates)

    # Determine which member directories to remove.
    to_remove: set[str] = set()
    for name, manifest_path in workspace_members.items():
        if name not in closure:
            # manifest_path is absolute; compute relative dir from workspace root.
            try:
                rel = (Path(manifest_path).parent).relative_to(slice_root.resolve())
            except ValueError:
                continue
            to_remove.add(str(rel))

    for rel_dir in to_remove:
        target = slice_root / rel_dir
        if target.is_dir():
            shutil.rmtree(target)

    # Rewrite the root Cargo.toml members.
    _rewrite_workspace_members(slice_root / "Cargo.toml", closure)

    # Verify the pruned workspace resolves.
    try:
        _cargo_metadata(slice_root)
    except RuntimeError as e:
        raise RuntimeError(
            f"pruned workspace fails cargo metadata — the slice cannot resolve. "
            f"Target crates: {target_crates}. Error: {e}"
        ) from e


def slice_repo(
    *,
    url: str | None,
    repo: Path | None,
    commit: str,
    paths: list[str],
    output: Path,
    exclude: list[str],
    tarball: bool,
    cargo_prune: bool = False,
    crates: list[str] | None = None,
) -> Path:
    exclude_set = set(_DEFAULT_EXCLUDE) | set(exclude)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "checkout"
        checkout = _checkout_repo(url, repo, commit, work)
        if paths:
            slice_root = output / "repo"
            if slice_root.exists():
                shutil.rmtree(slice_root)
            _copy_paths(checkout, slice_root, paths, exclude_set)
        else:
            slice_root = output / "repo"
            if slice_root.exists():
                shutil.rmtree(slice_root)
            shutil.copytree(
                checkout,
                slice_root,
                ignore=shutil.ignore_patterns(*exclude_set),
                dirs_exist_ok=True,
            )
            if (slice_root / ".git").exists():
                shutil.rmtree(slice_root / ".git")

        if cargo_prune and crates:
            _cargo_prune(slice_root, crates)

    stats = measure_repo_path(slice_root)
    manifest = {
        "commit": commit,
        "url": url,
        "paths": paths,
        "created": datetime.now(UTC).isoformat(),
        "file_count": stats["file_count"],
        "loc": stats["loc"],
        "license_files": _find_license_files(slice_root),
        "cargo_prune": cargo_prune,
        "crates": crates,
    }
    (output / "SLICE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if tarball:
        archive = output / "repo_snapshot.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for f in slice_root.rglob("*"):
                if f.is_file():
                    tar.add(f, arcname=f.relative_to(slice_root).as_posix())
        shutil.rmtree(slice_root)
        return archive
    return slice_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Slice a git repo at a commit")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="remote git URL to clone")
    src.add_argument("--repo", type=Path, help="local git repo path")
    p.add_argument("--commit", required=True, help="commit SHA or ref to checkout")
    p.add_argument(
        "--paths",
        default="",
        help="comma-separated paths to include (empty = whole repo minus excludes)",
    )
    p.add_argument("--exclude", default="", help="extra directory names to exclude")
    p.add_argument("--output", type=Path, required=True, help="output directory")
    p.add_argument(
        "--tarball",
        action="store_true",
        help="write repo_snapshot.tar.gz instead of repo/ directory",
    )
    p.add_argument(
        "--cargo-prune",
        action="store_true",
        help="prune Cargo workspace to only crates needed by --crates targets",
    )
    p.add_argument(
        "--crates",
        default="",
        help="comma-separated crate names for --cargo-prune (e.g. my-crate,other-crate)",
    )
    args = p.parse_args(argv)

    paths = [s.strip() for s in args.paths.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
    crates = [s.strip() for s in args.crates.split(",") if s.strip()]

    if args.cargo_prune and not crates:
        print(
            "error: --cargo-prune requires --crates with at least one crate name", file=sys.stderr
        )
        return 1

    try:
        out = slice_repo(
            url=args.url,
            repo=args.repo,
            commit=args.commit,
            paths=paths,
            output=args.output,
            exclude=exclude,
            tarball=args.tarball,
            cargo_prune=args.cargo_prune,
            crates=crates or None,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"slice written to {out}")
    print(f"manifest: {args.output / 'SLICE_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
