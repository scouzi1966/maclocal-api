"""Tests for the Cargo-aware workspace pruning logic in slice_repo.py.

Tests the pure-function Cargo.toml rewrite logic (no cargo needed) and
marks the integration test as skipped when cargo is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.slice_repo import (
    _cargo_prune,
    _rewrite_workspace_members,
    _workspace_dependency_closure,
)

# --- _workspace_dependency_closure (pure function) ------------------------------


def test_closure_single_crate() -> None:
    metadata = {
        "packages": [
            {"name": "a", "dependencies": []},
            {"name": "b", "dependencies": []},
        ]
    }
    assert _workspace_dependency_closure(metadata, ["a"]) == {"a"}


def test_closure_transitive() -> None:
    metadata = {
        "packages": [
            {"name": "app", "dependencies": [{"name": "lib"}]},
            {"name": "lib", "dependencies": [{"name": "core"}]},
            {"name": "core", "dependencies": []},
            {"name": "unrelated", "dependencies": []},
        ]
    }
    result = _workspace_dependency_closure(metadata, ["app"])
    assert result == {"app", "lib", "core"}


def test_closure_multiple_targets() -> None:
    metadata = {
        "packages": [
            {"name": "a", "dependencies": [{"name": "shared"}]},
            {"name": "b", "dependencies": [{"name": "shared"}]},
            {"name": "shared", "dependencies": []},
            {"name": "unused", "dependencies": []},
        ]
    }
    result = _workspace_dependency_closure(metadata, ["a", "b"])
    assert result == {"a", "b", "shared"}


# --- _rewrite_workspace_members (file I/O but no cargo) ------------------------


def test_rewrite_members_simple(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text(
        '[workspace]\nmembers = ["a", "b", "c"]\n\n[profile.dev]\n', encoding="utf-8"
    )
    _rewrite_workspace_members(cargo_toml, {"a", "b"})
    content = cargo_toml.read_text(encoding="utf-8")
    # "c" should be gone; "a" and "b" remain.
    assert '"c"' not in content
    assert '"a"' in content
    assert '"b"' in content
    # Other sections preserved.
    assert "[profile.dev]" in content


def test_rewrite_members_single(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text('[workspace]\nmembers = ["x", "y"]\n', encoding="utf-8")
    _rewrite_workspace_members(cargo_toml, {"x"})
    content = cargo_toml.read_text(encoding="utf-8")
    assert '"y"' not in content
    assert '"x"' in content


def test_rewrite_members_multiline(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text(
        '[workspace]\nmembers = [\n  "a",\n  "b",\n  "c",\n]\n\n[profile.dev]\n',
        encoding="utf-8",
    )
    _rewrite_workspace_members(cargo_toml, {"a", "c"})
    content = cargo_toml.read_text(encoding="utf-8")
    assert '"b"' not in content
    assert '"a"' in content
    assert '"c"' in content
    assert "[profile.dev]" in content


def test_rewrite_preserves_non_workspace_sections(tmp_path: Path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_toml.write_text(
        '[package]\nname = "root"\nversion = "0.1.0"\n\n[workspace]\nmembers = ["a", "b"]\n\n[dependencies]\n',
        encoding="utf-8",
    )
    _rewrite_workspace_members(cargo_toml, {"a"})
    content = cargo_toml.read_text(encoding="utf-8")
    assert "[package]" in content
    assert "[dependencies]" in content
    assert '"b"' not in content
    assert '"a"' in content


# --- integration test (requires cargo) ------------------------------------------


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_cargo_prune_integration(tmp_path: Path) -> None:
    """End-to-end cargo-prune test with a real Cargo workspace."""

    # Create a minimal two-crate workspace.
    root = tmp_path / "repo"
    root.mkdir()
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["keep", "drop"]\nresolver = "2"\n', encoding="utf-8"
    )
    keep = root / "keep"
    keep.mkdir()
    (keep / "Cargo.toml").write_text(
        '[package]\nname = "keep"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
    )
    (keep / "src").mkdir()
    (keep / "src" / "lib.rs").write_text("pub fn f() -> i32 { 1 }\n", encoding="utf-8")
    drop = root / "drop"
    drop.mkdir()
    (drop / "Cargo.toml").write_text(
        '[package]\nname = "drop"\nversion = "0.1.0"\nedition = "2021"\n', encoding="utf-8"
    )
    (drop / "src").mkdir()
    (drop / "src" / "lib.rs").write_text("pub fn g() -> i32 { 2 }\n", encoding="utf-8")

    # Generate Cargo.lock.
    subprocess.run(["cargo", "generate-lockfile"], cwd=root, check=True, capture_output=True)

    _cargo_prune(root, ["keep"])

    assert not (root / "drop").exists()
    assert (root / "keep").is_dir()
    cargo_content = (root / "Cargo.toml").read_text(encoding="utf-8")
    assert "drop" not in cargo_content
    assert "keep" in cargo_content
