"""Tests for language-aware default test commands."""

from __future__ import annotations

from pathlib import Path

from harness.agent.test_commands import default_test_command


def test_python_default(tmp_path: Path) -> None:
    cmd = default_test_command(tmp_path)
    assert "pytest" in cmd


def test_go_default(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/x\n\ngo 1.23\n")
    assert "go test" in default_test_command(tmp_path)


def test_rust_default(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "0.1.0"\n')
    assert "cargo test" in default_test_command(tmp_path)


def test_node_default(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"x"}')
    assert "node --test" in default_test_command(tmp_path) or "npm test" in default_test_command(
        tmp_path
    )
