"""Default test commands for the agent ``run_tests`` tool."""

from __future__ import annotations

from pathlib import Path


def default_test_command(workspace: Path) -> str:
    """Return a shell command that runs the workspace's primary test suite.

    Always suffixes with ``|| true`` so the agent sees output without aborting
    the tool loop on failure.
    """
    ws = workspace.resolve()
    if (ws / "Cargo.toml").exists():
        return "cargo test --quiet 2>&1 || true"
    if (ws / "go.mod").exists():
        return "go test ./... 2>&1 || true"
    if (ws / "package.json").exists():
        return (
            "if [ -d tests ] || [ -d test ]; then node --test 2>&1; else npm test 2>&1; fi || true"
        )
    if (ws / "pom.xml").exists() or (ws / "build.gradle").exists():
        return "mvn -q test 2>&1 || ./gradlew test 2>&1 || true"
    return "python -m pytest -q --tb=no 2>&1 || true"
