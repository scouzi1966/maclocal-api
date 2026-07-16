"""Shared task metadata helpers (scale tiers, agent budgets, workspace stats)."""

from __future__ import annotations

import re
import tarfile
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO_SCALES = frozenset({"micro", "small", "medium", "large"})
TASK_COMPLEXITIES = frozenset({"localized", "multi_file", "system", "architecture"})
SCALE_LOC_BOUNDS: dict[str, tuple[int, int | None]] = {
    "micro": (0, 500),
    "small": (500, 1_000),
    "medium": (1_000, 10_000),
    "large": (10_000, 50_000),
}
DEFAULT_MAX_STEPS = 50
DEFAULT_TIMEOUT_S: float | None = None
SCALE_DEFAULTS: dict[str, dict[str, int | float]] = {
    "micro": {"suggested_max_steps": 50, "suggested_timeout_s": 300},
    "small": {"suggested_max_steps": 60, "suggested_timeout_s": 600},
    "medium": {"suggested_max_steps": 100, "suggested_timeout_s": 1200},
    "large": {"suggested_max_steps": 150, "suggested_timeout_s": 1800},
}
MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024
LIST_FILES_CAP = 500
SEARCH_CODE_CAP = 100
CODE_SUFFIXES = frozenset(
    {".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".rs", ".rb", ".cs"}
)
_PLACEHOLDER_COMMIT = re.compile(r"^0+$|^0+1$|^0+2$")


def repo_scale(metadata: dict[str, Any]) -> str:
    raw = metadata.get("repo_scale")
    if raw in REPO_SCALES:
        return str(raw)
    return "micro"


def task_complexity(metadata: dict[str, Any]) -> str:
    raw = metadata.get("task_complexity")
    if raw in TASK_COMPLEXITIES:
        return str(raw)
    return "localized"


def infer_task_complexity_from_gold_patch(patch_text: str) -> str:
    """Infer task shape from the number of source files touched by a gold patch."""
    touched: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        suffix = Path(path).suffix.lower()
        if suffix in CODE_SUFFIXES:
            touched.add(path)

    n_files = len(touched)
    if n_files >= 3:
        return "system"
    if n_files == 2:
        return "multi_file"
    return "localized"


def agent_hints(metadata: dict[str, Any]) -> dict[str, Any]:
    hints = metadata.get("agent_hints")
    return hints if isinstance(hints, dict) else {}


def resolve_max_steps(metadata: dict[str, Any], cli_max_steps: int | None = None) -> int:
    """Resolve step budget from metadata; optional CLI value caps (does not raise)."""
    hints = agent_hints(metadata)
    raw = hints.get("suggested_max_steps", metadata.get("max_steps"))
    if isinstance(raw, int) and raw > 0:
        metadata_steps = raw
    else:
        scale = repo_scale(metadata)
        metadata_steps = int(
            SCALE_DEFAULTS.get(scale, {}).get("suggested_max_steps", DEFAULT_MAX_STEPS)
        )
    if cli_max_steps is not None:
        return min(cli_max_steps, metadata_steps)
    return metadata_steps


def resolve_verifier_timeout_s(metadata: dict[str, Any], default: int = 120) -> int:
    """Per-task verifier subprocess budget (``test_timeout_s``), not agent wall-clock."""
    raw = metadata.get("test_timeout_s")
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    return default


def resolve_agent_timeout_s(
    metadata: dict[str, Any], cli_timeout: float | None = None
) -> float | None:
    """Agent wall-clock budget from hints/scale defaults; CLI ``--timeout`` caps when set."""
    hints = agent_hints(metadata)
    hint_timeout = hints.get("suggested_timeout_s")
    if isinstance(hint_timeout, (int, float)) and hint_timeout > 0:
        timeout: float | None = float(hint_timeout)
    else:
        scale = repo_scale(metadata)
        default = SCALE_DEFAULTS.get(scale, {}).get("suggested_timeout_s")
        timeout = float(default) if isinstance(default, (int, float)) else DEFAULT_TIMEOUT_S
    if cli_timeout is not None:
        return min(cli_timeout, timeout) if timeout is not None else cli_timeout
    return timeout


def resolve_timeout_s(metadata: dict[str, Any], cli_timeout: float | None = None) -> float | None:
    """Backward-compatible alias for :func:`resolve_agent_timeout_s`."""
    return resolve_agent_timeout_s(metadata, cli_timeout)


def upstream_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    upstream = metadata.get("upstream")
    return upstream if isinstance(upstream, dict) else {}


_WORKSPACE_SKIP = frozenset(
    {"issue.md", ".vulcanbenchignore", ".coverage", ".gitignore", ".gitattributes"}
)


def measure_repo_path(root: Path) -> dict[str, int]:
    """Count files and code LOC under a repo directory or extracted workspace."""
    file_count = 0
    loc = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if ".git" in rel.parts or rel.name in _WORKSPACE_SKIP:
            continue
        file_count += 1
        if p.suffix.lower() in CODE_SUFFIXES:
            with suppress(OSError):
                loc += sum(1 for _ in p.read_text(encoding="utf-8", errors="replace").splitlines())
    return {"file_count": file_count, "loc": loc}


def snapshot_uncompressed_size(snapshot: Path) -> int:
    total = 0
    with tarfile.open(snapshot, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                total += member.size
    return total


def validate_scale_fields(task_root: Path, metadata: dict[str, Any]) -> list[str]:  # noqa: PLR0912
    """Return validation failure reasons for scale/provenance extensions."""
    reasons: list[str] = []
    scale = metadata.get("repo_scale")
    if scale is not None and scale not in REPO_SCALES:
        reasons.append(f"repo_scale must be one of {sorted(REPO_SCALES)}")
    complexity = metadata.get("task_complexity")
    if complexity is not None and complexity not in TASK_COMPLEXITIES:
        reasons.append(f"task_complexity must be one of {sorted(TASK_COMPLEXITIES)}")

    source = metadata.get("source")
    if source == "oss":
        if not metadata.get("base_commit"):
            reasons.append("oss tasks require metadata.base_commit")
        elif _PLACEHOLDER_COMMIT.match(str(metadata.get("base_commit"))):
            reasons.append("oss base_commit must be a real upstream SHA (not a placeholder)")
        upstream = upstream_dict(metadata)
        if not upstream.get("url"):
            reasons.append("oss tasks require metadata.upstream.url")

    allow_large = metadata.get("allow_large_snapshot") is True
    snap = task_root / "repo_snapshot.tar.gz"
    if snap.exists() and not allow_large:
        size = snapshot_uncompressed_size(snap)
        if size > MAX_SNAPSHOT_BYTES:
            reasons.append(
                f"repo_snapshot uncompressed size {size} exceeds {MAX_SNAPSHOT_BYTES} "
                "(set allow_large_snapshot: true to override)"
            )

    repo_dir = task_root / "repo"
    if repo_dir.is_dir() and isinstance(scale, str):
        stats = measure_repo_path(repo_dir)
        bounds = SCALE_LOC_BOUNDS.get(scale)
        if bounds is not None:
            _lo, hi = bounds
            loc = stats["loc"]
            if scale == "medium" and loc < 1_000:
                reasons.append(
                    f"repo_scale=medium but repo has only {loc} code LOC (expected >=1000)"
                )
            if scale == "large" and loc < 5_000:
                reasons.append(
                    f"repo_scale=large but repo has only {loc} code LOC (expected >=5000)"
                )
            if hi is not None and loc > hi * 2:
                reasons.append(f"repo_scale={scale} but repo has {loc} code LOC (expected <=~{hi})")

    test_timeout = metadata.get("test_timeout_s")
    if test_timeout is not None and (
        not isinstance(test_timeout, (int, float)) or test_timeout <= 0
    ):
        reasons.append("test_timeout_s must be a positive number when set")

    return reasons


def system_prompt_for_task(metadata: dict[str, Any], base_prompt: str) -> str:
    scale = repo_scale(metadata)
    if scale in ("medium", "large"):
        return (
            f"{base_prompt}\n\nThis is a {scale} codebase: use search_code and partial "
            "read_file (line ranges) to explore before editing. Prefer small, targeted "
            "changes across the modules that own the behavior."
        )
    return base_prompt
