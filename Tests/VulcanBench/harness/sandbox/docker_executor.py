"""Docker-backed tool executor.

Runs the agent's *code execution* (``run_command`` and friends) inside an
isolated, non-root, network-off, resource-limited container, while delegating
*file* operations to a host-side :class:`LocalToolExecutor` over the bind-mounted
workspace. Splitting it this way keeps file semantics identical to local runs
(same bytes, same path-escape guard) and confines the only untrusted thing -- the
shell commands a model asks to run -- to the container.

The workspace is bind-mounted at ``/workspace`` and the container runs as the
host UID/GID so files created inside stay writable and owned by the host user
(and never root). The container is detached and long-lived (``sleep infinity``);
each tool call is a ``docker exec`` into it. Always ``close()`` it (the agent
loop does so in a ``finally``).
"""

from __future__ import annotations

import contextlib
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

import docker

from harness.agent.local_executor import LocalToolExecutor
from harness.agent.protocol import RunCommandArgs, ToolProtocol
from harness.agent.test_commands import default_test_command

if TYPE_CHECKING:
    from harness.agent.protocol import (
        EditFileArgs,
        ListFilesArgs,
        ReadFileArgs,
        SearchCodeArgs,
    )

# Default image. Overridable via env or --image; this base image contains the
# seed-task toolchains used by in-container verification (Python, Go, Node).
DEFAULT_IMAGE = os.environ.get("VULCANBENCH_SANDBOX_IMAGE", "vulcanbench/sandbox:base")

_CONTAINER_WORKDIR = "/workspace"

# Writable paths for non-root containers (host UID/GID). Without these, `go test`
# fails trying to create ~/.cache/go-build under `/` when HOME is unset.
_SANDBOX_ENV = {
    "HOME": "/tmp",
    "GOCACHE": "/tmp/go-build",
    "GOPATH": "/tmp/go",
    "CARGO_HOME": "/tmp/cargo",
}


class SandboxError(RuntimeError):
    """Raised when the sandbox container cannot be created or used."""


def _decode(b: Any) -> str:
    """Decode a (possibly None) bytes chunk from a container exec stream."""
    if isinstance(b, (bytes, bytearray)):
        return bytes(b).decode("utf-8", errors="replace")
    return "" if b is None else str(b)


def _docker_available() -> bool:
    """True if a Docker daemon is reachable. Used for ``auto`` mode and tests."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class DockerToolExecutor(ToolProtocol):
    """Execute tools against a containerized, bind-mounted workspace."""

    def __init__(
        self,
        workspace: Path | str = ".",
        image: str = DEFAULT_IMAGE,
        network: bool = False,
        mem_limit: str = "2g",
        cpus: float = 2.0,
        pids_limit: int = 512,
        default_timeout: int = 120,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        # File ops run host-side over the shared bind mount.
        self._files = LocalToolExecutor(self.workspace)
        self.image = image
        self.default_timeout = default_timeout
        self._closed = False

        try:
            self._client = docker.from_env()
            self._client.ping()
        except Exception as e:
            raise SandboxError(
                f"could not connect to the Docker daemon ({e}). Is Docker running?"
            ) from e

        try:
            self._container = self._client.containers.run(
                image,
                command=["sleep", "infinity"],
                detach=True,
                working_dir=_CONTAINER_WORKDIR,
                volumes={str(self.workspace): {"bind": _CONTAINER_WORKDIR, "mode": "rw"}},
                environment=_SANDBOX_ENV,
                network_disabled=not network,
                mem_limit=mem_limit,
                nano_cpus=int(cpus * 1_000_000_000),
                pids_limit=pids_limit,
                user=f"{os.getuid()}:{os.getgid()}",
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                tty=False,
                auto_remove=False,
            )
        except Exception as e:
            raise SandboxError(f"failed to start sandbox container from {image!r}: {e}") from e

    # --- file operations: delegate to the host-side executor ------------------
    def list_files(self, args: ListFilesArgs) -> Any:
        return self._files.list_files(args)

    def read_file(self, args: ReadFileArgs) -> Any:
        return self._files.read_file(args)

    def search_code(self, args: SearchCodeArgs) -> Any:
        return self._files.search_code(args)

    def edit_file(self, args: EditFileArgs) -> Any:
        return self._files.edit_file(args)

    def security_scan(self, timeout_s: float | None = None) -> Any:
        # Static analysis reads files (no untrusted execution) -> host-side.
        return self._files.security_scan(timeout_s=timeout_s)

    # --- execution operations: run inside the container -----------------------
    def run_command(self, args: RunCommandArgs) -> dict[str, Any]:
        workdir = _CONTAINER_WORKDIR
        if args.cwd:
            # Keep the agent within the workspace; reuse the local guard.
            resolved = self._files._resolve(args.cwd)
            workdir = f"{_CONTAINER_WORKDIR}/{resolved.relative_to(self.workspace)}"
        timeout = args.timeout or self.default_timeout
        inner = f"cd {shlex.quote(workdir)} && timeout {timeout}s sh -c {shlex.quote(args.cmd)}"
        return self._exec(inner)

    def run_tests(self) -> dict[str, Any]:
        cmd = default_test_command(self.workspace)
        return self.run_command(RunCommandArgs(cmd=cmd))

    def run_lint(self) -> dict[str, Any]:
        return self.run_command(RunCommandArgs(cmd="ruff check . || true"))

    def run_build(self) -> dict[str, Any]:
        return {"ok": True}

    def _exec(self, shell_cmd: str) -> dict[str, Any]:
        try:
            result = self._container.exec_run(
                ["sh", "-c", shell_cmd], workdir=_CONTAINER_WORKDIR, demux=True
            )
        except Exception as e:
            raise SandboxError(f"container exec failed: {e}") from e
        raw = result.output
        out_b, err_b = raw if isinstance(raw, tuple) else (raw, None)
        return {
            "stdout": _decode(out_b),
            "stderr": _decode(err_b),
            "exit_code": result.exit_code,
        }

    # --- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        """Stop and remove the container. Safe to call more than once."""
        if getattr(self, "_closed", True):
            return
        container = getattr(self, "_container", None)
        self._closed = True
        if container is None:
            return
        with contextlib.suppress(Exception):
            container.stop(timeout=5)
        with contextlib.suppress(Exception):
            container.remove(force=True)

    def __enter__(self) -> DockerToolExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        with contextlib.suppress(Exception):
            self.close()
