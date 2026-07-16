"""Tests for the Docker sandbox executor.

The Docker daemon is not assumed to be running, so these mock the Docker SDK
(``docker.from_env``) to exercise construction, exec, and cleanup. One live test
is gated on ``_docker_available()`` and skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import docker
import pytest

from harness.agent.loop import _collect_manifest, _executor_runner
from harness.agent.protocol import EditFileArgs, ReadFileArgs, RunCommandArgs
from harness.sandbox.docker_executor import (
    _SANDBOX_ENV,
    DEFAULT_IMAGE,
    DockerToolExecutor,
    SandboxError,
    _docker_available,
)


class FakeExecResult:
    def __init__(self, exit_code: int, output: tuple[bytes | None, bytes | None]) -> None:
        self.exit_code = exit_code
        self.output = output


class FakeContainer:
    def __init__(self) -> None:
        self.exec_calls: list[dict[str, Any]] = []
        self.stopped = False
        self.removed = False

    def exec_run(self, cmd: Any, workdir: str | None = None, demux: bool = False) -> FakeExecResult:
        self.exec_calls.append({"cmd": cmd, "workdir": workdir, "demux": demux})
        return FakeExecResult(0, (b"hi\n", b""))

    def stop(self, timeout: int | None = None) -> None:
        self.stopped = True

    def remove(self, force: bool = False) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self) -> None:
        self.run_kwargs: dict[str, Any] | None = None
        self.container = FakeContainer()

    def run(self, image: str, **kwargs: Any) -> FakeContainer:
        self.run_kwargs = {"image": image, **kwargs}
        return self.container


class FakeClient:
    def __init__(self) -> None:
        self.containers = FakeContainers()

    def ping(self) -> bool:
        return True


@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(docker, "from_env", lambda: client)
    return client


def test_construction_kwargs(tmp_path: Path, fake_docker: FakeClient) -> None:
    ex = DockerToolExecutor(tmp_path, image="img:test", network=False, cpus=2.0)
    kw = fake_docker.containers.run_kwargs
    assert kw is not None
    assert kw["image"] == "img:test"
    assert kw["command"] == ["sleep", "infinity"]
    assert kw["working_dir"] == "/workspace"
    assert kw["volumes"] == {str(tmp_path.resolve()): {"bind": "/workspace", "mode": "rw"}}
    assert kw["network_disabled"] is True
    assert kw["nano_cpus"] == 2_000_000_000
    assert kw["pids_limit"] == 512
    assert kw["user"] == f"{os.getuid()}:{os.getgid()}"
    assert kw["environment"] == _SANDBOX_ENV
    assert "no-new-privileges" in kw["security_opt"]
    assert kw["cap_drop"] == ["ALL"]
    ex.close()


def test_default_image_is_all_language_base() -> None:
    if os.environ.get("VULCANBENCH_SANDBOX_IMAGE"):
        pytest.skip("default image overridden by environment")
    assert DEFAULT_IMAGE == "vulcanbench/sandbox:base"


def test_network_enabled_flag(tmp_path: Path, fake_docker: FakeClient) -> None:
    DockerToolExecutor(tmp_path, network=True).close()
    assert fake_docker.containers.run_kwargs["network_disabled"] is False


def test_run_command_execs_in_container(tmp_path: Path, fake_docker: FakeClient) -> None:
    ex = DockerToolExecutor(tmp_path, image="img")
    out = ex.run_command(RunCommandArgs(cmd="echo hi"))
    assert out == {"stdout": "hi\n", "stderr": "", "exit_code": 0}
    call = fake_docker.containers.container.exec_calls[-1]
    assert call["cmd"][:2] == ["sh", "-c"]
    assert "timeout" in call["cmd"][2]
    assert "echo hi" in call["cmd"][2]
    assert call["demux"] is True
    ex.close()


def test_run_tests_and_lint_exec(tmp_path: Path, fake_docker: FakeClient) -> None:
    ex = DockerToolExecutor(tmp_path)
    ex.run_tests()
    ex.run_lint()
    cmds = [c["cmd"][2] for c in fake_docker.containers.container.exec_calls]
    assert any("pytest" in c for c in cmds)
    assert any("ruff" in c for c in cmds)
    assert ex.run_build() == {"ok": True}
    ex.close()


def test_executor_runner_execs_in_container(tmp_path: Path, fake_docker: FakeClient) -> None:
    """The in-container verifier runner routes test commands through exec_run."""
    ex = DockerToolExecutor(tmp_path)
    runner = _executor_runner(ex)
    exit_code = runner("pytest test_x.py -q", tmp_path, 60)
    assert exit_code == 0  # FakeContainer.exec_run returns exit_code 0
    last = fake_docker.containers.container.exec_calls[-1]["cmd"][2]
    assert "pytest test_x.py -q" in last
    ex.close()


def test_manifest_versions_are_collected_in_container(
    tmp_path: Path, fake_docker: FakeClient
) -> None:
    ex = DockerToolExecutor(tmp_path)
    manifest = _collect_manifest("mock:model", None, False, ex)
    assert manifest["sandbox"]["mode"] == "docker"
    assert manifest["tools"]["git"] == "hi"
    assert manifest["runtime"]["python"] == "hi"
    cmds = "\n".join(c["cmd"][2] for c in fake_docker.containers.container.exec_calls)
    assert "git --version" in cmds
    assert "go version" in cmds
    assert "node --version" in cmds
    ex.close()


def test_file_ops_delegate_to_host(tmp_path: Path, fake_docker: FakeClient) -> None:
    ex = DockerToolExecutor(tmp_path)
    ex.edit_file(EditFileArgs(path="a.txt", old_string="", new_string="data\n"))
    # File written on the host side (the shared bind mount), no container exec.
    assert (tmp_path / "a.txt").read_text() == "data\n"
    assert ex.read_file(ReadFileArgs(path="a.txt")) == "data\n"
    assert fake_docker.containers.container.exec_calls == []
    ex.close()


def test_close_stops_and_removes(tmp_path: Path, fake_docker: FakeClient) -> None:
    ex = DockerToolExecutor(tmp_path)
    container = fake_docker.containers.container
    ex.close()
    assert container.stopped and container.removed
    ex.close()  # idempotent, no error


def test_context_manager(tmp_path: Path, fake_docker: FakeClient) -> None:
    with DockerToolExecutor(tmp_path) as ex:
        assert ex is not None
    assert fake_docker.containers.container.removed


def test_daemon_down_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> Any:
        raise ConnectionError("no daemon")

    monkeypatch.setattr(docker, "from_env", boom)
    with pytest.raises(SandboxError, match="Docker daemon"):
        DockerToolExecutor(tmp_path)


def test_docker_available_false_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker, "from_env", lambda: (_ for _ in ()).throw(OSError("down")))
    assert _docker_available() is False


def test_docker_available_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker, "from_env", FakeClient)
    assert _docker_available() is True


@pytest.mark.docker
@pytest.mark.skipif(not _docker_available(), reason="requires a running Docker daemon")
def test_live_go_test_without_gocache_export(tmp_path: Path) -> None:
    """Go verification must work when the agent does not export GOCACHE."""
    if os.environ.get("VULCANBENCH_SANDBOX_IMAGE"):
        image = os.environ["VULCANBENCH_SANDBOX_IMAGE"]
    else:
        image = DEFAULT_IMAGE
    ws = tmp_path / "goproj"
    (ws / "stack").mkdir(parents=True)
    (ws / "go.mod").write_text("module example.com/stackdemo\n\ngo 1.23\n")
    (ws / "stack" / "stack.go").write_text(
        "package stack\n\ntype Stack struct{ items []int }\n"
        "func New() *Stack { return &Stack{} }\n"
        "func (s *Stack) Push(v int) { s.items = append(s.items, v) }\n"
        "func (s *Stack) Len() int { return len(s.items) }\n"
    )
    (ws / "stack" / "pop.go").write_text(
        "package stack\n\nfunc (s *Stack) Pop() (int, bool) {\n"
        "  if len(s.items) == 0 { return 0, false }\n"
        "  i := len(s.items) - 1\n  v := s.items[i]\n"
        "  s.items = s.items[:i]\n  return v, true\n}\n"
    )
    (ws / "stack" / "pop_test.go").write_text(
        'package stack\n\nimport "testing"\n\n'
        "func TestPopLIFO(t *testing.T) {\n"
        "  s := New()\n  s.Push(1)\n  s.Push(2)\n"
        "  v, ok := s.Pop()\n"
        '  if !ok || v != 2 { t.Fatalf("got %d,%v want 2,true", v, ok) }\n'
        "}\n"
    )
    with DockerToolExecutor(ws, image=image) as ex:
        runner = _executor_runner(ex)
        assert runner("go test -run '^TestPopLIFO$' ./...", ws, 120) == 0


@pytest.mark.docker
@pytest.mark.skipif(not _docker_available(), reason="requires a running Docker daemon")
def test_live_run_command(tmp_path: Path) -> None:
    with DockerToolExecutor(tmp_path, image="python:3.12-slim") as ex:
        out = ex.run_command(RunCommandArgs(cmd="echo sandbox-ok"))
        assert out["exit_code"] == 0
        assert "sandbox-ok" in out["stdout"]
