"""End-to-end test of the real agent loop using the deterministic mock provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import harness.agent.loop as loop_mod
from harness.agent.loop import run_agent, run_synthetic_hello
from harness.agent.providers import LLMProvider, LLMResponse, TokenUsage, ToolInvocation
from harness.redaction import MAX_FIELD_CHARS


def test_run_agent_solves_hello_world(tmp_path: Path) -> None:
    res = run_agent(
        task_id="hello-world",
        model="mock:synthetic",
        output_dir=tmp_path,
        tasks_root=Path("tasks/v1"),
    )
    run_dir = tmp_path / res["run_id"]
    summary = res["summary"]

    # Functional success via the real verifier.
    assert summary["scores"]["functional"] == 1.0
    assert summary["scores"]["total"] > 0.9
    assert summary["finished"] is True
    assert summary["total_tokens"] > 0

    # Full evaluator populates the remaining metrics (mock judge is deterministic).
    assert summary["scores"]["quality"] is not None
    assert summary["scores"]["security"] == 1.0
    assert summary["scores"]["human_like"] == 0.8
    # Judge tokens are tracked separately, not folded into the run's token count.
    assert summary["scores"]["metric_details"]["human_like"]["judge_tokens"] > 0

    # Cost + latency captured (mock is priced at $0).
    assert summary["tokens"]["prompt"] > 0 and summary["tokens"]["completion"] > 0
    assert summary["cost_usd"] == 0.0
    assert summary["duration_s"] >= 0
    assert "started_at" in summary

    # Reproducibility manifest: runtime + sandbox + tool versions.
    manifest = summary["manifest"]
    assert manifest["model"] == "mock:synthetic"
    assert manifest["runtime"]["python"]
    assert manifest["sandbox"]["mode"] == "local"
    assert "git" in manifest["tools"]

    # Run records the task's scoring-definition hash (for drift detection).
    assert len(summary["task_hash"]) == 64

    # The agent actually wrote the file in its own workspace.
    assert (run_dir / "workspace" / "hello.py").exists()

    # A real (non-synthetic) git patch was captured.
    patch = (run_dir / "final.patch").read_text()
    assert "hello.py" in patch
    assert ".coverage" not in patch  # build artifacts are git-ignored

    # Trace contains the expected event sequence and a replay was generated.
    types = [
        json.loads(line)["type"] for line in (run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert "tool_call" in types
    assert "metric_computed" in types
    assert (run_dir / "replay.html").exists()


def test_run_agent_records_mock_effort_metadata(tmp_path: Path) -> None:
    res = run_agent(
        task_id="hello-world",
        model="mock:synthetic",
        output_dir=tmp_path,
        tasks_root=Path("tasks/v1"),
        judges=False,
        effort="low",
        experiment_id="experiment-test",
    )
    summary = res["summary"]
    assert summary["effort"] == {
        "requested": "low",
        "provider": "mock",
        "provider_value": None,
        "supported": False,
    }
    assert summary["experiment_id"] == "experiment-test"

    manifest_task = summary["manifest"]["task"]
    assert manifest_task["task_complexity"] == "localized"
    assert manifest_task["languages"] == ["python"]
    assert manifest_task["difficulty"] == "trivial"


def test_run_synthetic_hello_wrapper(tmp_path: Path) -> None:
    res = run_synthetic_hello(Path("tasks/v1/hello-world"), output_dir=tmp_path)
    assert res["summary"]["scores"]["functional"] == 1.0


def test_run_agent_closes_executor_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox executor is cleaned up even if the agent loop raises."""

    class FakeExecutor:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake = FakeExecutor()
    monkeypatch.setattr(loop_mod, "_make_executor", lambda *a, **k: fake)

    class BoomProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "mock"

        def complete(self, messages, tools):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_agent(
            task_id="hello-world",
            model="mock:boom",
            output_dir=tmp_path,
            provider=BoomProvider("boom"),
            tasks_root=Path("tasks/v1"),
        )
    assert fake.closed is True


def test_run_agent_wall_clock_budget_aborts(tmp_path: Path) -> None:
    """A zero/elapsed-exceeding budget aborts the loop cleanly before doing work."""
    res = run_agent(
        task_id="hello-world",
        model="mock:synthetic",
        output_dir=tmp_path,
        tasks_root=Path("tasks/v1"),
        judges=False,
        timeout_s=0.0,  # any elapsed time exceeds this -> abort at step 1
    )
    summary = res["summary"]
    assert summary["finished"] is False
    assert summary["scores"]["functional"] == 0.0  # agent never solved it
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / res["run_id"] / "trace.jsonl").read_text().splitlines()
    ]
    assert "budget_exceeded" in types
    assert "llm_request" not in types  # aborted before the first model call


def test_run_agent_budget_expiring_during_provider_does_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow provider response cannot mark the run finished after the deadline."""
    ticks = iter([0.0, 0.0, 0.0, 0.6])
    monkeypatch.setattr(loop_mod.time, "monotonic", lambda: next(ticks, 0.6))

    seen_timeouts: list[float | None] = []

    class SlowFinishProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "mock"

        def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            timeout_s: float | None = None,
        ) -> LLMResponse:
            seen_timeouts.append(timeout_s)
            return LLMResponse(
                content="FINISH: too late",
                usage=TokenUsage(prompt_tokens=3, completion_tokens=4),
            )

    res = run_agent(
        task_id="hello-world",
        model="mock:slow",
        output_dir=tmp_path,
        provider=SlowFinishProvider("slow"),
        tasks_root=Path("tasks/v1"),
        judges=False,
        timeout_s=0.5,
    )

    summary = res["summary"]
    assert seen_timeouts == [pytest.approx(0.5)]
    assert summary["finished"] is False
    assert summary["scores"]["budget_exceeded"] is True
    assert summary["verifier"]["budget_exceeded"] is True
    types = [
        json.loads(line)["type"]
        for line in (tmp_path / res["run_id"] / "trace.jsonl").read_text().splitlines()
    ]
    assert "llm_request" in types
    assert "llm_response" in types
    assert "budget_exceeded" in types
    assert "test_result" not in types


def test_final_patch_is_redacted_and_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "ghp_" + "A" * 40
    huge_patch = f"diff --git a/x b/x\n+{secret}\n+" + ("x" * (MAX_FIELD_CHARS + 5000))
    monkeypatch.setattr(loop_mod, "_git_diff", lambda workspace: huge_patch)
    monkeypatch.setattr(loop_mod, "_git_changed_files", lambda workspace: [])
    monkeypatch.setattr(loop_mod, "_verify", lambda *a, **k: (0.0, {"scores": {"functional": 0.0}}))

    class FinishProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "mock"

        def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            timeout_s: float | None = None,
        ) -> LLMResponse:
            return LLMResponse(content="FINISH: done")

    res = run_agent(
        task_id="hello-world",
        model="mock:finish",
        output_dir=tmp_path,
        provider=FinishProvider("finish"),
        tasks_root=Path("tasks/v1"),
        judges=False,
    )

    patch = (tmp_path / res["run_id"] / "final.patch").read_text(encoding="utf-8")
    assert secret not in patch
    assert "[REDACTED]" in patch
    assert "truncated" in patch
    assert len(patch) <= MAX_FIELD_CHARS + 80


def test_execute_tool_caps_run_command_timeout_to_remaining_budget() -> None:
    seen: list[int | None] = []

    class FakeExecutor:
        def execute(self, call):  # type: ignore[no-untyped-def]
            seen.append(call.args.get("timeout"))
            return SimpleNamespace(result={"ok": True}, error=None)

    tc = ToolInvocation(
        id="cmd",
        name="run_command",
        arguments={"cmd": "sleep 100", "timeout": 999},
    )
    observation = loop_mod._execute_tool(FakeExecutor(), tc, "run", 1, timeout_s=2.2)

    assert observation == {"result": {"ok": True}, "error": None}
    assert seen == [3]


def test_execute_tool_without_run_budget_preserves_requested_timeout() -> None:
    seen: list[int | None] = []

    class FakeExecutor:
        def execute(self, call):  # type: ignore[no-untyped-def]
            seen.append(call.args.get("timeout"))
            return SimpleNamespace(result={"ok": True}, error=None)

    tc = ToolInvocation(
        id="cmd",
        name="run_command",
        arguments={"cmd": "sleep 100", "timeout": 999},
    )
    observation = loop_mod._execute_tool(FakeExecutor(), tc, "run", 1)

    assert observation == {"result": {"ok": True}, "error": None}
    assert seen == [999]


def test_execute_tool_caps_search_timeout_to_remaining_budget() -> None:
    seen: list[int | None] = []

    class FakeExecutor:
        def execute(self, call):  # type: ignore[no-untyped-def]
            seen.append(call.args.get("timeout"))
            return SimpleNamespace(result=[], error=None)

    tc = ToolInvocation(
        id="search",
        name="search_code",
        arguments={"query": "needle", "timeout": 30},
    )
    observation = loop_mod._execute_tool(FakeExecutor(), tc, "run", 1, timeout_s=1.1)

    assert observation == {"result": [], "error": None}
    assert seen == [2]


def test_run_agent_no_judges_leaves_human_like_null(tmp_path: Path) -> None:
    res = run_agent(
        task_id="hello-world",
        model="mock:synthetic",
        output_dir=tmp_path,
        tasks_root=Path("tasks/v1"),
        judges=False,
    )
    assert res["summary"]["scores"]["human_like"] is None


def test_run_agent_unknown_tool_is_tolerated(tmp_path: Path) -> None:
    """A model requesting a bogus tool gets an error observation, not a crash."""

    class BadToolProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "mock"

        def complete(self, messages, tools):  # type: ignore[no-untyped-def]
            if not any(m.get("role") == "tool" for m in messages):
                return LLMResponse(
                    tool_calls=[ToolInvocation(id="x", name="frobnicate", arguments={})]
                )
            return LLMResponse(content="FINISH: gave up")

    res = run_agent(
        task_id="hello-world",
        model="mock:bad",
        output_dir=tmp_path,
        provider=BadToolProvider("bad"),
        tasks_root=Path("tasks/v1"),
    )
    # Unknown tool -> functional 0 but no crash, full trace still produced.
    assert res["summary"]["scores"]["functional"] == 0.0


def test_changed_files_are_captured_before_hidden_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hidden verifier tests must not be scored as agent-authored changes."""
    tasks_root = tmp_path / "tasks"
    task = tasks_root / "fix-f"
    (task / "repo").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    (task / "repo" / "m.py").write_text("def f():\n    return 1\n")
    (task / "tests" / "t_f2p.py").write_text(
        "from m import f\n\ndef test_two():\n    assert f() == 2\n"
    )
    (task / "metadata.json").write_text(
        json.dumps(
            {
                "id": "fix-f",
                "category": "bug_fix",
                "languages": ["python"],
                "difficulty": "trivial",
                "created": "2026-05-30",
                "source": "fixture",
                "decontamination_notes": "fixture",
                "tests": {
                    "fail_to_pass": [
                        {"name": "two", "cmd": f"{sys.executable} -m pytest t_f2p.py -q"}
                    ],
                    "pass_to_pass": [],
                },
            }
        )
    )
    (task / "issue.md").write_text("make f return 2")

    captured: list[list[str]] = []

    def fake_evaluate_run(**kwargs: Any) -> dict[str, Any]:
        captured.append(list(kwargs["changed_files"]))
        return {
            "functional": kwargs["functional"],
            "quality": None,
            "security": None,
            "efficiency": 1.0,
            "human_like": None,
            "total": 1.0,
            "metric_details": {},
        }

    class FixFProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "mock"

        def complete(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> LLMResponse:
            tool_steps = sum(1 for m in messages if m.get("role") == "tool")
            if tool_steps == 0:
                return LLMResponse(
                    tool_calls=[
                        ToolInvocation(id="read", name="read_file", arguments={"path": "m.py"})
                    ]
                )
            if tool_steps == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolInvocation(
                            id="edit",
                            name="edit_file",
                            arguments={
                                "path": "m.py",
                                "old_string": "return 1",
                                "new_string": "return 2",
                            },
                        )
                    ]
                )
            return LLMResponse(content="FINISH: done")

    monkeypatch.setattr(loop_mod, "evaluate_run", fake_evaluate_run)
    res = run_agent(
        task_id="fix-f",
        model="mock:fix-f",
        output_dir=tmp_path / "runs",
        provider=FixFProvider("fix-f"),
        tasks_root=tasks_root,
        judges=False,
    )

    run_dir = tmp_path / "runs" / res["run_id"]
    assert captured == [["m.py"]]
    assert (run_dir / "workspace" / "t_f2p.py").exists()
    assert "t_f2p.py" not in (run_dir / "final.patch").read_text(encoding="utf-8")
