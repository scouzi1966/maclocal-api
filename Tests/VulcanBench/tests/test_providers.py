"""Tests for the LLM provider interface."""

from __future__ import annotations

import pytest

from harness.agent import providers as P
from harness.agent.providers import (
    AnthropicProvider,
    LLMResponse,
    MockProvider,
    OpenAIProvider,
    TokenUsage,
    ZaiProvider,
    get_provider,
    parse_model_spec,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        ("anthropic:claude-opus-4-8", ("anthropic", "claude-opus-4-8")),
        ("zai:glm-5.2", ("zai", "glm-5.2")),
        ("mock:synthetic", ("mock", "synthetic")),
        ("openai:gpt-4o:extra", ("openai", "gpt-4o:extra")),
    ],
)
def test_parse_model_spec(spec: str, expected: tuple[str, str]) -> None:
    assert parse_model_spec(spec) == expected


@pytest.mark.parametrize("bad", ["gpt-4o", "openai:", ":model", ""])
def test_parse_model_spec_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_model_spec(bad)


def test_get_provider_unknown() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope:x")


def test_get_provider_returns_mock() -> None:
    p = get_provider("mock:synthetic")
    assert isinstance(p, MockProvider)
    assert p.name == "mock"
    assert p.spec == "mock:synthetic"


def test_get_provider_returns_zai() -> None:
    p = get_provider("zai:glm-5.2")
    assert isinstance(p, ZaiProvider)
    assert p.name == "zai"
    assert p.spec == "zai:glm-5.2"


def test_token_usage_total() -> None:
    assert TokenUsage(prompt_tokens=10, completion_tokens=5).total == 15


def test_llm_response_wants_tools() -> None:
    assert LLMResponse().wants_tools is False


def test_mock_provider_scripted_policy() -> None:
    """Mock walks read -> edit -> test -> finish based on tool-result count."""
    p = MockProvider("synthetic")
    msgs: list[dict[str, object]] = [{"role": "user", "content": "issue"}]

    r0 = p.complete(msgs, [])
    assert r0.tool_calls[0].name == "read_file"

    msgs.append({"role": "tool", "content": "..."})
    r1 = p.complete(msgs, [])
    assert r1.tool_calls[0].name == "edit_file"

    msgs.append({"role": "tool", "content": "..."})
    r2 = p.complete(msgs, [])
    assert r2.tool_calls[0].name == "run_tests"

    msgs.append({"role": "tool", "content": "..."})
    r3 = p.complete(msgs, [])
    assert not r3.wants_tools
    assert r3.content is not None and "FINISH" in r3.content


def test_loads_args() -> None:
    assert P._loads_args('{"a": 1}') == {"a": 1}
    assert P._loads_args({"a": 1}) == {"a": 1}
    assert P._loads_args("not json") == {}
    assert P._loads_args(None) == {}


def test_openai_tool_to_anthropic() -> None:
    tool = {"function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}}}
    out = P._openai_tool_to_anthropic(tool)
    assert out["name"] == "read_file"
    assert out["input_schema"] == {"type": "object"}


def test_to_anthropic_messages_conversion() -> None:
    messages = [
        {"role": "system", "content": "be good"},
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {"id": "t1", "function": {"name": "read_file", "arguments": '{"path": "x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    system, converted = P._to_anthropic_messages(messages)
    assert system == "be good"
    # assistant turn carries text then a tool_use block; tool turn -> tool_result
    assistant_blocks = converted[1]["content"]
    assert {b["type"] for b in assistant_blocks} == {"text", "tool_use"}
    tool_use = next(b for b in assistant_blocks if b["type"] == "tool_use")
    assert tool_use["name"] == "read_file"
    assert converted[2]["content"][0]["type"] == "tool_result"
    assert converted[2]["content"][0]["tool_use_id"] == "t1"


def test_openai_complete_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict[str, object] = {}

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen["payload"] = payload
        assert "chat/completions" in url
        assert payload["tools"]  # tools forwarded
        return {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = OpenAIProvider("gpt-4o").complete(
        [{"role": "user", "content": "hi"}], [{"function": {"name": "read_file"}}]
    )
    assert resp.content == "ok"
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "a"}
    assert resp.usage.total == 14
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["temperature"] == 0


def test_openai_gpt5_omits_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict[str, object] = {}

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = OpenAIProvider("gpt-5.5").complete([{"role": "user", "content": "hi"}], [])
    assert resp.content == "ok"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert "temperature" not in payload


def test_openai_complete_uses_budgeted_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: list[float] = []

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen.append(timeout)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = OpenAIProvider("gpt-4o").complete([], [], timeout_s=3.2)

    assert resp.content == "ok"
    assert seen == [3.2]


def test_openai_effort_uses_responses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict[str, object] = {}

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen["url"] = url
        seen["payload"] = payload
        return {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "read_file",
                    "arguments": '{"path": "a"}',
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ],
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = OpenAIProvider("gpt-5.1").complete(
        [{"role": "user", "content": "hi"}],
        [{"function": {"name": "read_file", "description": "read", "parameters": {}}}],
        effort="xhigh",
    )

    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert "responses" in seen["url"]
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert payload["tools"][0]["name"] == "read_file"
    assert resp.content == "ok"
    assert resp.tool_calls[0].arguments == {"path": "a"}
    assert resp.usage.total == 14


def test_openai_complete_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(P.ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider("gpt-4o").complete([], [])


def test_anthropic_complete_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        assert "/v1/messages" in url
        return {
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "u1", "name": "edit_file", "input": {"path": "z"}},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = AnthropicProvider("claude-opus-4-8").complete(
        [{"role": "user", "content": "hi"}], [{"function": {"name": "edit_file", "parameters": {}}}]
    )
    assert resp.content == "thinking"
    assert resp.tool_calls[0].name == "edit_file"
    assert resp.usage.total == 7


def test_anthropic_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(P.ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("claude-opus-4-8").complete([], [])


def test_anthropic_effort_and_no_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    seen: dict = {}  # type: ignore[type-arg]

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return {"content": [{"type": "text", "text": "ok"}], "usage": {}}

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = AnthropicProvider("claude-opus-4-8").complete(
        [{"role": "user", "content": "hi"}], [], effort="low"
    )
    assert resp.content == "ok"
    assert seen["output_config"] == {"effort": "low"}
    # Sampling params are rejected with a 400 by Opus 4.7+ — never send them.
    assert "temperature" not in seen
    assert "top_p" not in seen


def test_anthropic_no_effort_omits_output_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    seen: dict = {}  # type: ignore[type-arg]

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen.update(payload)
        return {"content": [{"type": "text", "text": "ok"}], "usage": {}}

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    AnthropicProvider("claude-opus-4-8").complete([{"role": "user", "content": "hi"}], [])
    assert "output_config" not in seen


def test_zai_complete_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "zai-test")
    seen: dict[str, object] = {}

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen["url"] = url
        assert payload["tools"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = ZaiProvider("glm-5.2").complete(
        [{"role": "user", "content": "hi"}], [{"function": {"name": "read_file"}}]
    )
    assert "chat/completions" in seen["url"]
    assert resp.content == "ok"
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "a"}
    assert resp.usage.total == 14


def test_zai_complete_uses_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "zai-test")
    monkeypatch.setenv("ZAI_BASE_URL", "https://custom.z.ai/v1")
    seen: dict[str, object] = {}

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen["url"] = url
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = ZaiProvider("glm-5.2").complete([], [])
    assert seen["url"] == "https://custom.z.ai/v1/chat/completions"
    assert resp.content == "ok"


def test_zai_complete_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(P.ProviderError, match="ZAI_API_KEY"):
        ZaiProvider("glm-5.2").complete([], [])


def test_zai_ignores_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "zai-test")
    seen: dict[str, object] = {}

    def fake_post(url, headers, payload, timeout=120):  # type: ignore[no-untyped-def]
        seen["url"] = url
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(P, "_http_post_json", fake_post)
    resp = ZaiProvider("glm-5.2").complete([], [], effort="high")
    assert "chat/completions" in seen["url"]
    assert "responses" not in str(seen["url"])
    assert resp.content == "ok"


def test_providers_do_not_stream_yet() -> None:
    assert get_provider("mock:synthetic").supports_streaming is False
    assert OpenAIProvider("gpt-4o").supports_streaming is False
    assert AnthropicProvider("claude-opus-4-8").supports_streaming is False
    assert ZaiProvider("glm-5.2").supports_streaming is False
