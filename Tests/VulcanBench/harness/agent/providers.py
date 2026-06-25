"""Generic LLM provider interface for the VulcanBench agent loop.

The goal is to evaluate *any* model behind a uniform contract. A provider takes
a list of chat messages plus the OpenAI-style tool schemas and returns an
:class:`LLMResponse` (assistant text, requested tool calls, and token usage).

Providers implemented:

- ``mock:<name>``      Deterministic, offline. Solves ``hello-world`` and powers
                       tests without network or API keys.
- ``openai:<model>``   OpenAI Chat Completions API by default, and the Responses
                       API when reasoning effort is supplied.
- ``anthropic:<model>`` Anthropic Messages API.
- ``zai:<model>``      Z.ai (Zhipu) OpenAI-compatible Chat Completions API.

Only the Python standard library is used for HTTP so the harness stays
dependency-light; ``tenacity`` provides retry/backoff.

Usage::

    provider = get_provider("openai:gpt-4o")
    resp = provider.complete(messages, tools)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response."""


class ToolInvocation(BaseModel):
    """A single tool call requested by the model."""

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    """Uniform response shape across all providers."""

    content: str | None = None
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Base class for all model backends."""

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def spec(self) -> str:
        return f"{self.name}:{self.model}"

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider id, e.g. ``openai``."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        """Return the next assistant turn given the conversation and tool schemas."""

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider implements token streaming (v1: always False)."""
        return False


def _http_post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 120
) -> dict[str, Any]:
    """POST JSON and parse the JSON response using only the stdlib."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {e.code} from {url}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise ProviderError(f"network error calling {url}: {e.reason}") from e


def _http_timeout(timeout_s: float | None) -> float:
    if timeout_s is None:
        return 120
    if timeout_s <= 0:
        raise ProviderError("run budget exhausted before provider call")
    return timeout_s


_RETRY = retry(
    retry=retry_if_exception_type(ProviderError),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)


class OpenAIProvider(LLMProvider):
    """OpenAI provider.

    Uses Chat Completions for the legacy/no-effort path. When ``effort`` is
    supplied, switches to the Responses API so reasoning effort can be passed in
    the official ``reasoning.effort`` field.
    """

    @property
    def name(self) -> str:
        return "openai"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        timeout = _http_timeout(timeout_s)
        if timeout_s is not None:
            return self._complete_once(messages, tools, timeout, effort)
        return self._complete_with_retry(messages, tools, timeout, effort)

    @_RETRY
    def _complete_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
        effort: str | None,
    ) -> LLMResponse:
        return self._complete_once(messages, tools, timeout, effort)

    def _complete_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
        effort: str | None,
    ) -> LLMResponse:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if effort is not None:
            return self._responses_complete(base, api_key, messages, tools, timeout, effort)

        return _chat_completions_complete(
            base,
            api_key,
            self.model,
            messages,
            tools,
            timeout,
            temperature=None if _openai_omits_chat_sampling(self.model) else 0,
        )

    def _responses_complete(
        self,
        base: str,
        api_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
        effort: str,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": _to_responses_input(messages),
            "reasoning": {"effort": effort},
        }
        if tools:
            payload["tools"] = [_openai_tool_to_responses(t) for t in tools]
            payload["tool_choice"] = "auto"
        body = _http_post_json(
            f"{base}/responses",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload,
            timeout=timeout,
        )
        return _parse_responses_body(body)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API.

    Reasoning effort maps to the API's ``output_config.effort`` field. No
    sampling parameters are sent: ``temperature``/``top_p``/``top_k`` are
    rejected with a 400 by Opus 4.7 and newer models.
    """

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        timeout = _http_timeout(timeout_s)
        if timeout_s is not None:
            return self._complete_once(messages, tools, timeout, effort)
        return self._complete_with_retry(messages, tools, timeout, effort)

    @_RETRY
    def _complete_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
        effort: str | None,
    ) -> LLMResponse:
        return self._complete_once(messages, tools, timeout, effort)

    def _complete_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
        effort: str | None,
    ) -> LLMResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        system, converted = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": converted,
        }
        if effort is not None:
            payload["output_config"] = {"effort": effort}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_openai_tool_to_anthropic(t) for t in tools]
        body = _http_post_json(
            f"{base}/v1/messages",
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
            timeout=timeout,
        )
        content_text: list[str] = []
        tool_calls: list[ToolInvocation] = []
        for block in body.get("content", []):
            if block.get("type") == "text":
                content_text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolInvocation(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )
        usage = body.get("usage", {})
        return LLMResponse(
            content="\n".join(content_text) or None,
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
            ),
            raw=body,
        )


class MockProvider(LLMProvider):
    """Deterministic, offline provider for tests and demos.

    It runs a tiny scripted policy: read the issue, then create the file the
    ``hello-world`` task asks for, then signal completion. This lets the *real*
    agent loop be exercised end-to-end with no network or API key.
    """

    @property
    def name(self) -> str:
        return "mock"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        del timeout_s, effort
        # Judge requests (human_like ensemble) carry a sentinel: answer with a
        # fixed JSON score so the metric is deterministic offline.
        if any("VULCANBENCH_JUDGE" in str(m.get("content", "")) for m in messages):
            return LLMResponse(
                content='{"score": 80, "rationale": "mock judge"}',
                usage=TokenUsage(prompt_tokens=120, completion_tokens=15),
            )
        # Decide the next action purely from what has happened so far.
        called = [m for m in messages if m.get("role") == "tool"]
        steps = len(called)
        usage = TokenUsage(prompt_tokens=50 + steps * 20, completion_tokens=20)
        if steps == 0:
            return LLMResponse(
                content="Reading the issue.",
                tool_calls=[
                    ToolInvocation(id="c1", name="read_file", arguments={"path": "issue.md"})
                ],
                usage=usage,
            )
        if steps == 1:
            return LLMResponse(
                content="Creating hello.py with the required output.",
                tool_calls=[
                    ToolInvocation(
                        id="c2",
                        name="edit_file",
                        arguments={
                            "path": "hello.py",
                            "old_string": "",
                            "new_string": 'print("hello from vulcanbench")\n',
                        },
                    )
                ],
                usage=usage,
            )
        if steps == 2:
            return LLMResponse(
                content="Running tests.",
                tool_calls=[ToolInvocation(id="c3", name="run_tests", arguments={})],
                usage=usage,
            )
        return LLMResponse(content="FINISH: implemented and verified.", usage=usage)


class ZaiProvider(LLMProvider):
    """Z.ai (Zhipu) OpenAI-compatible Chat Completions API.

    Uses ``/chat/completions`` only. Reasoning effort is not supported; pass
    ``--effort`` for metadata recording but it is ignored at the API layer.
    """

    @property
    def name(self) -> str:
        return "zai"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        del effort
        timeout = _http_timeout(timeout_s)
        if timeout_s is not None:
            return self._complete_once(messages, tools, timeout)
        return self._complete_with_retry(messages, tools, timeout)

    @_RETRY
    def _complete_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
    ) -> LLMResponse:
        return self._complete_once(messages, tools, timeout)

    def _complete_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout: float,
    ) -> LLMResponse:
        api_key = os.environ.get("ZAI_API_KEY")
        if not api_key:
            raise ProviderError("ZAI_API_KEY is not set")
        base = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4").rstrip("/")
        return _chat_completions_complete(base, api_key, self.model, messages, tools, timeout)


def _loads_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split out the system prompt and convert OpenAI-style turns to Anthropic.

    Tool results (OpenAI ``role: tool``) become Anthropic ``tool_result`` blocks
    on a ``user`` turn keyed by ``tool_use_id``.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(str(m.get("content", "")))
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": str(m.get("content", "")),
                        }
                    ],
                }
            )
        elif role == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": str(m["content"])})
            for tc in m["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "input": _loads_args(tc["function"].get("arguments", "{}")),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": role or "user", "content": str(m.get("content", ""))})
    return "\n".join(system_parts), out


def _to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the harness' chat-style transcript to Responses API input items."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "system":
            out.append({"role": "developer", "content": content})
        elif role in {"user", "assistant"}:
            if content:
                out.append({"role": role, "content": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    }
                )
        elif role == "tool":
            out.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id", ""),
                    "output": content,
                }
            )
        else:
            out.append({"role": "user", "content": content})
    return out


def _openai_tool_to_responses(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool["function"]
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _parse_responses_body(body: dict[str, Any]) -> LLMResponse:
    content_parts: list[str] = []
    tool_calls: list[ToolInvocation] = []
    for item in body.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for block in item.get("content") or []:
                if block.get("type") in {"output_text", "text"}:
                    content_parts.append(block.get("text", ""))
        elif item_type == "function_call":
            tool_calls.append(
                ToolInvocation(
                    id=item.get("call_id") or item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=_loads_args(item.get("arguments", "{}")),
                )
            )
    if body.get("output_text"):
        content_parts.append(str(body["output_text"]))

    usage = body.get("usage", {})
    return LLMResponse(
        content="\n".join(part for part in content_parts if part) or None,
        tool_calls=tool_calls,
        usage=TokenUsage(
            prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
        ),
        raw=body,
    )


def _parse_chat_completions_response(body: dict[str, Any]) -> LLMResponse:
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    tool_calls = [
        ToolInvocation(
            id=tc.get("id", ""),
            name=tc["function"]["name"],
            arguments=_loads_args(tc["function"].get("arguments", "{}")),
        )
        for tc in (msg.get("tool_calls") or [])
    ]
    usage = body.get("usage", {})
    return LLMResponse(
        content=msg.get("content"),
        tool_calls=tool_calls,
        usage=TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        ),
        raw=body,
    )


def _openai_omits_chat_sampling(model: str) -> bool:
    """True when Chat Completions rejects non-default ``temperature`` (GPT-5, o-series)."""
    name = model.strip().lower()
    return name.startswith(("gpt-5", "o1", "o2", "o3", "o4"))


def _chat_completions_complete(
    base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float,
    temperature: float | None = 0,
) -> LLMResponse:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = _http_post_json(
        f"{base}/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
        timeout=timeout,
    )
    return _parse_chat_completions_response(body)


def _openai_tool_to_anthropic(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "zai": ZaiProvider,
    "mock": MockProvider,
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:model`` into its parts.

    >>> parse_model_spec("openai:gpt-4o")
    ('openai', 'gpt-4o')
    """
    if ":" not in spec:
        raise ValueError(f"model spec must be 'provider:model', got {spec!r}")
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(f"model spec must be 'provider:model', got {spec!r}")
    return provider, model


def get_provider(spec: str) -> LLMProvider:
    """Construct a provider from a ``provider:model`` spec."""
    provider, model = parse_model_spec(spec)
    if provider not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; known: {known}")
    return _PROVIDERS[provider](model)
