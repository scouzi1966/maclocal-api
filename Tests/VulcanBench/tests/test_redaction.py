"""Tests for secret redaction and field-size caps on published artifacts."""

from __future__ import annotations

import pytest

from harness.redaction import MAX_FIELD_CHARS, REDACTED, redact, sanitize


def test_redacts_openai_key() -> None:
    out = redact("token=sk-abcdefghijklmnopqrstuvwxyz0123 done")
    assert "sk-abcdefghij" not in out
    assert REDACTED in out


def test_redacts_github_token() -> None:
    assert "ghp_" not in redact("ghp_0123456789abcdefghijklmnopqrstuvwx")
    assert "github_pat_" not in redact("github_pat_0123456789abcdefghij_ABCDEFG")


def test_redacts_bearer() -> None:
    out = redact("Authorization: Bearer abcdef0123456789abcdef0123")
    assert "abcdef0123456789" not in out
    assert "Bearer" in out  # only the token is masked


def test_redacts_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "supersecretvalue123")
    out = redact("the key is supersecretvalue123 in the log")
    assert "supersecretvalue123" not in out
    assert REDACTED in out


def test_redacts_zai_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "zai_supersecretvalue123456")
    out = redact("the key is zai_supersecretvalue123456 in the log")
    assert "zai_supersecretvalue123456" not in out
    assert REDACTED in out


def test_redacts_api_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VULCANBENCH_API_TOKEN", "tok_supersecretvalue123456")
    out = redact("auth tok_supersecretvalue123456 end")
    assert "tok_supersecretvalue123456" not in out


def test_short_env_value_not_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid masking innocuous short values that happen to match a secret env.
    monkeypatch.setenv("GH_TOKEN", "abc")
    assert redact("abc def") == "abc def"


def test_plain_text_untouched() -> None:
    assert redact("a normal log line with no secrets") == "a normal log line with no secrets"
    assert redact("") == ""


def test_sanitize_recurses_and_caps() -> None:
    big = "x" * (MAX_FIELD_CHARS + 500)
    data = {
        "stdout": big,
        "nested": {"key": "sk-abcdefghijklmnopqrstuvwxyz0123"},
        "list": ["ghp_0123456789abcdefghijklmnopqrstuvwx", 42, None],
    }
    out = sanitize(data)
    assert out["stdout"].endswith("chars]") and len(out["stdout"]) < len(big)
    assert out["nested"]["key"] == REDACTED
    assert out["list"][0] == REDACTED
    assert out["list"][1] == 42  # non-strings pass through
    assert out["list"][2] is None
