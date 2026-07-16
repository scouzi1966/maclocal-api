"""Tests for optional API write-through."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, ClassVar

import pytest

from harness.persistence import maybe_post_run_summary


class _Handler(BaseHTTPRequestHandler):
    received: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/api/runs":
            _Handler.received.append(json.loads(body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def api_server() -> str:
    _Handler.received.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_write_through_skipped_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VULCANBENCH_API_BASE", raising=False)
    assert maybe_post_run_summary({"run_id": "x"}) is False


def test_write_through_posts_summary(api_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VULCANBENCH_API_BASE", api_server)
    monkeypatch.setenv("VULCANBENCH_API_TOKEN", "secret")
    summary = {"run_id": "t-abc", "task_id": "hello-world", "scores": {"total": 1.0}}
    assert maybe_post_run_summary(summary) is True
    assert _Handler.received[0]["run_id"] == "t-abc"


def test_write_through_requires_token(api_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VULCANBENCH_API_BASE", api_server)
    monkeypatch.delenv("VULCANBENCH_API_TOKEN", raising=False)
    assert maybe_post_run_summary({"run_id": "x"}) is False
