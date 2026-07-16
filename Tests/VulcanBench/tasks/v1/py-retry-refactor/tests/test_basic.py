"""Invariants that hold before and after the refactor (regression guard)."""
import pytest

from resilient import DbClient, HttpClient


class Recorder:
    def __init__(self):
        self.delays = []

    def __call__(self, delay):
        self.delays.append(delay)


def flaky(n_failures, result="ok"):
    state = {"calls": 0}

    def op():
        state["calls"] += 1
        if state["calls"] <= n_failures:
            raise RuntimeError("transient")
        return result

    op.state = state
    return op


def always_fails():
    def op():
        raise RuntimeError("permanent")

    return op


def test_http_first_try_success_no_sleep():
    rec = Recorder()
    client = HttpClient(max_attempts=3, base_delay=1, max_delay=5, sleep=rec)
    op = flaky(0, "v")
    assert client.request_with_retry(op) == "v"
    assert rec.delays == []


def test_db_first_try_success_no_sleep():
    rec = Recorder()
    client = DbClient(max_attempts=3, base_delay=1, max_delay=5, sleep=rec)
    op = flaky(0, "v")
    assert client.query_with_retry(op) == "v"
    assert rec.delays == []


def test_http_runs_every_attempt():
    op = flaky(2, "v")
    client = HttpClient(max_attempts=3, base_delay=1, max_delay=5, sleep=Recorder())
    assert client.request_with_retry(op) == "v"
    assert op.state["calls"] == 3


def test_db_caps_delay():
    # base_delay grows past max_delay, so every recorded delay must equal the cap.
    rec = Recorder()
    client = DbClient(max_attempts=5, base_delay=10, max_delay=5, sleep=rec)
    with pytest.raises(RuntimeError):
        client.query_with_retry(always_fails())
    assert rec.delays  # at least one retry slept
    assert all(d == 5 for d in rec.delays)
