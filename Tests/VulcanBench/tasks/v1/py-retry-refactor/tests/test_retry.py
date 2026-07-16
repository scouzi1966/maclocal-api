"""Unified retry behavior — fails on the starting repo, passes after consolidation."""
import pytest

from resilient import DbClient, HttpClient


class Recorder:
    """Records the backoff delays instead of really sleeping."""

    def __init__(self):
        self.delays = []

    def __call__(self, delay):
        self.delays.append(delay)


def flaky(n_failures, result="ok"):
    """An operation that raises the first ``n_failures`` times, then returns ``result``."""
    state = {"calls": 0}

    def op():
        state["calls"] += 1
        if state["calls"] <= n_failures:
            raise RuntimeError(f"transient failure {state['calls']}")
        return result

    op.state = state
    return op


def always_fails():
    state = {"calls": 0}

    def op():
        state["calls"] += 1
        raise RuntimeError("permanent failure")

    op.state = state
    return op


def test_http_caps_backoff_delay():
    rec = Recorder()
    client = HttpClient(max_attempts=5, base_delay=1, max_delay=5, sleep=rec)
    op = always_fails()
    with pytest.raises(RuntimeError):
        client.request_with_retry(op)
    # 5 attempts -> 4 sleeps; 1,2,4,8 capped at max_delay=5 -> 1,2,4,5.
    assert rec.delays == [1, 2, 4, 5]


def test_db_runs_all_attempts():
    client = DbClient(max_attempts=3, base_delay=1, max_delay=5, sleep=Recorder())
    op = flaky(2, "rows")  # fails twice, succeeds on the 3rd attempt
    assert client.query_with_retry(op) == "rows"
    assert op.state["calls"] == 3


def test_http_and_db_behave_identically():
    rec_h, rec_d = Recorder(), Recorder()
    http = HttpClient(max_attempts=4, base_delay=1, max_delay=3, sleep=rec_h)
    db = DbClient(max_attempts=4, base_delay=1, max_delay=3, sleep=rec_d)
    op_h, op_d = always_fails(), always_fails()
    with pytest.raises(RuntimeError):
        http.request_with_retry(op_h)
    with pytest.raises(RuntimeError):
        db.query_with_retry(op_d)
    assert op_h.state["calls"] == op_d.state["calls"] == 4
    assert rec_h.delays == rec_d.delays == [1, 2, 3]
