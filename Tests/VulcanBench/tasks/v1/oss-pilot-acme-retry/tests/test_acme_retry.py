from acme.http import HttpClient
from acme.db import DbClient

class Recorder:
    def __init__(self):
        self.delays = []
    def __call__(self, s):
        self.delays.append(s)

def test_clients_import():
    assert HttpClient and DbClient
def test_retry_attempt_count():
    rec = Recorder()
    c = HttpClient(rec)
    n = {'i': 0}
    def boom():
        n['i'] += 1
        raise RuntimeError('x')
    try:
        c.request_with_retry(boom, max_attempts=3)
    except RuntimeError:
        pass
    assert n['i'] == 3

def test_retry_respects_max_delay():
    rec = Recorder()
    c = DbClient(rec)
    def boom():
        raise RuntimeError('x')
    try:
        c.query_with_retry(boom, max_attempts=4, base_delay=1.0, max_delay=2.0)
    except RuntimeError:
        pass
    assert all(d <= 2.0 for d in rec.delays)
