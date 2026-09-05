"""Deterministic clock checks and delayed HTTP/SSE fixtures; no model required."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ttft', ROOT / 'Scripts/measure-sse-ttft.py')
ttft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ttft)


class ClockTests(unittest.TestCase):
    def test_role_empty_and_malformed_events_do_not_start_clock(self):
        lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                 'data: not json\n',
                 'data: {"choices":[{"delta":{"content":""}}]}\n',
                 'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}\n',
                 'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                 'data: [DONE]\n']
        times = iter([2, 3, 4, 7, 9, 12])
        self.assertEqual(ttft.measure(iter(lines), 1_000_000,
                                     lambda: next(times) * 1_000_000), 6)
        self.assertEqual(list(times), [9, 12])  # Stop timing after first token.

    def test_empty_stream_does_not_report_zero_ttft(self):
        self.assertIsNone(ttft.measure(['data: [DONE]\n'], 0, lambda: 1))


class DelayedSSETests(unittest.TestCase):
    def run_fixture(self, mode, token_delay=0.2, tail_delay=0.3, request_timeout=10):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send_json(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.send_json({'data': [{'id': 'fixture'}], 'models': []})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not body.get('stream'):
                    self.send_json({'choices': [{'message': {'content': 'OK'}}],
                                    'usage': {'completion_tokens': 50}})
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()

                def event(payload):
                    data = ('data: ' + payload + '\n\n').encode()
                    self.wfile.write(f'{len(data):x}\r\n'.encode() + data + b'\r\n')
                    self.wfile.flush()

                try:
                    event('{"choices":[{"delta":{"role":"assistant"}}]}')
                    time.sleep(token_delay)
                    if mode != 'role-only':
                        key = 'reasoning_content' if mode == 'reasoning' else 'content'
                        event(json.dumps({'choices': [{'delta': {key: 'first'}}]}))
                    if mode == 'disconnect':
                        self.close_connection = True
                        return  # Missing final chunk: transport failure after first token.
                    time.sleep(tail_delay)
                    event('[DONE]')
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        parent = ROOT / '.build' / 'assertion-ttft-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=parent) as directory:
                env = dict(os.environ, MACAFM_SWIFT_TEST_SKIP='1',
                           AFM_ASSERTIONS_REPORT_DIR=directory,
                           AFM_ASSERTIONS_WORK_ROOT=directory + '/work',
                           AFM_ASSERTIONS_REQUEST_TIMEOUT=str(request_timeout))
                result = subprocess.run(
                    ['bash', str(ROOT / 'Scripts/test-assertions.sh'), '--model', 'fixture',
                     '--bin', '/usr/bin/true', '--section', '9', '--tier', 'full',
                     '--port', str(server.server_port)],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
                reports = list(Path(directory).glob('*.jsonl'))
                self.assertEqual(len(reports), 1, result.stdout + result.stderr)
                rows = [json.loads(line) for line in reports[0].read_text().splitlines()]
                return result, rows
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_delay_is_measured_live_and_tail_is_excluded(self):
        for mode in ['content', 'reasoning']:
            with self.subTest(mode=mode):
                result, rows = self.run_fixture(mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                row = next(row for row in rows if row['name'].startswith('TTFT'))
                elapsed = int(re.search(r'\((\d+)ms\)', row['name'])[1])
                self.assertGreaterEqual(elapsed, 190)  # Immediate role event is ignored.
                self.assertGreaterEqual(row['duration_ms'] - elapsed, 250)

    def test_delayed_first_token_fails_five_second_threshold(self):
        result, rows = self.run_fixture('content', token_delay=5.1, tail_delay=0)
        self.assertNotEqual(result.returncode, 0)
        row = next(row for row in rows if row['name'].startswith('TTFT'))
        self.assertEqual(row['status'], 'FAIL')
        self.assertGreaterEqual(int(row['actual'].removeprefix('FAIL: ')), 5000)

    def test_role_only_stream_fails_without_claiming_first_token(self):
        result, rows = self.run_fixture('role-only', token_delay=0, tail_delay=0)
        self.assertNotEqual(result.returncode, 0)
        row = next(row for row in rows if row['name'].startswith('TTFT'))
        self.assertEqual(row['status'], 'FAIL')
        self.assertIn('no content or reasoning token', row['actual'])

    def test_disconnect_after_first_token_retains_transport_failure(self):
        result, rows = self.run_fixture('disconnect', token_delay=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(rows[-1]['status'], 'FAIL')
        self.assertIn('engine unavailable', rows[-1]['actual'])
        self.assertEqual(len(rows), 3)

    def test_live_timeout_after_first_token_is_not_a_ttft_pass(self):
        result, rows = self.run_fixture('content', token_delay=0, tail_delay=1.2,
                                        request_timeout=1)
        self.assertNotEqual(result.returncode, 0)
        row = next(row for row in rows if row['name'].startswith('TTFT'))
        self.assertEqual(row['status'], 'FAIL')
        self.assertIn('request timeout', row['actual'])
        self.assertEqual(len(rows), 6)  # A live server can continue other assertions.


if __name__ == '__main__':
    unittest.main()
