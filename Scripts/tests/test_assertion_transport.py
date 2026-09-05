"""CPU-only end-to-end assertion runner tests using a fake curl executable."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FAKE_CURL = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ['FIXTURE_CALL_LOG'])
with log.open('a') as handle:
    handle.write(json.dumps(args) + '\n')
mode = os.environ['FIXTURE_MODE']
is_models = any(arg.endswith('/v1/models') for arg in args)
calls = log.read_text().splitlines()
if mode == 'preflight' or (mode == 'capabilities' and len(calls) == 2):
    sys.exit(7)
if mode == 'capabilities-http-error' and len(calls) == 2:
    sys.exit(22)
if mode == 'timeout-dead' and is_models and len(calls) > 2:
    sys.exit(7)
if is_models:
    print(json.dumps({'data': [{'id': 'fixture'}], 'models': []}))
    sys.exit(0)
if mode == 'http-error':
    sys.exit(22)
if mode == 'drop' or (mode == 'stream-drop' and '-N' in args):
    sys.exit(52)
if mode in ('timeout', 'timeout-dead'):
    sys.exit(28)
print(json.dumps({'choices': [{'message': {'content': 'AFM_PREFIX'},
                              'finish_reason': 'stop'}]}))
'''


class AssertionTransportTests(unittest.TestCase):
    def run_fixture(self, mode, section='1', tier='smoke'):
        parent = ROOT / '.build' / 'assertion-transport-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            work = Path(directory)
            fake_bin = work / 'bin'
            fake_bin.mkdir()
            curl = fake_bin / 'curl'
            curl.write_text(FAKE_CURL)
            curl.chmod(0o755)
            log = work / 'calls.jsonl'
            reports = work / 'reports'
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
                       MACAFM_SWIFT_TEST_SKIP='1', AFM_ASSERTIONS_REPORT_DIR=str(reports),
                       AFM_ASSERTIONS_WORK_ROOT=str(work / 'scratch'),
                       FIXTURE_MODE=mode, FIXTURE_CALL_LOG=str(log))
            result = subprocess.run(
                ['bash', str(ROOT / 'Scripts/test-assertions.sh'), '--model', 'fixture',
                 '--bin', '/usr/bin/true', '--section', section, '--tier', tier,
                 '--grammar-constraints'],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            jsonl = list(reports.glob('*.jsonl'))
            html = list(reports.glob('*.html'))
            self.assertEqual(len(jsonl), 1, result.stdout + result.stderr)
            self.assertEqual(len(html), 1, result.stdout + result.stderr)
            rows = [json.loads(line) for line in jsonl[0].read_text().splitlines()]
            self.assertIn('</html>', html[0].read_text())
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, rows, calls

    def assert_aborted(self, mode, section='1', tier='smoke'):
        result, rows, calls = self.run_fixture(mode, section, tier)
        self.assertNotEqual(result.returncode, 0)
        failures = [row for row in rows if 'engine unavailable' in row['actual']]
        self.assertEqual(len(failures), 1, result.stdout + result.stderr)
        self.assertEqual(rows[-1], failures[0])
        self.assertEqual(rows[-1]['status'], 'FAIL')
        self.assertEqual(rows[-1]['classification'], 'engine/runtime likely')
        self.assertNotEqual(rows[0]['status'], 'FAIL')  # Retain completed evidence.
        return rows, calls

    def test_json_transport_loss_preserves_partial_report_and_stops(self):
        rows, calls = self.assert_aborted('drop')
        self.assertEqual(len(calls), 4)
        self.assertEqual([row['status'] for row in rows], ['PASS'] * 3 + ['FAIL'])

    def test_header_transport_loss_cannot_pass_absent_header(self):
        rows, calls = self.assert_aborted('drop', '14')
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(rows), 3)

    def test_failed_thinking_probe_cannot_be_capability_skip(self):
        rows, calls = self.assert_aborted('drop', '4')
        self.assertFalse(any(row['status'] == 'SKIP' for row in rows))
        self.assertEqual(len(calls), 3)

    def test_stream_transport_loss_stops_at_failed_stream(self):
        _, calls = self.assert_aborted('stream-drop', '2', 'standard')
        self.assertIn('-N', calls[-1])
        self.assertEqual(sum('-N' in call for call in calls), 1)

    def test_direct_http_status_request_preserves_report_on_errexit(self):
        _, calls = self.assert_aborted('drop', '8')
        self.assertEqual(len(calls), 3)

    def test_concurrent_requests_share_transport_failure(self):
        _, calls = self.assert_aborted('drop', '7', 'standard')
        self.assertLessEqual(len(calls), 4)

    def test_preflight_and_capability_fetch_fail_closed(self):
        for mode in ['preflight', 'capabilities']:
            with self.subTest(mode=mode):
                self.assert_aborted(mode)

    def test_timeout_is_failed_transport_not_capability(self):
        rows, _ = self.assert_aborted('timeout-dead', '4')
        self.assertIn('curl exit 28', rows[-1]['actual'])

    def test_live_timeout_fails_assertion_without_claiming_server_death(self):
        result, rows, calls = self.run_fixture('timeout', '4')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(rows[-1]['status'], 'FAIL')
        self.assertIn('server still reachable', rows[-1]['actual'])
        self.assertNotIn('engine unavailable', rows[-1]['actual'])
        self.assertTrue(any(arg.endswith('/v1/models') for arg in calls[-1]))

    def test_http_error_response_remains_an_assertion_failure(self):
        result, rows, calls = self.run_fixture('http-error')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any('engine unavailable' in row['actual'] for row in rows))
        self.assertEqual(len(calls), 4)

    def test_failed_capability_endpoint_is_not_unsupported_model(self):
        result, rows, calls = self.run_fixture('capabilities-http-error')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(rows[-1]['status'], 'FAIL')
        self.assertIn('HTTP capability discovery failed', rows[-1]['actual'])
        self.assertFalse(any(row['status'] == 'SKIP' for row in rows))
        self.assertEqual(len(calls), 2)

    def test_success_adds_no_health_or_inference_requests(self):
        result, rows, calls = self.run_fixture('success')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(all(row['status'] == 'PASS' for row in rows))
        self.assertEqual(len(calls), 4)
        self.assertEqual(sum(any(arg.endswith('/v1/chat/completions') for arg in call)
                             for call in calls), 1)


if __name__ == '__main__':
    unittest.main()
