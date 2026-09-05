"""Execute the production final shell gate against CPU-only JSONL fixtures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/'Scripts/mlx-model-test.sh').read_text()
MARKER = '# ── Final recorded outcome gate'
GATE = SOURCE[SOURCE.index(MARKER):]


class FinalStatusTests(unittest.TestCase):
    def run_gate(self, records, initial_status=0):
        parent = ROOT/'.build/comprehensive-status-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            path = Path(directory)/'results.jsonl'
            if records is not None:
                path.write_text(records if isinstance(records, str) else
                                '\n'.join(json.dumps(record) for record in records) + '\n')
                before = hashlib.sha256(path.read_bytes()).hexdigest()
            result = subprocess.run(['bash', '-u', '-o', 'pipefail', '-c', GATE],
                                    env=dict(os.environ, RESULTS_FILE=str(path), OVERALL_STATUS=str(initial_status)),
                                    text=True, capture_output=True, timeout=10)
            if records is not None:
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            return result

    def test_transport_ok_does_not_hide_failed_assertions(self):
        result = self.run_gate([{'status': 'OK', 'assertion_status': 'fail', 'overall_status': 'fail'}])
        self.assertEqual(result.returncode, 1)
        self.assertIn('0 passed, 1 failed, 0 skipped', result.stdout)

    def test_each_failure_signal_overrides_success_and_skip(self):
        for field, value in [('status', 'FAIL'), ('transport_status', 'error'),
                             ('assertion_status', 'FAIL'), ('overall_status', ' fail '),
                             ('assertion_failures', ['wrong arguments']), ('error', 'connection lost')]:
            for status in ('OK', 'SKIP'):
                with self.subTest(field=field, status=status):
                    record = {'status': status, 'overall_status': 'pass', field: value}
                    self.assertEqual(self.run_gate([record]).returncode, 1)

    def test_legacy_success_and_normalized_success_pass(self):
        result = self.run_gate([{'status': 'OK'}, {'status': ' ok ', 'transport_status': 'PASS',
                                                'assertion_status': 'not_configured', 'overall_status': 'PASS'}])
        self.assertEqual(result.returncode, 0)
        self.assertIn('2 passed, 0 failed, 0 skipped', result.stdout)

    def test_capability_skip_is_not_a_pass_or_failure(self):
        result = self.run_gate([{'status': 'SKIP', 'transport_status': 'not_run',
                                'assertion_status': 'not_run', 'overall_status': 'skip'},
                               {'status': 'skip'}])
        self.assertEqual(result.returncode, 0)
        self.assertIn('0 passed, 0 failed, 2 skipped (not executed; not passes)', result.stdout)

    def test_metadata_excluded_and_mixed_results_counted_once(self):
        result = self.run_gate([{'_meta': True}, {'status': 'OK'}, {'status': 'SKIP'},
                               {'status': 'FAIL', 'overall_status': 'fail', 'error': 'load failed'}])
        self.assertEqual(result.returncode, 1)
        self.assertIn('1 passed, 1 failed, 1 skipped', result.stdout)

    def test_invalid_records_and_unknown_status_fail_closed(self):
        for record in (None, [], 'text', {}, {'status': 'unknown'},
                       {'status': 'OK', 'assertion_status': 'unknown'},
                       {'status': 'OK', 'transport_status': 'not_run'}):
            with self.subTest(record=record):
                self.assertEqual(self.run_gate([record]).returncode, 1)

    def test_prior_failure_is_not_erased_by_passing_results(self):
        self.assertEqual(self.run_gate([{'status': 'OK'}], initial_status=7).returncode, 7)

    def test_missing_results_fail_closed(self):
        self.assertEqual(self.run_gate(None).returncode, 1)

    def test_malformed_json_does_not_hide_following_records(self):
        result = self.run_gate('{broken json}\n{"status":"OK"}\n')
        self.assertEqual(result.returncode, 1)
        self.assertIn('1 passed, 1 failed, 0 skipped', result.stdout)


if __name__ == '__main__':
    unittest.main()
