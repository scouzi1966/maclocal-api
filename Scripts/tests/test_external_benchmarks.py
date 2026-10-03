#!/usr/bin/env python3
"""Offline completeness regression tests; never start a model."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'test-external-benchmarks.py'
spec = importlib.util.spec_from_file_location('external_benchmarks', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContextCompletenessTests(unittest.TestCase):
    def setUp(self):
        work = SCRIPT.parent.parent / '.build/external-benchmark-tests'
        work.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / 'benchmark_openai_fixture'
        self.results.mkdir()
        log = []
        for context in module.CONTEXTS:
            log.append(f'Benchmarking {context}.txt ...')
            for number in range(1, module.RUNS + 1):
                log.append(f'  Run {number}/{module.RUNS}...\n  generation_tps: 30.0\n  Completion tokens: 128')
            (self.results / f'response_{context}.txt').write_text('fixture response')
        (self.root / 'context.log').write_text('\n'.join(log))
        with (self.results / 'benchmark_results.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['context_size', 'generation_tps', 'prompt_tps_latency_adjusted', 'host_memory_gb'])
            writer.writerows([context, 30, 300, 40] for context in module.CONTEXTS)

    def test_complete_run_passes(self):
        self.assertTrue(module.validate_context(self.root)['passed'])

    def test_python_symlink_preserves_virtualenv_entrypoint(self):
        base = self.root / 'base-python'
        base.touch()
        interpreter = self.root / 'venv-python'
        interpreter.symlink_to(base)
        self.assertEqual(module.existing_path(interpreter), interpreter)

    def test_missing_input_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            module.existing_path(self.root / 'missing-python')

    def test_exit_zero_with_missing_trial_fails(self):
        path = self.root / 'context.log'
        path.write_text(path.read_text().replace('  Run 2/2...\n  generation_tps: 30.0\n  Completion tokens: 128', '', 1))
        self.assertFalse(module.validate_context(self.root)['passed'])

    def test_missing_saved_response_fails(self):
        (self.results / 'response_32k.txt').unlink()
        self.assertFalse(module.validate_context(self.root)['passed'])

    def test_nonfinite_metrics_fail(self):
        path = self.results / 'benchmark_results.csv'
        path.write_text(path.read_text().replace(',30,', ',nan,', 1))
        self.assertFalse(module.validate_context(self.root)['passed'])


class ServerConfigurationTests(unittest.TestCase):
    def arguments(self, **overrides):
        return SimpleNamespace(**dict(dict(binary=Path('/candidate/afm'), model=Path('/model'),
            port=9999, mtp=False, mtp_depth=3, prefill_step_size=None,
            concurrent_capacity=1), **overrides))

    def test_mtp_and_prefill_reach_both_suites(self):
        args = self.arguments(mtp=True, mtp_depth=4, prefill_step_size=8192)
        for phase in ('llmprobe', 'context'):
            command = module.server_command(args, phase)
            self.assertIn('--mtp', command)
            self.assertEqual(command[command.index('--mtp-depth') + 1], '4')
            self.assertEqual(command[command.index('--prefill-step-size') + 1], '8192')
            self.assertEqual(command[command.index('--port') + 1], '9999')

    def test_default_stays_non_speculative(self):
        command = module.server_command(self.arguments(), 'context')
        self.assertNotIn('--mtp', command)
        self.assertNotIn('--mtp-depth', command)
        self.assertNotIn('--prefill-step-size', command)
        self.assertIn('--no-think', command)
        self.assertNotIn('--concurrent', command)

    def test_server_capacity_reaches_both_suites(self):
        for phase in ('llmprobe', 'context'):
            command = module.server_command(self.arguments(concurrent_capacity=2), phase)
            self.assertEqual(command[command.index('--concurrent') + 1], '2')


class ServerOwnershipTests(unittest.TestCase):
    def test_own_listener_is_ready(self):
        with patch.object(module.subprocess, 'run', return_value=Mock(stdout='123\n')):
            self.assertTrue(module.listener_owned_by(Mock(pid=123), 9999))

    def test_no_listener_is_not_ready(self):
        with patch.object(module.subprocess, 'run', return_value=Mock(stdout='')):
            self.assertFalse(module.listener_owned_by(Mock(pid=123), 9999))

    def test_foreign_listener_is_rejected(self):
        with patch.object(module.subprocess, 'run', return_value=Mock(stdout='456\n')):
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                module.listener_owned_by(Mock(pid=123), 9999)


class ShellServerOwnershipTests(unittest.TestCase):
    def check_listener(self, relative_path, function, shell, owner):
        source = (SCRIPT.parent.parent / relative_path).read_text()
        function_source = function + '() {' + source.split(function + '() {', 1)[1].split('\n}\n', 1)[0] + '\n}'
        program = f"""
PORT=9999
SERVER_PID=123
TIMEOUT_LOAD=1
port=9999
server_pid=123
load_timeout=1
lsof() {{ echo {owner}; }}
kill() {{ return 0; }}
curl() {{ echo HTTP_CALLED > /dev/fd/3; return 0; }}
{function_source}
{function}
"""
        result = subprocess.run([shell, '-c', 'exec 3>&1\n' + program], capture_output=True, text=True)
        if owner == 123:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('HTTP_CALLED', result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('HTTP_CALLED', result.stdout)

    def test_comprehensive_own_listener(self):
        self.check_listener('Scripts/mlx-model-test.sh', 'wait_for_server', 'bash', 123)

    def test_comprehensive_foreign_listener(self):
        self.check_listener('Scripts/mlx-model-test.sh', 'wait_for_server', 'bash', 456)

    def test_promptfoo_own_listener(self):
        self.check_listener('Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh', 'wait_for_health', 'zsh', 123)

    def test_promptfoo_foreign_listener(self):
        self.check_listener('Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh', 'wait_for_health', 'zsh', 456)


if __name__ == '__main__':
    unittest.main()
