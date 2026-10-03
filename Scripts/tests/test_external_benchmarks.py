#!/usr/bin/env python3
"""Offline completeness regression tests; never start a model."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
