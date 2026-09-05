"""CPU-only checks for batch failure diagnostics; no server or GPU required."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
import tempfile
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1] / 'feature-mlx-concurrent-batch'
sys.path.insert(0, str(ROOT))
from batch_validation_report import BatchReport


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mixed = load('validate_mixed_workload')
responses = load('validate_responses')
prefix = load('validate_multiturn_prefix')


class AsyncContext:
    def __init__(self, value): self.value = value
    async def __aenter__(self): return self.value
    async def __aexit__(self, *_): return False


def response(text):
    return dict(text=text, completion_tokens=10, prompt_tokens=10, cached_tokens=0,
                pp_tok_s=10, tg_tok_s=10, ttft=0.1, wall_s=1)


class FailureReportingTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_errors_surface_before_output_judgment(self):
        class HTTPErrorResponse:
            def raise_for_status(self): raise RuntimeError('HTTP 503 unavailable')
            @property
            def content(self): raise AssertionError('HTTP error must not become empty output')
        session = SimpleNamespace(post=lambda *_args, **_kwargs: AsyncContext(HTTPErrorResponse()))
        for module in [responses, mixed, prefix]:
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(RuntimeError, 'HTTP 503'):
                    await module.send_request(session, 'fixture')

    async def test_mixed_failures_retain_distinct_evidence_without_blame(self):
        tests = [dict(name=name, prompt='fixture', expected=['needle'], min_tokens=min_tokens)
                 for name, min_tokens in [('timeout', 0), ('garbage', 0), ('short', 20),
                                          ('missing', 0), ('success', 0)]]
        outcomes = [TimeoutError('fixture timeout'), response(''), response('ok'),
                    response('other answer'), response('needle')]
        output = io.StringIO()
        with patch.object(mixed.aiohttp, 'ClientSession', return_value=AsyncContext(object())), \
             patch.object(mixed, 'send_request', AsyncMock(side_effect=outcomes)), \
             contextlib.redirect_stdout(output):
            passed, failed, rows, _ = await mixed.run_batch(5, tests)
            mixed.print_failure_summary(failed)
        self.assertEqual((passed, failed), (1, 4))
        self.assertEqual([row['status'] for row in rows],
                         ['EXCEPTION', 'GARBAGE', 'TOO_SHORT', 'MISSING', 'OK'])
        self.assertIn('fixture timeout', rows[0]['error'])
        self.assertIn('4 failed checks; cause remains unattributed', output.getvalue())
        self.assertNotIn('not code bugs', output.getvalue())
        self.assertNotIn('model answer mismatches', output.getvalue())

    async def test_multiturn_garbage_is_not_reported_as_missing_empty_list(self):
        conversation = dict(name='fixture', system='system',
                            turns=[dict(user='hi', expected=[])])
        output = io.StringIO()
        with patch.object(prefix.aiohttp, 'ClientSession', return_value=AsyncContext(object())), \
             patch.object(prefix, 'send_request', AsyncMock(return_value=response(''))), \
             contextlib.redirect_stdout(output):
            passed, failed, rows = await prefix.run_batch(1, [conversation])
        self.assertEqual((passed, failed), (0, 1))
        self.assertTrue(rows[0]['is_garbage'])
        self.assertIn('GARBAGE', output.getvalue())
        self.assertNotIn('missing []', output.getvalue())

    async def test_legitimate_missing_answer_keeps_specific_diagnostic(self):
        conversation = dict(name='fixture', system='system',
                            turns=[dict(user='hi', expected=['needle'])])
        output = io.StringIO()
        with patch.object(prefix.aiohttp, 'ClientSession', return_value=AsyncContext(object())), \
             patch.object(prefix, 'send_request', AsyncMock(return_value=response('other'))), \
             contextlib.redirect_stdout(output):
            passed, failed, rows = await prefix.run_batch(1, [conversation])
        self.assertEqual((passed, failed), (0, 1))
        self.assertFalse(rows[0]['is_garbage'])
        self.assertIn("missing ['needle']", output.getvalue())

    def test_success_summary_does_not_emit_failure_warning(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output): mixed.print_failure_summary(0)
        self.assertEqual(output.getvalue(), '')

    def test_raw_report_opt_in_preserves_full_text_errors_and_metrics(self):
        parent = ROOT.parents[1] / '.build' / 'batch-report-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            with patch.dict(os.environ, {'AFM_REPORT_DIR': directory}):
                report = BatchReport('fixture', 'model', 'endpoint')
                records = [dict(response('full output ' * 100), status='MISSING'),
                           dict(status='EXCEPTION', error='TimeoutError()')]
                batches = {1: dict(passed=0, failed=2, results=records)}
                with contextlib.redirect_stdout(io.StringIO()):
                    report.save(batches)
                    batches[2] = dict(passed=1, failed=0, results=[response('answer')])
                    report.save(batches)
                saved = json.loads(report.path.read_text())
                self.assertEqual(saved['batches']['1']['results'], records)
                self.assertEqual(saved['batches']['2']['results'][0]['text'], 'answer')
                self.assertEqual(len(list(Path(directory).glob('*.json'))), 1)
                self.assertFalse(list(Path(directory).glob('*.partial')))
                self.assertNotEqual(BatchReport('fixture', 'model', 'endpoint').path, report.path)

    def test_unset_report_directory_keeps_console_only_behavior(self):
        with patch.dict(os.environ):
            os.environ.pop('AFM_REPORT_DIR', None)
            report = BatchReport('fixture', 'model', 'endpoint')
            self.assertIsNone(report.path)
            report.save({'unserializable': object()})  # No serialization/file writes.

    async def test_each_validator_main_saves_exception_records(self):
        parent = ROOT.parents[1] / '.build' / 'batch-report-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        error = dict(name='fixture', status='EXCEPTION', error='HTTP 503 unavailable')

        async def known_batch(_size, records):
            records.append(error)
            return 0, 1

        with tempfile.TemporaryDirectory(dir=parent) as directory:
            for module, function, result in [
                (responses, 'run_validation', None),
                (mixed, 'run_batch', (0, 1, [error], 1)),
                (prefix, 'run_batch', (0, 1, [error])),
            ]:
                with self.subTest(module=module.__name__), \
                     patch.dict(os.environ, {'AFM_REPORT_DIR': directory}), \
                     patch.object(sys, 'argv', ['fixture', '1']), \
                     patch.object(module.aiohttp, 'ClientSession', return_value=AsyncContext(object())), \
                     patch.object(module, 'send_request', AsyncMock(return_value=response('warmup'))), \
                     patch.object(module.asyncio, 'sleep', AsyncMock()), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.ExitStack() as stack:
                    if module is mixed:
                        sampler = stack.enter_context(patch.object(mixed, 'MactopSampler'))
                        sampler.return_value.summary.return_value = None
                        stack.enter_context(patch.object(mixed, 'ASSERT_ACCEPTANCE', False))
                    operation = AsyncMock(side_effect=known_batch) if module is responses else AsyncMock(return_value=result)
                    stack.enter_context(patch.object(module, function, operation))
                    self.assertEqual(await module.main(), 1)
            reports = list(Path(directory).glob('*.json'))
            self.assertEqual(len(reports), 3)
            for path in reports:
                saved = json.loads(path.read_text())
                self.assertEqual(saved['batches']['1']['results'], [error])


if __name__ == '__main__': unittest.main()
