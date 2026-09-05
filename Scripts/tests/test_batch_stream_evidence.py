"""CPU-only wire-evidence tests against all three real streaming senders."""
import asyncio
import base64
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_batch_failure_reporting import load, ROOT
from batch_stream_evidence import StreamEvidence, MAX_CAPTURE_BYTES

MODULES = [load(name) for name in ('validate_responses', 'validate_mixed_workload', 'validate_multiturn_prefix')]


def frame(payload):
    return ('data: ' + json.dumps(payload, ensure_ascii=False) + '\n\n').encode()


class Response:
    status = 200
    headers = {'X-Request-ID': 'fixture-request'}
    def __init__(self, body, error=None): self.body, self.error = body, error
    def raise_for_status(self): pass
    @property
    def content(self):
        async def lines():
            for line in self.body.splitlines(keepends=True):
                yield line
            if self.error is not None: raise self.error
        return lines()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): return False


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        parent = ROOT.parents[1]/'.build/batch-stream-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    async def send(self, module, response, enabled=True):
        post = Mock(return_value=response)
        session = SimpleNamespace(post=post)
        argument = [{'role': 'user', 'content': 'fixture history'}] if module.__name__.endswith('prefix') else 'fixture'
        with patch.dict(os.environ, {'AFM_REPORT_DIR': str(self.root),
                                    'AFM_CAPTURE_STREAM_EVIDENCE': '1' if enabled else ''}), \
             patch.object(module, 'time', SimpleNamespace(monotonic=Mock(side_effect=[100, 100.25, 101]))):
            result = await module.send_request(session, argument, max_tokens=8)
        return result, post.call_args

    def documents(self):
        self.assertFalse(list(self.root.rglob('*.partial')))
        return [json.loads(path.read_text()) for path in self.root.rglob('*.json')]

    async def test_all_senders_preserve_payload_scoring_text_and_timing_values(self):
        body = (frame({'choices': [{'delta': {'content': 'héllo', 'reasoning_content': 'private reasoning'}}]})
                + frame({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
                + b'data: [DONE]\n\n')
        for module in MODULES:
            baseline, request = await self.send(module, Response(body), enabled=False)
            observed, captured_request = await self.send(module, Response(body))
            self.assertEqual(observed, baseline)
            self.assertEqual(captured_request, request)
        docs = self.documents()
        self.assertEqual(len(docs), 3)
        for doc in docs:
            self.assertEqual(base64.b64decode(doc['raw_sse_base64']), body[:-1])
            self.assertEqual(doc['visible_content'], 'héllo')
            self.assertEqual(doc['reasoning_content'], 'private reasoning')
            self.assertEqual(doc['finish_reason'], 'stop')
            self.assertTrue(doc['done_observed'])
            self.assertFalse(doc['iterator_eof'])  # Sender stops at DONE; no claim of socket EOF.
            self.assertEqual(doc['http_status'], 200)
            self.assertEqual(doc['http_headers']['X-Request-ID'], 'fixture-request')

    async def test_malformed_events_are_retained_without_changing_sender_result(self):
        body = b'data: {broken}\n\n' + frame({'choices': [{'delta': {'content': 'ok'}}]}) + b'data: [DONE]\n\n'
        for module in MODULES:
            baseline, _ = await self.send(module, Response(body), enabled=False)
            observed, _ = await self.send(module, Response(body))
            self.assertEqual(observed, baseline)
        for doc in self.documents():
            self.assertTrue(any(error['origin'] == 'sender' for error in doc['parse_errors']))
            self.assertEqual(doc['visible_content'], 'ok')

    async def test_eof_without_done_is_not_invented_as_clean_done(self):
        await self.send(MODULES[0], Response(frame({'choices': [{'delta': {'content': 'partial'}}]})))
        doc = self.documents()[0]
        self.assertFalse(doc['done_observed'])
        self.assertTrue(doc['iterator_eof'])
        self.assertIsNone(doc['finish_reason'])

    async def test_transport_failure_and_cancellation_preserve_partial_wire(self):
        body = frame({'choices': [{'delta': {'content': 'partial'}}]})
        for error in (RuntimeError('connection lost'), asyncio.CancelledError('fixture cancellation')):
            with self.assertRaises(type(error)):
                await self.send(MODULES[0], Response(body, error))
        docs = self.documents()
        self.assertEqual(len(docs), 2)
        for doc in docs:
            self.assertEqual(base64.b64decode(doc['raw_sse_base64']), body)
            self.assertEqual(doc['visible_content'], 'partial')
            self.assertIsNotNone(doc['request_error'])
            self.assertFalse(doc['iterator_eof'])

    async def test_http_rejection_does_not_read_body_or_become_empty_success(self):
        class Unavailable(Response):
            status = 503
            def raise_for_status(self): raise RuntimeError('HTTP 503')
            @property
            def content(self): raise AssertionError('must not read failed response')
        with self.assertRaisesRegex(RuntimeError, 'HTTP 503'):
            await self.send(MODULES[0], Unavailable(b''))
        self.assertEqual(self.documents()[0]['http_status'], 503)

    async def test_capture_limit_never_truncates_the_senders_input(self):
        body = frame({'choices': [{'delta': {'content': 'x'*100_000}}]}) + b'data: [DONE]\n\n'
        result, _ = await self.send(MODULES[0], Response(body))
        self.assertEqual(len(result[0]), 100_000)
        doc = self.documents()[0]
        self.assertTrue(doc['raw_truncated'])
        self.assertEqual(doc['retained_bytes'], doc['byte_limit'])
        self.assertLessEqual(doc['retained_bytes'], MAX_CAPTURE_BYTES)
        self.assertTrue(doc['done_observed'])
        self.assertTrue(doc['derived_channels_may_be_truncated'])

    async def test_evidence_write_failure_warns_without_changing_score_input(self):
        body = frame({'choices': [{'delta': {'content': 'ok'}}]})
        with patch.object(StreamEvidence, 'publish', side_effect=OSError('fixture disk full')), \
             contextlib.redirect_stderr(io.StringIO()) as log:
            result, _ = await self.send(MODULES[0], Response(body))
        self.assertEqual(result[0], 'ok')
        self.assertIn('stream evidence not saved', log.getvalue())

    async def test_report_directory_alone_preserves_original_session_and_creates_no_raw_artifact(self):
        for module in MODULES:
            with patch('batch_stream_evidence.StreamEvidence', side_effect=AssertionError('capture must be bypassed')):
                await self.send(module, Response(b''), enabled=False)
        self.assertEqual(self.documents(), [])

    async def test_capture_requires_report_directory(self):
        sender = MODULES[0].send_request
        with patch.dict(os.environ, {'AFM_REPORT_DIR': '', 'AFM_CAPTURE_STREAM_EVIDENCE': '1'}), \
             patch('batch_stream_evidence.StreamEvidence', side_effect=AssertionError('capture must be bypassed')):
            await sender(SimpleNamespace(post=Mock(return_value=Response(b''))), 'fixture')
        self.assertEqual(self.documents(), [])


if __name__ == '__main__': unittest.main()
