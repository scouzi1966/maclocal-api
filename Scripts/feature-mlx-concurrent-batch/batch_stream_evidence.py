"""Opt-in, bounded wire evidence; never feeds the validators' scoring paths."""
import asyncio
import base64
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time
import uuid

BYTES_PER_REQUEST_TOKEN = 2048
STREAM_OVERHEAD_BYTES = 64 * 1024
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_PARSE_ERRORS = 128
MAX_ERROR_MESSAGE_CHARS = 512
_active = ContextVar('batch_stream_evidence', default=None)


def record_stream_parse_error():
    capture = _active.get()
    if capture is not None:
        capture.parse_error(sys.exc_info()[1], 'sender')


class StreamEvidence:
    def __init__(self, root, suite):
        self.path = Path(root)/'stream-requests'/f'{suite}-{uuid.uuid4().hex}.json'
        self.started_at_unix = time.time()
        self.raw = bytearray()
        self.limit = STREAM_OVERHEAD_BYTES
        self.observed_bytes = self.lines_seen = self.parse_error_count = 0
        self.parse_errors = []
        self.request = self.endpoint = self.http_status = self.http_headers = None
        self.done_observed = self.iterator_eof = False

    def parse_error(self, error, origin):
        self.parse_error_count += 1
        if len(self.parse_errors) < MAX_PARSE_ERRORS:
            self.parse_errors.append(dict(origin=origin, line=self.lines_seen if origin == 'sender' else None,
                                          type=type(error).__name__, message=str(error)[:MAX_ERROR_MESSAGE_CHARS]))

    async def lines(self, source):
        async for line in source:
            self.lines_seen += 1
            self.observed_bytes += len(line)
            remaining = max(0, self.limit-len(self.raw))
            if remaining:
                self.raw.extend(line[:remaining])
            if line.startswith(b'data:') and line[5:].strip() == b'[DONE]':
                self.done_observed = True
            yield line
        self.iterator_eof = True

    def document(self, error):
        visible, reasoning, finish_reasons = [], [], []
        # Parse retained bytes after timing has been sampled. This observer
        # never replaces or repairs the sender's existing parser or text.
        event_data = []
        def event():
            if not event_data:
                return
            data = b'\n'.join(event_data)
            event_data.clear()
            if data.strip() == b'[DONE]':
                return
            try:
                chunk = json.loads(data)
                choices = chunk.get('choices') or []
                if choices:
                    choice = choices[0]
                    delta = choice.get('delta') or {}
                    if isinstance(delta.get('content'), str): visible.append(delta['content'])
                    if isinstance(delta.get('reasoning_content'), str): reasoning.append(delta['reasoning_content'])
                    if choice.get('finish_reason') is not None: finish_reasons.append(choice['finish_reason'])
            except Exception as parse_error:
                self.parse_error(parse_error, 'retained_sse_observer')
        for line in bytes(self.raw).splitlines():
            if not line:
                event()
            elif line.startswith(b'data:'):
                event_data.append(line[5:].lstrip(b' '))
        event()
        truncated = self.observed_bytes > len(self.raw)
        return dict(request=self.request, endpoint=self.endpoint, started_at_unix=self.started_at_unix,
                    http_status=self.http_status, http_headers=self.http_headers,
                    raw_sse_base64=base64.b64encode(self.raw).decode('ascii'),
                    raw_encoding='base64 (exact bytes consumed by the sender, not an independent socket capture)',
                    observed_bytes=self.observed_bytes, retained_bytes=len(self.raw),
                    byte_limit=self.limit, raw_truncated=truncated,
                    visible_content=''.join(visible), reasoning_content=''.join(reasoning),
                    finish_reason=finish_reasons[-1] if finish_reasons else None,
                    finish_reasons=finish_reasons, derived_channels_may_be_truncated=truncated,
                    done_observed=self.done_observed, iterator_eof=self.iterator_eof,
                    parse_errors=self.parse_errors, parse_error_count=self.parse_error_count,
                    parse_errors_truncated=self.parse_error_count > len(self.parse_errors),
                    request_error=None if error is None else dict(type=type(error).__name__, message=str(error)),
                    scoring_note='Observability only; original sender text and predicates are unchanged')

    def publish(self, error):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.partial')
        try:
            temporary.write_text(json.dumps(self.document(error), ensure_ascii=False, indent=2))
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)


class _Response:
    def __init__(self, response, capture):
        self.response, self.capture = response, capture

    @property
    def content(self):
        return self.capture.lines(self.response.content)

    def __getattr__(self, name):
        return getattr(self.response, name)


class _Post:
    def __init__(self, manager, capture):
        self.manager, self.capture = manager, capture

    async def __aenter__(self):
        response = await self.manager.__aenter__()
        self.capture.http_status = getattr(response, 'status', None)
        self.capture.http_headers = dict(getattr(response, 'headers', {}))
        return _Response(response, self.capture)

    async def __aexit__(self, *args):
        return await self.manager.__aexit__(*args)


class _Session:
    def __init__(self, session, capture):
        self.session, self.capture = session, capture

    def post(self, endpoint, **kwargs):
        self.capture.endpoint = endpoint
        self.capture.request = kwargs.get('json')
        tokens = int((self.capture.request or {}).get('max_tokens') or 0)
        self.capture.limit = min(MAX_CAPTURE_BYTES, STREAM_OVERHEAD_BYTES + max(0, tokens)*BYTES_PER_REQUEST_TOKEN)
        return _Post(self.session.post(endpoint, **kwargs), self.capture)


def capture_stream_evidence(suite):
    def decorate(sender):
        @wraps(sender)
        async def wrapped(session, *args, **kwargs):
            root = os.environ.get('AFM_REPORT_DIR')
            if os.environ.get('AFM_CAPTURE_STREAM_EVIDENCE') != '1' or not root:
                return await sender(session, *args, **kwargs)
            capture = StreamEvidence(root, suite)
            token = _active.set(capture)
            error = None
            try:
                return await sender(_Session(session, capture), *args, **kwargs)
            except BaseException as caught:
                error = caught
                raise
            finally:
                _active.reset(token)
                try:
                    # Serialize once off the event loop, after sender timing;
                    # await persistence so later conversation mutation is safe.
                    await asyncio.to_thread(capture.publish, error)
                except Exception as write_error:
                    # Evidence I/O must not change existing inference scoring.
                    print(f'WARNING: stream evidence not saved to {capture.path}: {write_error}', file=sys.stderr)
        return wrapped
    return decorate
