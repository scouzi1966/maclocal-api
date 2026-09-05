# Optional streaming evidence

The known-answer, mixed-workload, and multi-turn prefix validators can retain
diagnostic stream evidence when **both** `AFM_CAPTURE_STREAM_EVIDENCE=1` and
`AFM_REPORT_DIR=/path/to/diagnostic-report` are set. Report-directory configuration
alone keeps the original sender path and existing batch reports; it does not
enable this capture. Do not enable capture for primary performance comparisons
without first agreeing to its overhead.

Each request produces one atomically published JSON file in
`AFM_REPORT_DIR/stream-requests/`. It includes the exact request/history,
endpoint, HTTP status/headers, base64-encoded bytes consumed by the sender,
separate visible/reasoning text, finish reasons, observed `[DONE]`, iterator EOF,
parse errors, and partial evidence on a request exception or cancellation.
This is not an independent socket capture: the sender normally stops at `[DONE]`
without reading the following blank line or exhausting the response iterator.
Missing finish/DONE evidence must not be interpreted as proof of an engine bug.

The original request, parser, merged scoring text, assertions, and timing
expressions are unchanged. Parse errors from the existing sender and from the
post-request evidence observer are distinguished. Evidence never repairs the
scoring input. Lexical-gate concerns remain tracked separately in issue #265;
new evidence cannot retroactively reclassify responses that were not retained.
The prefix report's existing Turn 1 “cold” label is also not proof of a cold
cache: earlier batch sizes may have warmed its prefix. Use actual cache metrics.

## Bounds and performance caveats

Raw retention is capped at `min(16 MiB, 64 KiB + max_tokens * 2048 bytes)` per
request (about 8.06 MiB at the current 4096-token budget). The sender still
receives all bytes; overflow sets explicit truncation flags. Derived channel
text and finish reasons then describe only the retained prefix, while the
independent DONE observation still covers all consumed lines. Parse-error
details are capped at 128 records with 512-character messages.

Capture adds byte copying and line bookkeeping during streaming. After the
sender samples elapsed time/TTFT, a worker thread parses retained SSE and writes
one JSON document through an atomic rename; there are no per-token filesystem
writes. Serialization/base64 and decoded text temporarily allocate several
times the retained byte count, multiplied by concurrent requests. Persistence
is awaited before the next conversation turn, so it adds between-turn latency
and can affect aggregate wall time, concurrency overlap, and observed throughput.
Even individual timings can be indirectly affected by capture overhead. Treat
these as diagnostic runs, not equivalent performance baselines.

I/O failures warn on stderr and do not change inference scoring. Abrupt process
termination may leave no evidence (or an unpublished `.partial` file); normal
request exceptions and cancellation are captured on a best-effort basis.
Artifacts contain full prompts, responses, and response headers: review them
before sharing and keep them outside version control.
