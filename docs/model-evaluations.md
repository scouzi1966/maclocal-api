# Local model evaluations

`afm mlx -m <model> --eval` runs AFM's bundled `comprehensive` suite in-process.
It does not start an HTTP server, invoke Python, call a cloud service, or use an AI
judge. The MLX model is loaded once and reused for every case.

`--bench` is a discoverable alias for `--eval`. Select additional suites by repeating
`--eval-suite`; specifying a suite implies `--eval`:

```bash
afm mlx -m <model> --eval
afm mlx -m <model> --bench --no-open
afm mlx -m <model> --eval-suite comprehensive --eval-suite regression
afm mlx --eval-list
```

The bundled suite contains **all 91 labeled variants** from
`Scripts/test-llm-comprehensive.txt`, in the same order and with the same labels and
prompts. The mapping is one legacy `[@ label]` to one JSON case whose `id` is `label`.
It covers sampling, system/developer instructions, JSON, 29 stop-sequence variants,
logprobs, seeded generation, reasoning, six native tool-schema variants, streaming,
code, Unicode, multilingual output, longer-prefill timing, and arithmetic. Fifty-two
cases have content-specific deterministic string/JSON/tool/cross-case checks; the
remaining 39 make only a deterministic nonempty-output health check and retain
performance evidence without claiming semantic quality. A failed content check is
`missed`, not an infrastructure crash.

The legacy shell harness changed process-wide flags by restarting AFM between variants.
The native evaluator deliberately loads the model once. Per-request flags are translated
directly. The following process-only variants remain present for prompt/output/metric
coverage, and each generated case description records the adaptation:

| Legacy process behavior | Variants | Native load-once behavior |
|---|---:|---|
| Prefix cache off/on A/B | 6 (3 marked cached) | Uses the evaluator's single CLI-level cache setting; prompts and TTFT are retained, but the suite does not claim an off/on comparison |
| KV size / KV bits | 2 | Uses `--max-kv-size` compatibility behavior and the evaluator's CLI-level `--kv-bits` |
| Prefill step size | 2 | Uses one CLI-level `--prefill-step-size`; per-case TTFT is still retained |
| Raw reasoning extraction | 2 | Records separated visible output and reasoning rather than changing extraction mid-run |
| Verbose / very verbose | 2 | Direct in-process evaluation does not restart logging modes |
| Explicit legacy `xmlFunction` parser | 1 | Uses current native model-format detection; the tool-name assertion remains active |

The old file's 4,096/32,768-token ceilings are bounded to 512 in the distributed suite
so a default local run is comprehensive but finite; explicit smaller truncation cases
retain their original limits. The intended `streaming-seeded` case uses the native
streaming API (the old Python client actually sent every request with `stream=False`),
and `non-streaming-seeded` checks cross-path seeded equivalence.

## Results

Each run creates a unique directory beneath `~/.afm/evals` named with the UTC date and
time, sanitized model name, and selected suites. It contains:

- `run.json`: complete machine-readable metadata, aggregates inputs, and results.
- `results.jsonl`: one durable record per completed case, appended as the run proceeds.
- `suites.json`: exact suite definitions used for reproduction.
- `eval.log`: infrastructure errors without environment-variable dumps.
- `report.html`: a self-contained, HTML-escaped report.

The report includes AFM/model/suite identity, non-sensitive system information, every
prompt and output, extracted reasoning and tool calls, deterministic check results,
wall time, observable MLX prompt-time/TTFT, token counts, throughput, finish reason,
per-case generation parameters, aggregate token/latency/throughput totals, and a
reproducibility command. TTFT is only populated for native streaming cases; non-streaming
cases retain prompt and generation timing without mislabeling prefill time as TTFT. On a successful interactive run AFM opens it
with macOS `/usr/bin/open`; use `--no-open` for CI or remote sessions.

SIGINT/SIGTERM takes effect at the next case boundary. Completed JSONL records and the
latest HTML/JSON snapshots remain available. Schema/model-load and generation
infrastructure failures return nonzero. Ordinary deterministic quality misses remain
visible in the report but do not make the process fail.

## Custom suites

Create and validate a starter file:

```bash
afm mlx --eval-init regression
afm mlx --eval-validate regression
afm mlx -m <model> --eval-suite regression
```

Custom suites are JSON files directly under `~/.afm/evals`; result subdirectories are
not scanned as suites. A custom suite with the same `name` overrides a bundled suite.
The format is versioned and rejects unknown keys so misspelled options fail clearly:

```json
{
  "schemaVersion": 1,
  "name": "regression",
  "description": "Project-specific local checks.",
  "defaults": {
    "temperature": 0,
    "maxTokens": 128,
    "seed": 42
  },
  "cases": [
    {
      "id": "hello",
      "prompt": "Respond with exactly: hello",
      "expectations": { "exact": "hello" }
    }
  ]
}
```

Case `parameters` may set `temperature`, `maxTokens`, `topP`, `topK`, `minP`,
`repetitionPenalty`, `presencePenalty`, `seed`, `logprobs`, `topLogprobs`, `stop`,
OpenAI-compatible `tools`, `responseFormat`, and `streaming`. Cases may also add a
`developer` message. Case values override suite defaults; suite defaults override CLI
sampling values.

Available deterministic expectations are `exact`, `contains`, `notContains`,
`validJSON`, `minimumCharacters`, `maximumCharacters`, `toolCallName`, and
`caseSensitive`; bundled suites can additionally use `matchesCase` for seeded cross-path
checks. Checks are case-insensitive unless `caseSensitive` is true.

For safety and predictable local resource use, names cannot contain path separators,
suite files are capped at 5 MB and 1,000 cases, prompt/system text is capped at 64 KB
per field, generation parameters have bounded ranges, and scaffold mode never
overwrites a file. Suite files define inference inputs only: there are no commands,
hooks, environment interpolation, network callbacks, or arbitrary file paths.
