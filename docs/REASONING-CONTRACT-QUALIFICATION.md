# Reasoning and tool-turn qualification

Ordinary-answer thinking checks are insufficient. Before this regression set,
the assertion harness probed `/v1/chat/completions` with arithmetic questions,
and Responses controller tests fabricated plain chat answers. Neither traversed
the **real non-streaming chat finalizer with both reasoning and a tool call**.
Responses uses that finalizer for its external JSON **and** SSE transports, so
an apparently successful tool call could silently lose the reasoning item.

## Deterministic gates (no model inference)

These run with the normal `Scripts/test-assertions.sh --tier unit` suite. For a
targeted run, from the checkout containing the fix:

```sh
Scripts/swiftpm-reliable.sh test -c release --filter 'ResponsesReasoningContractTests|MLXChatCompletionsControllerStreamingTests.testReasoningContractRetainsThinkingAlongsideToolCallsInBothAPIs'
```

| Contract | How it is checked |
|---|---|
| Returned reasoning is not an empty user message | Capture the actual translated chat request; assert exact user/assistant/tool/user sequence and tool ID |
| Opaque/summary reasoning does not change history | Compare translated messages with/without echoed reasoning |
| Explicit template off survives medium effort | Assert forwarded kwargs, including an unrelated marker |
| Effort `none` defeats template on | Assert off while retaining other kwargs |
| Sampling and template settings reach both external transport paths | Assert `seed`, `top_k`, temperature, top-p, thinking and effort in captured requests |
| Reasoning survives a tool turn | Mock provider emits known reasoning + structured tool call; use the real chat controller and Responses adapter |
| Streaming/non-streaming consistency | Inspect structured reasoning/tool content in chat JSON/SSE and Responses JSON/SSE; require reasoning-specific delta/done delivery and matching `output_item.done`, not only terminal JSON |
| No raw thinking tags or hidden-text leakage | Split tags across streaming chunks; assert extracted reasoning and clean visible content |

The known mock reasoning **must** be preserved. Its absence fails a deterministic
engine test; it cannot be excused as model behavior. These tests do not load
weights or change runtime defaults, and add no inference overhead.

## Installed-binary gate (requires coordinated GPU access)

Start the exact release binary and checkpoint as for ordinary qualification.
Do not add `--no-think`; use a checkpoint documented to support reasoning,
tool calls, and per-request thinking control. The harness does not start, stop,
or reconfigure the server. It issues six sequential requests: on/off/effort-none
crossed with JSON/SSE, always requesting `read_file` for `README.md`.

```sh
python3 Scripts/test-reasoning-contracts.py \
  --base-url http://127.0.0.1:9999 --model '<exact model ID>' \
  --output-dir '/Volumes/edata/afm-benchmarks/<new-run>/reasoning-contracts'
```

Or include the same gate in existing assertion JSONL/HTML reporting:

```sh
Scripts/test-assertions.sh --tier standard --model '<exact model ID>' \
  --port 9999 --bin '<exact release binary>' --reasoning-contracts
```

All raw request/response bodies and a structured report are saved. Output
directories must be new to protect previous evidence. Explicitly capture the
binary/checkpoint hashes alongside the report in release qualification.

Exit `0` means all observable checks passed; `1` means failure; `2` means
uncovered reasoning retention. **Absent reasoning is not a pass** and does not
establish that a model lacks reasoning. A missing tool call is an unattributed
engine/model-boundary failure. Truncation is not successful coverage; increase
the explicitly recorded output budget and rerun when appropriate. The default
budget is 1,024 output tokens; `--max-output-tokens` adjusts it. A model that does
not support these features should not receive a claimed passing qualification.

Live output cannot prove effective option forwarding or reveal an inserted
empty user turn reliably. That is why the deterministic captured-request tests
are mandatory alongside installed-binary qualification. Do not infer quality
or throughput parity from these protocol tests.

## Reporter self-tests (CPU only)

```sh
python3 Scripts/test-reasoning-contracts.py --self-test
```

Mutation fixtures verify that missing reasoning, dropped tools, leaked split
tags, ignored off, truncated output, and missing SSE completion cannot become
false passes. These validate the harness, **not AFM**. Run the Swift and live
gates before claiming that the application regressions are fixed.

The SSE checker accepts both `response.reasoning_text.delta/done` with
`content_index` and `response.reasoning_summary_text.delta/done` with
`summary_index`. It checks item IDs, output indices, lifecycle order, delta
assembly, the matching completed item, and terminal consistency. A terminal
`response.completed` containing reasoning without those delivery events fails.
Content-only reasoning with an empty summary is not rejected solely for lacking
a summary; client-specific display/round-trip support requires separate evidence.
