# Responses reasoning qualification — 2026-10-09

## Confirmed fixes

1. Incoming Responses reasoning items no longer become empty user messages.
2. Responses forwards sampling/template controls to chat generation, including an explicit reasoning-off override.
3. Tool-call finalization and AFMKit response construction preserve generated reasoning beside tool calls.
4. Responses exposes public reasoning through summary parts and matching SSE lifecycle events. Content-only reasoning events passed generic protocol checks but the tested Codex client did not consume them. Summary events were consumed and echoed on the next request.
5. Structured-output finalization preserves an anchored leading reasoning block without parsing literal thinking tags inside JSON. Truncated leading reasoning is withheld from visible structured content. Tool content and log-probability suppression remain unchanged.

These are boundary changes, not model, kernel, cache-policy, or sampler optimizations. Responses SSE remains buffered until generation completes; this patch does not claim incremental-generation latency improvements.

## Evidence

Evidence root (untracked):
`/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007`

| Check | Result |
| --- | --- |
| Final targeted XCTest suite | 87 passed, 0 failed |
| Qwen tool qualification | 4 passed, 0 failed |
| Reporter self-tests | 16 passed, 0 failed |
| Structured-output regression before fix | 8 failures, all missing known reasoning |
| Live AFM reasoning contracts | 27 passed, 0 failed, 0 uncovered |
| Same live contracts on reference | 25 passed, 2 failed (`enable_thinking:false`); `reasoning.effort:none` passed |

The live comparison used the identical checkpoint:
`/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001`

Configuration: MTP off, prefix cache enabled, temperature 0, top-p 1, top-k 0, no tuning environment variables. Reference binary SHA: `46d017b5f6e49890e654dbefd2b750eda44b627a20f4ec7a01015a4d0d8ea1e0`. AFM candidate SHA: `a4f689e25a3d7bf887f8c9e365d15a49a07229cf717aa464acce59cccbc059da`.

### Actual Codex round trip

Both engines created a small Python function and verified its result using tools. Both exposed reasoning events to Codex, and both subsequent requests echoed a reasoning item.

| Engine | Elapsed | Output tokens | Requests | Result |
| --- | ---: | ---: | ---: | --- |
| AFM | 10.15 s | 186 | 2 | Completed; 6,731 cached input tokens |
| Reference | 13.29 s | 333 | 3 | Completed; reported cached usage is not comparable to AFM's |

This small task proves client delivery, not full-project quality or performance parity. Raw records: `reasoningSummaryCodex20261009/results/`.

### Prompt normalization diagnostic

The reference omits tool `strict:false` metadata from its rendered prompt. Removing only this redundant field from identical captured requests reduced the AFM/reference prompt-token difference from 21 to 1. The captured growing-prefix follow-up took AFM 2.80 s versus reference 2.58 s. Outputs differed (104 versus 102 tokens); this is not an isolated decode-throughput measurement. No production normalization default was changed.

Records: `reasoningStrictReplay20261009/`. All requests and raw SSE responses are retained. The AFM live checker in that directory passed all 27 assertions.

## Remaining qualification

- Larger dashboard comparison is running under `reasoningDashboard20261009/`; do not infer its result from the small task.
- Explain remaining full-project token-volume and elapsed-time divergence before claiming parity.
- AFM still reports zero reasoning-token usage despite delivering reasoning; accounting correctness is a separate remaining gap. Do not fabricate a token count from character length.
- Verify full release packaging and immutable AFMKit dependency pin before publishing. This diagnostic candidate uses paired local worktrees.
