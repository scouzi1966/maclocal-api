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
| Final targeted XCTest suite | 93 passed, 0 failed |
| Provider raw/no-tools contracts | 4 passed, 0 failed |
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

The reference omits tool `strict:false` metadata from its rendered prompt. Removing only this redundant field from identical captured requests reduced the AFM/reference prompt-token difference from 21 to 1. The captured growing-prefix follow-up took AFM 2.80 s versus reference 2.58 s. Outputs differed (104 versus 102 tokens); this is not an isolated decode-throughput measurement. At that diagnostic stage no production normalization default had changed; the later provider checkpoint below incorporates the verified normalization.

Records: `reasoningStrictReplay20261009/`. All requests and raw SSE responses are retained. The AFM live checker in that directory passed all 27 assertions.

### Exact prompt capture and Orbital bottleneck

The sequential diagnostic captures under `promptCaptureAFM20261009/comparison/`
reconstructed every token for the first three saved Orbital requests. All 21
extra AFM tokens are explained by four redundant `strict:false` fields plus
escaped forward slashes in tool descriptions. Outside those tool-definition
spans the token sequences match. This is a diagnostic result, not throughput
qualification: logging was enabled and generation was limited to one token.

The earlier complete Orbital run in `fixedThreeProjects20261008/` took AFM
747.903 seconds / 39,930 output tokens / 44 requests versus the reference's
359.689 seconds / 21,242 tokens / 25 requests. Acceptance was 11/11 versus
10/11. AFM spent 659.599 seconds decoding and 71.609 seconds prefilling;
non-request wall time was only 12.396 versus 13.623 seconds. The primary gap
was generation volume and extra turns, not shell-execution overhead. AFM
request 11 alone generated 11,068 tokens and took 184.441 seconds. It is the
next exact-request replay target before repeating the full project.

These older coding runs do not demonstrate end-to-end parity. Near-parity
claims for isolated decoding must not be generalized to agent completion time.

### Normalized candidate: saved request 11

Provider checkpoint `136d8e55` omits default-false strict metadata only in the
native tool-prompt shim and avoids needless forward-slash escaping. It does not
change strict enforcement, kernels, cache policy, or sampling. Provider tests:
20 Qwen tool tests plus 4 raw/no-tools tests passed; consumer 93 XCTest plus
4 Qwen qualification tests passed. Independent source review found no blocking
issue. Consumer checkpoint: `6839b05`.

The exact rendered text of saved Orbital request 11 now matches the reference.
Two whitespace tokenization differences remain: AFM uses checkpoint token 13488
for `\n  \n`; the reference splits it into 198 and 2228. The checkpoint's greedy
whitespace regex and explicit BPE merge support AFM's form. We did not change
AFM tokenization to imitate the reference difference or alter the release
reference binary. Capture: `orbital11Capture20261009/comparison/`.

The subsequent non-traced, cold replay (`orbital11Replay20261009/`) used the
original 16,384-token output limit and the unchanged release reference:

| Engine | Input tokens | Output tokens | Wall time |
| --- | ---: | ---: | ---: |
| AFM | 15,477 | 5,599 | 104.140 s |
| Reference | 15,479 | 6,570 | 115.290 s |

Both returned reasoning and an `exec_command` call, but generated different
code. This is not project acceptance or proof of end-to-end parity. It also
must not be treated as a controlled causal comparison to the old warm request
11: input serialization and cache state differed. AFM cold prefill was 12.07 s
and decode 91.97 s / 60.9 tok/s. Candidate binary SHA:
`0e8edd40a46341bab51da5ad7725a45e43ba622a7c4ef2eac1d6951646441156`.

The fresh full-project qualification is `orbitalNormalized20261009/`. Its
shared execution directory is `/Volumes/edata/afm-isolated-coding-20261009/orbital`,
outside prior results. Each engine starts a new ephemeral Codex process and a
new fixture copy containing only TASK.md, AGENTS.md, package files, and installed
dependencies. Initial file hashes and client/launcher hashes are recorded.
After an engine finishes and requests drain, its working tree is moved out to
the evidence archive; the next engine starts from a fresh copy at the same
path. Inference and acceptance runs are sequential. No generated solution is
manually repaired.

### Fresh normalized full-project pair

The completed `orbitalNormalized20261009/` pair retained the frozen candidate
SHA `0e8edd40a46341bab51da5ad7725a45e43ba622a7c4ef2eac1d6951646441156`
and release reference SHA above. MTP was off, prefix cache and reasoning were
on, with the same checkpoint, sampling, tools, initial file hashes, canonical
workspace path, Codex executable, launcher implementation, and catalog.
Inference and independent browser acceptance were sequential.

| Metric | AFM | Reference |
| --- | ---: | ---: |
| Agent wall time | 511.390 s | 1,078.934 s, budget-censored |
| Requests | 34 | 70 |
| Reported output tokens | 26,391 | 60,000 |
| Independent project acceptance | 11/11 | 11/11 |
| Compactions | 1 | 2 |
| Sum of HTTP request time | 496.149 s | 1,050.248 s |
| Remaining wall time | 15.241 s | 28.686 s |
| Delivered reasoning characters | 41,821 | 147,819 |
| Final summary | Present | Empty; incomplete response |

Reference request 070 consumed the final 1,498 tokens of the common 60,000-token
budget in reasoning and returned `status: incomplete` / `max_output_tokens`.
Codex nevertheless exited zero. This is not a normal agent completion, and
the ratio of these elapsed values must **not** be advertised as a 2.11x
completion-speed win. Passing artifact acceptance is a separate outcome from
finishing the agent task with a final report.

AFM improved over the earlier AFM Orbital run (747.903 s / 39,930 reported
output tokens / 44 requests, also 11/11 acceptance). Its weighted reported
decode rate was essentially unchanged, 60.54 versus 60.72 tok/s. The shorter
run primarily reflects fewer generated tokens and turns, not a new kernel
speedup. Multiple fixes and workspace normalization changed together; one
pair cannot assign causality or establish general engine parity.

Both models spent substantial time revisiting completed work. AFM requests
22–34 used 160.364 backend seconds / 7,786 tokens after implementation and
self-tests were complete. Reference requests 49–70, after its second valid
compaction, used 300.698 seconds / 17,988 tokens, mostly re-verification and
attempts to start a preview server despite the sandbox's socket restriction.
Some earlier late reference changes genuinely improved tests and documentation;
the entire tail must not be called wasted. Both engines preserved reasoning
and useful compaction handoffs in this run, with no recurrence of the previous
undeclared-tool failure.

Reporting caveats:

- Output-token figures are the frozen binaries' reported usage. A separate
  source audit found AFM's serial token counter incremented on detokenized
  chunks rather than every accepted token; buffered Unicode can undercount.
  The exact discrepancy cannot be reconstructed from final text alone.
- AFM delivered reasoning but still reported zero reasoning-token usage.
  This does not mean reasoning was absent or excluded from completion totals.
- AFM reported 484,959 cached input tokens. Reference API usage reported zero,
  while its server logs prove hot-cache reuse. Do not infer no reference cache
  reuse from its API usage field.
- Both engines buffered semantic reasoning/tool output until near request
  completion. Reference's earlier lifecycle events are not early reasoning.

The next paired run must separate these outcomes and verify local-preview
permissions before inference. A sandbox preflight confirmed loopback serving
works with the network proxy enabled and local binding allowed, while direct
and proxied external connections are denied. This is a **new harness profile**,
not an unchanged repeat of the socket-restricted comparison. The official
[Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
documents the proxy and local-binding controls. Keep both engines identical
within that new pair and retain this original evidence.

## Remaining qualification

### Accepted-token accounting checkpoint

Provider commit `f12b2d32c18ed9ff169b3c912e2502393a930a05` fixes the
serial-generation counter described above. It increments for each accepted
non-EOS/non-unknown token before detokenization, rather than only when the
detokenizer emits text. Cancellation and special-token filtering retain their
existing placement. No sampling, cache, kernel, or token-limit policy changes
are included, and no GPU read or retokenization was added.

The actual generation-path regression tests cover ASCII output, three UTF-8
tokens producing one euro character, and a token limit reached before a
Unicode character can be emitted. Before the fix, the latter two cases failed
(counts 1 instead of 3 and 0 instead of 2); after the fix all three passed.
Independent review found no blocking issue. Logs are
`generation-count-red-pinned.log` and `generation-count-green.log` under the
evidence root. The isolated test package uses the production dependency
revisions; an initial unconstrained dependency-resolution attempt failed on a
newer compiler-incompatible dependency and is not a production build failure.

The release-optimized qualification build completed in 214.33 seconds using
consumer `cb7d6194b5d136a571d4f848757fa430082a4e22` and this provider commit.
Binary: `acceptedTokenRuntime20261009/afm`; SHA-256:
`91d8945c2932db1dab366285d3474fcd8148717cc3e5e1758d357e0985bc37c6`.
This is a paired-worktree qualification binary, not a published nightly.
The fix does not reconstruct earlier counts and is not an explanation for
the large differences in generated code, number of turns, or elapsed time.
Reasoning-token breakdown remains a separate unresolved accounting item.

This binary is frozen for the sequential, reference-first
`orbitalPreview20261009/` pair. That profile permits local preview servers,
disables the Codex daemon for both clients, records incomplete/budget outcomes
explicitly, and cleans only verified newly created preview children before
archiving a workspace. It is not an unchanged repeat of the previous pair.
The startup preflight verified local binding and denied the tested direct and
proxied public connections; this is not a blanket claim of private-LAN
isolation. The fixture's older "no network" wording remains alongside the
new direct instruction allowing localhost. The reference's first compaction
repeated the older wording without the exception; retain this limitation
when interpreting verification behavior. Do not alter instructions midway
through a pair.

### Outstanding checks

- Larger dashboard comparison is under `reasoningDashboard20261009/`. AFM exhausted the 80-request harness cap after 1,106.41 seconds and 55,292 output tokens; the reference completed in 215.35 seconds, 14 requests and 12,477 tokens. Both generated projects passed 15/16 acceptance checks. The 429 is a harness budget response, not server overload. This is a failed completion qualification, not a release-ready result.
- AFM compaction request 20 supplied `tools:[]`, but its response contained a structured tool call and no answer. Codex resumed with an empty handoff and reread files. Earlier archived AFM compactions produced nonempty handoffs, so do not claim this explains every previous timing gap.
- The request-local no-tools parser fix and fail-closed consumer guard subsequently passed 93 consumer XCTest tests plus 4 Qwen qualification tests. Replaying the failed compaction now preserves emitted markup as ordinary text, with no callable output. The provider/model still emits tool-like text rather than a useful summary; the reference produced a proper handoff. Protocol repair is not behavioral parity.
- That replay is recorded under `noToolsCompactionReplay20261009/`, AFM SHA `48dcac1da94e94af60ddd44472ff27c769e839bfcc902f724e521c9ea5825625`. The diagnostic bounded output at 2,048 tokens; both stopped below it. AFM: 26,952 input/661 output, 32.29s wall, 61.3 decode tok/s. Reference: 26,882 input/1,197 output, 39.60s wall. These walls include prefill and different generated lengths; do not equate them with quality or isolated throughput.
- Explain remaining full-project token-volume and elapsed-time divergence before claiming parity.
- AFM still reports zero reasoning-token usage despite delivering reasoning; accounting correctness is a separate remaining gap. Do not fabricate a token count from character length.
- Verify full release packaging and immutable AFMKit dependency pin before publishing. This diagnostic candidate uses paired local worktrees.
