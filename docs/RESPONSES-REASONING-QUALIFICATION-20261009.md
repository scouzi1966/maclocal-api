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

### Completed preview-enabled pair

The `orbitalPreview20261009/` pair is complete. It used the accepted-token
candidate above, the unchanged release reference, and the identical checkpoint.
MTP was off, prefix caching and reasoning were on, and sampling was temperature
0 / top-p 1 / top-k 0 / seed 123. Initial fixture hashes and first-request contents
matched after excluding the engine's model alias and client metadata/IDs.
The reference ran first; inference and independent acceptance remained
sequential. Both engine processes exited and both workspace-cleanup checks
verified empty. Generated solutions were not manually repaired.

| Metric | AFM | Reference |
| --- | ---: | ---: |
| Agent wall time | 583.338 s | 1,079.247 s, budget-censored |
| Requests | 31 | 75 |
| Output tokens | 32,432 | 60,000 |
| Agent outcome | Completed; 1,822-character final report | Incomplete; empty final report |
| Independent project acceptance | 10/11 | 7/11 |
| Last observed self-authored test result | 11/11 | 10/11 |
| Compactions | 0 | 3, all valid |
| Sum of HTTP request time | 559.219 s | 1,062.174 s |
| Remaining wall time | 24.119 s | 17.073 s |
| Reported input/cache tokens | 541,524 / 516,443 (95.37% reuse) | 1,325,069 / 0; server logs prove reuse |

AFM's project-only generation logs reconcile to exactly 31 requests and 32,432
output tokens: 30.430 seconds prefilling and 522.512 seconds decoding, or
62.069 decode tok/s. This excludes the separate four-token startup warmup.
The reference logs do not supply an equivalent isolated decode duration for
this pair, so no same-pair pure-decode gap is claimed. Output tokens divided
by total agent wall time happen to be almost identical: 55.597 versus 55.594
tok/s. These are descriptive whole-run rates, not a controlled raw-throughput
comparison: prompts, generated solutions, and completion outcomes diverged.

The shorter AFM run follows fewer generated tokens and turns, not a new
kernel optimization. AFM's reported project-only decode rate is 62.07 versus
the earlier 60.72 tok/s; this is not a controlled regression measurement.
Differing context trajectories and the corrected token counter prevent an
isolated speedup claim. Nor may the
583.338/1,079.247 ratio be advertised as a normal-completion speedup: the
reference exhausted the common output budget and did not finish.

Both applications render real three-dimensional scenes and fit the mobile
viewport. Desktop and mobile screenshots were inspected. AFM passes animation,
raycasting, and browser-exception checks. Its single scored failure is a
lowercase `boreal` selection identifier where the evaluator expects `Boreal`.
The dropdown and details update work; nevertheless its lowercase option values
also disagree with the explicit task contract. Keep the score at 10/11, not
11/11, and classify this as generated-project behavior rather than an engine
transport bug.

The reference has the same selection-contract mismatch, plus a negative-delta
exception that stops animation, a null-distance failure in the exposed raycast
verification function, and the resulting browser-exception failure. The
raycast result does not prove all normal pointer selection is broken. Its
requests 1–12 used 229.982 backend seconds / 13,598 tokens on implementation
and pure-math tests. Requests 13–75 used another 832.192 seconds / 46,402 tokens
after starting a self-authored WebGL mock-test detour, including 193.681 seconds
/ 8,571 tokens for three useful compaction handoffs. That detour contained real
test additions and repairs; do not label the entire tail wasted work.

AFM's own final tests passed 11/11; these are separate from the independent
10/11 acceptance score. It actually served local previews, although background
preview processes ended between tool calls and led to additional restarts.
Eight preview-only requests consumed 49.142 backend seconds / 2,702 tokens.
This was not an observed network denial or proof of an engine defect. Reference
never attempted an actual preview. AFM's final report claimed the preview was
left running, but cleanup found no remaining process; this is a narrow
generated-report overclaim, not a verified running server. AFM created its own
temporary debug/log
files outside the project under `/tmp`; no reads of previous solutions or
evaluation files were observed. All retained qualification artifacts remain
on the evidence volume, not in `/tmp`.

These runs support the concrete reasoning, tool-boundary, cache-reuse and
accounting repairs. They do not establish universal coding-quality superiority,
reproducible completed-task parity, or release qualification. Preserve both
this pair and the prior, differently configured pair; do not pool their times
as interchangeable repetitions.

Provenance caveat: the round's `target.json` retains the stale inherited
descriptive version label `diagnostic-cache-fix-85ede4af`. The frozen binary
SHA-256 and explicit `cb7d619` / `f12b2d32` source commits above identify the
actual candidate; that inherited label does not. Preserve the original
manifest and this correction together rather than rewriting archived evidence.

Audit correction: the first writeup missed reference compaction request 068.
Enumerating all no-tools compaction requests identifies 020, 040, and 068.
The third used 63.716 seconds / 2,832 tokens and returned a useful handoff.
This corrects the compaction subtotal, not the overall elapsed/token totals
or acceptance scores.

### Reasoning usage and Codex compaction

A read-only audit matched the installed Codex 0.160.0 package to official
tag `rust-v0.160.0`, commit `a956835d020762cb2b570053af06f643a11c0ecc`.
Its [Responses usage conversion](https://github.com/openai/codex/blob/a956835d020762cb2b570053af06f643a11c0ecc/codex-rs/codex-api/src/sse/responses.rs)
copies `total_tokens` unchanged and stores reasoning-token detail separately.
Its [active-context calculation](https://github.com/openai/codex/blob/a956835d020762cb2b570053af06f643a11c0ecc/codex-rs/core/src/context_manager/history.rs)
uses the last total plus estimates for newer local items, not the reasoning
breakdown. A separate response-header switch concerns earlier encrypted
reasoning items; the inspected AFM responses contain plain summaries, not
encrypted content. Do not add that header as a speculative fix.

Concrete retained evidence: normalized AFM response 025 reported 27,061 input
plus 1,871 output = 28,932 total tokens, with reasoning detail zero. Request
026 was the no-tools compaction, matching the configured 28,000 threshold.
Thus zero reasoning detail is a reporting defect, not the cause of compaction
timing in this configuration when total usage is correct. Incorrect total
usage remains relevant, which is why the accepted-token counter fix matters.

### Identical-history latency replay and first-use sensitivity

The next diagnostic replayed six saved AFM histories and six saved reference
histories through both frozen engines, one request and one engine at a time.
Every model-normalized wire-request hash matched across engines. The candidate
was the accepted-token build, SHA
`91d8945c2932db1dab366285d3474fcd8148717cc3e5e1758d357e0985bc37c6`;
the reference and checkpoint were unchanged. MTP remained off, prefix caching
and thinking on, temperature 0, top-p 1, top-k 0, seed 123. The diagnostic capped
output at 512 tokens. No returned tool was executed and no generated response
was fed into the next captured history. These are latency probes, not completed
coding or quality scores. Selecting nonadjacent histories creates large prefix
jumps that differ from the original full-project cache trajectory.

Evidence directories under the root above:

- `orbitalMatchedAFM20261009`: AFM first, then reference, six AFM histories.
- `orbitalMatchedReference20261009`: reference first, then AFM, six reference histories.
- `orbitalMatchedAFMWarmRepeat20261009`: repeat of the six AFM histories, same
  binary and default residency policy, after the preceding runs.

The first AFM chain was strongly first-use sensitive:

| Saved AFM request | AFM first / warm-repeat wall (s) | Reference wall (s) | AFM / reference output tokens |
| --- | ---: | ---: | ---: |
| 001 | 93.742 / 6.202 | 6.388 | 67 / 69 |
| 002 | 7.897 / 3.030 | 1.886 | 121 / 61 |
| 012 | 34.745 / 9.668 | 9.732 | 103 / 104 |
| 013 | 9.765 / 8.981 | 4.058 | 512 / 233 |
| 030 | 33.380 / 13.483 | 12.839 | 512 / 512 |
| 031 | 9.181 / 9.011 | 8.233 | 512 / 512 |

All six AFM output arrays were identical after removing only generated item/call
IDs; their counts and input hashes also matched across the first and
warm-repeat chains. AFM's first 6,669-token prompt spent 92.578 seconds in
prefill, not model startup or decode. Its large-suffix request 012 spent 32.981
seconds in prefill and 1.687 seconds decoding. Cache-reuse lengths matched the
reference's physical server logs, including 6,638, 7,469, 17,371, 17,672 and
23,869 tokens. Reference API `cached_tokens: 0` does not mean it failed to reuse
those tokens. AFM warm request 031 is 9.44% slower in wall time; request 030 is
5.01% slower. These include prefill and are not pure-decode ratios.

The reverse-order chain produced these equal-output comparisons:

| Saved reference request | AFM wall (s) | Reference wall (s) | AFM latency difference | Outputs each |
| --- | ---: | ---: | ---: | ---: |
| 018 | 23.225 | 22.894 | +1.45% | 512 |
| 019 | 8.958 | 8.319 | +7.68% | 512 |
| 020, no-tools compaction | 29.454 | 28.983 | +1.63% | 512 |

The compaction input is not token-identical despite matching wire input:
AFM reports 27,435 versus 27,359 reference prompt tokens. A separate traced
capture localized all 76 extra AFM tokens to 19 empty historical thinking
blocks: `[248068, 271, 248069, 271]`, or `<think>\n\n</think>\n\n`, per older
assistant tool turn. There are no other token differences in that capture.
The reference defaults `preserve_thinking` to false; AFM leaves it undefined,
which this checkpoint's template treats as true. No default was changed to
imitate the reference. Evidence is in
`orbitalNoToolsPromptCapture20261009/audit-comparison/`; its traced diagnostic
reference executable is not the binary used for the timing table. The reference emitted its first
semantic reasoning chunk at 21.139 seconds, before completion at 28.983 seconds;
AFM buffered to completion. Tool-bearing responses in this diagnostic did not
show that semantic-streaming lead in either engine. Lifecycle first bytes must
not be counted as generated-token TTFT.

A source audit established a relevant policy difference, not complete causal
proof: AFM `mapped` residency does not warm the sidecar; the reference starts
background sequential warming by default, and its log records the 29.8 GiB
table warmed in 8.4 seconds. AFM already offers explicit
`--qwen-ngram-residency prewarm`, which waits for warming. Provider commit
`8dbe6a8c2031464cb283c177243076b43cf00074` replaced earlier unconditional
background warming with the explicit policy. No residency default was changed
in this qualification. Page residency, kernel preparation and filesystem/order
effects are not separately measured here; do not claim the whole cold delay
has a proven single cause or count startup warming as free performance.

The capped probes also exposed a Responses terminal-status defect: a salvaged partial tool
call at the output limit can be reported as completed. Preserve those records
as regression evidence; they are neither safe-to-execute tool calls nor a
512-token coding-quality result. A targeted length-status correction passed
the tests below. None of these diagnostics establish full-project or release
qualification.

### Token-limit status correction

The tool-call finalizer previously chose `tool_calls` before examining the
output cap; the non-streaming tool-response constructor also hardcoded that
reason. Consequently the Responses adapter could label a capped, salvaged
call as completed. The correction preserves `length` through finalization,
the tool-response constructor, the Responses resource, its output items and
its SSE terminal event. Arguments and reasoning remain available. Under-budget
calls and explicit-stop precedence keep their previous behavior.

The provider currently exposes a token count rather than exact EOS-versus-cap
termination evidence, so naturally complete calls at exactly the cap are
conservatively marked length, matching the existing text convention. Native
Qwen unfinished-envelope salvage remains unchanged. Accurate incomplete status
does **not** guarantee that clients will refuse to execute retained partial
arguments; changing that policy requires separate qualification.

Validation was sequential, through `Scripts/swiftpm-reliable.sh`:

- Restoring the old finalizer precedence reproduced four failing tests / 11
  failed assertions among seven regression tests (`tool-length-red.log`).
- Corrected consumer suite: 41 XCTest plus 15 Swift Testing cases passed,
  zero failures (`tool-length-green.log`), including eight new contracts.
- Provider constructors: three tests passed, zero failures
  (`tool-length-provider-green.log`).
- Independent review found no blocking issue; it prompted an added text-only
  incomplete-item regression. There are no new model forwards, tokenizations,
  GPU synchronizations, sampling or cache-policy changes.

The defaulted initializer parameter preserves ordinary source call sites, not
binary ABI or typed references to the previous initializer signature. These
repositories build the SwiftPM dependency from source. Integration still needs
an exact provider version bump.

The subsequent release build succeeded in 206.93 seconds from consumer
`17ee40c302c997b84e0693ac6a7fdfbd7b3f5c9e` and provider
`c01bfe4f08d6adab5eb10b4279c32d305144ffee`. Frozen executable:
`toolLengthRuntime20261009/afm`, SHA-256
`8483d51f434b9215dc101d7bb64a2ffdce3b0eb6f163ae33ea8e277ca1811f94`.
This is a paired-worktree qualification binary reporting `v0.9.20`, not a
published nightly or an immutable-provider release package.

Live replay (`orbitalToolLengthReplay20261009`) confirmed the saved request 013
now returns an incomplete resource and incomplete function-call item at 512
tokens, with identical arguments and reasoning. All six saved histories
preserved input hashes, token counts, generated content and cache reuse, apart
from generated IDs and the intentional status correction. The follow-on live
reasoning suite passed 27/27 assertions, zero failures or uncovered checks.

First-use prefill delays recurred after rebuilding: request 001 took 68.350 s
(67.170 s prefill, 1.102 s decode), and request 012 took 34.824 s. This run does
not qualify latency non-regression from its first-use samples. Request 031 took
9.166 s versus the prior AFM warm 9.011 s and reference 8.233 s: +1.73% versus
the previous AFM measurement and +11.33% versus reference, not a claimed 10%
gate pass. A one-second process sample during replay label 005 found the active
path in mapped n-gram gather/dequantization. Only two stack samples were
captured; this is not a complete profile or proof of all cold latency. That
request's timing is excluded explicitly in `diagnostic-interference.json`.

### Aborted browser-environment run

`orbitalConfirmed20261009` started with AFM using the unchanged local-preview
profile, checkpoint and fresh fixture. It was **aborted before the reference
started**. Neither final agent completion nor independent acceptance was
established; this is not a completed-project timing or parity result.

The local-preview preflight proved loopback HTTP access, not Chromium launch.
The coding agent's browser attempts encountered a missing expected browser,
macOS sandbox Mach-port permission failures and EGL initialization errors.
Meanwhile, persistent client instructions requested browser verification and
fixture instructions still said no network, with the localhost exception only
in the direct prompt. The agent spent substantial effort repairing this
environment instead of completing the application.

| Available phase | Requests | Generated tokens | Summed HTTP time (s) |
| --- | --- | ---: | ---: |
| Implementation through first successful build | 001–028 | 27,627 | 491.307 |
| Verification preparation / first compaction | 029–037 | 5,396 | 120.129 |
| Browser troubleshooting | 038–074 | 18,230 | 338.706 |
| Second compaction / recovery | 075–078 | 3,728 | 89.268 |
| All completed requests | 78 | 54,981 | 1,039.410 |

From the first blocked browser attempt onward, 41 requests consumed 21,958
tokens and 427.974 backend seconds. These are observed phases, not a claim that
all of that work was avoidable or that it measures engine throughput alone.
Both no-tools compactions (031 and 075) returned useful nonempty summaries and
no callable output. All 78 completed responses were HTTP 200/completed. Reported
cache reuse was 1,256,592 / 1,367,010 input tokens (91.92%). At the last completed
tool event, elapsed time was 1,094.105 seconds; that excludes in-flight work and
abort cleanup and is not completed-project time.

Scope contamination also occurred: request 048 invoked installed Chrome without
an isolated profile; request 064 attempted broad browser-name `pkill` commands.
Captured output does not establish whether unrelated processes received those
signals. The parent stopped only verified benchmark-owned processes; its
workspace cleanup reported no remaining processes and did not signal the user's
browser. The original workspace was archived outside the shared live project,
and `interruption.json` records the abort. No previous solution remains in the
next live project directory.

A new, separately labelled opt-in `isolated-cli-browser-independent-v1` profile
removes the instruction conflict without loosening application acceptance.
Both engines receive the same durable environment block, including compaction
requests: build the real application, run project tests/build/localhost HTTP,
do not search for or repair browsers, and leave actual browser evaluation to the
independent evaluator. Process cleanup is restricted in instructions to owned,
individually captured child PIDs. Those instructions are not claimed to be a
new OS-level isolation mechanism. Sixteen offline harness contracts passed.
The three acceptance script hashes are pinned to the prior frozen snapshot;
the 11 acceptance checks are unchanged. Original fixtures and old runs remain
untouched. New-profile results must not be pooled with the old profile as if
they were unchanged repetitions.

### Explicit prewarm diagnostic

`orbitalPrewarmProbe20261009` used the same frozen application binary and six
captured requests, with only `--qwen-ngram-residency prewarm` added to AFM.
The harness executed no generated tools and sent no extra warmup generation
requests. The existing engine startup behavior was retained. Startup to a
successful `/v1/models` response took 11.110 seconds, including a logged 7.5
seconds warming 29.8 GiB. All six request hashes, output arrays (apart from IDs)
and usage counts matched the preceding default-policy replay.

| Captured request | AFM prewarm wall (s) | Prior AFM warm wall (s) | Reference wall (s) | AFM / reference outputs |
| --- | ---: | ---: | ---: | ---: |
| 001 | 6.212 | 6.202 | 6.388 | 67 / 69 |
| 002 | 2.982 | 3.030 | 1.886 | 121 / 61 |
| 012 | 9.650 | 9.668 | 9.732 | 103 / 104 |
| 013 | 8.921 | 8.981 | 4.058 | 512 / 233 |
| 030 | 13.388 | 13.483 | 12.839 | 512 / 512 |
| 031 | 8.937 | 9.011 | 8.233 | 512 / 512 |

The six requests totaled 50.090 seconds, close to the earlier warm default
total of 50.374 seconds. The final two equal-output requests are 4.27% and 8.55%
slower than reference wall time. These are bounded latency probes, not coding
quality or pure-decode scores. The filesystem was already warm after prior
work; this run does not causally establish a cold-start improvement. No default
was changed. A source audit found no `F_NOCACHE` or analogous invalidation policy
in either sidecar's mapped/positional-read paths; warming policy remains a
relevant difference, not proof of the entire cold-prefill delay.

### Completed browser-independent Orbital comparison

`orbitalBrowserIndependent20261009` finished both engines sequentially under
`isolated-cli-browser-independent-v1`. The reference ran first; AFM followed
from the same empty fixture, not from the reference's solution. Both used the
same checkpoint, MTP off, prefix caching on, temperature 0, top-p 1, top-k 0,
seed 123 and medium reasoning. AFM retained default mapped n-gram residency.
The binaries were unchanged from the tool-limit qualification above. The
reference was release 26.10.1, executable SHA-256
`46d017b5f6e49890e654dbefd2b750eda44b627a20f4ec7a01015a4d0d8ea1e0`.

Independent audit confirmed equivalent initial workspace manifests and first
requests after removing engine/session identifiers. Their normalized request
SHA-256 was
`cb2be68491be823d5b4ef70c2f98a089738dacdfc5d44c045d1fe1074e25e543`.
No acceptance assertions were changed. No browser-repair detour, broad process
termination, or previous-solution access was observed. Both agents wrote some
preview logs/scripts under `/tmp`, a limited workspace-rule deviation. Both
owned-workspace cleanup records verified no remaining processes.

| Measurement | AFM | Reference |
| --- | ---: | ---: |
| Agent wall time (s) | 588.724 | 732.122 |
| Sum of HTTP request durations (s) | 559.482 | 708.013 |
| Other elapsed time (s) | 29.241 | 24.109 |
| Requests | 42 | 47 |
| Generated output tokens | 29,838 | 41,517 |
| Output tokens / summed HTTP second | 53.331 | 58.639 |
| First successful project build elapsed (s) | 265.200 | 490.484 |
| First-build output tokens | 15,094 | 27,569 |
| Physically reused / total input tokens | 729,134 / 797,463 | 827,392 / 904,107 |
| Physical cache reuse | 91.43% | 91.51% |
| Compactions | 1 | 1 |
| Compaction duration / outputs | 39.258 s / 1,153 | 42.425 s / 1,393 |
| Self-authored tests | 15/15 | 11/11 |
| Agent final answer | Present, 1,943 characters | Empty; incomplete |
| Original independent application checks | 8/10 executed; mobile not executed | 9/11 |

The AFM run used 19.6% less wall time than the reference run, which ended
without a final answer, while generating 28.1% fewer tokens. Its output per
summed HTTP second was 9.05% lower. These are different generated solutions and
conversation trajectories: neither rate nor the first-build improvement is a
matched-input raw-decode comparison. AFM's project-only server counters,
excluding the four-token startup warmup, give 66.680 seconds of prefill and
486.219 seconds of decode, or 61.367 decode tokens/second. The reference log
does not contain an equivalent decode-stage duration; no pure-decode ratio is
claimed. Reference API `cached_tokens` is still zero, but its server log
independently records the physical reuse in the table.

Both compactions returned useful nonempty summaries without callable output.
AFM request 026 supplied a 3,173-character handoff and preserved the persistent
environment restrictions. No recurrence of the earlier empty-handoff protocol
defect was seen in this pair.

The independent application failures are material:

- AFM initializes `last` with `performance.now()` and accepts a slightly older
  first animation-frame timestamp. Its negative delta reaches `advanceTime`,
  throws, and prevents the next frame from being scheduled. Time remains zero.
  Its successful pause assertion can consequently be vacuous; it is not proof
  of correct pause during motion. The original failure was reproduced by an
  unchanged evaluator rerun on the unchanged archived project.
- AFM's original nominal 8/11 includes an `evaluation setup/continuation` row:
  mobile screenshot capture timed out before the mobile-overflow assertion.
  This is **not** an executed mobile failure or pass. Its relationship to the
  stopped rendering remains incompletely established; do not automatically
  attribute it to unrelated infrastructure.
- Reference animation advances time despite its pause flag. Its other failed
  check is a lowercase selection value/diagnostic (`boreal` versus `Boreal`),
  not wholly broken visual selection. The task explicitly requires capitalized
  option values; the diagnostics field's casing is less explicitly specified.

A separate read-only mobile diagnostic found stable geometry and no horizontal
overflow on AFM's archived project. It also reproduced the negative-delta
exception. That diagnostic calls `__orbitDiagnostics()`, which itself renders,
and then captured a screenshot successfully in 0.370 seconds. Because it
forces rendering and has a longer capture deadline, it is not an unchanged
acceptance rerun and does not replace the original score. Source hashes before
and after the diagnostic matched. Evidence is retained under
`evaluation-rechecks/afm-01` and `evaluation-rechecks/afm-mobile-diagnostic`.

The reference's final request 047 spent 86.550 seconds generating 5,589 tokens
of repetitive reasoning. Its near-repeat detector ended generation, logging
`finish_reason=stop details=repetition_loop tier=near_repeat trim_start=1493`.
The wire nevertheless returned `status: completed`, no visible answer and no
repetition-stop cause. This was not a timeout or the harness output budget.
Source review confirms streaming Responses does not apply the logged tail
trim and maps this stop to completed; the client did not discard an answer
present on the wire. This does not prove AFM would generate the same loop from
the identical input.

AFM request 025 returned schema-invalid arguments wrapped under `arguments`
instead of top-level `cmd`; Codex rejected them. It cost 25.811 seconds and
1,520 output tokens. Generation versus parsing remains unattributed without
raw-output evidence. Preview-only work cost AFM 68.961 seconds / 3,548 tokens
versus reference 52.218 seconds / 2,989 tokens. These categories explain observed
work; subtracting them is not a verified counterfactual completion time.

This pair shows working cache reuse, useful compaction, and no recurrence of
the earlier large AFM elapsed-time deficit. It **does not establish successful
full-project quality parity or release qualification**. Both archived solutions,
all original reports, and the unsuccessful evaluator rerun remain intact.

### Native community 8-bit checkpoint

The downloaded checkpoint is
`/Volumes/edata/models/vesta-test-cache/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp`.
Its config SHA-256 is
`88793ae9905553224f329afde5859dfeb2e60401eb60e19dd0d6f222a3d7429d`.
All 36 indexed shards are present (194,858,557,249 bytes). The published HF
revision checked was `2a9de025436ea977720e4e3d6f369b185838e791`.
The repository does not publish `ngram_table.bin`; its PLE data is stored in
native quantized tensor shards. The pinned reference failed after weight
precomputation because it requires that sidecar. No paired 8-bit performance
result is available, and the user's instruction was to set aside conversion.

AFM initially rejected the shared `ngram_embedding.weight_scale` parameter.
Provider commit `283c3dd21bf7449b6405ccef80e79c007bf4a22c` adds shared-scale
loading and applies it once after row dequantization. This checkpoint's value
is BF16, shape `[1]`, exactly one. Twelve targeted tests passed, including
nested checkpoint loading, non-unit scales, strict legacy loading, q8 lookup
and the q4 CPU path. Independent source review found no blocking issue.

The release build took 115.97 seconds. Frozen executable
`sharedScaleRuntime20261009/afm` has SHA-256
`b361ca39d8072cee7d967a5f107e292625c65f51b7a116c04872643ae6daf661`.
It loads the untouched checkpoint and passed three API smoke requests covering
ordinary output, a growing cached conversation and a required tool call.
Growing-prefix reuse was 598 / 627 input tokens. Short-response decoding was
22.48–22.55 tok/s; reported peak MLX memory reached 180.3 GiB. The same smoke
with the 4-bit checkpoint passed, with 61.70–66.63 tok/s on its growing/tool
requests and 68.4 GiB peak MLX memory. These short samples and different storage
layouts do not establish a pure bit-width scaling law or coding-quality score.

`--mtp` on the native 8-bit checkpoint still fails preflight, reporting that no
compatible sidecar is resolved despite embedded MTP tensors. The current
validator assumes an older native MTP tensor layout; compatibility with the
checkpoint's heterogeneous quantized predictors remains unqualified. Do not
advertise this smoke as MTP support or vision-input qualification.

### Full 8-bit Orbital run and bounded optimization probes

`orbitalCommunity8bitCoding20261009` ran AFM alone with the same 20-minute
coding budget, fresh workspace and unchanged browser-independent acceptance
profile. MTP was off, prefix caching on, with no tuning environment settings.
The run timed out at 1,200.017 seconds with no final answer. Nine completed
responses delivered 22,423 tokens; request ten was interrupted and its saved
metrics report `BrokenPipeError`, without terminal usage. Do not equate the
delivered token count with every internally generated token. Summed HTTP
durations include post-timeout draining and exceed the coding wall limit, so
they must not be subtracted from that wall to produce a negative overhead.

The unfinished project passed only the two pure-math checks. Browser checks
found repeated `lastSelected` initialization errors and unavailable
diagnostics; mobile capture failed before the overflow check. The nominal
saved score is 2/11, including that setup/continuation row. This is an
unfinished-project outcome, not a completed coding-quality comparison.
Cleanup verified that the owned model, client and workspace processes stopped.
The app-server restart lost the original terminal handle but the OS processes
remained live; the run was monitored and was not restarted or duplicated.

The user proposed an 8-bit decode expectation near half the prior 4-bit rate,
approximately 30–33 tok/s. The 8-bit baseline remained around 22 tok/s.
The checkpoints have the same 48 layers, hidden width 2560, 512 experts,
10 routed experts/token and expert width 640, so that gap is not explained by
different expert counts or model depth.

Provider commit `ea0e1e3b2a284cb9b49090e93eda1bfc261e502c` extends the existing
resident row reader to q8/group-32 BF16 storage. Thirteen targeted tests passed,
including exact CPU/GPU equality across the actual 160-dimensional row width.
This is available through the existing explicit CPU lookup switch/profile;
the profile itself is opt-in. An earlier review confused the profile's
throughput defaults with ordinary serving defaults. A no-setting replay did
not activate CPU lookup and produced no speed improvement.

Controlled replays used the same captured Orbital requests 001 and 008, a
512-token cap, fresh sequential servers and no execution of generated tools.
All normalized request hashes and all output content (excluding generated IDs)
matched the baseline in both the CPU and fused-expert experiments below.

| Decode probe | Original GPU lookup | Explicit CPU row lookup | Experimental q8 expert fusion |
| --- | ---: | ---: | ---: |
| 245-output request: generation (s) | 11.084 | 10.855 | 10.535 |
| 245-output request: decode tok/s | 22.10 | 22.57 | 23.26 |
| 512-output request: generation (s) | 23.325 | 22.898 | 22.162 |
| 512-output request: decode tok/s | 21.95 | 22.36 | 23.10 |

CPU lookup improved these probes by about 2%, with no table copy or residency
change. That benefit does not justify a new default or explain most of the
gap. Its evidence is in `communityEightBitGPUReplay20261009`,
`communityEightBitCPUReplay20261009` (switch unset), and
`communityEightBitCPUEnabledReplay20261009` (switch one). The last folder has
an explicit provenance correction for a mistyped consumer commit; executable
identity, requests and results are unchanged.

The existing Qwen fused expert dispatcher accepts q4 only. Local experiment
`f592ea8a6949ed4662ed8c8ab2ddc502100356f8` adds an opt-in q8/group-64 scalar
decode path, using MLX's byte-wise dot/bias order and BF16 projection, activation
and weighted-reduction boundaries. Its real-dimension operator test passed a
2% normalized maximum-error bound against stock operations; the existing q4
test also passed. That tolerance test alone is not token or quality parity.
The captured whole-model output match is separate evidence. Its frozen binary
SHA-256 is `9480f8153e6b1f3b378d36668afb0778bbe473ec190e39a7ee80487087c207f7`.
The measured gain is only about 5%, still below the requested range. Review
prompted narrower geometry/expert-count guards; the tightened operator test
passed and the consumer release build passed (117.44 seconds). Default q8 fusion remains disabled. No new default or release was
published on the strength of these bounded probes.

Dispatch-only replay `communityEightBitDispatchDiagnostic20261009/` used provider
`12e10ec33c0e9c7a4635cebe7b09a15a0062bf72`, frozen binary SHA-256
`c29590da5a94a02a9fded1ee4ea7521a0d6cfba5fb9bb768b9d4a72a5ccbd937`,
the same captured first request, and a 16-token cap. With diagnostic logging and
experimental q8 fusion explicitly enabled, all 48 expert layers reported BF16
inputs, logits and scores, group 64, fused execution, and no MTP verification
policy. This rules out a silent expert-dispatch fallback for that probe; its
instrumented timing is excluded from performance comparisons.

Read-only source review also identified a distinct native-checkpoint fallback:
quantized HC injection prevents full HyperConnection fusion by default. The
source preserves this fallback because earlier fused reductions changed tool
decisions. No measurement yet attributes a fraction of runtime to this path,
and it has not been newly enabled. The requested 30–33 decode tok/s range remains
unmet; the measured q8 expert-fusion replay reached about 23 tok/s.

### Eight-bit gap isolation: controlled HC and combined experiments

All measurements below use frozen binary `c29590da5a94a02a9fded1ee4ea7521a0d6cfba5fb9bb768b9d4a72a5ccbd937`,
MTP off, prefix cache on, and the same saved agentic requests 001 and 008.
Each server runs alone. The 8-bit checkpoint and provider commit are unchanged
from the dispatch diagnostic above. Experimental flags are explicit, not defaults.

| 8-bit mode | First request decode tok/s | Growing-prefix request decode tok/s | Artifact folder |
|---|---:|---:|---|
| Ordinary, no tuning | 22.2 | 22.1 | `communityEightBitHCControl20261009` |
| Partial native HC normalization/mix | 23.7 | 23.6 | `communityEightBitHCEnabled20261009` |
| Partial native HC plus q8 expert fusion | 25.0 | 24.9 | `communityEightBitHCAndMoE20261009` |

The HC-only improvement is approximately 7%; combined improvement is approximately
13%. Both experiments preserve the two control outputs after normalizing generated
item/call identifiers and parsing JSON arguments. This is bounded replay evidence,
not broad quality qualification. The first request emits 245 tokens, the second
reaches its 512-token cap. All modes reuse 7,375 of 20,578 input tokens on request 008.
The prior quantized-HC quality concern remains; no default is changed.

Matched 4-bit checkpoint replays (`fourBitMatchedControl20261009`) using the same
binary and saved prompts decode at 60.4 tok/s on both requests. Its first output
is 87 tokens, not 245, so completion walls and quality are not equivalent across
quantizations. Second-request prefill is also distinct: 4-bit 54.38s versus
8-bit control 10.71s. Record this separately from decode and do not attribute
it to a decoded-token regression. Combined q8 is about 41% of this matched
4-bit decode rate; ordinary q8 is about 37%. Half-rate would be 30.2 tok/s.

Synchronization-heavy block profiles (`communityEightBitBlockProfile20261009`
and `fourBitBlockProfile20261009`) provide diagnostic graph evidence, not production
time attribution. They disable deferred HC scheduling and force a GPU evaluation
after every block. Last-ten-singleton median GPU operation counts are:

| Block | 4-bit diagnostic | 8-bit diagnostic |
|---|---:|---:|
| PLE | 56 | 200 |
| HC read | 288 | 2,016 |
| GDN | 288 | 288 |
| Attention | 397 | 397 |
| HC write | 96 | 672 |
| Routed/shared MLP | 720 | 1,104 |

These confirm additional native-checkpoint graph work in HC, MLP and PLE;
they do not prove these operation ratios survive normal compiled execution.
HC's real A/B gain is much smaller than its synchronized timing difference.
Next isolate stock q4/q8 projection costs at the actual 512-expert,
2,560/640-wide, ten-route geometry, and host graph/submission overhead before
claiming the remaining gap is exclusively quantization cost.

The cache-limit hypothesis is not a q4/q8 differentiator at this geometry.
Gate/up concatenation would retain approximately 900 MiB per layer for q4,
or 1,700 MiB for q8 (packed weights plus BF16 affine metadata). Both exceed
the same 512 MiB limit. Do not raise the limit across 48 layers merely to
test a proposed speedup; that would add large duplicate banks.

Host-only diagnostics preserve normal deferred scheduling and do not explicitly
synchronize after blocks. In `communityEightBitHostProfile20261009`, warmed
32-forward windows report approximately 38.6ms total, including 32.2ms in
submission. In `fourBitHostProfile20261009`, the second 32-forward window reports
13.77ms total, including 8.49ms in submission. `asyncEval` may wait inside MLX,
so submission is not pure CPU work; CPU call-stack capture is the next check.
Both runs use the same binary, prompt and 128-token cap; q4 stops after 87 tokens,
q8 reaches the cap. These are diagnostic windows, not equivalent coding outcomes.

The initial real-geometry stock projection microbenchmarks also expose an
important measurement limitation: per-call `eval` adds about 0.3ms of host and
synchronization overhead. Amortizing 32 independent singleton graphs in each
evaluation gives gate medians 0.105ms q4 / 0.109ms q8 and down medians
0.103ms q4 / 0.110ms q8. This synthetic single-bank probe is not a complete
model or serving-concurrency benchmark: it has different working-set behavior,
does not cover shared experts, HC, PLE, attention, or the vocabulary head,
and cannot establish an expected whole-model speed ratio. The initial pipeline
probe also did not squeeze its singleton output dimension before summing;
its pipeline result is excluded until the corrected diagnostic is rerun.

During this investigation, system swap usage was zero, pageouts/swapouts were
zero, and memory pressure was low. These snapshots do not support a system-wide
swap explanation, but do not rule out GPU allocation/residency or encoding costs.

The corrected operator diagnostic is committed in AFMKit as `b9eb5393`.
Both explicit q4/q8 processes passed; logs are `four-bit-projection-corrected-20261009.log`
and `eight-bit-projection-corrected-20261009.log`. Corrected stock expert-pipeline
amortized medians are 0.301ms q4 and 0.314ms q8. These synthetic numbers do not
include a whole-model working set or compiled model tails, and are not a
claim that q8 compute or bandwidth cost is only 4% higher in real inference.

Active decode stack capture succeeded under
`communityEightBitActiveDecodeSample20261009/afm-cache/decode-process-sample.txt`.
Its timestamp 18:42:12 precedes generation completion 18:42:22 and falls inside
the 11.21-second generation phase. The generation thread has 2,718 sampled
stacks: 1,984 enter `mlx_async_eval`, of which 1,502 reach the condition-variable
wait at `mlx/transforms.cpp:280` (`scheduler::wait_for_one`). This proves actual
scheduler waiting, not that the waiting is avoidable or caused solely by encoding.
Do not use the preceding capture that began after generation had completed:
the first visible function-call delta is too late to trigger this diagnostic.

Source inspection shows MLX's commit byte budget counts each referenced input's
whole allocation (`CommandEncoder::set_input_array`), not just the selected
expert rows. Ultra defaults are 50 ops or 50 MiB per command buffer. However,
the controlled `MLX_MAX_MB_PER_BUFFER=4096` A/B
(`communityEightBitEncoder4096MB20261009`) only improves decode to 23.0/22.9
tok/s from 22.2/22.1, with unchanged output lengths and cache-hit token counts.
Thus frequent commits contribute modestly; changing this budget alone does
not resolve the remaining gap. No scheduler limit or memory limit was patched.

### Whole-model isolation and negative experiments

The existing forward-only diagnostic now accepts the vision wrapper's text child
and supplies host token IDs, as AFM's iterator does. This removes HTTP, parsing,
sampling and detokenization, but retains the normal trunk and mutable caches.
Both checkpoints use a 512-token synthetic prefill and 32 fixed-token forwards.
This is a performance probe, not a coding or model-quality score.

| Mode | Full forward ms | Without vocabulary head ms | GPU ops/forward |
|---|---:|---:|---:|
| Fast q4 checkpoint, defaults | 14.448 | 13.542 | 1,370 |
| Native q8, defaults | 44.784 | 43.851 | 3,720 |
| Native q8, partial HC + expert fusion | 39.362 | 38.721 | 2,138 |
| Native q8, stock AR HC compilation prototype | 46.153 | 45.120 | 3,192 |
| Native q8, reverted control | 45.145 | 44.068 | 3,720 |

Logs respectively: `four-bit-whole-forward-benchmark-20261009.log`,
`eight-bit-whole-forward-benchmark-fixed-20261009.log`,
`eight-bit-fused-whole-forward-benchmark-20261009.log`,
`eight-bit-compiled-ar-hc-whole-forward-fixed-20261009.log`, and
`eight-bit-post-revert-whole-forward-20261009.log`.
Build/submission time includes GPU scheduler waits; final evaluation time is
another host wall interval, not a separate GPU hardware counter. The diagnostic
now labels these correctly. Head omission uses a continuing cache, not an
identical snapshot, so its roughly 1ms delta is approximate; the isolated head
probe independently agrees with that scale. The large slowdown persists without
the API or vocabulary head.

The reverted native control reports active MLX allocation 192,274,593,214 bytes,
peak 192,925,663,559, and allocation limit 522,268,023,193. Together with the
captured wait at `transforms.cpp:280`, this supports task-queue throttling rather
than exceeding the allocation limit. Increasing both encoder budgets to 4,096
MiB/1,000 ops did not resolve it: live decode remained 22.6/22.5 tok/s
(`communityEightBitEncoderBothBudgets20261009`). Waiting is evidence that work
has not completed, not proof that increasing a scheduler limit would help.

### What the actual kernels show

A temporary lookup-name hook in `metal/device.cpp`, bounded by explicit test
window markers, recorded 1,370 q4 and 2,138 partially fused q8 lookups per forward.
They agree with the command-buffer operation totals. The hook was removed;
no core logging or runtime default remains changed. Logs are
`four-bit-kernel-lookups-20261009.log` and
`eight-bit-fused-kernel-lookups-20261009.log`; these instrumented times are excluded.

The q4 path executes 96 compact HC-down and 96 compact HC-up/mix kernels per
forward. Native q8 executes **none** of those compact projection kernels, even
with `AFM_QWEN_FUSED_QUANTIZED_HC=1`. `Qwen4ExpHyperConnectionFusion.call` explicitly
returns a composed path for quantized injection: it fuses normalization and final
mixing but retains stock down/up/injection projections. This preserves MLX's
reductions because earlier custom replacements changed native tool decisions.
Thus "HC enabled" must not be presented as "full HC fusion enabled".

For the partially fused run, q8 has 713 ordinary quantized projection lookups
versus q4's 425, and 203 float32-to-BF16 copy lookups versus 11. These are stage
boundaries, not evidence that the entire model accidentally widens to FP32.
The remaining differences include native PLE row gathers/dequantization/scatter
instead of the q4 mapped-table path. Kernel counts locate work; they do not
assign a proportional share of wall time to it.

### Actual-bank component probes

All 96 real HC banks were measured with explicit pending injection, both eager
and compiled, using the same BF16 input. Reads are independent in this probe;
the full trunk's dependency and overlap behavior is intentionally not reproduced.

| HC mode | Eager sweep ms | Compiled sweep ms | Eager/compiled ops |
|---|---:|---:|---:|
| q4 compact path | 3.682 | 2.598 | 288 / 288 |
| Native q8 ordinary fallback | 25.142 | 4.195 | 2,688 / 1,632 |
| Native q8 partial fusion | 9.781 | 7.728 | 1,248 / 768 |

Logs: `four-bit-hc-banks-20261009.log`,
`eight-bit-default-hc-banks-equivalence-20261009.log`, and
`eight-bit-partial-hc-banks-equivalence-20261009.log`.
Compiled/eager output fields match bitwise (288/288) on the native test input.
An explicit partial-versus-stock check also matches 288/288 fields with zero
maximum difference (`eight-bit-hc-partial-vs-stock-20261009.log`). This is one
input with actual weights, not broad language/tool qualification.

The promising isolated stock-HC compilation was tested in the full model and
**regressed**, as the whole-forward table shows. That prototype was removed,
including its test-only model switch. It is not an implementation recommendation.
Independent component speedups cannot simply be added or projected into the trunk.

A dependent chain over the real 48 routed expert banks, with synthetic RMS
boundaries and dispersed fixed routes, distinguishes generic from specialized
execution. Logs are `four-bit-real-routed-chain-20261009.log` and
`eight-bit-real-routed-chain-20261009.log`.

| Routed chain | q4 ms | q8 ms |
|---|---:|---:|
| Stock eager | 21.572 | 22.220 |
| Stock compiled | 11.203 | 11.371 |
| Specialized fused eager | 4.243 | 5.226 |
| Specialized fused compiled | 4.153 | 5.145 |

The q4 specialized path is a serving default; q8 remains opt-in. These are not
complete decoder times: routes are fixed, normalization is synthetic, and router,
shared experts, HC, attention, and PLE are excluded. The contrast supports an
implementation-path penalty beyond a simple bit-width ratio, not a claim that
generic q8 should have the same latency as q4 in every real prompt.

### Scalar-construction experiment: correct but not faster

MLX Swift's BF16 scalar constructor schedules a Float32-to-BF16 cast. A temporary
opt-in CPU-leaf prototype bypassed it only for exactly representable normal
values/zeros, leaving nonfinite, subnormal, and nonexact values unchanged.
The scalar bit tests passed, and all 32 saved full-vocabulary logits tensors
matched bitwise against the original constructor.

Nevertheless, 245 removed operations did not improve full-model throughput:
native-leaf 45.776ms / 3,475 ops versus original 44.575ms / 3,720 ops. Logs and
captures are `eight-bit-native-scalar-full-forward-20261009.log`,
`eight-bit-stock-scalar-full-forward-20261009.log`, and their corresponding
`*-scalar-logits-20261009.safetensors` files. The prototype and its environment
switch were removed. The retained scalar test records bit-pattern compatibility;
no changed constructor or new default ships from this experiment.

### Interpretation and next implementation boundary

The supported evidence identifies a real execution-path difference: native q8
quantizes the tiny HC injection matrices, preventing the compact legacy HC path,
and lacks a default-qualified specialized expert path. Architecture fields agree
apart from the equivalent RoPE `type`/`rope_type` spelling. The labels also hide
mixed precision: q4 routers are quantized while native q8 routers are BF16; both
vocabulary heads are q8 (groups 64 and 128). Isolated router/head probes did not
show a large penalty from those latter differences.

Partial HC and expert optimizations recover about 12–13% end to end, but the
requested roughly half-q4 decode rate remains unmet. The analysis does **not**
claim exact additive production-time attribution for every remaining millisecond.
A meaningful next kernel experiment must retain native quantized projection
reductions and BF16 rounding boundaries while compacting HC execution, followed
by actual-model/API quality and latency qualification. Do not dequantize or
requantize the checkpoint, enable the old quality-changing fusion, or promote
the rejected compilation/scalar experiments merely to improve a benchmark.

### Outstanding checks

- Larger dashboard comparison is under `reasoningDashboard20261009/`. AFM exhausted the 80-request harness cap after 1,106.41 seconds and 55,292 output tokens; the reference completed in 215.35 seconds, 14 requests and 12,477 tokens. Both generated projects passed 15/16 acceptance checks. The 429 is a harness budget response, not server overload. This is a failed completion qualification, not a release-ready result.
- AFM compaction request 20 supplied `tools:[]`, but its response contained a structured tool call and no answer. Codex resumed with an empty handoff and reread files. Earlier archived AFM compactions produced nonempty handoffs, so do not claim this explains every previous timing gap.
- The request-local no-tools parser fix and fail-closed consumer guard subsequently passed 93 consumer XCTest tests plus 4 Qwen qualification tests. Replaying the failed compaction now preserves emitted markup as ordinary text, with no callable output. The provider/model still emits tool-like text rather than a useful summary; the reference produced a proper handoff. Protocol repair is not behavioral parity.
- That replay is recorded under `noToolsCompactionReplay20261009/`, AFM SHA `48dcac1da94e94af60ddd44472ff27c769e839bfcc902f724e521c9ea5825625`. The diagnostic bounded output at 2,048 tokens; both stopped below it. AFM: 26,952 input/661 output, 32.29s wall, 61.3 decode tok/s. Reference: 26,882 input/1,197 output, 39.60s wall. These walls include prefill and different generated lengths; do not equate them with quality or isolated throughput.
- Explain remaining full-project token-volume and elapsed-time divergence before claiming parity.
- AFM still reports zero reasoning-token usage despite delivering reasoning; accounting correctness is a separate remaining gap. Do not fabricate a token count from character length.
- Verify full release packaging and immutable AFMKit dependency pin before publishing. This diagnostic candidate uses paired local worktrees.
