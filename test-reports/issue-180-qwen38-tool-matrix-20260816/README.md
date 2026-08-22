# Qwen 3.8 Tool-Calling Qualification Artifacts

These artifacts support issue #180 and commit `427b8b3` on
`codex/fix-issue-180`.

## Scope

- Model: `mlx-community/Qwen3.8-27B-4bit`
- Benchmark: Berkeley Function Calling Leaderboard (BFCL) V4
- Selected categories: `simple_python`, `multiple`, `parallel`,
  `parallel_multiple`, and `irrelevance`
- Cases per category: 5 (25 total per matrix cell)
- Matrix: MTP off/on crossed with no-thinking, low, high, and max reasoning
- Sampling: temperature 0, maximum 512 output tokens

## Results

All eight matrix cells scored 24/25 (96%) across the selected categories:

- `simple_python`: 5/5
- `multiple`: 5/5
- `parallel`: 4/5
- `parallel_multiple`: 5/5
- `irrelevance`: 5/5

Every cell missed the same `parallel_3` case. The three calls were structurally
valid and parsed, but one or more argument values were semantically expanded
beyond BFCL's accepted exact values. This is a model argument-selection miss,
not a parser, streaming, transport, or schema failure.

The BFCL-generated overall CSV includes categories that were intentionally not
run. Its 78.33% aggregate is therefore not the score for this focused matrix.
Use `matrix/summary.csv` and the five selected category score files instead.

## Layout

- `matrix/summary.csv`: selected-category score and AFM throughput summary
- `matrix/<cell>/results`: raw BFCL model responses
- `matrix/<cell>/scores`: BFCL category scores
- `matrix/<cell>/server.log`: AFM startup and request logs
- `matrix/<cell>/bfcl.log`: BFCL generation/evaluation logs
- `swift-tool-qualification.log`: focused Release test run (143/143 passed)

The original persistent run remains under
`/Volumes/edata2/afm-benchmarks/qwen38-tool-matrix-20260816-084142` on the test
machine.

## Extended Real-Tool Coverage

The qualification now also exercises controls and schemas used by real coding
agents:

- named `tool_choice`
- `parallel_tool_calls: false`
- OpenCode `grep` arguments, including arrays and optional paths
- Pi `write` arguments, including camelCase keys and multiline Unicode
- OpenClaw `apply_patch` multiline unified diffs
- Hermes nested `todo` arrays and nullable values

The focused Release qualification contains 15 tests and passes 15/15. The live
Promptfoo matrix runs 106 cases under each of the default, adaptive XML, and
adaptive XML plus grammar profiles. It passes 295/318 overall (92.77%):

| Suite | Passed | Total |
| --- | ---: | ---: |
| Core tool calling | 39 | 39 |
| Agentic coding workflows | 12 | 12 |
| Framework schemas | 22 | 24 |
| OpenCode tools | 105 | 111 |
| Pi tools | 51 | 60 |
| OpenClaw tools | 30 | 36 |
| Hermes tools | 36 | 36 |

Raw live exports and a failure extract are stored at
`/Volumes/edata2/afm-benchmarks/issue180-tool-coverage-20260816/live-all-profiles`.

### Failure Attribution

None of the 23 remaining Promptfoo misses shows an AFM transport, SSE, JSON,
stream assembly, tool-call parser, or crash failure. AFM returned structured
`tool_calls` for every miss except the truncated `todowrite` responses.

- 20 are model policy or semantic argument choices. Examples include exploring
  with `ls` or `find` instead of immediately building, reading through a shell
  instead of the requested `read` tool, using `cat README.md` as an
  `apply_patch` argument, omitting a requested timeout, and choosing
  `skillCommands.enabled` instead of the expected `enableSkillCommands` key.
- 3 are the same verbose OpenCode `todowrite` generation reaching
  `finish_reason: length`. Increasing the case budget from 400 to 1024 did not
  make this model concise enough to complete the call. This is model behavior
  exposed by a finite harness budget, not an AFM parsing limitation.

The v0.9.16 release report's 41 Promptfoo misses also included approximately 15
false negatives from over-strict assertions. The updated harness now accepts
valid multiple search/read calls and conventional unified diff syntax. The
remaining failures are useful behavioral quality measurements and should not be
relaxed into passes.

## Independent Review Remediation

An independent review of PR #186 identified two streaming defects and three
assertion gaps. The follow-up fixes:

- preserve adjacent parallel tool calls emitted in one stream chunk;
- salvage only the final unclosed parameter and schema-coerce its wire delta so
  it matches the repaired final call;
- require every agentic search/read call to carry a relevant query or path;
- validate all advertised OpenCode and Pi schema fields; and
- require an actual README hunk with the requested replacement in the OpenClaw
  `apply_patch` case.

The expanded Release streaming regression set passes 29 XCTest and 28 Swift
Testing cases with zero failures. The affected live Promptfoo suites were rerun
with `mlx-community/Qwen3.8-27B-4bit` across all three parser profiles:

- core tool calling: 39/39;
- agentic coding workflows: 12/12;
- combined reviewer-remediation matrix: 51/51.

The raw rerun exports are stored at
`/Volumes/edata2/afm-benchmarks/issue180-review-fixes-20260816` on the test
machine.

The complete Release test suite was also rerun after these fixes. It passed:

- XCTest: 665 tests, 3 intentional skips, 0 failures;
- Swift Testing: 435 tests across 34 suites, 0 failures.

The captured full-suite log is stored at
`/Volumes/edata2/afm-benchmarks/issue180-review-fixes-20260816/full-release-tests.log`.
