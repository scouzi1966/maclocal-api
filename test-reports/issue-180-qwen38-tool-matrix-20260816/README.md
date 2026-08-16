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
