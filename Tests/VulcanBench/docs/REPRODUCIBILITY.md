# Reproducibility (v1)

Every run is designed to be inspectable and replayable without hidden state.

## Artifacts

Each run directory under `./runs/<run_id>/` contains:

| File | Purpose |
|------|---------|
| `trace.jsonl` | Append-only event log (LLM turns, tool calls, diffs, metrics) |
| `summary.json` | Final scores, manifest, cost, duration, task hash |
| `final.patch` | Git diff of agent edits (redacted + size-capped) |
| `replay.html` | Self-contained browser replay |

## Manifest

`summary.json` includes a `manifest` recording:

- Python/platform versions
- Sandbox mode, image name, network setting
- Model and judge model
- Toolchain versions (git, ruff, bandit, go, node, etc.)
- Task metadata snapshot (scale, complexity, languages)

## Task content hashing

`task_hash` in the summary is a deterministic SHA-256 of scoring-relevant task
files (repo, tests, issue, metadata tests, gold patch). The leaderboard and
report flag runs scored against a stale task definition.

## Replay

```bash
vulcanbench replay <run_id>
```

Opens or regenerates the HTML trace viewer. The summary's `replay_command` field
records the exact command.

## Sandbox isolation

Default `--sandbox docker` runs command execution and the functional verifier
inside a non-root, network-off container. File reads/edits use a host bind mount
over the same workspace path.

## Publishing artifacts

Traces, summaries, and patches pass through secret redaction before write
(`harness/redaction.py`). Safe to share for evaluation; do not use outputs to
train competing models (see README provider terms).
