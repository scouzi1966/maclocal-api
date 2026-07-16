# VulcanBench

[![CI](https://github.com/morganlinton/VulcanBench/actions/workflows/ci.yml/badge.svg)](https://github.com/morganlinton/VulcanBench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

Fully open-source benchmarking for LLMs on realistic, multi-file software
engineering tasks. VulcanBench measures how models perform across reasoning
effort, language, codebase scale, and task complexity — with full traces,
reproducible scoring, and a local dashboard.

**v0.5.1** — 52 gold-verified tasks, tool-calling agent (mock / OpenAI /
Anthropic / Z.ai), Docker sandbox, pre-run cost estimates with bundled priors
(`vulcanbench estimate`), `v1-compare` suite, five-metric scoring, suite runs,
and HTML replay.
See [docs/QUICKSTART.md](docs/QUICKSTART.md) to get started.

## One-command setup

```bash
git clone https://github.com/morganlinton/VulcanBench.git
cd VulcanBench
make setup
source .venv/bin/activate
vulcanbench --help
```

Dashboard + backend (the dashboard reads live data from the API):
```bash
pip install -e ".[backend]"
uvicorn backend.app:app --port 8000          # serves ./runs at /api/*
cd dashboard && npm install && npm run dev    # http://localhost:3000
```
The dashboard falls back to a friendly empty state if the backend isn't running.
Point it elsewhere with `NEXT_PUBLIC_API_BASE` (see `dashboard/.env.example`).

By default the API reads `./runs/` directly. For a durable, queryable store, set
`DATABASE_URL` (Postgres or SQLite) and the API switches to a database —
`POST /api/runs` and `/api/feedback` become writable, and
`python scripts/ingest_runs.py` bulk-loads existing runs. A Postgres is provided
by `docker compose up db`.

## Example run

```bash
# Offline, deterministic (no API key) — drives the real agent loop end to end.
# Real runs default to the Docker sandbox; --sandbox local is fine for the
# deterministic mock model.
vulcanbench run --task hello-world --model mock:synthetic --sandbox local

# Any real model via the generic provider interface:
export OPENAI_API_KEY=...      # or ANTHROPIC_API_KEY=...
vulcanbench run --task hello-world --model openai:gpt-4o
vulcanbench run --task hello-world --model anthropic:claude-opus-4-8
vulcanbench run --task hello-world --model zai:glm-5.2

# Each run prints all five metrics + cost, e.g.:
#   functional=1.0 quality=1.0 security=1.0 human_like=0.8 total=0.974 cost=$0.0
# and writes ./runs/<id>/{trace.jsonl, summary.json, replay.html, final.patch}

# Run a whole suite: repeat for signal, parallelize, and cap the spend.
vulcanbench run --suite v1 --model openai:gpt-4o --repeat 5 --max-concurrency 4 --max-cost 20.00

# Compare normalized reasoning effort on the same suite/model.
vulcanbench run --suite v1 --model openai:gpt-4o --effort low
vulcanbench effort-sweep --suite v1 --model openai:gpt-5.1 --efforts low,medium,high --repeat 3 --sandbox docker

# Fast micro/small sweep vs navigation-heavy medium/large tasks:
vulcanbench run --suite v1-micro --model openai:gpt-4o
vulcanbench run --suite v1-large --model openai:gpt-4o --repeat 5 --sandbox docker
vulcanbench leaderboard            # by model: pass@1 ± stderr, pass@k, cost, latency
vulcanbench leaderboard --by run   # per-run drill-down
vulcanbench report -o report.md    # shareable Markdown/JSON report (ranking,
                                   #   effort sensitivity, per-task breakdown,
                                   #   environment, drift flags)
vulcanbench calibrate              # empirical difficulty calibration from recorded runs
vulcanbench replay <id>

# Runs execute in an isolated container by default (see Sandbox below);
# build the image once with `make sandbox-image`.

# Use it as a CI regression gate (threshold must be in [0, 1]):
vulcanbench run --suite v1 --model openai:gpt-4o --repeat 5 --fail-under 0.8
```

The gate **fails closed**: it exits `4` if pass@1 is below the threshold, if
pass@1 is unavailable, *or if any suite run errored* — a CI gate never goes
green on a partial or unknown result.

Exit codes: `0` ok · `1` usage/error · `2` provider · `3` sandbox · `4` gate
failed (below `--fail-under`, or a run errored).

`final.patch` is a real `git diff` of the agent's edits; `replay.html` is fully
self-contained (open in any browser). Use `--no-judges` to skip the LLM judge
ensemble, `--timeout SECONDS` to cap a run's wall-clock. Traces, summaries, and
patches are secret-redacted and size-capped before they're written, so run
artifacts are safe to publish. See `make ci`, `make docker-up`, docs/.

## Models

Specify a model as `provider:model`:

- `mock:synthetic` — deterministic, offline; used by tests and demos.
- `openai:<model>` — OpenAI Chat Completions for normal runs, or the Responses
  API when `--effort` is supplied. Needs `OPENAI_API_KEY`.
- `anthropic:<model>` — Anthropic Messages API. Needs `ANTHROPIC_API_KEY`.
- `zai:<model>` — Z.ai (Zhipu) OpenAI-compatible Chat Completions API. Needs
  `ZAI_API_KEY`. Reasoning effort is not supported; `--effort` is recorded as
  metadata only.

`--effort` accepts `low`, `medium`, `high`, or `extra-high`. OpenAI runs map it
to the Responses API `reasoning.effort` field; Anthropic runs map it to the
Messages API `output_config.effort` field. `extra-high` maps to `xhigh` on both
providers and is opt-in for sweeps because support is model-dependent (e.g.
Claude Opus 4.7+). Mock and Z.ai runs accept the field as no-op metadata. Effort
labels are each provider's own scale — a cross-provider comparison at the same
label compares each model at its own setting, not a calibrated equivalence.

## Sandbox

The agent's tool execution can run in an isolated Docker container instead of on
the host:

```bash
# Build the base image once (git, ripgrep, ruff, bandit, radon, pytest):
docker build -t vulcanbench/sandbox:base -f sandbox/Dockerfile.base .

vulcanbench run --task hello-world --model openai:gpt-4o
# --sandbox local|docker|auto   (default: docker)
# --image vulcanbench/sandbox:base   (default: per-task metadata or vulcanbench/sandbox:base)
# --network                     (off by default; opt in for dependency installs)
```

- `docker` (default) runs tools in a non-root, **network-off**, resource-limited
  container (workspace bind-mounted, cleaned up after each run). It errors out
  if the daemon is unreachable — it never silently falls back to host execution.
- `local` runs the model's commands directly on the host — fast and Docker-free,
  but unsandboxed; opt in deliberately (fine for `mock:synthetic` and trusted
  dev loops).
- `auto` uses Docker when available. Falling back to host execution additionally
  requires `VULCANBENCH_ALLOW_HOST_EXEC=1`; otherwise it errors out.

File operations (read/edit/search) always run host-side over the shared mount;
command execution (`run_command`/`run_tests`/`run_lint`) **and the functional
verifier** run inside the container, so the whole run is reproduced in one
isolated environment. Build the all-language image with `docker build -t
vulcanbench/sandbox:base -f sandbox/Dockerfile.base .` (Python + Go + Node).

## Tasks

The `tasks/v1/` suite holds **52** multi-file, gold-verified tasks across Python,
Go, TypeScript, and Rust, plus the `hello-world` demo. Each task ships a starting
`repo/`, **hidden** `tests/` (never shown to the agent), declarative
`fail_to_pass`/`pass_to_pass` test commands in `metadata.json`, and a
`gold_patch.diff` reference solution.

The corpus spans difficulty (`easy` / `medium` / `hard`), `repo_scale`
(`micro` / `small` / `medium` / `large`), and `task_complexity` (`localized`,
`multi_file`, `system`, `architecture`) so it discriminates between weak and
strong models and shows when higher reasoning effort matters. Examples: an RFC
6901 JSON Pointer resolver
(`py-jsonpointer`, hard), a race-free bounded worker pool verified under
`go test -race` (`go-worker-pool`, hard), and a prototype-pollution-safe deep
merge (`ts-deep-merge`, hard).

```bash
make validate-tasks                              # validate every task
vulcanbench validate-task tasks/v1/<id>          # one task
```

Validation proves each task is real: the gold patch must solve it
(`functional == 1.0`), the `fail_to_pass` tests must genuinely fail *before* the
fix, and scoring must be deterministic over repeated runs.

**Provenance is labeled and checked.** Every task declares `source`
(`hand-authored` or `oss`) and an explicit `decontaminated` boolean. Hand-authored
tasks are written now (post-cutoff, so `decontaminated: true`); the validator
enforces that. An `oss` task (e.g. `oss-inflection-titleize`, sourced verbatim
from a real MIT-licensed repo with its LICENSE preserved) is honestly labeled
`decontaminated: false` — its fix predates model cutoffs — and the
`vulcanbench report` integrity section flags every run scored against it. Scaffold
one with `python scripts/import_oss_issues.py`. Format details:
[docs/TASK_CONTRIBUTION.md](docs/TASK_CONTRIBUTION.md).

## Architecture & Reproducibility

- Standardized tools (list/read/edit/search/run) behind one protocol, with
  interchangeable local and Docker executors (see [Sandbox](#sandbox))
- Every step captured as JSONL (llm, tool, diff, test, metric) + token usage
- Each run records its `vulcanbench replay <id>` command for reproduction
- Docker sandbox runs untrusted command execution in a non-root, network-off,
  resource-limited container

Full details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/METRICS.md](docs/METRICS.md), [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

## Documentation

| Doc | Purpose |
|-----|---------|
| [QUICKSTART](docs/QUICKSTART.md) | Setup, smoke test, first real run |
| [METRICS](docs/METRICS.md) | How the five scores are computed |
| [DEPLOYMENT](docs/DEPLOYMENT.md) | Hosted API + dashboard (optional) |
| [CONTRIBUTING](docs/CONTRIBUTING.md) | Add tasks, run CI locally |
| [ROADMAP](docs/ROADMAP.md) | Planned follow-ups |

Quality and security analyzers run when the relevant toolchains are on your
host (e.g. `bandit` for Python via the venv; `gosec` for Go if installed).
Otherwise those metrics report `null` with a reason — never a fabricated score.
Use `--no-judges` to skip the LLM judge ensemble and cut cost roughly threefold.

## License

Apache 2.0 (see LICENSE and NOTICE).

## Provider terms & data usage

VulcanBench is an independent evaluation harness. A few boundaries keep its use
consistent with the model providers' terms — please read these before running or
publishing results.

- **You bring your own keys, under your own agreement.** VulcanBench never
  bundles or shares API credentials. Each run uses the keys in your environment
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ZAI_API_KEY`), so every call is made
  under *your* account and *your* commercial/API agreement with that provider.
  You are responsible for staying within your provider's terms and usage
  policies.

- **Outputs are for evaluation, not training.** Recorded run artifacts (traces,
  patches, summaries) capture model outputs solely for scoring, inspection, and
  reproducibility. Both OpenAI and Anthropic prohibit using their outputs to
  develop or train competing models — do not use VulcanBench artifacts, or any
  published corpus of them, for that purpose. VulcanBench intentionally has no
  "export outputs as a training dataset" feature.

- **Trademarks & independence.** "OpenAI" and "GPT" are trademarks of OpenAI;
  "Anthropic" and "Claude" are trademarks of Anthropic; "Z.ai" and "GLM" are
  trademarks of Zhipu AI. VulcanBench is not affiliated with, sponsored by, or
  endorsed by any of these companies. Model and provider names are used only to
  identify the systems under test.

This is not legal advice; consult the current provider terms for authoritative
guidance.
