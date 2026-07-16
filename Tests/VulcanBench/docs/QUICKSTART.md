# QUICKSTART.md

One-command local setup and first run (v1).

## Prerequisites
- Python >=3.12
- Node.js >=20 (for dashboard)
- Docker Desktop (for real sandbox runs; optional for dry-run stubs)
- Git + Git LFS (`git lfs install`)

## Setup

```bash
git clone https://github.com/morganlinton/VulcanBench.git
cd VulcanBench
make setup
source .venv/bin/activate
vulcanbench --help
```

Dashboard:
```bash
cd dashboard
npm install
npm run dev
# open http://localhost:3000
```

## Smoke test (offline, free)

`mock:synthetic` is deterministic and free — use it to confirm the harness runs
end-to-end before spending any tokens. `--sandbox local` skips Docker, which is
fine here because the mock model's commands are canned (real models default to
the Docker sandbox):

```bash
vulcanbench run --task hello-world --model mock:synthetic --sandbox local
vulcanbench run --suite v1-micro --model mock:synthetic --no-judges --sandbox local
```

## Your first real run

**1. Build the sandbox image** (real runs execute model-written shell commands —
run them in Docker, not on your host):

```bash
make sandbox-image            # builds vulcanbench/sandbox:base (Python+Go+Node)
```

**2. Set the provider key** for the model you'll test:

```bash
export OPENAI_API_KEY=sk-...           # for openai:* models
export ANTHROPIC_API_KEY=sk-ant-...    # for anthropic:* models
export ZAI_API_KEY=...                 # for zai:* models (GLM)
```

**3. Start small and cheap** — one task, in Docker (the default), judges off,
with a spend cap:

```bash
vulcanbench run --task py-topo-sort-cycle \
  --model openai:gpt-4o-mini \
  --no-judges
vulcanbench run --task py-topo-sort-cycle \
  --model zai:glm-5.2 \
  --no-judges
vulcanbench replay --run-id <latest>   # self-contained HTML trace of the run
```

**4. Scale to the whole suite** with repeats (for pass@k ± stderr), bounded
parallelism, and a hard spend cap, then build a shareable report:

```bash
vulcanbench run --suite v1 \
  --model anthropic:claude-sonnet-4-6 \
  --sandbox docker --repeat 3 --max-concurrency 4 --max-cost 5.00
vulcanbench report --suite v1 -o report.md
```

> **Cost & safety notes**
> - Estimate spend before a benchmark (uses local `./runs` history when present,
>   bundled priors on fresh installs):
>   `vulcanbench estimate --suite v1-compare --model openai:gpt-5.5`
> - Preflight task health before a full suite spend:
>   `make validate-tasks-docker` (gold + verifiers inside Docker; builds base + Rust images)
> - `--judges` is **on by default** (a 3-model `human_like` ensemble reusing the
>   run model) — it roughly triples token cost/latency. Use `--no-judges` for
>   cheap functional-only runs.
> - `--max-cost` is a soft cap that stops launching new runs (suite runs only)
>   and requires a priced model; cost/latency are recorded per run regardless.
> - Default `--sandbox docker` runs the agent's shell commands in an isolated
>   container. Use `--sandbox local` only for trusted dev loops (e.g.
>   `mock:synthetic`). Override prices any time with
>   `VULCANBENCH_PRICING=/path/to/prices.json`.

See the full example in the README and `docs/ARCHITECTURE.md`. To add your own
task: `docs/TASK_CONTRIBUTION.md`.
