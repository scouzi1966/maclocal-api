# ARCHITECTURE.md (v1)

## High-Level

```
VulcanBench/
├── harness/          # Python CLI + engine (Typer, Pydantic v2, Docker)
│   ├── cli.py
│   ├── agent/        # Tool ABC + executors (Docker + Local)
│   ├── sandbox/      # DockerToolExecutor (non-root, network-off)
│   ├── evaluator/    # functional + quality + security + efficiency + human_like
│   └── tracer/       # JSONL + self-contained replay.html
├── tasks/v1/         # 52 gold-verified suite tasks (+ hello-world demo)
├── dashboard/        # Next.js App Router (leaderboard, trace viewer, submit)
├── backend/          # FastAPI + SQLModel (filesystem or Postgres)
├── sandbox/          # Dockerfile.base + task template
├── scripts/          # validate_tasks.py, ingest_runs.py, etc.
└── docker-compose.yml (local Postgres; see docker-compose.prod.yml for full stack)
```

## Data Flow (one eval run)

1. CLI `run` → resolves task metadata + repo snapshot
2. Sandbox launches isolated container (workspace bind-mounted; commands in-container)
3. Agent loop calls standardized tools (list/read/edit/search/run)
4. Every event → append-only `trace.jsonl`
5. On done: verifier + analyzers + judges → scores → `summary.json` + `final.patch` + `replay.html`
6. Optional write-through: `POST /api/runs` when `VULCANBENCH_API_BASE` + `VULCANBENCH_API_TOKEN` are set
7. Leaderboard aggregates from `./runs/` or the database

## Tool Contract

Standardized across local/Docker/replay:

- `list_files`, `read_file`, `search_code` (ripgrep), `edit_file`, `run_command`, `run_tests`, `run_lint`, etc.
- Exposed as OpenAI function-calling schema for provider compatibility.

## Reproducibility

- Every trace embeds environment manifest (toolchains, sandbox image, model)
- `task_hash` detects stale runs when task definitions change
- `vulcanbench replay <run_id>` reconstructs the run timeline

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [METRICS.md](METRICS.md) for details.

## Non-Functional (enforced)

- Strict typing (Pydantic + mypy), Ruff zero-warn, ≥80% coverage on harness core
- Non-root containers, secret redaction on artifacts
- Self-contained HTML replays
