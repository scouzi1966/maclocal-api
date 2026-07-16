# VulcanBench — Roadmap

Status as of **2026-06-19**. v1 MVP is **shipped** — real agent loop, Docker
sandbox, 52-task suite, five-metric evaluator, dashboard, and optional Postgres
persistence. This document tracks remaining work toward a hosted launch and v1.1.

Legend: ✅ done · 🟧 in progress / next · 🟨 polish · 🔮 future

---

## Shipped in v0.1.0

- ✅ Generic LLM providers (mock, OpenAI, Anthropic) with retry/timeout
- ✅ Tool-calling agent loop + git-diff patch capture
- ✅ Docker sandbox (default) + local/auto modes
- ✅ Five-metric evaluator + LLM judge ensemble
- ✅ 52 gold-verified tasks (Python, Go, TypeScript, Rust)
- ✅ Suite runs, effort sweeps, `--fail-under` CI gate, `--max-cost` cap
- ✅ Leaderboard, report, calibration, HTML replay
- ✅ FastAPI backend + Next.js dashboard (live API fetch)
- ✅ Optional Postgres via `DATABASE_URL`; manual + automatic ingest
- ✅ Task validation in CI; contributor docs and templates

---

## Launch follow-ups (v0.1.x)

### Tier A — OSS announcement ✅ (this release)

- ✅ Documentation reconciliation (QUICKSTART, ROADMAP, ARCHITECTURE, CONTRIBUTING)
- ✅ `dashboard/.env.example`, root `.env.example` completeness
- ✅ CHANGELOG + release workflow
- ✅ Secret redaction covers `VULCANBENCH_API_TOKEN`

### Tier B — Hosted stack 🟧

- ✅ Production Dockerfiles (backend + dashboard)
- ✅ `docker-compose.prod.yml` with backend, dashboard, Postgres
- ✅ CLI write-through to `POST /api/runs`
- ✅ Alembic initial migration
- ✅ MinIO removed from default dev stack (unused)
- 🟨 Production TLS termination, rate limits, monitoring (operator responsibility — see `docs/DEPLOYMENT.md`)

### Tier C — Distribution 🟧

- ✅ GitHub Release workflow on tag
- ✅ PyPI publish workflow (opt-in via repository secret)
- 🟨 Promote PyPI classifier from Beta when install-from-PyPI is validated

---

## v1.1 backlog

| Priority | Item | Notes |
|----------|------|-------|
| 🟧 | Java runtime + CI + tasks | Analyzers exist; no Java tasks yet |
| 🟧 | Provider streaming | Non-streaming complete() today |
| 🟧 | Task-PR validation CI | Auto-label + validate changed tasks on PR |
| 🟨 | Semantic `search_code` | ripgrep only; `semantic` flag reserved |
| 🟨 | `expected_metrics.json` baselines | Comparison not implemented |
| 🟨 | Non-Python quality depth | Go/TS/Rust partially covered |
| 🟨 | Multi-arch sandbox CI matrix | ARM64 + x86 gold-patch validation |
| 🟨 | Supply-chain scan in CI | cargo-audit only today |
| 🟨 | `docs/SPEC.md` | Formal benchmark spec (optional) |
| 🟨 | Dashboard automated tests | Lint + build only in CI |
| 🟨 | Docker integration test in default CI | Live test gated behind `@pytest.mark.docker` |

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) and [TASK_CONTRIBUTION.md](TASK_CONTRIBUTION.md).
