# Deployment (v1)

This guide covers hosting the VulcanBench dashboard and API for a public
leaderboard. The harness CLI (`vulcanbench run`) typically runs on evaluator
machines and optionally posts summaries to the API.

## Architecture

```text
Evaluators (CLI)  --POST /api/runs-->  Backend (FastAPI)  -->  Postgres
                                              ^
Dashboard (Next.js) ---- GET /api/* ----------+
```

Run artifacts (`trace.jsonl`, `replay.html`, patches) stay on disk under
`VULCANBENCH_RUNS_DIR` (default `/data/runs` in Docker). The database stores
queryable summaries for the leaderboard.

## Quick start (Docker Compose)

1. Copy `.env.example` to `.env` and set strong values for:
   - `POSTGRES_PASSWORD`
   - `VULCANBENCH_API_TOKEN`
   - `VULCANBENCH_CORS_ORIGINS` (your dashboard origin, e.g. `https://bench.example.com`)

2. Build and start:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

3. Open the dashboard at `http://localhost:3000` (API at `:8000`).

4. On evaluator machines, enable write-through:

```bash
export VULCANBENCH_API_BASE=https://api.bench.example.com
export VULCANBENCH_API_TOKEN=<same token as server>
vulcanbench run --task hello-world --model mock:synthetic --sandbox local
```

## TLS and secrets

- **TLS**: terminate HTTPS at a reverse proxy (nginx, Caddy, cloud LB). Do not
  expose Postgres publicly.
- **API token**: required for `POST /api/runs` and `/api/feedback`. Without it,
  writes return HTTP 503.
- **CORS**: set `VULCANBENCH_CORS_ORIGINS` to your dashboard origin(s) in
  production (avoid `*` when credentials matter).

## Database migrations

The backend container runs `alembic upgrade head` on start. For manual upgrades:

```bash
DATABASE_URL=postgresql+psycopg://... alembic upgrade head
```

Initial schema matches `backend/db.py` (`run`, `feedback` tables).

## Bulk ingest (legacy runs)

```bash
DATABASE_URL=... python scripts/ingest_runs.py ./runs
```

## PyPI install (CLI only)

```bash
pip install vulcanbench
```

The PyPI wheel ships the harness and backend packages. Clone the repository for
the full task corpus, sandbox Dockerfiles, and dashboard.

## Health check

```bash
curl -s http://localhost:8000/api/health
# {"status":"ok","store":"db"}
```
