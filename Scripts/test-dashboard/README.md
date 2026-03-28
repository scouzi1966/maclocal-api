# AFM Test Dashboard

Web-based test driver and result viewer for validating AFM binaries.

## Quick Start

```bash
python3 Scripts/test-dashboard/server.py
# Opens browser at http://localhost:8080
```

## What It Does

1. **Select** a binary, model, and test suites
2. **Run** pre-flight safety checks (relocated binary, Bundle.module audit, metallib)
3. **Watch** live test progress via Server-Sent Events
4. **Review** results with per-test drill-down and failure details
5. **Export** structured JSONL logs for coding agent investigation

## Architecture

- `server.py` — Python HTTP server (uses `openai` SDK when available, falls back to raw HTTP)
- `index.html` — Self-contained SPA (vanilla HTML/CSS/JS, no build step)

The server spawns test scripts as subprocesses, parses their output, and streams events to the browser via SSE.

## Ports

| Service | Port | Purpose |
|---------|------|---------|
| Dashboard | 8080 | Web UI + API |
| AFM (assertions) | 9998 | Server for assertion tests |
| AFM (batch/promptfoo) | 9999 | Server for batch and promptfoo tests |
| AFM (smart analysis) | 9877 | Server for comprehensive smart analysis |
| Promptfoo UI | 15500 | Interactive promptfoo result viewer |

## Logs

Structured JSONL logs are written to `test-reports/dashboard-logs/`. A symlink `LATEST.jsonl` always points to the current run.

Monitor from a coding agent:
```bash
tail -f test-reports/dashboard-logs/LATEST.jsonl | jq 'select(.type == "test_result")'
```

## Dependencies

```bash
pip install openai  # recommended — used for health checks and preflight validation
```

The server works without `openai` (falls back to raw HTTP), but with it installed, every interaction with AFM goes through the same SDK that users use — making the dashboard itself an API compatibility test.

## Options

```bash
python3 Scripts/test-dashboard/server.py --port 8080  # custom port
```
