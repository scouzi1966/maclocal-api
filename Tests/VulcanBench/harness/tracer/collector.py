from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.redaction import sanitize


class TraceCollector:
    def __init__(self, run_dir: Path, run_id: str, task_id: str, model: str) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.run_id = run_id
        self.task_id = task_id
        self.model = model
        self.step = 0
        self.events: list[dict[str, Any]] = []

    def record(self, event_type: str, data: dict[str, Any]) -> None:
        self.step += 1
        ev = {
            "ts": datetime.now(UTC).isoformat(),
            "step": self.step,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model": self.model,
            "type": event_type,
            # Redact secrets and cap field sizes before anything touches disk.
            "data": sanitize(data),
        }
        self.events.append(ev)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

    def finalize(
        self, scores: dict[str, Any], extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        summary = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model": self.model,
            "steps": self.step,
            "scores": scores,
            "artifacts": {
                "trace": str(self.trace_path),
                "replay_html": str(self.run_dir / "replay.html"),
            },
            "finished_at": datetime.now(UTC).isoformat(),
        }
        if extra:
            summary.update(extra)
        clean: dict[str, Any] = sanitize(summary)
        self.summary_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
        return clean


def generate_replay_html(trace_path: Path, output_path: Path) -> None:
    events = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>VulcanBench Replay</title>
<style>body{{font-family:ui-monospace,monospace;background:#0a0a0a;color:#ddd;margin:0;padding:20px}} .event{{border:1px solid #333;margin:4px 0;padding:8px;border-radius:4px}} .step{{color:#0f0}} .type{{color:#0af;font-weight:bold}} pre{{background:#111;padding:8px;overflow:auto;max-height:300px}} .header{{position:sticky;top:0;background:#111;padding:10px;z-index:10}}</style>
</head><body>
<div class="header"><h1>VulcanBench Replay</h1><p>Trace: {trace_path} | Events: {len(events)}</p><button onclick="location.reload()">Reload</button></div>
<div id="timeline">"""
    for ev in events:
        html += f'<div class="event"><span class="step">#{ev["step"]}</span> <span class="type">{ev["type"]}</span> <small>{ev["ts"]}</small><pre>{json.dumps(ev["data"], indent=2, ensure_ascii=False)}</pre></div>'
    html += """</div><script>console.log('self-contained replay ready');</script></body></html>"""
    output_path.write_text(html, encoding="utf-8")
