#!/usr/bin/env python3
"""
Render the per-request capture (requests.jsonl) into a self-contained
HTML report for review. Dark theme matching the video, filterable /
searchable / sortable table with expandable rows showing full prompt,
reasoning_content, and content.

Input : Scripts/demo/out/requests.jsonl  (one JSON object per request,
                                          written by concurrent_load_driver.py)
        Scripts/demo/out/trace.summary.json  (optional, for header metadata)
Output: Scripts/demo/out/requests.html

The HTML is fully self-contained (no network assets) so it can be zipped
and shared, opened offline, or attached to an issue.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Load + normalize input
# ---------------------------------------------------------------------------

def load_requests(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

CSS = r"""
:root {
  --bg:        #0B0F17;
  --panel:     #131926;
  --panel-alt: #0F1420;
  --border:    #1F2937;
  --text:      #F9FAFB;
  --text-dim:  #9CA3AF;
  --text-faint:#6B7280;
  --warm:      #F59E0B;
  --cool:      #22D3EE;
  --ok:        #34D399;
  --err:       #F87171;
  --accent:    #818CF8;
  --mono: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue",
               Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}
h1 {
  font-size: 26px;
  font-weight: 700;
  margin: 0 0 6px 0;
  letter-spacing: -0.01em;
}
.subtitle {
  color: var(--text-dim);
  font-size: 14px;
  margin-bottom: 28px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-bottom: 28px;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
}
.card .label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.card .value {
  font-size: 24px;
  font-weight: 700;
  color: var(--text);
}
.card .unit {
  font-size: 12px;
  color: var(--text-dim);
  font-weight: 400;
  margin-left: 4px;
}
.card.warm .value { color: var(--warm); }
.card.cool .value { color: var(--cool); }
.card.ok   .value { color: var(--ok); }
.card.err  .value { color: var(--err); }

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 16px;
  padding: 14px 16px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.controls input[type=search], .controls select {
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px 10px;
  font-size: 13px;
  font-family: inherit;
}
.controls input[type=search] {
  flex: 1;
  min-width: 200px;
}
.controls label {
  font-size: 12px;
  color: var(--text-dim);
  margin-right: 4px;
}
.controls .info {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-dim);
}

table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
}
thead {
  background: var(--panel-alt);
}
th {
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  font-weight: 600;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
th:hover { color: var(--text); }
th.sort-asc::after  { content: " ↑"; color: var(--accent); }
th.sort-desc::after { content: " ↓"; color: var(--accent); }
td {
  padding: 12px 14px;
  font-size: 13px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
tr.row-main { cursor: pointer; transition: background 0.1s; }
tr.row-main:hover { background: rgba(255,255,255,0.02); }
tr.row-main.err td { color: var(--err); }
tr.row-detail { display: none; background: var(--panel-alt); }
tr.row-detail.open { display: table-row; }
tr.row-detail td {
  border-bottom: 2px solid var(--border);
  padding: 16px 18px 20px;
}

.mono { font-family: var(--mono); font-size: 12px; }
.dim  { color: var(--text-dim); }
.pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  background: var(--border);
  color: var(--text-dim);
}
.pill.ok  { background: rgba(52, 211, 153, 0.14); color: var(--ok); }
.pill.err { background: rgba(248, 113, 113, 0.18); color: var(--err); }
.pill.len { background: rgba(245, 158, 11, 0.14); color: var(--warm); }
.pill.stop{ background: rgba(34, 211, 238, 0.14); color: var(--cool); }

.detail-grid {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 6px 18px;
  margin-bottom: 18px;
}
.detail-grid .k { color: var(--text-dim); font-size: 12px; padding-top: 2px; }
.detail-grid .v { color: var(--text); font-size: 12px; font-family: var(--mono); word-break: break-all; }

.section-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-dim);
  margin: 14px 0 6px 0;
  font-weight: 600;
}
.pre {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  font-family: var(--mono);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  max-height: 420px;
  overflow-y: auto;
}
.pre.reasoning {
  border-left: 3px solid var(--accent);
  color: var(--text-dim);
}
.footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--text-faint);
  font-size: 12px;
  display: flex;
  justify-content: space-between;
}
"""

JS = r"""
(function() {
  const rows = Array.from(document.querySelectorAll('tr.row-main'));
  const searchInput = document.getElementById('search');
  const filterStatus = document.getElementById('filter-status');
  const filterUser = document.getElementById('filter-user');
  const filterReasoning = document.getElementById('filter-reasoning');
  const info = document.getElementById('info');

  function matches(row) {
    const q = (searchInput.value || '').toLowerCase();
    if (q) {
      const hay = row.dataset.search || '';
      if (!hay.includes(q)) return false;
    }
    const st = filterStatus.value;
    if (st && row.dataset.status !== st) return false;
    const u = filterUser.value;
    if (u && row.dataset.user !== u) return false;
    const r = filterReasoning.value;
    if (r === 'yes' && row.dataset.reasoning !== '1') return false;
    if (r === 'no'  && row.dataset.reasoning === '1') return false;
    return true;
  }

  function applyFilter() {
    let n = 0;
    rows.forEach(r => {
      const ok = matches(r);
      r.style.display = ok ? '' : 'none';
      // Also hide the corresponding detail row
      const id = r.dataset.id;
      const d = document.getElementById('detail-' + id);
      if (d && !ok) d.classList.remove('open');
      if (d) d.style.display = ok ? '' : 'none';
      if (ok) n++;
    });
    info.textContent = n + ' of ' + rows.length + ' requests';
  }

  [searchInput, filterStatus, filterUser, filterReasoning].forEach(el => {
    el.addEventListener('input', applyFilter);
    el.addEventListener('change', applyFilter);
  });

  rows.forEach(row => {
    row.addEventListener('click', () => {
      const id = row.dataset.id;
      const d = document.getElementById('detail-' + id);
      if (d) d.classList.toggle('open');
    });
  });

  // Column sorting
  const headers = document.querySelectorAll('thead th[data-sort]');
  let sortState = { col: null, dir: 1 };
  headers.forEach(h => {
    h.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const col = h.dataset.sort;
      if (sortState.col === col) sortState.dir = -sortState.dir;
      else { sortState.col = col; sortState.dir = 1; }
      headers.forEach(x => x.classList.remove('sort-asc','sort-desc'));
      h.classList.add(sortState.dir > 0 ? 'sort-asc' : 'sort-desc');
      sortRows(col, sortState.dir);
    });
  });

  function sortRows(col, dir) {
    const tbody = document.querySelector('tbody');
    const pairs = rows.map(r => {
      const d = document.getElementById('detail-' + r.dataset.id);
      return { main: r, detail: d, key: r.dataset['sort' + col] || '' };
    });
    pairs.sort((a, b) => {
      const av = isNaN(+a.key) ? a.key : +a.key;
      const bv = isNaN(+b.key) ? b.key : +b.key;
      if (av < bv) return -dir;
      if (av > bv) return  dir;
      return 0;
    });
    pairs.forEach(p => {
      tbody.appendChild(p.main);
      if (p.detail) tbody.appendChild(p.detail);
    });
  }

  applyFilter();
})();
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preview(text: str, n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) > n:
        return text[: n - 1] + "…"
    return text


def classify(rec: dict[str, Any]) -> str:
    if rec.get("error"):
        return "error"
    fr = rec.get("finish_reason") or ""
    if fr == "stop":
        return "ok"
    if fr == "length":
        return "length"
    return "other"


def status_pill(rec: dict[str, Any]) -> str:
    cls = classify(rec)
    if cls == "error":
        return '<span class="pill err">error</span>'
    if cls == "length":
        return '<span class="pill len">length</span>'
    if cls == "ok":
        return '<span class="pill ok">stop</span>'
    fr = rec.get("finish_reason") or "?"
    return f'<span class="pill stop">{html.escape(fr)}</span>'


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(records: list[dict[str, Any]], summary: dict[str, Any], out_path: Path) -> None:
    cfg = summary.get("cfg", {}) if isinstance(summary, dict) else {}
    model = cfg.get("model", "(unknown model)")
    display_model = model.split("/", 1)[-1] if "/" in model else model
    target_users = cfg.get("target_users", "?")
    mode = cfg.get("mode", "?")
    ramp_s = cfg.get("ramp_s", "?")
    hold_s = cfg.get("hold_s", "?")

    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    lengths = [classify(r) for r in records]
    finished_stop = lengths.count("ok")
    finished_length = lengths.count("length")

    pt_vals = [r["usage"]["prompt_tokens"] for r in records if isinstance(r.get("usage"), dict) and "prompt_tokens" in r["usage"]]
    ct_vals = [r["usage"]["completion_tokens"] for r in records if isinstance(r.get("usage"), dict) and "completion_tokens" in r["usage"]]
    wall_vals = [r["wall_seconds"] for r in records if isinstance(r.get("wall_seconds"), (int, float))]

    def _tg_per_sec_wall(r: dict[str, Any]) -> float | None:
        """User-observed decode throughput: completion_tokens / wall_seconds.
        Lower bound on the model's actual decode rate because wall includes
        queue wait + prefill + decode."""
        cached = r.get("tg_per_sec_wall")
        if isinstance(cached, (int, float)):
            return float(cached)
        u = r.get("usage") or {}
        ct = u.get("completion_tokens")
        wall = r.get("wall_seconds")
        if isinstance(ct, int) and ct > 0 and isinstance(wall, (int, float)) and wall > 0:
            return ct / wall
        return None

    def _tg_per_sec_pure(r: dict[str, Any]) -> float | None:
        """Server-reported pure decode throughput — what the GPU actually
        produces during this request's decode phase, excluding queue and
        prefill. Read from usage.completion_tokens_per_second, falling
        back to completion_tokens / completion_time."""
        cached = r.get("tg_per_sec_pure")
        if isinstance(cached, (int, float)):
            return float(cached)
        u = r.get("usage") or {}
        pure = u.get("completion_tokens_per_second")
        if isinstance(pure, (int, float)) and pure > 0:
            return float(pure)
        ct = u.get("completion_tokens")
        comp_time = u.get("completion_time")
        if isinstance(ct, int) and ct > 0 and isinstance(comp_time, (int, float)) and comp_time > 0:
            return ct / float(comp_time)
        return None

    tgs_vals = [v for v in (_tg_per_sec_wall(r) for r in records) if v is not None]
    tgs_pure_vals = [v for v in (_tg_per_sec_pure(r) for r in records) if v is not None]

    def _prefill_tps(r: dict[str, Any]) -> float | None:
        """Per-request prefill rate — how fast the server tokenized +
        prefilled this request's prompt, from the usage field."""
        u = r.get("usage") or {}
        rate = u.get("prompt_tokens_per_second")
        if isinstance(rate, (int, float)) and rate > 0:
            return float(rate)
        pt = u.get("prompt_tokens")
        pt_time = u.get("prompt_time")
        if isinstance(pt, int) and pt > 0 and isinstance(pt_time, (int, float)) and pt_time > 0:
            return pt / float(pt_time)
        return None

    def _prefill_time(r: dict[str, Any]) -> float | None:
        u = r.get("usage") or {}
        t = u.get("prompt_time")
        if isinstance(t, (int, float)):
            return float(t)
        return None

    def _cached_tokens(r: dict[str, Any]) -> int:
        u = r.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        n = details.get("cached_tokens", 0)
        return int(n) if isinstance(n, (int, float)) else 0

    pp_tps_vals = [v for v in (_prefill_tps(r) for r in records) if v is not None]
    pp_time_vals = [v for v in (_prefill_time(r) for r in records) if v is not None]
    cached_sum = sum(_cached_tokens(r) for r in records)
    pp_sum = sum(pt_vals) if pt_vals else 0
    cache_hit_ratio = (cached_sum / (pp_sum + cached_sum)) if (pp_sum + cached_sum) > 0 else 0.0

    def fmt_avg(v: list[float]) -> str:
        if not v:
            return "—"
        return f"{statistics.mean(v):.0f}"

    def fmt_median(v: list[float]) -> str:
        if not v:
            return "—"
        return f"{statistics.median(v):.1f}"

    # Build table rows
    row_html: list[str] = []
    for idx, r in enumerate(records):
        cls = classify(r)
        corr = r.get("corr_id") or ""
        user = r.get("user_id", "?")
        seq = r.get("req_seq", "?")
        chatcmpl = r.get("chatcmpl_id") or ""
        prompt = r.get("prompt") or ""
        content = r.get("content") or ""
        reasoning = r.get("reasoning_content") or ""
        usage = r.get("usage") or {}
        pt = usage.get("prompt_tokens", "?")
        ct = usage.get("completion_tokens", "?")
        wall = r.get("wall_seconds", "?")
        finish = r.get("finish_reason") or ""
        model_id = r.get("model", "")
        error_msg = r.get("error") or ""
        has_reasoning = "1" if reasoning else "0"
        t_start = r.get("t_start_rel", "?")
        t_end = r.get("t_end_rel", "?")
        max_tokens_req = r.get("max_tokens_requested", "?")
        system_prompt = r.get("system") or ""
        mt_req_display = f"{max_tokens_req}" if max_tokens_req != "?" else "?"
        tgs = _tg_per_sec_wall(r)
        tgs_display = f"{tgs:.1f}" if tgs is not None else "—"
        tgs_sort = f"{tgs:.3f}" if tgs is not None else "0"
        tgs_pure = _tg_per_sec_pure(r)
        tgs_pure_display = f"{tgs_pure:.1f}" if tgs_pure is not None else "—"
        tgs_pure_sort = f"{tgs_pure:.3f}" if tgs_pure is not None else "0"
        pp_tps = _prefill_tps(r)
        pp_tps_display = f"{pp_tps:.0f}" if pp_tps is not None else "—"
        pp_tps_sort = f"{pp_tps:.3f}" if pp_tps is not None else "0"
        pp_time = _prefill_time(r)
        pp_time_display = f"{pp_time:.2f}s" if pp_time is not None else "—"
        pp_time_sort = f"{pp_time:.3f}" if pp_time is not None else "0"
        cached = _cached_tokens(r)
        cached_display = str(cached) if cached > 0 else "—"

        search_blob = f"{corr} {chatcmpl} {prompt} {content} {reasoning} {error_msg}".lower()
        row_main_cls = "row-main"
        if cls == "error":
            row_main_cls += " err"
        row_html.append(f"""
<tr class="{row_main_cls}"
    data-id="{idx}"
    data-search="{html.escape(search_blob, quote=True)}"
    data-status="{cls}"
    data-user="{user}"
    data-reasoning="{has_reasoning}"
    data-sortcorr="{html.escape(corr)}"
    data-sortuser="{user}"
    data-sortseq="{seq}"
    data-sortpt="{pt if isinstance(pt, int) else 0}"
    data-sortct="{ct if isinstance(ct, int) else 0}"
    data-sortwall="{wall if isinstance(wall, (int, float)) else 0}"
    data-sorttgs="{tgs_sort}"
    data-sorttgspure="{tgs_pure_sort}"
    data-sortpptps="{pp_tps_sort}"
    data-sortpptime="{pp_time_sort}"
    data-sortcached="{cached}"
    data-sortstart="{t_start if isinstance(t_start, (int, float)) else 0}">
  <td class="mono dim">{html.escape(corr)}</td>
  <td class="mono">u{user:03d}</td>
  <td class="mono dim">#{seq}</td>
  <td class="mono dim">{html.escape(chatcmpl)}</td>
  <td>{pt}</td>
  <td class="dim">{cached_display}</td>
  <td>{pp_time_display}</td>
  <td>{pp_tps_display}</td>
  <td>{ct}</td>
  <td>{wall if wall == "?" else f"{wall:.2f}"}s</td>
  <td>{tgs_display}</td>
  <td>{tgs_pure_display}</td>
  <td>{status_pill(r)}</td>
  <td>{"✦ " if reasoning else ""}{html.escape(preview(content) or preview(reasoning) or "—")}</td>
</tr>
<tr class="row-detail" id="detail-{idx}">
  <td colspan="14">
    <div class="detail-grid">
      <div class="k">corr_id</div>        <div class="v">{html.escape(corr)}</div>
      <div class="k">chatcmpl_id</div>    <div class="v">{html.escape(chatcmpl)}</div>
      <div class="k">model</div>          <div class="v">{html.escape(model_id)}</div>
      <div class="k">user / seq</div>     <div class="v">user {user} · req #{seq}</div>
      <div class="k">timing</div>         <div class="v">t=[{t_start}..{t_end}] ({wall if wall == "?" else f"{wall:.3f}"}s wall)</div>
      <div class="k">tokens</div>         <div class="v">prompt={pt}, completion={ct} (max_tokens requested={mt_req_display})</div>
      <div class="k">prefill time</div>   <div class="v">{pp_time_display} · {pp_tps_display} tok/s · {cached_display} cached from radix</div>
      <div class="k">tg/s wall</div>      <div class="v">{tgs_display} tok/s (user-observed: completion / wall, includes queue+prefill+decode)</div>
      <div class="k">tg/s pure</div>      <div class="v">{tgs_pure_display} tok/s (server-reported pure decode rate)</div>
      <div class="k">finish_reason</div>  <div class="v">{html.escape(finish) or "—"}</div>
      {f'<div class="k">error</div><div class="v" style="color:var(--err)">{html.escape(error_msg)}</div>' if error_msg else ""}
    </div>
    {f'<div class="section-label">system prompt</div><div class="pre">{html.escape(system_prompt)}</div>' if system_prompt else ""}
    <div class="section-label">user prompt</div>
    <div class="pre">{html.escape(prompt)}</div>
    {f'<div class="section-label">reasoning_content (from &lt;think&gt;...)</div><div class="pre reasoning">{html.escape(reasoning)}</div>' if reasoning else ""}
    <div class="section-label">assistant content</div>
    <div class="pre">{html.escape(content) or '<span class="dim">(empty)</span>'}</div>
  </td>
</tr>
""")

    # User filter options
    user_ids = sorted({r.get("user_id", 0) for r in records})
    user_options = "".join(
        f'<option value="{u}">u{u:03d}</option>' for u in user_ids[:500]
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFM demo · request review</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
  <h1>AFM demo · request review</h1>
  <div class="subtitle">
    {html.escape(display_model)}  ·  {target_users} concurrent users  ·  ramp {ramp_s}s + hold {hold_s}s  ·  mode: {html.escape(str(mode))}
  </div>

  <div class="cards">
    <div class="card"><div class="label">Requests</div><div class="value">{total}</div></div>
    <div class="card ok"><div class="label">Finished (stop)</div><div class="value">{finished_stop}</div></div>
    <div class="card warm"><div class="label">Hit max_tokens</div><div class="value">{finished_length}</div></div>
    <div class="card err"><div class="label">Errors</div><div class="value">{errors}</div></div>
    <div class="card warm"><div class="label">Avg prompt tok</div><div class="value">{fmt_avg(pt_vals)}</div></div>
    <div class="card cool"><div class="label">Avg completion tok</div><div class="value">{fmt_avg(ct_vals)}</div></div>
    <div class="card"><div class="label">Median wall</div><div class="value">{fmt_median(wall_vals)}<span class="unit">s</span></div></div>
    <div class="card cool"><div class="label">Median tg/s — wall</div><div class="value">{fmt_median(tgs_vals)}<span class="unit">tok/s</span></div></div>
    <div class="card cool"><div class="label">Median tg/s — pure</div><div class="value">{fmt_median(tgs_pure_vals)}<span class="unit">tok/s</span></div></div>
    <div class="card warm"><div class="label">Median prefill tok/s</div><div class="value">{fmt_median(pp_tps_vals)}<span class="unit">tok/s</span></div></div>
    <div class="card warm"><div class="label">Median prefill time</div><div class="value">{fmt_median(pp_time_vals)}<span class="unit">s</span></div></div>
    <div class="card"><div class="label">Total prompt tok</div><div class="value">{pp_sum}</div></div>
    <div class="card ok"><div class="label">Total cached tok</div><div class="value">{cached_sum}</div></div>
    <div class="card ok"><div class="label">Cache hit ratio</div><div class="value">{cache_hit_ratio*100:.1f}<span class="unit">%</span></div></div>
  </div>

  <div class="controls">
    <input type="search" id="search" placeholder="Search prompt, content, corr_id, chatcmpl…">
    <label>status</label>
    <select id="filter-status">
      <option value="">all</option>
      <option value="ok">stop</option>
      <option value="length">hit max_tokens</option>
      <option value="error">errors</option>
    </select>
    <label>user</label>
    <select id="filter-user">
      <option value="">all</option>
      {user_options}
    </select>
    <label>reasoning</label>
    <select id="filter-reasoning">
      <option value="">all</option>
      <option value="yes">with reasoning</option>
      <option value="no">without reasoning</option>
    </select>
    <span class="info" id="info">{total} of {total} requests</span>
  </div>

  <table>
    <thead>
      <tr>
        <th data-sort="corr">corr_id</th>
        <th data-sort="user">user</th>
        <th data-sort="seq">#</th>
        <th data-sort="corr">chatcmpl</th>
        <th data-sort="pt">p tok</th>
        <th data-sort="cached">cached</th>
        <th data-sort="pptime">p time</th>
        <th data-sort="pptps">p tok/s</th>
        <th data-sort="ct">c tok</th>
        <th data-sort="wall">wall</th>
        <th data-sort="tgs">tg/s<br><span class="dim" style="font-weight:400;text-transform:none;letter-spacing:0">wall</span></th>
        <th data-sort="tgspure">tg/s<br><span class="dim" style="font-weight:400;text-transform:none;letter-spacing:0">pure</span></th>
        <th>status</th>
        <th>preview</th>
      </tr>
    </thead>
    <tbody>
      {"".join(row_html)}
    </tbody>
  </table>

  <div class="footer">
    <div>✦ = has reasoning_content · click any row to expand</div>
    <div>afm demo · Swift + MLX on Apple Silicon</div>
  </div>
</div>
<script>{JS}</script>
</body>
</html>
"""
    out_path.write_text(html_doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="Scripts/demo/out/requests.jsonl")
    ap.add_argument("--summary", default="Scripts/demo/out/trace.summary.json")
    ap.add_argument("--output", default="Scripts/demo/out/requests.html")
    args = ap.parse_args()

    req_path = Path(args.requests)
    if not req_path.exists():
        print(f"ERROR: requests file not found: {req_path}", file=sys.stderr)
        sys.exit(1)

    records = load_requests(req_path)
    summary = load_summary(Path(args.summary))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(records, summary, out_path)
    print(f"[html] wrote {out_path} ({len(records)} requests)")


if __name__ == "__main__":
    main()
