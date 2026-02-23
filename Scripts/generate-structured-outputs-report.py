#!/usr/bin/env python3
"""Generate HTML report from structured-outputs JSONL results.

Usage:
    # From a specific JSONL file
    python3 Scripts/generate-structured-outputs-report.py test-reports/structured-outputs-20260223_131353.jsonl

    # Called automatically by test-structured-outputs.sh
"""
import json, html, datetime, sys, os

if len(sys.argv) < 2:
    print("Usage: generate-structured-outputs-report.py <results.jsonl>")
    sys.exit(1)

jsonl_path = sys.argv[1]
if not os.path.exists(jsonl_path):
    print(f"ERROR: File not found: {jsonl_path}")
    sys.exit(1)

results = []
with open(jsonl_path) as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

if not results:
    print("ERROR: No results found in file")
    sys.exit(1)

passed = [r for r in results if r["status"] == "PASS"]
failed = [r for r in results if r["status"] == "FAIL"]
skipped = [r for r in results if r["status"] == "SKIP"]

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
total_elapsed = sum(r.get("elapsed_s", 0) for r in results)

# Group by section
sections = {}
for r in results:
    sec = r["section"]
    if sec not in sections:
        sections[sec] = []
    sections[sec].append(r)

# Build section HTML
section_html = ""
for sec_name, tests in sections.items():
    sec_pass = sum(1 for t in tests if t["status"] == "PASS")
    sec_fail = sum(1 for t in tests if t["status"] == "FAIL")
    sec_skip = sum(1 for t in tests if t["status"] == "SKIP")
    sec_total = len(tests)
    sec_elapsed = sum(t.get("elapsed_s", 0) for t in tests)

    if sec_fail > 0:
        sec_badge = f'<span class="badge badge-fail">{sec_fail} FAIL</span>'
    elif sec_skip > 0 and sec_pass == 0:
        sec_badge = '<span class="badge badge-skip">SKIPPED</span>'
    else:
        sec_badge = '<span class="badge badge-pass">ALL PASS</span>'

    rows = ""
    for t in tests:
        status = t["status"]
        if status == "PASS":
            status_class = "status-ok"
            icon = "&#10003;"
        elif status == "FAIL":
            status_class = "status-fail"
            icon = "&#10007;"
        else:
            status_class = "status-skip"
            icon = "&#8674;"

        detail = html.escape(t.get("detail", "") or "")
        elapsed = t.get("elapsed_s", 0)
        prompt = html.escape(t.get("prompt", "") or "")
        request = t.get("request")

        # For PASS results with JSON detail, try to pretty-format it
        detail_display = detail
        if status == "PASS" and detail.startswith("{"):
            try:
                parsed = json.loads(t.get("detail", ""))
                detail_display = '<code class="json-detail">' + html.escape(json.dumps(parsed, indent=2)) + '</code>'
            except:
                detail_display = html.escape(detail[:200])
        elif len(detail) > 200:
            detail_display = html.escape(detail[:200]) + "..."

        # Build request display: show prompt + full request payload
        request_html = ""
        if request:
            try:
                request_pretty = json.dumps(request, indent=2)
                row_id = f"req-{sec_name.replace(':', '-').replace(' ', '_')}-{len(rows)}"
                request_html = f'''<div class="request-toggle" onclick="toggleRequest(this)">&#9654; Show request payload</div>
<code class="request-detail" style="display:none">{html.escape(request_pretty)}</code>'''
            except:
                pass
        elif prompt:
            request_html = f'<div class="prompt-text">{prompt}</div>'

        rows += f"""<tr>
  <td class="{status_class}">{icon} {status}</td>
  <td>{html.escape(t["name"])}{request_html}</td>
  <td class="mono elapsed">{elapsed}s</td>
  <td class="detail">{detail_display}</td>
</tr>
"""

    rows += f"""<tr class="section-summary-row">
  <td colspan="2" style="text-align:right; color:#8b949e; font-size:0.8rem;">Section total</td>
  <td class="mono elapsed" style="font-weight:600;">{sec_elapsed:.1f}s</td>
  <td></td>
</tr>
"""

    section_html += f"""
<div class="section">
  <h3 class="section-header" onclick="toggleSection(this)">
    <span class="toggle-icon">&#9654;</span>
    {html.escape(sec_name)}
    <span class="section-stats">{sec_pass}/{sec_total} passed &middot; {sec_elapsed:.1f}s {sec_badge}</span>
  </h3>
  <div class="section-body">
    <table>
      <tr><th>Status</th><th>Test / Prompt</th><th>Time</th><th>Detail / Output</th></tr>
      {rows}
    </table>
  </div>
</div>
"""

# Overall status
if len(failed) == 0:
    overall_class = "green"
    overall_text = "ALL TESTS PASSED"
else:
    overall_class = "red"
    overall_text = f"{len(failed)} TEST{'S' if len(failed) != 1 else ''} FAILED"

report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Structured Outputs Test Report &mdash; {now}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
  h1 {{ color: #58a6ff; margin-bottom: 0.5rem; }}
  h2 {{ color: #58a6ff; margin: 2rem 0 1rem; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }}
  .meta {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .summary {{ display: flex; gap: 1.5rem; margin: 1.5rem 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem 1.5rem; min-width: 140px; }}
  .card .label {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 0.3rem; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .card .value.green {{ color: #3fb950; }}
  .card .value.red {{ color: #f85149; }}
  .card .value.blue {{ color: #58a6ff; }}
  .card .value.yellow {{ color: #d29922; }}
  .overall {{ text-align: center; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; font-size: 1.3rem; font-weight: 700; }}
  .overall.green {{ background: #0d2818; border: 2px solid #3fb950; color: #3fb950; }}
  .overall.red {{ background: #2d1117; border: 2px solid #f85149; color: #f85149; }}

  .section {{ margin: 0.8rem 0; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }}
  .section-header {{ display: flex; align-items: center; gap: 0.8rem; padding: 0.8rem 1.2rem; background: #161b22; cursor: pointer; user-select: none; color: #c9d1d9; font-size: 0.95rem; font-weight: 500; }}
  .section-header:hover {{ background: #1c2129; }}
  .toggle-icon {{ font-size: 0.7rem; color: #8b949e; transition: transform 0.2s; display: inline-block; width: 1rem; }}
  .toggle-icon.open {{ transform: rotate(90deg); }}
  .section-stats {{ margin-left: auto; color: #8b949e; font-size: 0.8rem; display: flex; gap: 0.5rem; align-items: center; }}
  .section-body {{ border-top: 1px solid #21262d; }}
  .section-summary-row td {{ border-top: 1px solid #21262d !important; }}

  .badge {{ padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .badge-pass {{ background: #0d2818; color: #3fb950; }}
  .badge-fail {{ background: #2d1117; color: #f85149; }}
  .badge-skip {{ background: #2d2200; color: #d29922; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #0d1117; color: #58a6ff; text-align: left; padding: 0.5rem 1rem; font-weight: 600; font-size: 0.85rem; }}
  td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #161b22; font-size: 0.9rem; vertical-align: top; }}
  tr:hover {{ background: #161b22; }}
  .status-ok {{ color: #3fb950; font-weight: 600; white-space: nowrap; }}
  .status-fail {{ color: #f85149; font-weight: 600; white-space: nowrap; }}
  .status-skip {{ color: #d29922; font-weight: 600; white-space: nowrap; }}
  .mono {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.85rem; }}
  .elapsed {{ white-space: nowrap; color: #8b949e; }}
  .detail {{ color: #8b949e; font-size: 0.8rem; max-width: 500px; word-break: break-word; }}
  .json-detail {{ display: block; white-space: pre-wrap; font-family: 'SF Mono', Menlo, monospace; font-size: 0.78rem; color: #7ee787; background: #0d1117; padding: 0.4rem 0.6rem; border-radius: 4px; margin-top: 0.2rem; max-height: 120px; overflow-y: auto; }}
  .prompt-text {{ color: #8b949e; font-size: 0.78rem; font-style: italic; margin-top: 0.3rem; }}
  .request-toggle {{ color: #58a6ff; font-size: 0.75rem; cursor: pointer; margin-top: 0.3rem; user-select: none; }}
  .request-toggle:hover {{ text-decoration: underline; }}
  .request-detail {{ display: block; white-space: pre-wrap; font-family: 'SF Mono', Menlo, monospace; font-size: 0.72rem; color: #c9d1d9; background: #0d1117; padding: 0.5rem 0.7rem; border-radius: 4px; margin-top: 0.3rem; max-height: 300px; overflow-y: auto; border: 1px solid #21262d; }}

  .expand-all {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; margin-bottom: 1rem; }}
  .expand-all:hover {{ background: #30363d; }}

  .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #21262d; color: #484f58; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>Structured Outputs Test Report</h1>
<p class="meta">Generated {now} &middot; afm v0.9.4 &middot; Apple M-series &middot; macOS 26</p>

<div class="overall {overall_class}">{overall_text}</div>

<div class="summary">
  <div class="card"><div class="label">Total Tests</div><div class="value blue">{len(results)}</div></div>
  <div class="card"><div class="label">Passed</div><div class="value green">{len(passed)}</div></div>
  <div class="card"><div class="label">Failed</div><div class="value red">{len(failed)}</div></div>
  <div class="card"><div class="label">Skipped</div><div class="value yellow">{len(skipped)}</div></div>
  <div class="card"><div class="label">Models</div><div class="value blue">{len(sections)}</div></div>
  <div class="card"><div class="label">Total Time</div><div class="value blue" style="font-size:1.4rem;">{total_elapsed:.1f}s</div></div>
</div>

<h2>Test Results by Backend / Model</h2>
<button class="expand-all" onclick="toggleAll()">Expand / Collapse All</button>

{section_html}

<div class="footer">
  Source data: <code>{html.escape(os.path.basename(jsonl_path))}</code>
</div>

<script>
function toggleSection(header) {{
  var body = header.nextElementSibling;
  var icon = header.querySelector('.toggle-icon');
  if (body.style.display === 'none' || body.style.display === '') {{
    body.style.display = 'block';
    icon.classList.add('open');
  }} else {{
    body.style.display = 'none';
    icon.classList.remove('open');
  }}
}}

var allExpanded = false;
function toggleAll() {{
  allExpanded = !allExpanded;
  document.querySelectorAll('.section-body').forEach(function(b) {{
    b.style.display = allExpanded ? 'block' : 'none';
  }});
  document.querySelectorAll('.toggle-icon').forEach(function(i) {{
    if (allExpanded) i.classList.add('open');
    else i.classList.remove('open');
  }});
}}

function toggleRequest(el) {{
  var code = el.nextElementSibling;
  if (code.style.display === 'none') {{
    code.style.display = 'block';
    el.innerHTML = '&#9660; Hide request payload';
  }} else {{
    code.style.display = 'none';
    el.innerHTML = '&#9654; Show request payload';
  }}
}}

// Auto-expand failed sections
document.querySelectorAll('.badge-fail').forEach(function(badge) {{
  var section = badge.closest('.section');
  if (section) {{
    var body = section.querySelector('.section-body');
    var icon = section.querySelector('.toggle-icon');
    if (body) body.style.display = 'block';
    if (icon) icon.classList.add('open');
  }}
}});
</script>

</body>
</html>
"""

# Output path: same directory and base name as the JSONL, with .html extension
html_path = jsonl_path.rsplit(".", 1)[0] + ".html"
with open(html_path, "w") as f:
    f.write(report)

print(f"Report: {html_path}")
print(f"  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped out of {len(results)} tests across {len(sections)} models")
