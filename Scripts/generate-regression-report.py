#!/usr/bin/env python3
"""Generate HTML regression test report from JSONL results."""
import json, html, datetime, sys, os

results = []
with open("/tmp/regression-test-results.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

passed = [r for r in results if r["status"] == "PASS"]
failed = [r for r in results if r["status"] == "FAIL"]
skipped = [r for r in results if r["status"] == "SKIP"]

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

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

    if sec_fail > 0:
        sec_badge = f'<span class="badge badge-fail">{sec_fail} FAIL</span>'
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
        rows += f"""<tr>
  <td class="{status_class}">{icon} {status}</td>
  <td>{html.escape(t["name"])}</td>
  <td class="mono">{elapsed}s</td>
  <td class="detail" title="{detail}">{detail[:120]}</td>
</tr>
"""

    section_html += f"""
<div class="section">
  <h3 class="section-header" onclick="toggleSection(this)">
    <span class="toggle-icon">&#9654;</span>
    {html.escape(sec_name)}
    <span class="section-stats">{sec_pass}/{sec_total} passed {sec_badge}</span>
  </h3>
  <div class="section-body">
    <table>
      <tr><th>Status</th><th>Test</th><th>Time</th><th>Detail</th></tr>
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
<title>AFM Regression Test Report — {now}</title>
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

  .badge {{ padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .badge-pass {{ background: #0d2818; color: #3fb950; }}
  .badge-fail {{ background: #2d1117; color: #f85149; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #0d1117; color: #58a6ff; text-align: left; padding: 0.5rem 1rem; font-weight: 600; font-size: 0.85rem; }}
  td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #161b22; font-size: 0.9rem; }}
  tr:hover {{ background: #161b22; }}
  .status-ok {{ color: #3fb950; font-weight: 600; }}
  .status-fail {{ color: #f85149; font-weight: 600; }}
  .status-skip {{ color: #d29922; font-weight: 600; }}
  .mono {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.85rem; }}
  .detail {{ color: #8b949e; font-size: 0.8rem; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  .expand-all {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; margin-bottom: 1rem; }}
  .expand-all:hover {{ background: #30363d; }}
</style>
</head>
<body>
<h1>AFM Regression Test Report</h1>
<p class="meta">Generated {now} &middot; v0.9.4 &middot; Apple M-series &middot; macOS 26</p>

<div class="overall {overall_class}">{overall_text}</div>

<div class="summary">
  <div class="card"><div class="label">Total Tests</div><div class="value blue">{len(results)}</div></div>
  <div class="card"><div class="label">Passed</div><div class="value green">{len(passed)}</div></div>
  <div class="card"><div class="label">Failed</div><div class="value red">{len(failed)}</div></div>
  <div class="card"><div class="label">Skipped</div><div class="value yellow">{len(skipped)}</div></div>
  <div class="card"><div class="label">Sections</div><div class="value blue">{len(sections)}</div></div>
</div>

<h2>Test Results by Section</h2>
<button class="expand-all" onclick="toggleAll()">Expand / Collapse All</button>

{section_html}

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

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outpath = os.path.join(script_dir, f"regression-report-{timestamp}.html")
with open(outpath, "w") as f:
    f.write(report)

print(f"Report: {outpath}")
print(f"  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped out of {len(results)} tests")
