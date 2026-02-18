#!/usr/bin/env python3
import json, html, datetime, re

results = []
with open("/tmp/mlx-test-results.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

ok = [r for r in results if r["status"] == "OK"]
fail = [r for r in results if r["status"] == "FAIL"]
ok_sorted = sorted(ok, key=lambda r: r.get("tokens_per_sec", 0), reverse=True)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# Build per-model response sections
model_responses = ""
for i, r in enumerate(ok_sorted):
    model_id = r["model"].replace("/", "_").replace(".", "_")
    content = r.get("content", r.get("content_preview", ""))
    # Escape for embedding in JS string
    content_js = json.dumps(content)
    model_responses += f"""
<div class="response-section" id="resp-{i}">
  <h3 class="response-header" onclick="toggleResponse({i})">
    <span class="toggle-icon" id="icon-{i}">&#9654;</span>
    <span class="mono">{html.escape(r["model"])}</span>
    <span class="response-meta">{r["completion_tokens"]} tokens &middot; {r["tokens_per_sec"]:.1f} tok/s</span>
  </h3>
  <div class="response-body" id="body-{i}" style="display:none">
    <div class="rendered-content" id="content-{i}"></div>
  </div>
</div>
<script>responseData[{i}] = {content_js};</script>
"""

# Build failed model rows
fail_rows = ""
if fail:
    for r in fail:
        error = r.get("error", "Unknown error")
        error_clean = error.replace("\\n", " ").replace('\\"', '"').strip()
        if "loadFailed" in error_clean:
            m = re.search(r'loadFailed\("([^"]+)"\)', error_clean)
            if m:
                error_clean = m.group(1)
        elif "loading model" in error_clean:
            error_clean = "Model loading stalled (timeout)"
        error_clean = error_clean[:200]
        fail_rows += f"""<tr>
  <td class="mono">{html.escape(r["model"].strip())}</td>
  <td class="error-text" title="{html.escape(error_clean)}">{html.escape(error_clean)}</td>
  <td>{r["load_time_s"]}</td>
</tr>
"""

# Build performance table rows
perf_rows = ""
max_tps = ok_sorted[0]["tokens_per_sec"] if ok_sorted else 1
for i, r in enumerate(ok_sorted):
    tps = r["tokens_per_sec"]
    pct = (tps / max_tps) * 100
    if pct > 60:
        color = "#3fb950"
    elif pct > 30:
        color = "#d29922"
    else:
        color = "#da6d28"
    preview = html.escape(r.get("content_preview", "")[:200])
    perf_rows += f"""<tr onclick="scrollToResponse({i})" style="cursor:pointer" title="Click to view full response">
  <td class="rank">{i+1}</td>
  <td class="mono">{html.escape(r["model"])}</td>
  <td class="status-ok">OK</td>
  <td>{r["load_time_s"]}</td>
  <td>{r["completion_tokens"]}</td>
  <td>{r["gen_time_s"]}</td>
  <td><div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div><div class="bar-label">{tps:.1f}</div></div></td>
  <td class="preview" title="{preview}">{preview[:120]}</td>
</tr>
"""

best_tps = f"{ok_sorted[0]['tokens_per_sec']:.1f}" if ok_sorted else "N/A"
best_model = html.escape(ok_sorted[0]["model"]) if ok_sorted else "N/A"

report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLX Model Test Report — {now}</title>
<!-- MathJax for LaTeX rendering -->
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{ fontCache: 'global' }},
  startup: {{ typeset: false }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
<!-- marked.js for Markdown rendering -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
  h1 {{ color: #58a6ff; margin-bottom: 0.5rem; }}
  h2 {{ color: #58a6ff; margin: 2rem 0 1rem; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }}
  .summary {{ display: flex; gap: 1.5rem; margin: 1.5rem 0; flex-wrap: wrap; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem 1.5rem; min-width: 160px; }}
  .card .label {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 0.3rem; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .card .value.green {{ color: #3fb950; }}
  .card .value.red {{ color: #f85149; }}
  .card .value.blue {{ color: #58a6ff; }}
  .card .value.yellow {{ color: #d29922; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th {{ background: #161b22; color: #58a6ff; text-align: left; padding: 0.7rem 1rem; font-weight: 600; border-bottom: 2px solid #30363d; position: sticky; top: 0; z-index: 10; }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #161b22; }}
  .status-ok {{ color: #3fb950; font-weight: 600; }}
  .status-fail {{ color: #f85149; font-weight: 600; }}
  .bar {{ background: #21262d; border-radius: 4px; height: 20px; position: relative; overflow: hidden; min-width: 60px; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .bar-label {{ position: absolute; right: 6px; top: 1px; font-size: 0.75rem; color: #c9d1d9; font-weight: 600; }}
  .mono {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.85rem; }}
  .preview {{ max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.8rem; color: #8b949e; }}
  .error-text {{ color: #f85149; font-size: 0.85rem; max-width: 500px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .meta {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .rank {{ color: #8b949e; font-size: 0.85rem; width: 30px; text-align: center; }}

  /* Response sections */
  .response-section {{ margin: 0.5rem 0; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }}
  .response-header {{ display: flex; align-items: center; gap: 0.8rem; padding: 0.8rem 1.2rem; background: #161b22; cursor: pointer; user-select: none; color: #c9d1d9; font-size: 0.95rem; font-weight: 500; }}
  .response-header:hover {{ background: #1c2129; }}
  .toggle-icon {{ font-size: 0.7rem; color: #8b949e; transition: transform 0.2s; display: inline-block; width: 1rem; }}
  .toggle-icon.open {{ transform: rotate(90deg); }}
  .response-meta {{ margin-left: auto; color: #8b949e; font-size: 0.8rem; font-family: -apple-system, sans-serif; }}
  .response-body {{ padding: 1.5rem; background: #0d1117; border-top: 1px solid #21262d; }}

  /* Rendered markdown content */
  .rendered-content {{ line-height: 1.7; font-size: 0.95rem; }}
  .rendered-content h1, .rendered-content h2, .rendered-content h3 {{ color: #58a6ff; margin: 1.2rem 0 0.6rem; }}
  .rendered-content h1 {{ font-size: 1.4rem; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; }}
  .rendered-content h2 {{ font-size: 1.2rem; }}
  .rendered-content h3 {{ font-size: 1.05rem; }}
  .rendered-content p {{ margin: 0.6rem 0; }}
  .rendered-content ul, .rendered-content ol {{ margin: 0.6rem 0 0.6rem 1.5rem; }}
  .rendered-content li {{ margin: 0.3rem 0; }}
  .rendered-content code {{ background: #161b22; padding: 0.15rem 0.4rem; border-radius: 4px; font-family: 'SF Mono', Menlo, monospace; font-size: 0.85em; color: #e6edf3; }}
  .rendered-content pre {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; overflow-x: auto; margin: 0.8rem 0; }}
  .rendered-content pre code {{ background: none; padding: 0; }}
  .rendered-content blockquote {{ border-left: 3px solid #30363d; padding-left: 1rem; color: #8b949e; margin: 0.6rem 0; }}
  .rendered-content strong {{ color: #e6edf3; }}
  .rendered-content hr {{ border: none; border-top: 1px solid #21262d; margin: 1rem 0; }}
  .rendered-content table {{ border: 1px solid #30363d; }}
  .rendered-content th, .rendered-content td {{ border: 1px solid #30363d; padding: 0.4rem 0.8rem; }}

  /* MathJax overrides for dark theme */
  mjx-container {{ color: #c9d1d9 !important; }}
  mjx-container svg {{ fill: #c9d1d9; }}

  .expand-all {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; margin-bottom: 1rem; }}
  .expand-all:hover {{ background: #30363d; }}
</style>
</head>
<body>
<h1>MLX Model Test Report</h1>
<p class="meta">Generated {now} &middot; AFM MLX Backend &middot; Apple M4 Ultra (512GB) &middot; 3000 max tokens &middot; temp 0.7</p>

<div class="summary">
  <div class="card"><div class="label">Models Tested</div><div class="value blue">{len(results)}</div></div>
  <div class="card"><div class="label">Passed</div><div class="value green">{len(ok)}</div></div>
  <div class="card"><div class="label">Failed</div><div class="value red">{len(fail)}</div></div>
  <div class="card"><div class="label">Best tok/s</div><div class="value yellow">{best_tps}</div></div>
  <div class="card"><div class="label">Fastest Model</div><div class="value" style="font-size:1rem;color:#d29922">{best_model}</div></div>
</div>

<h2>Performance Ranking (by tokens/sec)</h2>
<p style="color:#8b949e;font-size:0.85rem;margin-bottom:0.5rem">Click a row to jump to its full response below.</p>
<table>
<tr>
  <th>#</th>
  <th>Model</th>
  <th>Status</th>
  <th>Load (s)</th>
  <th>Tokens</th>
  <th>Gen Time (s)</th>
  <th style="min-width:200px">Tokens/sec</th>
  <th>Response Preview</th>
</tr>
{perf_rows}
</table>

{"<h2>Failed Models</h2>" + chr(10) + "<table>" + chr(10) + "<tr><th>Model</th><th>Error</th><th>Load Time (s)</th></tr>" + chr(10) + fail_rows + "</table>" if fail else ""}

<h2>Full Model Responses</h2>
<button class="expand-all" onclick="toggleAll()">Expand / Collapse All</button>

<script>var responseData = {{}};</script>
{model_responses}

<script>
// Toggle individual response
function toggleResponse(idx) {{
  var body = document.getElementById('body-' + idx);
  var icon = document.getElementById('icon-' + idx);
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    icon.classList.add('open');
    renderContent(idx);
  }} else {{
    body.style.display = 'none';
    icon.classList.remove('open');
  }}
}}

// Scroll from perf table to response
function scrollToResponse(idx) {{
  var el = document.getElementById('resp-' + idx);
  var body = document.getElementById('body-' + idx);
  var icon = document.getElementById('icon-' + idx);
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    icon.classList.add('open');
    renderContent(idx);
  }}
  el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
}}

// Expand/collapse all
var allExpanded = false;
function toggleAll() {{
  allExpanded = !allExpanded;
  for (var key in responseData) {{
    var body = document.getElementById('body-' + key);
    var icon = document.getElementById('icon-' + key);
    if (allExpanded) {{
      body.style.display = 'block';
      icon.classList.add('open');
      renderContent(parseInt(key));
    }} else {{
      body.style.display = 'none';
      icon.classList.remove('open');
    }}
  }}
}}

// Render markdown + LaTeX for a response
var rendered = {{}};
function renderContent(idx) {{
  if (rendered[idx]) return;
  rendered[idx] = true;
  var raw = responseData[idx] || '';

  // Strip <think>...</think> blocks (some models emit reasoning)
  raw = raw.replace(/<think>[\\s\\S]*?<\\/think>/g, '');
  // Strip channel markers like <|channel|>analysis<|message|>
  raw = raw.replace(/<\\|channel\\|>[^<]*<\\|message\\|>/g, '');

  var el = document.getElementById('content-' + idx);

  // Use marked to parse markdown
  if (typeof marked !== 'undefined') {{
    el.innerHTML = marked.parse(raw);
  }} else {{
    el.innerHTML = '<pre>' + raw.replace(/</g, '&lt;') + '</pre>';
  }}

  // Typeset MathJax
  if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {{
    MathJax.typesetPromise([el]).catch(function(err) {{ console.warn('MathJax error:', err); }});
  }}
}}
</script>

</body>
</html>
"""

with open("/tmp/mlx-model-report.html", "w") as f:
    f.write(report)

print(f"Report written to /tmp/mlx-model-report.html")
print(f"  {len(ok)} passed, {len(fail)} failed out of {len(results)} models")
