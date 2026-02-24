#!/usr/bin/env python3
import json, html, datetime, re, os, sys, shutil, subprocess

results = []
with open("/tmp/mlx-test-results.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

# Tag each result with its original JSONL line index
for idx, r in enumerate(results):
    r["_jsonl_idx"] = idx

ok = [r for r in results if r["status"] == "OK"]
fail = [r for r in results if r["status"] == "FAIL"]
ok_sorted = sorted(ok, key=lambda r: r.get("tokens_per_sec", 0), reverse=True)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# Check for smart analysis file
smart_report_content = ""
output_base = os.environ.get("REPORT_OUTPUT_DIR", "")
if output_base:
    report_dir = os.path.join(output_base, "test-reports")
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    report_dir = os.path.join(project_dir, "test-reports")

# Look for smart analysis files — use SMART_TIMESTAMP from current run if set,
# otherwise fall back to most recent timestamp in the directory.
ai_scores = {}  # maps JSONL line index -> average score (1-5)
smart_analyses = []  # list of {"tool": name, "content": markdown, "scores": {idx: score}}

if os.path.isdir(report_dir):
    # Find smart analysis files and group by timestamp
    smart_files = sorted(
        [f for f in os.listdir(report_dir) if f.startswith("smart-analysis-") and f.endswith(".md")],
        reverse=True
    )
    if smart_files:
        # Only use smart analysis if SMART_TIMESTAMP was explicitly set by the test run.
        # Without it, we'd pick up stale analysis files from previous --smart runs.
        ts = os.environ.get("SMART_TIMESTAMP", "")
        if ts:
            # Load only files matching this exact timestamp
            for fname in smart_files:
                m_ts = re.search(r'(\d{8}_\d{6})\.md$', fname)
                if not m_ts or m_ts.group(1) != ts:
                    continue
                # Extract tool name: smart-analysis-TOOL-TS.md or smart-analysis-TS.md (legacy=claude)
                m_tool = re.match(r'smart-analysis-(?:(.+?)-)?(\d{8}_\d{6})\.md$', fname)
                tool_name = m_tool.group(1) if (m_tool and m_tool.group(1)) else "claude"
                try:
                    with open(os.path.join(report_dir, fname)) as sf:
                        content = sf.read()
                    scores = {}
                    m_scores = re.search(r'<!-- AI_SCORES (\[.*?\]) -->', content)
                    if m_scores:
                        for entry in json.loads(m_scores.group(1)):
                            scores[entry["i"]] = entry["s"]
                    smart_analyses.append({"tool": tool_name, "content": content, "scores": scores})
                except:
                    pass

    # Build per-tool score maps (no averaging — one column per tool)
    # ai_scores kept as average for backwards compat (used by fail table)
    if smart_analyses:
        all_indices = set()
        for sa in smart_analyses:
            all_indices.update(sa["scores"].keys())
        for idx in all_indices:
            tool_scores = [sa["scores"][idx] for sa in smart_analyses if idx in sa["scores"]]
            if tool_scores:
                ai_scores[idx] = round(sum(tool_scores) / len(tool_scores))

    # Legacy compat: single content string for old code paths
    smart_report_content = smart_analyses[0]["content"] if smart_analyses else ""

has_ai_scores = len(ai_scores) > 0


def config_badge(label, value, color="#8b949e"):
    """Generate an inline config badge."""
    if not value and value != 0:
        return ""
    return f'<span class="config-badge" style="border-color:{color}">{html.escape(str(label))}: <strong>{html.escape(str(value))}</strong></span>'


def config_panel(r):
    """Build a config details panel for a result."""
    badges = []
    # Temperature
    temp = r.get("temperature", "")
    if temp != "":
        badges.append(config_badge("temp", temp, "#d29922"))
    # Max tokens
    mt = r.get("max_tokens", "")
    if mt != "":
        badges.append(config_badge("max_tokens", mt, "#58a6ff"))
    # Label
    label = r.get("label", "")
    if label:
        badges.append(config_badge("variant", label, "#a371f7"))
    # Optional sampling params
    for key, label_text, color in [
        ("top_p", "top_p", "#58a6ff"),
        ("top_k", "top_k", "#58a6ff"),
        ("min_p", "min_p", "#58a6ff"),
        ("seed", "seed", "#a371f7"),
        ("presence_penalty", "pres_pen", "#f0883e"),
        ("repetition_penalty", "rep_pen", "#f0883e"),
        ("frequency_penalty", "freq_pen", "#f0883e"),
        ("logprobs", "logprobs", "#a371f7"),
        ("top_logprobs", "top_logprobs", "#a371f7"),
        ("stop", "stop", "#d29922"),
        ("response_format", "resp_fmt", "#d29922"),
    ]:
        val = r.get(key)
        if val is not None:
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            badges.append(config_badge(label_text, val, color))
    # AFM args
    afm = r.get("afm_args", "")
    if afm:
        badges.append(config_badge("afm", afm, "#f0883e"))
    # System prompt
    sp = r.get("system_prompt", "")
    if sp:
        short_sp = sp[:80] + ("..." if len(sp) > 80 else "")
        badges.append(config_badge("system", short_sp, "#3fb950"))
    # Feature verification badges
    finish = r.get("finish_reason")
    if finish:
        fc = "#3fb950" if finish == "stop" else "#d29922" if finish == "length" else "#58a6ff"
        badges.append(config_badge("finish", finish, fc))
    lpc = r.get("logprobs_count", 0)
    if lpc:
        badges.append(config_badge("logprobs", f'{lpc} tokens', "#a371f7"))
    vjson = r.get("is_valid_json")
    if vjson is True:
        badges.append(config_badge("json", "valid", "#3fb950"))
    elif vjson is False:
        badges.append(config_badge("json", "INVALID", "#f85149"))
    # Timing
    badges.append(config_badge("load", f'{r.get("load_time_s", "?")}s', "#8b949e"))
    badges.append(config_badge("gen", f'{r.get("gen_time_s", "?")}s', "#8b949e"))
    badges.append(config_badge("prompt_tok", r.get("prompt_tokens", "?"), "#8b949e"))
    badges.append(config_badge("comp_tok", r.get("completion_tokens", "?"), "#8b949e"))
    tps = r.get("tokens_per_sec", 0)
    if tps:
        badges.append(config_badge("tok/s", f'{tps:.1f}', "#d29922"))

    return " ".join(badges)


# Build per-model response sections
model_responses = ""
for i, r in enumerate(ok_sorted):
    content = r.get("content", "") or r.get("content_preview", "")
    reasoning = r.get("reasoning_content", "")
    # Show reasoning if content is empty, or prepend it if both exist
    if not content and reasoning:
        content = f"<details open><summary><strong>🧠 Reasoning</strong> <em>(model used all tokens thinking — no response emitted)</em></summary>\n\n{reasoning}\n\n</details>"
    elif content and reasoning:
        content = f"<details><summary><strong>🧠 Reasoning</strong></summary>\n\n{reasoning}\n\n</details>\n\n{content}"
    content_js = json.dumps(content)
    prompt_text = r.get("prompt", "")
    prompt_html = html.escape(prompt_text[:500]) + ("..." if len(prompt_text) > 500 else "")
    panel = config_panel(r)

    model_responses += f"""
<div class="response-section" id="resp-{i}">
  <h3 class="response-header" onclick="toggleResponse({i})">
    <span class="toggle-icon" id="icon-{i}">&#9654;</span>
    <span class="mono">{html.escape(r["model"])}</span>
    <span class="response-meta">{r["completion_tokens"]} tokens &middot; {r["tokens_per_sec"]:.1f} tok/s</span>
  </h3>
  <div class="response-body" id="body-{i}" style="display:none">
    <div class="config-panel">{panel}</div>
    <div class="prompt-box"><span class="prompt-label">PROMPT</span> {prompt_html}</div>
    <div class="rendered-content" id="content-{i}"></div>
  </div>
</div>
<script>responseData[{i}] = {content_js};</script>
"""

def ai_score_cell(score):
    """Render an AI score (1-5) as a colored cell."""
    if score is None:
        return '<td class="ai-score" style="color:#8b949e">—</td>'
    colors = {5: "#3fb950", 4: "#58a6ff", 3: "#d29922", 2: "#f0883e", 1: "#f85149"}
    labels = {5: "5", 4: "4", 3: "3", 2: "2", 1: "1"}
    c = colors.get(score, "#8b949e")
    return f'<td class="ai-score" style="color:{c};font-weight:700">{labels.get(score, "?")}</td>'

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

        label = r.get("label", "")
        label_html = f' <span class="variant-tag">{html.escape(label)}</span>' if label else ""
        afm = r.get("afm_args", "")
        afm_html = f'<span class="afm-tag">{html.escape(afm)}</span>' if afm else ""
        temp = r.get("temperature", "")
        temp_html = f'<span class="temp-tag">t={temp}</span>' if temp != "" else ""

        score_tds = ""
        if has_ai_scores:
            jsonl_idx = r.get("_jsonl_idx")
            for sa in smart_analyses:
                s = sa["scores"].get(jsonl_idx)
                score_tds += ai_score_cell(s)
        fail_rows += f"""<tr>
  <td class="mono">{html.escape(r["model"].strip())}{label_html}</td>
  <td class="error-text" title="{html.escape(error_clean)}">{html.escape(error_clean)}</td>
  {score_tds}
  <td>{temp_html} {afm_html}</td>
  <td>{r.get("load_time_s", "?")}</td>
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
    temp = r.get("temperature", "?")
    label = r.get("label", "")
    label_html = f' <span class="variant-tag">{html.escape(label)}</span>' if label else ""
    afm = r.get("afm_args", "")
    afm_html = f'<br><span class="afm-tag">{html.escape(afm)}</span>' if afm else ""
    prompt = r.get("prompt", "")
    prompt_short = html.escape(prompt[:60]) + ("..." if len(prompt) > 60 else "")
    # One score cell per AI tool
    score_tds = ""
    if has_ai_scores:
        jsonl_idx = r.get("_jsonl_idx")
        for sa in smart_analyses:
            s = sa["scores"].get(jsonl_idx)
            score_tds += ai_score_cell(s)

    perf_rows += f"""<tr onclick="scrollToResponse({i})" style="cursor:pointer" title="Click to view full response">
  <td class="rank">{i+1}</td>
  <td class="mono">{html.escape(r["model"])}{label_html}{afm_html}</td>
  <td class="status-ok">OK</td>
  {score_tds}
  <td>{temp}</td>
  <td>{r["load_time_s"]}</td>
  <td>{r["completion_tokens"]}</td>
  <td>{r["gen_time_s"]}</td>
  <td><div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div><div class="bar-label">{tps:.1f}</div></div></td>
  <td class="preview" title="{html.escape(prompt)}">{prompt_short}</td>
</tr>
"""

best_tps = f"{ok_sorted[0]['tokens_per_sec']:.1f}" if ok_sorted else "N/A"
best_model = html.escape(ok_sorted[0]["model"]) if ok_sorted else "N/A"

# AI score column headers — one per tool
if has_ai_scores:
    ai_score_header = "".join(
        f'<th title="{html.escape(sa["tool"])} score">{html.escape(sa["tool"][:6])}</th>'
        for sa in smart_analyses
    )
else:
    ai_score_header = ""

# Smart analysis sections — one per tool
smart_section = ""
if smart_analyses:
    smart_section += '<h2>AI Analysis (--smart)</h2>\n'
    for si, sa in enumerate(smart_analyses):
        tool = html.escape(sa["tool"])
        # Strip the AI_SCORES line from displayed content
        display_content = re.sub(r'<!-- AI_SCORES \[.*?\] -->\s*$', '', sa["content"]).strip()
        content_js = json.dumps(display_content)
        num_tools = len(smart_analyses)
        tool_label = f"{tool} Analysis" if num_tools > 1 else "Quality, Anomalies &amp; Recommendations"
        score_summary = ""
        if sa["scores"]:
            vals = list(sa["scores"].values())
            avg = sum(vals) / len(vals)
            score_summary = f' &middot; avg score: {avg:.1f}/5'
        smart_section += f"""
<div class="response-section">
  <h3 class="response-header" onclick="toggleSmart_{si}()">
    <span class="toggle-icon" id="smart-icon-{si}">&#9654;</span>
    <span>{tool_label}{score_summary}</span>
  </h3>
  <div class="response-body" id="smart-body-{si}" style="display:none">
    <div class="rendered-content" id="smart-content-{si}"></div>
  </div>
</div>
<script>
var smartData_{si} = {content_js};
var smartRendered_{si} = false;
function toggleSmart_{si}() {{
  var body = document.getElementById('smart-body-{si}');
  var icon = document.getElementById('smart-icon-{si}');
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    icon.classList.add('open');
    if (!smartRendered_{si}) {{
      smartRendered_{si} = true;
      var el = document.getElementById('smart-content-{si}');
      if (typeof marked !== 'undefined') {{
        el.innerHTML = marked.parse(smartData_{si});
      }} else {{
        el.innerHTML = '<pre>' + smartData_{si}.replace(/</g, '&lt;') + '</pre>';
      }}
    }}
  }} else {{
    body.style.display = 'none';
    icon.classList.remove('open');
  }}
}}
</script>
"""

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
  .ai-score {{ text-align: center; font-size: 1rem; width: 35px; }}

  /* Config badges */
  .config-panel {{ margin-bottom: 1rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }}
  .config-badge {{ display: inline-block; font-size: 0.75rem; padding: 0.2rem 0.6rem; border: 1px solid #30363d; border-radius: 12px; color: #c9d1d9; font-family: 'SF Mono', Menlo, monospace; white-space: nowrap; }}
  .config-badge strong {{ color: #e6edf3; }}

  /* Prompt box */
  .prompt-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #c9d1d9; line-height: 1.5; }}
  .prompt-label {{ display: inline-block; background: #58a6ff; color: #0d1117; font-size: 0.65rem; font-weight: 700; padding: 0.1rem 0.4rem; border-radius: 3px; margin-right: 0.5rem; vertical-align: middle; letter-spacing: 0.05em; }}

  /* Variant / AFM tags */
  .variant-tag {{ display: inline-block; font-size: 0.7rem; padding: 0.1rem 0.4rem; background: #a371f7; color: #0d1117; border-radius: 3px; margin-left: 0.4rem; font-weight: 600; font-family: -apple-system, sans-serif; }}
  .afm-tag {{ display: inline-block; font-size: 0.7rem; padding: 0.1rem 0.4rem; background: #21262d; border: 1px solid #f0883e; color: #f0883e; border-radius: 3px; font-family: 'SF Mono', Menlo, monospace; }}
  .temp-tag {{ display: inline-block; font-size: 0.7rem; padding: 0.1rem 0.4rem; background: #21262d; border: 1px solid #d29922; color: #d29922; border-radius: 3px; font-family: 'SF Mono', Menlo, monospace; }}

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
<p class="meta">Generated {now} &middot; AFM MLX Backend</p>

<div class="summary">
  <div class="card"><div class="label">Test Runs</div><div class="value blue">{len(results)}</div></div>
  <div class="card"><div class="label">Passed</div><div class="value green">{len(ok)}</div></div>
  <div class="card"><div class="label">Failed</div><div class="value red">{len(fail)}</div></div>
  <div class="card"><div class="label">Best tok/s</div><div class="value yellow">{best_tps}</div></div>
  <div class="card"><div class="label">Fastest</div><div class="value" style="font-size:1rem;color:#d29922">{best_model}</div></div>
</div>

<h2>Performance Ranking (by tokens/sec)</h2>
<p style="color:#8b949e;font-size:0.85rem;margin-bottom:0.5rem">Click a row to jump to its full response below.</p>
<table>
<tr>
  <th>#</th>
  <th>Model / Config</th>
  <th>Status</th>
  {ai_score_header}
  <th>Temp</th>
  <th>Load (s)</th>
  <th>Tokens</th>
  <th>Gen (s)</th>
  <th style="min-width:200px">Tokens/sec</th>
  <th>Prompt</th>
</tr>
{perf_rows}
</table>

{"<h2>Failed Runs</h2>" + chr(10) + "<table>" + chr(10) + "<tr><th>Model</th><th>Error</th>" + (ai_score_header) + "<th>Config</th><th>Load (s)</th></tr>" + chr(10) + fail_rows + "</table>" if fail else ""}

{smart_section}

<h2>Full Responses</h2>
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

os.makedirs(report_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
html_path = os.path.join(report_dir, f"mlx-model-report-{timestamp}.html")
jsonl_path = os.path.join(report_dir, f"mlx-model-report-{timestamp}.jsonl")

with open(html_path, "w") as f:
    f.write(report)

# Copy results JSONL alongside the HTML report with matching timestamp
shutil.copy2("/tmp/mlx-test-results.jsonl", jsonl_path)

print(f"Report: {html_path}")
print(f"  Data: {jsonl_path}")
print(f"  {len(ok)} passed, {len(fail)} failed out of {len(results)} models")

# Auto-open the HTML report on macOS
if sys.platform == "darwin":
    subprocess.run(["open", html_path])
