#!/usr/bin/env python3
"""
End-to-end test suite for --tool-call-parser options.
Tests each parser with a compatible locally-cached model.

Usage:
  python3 Scripts/tests/test-tool-call-parsers.py [--parsers hermes,gemma,qwen3_xml] [--port 9998]

Outputs:
  test-reports/tool-call-parsers-<timestamp>.jsonl   — machine-readable results
  test-reports/tool-call-parsers-<timestamp>.html    — human-readable report
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import datetime
import argparse
import traceback
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

AFM_BINARY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../.build/arm64-apple-macosx/release/afm",
)
MODEL_CACHE = os.environ.get(
    "MACAFM_MLX_MODEL_CACHE", "/Volumes/edata/models/vesta-test-cache"
)
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../test-reports")

# Parser → (model_id, reason)
PARSER_MODELS = {
    "hermes":     ("mlx-community/Qwen3-30B-A3B-4bit", "ChatML tokens available"),
    "llama3_json": ("mlx-community/Llama-3.3-70B-Instruct-4bit-DWQ", "Llama 3.3 70B"),
    "gemma":      ("mlx-community/functiongemma-270m-it-bf16", "FunctionGemma 270M"),
    "mistral":    ("mlx-community/mistralai_Devstral-Small-2-24B-Instruct-2512-MLX-8Bit", "Devstral Small 2 24B 8-bit"),
    "qwen3_xml":  ("mlx-community/Qwen3-Coder-Next-4bit", "Native Qwen3 XML"),
}

# Regression tests: models with vendor auto-detection (no --tool-call-parser).
# Verifies that adding new parsers didn't break existing auto-detect paths.
# Format: (model_id, expected_format_description, large_flag)
REGRESSION_MODELS = [
    ("mlx-community/GLM-4.7-Flash-4bit",  "auto-detect .glm4",     True),
    ("mlx-community/Qwen3-30B-A3B-4bit",  "auto-detect qwen3_moe", False),
    ("mlx-community/Qwen3-Coder-Next-4bit", "auto-detect .xmlFunction", True),
]

# Tools for testing
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'San Francisco'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 3 * 4'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Test cases: (name, prompt, expected_tool_name_or_None, stream)
TEST_CASES = [
    ("weather_nonstream", "What is the weather in Tokyo?", "get_weather", False),
    ("weather_stream",    "What is the weather in London right now?", "get_weather", True),
    ("calc_nonstream",    "Use the calculator to compute 17 * 23 + 5", "calculate", False),
    ("calc_stream",       "Please calculate 99 * 101 using the tool", "calculate", True),
]


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_for_server(port, proc=None, timeout=180):
    """Wait for server to respond on /v1/models."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    attempts = 0
    while time.time() < deadline:
        # Check if process died
        if proc and proc.poll() is not None:
            log(f"  Server process exited with code {proc.returncode}")
            if hasattr(proc, '_log_path') and os.path.exists(proc._log_path):
                with open(proc._log_path) as f:
                    tail = f.read()[-500:]
                log(f"  Server log tail: {tail}")
            return False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError) as e:
            if attempts % 10 == 0 and attempts > 0:
                log(f"  Still waiting... ({attempts} attempts, {e.__class__.__name__})")
            pass
        attempts += 1
        time.sleep(2)
    # Timeout — dump server log
    if proc and hasattr(proc, '_log_path') and os.path.exists(proc._log_path):
        with open(proc._log_path) as f:
            tail = f.read()[-1000:]
        log(f"  Server log tail on timeout:\n{tail}")
    return False


def start_server(model_id, parser, port):
    """Start afm mlx server, return subprocess. parser=None means no --tool-call-parser."""
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    env["AFM_DEBUG"] = "1"

    cmd = [
        AFM_BINARY, "mlx",
        "-m", model_id,
        "--port", str(port),
        "--max-tokens", "1024",
    ]
    if parser:
        cmd.extend(["--tool-call-parser", parser])
    label = parser or "auto"
    log(f"Starting server: {' '.join(cmd)}")
    # Write server output to a temp log file for debugging
    log_path = os.path.join(REPORT_DIR, f"server-{label}-{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # new session for clean kill
    )
    proc._log_file = log_file  # keep reference to close later
    proc._log_path = log_path
    return proc


def kill_server(proc):
    """Kill server process group."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception:
                pass
    if hasattr(proc, '_log_file'):
        try:
            proc._log_file.close()
        except Exception:
            pass


def send_request(port, prompt, tools, stream=False, timeout=120):
    """Send chat completion request, return parsed response dict."""
    import urllib.request

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. When the user asks about weather or calculations, ALWAYS use the provided tools. Do not answer directly."},
            {"role": "user", "content": prompt},
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": stream,
    }

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    start = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        return {
            "elapsed": elapsed,
            "choices": body.get("choices", []),
            "usage": body.get("usage", {}),
            "raw": body,
        }
    else:
        # Streaming: collect SSE chunks
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tool_calls = {}  # index → {id, name, arguments}
            content_parts = []
            finish_reason = None
            usage = {}

            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break
                chunk = json.loads(payload_str)
                delta = chunk["choices"][0].get("delta", {})
                fr = chunk["choices"][0].get("finish_reason")
                if fr:
                    finish_reason = fr
                if chunk.get("usage"):
                    usage = chunk["usage"]

                if "tool_calls" in delta and delta["tool_calls"]:
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if idx not in tool_calls:
                            tool_calls[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                        if tc.get("id"):
                            tool_calls[idx]["id"] = tc["id"]
                        func = tc.get("function", {})
                        if func.get("name"):
                            tool_calls[idx]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_calls[idx]["arguments"] += func["arguments"]

                if delta.get("content"):
                    content_parts.append(delta["content"])

        elapsed = time.time() - start
        # Build synthetic choices structure
        tc_list = []
        for idx in sorted(tool_calls.keys()):
            tc = tool_calls[idx]
            tc_list.append({
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            })
        message = {
            "content": "".join(content_parts),
            "tool_calls": tc_list if tc_list else None,
        }
        choices = [{"message": message, "finish_reason": finish_reason}]
        return {"elapsed": elapsed, "choices": choices, "usage": usage, "raw": None}


def evaluate_response(result, expected_tool):
    """Evaluate a response, return (pass, details_dict)."""
    details = {
        "elapsed_s": round(result["elapsed"], 2),
    }

    if not result["choices"]:
        return False, {**details, "error": "No choices in response"}

    choice = result["choices"][0]
    msg = choice.get("message", {})
    finish = choice.get("finish_reason", "")

    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content", "") or ""

    if expected_tool:
        if not tool_calls:
            return False, {
                **details,
                "error": "Expected tool_calls but got none",
                "finish_reason": finish,
                "content_snippet": content[:200],
            }

        tc = tool_calls[0]
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "")

        # Check tool name
        if name != expected_tool:
            return False, {
                **details,
                "error": f"Expected tool '{expected_tool}' but got '{name}'",
                "tool_calls": tool_calls,
            }

        # Check arguments are valid JSON
        try:
            args = json.loads(args_str)
            if not isinstance(args, dict):
                return False, {
                    **details,
                    "error": f"Arguments not a dict: {type(args).__name__}",
                    "arguments_raw": args_str,
                }
        except json.JSONDecodeError as e:
            return False, {
                **details,
                "error": f"Invalid JSON arguments: {e}",
                "arguments_raw": args_str[:500],
            }

        # Check expected argument keys
        if expected_tool == "get_weather" and "city" not in args:
            return False, {
                **details,
                "error": f"Missing 'city' key in arguments",
                "arguments": args,
            }
        if expected_tool == "calculate" and "expression" not in args:
            return False, {
                **details,
                "error": f"Missing 'expression' key in arguments",
                "arguments": args,
            }

        return True, {
            **details,
            "tool_name": name,
            "arguments": args,
            "finish_reason": finish,
            "num_tool_calls": len(tool_calls),
        }
    else:
        # Expected no tool call — just a text response
        if tool_calls:
            return False, {
                **details,
                "error": "Got unexpected tool_calls",
                "tool_calls": tool_calls,
            }
        return True, {
            **details,
            "content_snippet": content[:200],
            "finish_reason": finish,
        }


def run_parser_tests(parser, model_id, port, results):
    """Run all test cases for a parser. Appends to results list."""
    log(f"─── Testing parser: {parser} with {model_id} ───")

    proc = None
    try:
        proc = start_server(model_id, parser, port)
        log(f"Waiting for server (pid={proc.pid})...")

        if not wait_for_server(port, proc=proc, timeout=240):
            results.append({
                "parser": parser,
                "model": model_id,
                "test": "server_start",
                "passed": False,
                "details": {"error": "Server failed to start within 240s"},
                "timestamp": datetime.datetime.now().isoformat(),
            })
            return

        results.append({
            "parser": parser,
            "model": model_id,
            "test": "server_start",
            "passed": True,
            "details": {"message": "Server started successfully"},
            "timestamp": datetime.datetime.now().isoformat(),
        })

        for test_name, prompt, expected_tool, stream in TEST_CASES:
            full_name = f"{test_name}"
            log(f"  Test: {full_name} (stream={stream})")
            try:
                result = send_request(port, prompt, TOOLS, stream=stream, timeout=120)
                passed, details = evaluate_response(result, expected_tool)
                status = "PASS" if passed else "FAIL"
                log(f"    {status}: {json.dumps(details, default=str)}")
                results.append({
                    "parser": parser,
                    "model": model_id,
                    "test": full_name,
                    "stream": stream,
                    "passed": passed,
                    "details": details,
                    "timestamp": datetime.datetime.now().isoformat(),
                })
            except Exception as e:
                log(f"    ERROR: {e}")
                results.append({
                    "parser": parser,
                    "model": model_id,
                    "test": full_name,
                    "stream": stream,
                    "passed": False,
                    "details": {"error": str(e), "traceback": traceback.format_exc()},
                    "timestamp": datetime.datetime.now().isoformat(),
                })

    finally:
        if proc:
            log(f"Stopping server (pid={proc.pid})...")
            kill_server(proc)
            log("Server stopped.")
        # Pause to release port and GPU memory
        time.sleep(5)


def run_regression_tests(model_id, description, port, results):
    """Run tool calling tests WITHOUT --tool-call-parser (auto-detect). Appends to results list."""
    parser_label = f"auto:{model_id.split('/')[-1]}"
    log(f"─── Regression: {model_id} ({description}) ───")

    proc = None
    try:
        proc = start_server(model_id, None, port)  # parser=None
        log(f"Waiting for server (pid={proc.pid})...")

        if not wait_for_server(port, proc=proc, timeout=240):
            results.append({
                "parser": parser_label,
                "model": model_id,
                "test": "server_start",
                "passed": False,
                "details": {"error": "Server failed to start within 240s", "description": description},
                "timestamp": datetime.datetime.now().isoformat(),
            })
            return

        results.append({
            "parser": parser_label,
            "model": model_id,
            "test": "server_start",
            "passed": True,
            "details": {"message": "Server started successfully", "description": description},
            "timestamp": datetime.datetime.now().isoformat(),
        })

        # Run subset of tests: one sync, one streaming
        regression_cases = [
            ("weather_nonstream", "What is the weather in Tokyo?", "get_weather", False),
            ("calc_stream",      "Please calculate 99 * 101 using the tool", "calculate", True),
        ]

        for test_name, prompt, expected_tool, stream in regression_cases:
            log(f"  Test: {test_name} (stream={stream})")
            try:
                result = send_request(port, prompt, TOOLS, stream=stream, timeout=180)
                passed, details = evaluate_response(result, expected_tool)
                status = "PASS" if passed else "FAIL"
                log(f"    {status}: {json.dumps(details, default=str)}")
                results.append({
                    "parser": parser_label,
                    "model": model_id,
                    "test": test_name,
                    "stream": stream,
                    "passed": passed,
                    "details": details,
                    "timestamp": datetime.datetime.now().isoformat(),
                })
            except Exception as e:
                log(f"    ERROR: {e}")
                results.append({
                    "parser": parser_label,
                    "model": model_id,
                    "test": test_name,
                    "stream": stream,
                    "passed": False,
                    "details": {"error": str(e), "traceback": traceback.format_exc()},
                    "timestamp": datetime.datetime.now().isoformat(),
                })

    finally:
        if proc:
            log(f"Stopping server (pid={proc.pid})...")
            kill_server(proc)
            log("Server stopped.")
        time.sleep(5)


def generate_html_report(results, timestamp, jsonl_path):
    """Generate a styled HTML report."""
    total = sum(1 for r in results if r["test"] not in ("server_start", "skipped") and r.get("model"))
    passed = sum(1 for r in results if r["test"] not in ("server_start", "skipped") and r.get("model") and r["passed"])
    failed = total - passed
    skipped = sum(1 for r in results if not r.get("model"))

    # Group by parser
    parsers_seen = []
    parser_results = {}
    for r in results:
        p = r["parser"]
        if p not in parser_results:
            parser_results[p] = []
            parsers_seen.append(p)
        parser_results[p].append(r)

    all_pass = failed == 0 and total > 0
    badge_color = "#22c55e" if all_pass else "#ef4444"
    badge_text = f"{passed}/{total} passed" if total > 0 else "no tests"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tool Call Parser Test Report — {timestamp}</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
           --pass: #22c55e; --fail: #ef4444; --skip: #f59e0b; --accent: #38bdf8; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Segoe UI', sans-serif;
          background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.6; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .summary {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
           padding: 1rem 1.5rem; flex: 1; min-width: 140px; }}
  .stat .label {{ font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 1.75rem; font-weight: 700; }}
  .stat .value.pass {{ color: var(--pass); }}
  .stat .value.fail {{ color: var(--fail); }}
  .stat .value.skip {{ color: var(--skip); }}
  .badge {{ display: inline-block; background: {badge_color}; color: #fff; font-size: 0.8rem;
            font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 99px; margin-left: 0.5rem;
            vertical-align: middle; }}
  .parser-section {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px;
                     margin-bottom: 1.5rem; overflow: hidden; }}
  .parser-header {{ padding: 1rem 1.5rem; border-bottom: 1px solid var(--border);
                    display: flex; align-items: center; justify-content: space-between; }}
  .parser-header h2 {{ font-size: 1.1rem; font-weight: 600; }}
  .parser-header .model {{ color: #94a3b8; font-size: 0.85rem; font-family: monospace; }}
  .parser-header .parser-badge {{ font-size: 0.75rem; padding: 0.15rem 0.6rem; border-radius: 6px;
                                  font-weight: 600; }}
  .parser-badge.pass {{ background: rgba(34,197,94,0.15); color: var(--pass); }}
  .parser-badge.fail {{ background: rgba(239,68,68,0.15); color: var(--fail); }}
  .parser-badge.skip {{ background: rgba(245,158,11,0.15); color: var(--skip); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; font-size: 0.75rem; text-transform: uppercase; color: #64748b;
       padding: 0.6rem 1rem; letter-spacing: 0.05em; }}
  td {{ padding: 0.6rem 1rem; border-top: 1px solid var(--border); font-size: 0.9rem; }}
  td.test-name {{ font-family: monospace; font-weight: 500; }}
  .result-pass {{ color: var(--pass); font-weight: 600; }}
  .result-fail {{ color: var(--fail); font-weight: 600; }}
  .details {{ font-size: 0.8rem; color: #94a3b8; max-width: 400px; word-break: break-word; }}
  .details code {{ background: rgba(56,189,248,0.1); color: var(--accent); padding: 0.1rem 0.3rem;
                   border-radius: 3px; font-size: 0.8rem; }}
  .stream-badge {{ font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 4px;
                   background: rgba(56,189,248,0.12); color: var(--accent); margin-left: 0.5rem; }}
  footer {{ text-align: center; margin-top: 2rem; color: #475569; font-size: 0.8rem; }}
  footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<div class="container">
  <h1>Tool Call Parser Test Report <span class="badge">{badge_text}</span></h1>
  <div class="subtitle">Generated {timestamp} &mdash; JSONL: <code>{os.path.basename(jsonl_path)}</code></div>

  <div class="summary">
    <div class="stat"><div class="label">Passed</div><div class="value pass">{passed}</div></div>
    <div class="stat"><div class="label">Failed</div><div class="value fail">{failed}</div></div>
    <div class="stat"><div class="label">Skipped</div><div class="value skip">{skipped}</div></div>
    <div class="stat"><div class="label">Total Tests</div><div class="value">{total}</div></div>
  </div>
"""

    for parser in parsers_seen:
        p_results = parser_results[parser]
        model = p_results[0].get("model") or "—"
        p_tests = [r for r in p_results if r["test"] != "server_start"]
        p_total = len(p_tests)
        p_passed = sum(1 for r in p_tests if r["passed"])
        is_skipped = not p_results[0].get("model")

        if is_skipped:
            badge_cls = "skip"
            badge_label = "SKIPPED"
        elif p_passed == p_total and p_total > 0:
            badge_cls = "pass"
            badge_label = f"{p_passed}/{p_total} PASS"
        else:
            badge_cls = "fail"
            badge_label = f"{p_passed}/{p_total} PASS"

        reason = p_results[0].get("details", {}).get("reason", "")
        model_display = model if model != "—" else f"— {reason}"

        is_regression = parser.startswith("auto:")
        heading = f"Regression: {parser.replace('auto:', '')}" if is_regression else f"--tool-call-parser {parser}"

        html += f"""
  <div class="parser-section">
    <div class="parser-header">
      <div>
        <h2>{heading}</h2>
        <div class="model">{model_display}</div>
      </div>
      <span class="parser-badge {badge_cls}">{badge_label}</span>
    </div>
"""

        if is_skipped:
            html += f'    <div style="padding: 1rem 1.5rem; color: #94a3b8;">No compatible model available in cache. {reason}</div>\n'
        else:
            html += """    <table>
      <thead><tr><th>Test</th><th>Mode</th><th>Result</th><th>Time</th><th>Details</th></tr></thead>
      <tbody>
"""
            # Include server_start row
            for r in p_results:
                test = r["test"]
                is_pass = r["passed"]
                details = r.get("details", {})
                elapsed = details.get("elapsed_s", "—")
                stream = r.get("stream", False)

                result_cls = "result-pass" if is_pass else "result-fail"
                result_text = "PASS" if is_pass else "FAIL"
                mode = ""
                if test == "server_start":
                    mode = "—"
                elif stream:
                    mode = '<span class="stream-badge">SSE</span>'
                else:
                    mode = "sync"

                # Build details string
                detail_parts = []
                if details.get("tool_name"):
                    detail_parts.append(f'tool: <code>{details["tool_name"]}</code>')
                if details.get("arguments"):
                    args_str = json.dumps(details["arguments"], ensure_ascii=False)
                    if len(args_str) > 80:
                        args_str = args_str[:80] + "..."
                    detail_parts.append(f'args: <code>{args_str}</code>')
                if details.get("error"):
                    err = details["error"]
                    if len(err) > 120:
                        err = err[:120] + "..."
                    detail_parts.append(f'<span style="color:var(--fail)">{err}</span>')
                if details.get("content_snippet"):
                    snippet = details["content_snippet"][:100]
                    detail_parts.append(f'content: "{snippet}"')
                if details.get("message"):
                    detail_parts.append(details["message"])
                if details.get("finish_reason"):
                    detail_parts.append(f'finish: <code>{details["finish_reason"]}</code>')

                detail_html = "<br>".join(detail_parts) if detail_parts else "—"

                html += f'        <tr><td class="test-name">{test}</td><td>{mode}</td>'
                html += f'<td class="{result_cls}">{result_text}</td><td>{elapsed}</td>'
                html += f'<td class="details">{detail_html}</td></tr>\n'

            html += "      </tbody>\n    </table>\n"

        html += "  </div>\n"

    html += f"""
  <footer>
    afm tool-call-parser test suite &mdash; <a href="https://github.com/scouzi1966/maclocal-api">maclocal-api</a>
  </footer>
</div>
</body>
</html>
"""
    return html


def main():
    ap = argparse.ArgumentParser(description="Test tool call parsers")
    ap.add_argument("--parsers", default=None, help="Comma-separated parser names to test (default: all with models)")
    ap.add_argument("--port", type=int, default=9998, help="Port for test server (default: 9998)")
    ap.add_argument("--include-large", action="store_true", help="Include large models (>30GB: Llama 70B, Qwen3-Coder-Next 42GB, etc.)")
    ap.add_argument("--no-regression", action="store_true", help="Skip regression tests (auto-detect without parser)")
    ap.add_argument("--regression-only", action="store_true", help="Only run regression tests")
    args = ap.parse_args()

    # Resolve binary
    if not os.path.isfile(AFM_BINARY):
        print(f"ERROR: Binary not found at {AFM_BINARY}")
        print("Run: swift build -c release")
        sys.exit(1)

    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(REPORT_DIR, f"tool-call-parsers-{timestamp}.jsonl")
    html_path = os.path.join(REPORT_DIR, f"tool-call-parsers-{timestamp}.html")

    # Large model thresholds (model_id substring → skip unless --include-large)
    LARGE_MODELS = ["Coder-Next", "70B", "397B", "Kimi-K2"]

    results = []

    # ── Parser tests ──
    if not args.regression_only:
        if args.parsers:
            selected = [p.strip() for p in args.parsers.split(",")]
        else:
            selected = list(PARSER_MODELS.keys())

        # Filter large models unless --include-large
        if not args.include_large:
            filtered = []
            for p in selected:
                model_id, _ = PARSER_MODELS.get(p, (None, ""))
                if model_id and any(tag in model_id for tag in LARGE_MODELS):
                    log(f"Skipping {p} (large model: {model_id}). Use --include-large to include.")
                else:
                    filtered.append(p)
            selected = filtered

        log(f"═══ Tool Call Parser Test Suite ═══")
        log(f"Binary: {AFM_BINARY}")
        log(f"Cache: {MODEL_CACHE}")
        log(f"Parsers: {', '.join(selected)}")
        log(f"Port: {args.port}")
        log(f"")

        for parser in selected:
            model_id, reason = PARSER_MODELS.get(parser, (None, "Unknown parser"))

            if not model_id:
                log(f"─── SKIP parser: {parser} — {reason} ───")
                results.append({
                    "parser": parser,
                    "model": None,
                    "test": "skipped",
                    "passed": False,
                    "details": {"reason": reason},
                    "timestamp": datetime.datetime.now().isoformat(),
                })
                continue

            # Verify model exists on disk
            model_short = model_id.split("/")[-1]
            model_dir = os.path.join(MODEL_CACHE, "mlx-community", model_short)
            if not os.path.isdir(model_dir):
                log(f"─── SKIP parser: {parser} — model dir not found: {model_dir} ───")
                results.append({
                    "parser": parser,
                    "model": model_id,
                    "test": "skipped",
                    "passed": False,
                    "details": {"reason": f"Model directory not found: {model_dir}"},
                    "timestamp": datetime.datetime.now().isoformat(),
                })
                continue

            run_parser_tests(parser, model_id, args.port, results)

    # ── Regression tests (no --tool-call-parser) ──
    if not args.no_regression:
        log(f"")
        log(f"═══ Regression Tests (auto-detect, no parser override) ═══")
        for model_id, description, is_large in REGRESSION_MODELS:
            if is_large and not args.include_large:
                label = f"auto:{model_id.split('/')[-1]}"
                log(f"─── SKIP regression: {model_id} (large). Use --include-large ───")
                results.append({
                    "parser": label,
                    "model": None,
                    "test": "skipped",
                    "passed": False,
                    "details": {"reason": f"Large model, skipped without --include-large"},
                    "timestamp": datetime.datetime.now().isoformat(),
                })
                continue

            model_short = model_id.split("/")[-1]
            model_dir = os.path.join(MODEL_CACHE, "mlx-community", model_short)
            if not os.path.isdir(model_dir):
                label = f"auto:{model_short}"
                log(f"─── SKIP regression: {model_id} — not found ───")
                results.append({
                    "parser": label,
                    "model": model_id,
                    "test": "skipped",
                    "passed": False,
                    "details": {"reason": f"Model directory not found: {model_dir}"},
                    "timestamp": datetime.datetime.now().isoformat(),
                })
                continue

            run_regression_tests(model_id, description, args.port, results)

    # Write JSONL
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    log(f"JSONL: {jsonl_path}")

    # Write HTML
    html = generate_html_report(results, timestamp, jsonl_path)
    with open(html_path, "w") as f:
        f.write(html)
    log(f"HTML:  {html_path}")

    # Summary
    total = sum(1 for r in results if r["test"] not in ("server_start", "skipped") and r.get("model"))
    passed = sum(1 for r in results if r["test"] not in ("server_start", "skipped") and r.get("model") and r["passed"])
    log(f"")
    log(f"═══ Results: {passed}/{total} tests passed ═══")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
