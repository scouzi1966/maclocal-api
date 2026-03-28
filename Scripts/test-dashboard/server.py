#!/usr/bin/env python3
"""
AFM Test Dashboard Server

A standalone Python HTTP server (stdlib only) that orchestrates AFM test suites
and streams progress via SSE. Serves a web dashboard at http://localhost:8080.

Usage:
    python3 Scripts/test-dashboard/server.py [--port 8080]
"""

import argparse
import datetime
import json
import os
import pathlib
import queue
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, IO, Optional

try:
    from openai import OpenAI as _OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def _openai_client(port: int) -> Any:
    """Create an OpenAI client pointing at the local AFM server."""
    if not _HAS_OPENAI:
        return None
    return _OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")


def _wait_for_server(port: int, timeout: float = 120.0, poll: float = 1.0,
                     proc: Optional[subprocess.Popen] = None) -> bool:
    """Wait for AFM server to be ready using OpenAI SDK (preferred) or raw HTTP fallback."""
    deadline = time.monotonic() + timeout
    client = _openai_client(port)

    while time.monotonic() < deadline:
        if _stop_requested.is_set():
            return False
        if proc and proc.poll() is not None:
            return False
        try:
            if client:
                # Use the SDK — this is itself an API compatibility test
                models = client.models.list()
                if models.data:
                    return True
            else:
                # Fallback to raw HTTP if openai not installed
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(poll)
    return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CACHE = "/Volumes/edata/models/vesta-test-cache"
HEALTH_POLL_INTERVAL = 1.0  # seconds
HEALTH_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------

SUITES: dict[str, dict[str, Any]] = {
    "unit": {
        "label": "Unit Tests",
        "cmd": ["swift", "test"],
        "needs_server": False,
        "port": None,
        "est_minutes": 0.1,
        "parse_pattern": r"Test run with (\d+) tests.*passed",
    },
    "assertions-smoke": {
        "label": "Assertions (smoke)",
        "cmd_template": [
            "./Scripts/test-assertions.sh",
            "--tier", "smoke",
            "--model", "{model}",
            "--port", "9998",
            "--bin", "{binary}",
        ],
        "needs_server": True,
        "port": 9998,
        "server_flags": [
            "--tool-call-parser", "afm_adaptive_xml",
            "--enable-prefix-caching",
            "--enable-grammar-constraints",
        ],
        "est_minutes": 2,
        "parse_pattern": r"(\u2705|\u274c)\s+(.+)",
    },
    "assertions-standard": {
        "label": "Assertions (standard)",
        "cmd_template": [
            "./Scripts/test-assertions.sh",
            "--tier", "standard",
            "--model", "{model}",
            "--port", "9998",
            "--bin", "{binary}",
            "--grammar-constraints",
        ],
        "needs_server": True,
        "port": 9998,
        "server_flags": [
            "--tool-call-parser", "afm_adaptive_xml",
            "--enable-prefix-caching",
            "--enable-grammar-constraints",
        ],
        "est_minutes": 5,
        "parse_pattern": r"(\u2705|\u274c)\s+(.+)",
    },
    "assertions-full": {
        "label": "Assertions (full)",
        "cmd_template": [
            "./Scripts/test-assertions.sh",
            "--tier", "full",
            "--model", "{model}",
            "--port", "9998",
            "--bin", "{binary}",
            "--grammar-constraints",
        ],
        "needs_server": True,
        "port": 9998,
        "server_flags": [
            "--tool-call-parser", "afm_adaptive_xml",
            "--enable-prefix-caching",
            "--enable-grammar-constraints",
        ],
        "est_minutes": 15,
        "parse_pattern": r"(\u2705|\u274c)\s+(.+)",
    },
    "assertions-grammar": {
        "label": "Assertions + Grammar + Forced Parser",
        "cmd_template": [
            "./Scripts/test-assertions-multi.sh",
            "--models", "{model}",
            "--tier", "full",
            "--also-forced-parser", "qwen3_xml",
            "--grammar-constraints",
        ],
        "needs_server": False,
        "port": 9998,
        "est_minutes": 30,
        "parse_pattern": r"(\u2705|\u274c)\s+(.+)",
        "env_extras": {"AFM_BINARY": "{binary}"},
    },
    "smart-analysis": {
        "label": "Comprehensive Smart Analysis",
        "cmd_template": [
            "./Scripts/mlx-model-test.sh",
            "--model", "{model}",
            "--prompts", "Scripts/test-llm-comprehensive.txt",
            "--smart", "1:claude",
        ],
        "needs_server": False,
        "port": 9877,
        "est_minutes": 60,
        "parse_pattern": r"\[(\d+)/(\d+)\].*score=(\d)",
        "env_extras": {"AFM_BIN": "{binary}"},
    },
    "promptfoo": {
        "label": "Promptfoo Agentic Evals",
        "cmd_template": [
            "./Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh",
            "all",
        ],
        "needs_server": False,
        "port": 9999,
        "est_minutes": 90,
        "parse_pattern": r"Results: [\u2713\u2717] (\d+) passed, (\d+) failed",
        "env_extras": {"AFM_MODEL": "{model}", "AFM_BINARY": "{binary}"},
    },
    "batch-correctness": {
        "label": "Batch Correctness B={1,2,4,8}",
        "cmd_template": [
            "python3",
            "Scripts/feature-mlx-concurrent-batch/validate_responses.py",
        ],
        "needs_server": True,
        "port": 9999,
        "server_flags": ["--concurrent", "8"],
        "est_minutes": 12,
        "parse_pattern": r"(PASS|FAIL)",
    },
    "batch-mixed": {
        "label": "Batch Mixed Workload",
        "cmd_template": [
            "python3",
            "Scripts/feature-mlx-concurrent-batch/validate_mixed_workload.py",
        ],
        "needs_server": True,
        "port": 9999,
        "server_flags": ["--concurrent", "8"],
        "est_minutes": 20,
        "parse_pattern": r"(PASS|FAIL)",
    },
    "batch-multiturn": {
        "label": "Batch Multiturn Prefix",
        "cmd_template": [
            "python3",
            "Scripts/feature-mlx-concurrent-batch/validate_multiturn_prefix.py",
        ],
        "needs_server": True,
        "port": 9999,
        "server_flags": ["--concurrent", "8", "--enable-prefix-caching"],
        "est_minutes": 20,
        "parse_pattern": r"(PASS|FAIL)",
    },
    "openai-compat": {
        "label": "OpenAI Compat Evals",
        "cmd_template": [
            "python3",
            "Scripts/feature-codex-optimize-api/test-openai-compat-evals.py",
            "--start-server",
            "--model", "{model}",
        ],
        "needs_server": False,
        "port": 9999,
        "est_minutes": 8,
        "parse_pattern": r"(PASS|FAIL|\u2705|\u274c)",
        "env_extras": {"AFM_BINARY": "{binary}"},
    },
    "guided-json": {
        "label": "Guided JSON Evals",
        "cmd_template": [
            "python3",
            "Scripts/feature-codex-optimize-api/test-guided-json-evals.py",
            "--start-server",
            "--model", "{model}",
        ],
        "needs_server": False,
        "port": 9999,
        "est_minutes": 12,
        "parse_pattern": r"(PASS|FAIL|\u2705|\u274c)",
        "env_extras": {"AFM_BINARY": "{binary}"},
    },
    "gpu-profile": {
        "label": "GPU Profile",
        "cmd_template": [
            "python3",
            "Scripts/gpu-profile-report.py",
            "{model}",
        ],
        "needs_server": False,
        "port": None,
        "est_minutes": 1,
        "parse_pattern": r"Report:",
        "env_extras": {"AFM_BIN": "{binary}"},
    },
}


# ---------------------------------------------------------------------------
# Globals (shared state)
# ---------------------------------------------------------------------------

REPO_ROOT: str = ""
SCRIPT_DIR: str = ""
LOG_DIR: str = ""

# SSE subscribers: list of queue.Queue instances, one per connected client
_sse_lock = threading.Lock()
_sse_subscribers: list[queue.Queue] = []

# Active run state
_run_lock = threading.Lock()
_active_run_id: Optional[str] = None
_active_procs: list[subprocess.Popen] = []
_active_threads: list[threading.Thread] = []
_stop_requested = threading.Event()
_promptfoo_proc: Optional[subprocess.Popen] = None
_user_afm_proc: Optional[subprocess.Popen] = None
_user_afm_log: Optional[IO] = None

# JSONL log file handle for current run
_log_lock = threading.Lock()
_log_fh = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _make_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _emit_sse(event: dict) -> None:
    """Send an event to all SSE subscribers and write to JSONL log."""
    event.setdefault("timestamp", _now_iso())
    payload = json.dumps(event, ensure_ascii=False)

    with _sse_lock:
        dead: list[queue.Queue] = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)

    _write_log(payload)


def _write_log(line: str) -> None:
    with _log_lock:
        if _log_fh is not None:
            try:
                _log_fh.write(line + "\n")
                _log_fh.flush()
            except Exception:
                pass


def _open_log(run_id: str):
    global _log_fh
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{run_id}.jsonl")
    with _log_lock:
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass
        _log_fh = open(log_path, "a", encoding="utf-8")

    # Symlink LATEST.jsonl
    latest = os.path.join(LOG_DIR, "LATEST.jsonl")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(f"{run_id}.jsonl", latest)
    except OSError:
        pass


def _close_log():
    global _log_fh
    with _log_lock:
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass
            _log_fh = None


def _detect_binary() -> dict:
    """Detect the AFM binary, checking release paths in order."""
    candidates = [
        os.path.join(REPO_ROOT, ".build", "arm64-apple-macosx", "release", "afm"),
        os.path.join(REPO_ROOT, ".build", "release", "afm"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            version = _get_version(path)
            return {"path": path, "version": version, "exists": True}
    return {"path": "", "version": "", "exists": False}


def _get_version(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=10,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return f"error: {e}"


def _list_models() -> list[dict]:
    """Run list-models.sh and parse output."""
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    script = os.path.join(REPO_ROOT, "Scripts", "list-models.sh")
    if not os.path.isfile(script):
        return []
    try:
        result = subprocess.run(
            [script],
            capture_output=True, text=True, timeout=30,
            cwd=REPO_ROOT, env=env,
        )
        models = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("MACAFM") or line.startswith("Error") or "models found" in line:
                continue
            # Format: "org/model                     19.0 GB"
            m = re.match(r"^(\S+)\s+([\d.]+)\s+GB", line)
            if m:
                models.append({"id": m.group(1), "size_gb": float(m.group(2))})
        return models
    except Exception:
        return []


def _check_metallib(binary: str) -> dict:
    """Check for default.metallib next to the binary or in a bundle."""
    binary_dir = os.path.dirname(binary)
    # Loose metallib
    loose = os.path.join(binary_dir, "default.metallib")
    if os.path.isfile(loose):
        return {"status": "pass", "location": loose}
    # Check in a .bundle directory next to binary
    for entry in os.listdir(binary_dir):
        full = os.path.join(binary_dir, entry)
        if entry.endswith(".bundle") and os.path.isdir(full):
            bundle_metallib = os.path.join(full, "Contents", "Resources", "default.metallib")
            if os.path.isfile(bundle_metallib):
                return {"status": "pass", "location": bundle_metallib}
            # Also check flat bundle
            flat = os.path.join(full, "default.metallib")
            if os.path.isfile(flat):
                return {"status": "pass", "location": flat}
    return {"status": "fail", "location": ""}


def _check_relocated(binary: str) -> dict:
    """Copy binary + metallib to a temp dir, start it, and verify via OpenAI SDK."""
    tmpdir = None
    proc = None
    test_port = 19876  # ephemeral port for relocated test
    try:
        tmpdir = tempfile.mkdtemp(prefix="afm-relocate-")
        dst_bin = os.path.join(tmpdir, "afm")
        shutil.copy2(binary, dst_bin)
        os.chmod(dst_bin, 0o755)

        # Copy metallib and bundles from binary dir
        binary_dir = os.path.dirname(binary)
        for entry in os.listdir(binary_dir):
            src = os.path.join(binary_dir, entry)
            dst = os.path.join(tmpdir, entry)
            if entry == "afm":
                continue
            if entry.endswith(".metallib"):
                shutil.copy2(src, dst)
            elif entry.endswith(".bundle") and os.path.isdir(src):
                shutil.copytree(src, dst)

        env = os.environ.copy()
        env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE

        # Start the relocated binary as a server
        proc = subprocess.Popen(
            [dst_bin, "mlx", "-m", "mlx-community/SmolLM3-3B-4bit",
             "--port", str(test_port), "--max-tokens", "16"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=tmpdir, env=env,
        )

        # Wait for it to be ready
        if not _wait_for_server(test_port, timeout=60, poll=1.0, proc=proc):
            return {"status": "fail", "error": "Relocated server failed to start"}

        # Use OpenAI SDK to send a completion request (the real test)
        client = _openai_client(test_port)
        if client:
            response = client.chat.completions.create(
                model="mlx-community/SmolLM3-3B-4bit",
                messages=[{"role": "user", "content": "Say hi"}],
                max_tokens=5,
            )
            if response.choices and response.choices[0].message.content:
                return {"status": "pass", "response": response.choices[0].message.content[:50]}
            return {"status": "fail", "error": "Empty response from relocated binary"}
        else:
            # Fallback: raw HTTP if openai not installed
            import json as _json
            req_data = _json.dumps({
                "model": "mlx-community/SmolLM3-3B-4bit",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 5,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{test_port}/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            if resp.status == 200:
                return {"status": "pass"}
            return {"status": "fail", "error": f"HTTP {resp.status}"}

    except Exception as e:
        return {"status": "fail", "error": str(e)}
    finally:
        if proc and proc.poll() is None:
            _kill_proc(proc)
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def _check_bundle_module() -> dict:
    """Check for Bundle.module references in Swift sources."""
    sources_dir = os.path.join(REPO_ROOT, "Sources")
    if not os.path.isdir(sources_dir):
        return {"status": "pass", "hits": 0}
    try:
        result = subprocess.run(
            ["grep", "-r", "Bundle.module", sources_dir, "--include=*.swift"],
            capture_output=True, text=True, timeout=15,
        )
        # Count non-comment hits
        hits = 0
        for line in result.stdout.splitlines():
            # Skip lines that are comments
            stripped = line.split(":", 2)[-1].strip() if ":" in line else line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            hits += 1
        status = "pass" if hits == 0 else "warn"
        return {"status": status, "hits": hits}
    except Exception as e:
        return {"status": "error", "error": str(e), "hits": -1}


def _run_preflight(binary: str) -> dict:
    """Run all pre-flight checks."""
    version_str = _get_version(binary)
    has_sha = bool(re.search(r"-[0-9a-f]{7,}", version_str))
    version_result = {
        "status": "pass" if has_sha else "warn",
        "value": version_str,
    }

    metallib_result = _check_metallib(binary)
    relocated_result = _check_relocated(binary)
    bundle_result = _check_bundle_module()

    return {
        "version": version_result,
        "metallib": metallib_result,
        "relocated": relocated_result,
        "bundle_module": bundle_result,
    }


# ---------------------------------------------------------------------------
# Server lifecycle for suites that need_server
# ---------------------------------------------------------------------------

def _start_afm_server(binary: str, model: str, port: int,
                      extra_flags: list[str]) -> subprocess.Popen:
    """Start an AFM server process and wait for health."""
    cmd = [binary, "mlx", "-m", model, "--port", str(port)]
    cmd.extend(extra_flags)

    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE

    _emit_sse({
        "type": "server_start",
        "port": port,
        "cmd": " ".join(cmd),
    })

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )

    # Drain stdout in background to prevent buffer deadlock
    def _drain():
        try:
            for line in proc.stdout:
                _emit_sse({
                    "type": "server_log",
                    "port": port,
                    "line": line.rstrip("\n"),
                })
        except Exception:
            pass

    drain_t = threading.Thread(target=_drain, daemon=True)
    drain_t.start()

    # Wait for server readiness (uses OpenAI SDK if available)
    healthy = _wait_for_server(port, timeout=HEALTH_TIMEOUT, poll=HEALTH_POLL_INTERVAL, proc=proc)

    if healthy:
        _emit_sse({"type": "server_ready", "port": port})
    else:
        _emit_sse({"type": "server_timeout", "port": port})
        proc.terminate()

    return proc


def _kill_proc(proc: subprocess.Popen) -> None:
    """Terminate a subprocess tree gracefully, then force-kill."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _expand_template(template: list[str], binary: str, model: str) -> list[str]:
    """Expand {binary} and {model} in command templates."""
    return [s.format(binary=binary, model=model) for s in template]


def _expand_env(env_extras: dict[str, str], binary: str, model: str) -> dict[str, str]:
    return {k: v.format(binary=binary, model=model) for k, v in env_extras.items()}


def _run_suite(suite_name: str, suite_def: dict, binary: str, model: str,
               afm_server: Optional[subprocess.Popen]) -> dict:
    """
    Run a single test suite. Returns summary dict.
    Called from the runner thread.
    """
    start = time.monotonic()
    passed = 0
    failed = 0
    lines_captured: list[str] = []

    _emit_sse({"type": "suite_start", "suite": suite_name})

    # Build command
    if "cmd" in suite_def:
        cmd = list(suite_def["cmd"])
    else:
        cmd = _expand_template(suite_def["cmd_template"], binary, model)

    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    if "env_extras" in suite_def:
        env.update(_expand_env(suite_def["env_extras"], binary, model))

    pattern = re.compile(suite_def["parse_pattern"])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        with _run_lock:
            _active_procs.append(proc)

        # Parse output line by line
        for raw_line in proc.stdout:
            if _stop_requested.is_set():
                _kill_proc(proc)
                break

            line = raw_line.rstrip("\n")
            lines_captured.append(line)

            _emit_sse({
                "type": "stdout",
                "suite": suite_name,
                "line": line,
            })

            m = pattern.search(line)
            if m:
                groups = m.groups()
                _parse_match(suite_name, suite_def, groups, line)
                # Count pass/fail from common indicators
                first = groups[0] if groups else ""
                if first in ("\u2705", "PASS") or "passed" in line.lower():
                    if first in ("\u2705", "PASS"):
                        passed += 1
                    else:
                        # "Test run with N tests passed"
                        try:
                            passed += int(groups[0])
                        except (ValueError, IndexError):
                            pass
                elif first in ("\u274c", "FAIL"):
                    failed += 1

        proc.wait(timeout=300)
        returncode = proc.returncode

    except Exception as e:
        _emit_sse({
            "type": "suite_error",
            "suite": suite_name,
            "error": str(e),
        })
        returncode = -1

    finally:
        with _run_lock:
            if proc in _active_procs:
                _active_procs.remove(proc)

    duration_s = round(time.monotonic() - start, 1)

    _emit_sse({
        "type": "suite_end",
        "suite": suite_name,
        "passed": passed,
        "failed": failed,
        "duration_s": duration_s,
        "returncode": returncode,
    })

    return {
        "suite": suite_name,
        "passed": passed,
        "failed": failed,
        "duration_s": duration_s,
        "returncode": returncode,
    }


def _parse_match(suite_name: str, suite_def: dict, groups: tuple, line: str) -> None:
    """Emit fine-grained test_pass / test_fail events from regex matches."""
    first = groups[0] if groups else ""

    if first == "\u2705":
        test_name = groups[1] if len(groups) > 1 else line
        _emit_sse({
            "type": "test_pass",
            "suite": suite_name,
            "test": test_name.strip(),
        })
    elif first == "\u274c":
        test_name = groups[1] if len(groups) > 1 else line
        _emit_sse({
            "type": "test_fail",
            "suite": suite_name,
            "test": test_name.strip(),
            "error": line.strip(),
        })
    elif first == "PASS":
        _emit_sse({"type": "test_pass", "suite": suite_name, "test": line.strip()})
    elif first == "FAIL":
        _emit_sse({"type": "test_fail", "suite": suite_name, "test": line.strip()})


def _apply_suite_options(suite_name: str, suite_def: dict, options: dict) -> dict:
    """Apply user-specified per-suite options to the suite definition.
    Returns a modified copy of the suite_def (never mutates original)."""
    opts = options.get(suite_name, {})
    if not opts:
        return suite_def

    sdef = {k: (list(v) if isinstance(v, list) else v) for k, v in suite_def.items()}
    if "cmd_template" in sdef:
        sdef["cmd_template"] = list(sdef["cmd_template"])
    if "server_flags" in sdef:
        sdef["server_flags"] = list(sdef["server_flags"])

    # Assertions: --section, --grammar-constraints toggle
    if suite_name.startswith("assertions-") and suite_name != "assertions-grammar":
        if opts.get("section"):
            sdef["cmd_template"].extend(["--section", str(opts["section"])])
        if opts.get("grammar") is False:
            # Remove --grammar-constraints from cmd if present
            sdef["cmd_template"] = [a for a in sdef["cmd_template"] if a != "--grammar-constraints"]
            # Remove from server flags too
            if "server_flags" in sdef:
                sdef["server_flags"] = [f for f in sdef["server_flags"] if f != "--enable-grammar-constraints"]

    # Assertions with grammar + forced parser: tier, parser
    if suite_name == "assertions-grammar":
        if opts.get("tier"):
            sdef["cmd_template"] = [opts["tier"] if a == "full" and i > 0 and sdef["cmd_template"][i-1] == "--tier" else a
                                     for i, a in enumerate(sdef["cmd_template"])]
        if opts.get("forced_parser") and opts["forced_parser"] != "none":
            sdef["cmd_template"] = [opts["forced_parser"] if a == "qwen3_xml" and i > 0 and sdef["cmd_template"][i-1] == "--also-forced-parser" else a
                                     for i, a in enumerate(sdef["cmd_template"])]
        elif opts.get("forced_parser") == "none":
            # Remove --also-forced-parser entirely
            new_cmd = []
            skip_next = False
            for a in sdef["cmd_template"]:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--also-forced-parser":
                    skip_next = True
                    continue
                new_cmd.append(a)
            sdef["cmd_template"] = new_cmd

    # Smart analysis: judge, batch_mode, tests
    if suite_name == "smart-analysis":
        judge = opts.get("judge", "claude")
        batch = opts.get("batch_mode", "1")
        smart_val = f"{batch}:{judge}" if batch != "0" else judge
        sdef["cmd_template"] = [smart_val if a.startswith("1:") or a in ("claude", "codex") and i > 0 and sdef["cmd_template"][i-1] == "--smart" else a
                                 for i, a in enumerate(sdef["cmd_template"])]
        if opts.get("tests"):
            sdef["cmd_template"].extend(["--tests", str(opts["tests"])])

    # Promptfoo: mode
    if suite_name == "promptfoo":
        mode = opts.get("mode", "all")
        if mode != "all":
            sdef["cmd_template"] = [mode if a == "all" else a for a in sdef["cmd_template"]]

    # Batch tests: batch_sizes, concurrent, prefix_caching
    if suite_name.startswith("batch-"):
        if opts.get("concurrent"):
            if "server_flags" in sdef:
                new_flags = []
                skip_next = False
                for f in sdef["server_flags"]:
                    if skip_next:
                        skip_next = False
                        continue
                    if f == "--concurrent":
                        new_flags.extend(["--concurrent", str(opts["concurrent"])])
                        skip_next = True
                        continue
                    new_flags.append(f)
                sdef["server_flags"] = new_flags
        if opts.get("batch_sizes"):
            sdef["cmd_template"].extend(opts["batch_sizes"].split(","))
        if suite_name == "batch-multiturn" and opts.get("prefix_caching") is False:
            if "server_flags" in sdef:
                sdef["server_flags"] = [f for f in sdef["server_flags"] if f != "--enable-prefix-caching"]

    # GPU profile: max_tokens
    if suite_name == "gpu-profile" and opts.get("max_tokens"):
        sdef["cmd_template"].append(str(opts["max_tokens"]))

    return sdef


def _orchestrate_run(run_id: str, binary: str, model: str,
                     suite_names: list[str],
                     suite_options: Optional[dict] = None) -> None:
    """
    Main orchestration function. Groups suites by port, runs server-backed
    suites sequentially within each port group, runs non-server suites first.
    """
    global _active_run_id

    _open_log(run_id)

    _emit_sse({
        "type": "config",
        "run_id": run_id,
        "binary": binary,
        "model": model,
        "suites": suite_names,
    })

    total_passed = 0
    total_failed = 0
    results: list[dict] = []

    # Partition into: no-server, and port groups
    no_server: list[str] = []
    port_groups: dict[int, list[str]] = {}

    for name in suite_names:
        sdef = SUITES.get(name)
        if not sdef:
            continue
        if not sdef["needs_server"]:
            no_server.append(name)
        else:
            port = sdef["port"] or 0
            port_groups.setdefault(port, []).append(name)

    if suite_options is None:
        suite_options = {}

    # 1. Run non-server suites first
    for name in no_server:
        if _stop_requested.is_set():
            break
        sdef = _apply_suite_options(name, SUITES[name], suite_options)
        r = _run_suite(name, sdef, binary, model, None)
        results.append(r)
        total_passed += r["passed"]
        total_failed += r["failed"]

    # 2. Run port groups sequentially
    for port, names in sorted(port_groups.items()):
        if _stop_requested.is_set():
            break

        # Determine server flags — use the first suite's flags (after applying options)
        first_sdef = _apply_suite_options(names[0], SUITES[names[0]], suite_options)
        server_flags = first_sdef.get("server_flags", [])

        # Start server
        afm_proc = _start_afm_server(binary, model, port, server_flags)

        with _run_lock:
            _active_procs.append(afm_proc)

        try:
            # Check server is alive
            if afm_proc.poll() is not None:
                _emit_sse({
                    "type": "server_error",
                    "port": port,
                    "error": "Server exited prematurely",
                })
                continue

            for name in names:
                if _stop_requested.is_set():
                    break
                sdef = _apply_suite_options(name, SUITES[name], suite_options)
                r = _run_suite(name, sdef, binary, model, afm_proc)
                results.append(r)
                total_passed += r["passed"]
                total_failed += r["failed"]

        finally:
            _kill_proc(afm_proc)
            with _run_lock:
                if afm_proc in _active_procs:
                    _active_procs.remove(afm_proc)

    _emit_sse({
        "type": "done",
        "run_id": run_id,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "results": results,
    })

    _close_log()

    with _run_lock:
        _active_run_id = None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the test dashboard."""

    # Suppress default logging per request (we log ourselves)
    def log_message(self, fmt, *args):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"[{ts}] {fmt % args}\n")

    # ── Routing ──────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = dict(urllib.parse.parse_qsl(parsed.query))

        if path == "/" or path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif path == "/api/binary":
            self._json_response(_detect_binary())
        elif path == "/api/models":
            self._json_response(_list_models())
        elif path == "/api/preflight":
            binary = query.get("binary", "")
            if not binary:
                self._json_response({"error": "missing 'binary' parameter"}, 400)
                return
            self._json_response(_run_preflight(binary))
        elif path == "/api/events":
            self._handle_sse()
        elif path == "/api/suites":
            self._json_response(_get_suites_info())
        elif path == "/api/results":
            self._json_response(_list_results())
        elif path.startswith("/api/results/"):
            run_id = path.split("/api/results/", 1)[1]
            self._serve_result(run_id)
        elif path == "/api/afm/logs":
            self._handle_afm_logs()
        elif path.startswith("/api/reports/"):
            rel = path.split("/api/reports/", 1)[1]
            self._serve_report_file(rel)
        else:
            self._not_found()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/run":
            self._handle_run()
        elif path == "/api/stop":
            self._handle_stop()
        elif path == "/api/promptfoo-view":
            self._handle_promptfoo_view()
        elif path == "/api/afm/start":
            self._handle_afm_start()
        elif path == "/api/afm/stop":
            self._handle_afm_stop()
        else:
            self._not_found()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filename: str, content_type: str):
        filepath = os.path.join(SCRIPT_DIR, filename)
        if not os.path.isfile(filepath):
            self._not_found()
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_report_file(self, rel_path: str):
        """Serve a file from the test-reports/ directory."""
        report_dir = os.path.join(REPO_ROOT, "test-reports")
        # Prevent directory traversal
        safe = os.path.normpath(os.path.join(report_dir, rel_path))
        if not safe.startswith(report_dir):
            self._not_found()
            return
        if not os.path.isfile(safe):
            self._not_found()
            return

        ext = os.path.splitext(safe)[1].lower()
        ct_map = {
            ".html": "text/html",
            ".json": "application/json",
            ".jsonl": "application/jsonl",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".csv": "text/csv",
        }
        ct = ct_map.get(ext, "application/octet-stream")

        with open(safe, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not Found\n")

    # ── SSE ──────────────────────────────────────────────────────────────

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=1000)
        with _sse_lock:
            _sse_subscribers.append(q)

        try:
            # Send initial heartbeat
            self.wfile.write(b": heartbeat\n\n")
            self.wfile.flush()

            while True:
                try:
                    payload = q.get(timeout=15)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive comment
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            with _sse_lock:
                if q in _sse_subscribers:
                    _sse_subscribers.remove(q)

    # ── API handlers ─────────────────────────────────────────────────────

    def _handle_run(self):
        global _active_run_id

        with _run_lock:
            if _active_run_id is not None:
                self._json_response(
                    {"error": f"Run {_active_run_id} already active"}, 409
                )
                return

        body = self._read_body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        binary = data.get("binary", "")
        model = data.get("model", "")
        suite_names = data.get("suites", [])
        suite_options = data.get("options", {})  # per-suite options from UI

        if not binary or not model or not suite_names:
            self._json_response(
                {"error": "Required fields: binary, model, suites"}, 400
            )
            return

        # Validate suite names
        invalid = [s for s in suite_names if s not in SUITES]
        if invalid:
            self._json_response(
                {"error": f"Unknown suites: {invalid}"}, 400
            )
            return

        run_id = _make_run_id()

        with _run_lock:
            _active_run_id = run_id
            _stop_requested.clear()

        _emit_sse({
            "type": "trigger",
            "action": "user_clicked_run",
            "run_id": run_id,
            "binary": binary,
            "model": model,
            "suites": suite_names,
            "options": suite_options,
        })

        t = threading.Thread(
            target=_orchestrate_run,
            args=(run_id, binary, model, suite_names, suite_options),
            daemon=True,
        )
        t.start()
        with _run_lock:
            _active_threads.append(t)

        self._json_response({"run_id": run_id, "suites": suite_names})

    def _handle_stop(self):
        _emit_sse({"type": "trigger", "action": "user_clicked_stop"})
        _stop_requested.set()

        with _run_lock:
            for proc in list(_active_procs):
                _kill_proc(proc)
            _active_procs.clear()

        self._json_response({"status": "stopping"})

    def _handle_promptfoo_view(self):
        global _promptfoo_proc
        if _promptfoo_proc and _promptfoo_proc.poll() is None:
            self._json_response({"url": "http://localhost:15500", "status": "already_running"})
            return
        try:
            _promptfoo_proc = subprocess.Popen(
                ["promptfoo", "view", "-y"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=REPO_ROOT,
            )
            self._json_response({"url": "http://localhost:15500", "status": "started"})
        except FileNotFoundError:
            self._json_response({"error": "promptfoo not found in PATH"}, 500)

    # ── AFM Server Management ────────────────────────────────────────────

    def _handle_afm_start(self):
        """Start AFM server with user-specified options."""
        global _user_afm_proc, _user_afm_log

        if _user_afm_proc and _user_afm_proc.poll() is None:
            self._json_response({"error": "AFM server already running", "pid": _user_afm_proc.pid}, 409)
            return

        body = self._read_body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        binary = data.get("binary", "")
        model = data.get("model", "")
        opts = data.get("opts", {})

        if not binary or not model:
            self._json_response({"error": "Required: binary, model"}, 400)
            return

        # Build command from options
        cmd = [binary, "mlx", "-m", model]
        cmd.extend(["--port", str(opts.get("port", 9998))])

        hostname = opts.get("hostname", "127.0.0.1")
        if hostname != "127.0.0.1":
            cmd.extend(["-H", hostname])

        max_tokens = opts.get("maxTokens")
        if max_tokens and max_tokens != 8192:
            cmd.extend(["--max-tokens", str(max_tokens)])

        temp = opts.get("temperature")
        if temp is not None and temp != "":
            cmd.extend(["-t", str(temp)])

        top_p = opts.get("topP")
        if top_p is not None and top_p != "":
            cmd.extend(["--top-p", str(top_p)])

        top_k = opts.get("topK")
        if top_k and int(top_k) > 0:
            cmd.extend(["--top-k", str(top_k)])

        min_p = opts.get("minP")
        if min_p and float(min_p) > 0:
            cmd.extend(["--min-p", str(min_p)])

        concurrent = opts.get("concurrent")
        if concurrent and int(concurrent) > 1:
            cmd.extend(["--concurrent", str(concurrent)])

        seed = opts.get("seed")
        if seed is not None and seed != "":
            cmd.extend(["--seed", str(seed)])

        presence = opts.get("presencePenalty")
        if presence and float(presence) > 0:
            cmd.extend(["--presence-penalty", str(presence)])

        rep = opts.get("repetitionPenalty")
        if rep and float(rep) > 0:
            cmd.extend(["--repetition-penalty", str(rep)])

        max_kv = opts.get("maxKvSize")
        if max_kv:
            cmd.extend(["--max-kv-size", str(max_kv)])

        kv_bits = opts.get("kvBits")
        if kv_bits:
            cmd.extend(["--kv-bits", str(kv_bits)])

        prefill = opts.get("prefillStepSize")
        if prefill:
            cmd.extend(["--prefill-step-size", str(prefill)])

        parser = opts.get("toolCallParser")
        if parser:
            cmd.extend(["--tool-call-parser", parser])

        stop = opts.get("stop")
        if stop:
            cmd.extend(["--stop", stop])

        if opts.get("enablePrefixCaching"):
            cmd.append("--enable-prefix-caching")
        if opts.get("enableGrammarConstraints"):
            cmd.append("--enable-grammar-constraints")
        if opts.get("noThink"):
            cmd.append("--no-think")
        if opts.get("noStreaming"):
            cmd.append("--no-streaming")
        if opts.get("raw"):
            cmd.append("--raw")
        if opts.get("vlm"):
            cmd.append("--vlm")
        if opts.get("webui"):
            cmd.append("--webui")
        if opts.get("verbose"):
            cmd.append("-v")
        if opts.get("gpuProfile"):
            cmd.append("--gpu-profile")

        # Start the server
        env = os.environ.copy()
        env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE

        log_path = os.path.join(LOG_DIR, "afm-server.log")
        _user_afm_log = open(log_path, "w")

        try:
            _user_afm_proc = subprocess.Popen(
                cmd,
                stdout=_user_afm_log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=REPO_ROOT,
            )
            _emit_sse({
                "type": "trigger",
                "action": "user_started_afm_server",
                "command": " ".join(cmd),
                "pid": _user_afm_proc.pid,
            })

            # Wait for server readiness (uses OpenAI SDK if available)
            port = opts.get("port", 9998)
            healthy = _wait_for_server(port, timeout=120, poll=1.0, proc=_user_afm_proc)

            if healthy:
                self._json_response({
                    "pid": _user_afm_proc.pid,
                    "port": port,
                    "command": " ".join(cmd),
                    "status": "running",
                })
            else:
                _kill_proc(_user_afm_proc)
                self._json_response({"error": "Server failed to start (health check timeout)"}, 500)

        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_afm_stop(self):
        """Stop user-started AFM server."""
        global _user_afm_proc, _user_afm_log
        if _user_afm_proc and _user_afm_proc.poll() is None:
            _kill_proc(_user_afm_proc)
            _emit_sse({"type": "trigger", "action": "user_stopped_afm_server"})
        _user_afm_proc = None
        if _user_afm_log:
            _user_afm_log.close()
            _user_afm_log = None
        self._json_response({"status": "stopped"})

    def _handle_afm_logs(self):
        """Return recent AFM server log lines."""
        log_path = os.path.join(LOG_DIR, "afm-server.log")
        lines = []
        if os.path.isfile(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()[-500:]  # last 500 lines
        running = _user_afm_proc is not None and _user_afm_proc.poll() is None
        self._json_response({"lines": [l.rstrip() for l in lines], "running": running})

    # ── Results / Suites ─────────────────────────────────────────────────

    def _serve_result(self, run_id: str):
        """Return the full JSONL content for a specific run."""
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", run_id)
        path = os.path.join(LOG_DIR, f"{safe_id}.jsonl")
        if not os.path.isfile(path):
            self._not_found()
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/jsonl")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def _get_suites_info() -> list[dict]:
    """Return suite metadata for the UI."""
    out = []
    for name, sdef in SUITES.items():
        out.append({
            "id": name,
            "label": sdef["label"],
            "needs_server": sdef["needs_server"],
            "port": sdef.get("port"),
            "est_minutes": sdef.get("est_minutes", 0),
        })
    return out


def _list_results() -> list[dict]:
    """List past runs from dashboard-logs/."""
    if not os.path.isdir(LOG_DIR):
        return []

    runs = []
    for fname in sorted(os.listdir(LOG_DIR), reverse=True):
        if not fname.endswith(".jsonl") or fname == "LATEST.jsonl":
            continue
        run_id = fname.replace(".jsonl", "")
        fpath = os.path.join(LOG_DIR, fname)

        # Quick parse for summary
        suites: list[str] = []
        total_passed = 0
        total_failed = 0
        date_str = ""

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    etype = ev.get("type", "")
                    if etype == "config":
                        suites = ev.get("suites", [])
                        date_str = ev.get("timestamp", "")
                    elif etype == "done":
                        total_passed = ev.get("total_passed", 0)
                        total_failed = ev.get("total_failed", 0)
        except Exception:
            continue

        runs.append({
            "id": run_id,
            "date": date_str,
            "suites": suites,
            "passed": total_passed,
            "failed": total_failed,
        })

    return runs


# ---------------------------------------------------------------------------
# CORS preflight support
# ---------------------------------------------------------------------------

class CORSHandler(DashboardHandler):
    """Extends DashboardHandler with CORS OPTIONS support."""

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum, frame):
    print(f"\n[dashboard] Received signal {signum}, shutting down...")
    _stop_requested.set()

    # Kill all subprocesses
    with _run_lock:
        for proc in list(_active_procs):
            _kill_proc(proc)
        _active_procs.clear()

    if _promptfoo_proc and _promptfoo_proc.poll() is None:
        _kill_proc(_promptfoo_proc)

    if _user_afm_proc and _user_afm_proc.poll() is None:
        _kill_proc(_user_afm_proc)

    _close_log()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global REPO_ROOT, SCRIPT_DIR, LOG_DIR

    parser = argparse.ArgumentParser(description="AFM Test Dashboard Server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to serve dashboard on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser on start")
    args = parser.parse_args()

    # Detect repo root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        REPO_ROOT = result.stdout.strip()
    except Exception:
        REPO_ROOT = os.getcwd()

    if not REPO_ROOT or not os.path.isdir(REPO_ROOT):
        print("Error: Could not detect repository root.", file=sys.stderr)
        sys.exit(1)

    SCRIPT_DIR = os.path.join(REPO_ROOT, "Scripts", "test-dashboard")
    LOG_DIR = os.path.join(REPO_ROOT, "test-reports", "dashboard-logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Change to repo root so relative paths in suite commands work
    os.chdir(REPO_ROOT)

    # Install signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    server = HTTPServer(("0.0.0.0", args.port), CORSHandler)
    server.daemon_threads = True

    url = f"http://localhost:{args.port}"
    print(f"[dashboard] Serving at {url}")
    print(f"[dashboard] Repo root: {REPO_ROOT}")
    print(f"[dashboard] Log dir:   {LOG_DIR}")
    print(f"[dashboard] Press Ctrl+C to stop")

    if not args.no_browser:
        # Open browser after a short delay to let the server start
        def _open():
            time.sleep(0.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _close_log()
        print("[dashboard] Server stopped.")


if __name__ == "__main__":
    main()
