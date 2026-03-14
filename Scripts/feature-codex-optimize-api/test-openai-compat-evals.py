#!/usr/bin/env python3
"""
Reusable OpenAI-compat eval bundle for local AFM servers.

Runs a compact compatibility matrix against a local OpenAI-style endpoint:
  - GET /v1/models
  - openai-python non-stream chat completion
  - openai-python non-stream chat completion with logprobs
  - openai-python streaming chat completion with usage chunk
  - openai-python streaming chat completion with logprobs
  - vllm bench serve smoke benchmark

Usage examples:
  python3 Scripts/feature-codex-optimize-api/test-openai-compat-evals.py --base-url http://127.0.0.1:9999/v1
  python3 Scripts/feature-codex-optimize-api/test-openai-compat-evals.py --start-server --model mlx-community/Qwen3.5-35B-A3B-4bit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AFM_BINARY = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "afm"
REPORT_DIR = SCRIPT_DIR / "results"
MODEL_CACHE = os.environ.get("MACAFM_MLX_MODEL_CACHE", "/Volumes/edata/models/vesta-test-cache")
DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
DEFAULT_TOKENIZER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def run_subprocess(cmd: list[str], timeout: int = 120, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def wait_for_server(base_url: str, timeout: int = 180, proc: subprocess.Popen[str] | None = None) -> bool:
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(models_url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def start_server(model: str, port: int, extra_args: list[str]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    cmd = [
        str(AFM_BINARY),
        "mlx",
        "-m",
        model,
        "--port",
        str(port),
        *extra_args,
    ]
    log(f"Starting server: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop_server(proc: subprocess.Popen[str] | None) -> None:
    if not proc or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_models_probe(base_url: str) -> dict:
    started = time.time()
    body = http_get_json(f"{base_url.rstrip('/')}/models")
    data = body.get("data", [])
    return {
        "name": "models_endpoint",
        "ok": bool(data),
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "count": len(data),
            "first_id": data[0].get("id") if data else None,
        },
    }


def run_openai_python_nonstream(base_url: str, model: str) -> dict:
    code = f"""
from openai import OpenAI
client = OpenAI(base_url={base_url!r}, api_key='test')
resp = client.chat.completions.create(
    model={model!r},
    messages=[{{'role':'user','content':'Reply with exactly: ok'}}],
    max_tokens=8,
    temperature=0,
)
print(resp.model_dump_json())
"""
    started = time.time()
    result = run_subprocess(["python3", "-c", code], timeout=120)
    ok = result.returncode == 0
    payload = json.loads(result.stdout) if ok and result.stdout.strip() else {}
    return {
        "name": "openai_python_nonstream",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "finish_reason": (((payload.get("choices") or [{}])[0]).get("finish_reason")) if payload else None,
            "content": ((((payload.get("choices") or [{}])[0]).get("message") or {}).get("content")) if payload else None,
            "usage": payload.get("usage"),
            "stderr": result.stderr[-400:] if not ok else None,
        },
    }


def run_openai_python_stream(base_url: str, model: str) -> dict:
    code = f"""
import json
from openai import OpenAI
client = OpenAI(base_url={base_url!r}, api_key='test')
stream = client.chat.completions.create(
    model={model!r},
    messages=[{{'role':'user','content':'Reply with exactly: ok'}}],
    max_tokens=8,
    temperature=0,
    stream=True,
    stream_options={{'include_usage': True}},
)
chunks = list(stream)
content = ''.join((c.choices[0].delta.content or '') for c in chunks if c.choices)
finish = next((c.choices[0].finish_reason for c in reversed(chunks) if c.choices and c.choices[0].finish_reason), None)
last = chunks[-1]
print(json.dumps({{
    'chunk_count': len(chunks),
    'last_choices': len(last.choices),
    'last_usage': last.usage.model_dump() if last.usage else None,
    'content': content,
    'finish_reason': finish,
}}))
"""
    started = time.time()
    result = run_subprocess(["python3", "-c", code], timeout=120)
    ok = result.returncode == 0
    payload = json.loads(result.stdout) if ok and result.stdout.strip() else {}
    return {
        "name": "openai_python_stream",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "chunk_count": payload.get("chunk_count"),
            "last_choices": payload.get("last_choices"),
            "last_usage": payload.get("last_usage"),
            "content": payload.get("content"),
            "finish_reason": payload.get("finish_reason"),
            "stderr": result.stderr[-400:] if not ok else None,
        },
    }


def run_openai_python_nonstream_logprobs(base_url: str, model: str) -> dict:
    code = f"""
import json
from openai import OpenAI
client = OpenAI(base_url={base_url!r}, api_key='test')
resp = client.chat.completions.create(
    model={model!r},
    messages=[{{'role':'user','content':'Reply with exactly: ok'}}],
    max_tokens=8,
    temperature=0,
    logprobs=True,
    top_logprobs=3,
)
choice = resp.choices[0]
content = choice.message.content
lp = choice.logprobs
first = (lp.content[0].model_dump() if lp and lp.content else None)
print(json.dumps({{
    'content': content,
    'finish_reason': choice.finish_reason,
    'logprobs_present': lp is not None,
    'token_count': len(lp.content) if lp and lp.content else 0,
    'first_token': first,
}}))
"""
    started = time.time()
    result = run_subprocess(["python3", "-c", code], timeout=120)
    ok = result.returncode == 0
    payload = json.loads(result.stdout) if ok and result.stdout.strip() else {}
    return {
        "name": "openai_python_nonstream_logprobs",
        "ok": ok and payload.get("logprobs_present") and payload.get("token_count", 0) > 0,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "finish_reason": payload.get("finish_reason"),
            "content": payload.get("content"),
            "logprobs_present": payload.get("logprobs_present"),
            "token_count": payload.get("token_count"),
            "first_token": payload.get("first_token"),
            "stderr": result.stderr[-400:] if not ok else None,
        },
    }


def run_openai_python_stream_logprobs(base_url: str, model: str) -> dict:
    code = f"""
import json
from openai import OpenAI
client = OpenAI(base_url={base_url!r}, api_key='test')
stream = client.chat.completions.create(
    model={model!r},
    messages=[{{'role':'user','content':'Reply with exactly: ok'}}],
    max_tokens=8,
    temperature=0,
    stream=True,
    logprobs=True,
    top_logprobs=3,
    stream_options={{'include_usage': True}},
)
chunks = list(stream)
content = ''.join((c.choices[0].delta.content or '') for c in chunks if c.choices)
finish = next((c.choices[0].finish_reason for c in reversed(chunks) if c.choices and c.choices[0].finish_reason), None)
logprob_chunks = [c for c in chunks if c.choices and c.choices[0].logprobs and c.choices[0].logprobs.content]
first = logprob_chunks[0].choices[0].logprobs.content[0].model_dump() if logprob_chunks else None
last = chunks[-1]
print(json.dumps({{
    'content': content,
    'finish_reason': finish,
    'logprob_chunk_count': len(logprob_chunks),
    'first_logprob_token': first,
    'last_choices': len(last.choices),
    'last_usage': last.usage.model_dump() if last.usage else None,
}}))
"""
    started = time.time()
    result = run_subprocess(["python3", "-c", code], timeout=120)
    ok = result.returncode == 0
    payload = json.loads(result.stdout) if ok and result.stdout.strip() else {}
    return {
        "name": "openai_python_stream_logprobs",
        "ok": ok and payload.get("logprob_chunk_count", 0) > 0 and payload.get("last_choices") == 0,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "finish_reason": payload.get("finish_reason"),
            "content": payload.get("content"),
            "logprob_chunk_count": payload.get("logprob_chunk_count"),
            "first_logprob_token": payload.get("first_logprob_token"),
            "last_choices": payload.get("last_choices"),
            "last_usage": payload.get("last_usage"),
            "stderr": result.stderr[-400:] if not ok else None,
        },
    }


def run_vllm_bench(base_root_url: str, model: str, tokenizer: str) -> dict:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--base-url",
        base_root_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--random-input-len",
        "64",
        "--random-output-len",
        "32",
        "--num-prompts",
        "4",
    ]
    started = time.time()
    result = run_subprocess(cmd, timeout=240)
    output = f"{result.stdout}\n{result.stderr}"
    lines = [line.strip() for line in output.splitlines()]

    def extract_metric(prefix: str) -> str | None:
        for line in lines:
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return None

    ok = result.returncode == 0 and extract_metric("Failed requests") == "0"
    return {
        "name": "vllm_bench_smoke",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "successful_requests": extract_metric("Successful requests"),
            "failed_requests": extract_metric("Failed requests"),
            "total_generated_tokens": extract_metric("Total generated tokens"),
            "output_token_throughput": extract_metric("Output token throughput (tok/s)"),
            "stderr_tail": result.stderr[-400:] if result.stderr else None,
        },
    }


def write_report(results: list[dict], report_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generated_at": dt.datetime.now().isoformat(), "results": results}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reusable OpenAI-compat evals against AFM or another local OpenAI-style server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9999/v1", help="Base API URL including /v1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id to send in chat completion requests")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer repo for vllm bench")
    parser.add_argument("--start-server", action="store_true", help="Start afm locally instead of reusing an existing server")
    parser.add_argument("--port", type=int, default=9999, help="Port to bind when using --start-server")
    parser.add_argument("--server-arg", action="append", default=[], help="Extra arg to pass through when using --start-server")
    parser.add_argument("--report-name", default=None, help="Optional JSON report filename")
    args = parser.parse_args()

    server_proc = None
    results: list[dict] = []

    try:
        if args.start_server:
            server_proc = start_server(args.model, args.port, args.server_arg)
            if not wait_for_server(args.base_url, proc=server_proc):
                log("Server failed to become ready")
                return 1

        results.append(run_models_probe(args.base_url))
        results.append(run_openai_python_nonstream(args.base_url, args.model))
        results.append(run_openai_python_stream(args.base_url, args.model))
        results.append(run_openai_python_nonstream_logprobs(args.base_url, args.model))
        results.append(run_openai_python_stream_logprobs(args.base_url, args.model))
        results.append(run_vllm_bench(args.base_url[:-3] if args.base_url.endswith("/v1") else args.base_url, args.model, args.tokenizer))

        ok = all(item["ok"] for item in results)
        report_name = args.report_name or f"openai-compat-evals-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        report_path = REPORT_DIR / report_name
        write_report(results, report_path)

        for item in results:
            status = "PASS" if item["ok"] else "FAIL"
            log(f"{status} {item['name']} ({item['elapsed_s']}s)")
        log(f"Report: {report_path}")
        return 0 if ok else 1
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    sys.exit(main())
