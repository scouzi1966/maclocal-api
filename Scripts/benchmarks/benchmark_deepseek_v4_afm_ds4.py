#!/usr/bin/env python3
"""Release benchmark for AFM MLX and canonical antirez/ds4 servers.

The runners execute sequentially so two large DeepSeek models never compete for
unified memory. Raw responses and server logs are retained with the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPT = (
    "Count upward from 1, separated only by commas. Continue until stopped."
)
STATS_RE = re.compile(
    r"\[STATS\].*?tg:\s*(\d+)\s+tok,\s*([0-9.]+)s\s*\(([0-9.]+)\s+tok/s\)"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare Release AFM MLX with canonical antirez/ds4."
    )
    parser.add_argument("--afm-model", type=Path)
    parser.add_argument("--ds4-model", type=Path)
    parser.add_argument(
        "--afm-binary",
        type=Path,
        default=root / ".build/arm64-apple-macosx/release/afm",
    )
    parser.add_argument(
        "--ds4-binary", type=Path, default=root / "vendor/ds4/ds4-server"
    )
    parser.add_argument("--afm-kernels", default="native")
    parser.add_argument(
        "--afm-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional AFM environment entry; may be repeated.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--afm-port", type=int, default=19997)
    parser.add_argument("--ds4-port", type=int, default=19998)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Research/benchmarks/deepseek-v4-afm-ds4",
    )
    args = parser.parse_args()
    if not args.afm_model and not args.ds4_model:
        parser.error("provide --afm-model, --ds4-model, or both")
    if args.runs < 1 or args.tokens < 1 or args.warmup_tokens < 0:
        parser.error("runs and tokens must be positive; warmup tokens cannot be negative")
    return args


def require_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} not found: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise RuntimeError(f"{label} is not executable: {resolved}")
    return resolved


def require_model(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"{label} not found: {resolved}")
    return resolved


def parse_environment(entries: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key:
            raise RuntimeError(f"invalid --afm-env value: {entry!r}")
        result[key] = value
    return result


def http_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=900) as response:
            return json.loads(response.read())
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body[:500]}") from error


def wait_until_ready(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    # Both servers expose this OpenAI endpoint; canonical ds4 has no /health.
    health_url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            http_json(health_url)
            return
        except (URLError, RuntimeError, json.JSONDecodeError):
            time.sleep(0.5)
    raise RuntimeError(f"server did not become ready within {timeout:.0f}s")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def completion_request(port: int, prompt: str, tokens: int) -> tuple[dict[str, Any], float]:
    payload = {
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    response = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    return response, time.perf_counter() - started


def response_record(response: dict[str, Any], elapsed: float) -> dict[str, Any]:
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
    return {
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "wall_tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0,
        "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def run_backend(
    name: str,
    command: list[str],
    environment: dict[str, str],
    port: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    backend_dir = output_dir / name
    backend_dir.mkdir(parents=True, exist_ok=True)
    log_path = backend_dir / "server.log"
    print(f"[{name}] starting: {' '.join(command)}", flush=True)
    startup_started = time.perf_counter()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            wait_until_ready(process, port, args.startup_timeout)
            startup_seconds = time.perf_counter() - startup_started
            warmup_seconds = 0.0
            if args.warmup_tokens:
                _, warmup_seconds = completion_request(
                    port, "Reply with the word ready.", args.warmup_tokens
                )
            records = []
            for run_index in range(1, args.runs + 1):
                response, elapsed = completion_request(port, args.prompt, args.tokens)
                response_path = backend_dir / f"run-{run_index}.json"
                response_path.write_text(json.dumps(response, indent=2) + "\n")
                record = response_record(response, elapsed)
                record["run"] = run_index
                records.append(record)
                print(
                    f"[{name}] run {run_index}: "
                    f"{record['wall_tokens_per_second']:.2f} tok/s "
                    f"sha={record['content_sha256'][:12]}",
                    flush=True,
                )
        finally:
            stop_process(process)

    log_text = log_path.read_text(errors="replace")
    server_stats = [
        {
            "completion_tokens": int(match.group(1)),
            "seconds": float(match.group(2)),
            "tokens_per_second": float(match.group(3)),
        }
        for match in STATS_RE.finditer(log_text)
    ]
    measured = [record["wall_tokens_per_second"] for record in records]
    return {
        "name": name,
        "command": command,
        "startup_seconds": startup_seconds,
        "warmup_seconds": warmup_seconds,
        "runs": records,
        "average_wall_tokens_per_second": sum(measured) / len(measured),
        "server_stats": server_stats,
    }


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True)
    results: dict[str, Any] = {
        "created_at": timestamp,
        "prompt": args.prompt,
        "runs": args.runs,
        "tokens": args.tokens,
        "warmup_tokens": args.warmup_tokens,
        "context": args.context,
        "backends": [],
    }

    if args.afm_model:
        afm_binary = require_file(args.afm_binary, "AFM Release binary", executable=True)
        if "/release/" not in str(afm_binary):
            raise RuntimeError(f"refusing non-Release AFM binary: {afm_binary}")
        afm_model = require_model(args.afm_model, "AFM model")
        environment = os.environ.copy()
        environment.update(parse_environment(args.afm_env))
        command = [
            str(afm_binary), "mlx", "-m", str(afm_model),
            "-p", str(args.afm_port), "--no-think", "-t", "0",
            "--mlx-kernels", args.afm_kernels,
        ]
        results["backends"].append(
            run_backend("afm", command, environment, args.afm_port, args, output_dir)
        )

    if args.ds4_model:
        ds4_binary = require_file(args.ds4_binary, "canonical ds4-server", executable=True)
        ds4_model = require_file(args.ds4_model, "DS4 GGUF")
        command = [
            str(ds4_binary), "-m", str(ds4_model), "--metal",
            "--host", "127.0.0.1", "--port", str(args.ds4_port),
            "--ctx", str(args.context), "--tokens", str(args.tokens),
        ]
        results["backends"].append(
            run_backend("ds4", command, os.environ.copy(), args.ds4_port, args, output_dir)
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
