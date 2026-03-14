#!/usr/bin/env python3
"""
Guided JSON / structured output eval bundle.

Focus:
  - AFM `--guided-json` parity with OpenAI-style strict `json_schema`
  - openai-python structured output behavior
  - real-world schema fixtures instead of only toy objects
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
from typing import Any

import jsonschema
from openai import OpenAI
from pydantic import BaseModel


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AFM_BINARY = REPO_ROOT / ".build" / "arm64-apple-macosx" / "release" / "afm"
CASES_PATH = SCRIPT_DIR / "guided-json-cases.json"
REPORT_DIR = SCRIPT_DIR / "results"
MODEL_CACHE = os.environ.get("MACAFM_MLX_MODEL_CACHE", str(Path.home() / ".cache" / "macafm" / "models"))
DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"


class PersonRecord(BaseModel):
    name: str
    age: int
    occupation: str


def ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def run_subprocess(cmd: list[str], timeout: int = 240, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        shell=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_response_format(case: dict) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": case["schema_name"],
            "strict": True,
            "schema": case["schema"],
        },
    }


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


def start_server(model: str, port: int, extra_args: list[str], mode: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    if mode == "foundation":
        cmd = [
            str(AFM_BINARY),
            "--port",
            str(port),
            *extra_args,
        ]
    else:
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
        shell=False,
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


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


def validate_instance(schema: dict, instance: dict) -> tuple[bool, str | None]:
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return True, None
    except jsonschema.ValidationError as exc:
        return False, exc.message


def failed_result(name: str, started: float, exc: Exception) -> dict:
    return {
        "name": name,
        "ok": False,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    }


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


def run_api_schema_case(client: OpenAI, model: str, case: dict) -> dict:
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["prompt"]}],
            max_tokens=400,
            temperature=0,
            response_format=build_response_format(case),
        )
    except Exception as exc:
        return failed_result(f"api_json_schema::{case['id']}", started, exc)
    content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "name": f"api_json_schema::{case['id']}",
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "details": {"error": f"invalid JSON: {exc}", "content": content},
        }

    valid, err = validate_instance(case["schema"], parsed)
    return {
        "name": f"api_json_schema::{case['id']}",
        "ok": valid,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "finish_reason": response.choices[0].finish_reason,
            "content": content,
            "parsed": parsed,
            "usage": response.usage.model_dump() if response.usage else None,
            "validation_error": err,
        },
    }


def run_api_stream_schema_case(client: OpenAI, model: str, case: dict) -> dict:
    started = time.time()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["prompt"]}],
            max_tokens=400,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            response_format=build_response_format(case),
        )
    except Exception as exc:
        return failed_result(f"api_stream_json_schema::{case['id']}", started, exc)

    parts: list[str] = []
    usage = None
    final_finish_reason = None
    saw_usage_chunk = False
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
            saw_usage_chunk = len(chunk.choices) == 0
        for choice in chunk.choices:
            if choice.delta.content:
                parts.append(choice.delta.content)
            if choice.finish_reason:
                final_finish_reason = choice.finish_reason

    content = "".join(parts)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "name": f"api_stream_json_schema::{case['id']}",
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "details": {
                "error": f"invalid JSON: {exc}",
                "content": content,
                "chunk_count": chunk_count,
                "saw_usage_chunk": saw_usage_chunk,
                "usage": usage,
                "finish_reason": final_finish_reason,
            },
        }

    valid, err = validate_instance(case["schema"], parsed)
    return {
        "name": f"api_stream_json_schema::{case['id']}",
        "ok": valid and err is None and saw_usage_chunk and usage is not None,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "content": content,
            "parsed": parsed,
            "validation_error": err,
            "chunk_count": chunk_count,
            "saw_usage_chunk": saw_usage_chunk,
            "usage": usage,
            "finish_reason": final_finish_reason,
        },
    }


def run_api_conflict_schema_case(client: OpenAI, model: str, case: dict) -> dict:
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["prompt"]}],
            max_tokens=200,
            temperature=0,
            response_format=build_response_format(case),
        )
    except Exception as exc:
        return failed_result(f"api_conflict_json_schema::{case['id']}", started, exc)
    message = response.choices[0].message
    content = message.content or ""
    refusal = getattr(message, "refusal", None)

    parsed = None
    validation_error = None
    ok = False
    if refusal:
        ok = True
    else:
        try:
            parsed = json.loads(content)
            valid, validation_error = validate_instance(case["schema"], parsed)
            ok = valid
        except json.JSONDecodeError as exc:
            validation_error = str(exc)

    return {
        "name": f"api_conflict_json_schema::{case['id']}",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "content": content,
            "parsed": parsed,
            "refusal": refusal,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
            "validation_error": validation_error,
        },
    }


def run_api_truncation_schema_case(client: OpenAI, model: str, case: dict) -> dict:
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["prompt"]}],
            max_tokens=case.get("max_tokens", 24),
            temperature=0,
            response_format=build_response_format(case),
        )
    except Exception as exc:
        return failed_result(f"api_truncation_json_schema::{case['id']}", started, exc)
    content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason
    refusal = getattr(response.choices[0].message, "refusal", None)

    parsed = None
    validation_error = None
    valid = False
    try:
        parsed = json.loads(content)
        valid, validation_error = validate_instance(case["schema"], parsed)
    except json.JSONDecodeError as exc:
        validation_error = str(exc)

    completion_tokens = ((response.usage.model_dump() if response.usage else {}) or {}).get("completion_tokens")
    hit_budget = completion_tokens == case.get("max_tokens", 24)
    ok = ((hit_budget or finish_reason == "length") and not valid) or refusal is not None
    return {
        "name": f"api_truncation_json_schema::{case['id']}",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "finish_reason": finish_reason,
            "hit_budget": hit_budget,
            "content": content,
            "parsed": parsed,
            "validation_error": validation_error,
            "refusal": refusal,
            "usage": response.usage.model_dump() if response.usage else None,
        },
    }


def run_api_unsupported_schema_case(client: OpenAI, model: str, case: dict) -> dict:
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["prompt"]}],
            max_tokens=120,
            temperature=0,
            response_format=build_response_format(case),
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        valid, validation_error = validate_instance(case["schema"], parsed)
        return {
            "name": f"api_unsupported_json_schema::{case['id']}",
            "ok": valid,
            "elapsed_s": round(time.time() - started, 3),
            "details": {
                "mode": "accepted",
                "content": content,
                "parsed": parsed,
                "validation_error": validation_error,
                "finish_reason": response.choices[0].finish_reason,
            },
        }
    except Exception as exc:
        return {
            "name": f"api_unsupported_json_schema::{case['id']}",
            "ok": True,
            "elapsed_s": round(time.time() - started, 3),
            "details": {"mode": "rejected", "error_type": type(exc).__name__, "message": str(exc)},
        }


def run_sdk_parse_case(client: OpenAI, model: str, case: dict) -> dict:
    started = time.time()
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": case["prompt"]}],
            temperature=0,
            max_tokens=128,
            response_format=PersonRecord,
        )
    except Exception as exc:
        return failed_result(f"openai_python_parse::{case['id']}", started, exc)
    message = response.choices[0].message
    ok = message.parsed is not None and message.refusal is None
    return {
        "name": f"openai_python_parse::{case['id']}",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "content": message.content,
            "parsed": message.parsed.model_dump() if message.parsed else None,
            "refusal": message.refusal,
        },
    }


def run_cli_guided_json_case(model: str, case: dict, *, no_think: bool = False) -> dict:
    started = time.time()
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    mode = "no_think" if no_think else "default"
    cmd = [
        str(AFM_BINARY),
        "mlx",
        "-m",
        model,
        "-s",
        case["prompt"],
        "--max-tokens",
        "400",
        "--temperature",
        "0",
        "--guided-json",
        json.dumps(case["schema"], separators=(",", ":")),
    ]
    if no_think:
        cmd.append("--no-think")
    result = run_subprocess(cmd, timeout=240, env=env)
    output = result.stdout.strip()
    if result.returncode != 0:
        return {
            "name": f"cli_guided_json::{mode}::{case['id']}",
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "details": {"stderr": result.stderr[-500:], "stdout": output[-500:]},
        }
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return {
            "name": f"cli_guided_json::{mode}::{case['id']}",
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "details": {"error": f"invalid JSON: {exc}", "stdout": output},
        }
    valid, err = validate_instance(case["schema"], parsed)
    return {
        "name": f"cli_guided_json::{mode}::{case['id']}",
        "ok": valid,
        "elapsed_s": round(time.time() - started, 3),
        "details": {"parsed": parsed, "validation_error": err},
    }


def run_cli_invalid_schema(model: str) -> dict:
    started = time.time()
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    cmd = [
        str(AFM_BINARY),
        "mlx",
        "-m",
        model,
        "-s",
        "Hello",
        "--guided-json",
        "not valid json",
    ]
    result = run_subprocess(cmd, timeout=120, env=env)
    stderr_tail = result.stderr[-400:]
    error_indicators = (
        "guided-json",
        "guided json",
        "invalid json",
        "json",
        "schema",
    )
    matched_indicator = next(
        (marker for marker in error_indicators if marker in stderr_tail.lower()),
        None,
    )
    ok = result.returncode != 0 and matched_indicator is not None
    return {
        "name": "cli_guided_json::invalid_schema_error",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "returncode": result.returncode,
            "stderr": stderr_tail,
            "error_indicator": matched_indicator,
        },
    }


def run_cli_invalid_schema_no_think(model: str) -> dict:
    started = time.time()
    env = os.environ.copy()
    env["MACAFM_MLX_MODEL_CACHE"] = MODEL_CACHE
    cmd = [
        str(AFM_BINARY),
        "mlx",
        "-m",
        model,
        "-s",
        "Hello",
        "--no-think",
        "--guided-json",
        "not valid json",
    ]
    result = run_subprocess(cmd, timeout=120, env=env)
    stderr_tail = result.stderr[-400:]
    error_indicators = (
        "guided-json",
        "guided json",
        "invalid json",
        "json",
        "schema",
    )
    matched_indicator = next(
        (marker for marker in error_indicators if marker in stderr_tail.lower()),
        None,
    )
    ok = result.returncode != 0 and matched_indicator is not None
    return {
        "name": "cli_guided_json::no_think::invalid_schema_error",
        "ok": ok,
        "elapsed_s": round(time.time() - started, 3),
        "details": {
            "returncode": result.returncode,
            "stderr": stderr_tail,
            "error_indicator": matched_indicator,
        },
    }


def write_report(results: list[dict], report_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generated_at": dt.datetime.now().isoformat(), "results": results}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guided-json / structured output evals against AFM.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9999/v1", help="Base API URL including /v1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id")
    parser.add_argument("--start-server", action="store_true", help="Start afm locally instead of using an existing server")
    parser.add_argument("--port", type=int, default=9999, help="Port to bind when using --start-server")
    parser.add_argument("--server-mode", choices=["mlx", "foundation"], default="mlx", help="Server mode to launch when using --start-server")
    parser.add_argument("--server-arg", action="append", default=[], help="Extra arg to pass through when using --start-server")
    parser.add_argument("--run-cli", action="store_true", help="Also run expensive single-prompt CLI --guided-json checks")
    parser.add_argument("--run-cli-no-think", action="store_true", help="Also run CLI --guided-json checks with --no-think")
    parser.add_argument("--report-name", default=None, help="Optional JSON report filename")
    args = parser.parse_args()

    server_proc = None
    results: list[dict] = []

    try:
        if args.start_server:
            server_proc = start_server(args.model, args.port, args.server_arg, args.server_mode)
            if not wait_for_server(args.base_url, proc=server_proc):
                log("Server failed to become ready")
                return 1

        client = OpenAI(base_url=args.base_url, api_key="test")
        results.append(run_models_probe(args.base_url))

        for case in load_cases():
            if case["kind"] == "schema":
                results.append(run_api_schema_case(client, args.model, case))
                if args.run_cli:
                    results.append(run_cli_guided_json_case(args.model, case))
                if args.run_cli_no_think:
                    results.append(run_cli_guided_json_case(args.model, case, no_think=True))
            elif case["kind"] == "stream_schema":
                results.append(run_api_stream_schema_case(client, args.model, case))
            elif case["kind"] == "conflict_schema":
                results.append(run_api_conflict_schema_case(client, args.model, case))
            elif case["kind"] == "truncation_schema":
                results.append(run_api_truncation_schema_case(client, args.model, case))
            elif case["kind"] == "schema_edge":
                results.append(run_api_schema_case(client, args.model, case))
                if args.run_cli:
                    results.append(run_cli_guided_json_case(args.model, case))
                if args.run_cli_no_think:
                    results.append(run_cli_guided_json_case(args.model, case, no_think=True))
            elif case["kind"] == "unsupported_schema":
                results.append(run_api_unsupported_schema_case(client, args.model, case))
            elif case["kind"] == "parse":
                results.append(run_sdk_parse_case(client, args.model, case))

        if args.run_cli:
            results.append(run_cli_invalid_schema(args.model))
        if args.run_cli_no_think:
            results.append(run_cli_invalid_schema_no_think(args.model))

        ok = all(item["ok"] for item in results)
        report_name = args.report_name or f"guided-json-evals-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
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
