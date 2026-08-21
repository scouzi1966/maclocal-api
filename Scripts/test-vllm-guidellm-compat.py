#!/usr/bin/env python3
"""Qualify AFM's vLLM metrics and GuideLLM HTTP/report contracts."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class QualificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def durable_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    require(
        path == Path("/Volumes/edata")
        or path == Path("/Volumes/edata2")
        or Path("/Volumes/edata") in path.parents
        or Path("/Volumes/edata2") in path.parents,
        "artifact directory must be on /Volumes/edata or /Volumes/edata2",
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 120,
) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return (
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {key.lower(): value for key, value in error.headers.items()},
            error.read(),
        )


def json_request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    expected: int = 200,
) -> dict[str, Any]:
    status, _headers, data = request(base_url, method, path, body)
    require(status == expected, f"{method} {path}: expected {expected}, got {status}: {data[:500]!r}")
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as error:
        raise QualificationError(f"{method} {path}: invalid JSON: {error}") from error
    require(isinstance(parsed, dict), f"{method} {path}: response is not an object")
    return parsed


def parse_sse(data: bytes) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    done = 0
    for block in data.decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n\n"):
        payload_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not payload_lines:
            continue
        payload = "\n".join(payload_lines)
        if payload == "[DONE]":
            done += 1
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise QualificationError(f"invalid SSE JSON: {payload[:500]!r}: {error}") from error
        require(isinstance(event, dict), "SSE event is not an object")
        events.append(event)
    return events, done


def streaming_completion(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    status, headers, data = request(base_url, "POST", path, payload, timeout=300)
    require(status == 200, f"streaming {path}: HTTP {status}: {data[:500]!r}")
    require("text/event-stream" in headers.get("content-type", ""), f"streaming {path}: wrong content type")
    events, done = parse_sse(data)
    require(done == 1, f"streaming {path}: expected one [DONE], got {done}")
    require(events, f"streaming {path}: no JSON events")
    errors = [event for event in events if "error" in event]
    require(not errors, f"streaming {path}: OpenAI error event: {errors[0] if errors else ''}")
    usage_events = [event for event in events if event.get("usage") is not None]
    require(len(usage_events) == 1, f"streaming {path}: expected one final usage event, got {len(usage_events)}")
    usage = usage_events[0]["usage"]
    require(usage.get("prompt_tokens", 0) > 0, f"streaming {path}: prompt_tokens is not positive")
    require(usage.get("completion_tokens", 0) > 0, f"streaming {path}: completion_tokens is not positive")
    finish_reasons = [
        choice.get("finish_reason")
        for event in events
        for choice in event.get("choices", [])
        if choice.get("finish_reason") is not None
    ]
    require(len(finish_reasons) == 1, f"streaming {path}: expected one finish reason, got {finish_reasons}")
    text = "".join(
        str(
            choice.get("text")
            or choice.get("delta", {}).get("content")
            or choice.get("delta", {}).get("reasoning_content")
            or choice.get("delta", {}).get("reasoning")
            or ""
        )
        for event in events
        for choice in event.get("choices", [])
    )
    require(text, f"streaming {path}: no generated text")
    return {"finish_reason": finish_reasons[0], "usage": usage, "text": text, "events": len(events)}


def parse_metrics(text: str) -> tuple[dict[str, list[tuple[dict[str, str], float]]], dict[str, str]]:
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError as error:
        raise QualificationError("prometheus_client is required for metrics qualification") from error

    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    types: dict[str, str] = {}
    try:
        families = list(text_string_to_metric_families(text))
    except Exception as error:
        raise QualificationError(f"prometheus_client rejected /metrics: {error}") from error
    for family in families:
        family_name = family.name
        types[family_name] = family.type
        for sample in family.samples:
            value = float(sample.value)
            require(math.isfinite(value), f"non-finite metric value for {sample.name}")
            samples.setdefault(sample.name, []).append((dict(sample.labels), value))
    return samples, types


def metric_value(
    samples: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    labels = labels or {}
    matches = [value for found_labels, value in samples.get(name, []) if all(found_labels.get(k) == v for k, v in labels.items())]
    require(len(matches) == 1, f"expected one sample for {name} labels={labels}, got {len(matches)}")
    return matches[0]


def check_promtool(metrics_text: str, explicit_path: str | None) -> str:
    path = explicit_path or shutil.which("promtool")
    if not path:
        return "not-installed"
    parser_result = subprocess.run(
        [path, "check", "metrics", "--extended", "--lint=none"],
        input=metrics_text,
        text=True,
        capture_output=True,
        check=False,
    )
    require(
        parser_result.returncode == 0,
        f"promtool parser rejected /metrics: {parser_result.stderr or parser_result.stdout}",
    )
    lint_result = subprocess.run(
        [path, "check", "metrics"],
        input=metrics_text,
        text=True,
        capture_output=True,
        check=False,
    )
    diagnostics = [line for line in (lint_result.stderr + lint_result.stdout).splitlines() if line]
    abbreviated_unit_diagnostics = {
        "afm:snapshot_timestamp_ms metric names should not contain abbreviated units",
        "vllm:avg_generation_throughput_toks_per_s metric names should not contain abbreviated units",
        "vllm:avg_prompt_throughput_toks_per_s metric names should not contain abbreviated units",
    }
    unexpected = [
        line
        for line in diagnostics
        if line not in abbreviated_unit_diagnostics
        and (
            not line.startswith(("afm:", "vllm:"))
            or line.split(" metric names", 1)[-1] != " should not contain ':'"
        )
    ]
    require(not unexpected, f"promtool reported unexpected lint diagnostics: {unexpected}")
    require(
        lint_result.returncode == 0 or diagnostics,
        f"promtool lint failed without a diagnostic (exit {lint_result.returncode})",
    )
    suffix = "known vLLM-compatibility style lint only" if diagnostics else "clean lint"
    return f"{path} ({suffix})"


def assert_metric_parity(samples: dict[str, list[tuple[dict[str, str], float]]]) -> dict[str, float]:
    pairs = {
        "generated_tokens": ("afm:generation_tokens_total", "vllm:generation_tokens_total"),
        "running_requests": ("afm:num_requests_running", "vllm:num_requests_running"),
        "waiting_requests": ("afm:num_requests_waiting", "vllm:num_requests_waiting"),
    }
    values: dict[str, float] = {}
    for key, (afm_name, vllm_name) in pairs.items():
        afm = metric_value(samples, afm_name)
        vllm = metric_value(samples, vllm_name)
        require(afm == vllm, f"metric parity failed for {key}: AFM={afm}, vLLM={vllm}")
        values[key] = afm
    computed = metric_value(samples, "afm:prompt_tokens_total")
    vllm_computed = metric_value(samples, "vllm:prompt_tokens_by_source_total", {"source": "local_compute"})
    require(computed == vllm_computed, f"computed prompt token parity failed: {computed} != {vllm_computed}")
    values["computed_prompt_tokens"] = computed
    return values


def fetch_metrics(base_url: str, artifact: Path, promtool: str | None) -> tuple[dict[str, list[tuple[dict[str, str], float]]], dict[str, float], str]:
    status, headers, data = request(base_url, "GET", "/metrics")
    require(status == 200, f"GET /metrics returned {status}")
    require("text/plain" in headers.get("content-type", ""), "GET /metrics did not return text/plain")
    text = data.decode()
    artifact.write_text(text)
    samples, _types = parse_metrics(text)
    required = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:inter_token_latency_seconds_bucket",
        "vllm:e2e_request_latency_seconds_bucket",
    ]
    missing = [name for name in required if name not in samples]
    require(not missing, f"missing required vLLM samples: {missing}")
    parity = assert_metric_parity(samples)
    return samples, parity, check_promtool(text, promtool)


def qualify_http(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = durable_directory(args.artifact_dir)
    base_url = args.base_url.rstrip("/")
    models_one = json_request(base_url, "GET", "/v1/models")
    time.sleep(0.05)
    models_two = json_request(base_url, "GET", "/v1/models")
    entries_one = models_one.get("data")
    entries_two = models_two.get("data")
    require(isinstance(entries_one, list) and entries_one, "/v1/models returned no models")
    require(entries_one == entries_two, "/v1/models ordering or created timestamps are unstable")
    model = args.model or entries_one[0].get("id")
    require(isinstance(model, str) and model, "could not select a model")
    if "loaded" in entries_one[0]:
        require(entries_one[0]["loaded"] is True, "loaded model is not first")

    before, before_parity, promtool = fetch_metrics(base_url, artifact_dir / "metrics-before.prom", args.promtool)

    nonstream = json_request(
        base_url,
        "POST",
        "/v1/completions",
        {"model": model, "prompt": "Reply with one short word.", "max_tokens": 4, "temperature": 0},
    )
    usage = nonstream.get("usage", {})
    require(usage.get("prompt_tokens", 0) > 0 and usage.get("completion_tokens", 0) > 0, "non-stream completion lacks exact usage")
    require(usage.get("total_tokens") == usage.get("prompt_tokens") + usage.get("completion_tokens"), "non-stream completion usage is not exact")
    require(nonstream.get("choices", [{}])[0].get("finish_reason") in {"stop", "length"}, "invalid non-stream finish reason")

    common_stream = {
        "model": model,
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    raw_stream = streaming_completion(
        base_url, "/v1/completions", {**common_stream, "prompt": "Count upward using words."}
    )
    chat_stream = streaming_completion(
        base_url,
        "/v1/chat/completions",
        {**common_stream, "messages": [{"role": "user", "content": "Count upward using words."}]},
    )
    require(raw_stream["finish_reason"] == "length", "raw ignore_eos did not stop at max_tokens")
    require(chat_stream["finish_reason"] == "length", "chat ignore_eos did not stop at max_tokens")

    bad = json_request(
        base_url,
        "POST",
        "/v1/completions",
        {"model": model, "prompt": ["not", "supported"]},
        expected=400,
    )
    error = bad.get("error", {})
    require(error.get("param") == "prompt", "prompt-array validation error does not identify param=prompt")

    concurrency = max(2, args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                streaming_completion,
                base_url,
                "/v1/completions",
                {**common_stream, "prompt": f"Emit a short numbered item {index}."},
            )
            for index in range(concurrency)
        ]
        concurrent_results = [future.result() for future in futures]

    required_success_delta = concurrency + 3
    deadline = time.monotonic() + 2
    while True:
        after, after_parity, _ = fetch_metrics(base_url, artifact_dir / "metrics-after.prom", args.promtool)
        generated_delta = metric_value(after, "vllm:generation_tokens_total") - metric_value(before, "vllm:generation_tokens_total")
        prompt_delta = metric_value(after, "vllm:prompt_tokens_total") - metric_value(before, "vllm:prompt_tokens_total")
        success_delta = sum(value for _labels, value in after["vllm:request_success_total"]) - sum(
            value for _labels, value in before["vllm:request_success_total"]
        )
        if success_delta >= required_success_delta or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    require(generated_delta > 0, "vLLM generation token counter did not increase")
    require(prompt_delta > 0, "vLLM prompt token counter did not increase")
    require(success_delta >= required_success_delta, f"vLLM success counter delta too small: {success_delta}")

    result = {
        "status": "passed",
        "base_url": base_url,
        "model": model,
        "models": len(entries_one),
        "promtool": promtool,
        "before_parity": before_parity,
        "after_parity": after_parity,
        "metric_deltas": {
            "generation_tokens": generated_delta,
            "prompt_tokens": prompt_delta,
            "successful_requests": success_delta,
        },
        "raw_stream": {k: v for k, v in raw_stream.items() if k != "text"},
        "chat_stream": {k: v for k, v in chat_stream.items() if k != "text"},
        "concurrent_requests": len(concurrent_results),
    }
    (artifact_dir / "http-contract-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def nested_value(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        require(isinstance(current, dict) and part in current, f"GuideLLM report is missing {path}")
        current = current[part]
    return current


def positive_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0, f"{label} must be positive, got {value!r}")
    return float(value)


def nonnegative_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0,
        f"{label} must be nonnegative, got {value!r}",
    )
    return float(value)


def qualify_guidellm(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = durable_directory(args.artifact_dir)
    json_path = Path(args.json).resolve()
    csv_path = Path(args.csv).resolve()
    html_path = Path(args.html).resolve()
    for path in [json_path, csv_path, html_path]:
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty GuideLLM output: {path}")
    report = json.loads(json_path.read_text())
    benchmarks = report.get("benchmarks")
    require(isinstance(benchmarks, list) and benchmarks, "GuideLLM JSON has no benchmarks")
    checked: list[dict[str, float]] = []
    requires_throughput = False
    positive_paths = {
        "latency_seconds": "metrics.request_latency.successful.mean",
        "prompt_tokens": "metrics.prompt_token_count.successful.mean",
        "output_tokens": "metrics.output_token_count.successful.mean",
        "time_per_output_token_ms": "metrics.time_per_output_token_ms.successful.mean",
    }
    stream_paths = {
        "ttft_ms": "metrics.time_to_first_token_ms.successful.mean",
        "itl_ms": "metrics.inter_token_latency_ms.successful.mean",
    }
    for index, benchmark in enumerate(benchmarks):
        require(nested_value(benchmark, "metrics.request_totals.errored") == 0, f"benchmark {index} has protocol errors")
        require(nested_value(benchmark, "metrics.request_totals.incomplete") == 0, f"benchmark {index} has incomplete requests")
        successful = positive_number(
            nested_value(benchmark, "metrics.request_totals.successful"),
            f"benchmark {index} successful requests",
        )
        requires_throughput = requires_throughput or successful > 1
        metrics = {
            name: positive_number(nested_value(benchmark, path), f"benchmark {index} {name}")
            for name, path in positive_paths.items()
        }
        throughput = nested_value(benchmark, "metrics.output_tokens_per_second.successful.mean")
        metrics["throughput_tokens_per_second"] = (
            positive_number(throughput, f"benchmark {index} throughput_tokens_per_second")
            if successful > 1
            else nonnegative_number(throughput, f"benchmark {index} throughput_tokens_per_second")
        )
        metrics.update(
            {
                name: (
                    positive_number(nested_value(benchmark, path), f"benchmark {index} {name}")
                    if args.streaming
                    else nonnegative_number(nested_value(benchmark, path), f"benchmark {index} {name}")
                )
                for name, path in stream_paths.items()
            }
        )
        checked.append(metrics)

    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    require(len(rows) >= 4, "GuideLLM CSV does not contain headers and benchmark rows")
    csv_text = csv_path.read_text()
    csv_labels = (
        ["Time to First Token", "Inter Token Latency", "Output Tokens/Sec"]
        if args.streaming
        else ["Request Latency", "Time per Output Token", "Successful Output Tokens"]
    )
    if requires_throughput and not args.streaming:
        csv_labels.append("Token Throughput")
    for label in csv_labels:
        require(label in csv_text, f"GuideLLM CSV is missing {label}")
    html_text = html_path.read_text(errors="replace").lower()
    require("<html" in html_text and "benchmark" in html_text, "GuideLLM HTML report is not a benchmark document")

    result = {
        "status": "passed",
        "guidellm_version": report.get("metadata", {}).get("guidellm_version"),
        "streaming": args.streaming,
        "benchmarks": len(benchmarks),
        "metrics": checked,
        "outputs": {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)},
    }
    (artifact_dir / "guidellm-report-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    http = subparsers.add_parser("http", help="qualify a running AFM server")
    http.add_argument("--base-url", required=True)
    http.add_argument("--model")
    http.add_argument("--artifact-dir", required=True)
    http.add_argument("--concurrency", type=int, default=4)
    http.add_argument("--promtool")
    http.set_defaults(handler=qualify_http)
    report = subparsers.add_parser("guidellm-report", help="qualify GuideLLM JSON/CSV/HTML outputs")
    report.add_argument("--json", required=True)
    report.add_argument("--csv", required=True)
    report.add_argument("--html", required=True)
    report.add_argument("--artifact-dir", required=True)
    report.add_argument(
        "--streaming",
        action="store_true",
        help="require positive GuideLLM TTFT and ITL metrics",
    )
    report.set_defaults(handler=qualify_guidellm)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = args.handler(args)
    except QualificationError as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
