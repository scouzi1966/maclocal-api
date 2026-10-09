#!/usr/bin/env python3
"""Responses reasoning/tool-turn qualification; stdlib only, no server lifecycle.

Deterministic adapter assertions live in XCTest. These live checks deliberately
report missing reasoning as UNCOVERED, never evidence that extraction works.
Use --self-test to validate the reporter without network or inference.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import unittest
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 120
DEFAULT_OUTPUT_TOKENS = 1024
TAGS = re.compile(r"</?(?:think|thinking|analysis)>|\[/?THINK\]", re.I)
PROMPT = "Before changing README.md, reason briefly about why you must read it first, then call read_file for README.md."
TOOL = {"type": "function", "name": "read_file", "description": "Read a file before editing it.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}


def decode_response(raw, streaming):
    if not streaming:
        return json.loads(raw), []
    events = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        events.append(json.loads(data))
    terminals = [event for event in events if event.get("type") in ("response.completed", "response.incomplete")]
    if len(terminals) != 1:
        raise ValueError(f"expected one terminal response, got {len(terminals)}")
    if any(event.get("type") in ("error", "response.failed") for event in events):
        raise ValueError("error event in stream")
    return terminals[0]["response"], events


def inspect_response(resource, events, thinking):
    """Return independent assertions, not one misleading aggregate pass."""
    checks = []
    def add(name, status, detail=""):
        checks.append({"name": name, "status": status, "detail": detail})
    if resource.get("status") != "completed":
        add("complete generation", "FAIL", f"status={resource.get('status')}; do not count truncated reasoning as success")
    else:
        add("complete generation", "PASS")
    output = resource.get("output")
    if not isinstance(output, list):
        add("output schema", "FAIL", "output must be an array")
        return checks
    calls = [item for item in output if item.get("type") == "function_call"]
    valid_call = len(calls) == 1 and calls[0].get("name") == "read_file" and bool(calls[0].get("call_id"))
    try:
        valid_call = valid_call and json.loads(calls[0]["arguments"]) == {"path": "README.md"}
    except (IndexError, KeyError, TypeError, ValueError):
        valid_call = False
    add("declared tool and arguments", "PASS" if valid_call else "FAIL", "engine/model boundary; missing call is not a reasoning pass" if not valid_call else "")
    reasoning = "".join(part.get("text", "") for item in output if item.get("type") == "reasoning"
                        for part in item.get("content", []) + item.get("summary", [])).strip()
    visible = "".join(part.get("text", "") for item in output if item.get("type") == "message"
                      for part in item.get("content", []))
    # Inspect reconstructed text too: a leaked tag may span separate SSE deltas.
    visible_deltas = "".join(event.get("delta", "") for event in events if event.get("type") == "response.output_text.delta")
    leaked = TAGS.search(visible + visible_deltas + reasoning)
    add("no raw thinking tag leakage", "FAIL" if leaked else "PASS", leaked.group() if leaked else "")
    if thinking:
        add("reasoning retained beside tool call", "PASS" if reasoning and valid_call else "UNCOVERED",
            "No observable reasoning+tool pair: inspect deterministic finalizer tests; do not infer model incapability" if not (reasoning and valid_call) else "")
    else:
        add("explicit thinking off", "FAIL" if reasoning else "PASS", "reasoning was returned despite explicit off" if reasoning else "")
    return checks


def run_live(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for mode, effort, enabled in (("on", "medium", True), ("off", "medium", False), ("none", "none", True)):
        for streaming in (False, True):
            label = f"{mode}-{'stream' if streaming else 'json'}"
            body = {"model": args.model, "input": PROMPT, "tools": [TOOL], "tool_choice": "required",
                    "temperature": 0, "top_p": 1, "top_k": 0, "seed": 123,
                    "max_output_tokens": args.max_output_tokens, "stream": streaming,
                    "reasoning": {"effort": effort}, "chat_template_kwargs": {"enable_thinking": enabled}}
            (output_dir / f"{label}.request.json").write_text(json.dumps(body, indent=2) + "\n")
            started = time.monotonic()
            try:
                request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/responses",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    raw = response.read().decode()
                (output_dir / f"{label}.response.txt").write_text(raw)
                resource, events = decode_response(raw, streaming)
                checks = inspect_response(resource, events, mode == "on")
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError):
                    (output_dir / f"{label}.response.txt").write_bytes(error.read())
                checks = [{"name": "transport/schema", "status": "FAIL", "detail": str(error)}]
            results.append({"case": label, "seconds": time.monotonic() - started, "checks": checks})
            print(label + ": " + ", ".join(f"{item['name']}={item['status']}" for item in checks), flush=True)
    totals = {status: sum(item["status"] == status for result in results for item in result["checks"])
              for status in ("PASS", "FAIL", "UNCOVERED")}
    report = {"model": args.model, "base_url": args.base_url, "totals": totals, "results": results,
              "scope": "Live observable Responses contracts; not proof of effective template forwarding or model quality parity."}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(totals), flush=True)
    # Uncovered is a distinct non-success exit, never silently promoted to pass.
    return 1 if totals["FAIL"] else 2 if totals["UNCOVERED"] else 0


class ReporterTests(unittest.TestCase):
    def fixture(self):
        return {"status": "completed", "output": [
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Inspect first."}], "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": '{"path":"README.md"}'}]}

    def statuses(self, resource, events=None, thinking=True):
        return {item["name"]: item["status"] for item in inspect_response(resource, events or [], thinking)}

    def test_valid_reasoning_and_tool(self):
        self.assertEqual(set(self.statuses(self.fixture()).values()), {"PASS"})

    def test_dropped_reasoning_is_uncovered_not_pass(self):
        resource = self.fixture()
        resource["output"].pop(0)
        self.assertEqual(self.statuses(resource)["reasoning retained beside tool call"], "UNCOVERED")

    def test_missing_tool_fails_and_does_not_cover_retention(self):
        resource = self.fixture()
        resource["output"].pop()
        result = self.statuses(resource)
        self.assertEqual(result["declared tool and arguments"], "FAIL")
        self.assertEqual(result["reasoning retained beside tool call"], "UNCOVERED")

    def test_reasoning_when_explicitly_off_fails(self):
        self.assertEqual(self.statuses(self.fixture(), thinking=False)["explicit thinking off"], "FAIL")

    def test_tag_split_across_deltas_fails(self):
        events = [{"type": "response.output_text.delta", "delta": part} for part in ("<thi", "nk>hidden", "</think>")]
        self.assertEqual(self.statuses(self.fixture(), events)["no raw thinking tag leakage"], "FAIL")

    def test_terminal_sse_and_json_have_same_logical_checks(self):
        fixture = self.fixture()
        raw = "event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": fixture}) + "\n\ndata: [DONE]\n\n"
        resource, events = decode_response(raw, True)
        self.assertEqual(self.statuses(resource, events), self.statuses(json.loads(json.dumps(fixture))))

    def test_incomplete_generation_fails(self):
        resource = self.fixture()
        resource["status"] = "incomplete"
        self.assertEqual(self.statuses(resource)["complete generation"], "FAIL")

    def test_missing_terminal_fails(self):
        with self.assertRaises(ValueError):
            decode_response('data: {"type":"response.created"}\n\ndata: [DONE]\n\n', True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:9999")
    parser.add_argument("--model")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    args = parser.parse_args()
    if args.self_test:
        return not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReporterTests)).wasSuccessful()
    if not args.model or not args.output_dir:
        parser.error("live tests require --model and a new --output-dir; no inference runs implicitly")
    if args.timeout <= 0 or args.max_output_tokens <= 0:
        parser.error("timeouts and token budget must be positive")
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
