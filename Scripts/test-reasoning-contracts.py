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


def reasoning_stream_errors(output, events):
    """Validate delivered item/delta/done events, not just terminal JSON.

    Either public reasoning event family is accepted; no requirement to copy
    raw thinking into a summary solely for a particular client UI.
    """
    errors = []
    reasoning_items = {item.get("id"): (index, item) for index, item in enumerate(output)
                       if item.get("type") == "reasoning"}
    families = (("response.reasoning_text", "content", "content_index"),
                ("response.reasoning_summary_text", "summary", "summary_index"))
    for event in events:
        if any(event.get("type", "").startswith(family + ".") for family, _, _ in families):
            if event.get("item_id") not in reasoning_items:
                errors.append("reasoning event has no matching terminal reasoning item")
    for item_id, (output_index, item) in reasoning_items.items():
        if not item_id:
            errors.append("reasoning item has no id")
            continue
        added = [index for index, event in enumerate(events) if event.get("type") == "response.output_item.added"
                 and event.get("item", {}).get("id") == item_id]
        done = [(index, event) for index, event in enumerate(events) if event.get("type") == "response.output_item.done"
                and event.get("item", {}).get("id") == item_id]
        if len(added) != 1 or len(done) != 1 or done[0][1].get("item") != item:
            errors.append("reasoning item added/done missing, duplicated, or inconsistent with terminal output")
            continue
        delivered_parts = 0
        for family, field, index_field in families:
            for part_index, part in enumerate(item.get(field) or []):
                text = part.get("text", "")
                if not text:
                    continue
                matching = [(index, event) for index, event in enumerate(events)
                            if event.get("item_id") == item_id and event.get(index_field) == part_index
                            and event.get("type") in (family + ".delta", family + ".done")]
                deltas = [(index, event) for index, event in matching if event["type"].endswith(".delta")]
                completed = [(index, event) for index, event in matching if event["type"].endswith(".done")]
                if not deltas or len(completed) != 1:
                    errors.append(f"{field}[{part_index}] lacks reasoning-specific delta/done delivery")
                    continue
                if any(event.get("output_index") != output_index for _, event in matching):
                    errors.append("reasoning event output_index mismatch")
                if "".join(event.get("delta", "") for _, event in deltas) != text or completed[0][1].get("text") != text:
                    errors.append("reasoning deltas/done differ from terminal content")
                if not (added[0] < min(index for index, _ in deltas)
                        <= max(index for index, _ in deltas) < completed[0][0] < done[0][0]):
                    errors.append("reasoning lifecycle order is invalid")
                delivered_parts += 1
        if not delivered_parts:
            errors.append("reasoning item contains no delivered text part")
    return errors


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
    reasoning_deltas = "".join(event.get("delta", "") for event in events
                               if event.get("type") in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"))
    leaked = TAGS.search(visible + visible_deltas + reasoning + reasoning_deltas)
    add("no raw thinking tag leakage", "FAIL" if leaked else "PASS", leaked.group() if leaked else "")
    if thinking:
        add("reasoning retained beside tool call", "PASS" if reasoning and valid_call else "UNCOVERED",
            "No observable reasoning+tool pair: inspect deterministic finalizer tests; do not infer model incapability" if not (reasoning and valid_call) else "")
    else:
        has_reasoning = bool(reasoning or reasoning_deltas)
        add("explicit thinking off", "FAIL" if has_reasoning else "PASS", "reasoning was returned despite explicit off" if has_reasoning else "")
    if events:
        errors = reasoning_stream_errors(output, events)
        add("reasoning SSE delivery matches terminal output", "FAIL" if errors else "PASS", "; ".join(errors))
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
            {"type": "reasoning", "id": "rs_1", "content": [{"type": "reasoning_text", "text": "Inspect first."}], "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": '{"path":"README.md"}'}]}

    def statuses(self, resource, events=None, thinking=True):
        return {item["name"]: item["status"] for item in inspect_response(resource, events or [], thinking)}

    def stream_fixture(self, summary=False):
        resource = self.fixture()
        item = resource["output"][0]
        field, index_field, family = "content", "content_index", "response.reasoning_text"
        if summary:
            item["summary"] = [{"type": "summary_text", "text": "Inspect first."}]
            item["content"] = []
            field, index_field, family = "summary", "summary_index", "response.reasoning_summary_text"
        base = {"item_id": "rs_1", "output_index": 0, index_field: 0}
        events = [
            {"type": "response.output_item.added", "item": {"id": "rs_1", "type": "reasoning"}},
            {**base, "type": family + ".delta", "delta": "Inspect "},
            {**base, "type": family + ".delta", "delta": "first."},
            {**base, "type": family + ".done", "text": item[field][0]["text"]},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": resource},
        ]
        return resource, events

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
        fixture, fixture_events = self.stream_fixture()
        raw = "".join("data: " + json.dumps(event) + "\n\n" for event in fixture_events) + "data: [DONE]\n\n"
        resource, events = decode_response(raw, True)
        statuses = self.statuses(resource, events)
        self.assertEqual(statuses.pop("reasoning SSE delivery matches terminal output"), "PASS")
        self.assertEqual(statuses, self.statuses(json.loads(json.dumps(fixture))))

    def test_summary_channel_is_accepted_without_content(self):
        resource, events = self.stream_fixture(summary=True)
        self.assertEqual(set(self.statuses(resource, events).values()), {"PASS"})

    def test_terminal_only_reasoning_fails_stream_delivery(self):
        resource = self.fixture()
        events = [{"type": "response.completed", "response": resource}]
        self.assertEqual(self.statuses(resource, events)["reasoning SSE delivery matches terminal output"], "FAIL")

    def test_item_events_without_deltas_fail(self):
        resource, events = self.stream_fixture()
        events = [event for event in events if not event["type"].endswith(".delta")]
        self.assertEqual(self.statuses(resource, events)["reasoning SSE delivery matches terminal output"], "FAIL")

    def test_deltas_that_disagree_with_terminal_fail(self):
        resource, events = self.stream_fixture()
        events[1]["delta"] = "Wrong "
        self.assertEqual(self.statuses(resource, events)["reasoning SSE delivery matches terminal output"], "FAIL")

    def test_wrong_item_id_fails(self):
        resource, events = self.stream_fixture()
        events[1]["item_id"] = "missing"
        self.assertEqual(self.statuses(resource, events)["reasoning SSE delivery matches terminal output"], "FAIL")

    def test_wrong_output_index_fails(self):
        resource, events = self.stream_fixture()
        events[1]["output_index"] = 2
        self.assertEqual(self.statuses(resource, events)["reasoning SSE delivery matches terminal output"], "FAIL")

    def test_done_before_delta_fails(self):
        resource, events = self.stream_fixture()
        events[1], events[3] = events[3], events[1]
        self.assertEqual(self.statuses(resource, events)["reasoning SSE delivery matches terminal output"], "FAIL")

    def test_stream_only_reasoning_fails_off(self):
        resource, events = self.stream_fixture()
        resource["output"].pop(0)
        self.assertEqual(self.statuses(resource, events, thinking=False)["explicit thinking off"], "FAIL")

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
