"""Deterministic response assertions for mlx-model-test.sh."""

import json


def extract_score_payload(text):
    """Extract the last valid {score, reason} object from CLI output."""
    decoder = json.JSONDecoder()
    payload = None
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("score"), int):
            payload = candidate
    return payload


def expectation_for_prompt(config):
    """Give synthetic baselines their own oracle, not the variant oracle."""
    if config.get("prompt_idx", 0) < config.get("num_baseline", 0):
        # The global baseline is deliberately unrelated to every supplied tool.
        # A tool-enabled variant must therefore answer normally, not guess a call.
        return {"tool_calls": []} if config.get("tools") else {}
    return config.get("expect") or {}


def _validate_schema(value, schema, path="$"):
    failures = []
    schema_type = schema.get("type")
    type_matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if schema_type and not type_matches.get(schema_type, True):
        return [f"{path} expected type {schema_type}"]
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                failures.append(f"{path}.{key} is required")
        for key, child_schema in schema.get("properties", {}).items():
            if key in value:
                failures.extend(_validate_schema(value[key], child_schema, f"{path}.{key}"))
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            failures.extend(_validate_schema(item, schema["items"], f"{path}[{index}]"))
    return failures


def evaluate_expectations(
    expectation,
    *,
    content,
    finish_reason,
    logprobs_count,
    tool_calls,
):
    """Return (valid_json, failures) for a response and declarative expectation."""
    parsed_json = None
    valid_json = None
    if expectation.get("valid_json") is not None or expectation.get("json_schema") is not None:
        try:
            parsed_json = json.loads(content)
            valid_json = True
        except (json.JSONDecodeError, TypeError, ValueError):
            valid_json = False

    failures = []
    if expectation.get("valid_json") is not None and valid_json != expectation["valid_json"]:
        failures.append(f"valid_json expected {expectation['valid_json']}, got {valid_json}")
    if expectation.get("finish_reason") is not None and finish_reason != expectation["finish_reason"]:
        failures.append(
            f"finish_reason expected {expectation['finish_reason']!r}, got {finish_reason!r}"
        )
    if expectation.get("logprobs_min") is not None and logprobs_count < int(expectation["logprobs_min"]):
        failures.append(
            f"logprobs_count expected >= {expectation['logprobs_min']}, got {logprobs_count}"
        )

    if expectation.get("content_equals") is not None and content != expectation["content_equals"]:
        failures.append(
            f"content expected exactly {expectation['content_equals']!r}, got {content!r}"
        )
    if expectation.get("content_contains") is not None:
        required = expectation["content_contains"]
        required = required if isinstance(required, list) else [required]
        for value in required:
            if value not in content:
                failures.append(f"content expected to contain {value!r}")
    if expectation.get("content_not_contains") is not None:
        forbidden = expectation["content_not_contains"]
        forbidden = forbidden if isinstance(forbidden, list) else [forbidden]
        for value in forbidden:
            if value in content:
                failures.append(f"content must not contain {value!r}")
    if expectation.get("json_schema") is not None:
        failures.extend(_validate_schema(parsed_json, expectation["json_schema"]))

    if expectation.get("tool_calls") is not None:
        expected_calls = expectation["tool_calls"]
        if len(tool_calls) != len(expected_calls):
            failures.append(f"tool_calls count expected {len(expected_calls)}, got {len(tool_calls)}")
        unmatched_calls = list(enumerate(tool_calls))
        for expected_call in expected_calls:
            expected_name = expected_call.get("name")
            match_offset = next(
                (
                    offset
                    for offset, (_, call) in enumerate(unmatched_calls)
                    if call.get("function", {}).get("name") == expected_name
                ),
                None,
            )
            if match_offset is None:
                failures.append(
                    f"tool_calls missing expected function {expected_name!r}"
                )
                continue
            actual_index, actual_call = unmatched_calls.pop(match_offset)
            actual_name = actual_call["function"]["name"]
            try:
                actual_arguments = json.loads(actual_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError, ValueError):
                actual_arguments = None
            for key, expected_value in expected_call.get("arguments", {}).items():
                actual_value = actual_arguments.get(key) if isinstance(actual_arguments, dict) else None
                if actual_value != expected_value:
                    failures.append(
                        f"tool_calls[{actual_index}] ({actual_name}).arguments.{key} "
                        f"expected {expected_value!r}, got {actual_value!r}"
                    )

    return valid_json, failures
