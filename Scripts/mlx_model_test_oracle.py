"""Deterministic response assertions for mlx-model-test.sh."""

import json
import os
import re
from pathlib import Path


def configuration_allows_safe_partial_cache_miss(config):
    """Return whether correctness may require a cold fallback for a hybrid cache."""
    text_config = config.get("text_config") or {}
    layer_types = (
        text_config.get("layer_types")
        or config.get("layer_types")
        or text_config.get("layers_block_type")
        or config.get("layers_block_type")
        or []
    )
    recurrent_markers = ("linear", "mamba", "ssm", "delta", "recurrent")
    if any(
        any(marker in str(layer_type).lower() for marker in recurrent_markers)
        for layer_type in layer_types
    ):
        return True

    model_type = str(text_config.get("model_type") or config.get("model_type") or "").lower()
    architectures = text_config.get("architectures") or config.get("architectures") or []
    architecture_text = " ".join(str(architecture).lower() for architecture in architectures)
    return any(
        marker in model_type or marker in architecture_text
        for marker in ("deepseek_v4", "deepseekv4")
    )


def inspect_model_safe_partial_cache_miss(model, environ=None):
    """Return a local checkpoint's cache policy, or None when it cannot be inspected."""
    environ = os.environ if environ is None else environ
    model_path = Path(model).expanduser()
    candidates = [model_path / "config.json"]

    mac_cache_root = None
    if environ.get("MACAFM_MLX_MODEL_CACHE"):
        mac_cache_root = Path(environ["MACAFM_MLX_MODEL_CACHE"]).expanduser()
        candidates.extend(
            (
                mac_cache_root / model / "config.json",
                mac_cache_root / "models" / model / "config.json",
            )
        )

    cache_roots = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if environ.get(key):
            cache_roots.append(Path(environ[key]).expanduser())
    if environ.get("HF_HOME"):
        cache_roots.append(Path(environ["HF_HOME"]).expanduser() / "hub")
    cache_roots.append(Path(environ.get("HOME", str(Path.home()))) / ".cache/huggingface/hub")

    repo_directory_name = "models--" + model.replace("/", "--")
    for cache_root in cache_roots:
        candidates.append(cache_root / model / "config.json")
        repo_directory = cache_root / repo_directory_name
        main_ref = repo_directory / "refs/main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(repo_directory / "snapshots" / revision / "config.json")

    for config_path in candidates:
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return configuration_allows_safe_partial_cache_miss(config)
    return None


def model_allows_safe_partial_cache_miss(model, environ=None):
    """Inspect a local checkpoint without depending on a running AFM server."""
    return inspect_model_safe_partial_cache_miss(model, environ=environ) is True


def evaluation_lane(*, model, afm_args="", is_baseline=False, has_expectation=True):
    """Separate native protocol checks from explicit cross-family parser experiments."""
    if is_baseline:
        return "model_agent_behavior"
    match = re.search(r"(?:^|\s)--tool-call-parser\s+(\S+)", afm_args or "")
    if match:
        parser = match.group(1).lower()
        model_name = (model or "").lower()
        native_family = (
            (parser.startswith("qwen") and "qwen" in model_name)
            or (parser.startswith("deepseek") and "deepseek" in model_name)
            or (parser.startswith("nemotron") and "nemotron" in model_name)
            or (parser.startswith("muse") and "muse" in model_name)
        )
        if not native_family:
            return "forced_parser_compatibility"
    if not has_expectation:
        return "model_agent_behavior"
    return "native_protocol"


def classify_result_likelihood(
    *, model, label, afm_args="", is_baseline=False, status="OK", failures=None
):
    """Return a conservative failure bucket without claiming component ownership."""
    failures = failures or []
    if status == "SKIP":
        return "capability unavailable"
    if status != "OK":
        return "engine/runtime likely"
    if not failures:
        return "conformant"
    lane = evaluation_lane(model=model, afm_args=afm_args, is_baseline=is_baseline)
    if lane == "forced_parser_compatibility":
        return "forced-parser compatibility experiment"
    normalized_label = (label or "").lower()
    engine_prefixes = (
        "stop-", "logprobs", "agent-cached", "cache-", "batch-", "kv-",
        "guided-", "grammar-", "structured-", "seed-",
    )
    if normalized_label.startswith(engine_prefixes):
        return "engine/runtime likely"
    if normalized_label.startswith("tool-call"):
        return "parser/model boundary needs triage"
    return "model behavior likely"


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
    prompt_expectations = config.get("expect_by_prompt") or []
    relative_index = config.get("prompt_idx", 0) - config.get("num_baseline", 0)
    if 0 <= relative_index < len(prompt_expectations):
        return prompt_expectations[relative_index] or {}
    return config.get("expect") or {}


def request_configuration_fields(config, *, expectation=None):
    """Return the request configuration that must survive every result path."""
    fields = {
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
        "max_completion_tokens": config.get("max_completion_tokens"),
        "system_prompt": config.get("system", ""),
        "developer_prompt": config.get("developer", ""),
        "server_instructions": config.get("instructions", ""),
        "required_capabilities": config.get("requires") or [],
        "afm_args": config.get("afm_args", ""),
        "safe_partial_cache_miss": bool(config.get("safe_partial_cache_miss", False)),
    }
    for key in (
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "repetition_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "media",
        "tools",
        "stream",
    ):
        if key in config and config[key] is not None:
            fields[key] = config[key]
    if expectation:
        fields["expect"] = expectation
    return fields


def transport_failure_record(config, *, prompt, error, load_time_s):
    """Build a complete result for a request/client/transport failure."""
    is_baseline = config.get("prompt_idx", 0) < config.get("num_baseline", 0)
    expectation = expectation_for_prompt(config)
    model = config.get("model", "")
    label = config.get("label", "")
    afm_args = config.get("afm_args", "")
    record = {
        "model": model,
        "label": label,
        "prompt": prompt,
        "is_baseline": is_baseline,
        "status": "FAIL",
        "transport_status": "fail",
        "assertion_status": "not_run",
        "overall_status": "fail",
        "evaluation_lane": "native_protocol",
        "failure_classification": classify_result_likelihood(
            model=model,
            label=label,
            afm_args=afm_args,
            is_baseline=is_baseline,
            status="FAIL",
        ),
        "error": error,
        "load_time_s": load_time_s,
    }
    record.update(request_configuration_fields(config, expectation=expectation))
    return record


def failed_run_records(config, *, error, load_time_s):
    """Build one classified transport-failure record for each planned prompt."""
    prompts = config.get("prompts") or [""]
    records = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_config = dict(config, prompt_idx=prompt_index)
        records.append(
            transport_failure_record(
                prompt_config,
                prompt=prompt,
                error=error,
                load_time_s=load_time_s,
            )
        )
    return records


def skipped_run_records(config, *, reason):
    """Build complete capability-skip records without losing request metadata."""
    prompts = config.get("prompts") or [""]
    records = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_config = dict(config, prompt_idx=prompt_index)
        is_baseline = prompt_index < config.get("num_baseline", 0)
        expectation = expectation_for_prompt(prompt_config)
        model = config.get("model", "")
        label = config.get("label", "")
        afm_args = config.get("afm_args", "")
        record = {
            "model": model,
            "label": label,
            "prompt": prompt,
            "is_baseline": is_baseline,
            "status": "SKIP",
            "transport_status": "not_run",
            "assertion_status": "not_run",
            "overall_status": "skip",
            "skip_reason": reason,
            "evaluation_lane": evaluation_lane(
                model=model,
                afm_args=afm_args,
                is_baseline=is_baseline,
                has_expectation=bool(expectation),
            ),
            "failure_classification": classify_result_likelihood(
                model=model,
                label=label,
                afm_args=afm_args,
                is_baseline=is_baseline,
                status="SKIP",
            ),
        }
        record.update(request_configuration_fields(config, expectation=expectation))
        records.append(record)
    return records


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
    cached_input_tokens=0,
    safe_partial_cache_miss=False,
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
    if (
        expectation.get("cached_input_tokens_min") is not None
        and cached_input_tokens < int(expectation["cached_input_tokens_min"])
        and not (
            expectation.get("allow_safe_partial_cache_miss") is True
            and safe_partial_cache_miss
            and cached_input_tokens == 0
        )
    ):
        failures.append(
            "cached_input_tokens expected >= "
            f"{expectation['cached_input_tokens_min']}, got {cached_input_tokens}"
        )
    if (
        expectation.get("cached_input_tokens_max") is not None
        and cached_input_tokens > int(expectation["cached_input_tokens_max"])
    ):
        failures.append(
            "cached_input_tokens expected <= "
            f"{expectation['cached_input_tokens_max']}, got {cached_input_tokens}"
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
