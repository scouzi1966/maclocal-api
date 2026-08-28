#!/usr/bin/env python3
"""Reapply deterministic MLX result oracles without rerunning inference."""

import argparse
import json
from pathlib import Path

from mlx_model_test_oracle import (
    classify_result_likelihood,
    evaluate_expectations,
    inspect_model_safe_partial_cache_miss,
    model_allows_safe_partial_cache_miss,
)


def paths_refer_to_same_file(input_path, output_path):
    if input_path.resolve() == output_path.resolve():
        return True
    try:
        return output_path.exists() and input_path.samefile(output_path)
    except OSError:
        return False


def reassess_record(
    record,
    *,
    safe_cache_labels=frozenset(),
    reinspect_cache_policy=False,
):
    if record.get("_meta"):
        updated = dict(record)
        updated["oracle_reassessment"] = "recurrent-cache-aware-v1"
        return updated
    if record.get("status") != "OK":
        return record

    updated = dict(record)
    if reinspect_cache_policy:
        inspected_policy = inspect_model_safe_partial_cache_miss(
            record.get("model", "")
        )
        if inspected_policy is None:
            if "safe_partial_cache_miss" not in record:
                raise ValueError(
                    "cache policy reinspection unavailable and record has no "
                    f"captured policy for model {record.get('model', '')!r}"
                )
            safe_partial_cache_miss = bool(record["safe_partial_cache_miss"])
            updated["cache_policy_provenance"] = (
                "recorded-result-reinspection-unavailable"
            )
        else:
            safe_partial_cache_miss = inspected_policy
            updated["cache_policy_provenance"] = (
                "explicit-local-checkpoint-reinspection"
            )
    elif "safe_partial_cache_miss" in record:
        safe_partial_cache_miss = bool(record["safe_partial_cache_miss"])
        updated["cache_policy_provenance"] = "recorded-result"
    else:
        safe_partial_cache_miss = model_allows_safe_partial_cache_miss(
            record.get("model", "")
        )
        updated["cache_policy_provenance"] = "legacy-local-checkpoint-inspection"
    updated["safe_partial_cache_miss"] = safe_partial_cache_miss
    expectation = dict(record.get("expect") or {})
    if not expectation:
        return updated
    if record.get("label") in safe_cache_labels:
        expectation["allow_safe_partial_cache_miss"] = True
        updated["expect"] = expectation

    valid_json, failures = evaluate_expectations(
        expectation,
        content=record.get("content", ""),
        finish_reason=record.get("finish_reason"),
        logprobs_count=record.get("logprobs_count", 0),
        tool_calls=record.get("tool_calls") or [],
        cached_input_tokens=record.get("cached_input_tokens", 0),
        safe_partial_cache_miss=safe_partial_cache_miss,
    )
    updated["assertion_failures"] = failures
    updated["assertion_status"] = "fail" if failures else "pass"
    updated["overall_status"] = "fail" if failures else "pass"
    updated["failure_classification"] = classify_result_likelihood(
        model=record.get("model", ""),
        label=record.get("label", ""),
        afm_args=record.get("afm_args", ""),
        is_baseline=record.get("is_baseline", False),
        status=record.get("status", "OK"),
        failures=failures,
    )
    if valid_json is not None:
        updated["is_valid_json"] = valid_json
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Reapply deterministic oracles to an AFM JSONL result file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-safe-partial-cache-label",
        action="append",
        default=[],
        help="explicit scenario label allowed to use recurrent safe cold fallback",
    )
    parser.add_argument(
        "--reinspect-cache-policy",
        action="store_true",
        help=(
            "derive cache policy from the local checkpoint even when the input "
            "record contains a previously captured policy"
        ),
    )
    args = parser.parse_args()

    if paths_refer_to_same_file(args.input, args.output):
        parser.error("input and output must differ so raw inference data stays immutable")

    records = []
    with args.input.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(
                    reassess_record(
                        json.loads(line),
                        safe_cache_labels=frozenset(args.allow_safe_partial_cache_label),
                        reinspect_cache_policy=args.reinspect_cache_policy,
                    )
                )
            except json.JSONDecodeError as error:
                raise SystemExit(f"{args.input}:{line_number}: {error}") from error
            except ValueError as error:
                raise SystemExit(f"{args.input}:{line_number}: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
