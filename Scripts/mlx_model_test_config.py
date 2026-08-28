"""Parser for mlx-model-test prompt/config files."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
from pathlib import Path


INTEGER_PARAMETERS = {
    "max_tokens",
    "max_completion_tokens",
    "top_k",
    "seed",
    "top_logprobs",
}
FLOAT_PARAMETERS = {
    "temperature",
    "top_p",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "frequency_penalty",
}
BOOLEAN_PARAMETERS = {"logprobs"}
JSON_PARAMETERS = {"tools", "expect", "expect_by_prompt"}


def capture_prompts_snapshot(
    prompts_file: str | Path, results_file: str | Path
) -> tuple[Path, str]:
    """Atomically snapshot the exact prompt bytes that inference will consume."""
    source_path = Path(prompts_file)
    results_path = Path(results_file).resolve()
    destination = (
        results_path.with_suffix(".prompts.txt")
        if results_path.suffix == ".jsonl"
        else Path(f"{results_path}.prompts.txt")
    )
    contents = source_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()

    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(temporary_descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    if destination.is_symlink() or not destination.is_file():
        raise ValueError("captured prompt snapshot is not a regular file")
    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError("captured prompt snapshot failed digest verification")
    return destination, digest


def publish_report_atomically(temporary_file: str | Path, report_file: str | Path) -> None:
    """Publish a completed report without accepting a directory as its target."""
    temporary_path = Path(temporary_file)
    report_path = Path(report_file)
    if report_path.is_dir():
        raise IsADirectoryError(f"report destination is a directory: {report_path}")
    os.replace(temporary_path, report_path)


def materialize_verified_prompts_snapshot(filepath: str | Path) -> Path | None:
    """Copy a verified sibling prompt snapshot to a private judge input file."""
    results_path = Path(filepath).resolve()
    with Path(filepath).open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not record.get("_meta"):
                return None

            snapshot_name = record.get("prompts_snapshot")
            expected_digest = record.get("prompts_sha256")
            if not isinstance(snapshot_name, str) or not isinstance(
                expected_digest, str
            ):
                return None
            if Path(snapshot_name).name != snapshot_name:
                raise ValueError("prompt snapshot must be a sibling of the results file")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
                raise ValueError("prompt snapshot metadata has an invalid SHA-256 digest")

            snapshot_path = results_path.parent / snapshot_name
            try:
                descriptor = os.open(snapshot_path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as error:
                raise ValueError(
                    f"prompt snapshot is unavailable or unsafe: {snapshot_path}"
                ) from error
            try:
                snapshot_stat = os.fstat(descriptor)
                if not stat.S_ISREG(snapshot_stat.st_mode):
                    raise ValueError("prompt snapshot must be a regular file")
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    contents = handle.read()
            finally:
                os.close(descriptor)

            actual_digest = hashlib.sha256(contents).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(
                    f"prompt snapshot digest mismatch: expected {expected_digest}, "
                    f"got {actual_digest}"
                )

            report_directory = results_path.parent / "test-reports"
            report_directory.mkdir(parents=True, exist_ok=True)
            copy_descriptor, copy_path = tempfile.mkstemp(
                prefix=".verified-prompts-",
                suffix=".txt",
                dir=report_directory,
            )
            with os.fdopen(copy_descriptor, "wb") as copy_handle:
                copy_handle.write(contents)
            return Path(copy_path)
    return None


def results_metadata_declares_prompts(filepath: str | Path) -> bool:
    """Return whether a run metadata record says it used a prompts file."""
    with Path(filepath).open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not record.get("_meta"):
                return False
            if record.get("prompts_file") or record.get("prompts_snapshot"):
                return True
            command = record.get("test_command", "")
            if not isinstance(command, str):
                return False
            try:
                arguments = shlex.split(command)
            except ValueError as error:
                raise ValueError("legacy test command metadata is malformed") from error
            return any(
                argument == "--prompts" or argument.startswith("--prompts=")
                for argument in arguments
            )
    return False


def _json_string(value: str, parameter: str) -> str:
    parsed = json.loads(value)
    if not isinstance(parsed, str):
        raise ValueError(f"{parameter} must contain a JSON string")
    return parsed


def _parse_stop(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return [item.strip() for item in value.split(",")]
    return parsed if isinstance(parsed, list) else [str(parsed)]


def _parse_response_format(value: str):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _parse_parameter(line: str):
    if ":" not in line:
        return None
    key, raw_value = line.split(":", 1)
    key = key.strip()
    value = raw_value.strip()

    if key in INTEGER_PARAMETERS:
        return key, int(value)
    if key in FLOAT_PARAMETERS:
        return key, float(value)
    if key in BOOLEAN_PARAMETERS:
        return key, value.lower() == "true"
    if key == "stop":
        return key, _parse_stop(value)
    if key == "response_format":
        return key, _parse_response_format(value)
    if key in {"system", "developer", "instructions", "afm"}:
        return key, value
    if key in {"system_json", "developer_json", "instructions_json"}:
        target = key.removesuffix("_json")
        return target, _json_string(value, key)
    if key in JSON_PARAMETERS:
        return key, json.loads(value)
    if key == "media":
        return key, [item.strip() for item in value.split(",")]
    if key == "requires":
        return key, [item.strip() for item in value.split(",") if item.strip()]
    return None


def parse_prompts_file(filepath: str | Path) -> dict:
    config = {"defaults": {}, "all": [], "runs": []}
    model_sections: dict[str, dict] = {}
    current_section: str | None = None

    with Path(filepath).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"^\[(.+)\]$", line)
            if match:
                section_name = match.group(1).strip()
                if section_name == "all":
                    current_section = "all"
                    continue
                if section_name in model_sections:
                    raise ValueError(
                        f"line {line_number}: duplicate section [{section_name}]"
                    )
                current_section = section_name
                if section_name.startswith("@ "):
                    model_id = ""
                    label = section_name[2:].strip()
                elif " @ " in section_name:
                    model_id, label = section_name.split(" @ ", 1)
                    model_id = model_id.strip()
                    label = label.strip()
                else:
                    model_id = section_name
                    label = ""
                model_sections[section_name] = {
                    "model": model_id,
                    "label": label,
                    "prompts": [],
                    "params": {},
                    "afm": "",
                    "skip": False,
                }
                continue

            if current_section == "all":
                config["all"].append(line)
                continue

            parsed_parameter = _parse_parameter(line)
            if current_section is None:
                if parsed_parameter is not None:
                    key, value = parsed_parameter
                    if key == "afm":
                        config["defaults"]["afm"] = value
                    else:
                        config["defaults"][key] = value
                continue

            section = model_sections[current_section]
            if line == "skip":
                section["skip"] = True
            elif line.startswith("afm:"):
                section["afm"] = line.split(":", 1)[1].strip()
            elif parsed_parameter is not None:
                key, value = parsed_parameter
                section["params"][key] = value
            else:
                section["prompts"].append(line)

    config["runs"] = list(model_sections.values())
    return config


def expand_template_runs(config: dict, models: list[str]) -> dict:
    expanded_config = copy.deepcopy(config)
    expanded_runs = []
    for run in expanded_config["runs"]:
        if run["model"]:
            expanded_runs.append(run)
            continue
        for model in models:
            expanded = copy.deepcopy(run)
            expanded["model"] = model
            expanded_runs.append(expanded)
    expanded_config["runs"] = expanded_runs
    return expanded_config


def parse_ai_intent_specs(filepath: str | Path) -> dict[str, dict[str, list[str]]]:
    """Map model + label to the AI intent immediately preceding its section.

    Template sections use an empty model key and serve as the fallback for every
    expanded model. Named sections remain model-specific even when labels repeat.
    """
    specs: dict[str, dict[str, list[str]]] = {}
    comment_buffer: list[str] = []

    with Path(filepath).open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                comment_buffer.append(stripped)
                continue

            match = re.match(r"^\[(.+)\]$", stripped)
            if match:
                section_name = match.group(1).strip()
                model = ""
                label = ""
                if section_name.startswith("@ "):
                    label = section_name[2:].strip()
                elif " @ " in section_name:
                    model, label = section_name.split(" @ ", 1)
                    model = model.strip()
                    label = label.strip()
                if label:
                    intent_lines = [
                        line.replace("# AI:", "", 1).strip()
                        for line in comment_buffer
                        if "# AI:" in line
                    ]
                    if intent_lines:
                        specs.setdefault(model, {})[label] = intent_lines
                comment_buffer = []
                continue

            if (
                stripped
                and stripped != "skip"
                and _parse_parameter(stripped) is None
            ):
                comment_buffer = []

    return specs


def ai_intent_for_result(
    specs: dict[str, dict[str, list[str]]], model: str, label: str
) -> list[str]:
    if not label:
        return []
    return specs.get(model, {}).get(label) or specs.get("", {}).get(label) or []
