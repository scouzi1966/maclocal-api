"""Deterministic response contracts and pluggable semantic review."""
import asyncio
import json
import os
import shlex


REPLACEMENT_CHAR = "\ufffd"
MAX_REPLACEMENT_CHARS = 5
DEFAULT_SEMANTIC_JUDGE_TIMEOUT_S = 30.0
DEFAULT_SEMANTIC_JUDGE_MAX_OUTPUT_BYTES = 1024 * 1024


def observe_lexical(text, expected):
    """Return lexical evidence without treating prose markers as contracts."""
    lower = text.lower()
    missing = [marker for marker in expected if marker.lower() not in lower]
    return {
        "classification": "review_evidence",
        "expected": list(expected),
        "missing": missing,
        "observed": [marker for marker in expected if marker.lower() in lower],
        "ok": not missing,
    }


def evaluate_integrity_contract(
    response,
    min_tokens=0,
    require_visible_text=True,
    require_finish_reason=True,
    require_done=True,
):
    failures = []
    visible = response.get("visible_text", "")
    completion_tokens = int(response.get("completion_tokens") or 0)
    replacement_count = visible.count(REPLACEMENT_CHAR)

    if require_visible_text and len(visible.strip()) < 2:
        failures.append("empty_or_near_empty_visible_text")
    if replacement_count > MAX_REPLACEMENT_CHARS:
        failures.append("excess_replacement_characters")
    if min_tokens and completion_tokens < min_tokens:
        failures.append("completion_below_minimum")
    if require_finish_reason and not response.get("finish_reason"):
        failures.append("missing_finish_reason")
    if require_done and not response.get("done_observed"):
        failures.append("missing_sse_done")
    if int(response.get("parse_error_count") or 0):
        failures.append("sender_parse_error")

    return {
        "classification": "deterministic",
        "ok": not failures,
        "failures": failures,
        "completion_tokens": completion_tokens,
        "min_tokens": min_tokens,
        "replacement_count": replacement_count,
    }


def evaluate_cache_contract(response, contract=None):
    """Evaluate only explicitly requested cache accounting boundaries."""
    contract = contract or {}
    failures = []
    prompt_tokens = int(response.get("prompt_tokens") or 0)
    cached_tokens = int(response.get("cached_tokens") or 0)

    expected_prompt = contract.get("expected_prompt_tokens")
    minimum_cached = contract.get("minimum_cached_tokens")
    maximum_cached = contract.get("maximum_cached_tokens")
    if expected_prompt is not None and prompt_tokens != int(expected_prompt):
        failures.append("unexpected_prompt_token_count")
    if minimum_cached is not None and cached_tokens < int(minimum_cached):
        failures.append("cached_tokens_below_minimum")
    if maximum_cached is not None and cached_tokens > int(maximum_cached):
        failures.append("cached_tokens_above_maximum")

    return {
        "classification": "deterministic_opt_in",
        "ok": not failures,
        "failures": failures,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
    }


def _normalise_semantic_result(payload, requirements):
    judged = payload.get("requirements")
    if not isinstance(judged, list):
        raise ValueError("semantic judge response lacks a requirements list")

    expected_ids = {requirement["id"] for requirement in requirements}
    judged_ids = set()
    normalised = []
    for item in judged:
        if not isinstance(item, dict):
            raise ValueError("semantic judge requirement lacks id")
        requirement_id = item.get("id")
        if not isinstance(requirement_id, str) or not requirement_id:
            raise ValueError("semantic judge requirement lacks id")
        if requirement_id in judged_ids:
            raise ValueError(f"semantic judge duplicated requirement: {requirement_id}")
        if type(item.get("passed")) is not bool:
            raise ValueError(f"semantic judge passed flag is not boolean: {requirement_id}")
        evidence = item.get("evidence", "")
        if evidence is None:
            evidence = ""
        if not isinstance(evidence, str):
            raise ValueError(f"semantic judge evidence is not text: {requirement_id}")
        judged_ids.add(requirement_id)
        normalised.append({
            "id": requirement_id,
            "passed": item["passed"],
            "evidence": evidence[:2000],
        })

    missing_ids = sorted(expected_ids - judged_ids)
    if missing_ids:
        raise ValueError(f"semantic judge omitted requirements: {', '.join(missing_ids)}")

    return {
        "classification": "semantic_review",
        "status": "evaluated",
        "ok": all(item["passed"] for item in normalised),
        "requirements": normalised,
        "summary": str(payload.get("summary", ""))[:4000],
    }


def _positive_environment_number(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


async def _run_semantic_judge(command, payload, timeout_s, max_output_bytes):
    process = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def send_input():
        process.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

    async def read_limited(stream):
        return await stream.read(max_output_bytes + 1)

    try:
        stdout, stderr, _, _ = await asyncio.wait_for(
            asyncio.gather(
                read_limited(process.stdout),
                read_limited(process.stderr),
                send_input(),
                process.wait(),
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise
    except Exception:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise

    if len(stdout) > max_output_bytes:
        raise ValueError("semantic judge stdout exceeded the output limit")
    if len(stderr) > max_output_bytes:
        raise ValueError("semantic judge stderr exceeded the output limit")
    if process.returncode != 0:
        raise ValueError(
            f"semantic judge exited {process.returncode}: "
            f"{stderr.decode('utf-8', errors='replace')[:1000]}"
        )
    return stdout.decode("utf-8")


async def evaluate_semantic_contract(transcript, response, requirements):
    """Run an operator-selected semantic judge; never affect deterministic score."""
    command = os.environ.get("AFM_SEMANTIC_JUDGE_COMMAND", "").strip()
    if not command:
        return {
            "classification": "semantic_review",
            "status": "not_evaluated",
            "ok": None,
            "requirements": requirements,
        }
    if not requirements:
        return {
            "classification": "semantic_review",
            "status": "no_requirements",
            "ok": None,
            "requirements": [],
        }

    payload = {
        "messages": transcript,
        "prompt": transcript[-1]["content"] if transcript else "",
        "response": response.get("visible_text", ""),
        "requirements": requirements,
    }
    try:
        timeout_s = _positive_environment_number(
            "AFM_SEMANTIC_JUDGE_TIMEOUT_S",
            DEFAULT_SEMANTIC_JUDGE_TIMEOUT_S,
        )
        max_output_bytes = int(_positive_environment_number(
            "AFM_SEMANTIC_JUDGE_MAX_OUTPUT_BYTES",
            DEFAULT_SEMANTIC_JUDGE_MAX_OUTPUT_BYTES,
        ))
        stdout = await _run_semantic_judge(command, payload, timeout_s, max_output_bytes)
        return _normalise_semantic_result(
            json.loads(stdout), requirements
        )
    except Exception as error:
        return {
            "classification": "semantic_review",
            "status": "error",
            "ok": None,
            "requirements": requirements,
            "error": f"{type(error).__name__}: {error}",
        }
