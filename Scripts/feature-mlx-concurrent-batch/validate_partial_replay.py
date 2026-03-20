#!/usr/bin/env python3
"""Validate partial prefix-cache replay against a cold replay baseline.

For each suffix length:
1. Send a warmup prompt containing only the shared prefix.
2. Send the full prompt once to exercise partial cache restore.
3. Send the same full prompt again; in normal safe mode this exact full-prefix
   hit is bypassed, so this acts as a cold baseline for the same full prompt.

The cached and cold outputs should match under deterministic settings if
partial replay is restoring state correctly.
"""

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass


def send(url: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "stream": False,
        "temperature": 0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def extract(result: dict) -> tuple[int | None, int | None, str]:
    usage = result.get("usage", {})
    ptd = usage.get("prompt_tokens_details") or {}
    return (
        usage.get("prompt_tokens"),
        ptd.get("cached_tokens"),
        result.get("choices", [{}])[0].get("message", {}).get("content", ""),
    )


@dataclass
class SweepCase:
    suffix_units: int
    warm_prompt: str
    full_prompt: str


def build_case(tag: str, suffix_units: int) -> SweepCase:
    prefix = (
        f"Cache validation tag {tag}. "
        "You are running a deterministic replay validation. "
        "Read the whole prompt carefully before answering. "
        "The final instruction at the end overrides all earlier examples. "
        "Do not explain. Do not add punctuation. "
    )
    suffix_filler = " ".join(
        f"FILLER{i}-{tag}" for i in range(1, suffix_units + 1)
    )
    final = f" Final answer: reply with exactly TOKEN-{tag}."
    warm_prompt = prefix
    full_prompt = prefix + suffix_filler + final
    return SweepCase(suffix_units=suffix_units, warm_prompt=warm_prompt, full_prompt=full_prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9999/v1/chat/completions")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-35B-A3B-4bit")
    parser.add_argument(
        "--suffix-units",
        default="1,2,4,8,16,32",
        help="Comma-separated suffix filler unit counts to test",
    )
    args = parser.parse_args()

    failures = 0
    for idx, raw in enumerate(args.suffix_units.split(","), start=1):
        suffix_units = int(raw.strip())
        tag = f"PARTIAL-{idx}-{suffix_units}"
        case = build_case(tag, suffix_units)

        warm = send(args.url, args.model, case.warm_prompt)
        warm_prompt_tokens, warm_cached_tokens, warm_content = extract(warm)

        cached = send(args.url, args.model, case.full_prompt)
        cached_prompt_tokens, cached_cached_tokens, cached_content = extract(cached)

        cold = send(args.url, args.model, case.full_prompt)
        cold_prompt_tokens, cold_cached_tokens, cold_content = extract(cold)

        ok = cached_content == cold_content
        if not ok:
            failures += 1

        record = {
            "suffix_units": suffix_units,
            "warm": {
                "prompt_tokens": warm_prompt_tokens,
                "cached_tokens": warm_cached_tokens,
                "content": warm_content,
            },
            "cached_partial": {
                "prompt_tokens": cached_prompt_tokens,
                "cached_tokens": cached_cached_tokens,
                "content": cached_content,
            },
            "cold_replay": {
                "prompt_tokens": cold_prompt_tokens,
                "cached_tokens": cold_cached_tokens,
                "content": cold_content,
            },
            "matches": ok,
        }
        print(json.dumps(record))
        sys.stdout.flush()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
