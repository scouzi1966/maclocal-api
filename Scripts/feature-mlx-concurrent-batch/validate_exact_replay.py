#!/usr/bin/env python3
"""Validate exact prompt replay with prefix cache.

Runs the same prompt repeatedly and reports:
- prompt_tokens
- cached_tokens
- content equality against the cold baseline

Use with the default safety guard, then optionally restart AFM with:
  AFM_PREFIX_CACHE_ALLOW_UNSAFE_EXACT_REPLAY=1
to compare behavior when exact replay bypass is disabled.
"""

import argparse
import json
import sys
import urllib.request


def send(url: str, model: str, prompt: str, stream: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "stream": stream,
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
        if not stream:
            return json.load(resp)

        text = ""
        usage = {}
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "usage" in chunk:
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                text += delta.get("content", "") or delta.get("reasoning_content", "")
        return {
            "choices": [{"message": {"content": text}}],
            "usage": usage,
        }


def extract(result: dict) -> tuple[int | None, int | None, str]:
    usage = result.get("usage", {})
    ptd = usage.get("prompt_tokens_details") or {}
    return (
        usage.get("prompt_tokens"),
        ptd.get("cached_tokens"),
        result.get("choices", [{}])[0].get("message", {}).get("content", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9999/v1/chat/completions")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-35B-A3B-4bit")
    parser.add_argument("--prompt", default="Reply with exactly BASELINE-ALBATROSS and nothing else.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    baseline = None
    for idx in range(1, args.runs + 1):
        result = send(args.url, args.model, args.prompt, args.stream)
        prompt_tokens, cached_tokens, content = extract(result)
        if baseline is None:
            baseline = content
        record = {
            "run": idx,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "content": content,
            "matches_baseline": content == baseline,
        }
        print(json.dumps(record))
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
