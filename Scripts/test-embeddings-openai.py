#!/usr/bin/env python3
import json
import os
import sys

try:
    from openai import OpenAI
except Exception as exc:
    print(f"OpenAI SDK not available: {exc}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    base_url = os.environ.get("AFM_BASE_URL", "http://127.0.0.1:9998/v1")
    model = os.environ.get("AFM_EMBED_MODEL", "apple-nl-contextual-en")

    client = OpenAI(api_key="not-needed", base_url=base_url)
    result = client.embeddings.create(
        model=model,
        input=["hello world", "goodbye world"],
        dimensions=64,
    )

    payload = {
        "model": result.model,
        "count": len(result.data),
        "dimensions": len(result.data[0].embedding),
        "prompt_tokens": result.usage.prompt_tokens,
        "total_tokens": result.usage.total_tokens,
    }
    print(json.dumps(payload))

    assert payload["count"] == 2
    assert payload["dimensions"] == 64
    assert payload["prompt_tokens"] > 0
    assert payload["total_tokens"] >= payload["prompt_tokens"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
