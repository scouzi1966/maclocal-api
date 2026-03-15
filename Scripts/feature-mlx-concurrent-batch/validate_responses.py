#!/usr/bin/env python3
"""Validate that batched generation produces coherent, correct responses.

Sends known-answer questions at various batch sizes and checks responses
contain expected content. Catches broken masks, corrupt KV cache, or
garbage output that throughput-only tests would miss.

Usage:
    python3 validate_responses.py              # test B=1,2,4,8
    python3 validate_responses.py 1 4          # test specific batch sizes

Prerequisites:
    pip install aiohttp
    Server running on port 9999
"""
import asyncio, aiohttp, json, time, sys, re

URL = "http://localhost:9999/v1/chat/completions"

# Each entry: (prompt, expected_substrings, description)
# At least one substring must appear (case-insensitive) for the test to pass.
VALIDATIONS = [
    (
        "What is 2+2? Answer with just the number.",
        ["4"],
        "basic arithmetic"
    ),
    (
        "What is the capital of France? Answer in one word.",
        ["paris"],
        "capital of France"
    ),
    (
        "What is the chemical symbol for water? Answer in one word.",
        ["h2o"],
        "chemical formula"
    ),
    (
        "Name the largest planet in our solar system. Answer in one word.",
        ["jupiter"],
        "largest planet"
    ),
    (
        "What color do you get when you mix red and blue? Answer in one word.",
        ["purple", "violet"],
        "color mixing"
    ),
    (
        "In what year did World War II end? Answer with just the year.",
        ["1945"],
        "WWII end year"
    ),
    (
        "What is the square root of 144? Answer with just the number.",
        ["12"],
        "square root"
    ),
    (
        "Who wrote Romeo and Juliet? Answer with just the name.",
        ["shakespeare"],
        "Shakespeare authorship"
    ),
]


async def send_request(session, prompt, max_tokens=200):
    """Send a streaming request, return full text."""
    payload = {
        "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.3,
    }
    text = ""
    start = time.monotonic()

    async with session.post(URL, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or delta.get("reasoning_content", "")
                if content:
                    text += content
            except:
                pass

    elapsed = time.monotonic() - start
    return text, elapsed


def check_response(text, expected_substrings):
    """Check if response contains at least one expected substring."""
    lower = text.lower()
    for sub in expected_substrings:
        if sub.lower() in lower:
            return True, sub
    return False, None


async def run_validation(batch_size):
    """Run all validations at the given concurrency level."""
    print(f"\n{'='*70}")
    print(f"  Validating B={batch_size}")
    print(f"{'='*70}")

    passed = 0
    failed = 0
    errors = []

    async with aiohttp.ClientSession() as session:
        # Send batch_size requests concurrently, cycling through validations
        for batch_start in range(0, len(VALIDATIONS), batch_size):
            batch = VALIDATIONS[batch_start:batch_start + batch_size]
            tasks = [send_request(session, v[0]) for v in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (prompt, expected, desc), result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    errors.append((desc, f"EXCEPTION: {result}"))
                    print(f"  FAIL  {desc}: exception {result}")
                    continue

                text, elapsed = result
                ok, matched = check_response(text, expected)
                # Also check for obvious garbage
                is_garbage = len(text.strip()) < 2 or text.count('\ufffd') > 5

                if is_garbage:
                    failed += 1
                    preview = text[:80].replace('\n', ' ')
                    errors.append((desc, f"GARBAGE: '{preview}'"))
                    print(f"  FAIL  {desc}: garbage output '{preview}...'")
                elif ok:
                    passed += 1
                    preview = text[:60].replace('\n', ' ')
                    print(f"  OK    {desc}: found '{matched}' ({elapsed:.1f}s) | {preview}...")
                else:
                    failed += 1
                    preview = text[:100].replace('\n', ' ')
                    errors.append((desc, f"MISSING {expected}: '{preview}'"))
                    print(f"  FAIL  {desc}: expected {expected} in '{preview}...'")

    print(f"{'='*70}")
    print(f"  B={batch_size}: {passed}/{passed+failed} passed")
    if errors:
        print(f"  Failures:")
        for desc, msg in errors:
            print(f"    - {desc}: {msg}")
    print(f"{'='*70}")

    return passed, failed


async def main():
    batch_sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 4, 8]

    total_passed = 0
    total_failed = 0

    for bs in batch_sizes:
        p, f = await run_validation(bs)
        total_passed += p
        total_failed += f
        await asyncio.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"  TOTAL: {total_passed}/{total_passed+total_failed} passed across {len(batch_sizes)} batch sizes")
    if total_failed > 0:
        print(f"  *** {total_failed} FAILURES — output may be corrupt ***")
    else:
        print(f"  All responses coherent and correct.")
    print(f"{'='*70}")

asyncio.run(main())
