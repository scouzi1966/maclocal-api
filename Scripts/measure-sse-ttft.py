#!/usr/bin/env python3
"""Measure the first visible/reasoning token while draining an active SSE stream."""
import argparse
import json
import sys
import time


def measure(lines, started_ns, clock=time.monotonic_ns):
    first_token_ms = None
    for line in lines:
        # Timestamp receipt before JSON parsing; role-only and empty deltas do
        # not constitute a token. Continue draining to preserve curl failures
        # and avoid creating an artificial broken pipe after the first token.
        if first_token_ms is not None:
            continue
        received_ns = clock()
        if not line.startswith('data:'):
            continue
        try:
            event = json.loads(line[len('data:'):].strip())
            for choice in event.get('choices', []):
                delta = choice.get('delta') or {}
                if any(isinstance(delta.get(key), str) and delta[key]
                       for key in ('content', 'reasoning_content')):
                    first_token_ms = max(0, (received_ns - started_ns) // 1_000_000)
                    break
        except (ValueError, AttributeError, TypeError):
            continue
    return first_token_ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-ns', type=int, required=True)
    args = parser.parse_args()
    result = measure(sys.stdin, args.start_ns)
    print(result if result is not None else 'no content or reasoning token received')
