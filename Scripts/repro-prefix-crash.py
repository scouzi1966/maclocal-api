#!/usr/bin/env python3
"""
Minimal reproducer for prefix cache broadcast_shapes crash (Issue #47).

Strategy: Build up diverse radix cache entries as fast as possible by
sending rapid sequential requests with shared system prompts but
different user messages and tool configurations. The crash happens
when cache has many entries and a restore hits a shape mismatch
between Mamba and KVCacheSimple layers.

Usage: python3 Scripts/repro-prefix-crash.py [--port 9999] [--max-tokens 128]
"""

import asyncio
import aiohttp
import json
import time
import random
import argparse
from datetime import datetime

BASE_URL = "http://127.0.0.1:9999"
MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"

# Shared system prompts — creates prefix overlap for cache hits
SYSTEMS = [
    "You are a helpful assistant. Be concise.",
    "You are a helpful engineering assistant. Be precise and technical.",
    "You are a data engineering assistant. Be precise with numbers.",
    "You are a security-focused assistant. Consider threats and best practices.",
]

# Tools — some requests have them, some don't (different tokenization)
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Do math", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search docs", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

# Short diverse user messages — keep output short, maximize request throughput
QUESTIONS = [
    "What is 2+2?",
    "Name 3 planets.",
    "What color is the sky?",
    "Define recursion in one sentence.",
    "What's the capital of Japan?",
    "Is Python compiled or interpreted?",
    "What does HTTP stand for?",
    "What is a hash table?",
    "Name a sorting algorithm.",
    "What is TCP/IP?",
    "What is REST?",
    "What is a mutex?",
    "What is garbage collection?",
    "What is SQL injection?",
    "What is OAuth?",
    "What is a load balancer?",
    "What does ACID mean in databases?",
    "What is DNS?",
    "What is a container?",
    "What is CI/CD?",
    "How does TLS work?",
    "What is a microservice?",
    "What is eventual consistency?",
    "What is a B-tree?",
    "What is MapReduce?",
    "Define idempotency.",
    "What is CORS?",
    "What is a webhook?",
    "What is gRPC?",
    "What is sharding?",
]

# Multi-turn prefixes — reuse same system+first exchange, vary second turn
FIRST_EXCHANGES = [
    ("What's a linked list?", "A linked list is a data structure where each element points to the next."),
    ("What's a binary tree?", "A binary tree is a tree data structure where each node has at most two children."),
    ("Explain HTTP methods.", "The main HTTP methods are GET, POST, PUT, DELETE, and PATCH."),
    ("What's a deadlock?", "A deadlock occurs when two or more processes are waiting for each other to release resources."),
]

SECOND_TURNS = [
    "Give me an example.",
    "Show me the code.",
    "What's the time complexity?",
    "How is this used in production?",
    "Compare it to the alternative.",
    "What are the edge cases?",
    "Explain it differently.",
    "When should I NOT use this?",
]


async def send_request(http, messages, tools=None, streaming=False, max_tokens=128):
    """Send one request, return (success, elapsed_ms, info)."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": random.choice([0.0, 0.3, 0.7]),
    }
    if tools:
        body["tools"] = tools
    if streaming:
        body["stream"] = True

    t0 = time.time()
    try:
        async with http.post(f"{BASE_URL}/v1/chat/completions", json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                return False, int((time.time()-t0)*1000), f"HTTP {resp.status}: {text[:100]}"

            if streaming:
                async for line in resp.content:
                    pass  # drain stream
                return True, int((time.time()-t0)*1000), "stream"
            else:
                data = await resp.json()
                usage = data.get("usage", {})
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                return True, int((time.time()-t0)*1000), f"cached={cached}"
    except Exception as e:
        return False, int((time.time()-t0)*1000), str(e)[:100]


async def main():
    global BASE_URL, MODEL

    parser = argparse.ArgumentParser(description="Reproduce prefix cache crash")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--max-tokens", type=int, default=128, help="Keep low for faster cycling")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--max-requests", type=int, default=500)
    args = parser.parse_args()

    BASE_URL = f"http://127.0.0.1:{args.port}"
    MODEL = args.model

    print(f"Prefix cache crash reproducer")
    print(f"  Server: {BASE_URL}")
    print(f"  max_tokens: {args.max_tokens} (low = faster cycling)")
    print(f"  Target: build diverse cache entries, trigger shape mismatch")
    print()

    # Verify server
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/v1/models") as r:
            if r.status != 200:
                print(f"Server not reachable")
                return
            print("Server OK")

    ok_count = 0
    err_count = 0
    start = time.time()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
        for i in range(1, args.max_requests + 1):
            # Randomly pick a request pattern
            pattern = random.choice([
                "simple", "simple", "simple",  # 30% simple
                "tools", "tools",               # 20% with tools
                "multi-turn",                   # 10% multi-turn
                "multi-turn-tools",             # 10% multi-turn with tools
                "stream", "stream",             # 20% streaming
                "stream-tools",                 # 10% streaming with tools
            ])

            system = random.choice(SYSTEMS)
            messages = [{"role": "system", "content": system}]

            tools = None
            streaming = False

            if "tools" in pattern:
                n = random.randint(1, 3)
                tools = random.sample(TOOLS, n)

            if "stream" in pattern:
                streaming = True

            if "multi-turn" in pattern:
                first_q, first_a = random.choice(FIRST_EXCHANGES)
                messages.append({"role": "user", "content": first_q})
                messages.append({"role": "assistant", "content": first_a})
                messages.append({"role": "user", "content": random.choice(SECOND_TURNS)})
            else:
                messages.append({"role": "user", "content": random.choice(QUESTIONS)})

            success, ms, info = await send_request(http, messages, tools, streaming, args.max_tokens)

            if success:
                ok_count += 1
                marker = "+"
            else:
                err_count += 1
                marker = "X"
                if "Cannot connect" in info or "Server disconnected" in info:
                    elapsed = time.time() - start
                    print(f"\n  SERVER CRASHED after {ok_count} successful requests ({elapsed:.1f}s)")
                    print(f"  Last error: {info}")
                    print(f"  Total: {ok_count} OK, {err_count} errors")
                    return

            if i % 5 == 0 or not success:
                elapsed = time.time() - start
                rps = ok_count / elapsed if elapsed > 0 else 0
                print(f"  [{i:4d}] {marker} {pattern:20s} {ms:6d}ms {info:30s} | ok={ok_count} err={err_count} rps={rps:.2f}")

    elapsed = time.time() - start
    print(f"\n  Completed {args.max_requests} requests without crash!")
    print(f"  {ok_count} OK, {err_count} errors in {elapsed:.1f}s ({ok_count/elapsed:.2f} rps)")


if __name__ == "__main__":
    asyncio.run(main())
