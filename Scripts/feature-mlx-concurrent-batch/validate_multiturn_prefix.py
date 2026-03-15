#!/usr/bin/env python3
"""Multi-turn prefix cache validation with long system prompts.

Simulates realistic agent/chatbot usage:
  - Shared long system prompts across turns (prefix cache hits)
  - Multi-turn conversations (growing context)
  - Concurrent users with different system prompts
  - Measures cached_tokens, pp, tg, TTFT

Usage:
    python3 validate_multiturn_prefix.py              # test B=1,2,4,8
    python3 validate_multiturn_prefix.py 1 4          # specific batch sizes
    python3 validate_multiturn_prefix.py --label "overlap+prefix" 1 2 4 8
"""
import asyncio, aiohttp, json, time, sys

URL = "http://localhost:9999/v1/chat/completions"
MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"

# ─── Long system prompts (simulate agent instructions) ────────────────────────

CODING_AGENT_SYSTEM = """You are an expert software engineer and code reviewer. You have deep expertise in:
- Python, TypeScript, Rust, Go, Swift, and C++
- System design and distributed systems architecture
- Database design (PostgreSQL, Redis, DynamoDB)
- Cloud infrastructure (AWS, GCP, Kubernetes)
- CI/CD pipelines, testing strategies, and deployment
- Security best practices and OWASP top 10

When reviewing code:
1. Check for correctness, edge cases, and error handling
2. Evaluate performance implications and algorithmic complexity
3. Assess security vulnerabilities (injection, XSS, CSRF, etc.)
4. Review naming conventions, code organization, and readability
5. Suggest specific improvements with code examples
6. Consider backward compatibility and migration paths

When writing code:
1. Follow the language's idiomatic patterns and conventions
2. Write comprehensive error handling with meaningful messages
3. Include type annotations and documentation
4. Consider testability and write unit tests when appropriate
5. Use appropriate design patterns without over-engineering
6. Optimize for readability first, performance when measured

You are working in a large monorepo with microservices architecture. The codebase uses:
- gRPC for service-to-service communication
- PostgreSQL with Prisma ORM for data persistence
- Redis for caching and pub/sub
- Docker containers orchestrated by Kubernetes
- GitHub Actions for CI/CD
- Terraform for infrastructure as code

Always respond with well-structured, production-quality code. Explain your reasoning for architectural decisions. If the user's request is ambiguous, ask for clarification rather than making assumptions."""

RESEARCH_AGENT_SYSTEM = """You are a research assistant with expertise in scientific methodology, data analysis, and academic writing. Your capabilities include:

- Literature review and synthesis across multiple domains
- Statistical analysis: hypothesis testing, regression, Bayesian methods
- Experimental design: RCTs, quasi-experimental designs, observational studies
- Data visualization best practices (following Tufte's principles)
- Academic writing: APA, Chicago, IEEE citation styles
- Critical evaluation of research methodology and claims

When analyzing research:
1. Evaluate study design, sample size, and statistical power
2. Check for common biases: selection, confirmation, survivorship, publication
3. Assess effect sizes and practical significance, not just p-values
4. Consider replicability and generalizability of findings
5. Look for conflicts of interest and funding sources
6. Compare findings with existing literature and meta-analyses

When writing research summaries:
1. Start with the key finding and its significance
2. Describe methodology with enough detail for evaluation
3. Present results with appropriate statistical notation
4. Discuss limitations honestly and thoroughly
5. Suggest future research directions
6. Use precise language — avoid hedging without reason, but qualify claims appropriately

Your knowledge spans: physics, chemistry, biology, medicine, computer science, economics, psychology, sociology, and environmental science. When uncertain about domain-specific details, say so explicitly rather than confabulating. Always distinguish between established consensus, emerging evidence, and speculation.

For mathematical and statistical content, use LaTeX notation when it aids clarity. For code-related analysis, prefer Python with numpy, scipy, pandas, and matplotlib."""

CREATIVE_AGENT_SYSTEM = """You are a creative writing assistant and storytelling expert. You understand:

- Narrative structure: three-act, hero's journey, in medias res, frame narratives
- Character development: motivation, arc, voice, internal conflict
- World-building: consistency, depth, sensory detail, cultural coherence
- Prose style: rhythm, pacing, tone, register, point of view
- Genre conventions: literary fiction, sci-fi, fantasy, thriller, horror, romance
- Dialogue: subtext, dialect, pacing, revealing character through speech
- Revision techniques: show don't tell, cutting adverbs, tightening prose

Your approach to creative work:
1. Understand the author's intent and target audience
2. Respect the established tone, style, and world rules
3. Create vivid, specific details rather than generic descriptions
4. Develop authentic character voices that serve the story
5. Balance exposition with action and dialogue
6. Use sensory language that engages multiple senses
7. Create tension through conflict, stakes, and pacing

When giving feedback on creative writing:
1. Start with what works well — identify strengths
2. Address structural issues before line-level edits
3. Explain WHY something works or doesn't, not just WHAT to change
4. Offer specific alternatives, not just "make it better"
5. Consider the author's style and intent, not your preferences
6. Be honest but constructive — vague praise helps no one

You have read extensively across world literature, from Homer to contemporary fiction. You understand literary analysis, cultural context, and the evolution of storytelling traditions across cultures and time periods."""

# ─── Conversation scenarios ───────────────────────────────────────────────────

CONVERSATIONS = [
    {
        "name": "coding-3turn",
        "system": CODING_AGENT_SYSTEM,
        "turns": [
            {
                "user": "Write a Python function to validate email addresses using regex. Include edge cases.",
                "expected": ["def ", "re.", "import re", "@"],
                "max_tokens": 4096,
            },
            {
                "user": "Now add unit tests for that function using pytest. Cover valid, invalid, and edge cases.",
                "expected": ["def test", "pytest", "assert"],
                "max_tokens": 4096,
            },
            {
                "user": "Good. Now refactor the validation to also check MX records using dnspython. Keep the regex as a fast pre-filter.",
                "expected": ["dns", "MX", "def "],
                "max_tokens": 4096,
            },
        ],
    },
    {
        "name": "research-3turn",
        "system": RESEARCH_AGENT_SYSTEM,
        "turns": [
            {
                "user": "Explain the difference between Type I and Type II errors in hypothesis testing, with examples from clinical trials.",
                "expected": ["type i", "type ii", "null hypothesis"],
                "max_tokens": 4096,
            },
            {
                "user": "How does sample size affect statistical power? Walk through a power analysis for a two-sample t-test.",
                "expected": ["power", "sample size", "effect"],
                "max_tokens": 4096,
            },
            {
                "user": "Now compare frequentist vs Bayesian approaches to the same clinical trial scenario. When should we prefer each?",
                "expected": ["bayesian", "frequentist", "prior"],
                "max_tokens": 4096,
            },
        ],
    },
    {
        "name": "creative-3turn",
        "system": CREATIVE_AGENT_SYSTEM,
        "turns": [
            {
                "user": "Write the opening scene of a noir detective story set in a rain-soaked Tokyo alley. First person, present tense.",
                "expected": ["rain", "tokyo"],
                "max_tokens": 4096,
            },
            {
                "user": "Continue the story. The detective finds a clue — a business card with a number that doesn't exist. Build tension.",
                "expected": ["card", "number"],
                "max_tokens": 4096,
            },
            {
                "user": "Now write the confrontation scene where the detective meets the antagonist. Use subtext in the dialogue — they both know more than they say.",
                "expected": ["said", "voice"],
                "max_tokens": 4096,
            },
        ],
    },
    {
        "name": "coding-longdecode",
        "system": CODING_AGENT_SYSTEM,
        "turns": [
            {
                "user": (
                    "Implement a complete LRU cache in Rust with the following requirements: "
                    "generic key/value types, O(1) get/put, configurable capacity, "
                    "thread-safe with fine-grained locking, iterator support, "
                    "and TTL-based expiration. Include comprehensive tests."
                ),
                "expected": ["struct", "impl", "fn ", "pub"],
                "max_tokens": 4096,
                "min_tokens": 500,
            },
        ],
    },
    {
        "name": "research-longdecode",
        "system": RESEARCH_AGENT_SYSTEM,
        "turns": [
            {
                "user": (
                    "Write a comprehensive literature review on transformer architecture "
                    "improvements from 2020-2025. Cover: efficient attention mechanisms "
                    "(linear attention, sparse attention, flash attention), mixture of experts, "
                    "state space models (Mamba), retrieval-augmented generation, "
                    "and constitutional AI. Include citations and compare approaches."
                ),
                "expected": ["attention", "transformer", "mamba"],
                "max_tokens": 4096,
                "min_tokens": 500,
            },
        ],
    },
]


# ─── Request sender ───────────────────────────────────────────────────────────

async def send_request(session, messages, max_tokens=1024):
    """Send streaming request, return full stats dict."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.3,
    }
    text = ""
    ttft = None
    usage = {}
    timings = {}
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
                if "usage" in chunk:
                    usage = chunk["usage"]
                if "timings" in chunk:
                    timings = chunk["timings"]
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or delta.get("reasoning_content", "")
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    text += content
            except Exception:
                pass

    elapsed = time.monotonic() - start

    # Extract cached tokens from usage
    cached = 0
    ptd = usage.get("prompt_tokens_details", {})
    if isinstance(ptd, dict):
        cached = ptd.get("cached_tokens", 0)

    return {
        "text": text,
        "wall_s": elapsed,
        "ttft": ttft or 0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cached_tokens": cached,
        "pp_tok_s": usage.get("prompt_tokens_per_second", 0),
        "tg_tok_s": usage.get("completion_tokens_per_second", 0),
        "prompt_time_s": usage.get("prompt_time", 0),
        "completion_time_s": usage.get("completion_time", 0),
    }


def check_response(text, expected):
    lower = text.lower()
    missing = [s for s in expected if s.lower() not in lower]
    is_garbage = len(text.strip()) < 2 or text.count('\ufffd') > 5
    return {"missing": missing, "ok": len(missing) == 0 and not is_garbage,
            "is_garbage": is_garbage}


# ─── Conversation runner ──────────────────────────────────────────────────────

async def run_conversation(session, conv):
    """Run a multi-turn conversation sequentially, return list of turn results."""
    system_msg = {"role": "system", "content": conv["system"]}
    messages = [system_msg]
    turn_results = []

    for i, turn in enumerate(conv["turns"]):
        messages.append({"role": "user", "content": turn["user"]})
        r = await send_request(session, messages, turn.get("max_tokens", 1024))

        check = check_response(r["text"], turn["expected"])
        r["turn"] = i + 1
        r["name"] = f"{conv['name']}/t{i+1}"
        r["ok"] = check["ok"]
        r["missing"] = check["missing"]
        r["min_tokens"] = turn.get("min_tokens", 0)

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": r["text"]})
        turn_results.append(r)

    return turn_results


async def run_batch(batch_size, conversations):
    """Run conversations at given concurrency."""
    passed = 0
    failed = 0
    all_results = []

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(conversations), batch_size):
            batch = conversations[batch_start:batch_start + batch_size]
            tasks = [run_conversation(session, c) for c in batch]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            for conv, outcome in zip(batch, outcomes):
                if isinstance(outcome, Exception):
                    failed += len(conv["turns"])
                    print(f"  FAIL  {conv['name']}: exception {outcome}")
                    continue

                for r in outcome:
                    all_results.append(r)
                    pt = r["prompt_tokens"]
                    ct = r["completion_tokens"]
                    cached = r["cached_tokens"]
                    pp = r["pp_tok_s"]
                    tg = r["tg_tok_s"]
                    ttft = r["ttft"]
                    cache_pct = (cached / pt * 100) if pt > 0 else 0

                    if r["ok"]:
                        passed += 1
                        too_short = r["min_tokens"] > 0 and ct < r["min_tokens"] * 0.5
                        if too_short:
                            failed += 1
                            passed -= 1
                            print(f"  FAIL  {r['name']:30s}  TOO SHORT ({ct} tok)")
                        else:
                            print(f"  OK    {r['name']:30s}  "
                                  f"pp={pt:5d} ({cached:4d} cached {cache_pct:4.0f}%) {pp:7.1f} t/s  "
                                  f"tg={ct:4d} tok {tg:6.1f} t/s  "
                                  f"TTFT={ttft:.2f}s  wall={r['wall_s']:.1f}s")
                    else:
                        failed += 1
                        if r.get("is_garbage"):
                            print(f"  FAIL  {r['name']:30s}  GARBAGE")
                        else:
                            print(f"  FAIL  {r['name']:30s}  missing {r['missing']}")

    return passed, failed, all_results


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    args = sys.argv[1:]
    label = ""
    if "--label" in args:
        idx = args.index("--label")
        label = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    batch_sizes = [int(x) for x in args] if args else [1, 2, 4, 8]
    total_passed = 0
    total_failed = 0
    all_batch_summaries = {}

    # Warm up: first request primes the prefix cache
    print("Warming up prefix cache...")
    async with aiohttp.ClientSession() as session:
        await send_request(session, [
            {"role": "system", "content": CODING_AGENT_SYSTEM},
            {"role": "user", "content": "Say hello."}
        ], max_tokens=10)
        await send_request(session, [
            {"role": "system", "content": RESEARCH_AGENT_SYSTEM},
            {"role": "user", "content": "Say hello."}
        ], max_tokens=10)
        await send_request(session, [
            {"role": "system", "content": CREATIVE_AGENT_SYSTEM},
            {"role": "user", "content": "Say hello."}
        ], max_tokens=10)
    print("Cache primed.\n")

    total_turns = sum(len(c["turns"]) for c in CONVERSATIONS)

    for bs in batch_sizes:
        print(f"{'='*120}")
        hdr = f"  B={bs} — {len(CONVERSATIONS)} conversations, {total_turns} turns, multi-turn + prefix cache"
        if label:
            hdr += f"  [{label}]"
        print(hdr)
        print(f"{'='*120}")

        p, f, results = await run_batch(bs, CONVERSATIONS)
        total_passed += p
        total_failed += f

        ok = [r for r in results if r.get("ok")]
        if ok:
            total_prompt = sum(r["prompt_tokens"] for r in ok)
            total_compl = sum(r["completion_tokens"] for r in ok)
            total_cached = sum(r["cached_tokens"] for r in ok)
            avg_pp = sum(r["pp_tok_s"] for r in ok) / len(ok)
            avg_tg = sum(r["tg_tok_s"] for r in ok) / len(ok)
            avg_ttft = sum(r["ttft"] for r in ok) / len(ok)
            cache_hit_pct = (total_cached / total_prompt * 100) if total_prompt > 0 else 0
            max_wall = max(r["wall_s"] for r in ok)

            # Separate turn 1 (cold) vs turn 2+ (warm cache)
            t1 = [r for r in ok if r.get("turn") == 1]
            t2plus = [r for r in ok if r.get("turn", 1) > 1]

            print(f"  {'─'*116}")
            print(f"  Totals: {total_prompt} prompt ({total_cached} cached, {cache_hit_pct:.0f}%) + "
                  f"{total_compl} completion = {total_prompt + total_compl} tokens")
            print(f"  Avg pp: {avg_pp:.1f} tok/s   Avg tg: {avg_tg:.1f} tok/s   "
                  f"Avg TTFT: {avg_ttft:.2f}s")
            if t1:
                t1_pp = sum(r["pp_tok_s"] for r in t1) / len(t1)
                t1_ttft = sum(r["ttft"] for r in t1) / len(t1)
                t1_cached = sum(r["cached_tokens"] for r in t1)
                t1_prompt = sum(r["prompt_tokens"] for r in t1)
                t1_cache_pct = (t1_cached / t1_prompt * 100) if t1_prompt > 0 else 0
                print(f"  Turn 1 (cold):  pp={t1_pp:.0f} tok/s  TTFT={t1_ttft:.2f}s  "
                      f"cache={t1_cache_pct:.0f}%")
            if t2plus:
                t2_pp = sum(r["pp_tok_s"] for r in t2plus) / len(t2plus)
                t2_ttft = sum(r["ttft"] for r in t2plus) / len(t2plus)
                t2_cached = sum(r["cached_tokens"] for r in t2plus)
                t2_prompt = sum(r["prompt_tokens"] for r in t2plus)
                t2_cache_pct = (t2_cached / t2_prompt * 100) if t2_prompt > 0 else 0
                print(f"  Turn 2+ (warm): pp={t2_pp:.0f} tok/s  TTFT={t2_ttft:.2f}s  "
                      f"cache={t2_cache_pct:.0f}%")

        print(f"  Result: {p}/{p+f} passed")
        print(f"{'='*120}")

        all_batch_summaries[bs] = {"passed": p, "failed": f, "results": ok if ok else []}
        await asyncio.sleep(1)

    # ─── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    title = f"  SUMMARY — {MODEL} | multi-turn + prefix cache"
    if label:
        title += f" | {label}"
    print(title)
    print(f"{'='*130}")
    print(f"  {'B':>3s}  {'Pass':>5s}  {'PP tok':>6s}  {'Cached':>6s}  {'Cache%':>6s}  "
          f"{'PP t/s':>7s}  {'TG tok':>6s}  {'TG t/s':>7s}  "
          f"{'TTFT':>6s}  {'T1 TTFT':>7s}  {'T2+ TTFT':>8s}")
    print(f"  {'───':>3s}  {'─────':>5s}  {'──────':>6s}  {'──────':>6s}  {'──────':>6s}  "
          f"{'───────':>7s}  {'──────':>6s}  {'───────':>7s}  "
          f"{'──────':>6s}  {'───────':>7s}  {'────────':>8s}")

    for bs in batch_sizes:
        br = all_batch_summaries.get(bs)
        if not br or not br["results"]:
            continue
        ok = br["results"]
        p_f = f"{br['passed']}/{br['passed']+br['failed']}"
        total_prompt = sum(r["prompt_tokens"] for r in ok)
        total_compl = sum(r["completion_tokens"] for r in ok)
        total_cached = sum(r["cached_tokens"] for r in ok)
        cache_pct = (total_cached / total_prompt * 100) if total_prompt > 0 else 0
        avg_pp = sum(r["pp_tok_s"] for r in ok) / len(ok)
        avg_tg = sum(r["tg_tok_s"] for r in ok) / len(ok)
        avg_ttft = sum(r["ttft"] for r in ok) / len(ok)

        t1 = [r for r in ok if r.get("turn") == 1]
        t2plus = [r for r in ok if r.get("turn", 1) > 1]
        t1_ttft = (sum(r["ttft"] for r in t1) / len(t1)) if t1 else 0
        t2_ttft = (sum(r["ttft"] for r in t2plus) / len(t2plus)) if t2plus else 0

        print(f"  {bs:3d}  {p_f:>5s}  {total_prompt:6d}  {total_cached:6d}  {cache_pct:5.0f}%  "
              f"{avg_pp:7.1f}  {total_compl:6d}  {avg_tg:7.1f}  "
              f"{avg_ttft:5.2f}s  {t1_ttft:6.2f}s  {t2_ttft:7.2f}s")

    print(f"{'='*130}")
    print(f"  TOTAL: {total_passed}/{total_passed+total_failed} passed across {len(batch_sizes)} batch sizes")
    print(f"{'='*130}")


asyncio.run(main())
