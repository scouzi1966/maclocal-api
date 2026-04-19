#!/usr/bin/env python3
"""
Concurrent load driver for AFM — produces a time-series metrics trace
suitable for rendering into the concurrent-throughput demo video.

Ramps virtual users from 0 to TARGET_USERS over RAMP_S seconds, holds
at TARGET_USERS for HOLD_S seconds, then tears down. Each virtual user
loops: pick a random prompt from the configured pool, open a streaming
chat completion, count completion tokens as they arrive, close, repeat.

Metrics are sampled every SAMPLE_MS milliseconds and written as JSONL:

    {"t": 0.25, "active": 3, "agg_tps": 45.2, "completed": 0, "inflight": 3}

Where:
    t         — seconds since start
    active    — virtual users attempting work right now (ramp target)
    agg_tps   — aggregate completion-token rate, window-averaged over
                --smoothing-window-s (default 2s), reliable for peak/sustained
    completed — cumulative completed requests
    inflight  — requests currently streaming (may lag below `active`
                when the server queues; gap = queued)

The driver uses only stdlib + aiohttp. Run directly or via the wrapper
script Scripts/demo-concurrent-throughput.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Run: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_METRIC_GEN_RX    = re.compile(r'^afm:generation_tokens_total\{[^}]*\}\s+(\d+)')
_METRIC_PP_RX     = re.compile(r'^afm:prompt_tokens_total\{[^}]*\}\s+(\d+)')
_METRIC_RUN_RX    = re.compile(r'^afm:num_requests_running\{[^}]*\}\s+([\d.]+)')
_METRIC_WAIT_RX   = re.compile(r'^afm:num_requests_waiting\{[^}]*\}\s+([\d.]+)')
_METRIC_CHIT_RX   = re.compile(r'^afm:radix_cache_hits_total\{[^}]*\}\s+(\d+)')
_METRIC_CMISS_RX  = re.compile(r'^afm:radix_cache_misses_total\{[^}]*\}\s+(\d+)')


async def poll_metrics(session: aiohttp.ClientSession, state: "LoadState", cfg: dict[str, Any]) -> None:
    """
    Polls /metrics every sample_ms and publishes ground-truth numbers
    onto LoadState. If the endpoint returns 404 (old AFM build without
    the metrics route), the poller exits cleanly and the sampler falls
    back to the SSE-event-count proxy.
    """
    base = cfg["endpoint"].rsplit("/v1/", 1)[0]
    url = f"{base}/metrics"
    interval = max(0.1, cfg["sample_ms"] / 1000.0)
    # Window-averaged tps. Single-sample deltas are spiky because each batched
    # decode step emits B tokens at once; a step straddling a polling boundary
    # produces a double-counted peak one sample and an under-count the next.
    # Window over --smoothing-window-s matches the proxy path behavior and
    # makes agg_tps reliable regardless of which source feeds it.
    tps_window_s = max(1.0, float(cfg.get("smoothing_window_s", 2.0)))
    samples: deque[tuple[float, int, int]] = deque()

    # Initial probe
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as r:
            if r.status == 404:
                state.server_metrics_available = False
                sys.stderr.write("[poll] /metrics endpoint not available on this AFM build — "
                                 "falling back to SSE-event proxy for tok/sec\n")
                return
            if r.status != 200:
                state.server_metrics_available = False
                return
            _ = await r.text()
    except Exception:
        state.server_metrics_available = False
        return

    state.server_metrics_available = True

    while not state.stop.is_set():
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as r:
                if r.status != 200:
                    await asyncio.sleep(interval)
                    continue
                text = await r.text()
        except Exception:
            await asyncio.sleep(interval)
            continue

        gen = pp = 0
        run = wait = 0.0
        chit = cmiss = 0
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = _METRIC_GEN_RX.match(line)
            if m: gen = int(m.group(1)); continue
            m = _METRIC_PP_RX.match(line)
            if m: pp = int(m.group(1)); continue
            m = _METRIC_RUN_RX.match(line)
            if m: run = float(m.group(1)); continue
            m = _METRIC_WAIT_RX.match(line)
            if m: wait = float(m.group(1)); continue
            m = _METRIC_CHIT_RX.match(line)
            if m: chit = int(m.group(1)); continue
            m = _METRIC_CMISS_RX.match(line)
            if m: cmiss = int(m.group(1)); continue

        now = time.monotonic()
        samples.append((now, gen, pp))
        cutoff = now - tps_window_s
        while len(samples) > 2 and samples[0][0] < cutoff:
            samples.popleft()
        if len(samples) >= 2:
            t0, g0, p0 = samples[0]
            dt = now - t0
            if dt > 0:
                state.server_gen_tps     = (gen - g0) / dt
                state.server_prefill_tps = (pp - p0) / dt
        state.server_gen_total     = gen
        state.server_prefill_total = pp
        state.server_inflight      = int(run)
        state.server_waiting       = int(wait)
        state.server_cache_hits    = chit
        state.server_cache_misses  = cmiss

        await asyncio.sleep(interval)


class LoadState:
    """
    Mutable metrics shared between virtual users and the sampler.

    All mutations happen on the asyncio event loop thread, so plain int
    increments (inflight += 1 etc) are atomic. We do NOT use an
    asyncio.Lock because it becomes a serialization point under heavy
    load (200 users each acquiring the lock on every SSE line can
    starve the metrics sampler, which is exactly what killed the last
    run at t=2s). For the same reason recent_tokens is a raw deque;
    append/popleft are both O(1) and atomic on CPython.
    """

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.wall_clock_start: float = 0.0
        self.active_target: int = 0      # virtual users we currently want running
        self.inflight: int = 0           # requests actively streaming
        self.completed: int = 0          # finished requests (success or error)
        self.errors: int = 0
        # Sliding window of (timestamp, tokens_in_chunk) within the last window
        self.recent_tokens: deque[tuple[float, int]] = deque()
        self.stop = asyncio.Event()
        # Per-request capture: async queue so virtual users enqueue completed
        # records and a single writer drains them to JSONL.
        self.request_queue: asyncio.Queue = asyncio.Queue()

        # Ground-truth metrics polled from server /metrics endpoint.
        # When server_metrics_available is True, the sampler uses these
        # instead of the SSE-event-count proxy.
        self.server_metrics_available: bool = False
        self.server_gen_total: int = 0
        self.server_prefill_total: int = 0
        self.server_gen_tps: float = 0.0
        self.server_prefill_tps: float = 0.0
        self.server_inflight: int = 0
        self.server_waiting: int = 0
        self.server_cache_hits: int = 0
        self.server_cache_misses: int = 0


# ---------------------------------------------------------------------------
# Virtual user
# ---------------------------------------------------------------------------

async def virtual_user(
    user_id: int,
    session: aiohttp.ClientSession,
    state: LoadState,
    cfg: dict[str, Any],
    exit_event: asyncio.Event | None = None,
) -> None:
    """
    Loops until state.stop is set OR exit_event is set. Each iteration:
      1. Generates a correlation ID and sends it in X-Request-ID (Vapor
         logs this verbatim on the server side so the driver's records
         can be cross-checked against the server log).
      2. Streams one chat completion, accumulating content and
         reasoning_content separately from SSE deltas, plus token counts
         for the live aggregate-throughput sliding window.
      3. Extracts the chatcmpl ID, finish_reason, and usage from the
         final SSE chunk.
      4. Enqueues one per-request record on state.request_queue for the
         writer task to persist to out/requests.jsonl.

    max_tokens is left long (default 3000) so most responses finish on
    natural EOS rather than being hard-truncated — this also spreads
    completion times across a much wider window than a fixed ceiling,
    which keeps the batch scheduler pipeline full and avoids the
    mass-completion dips we saw at max_tokens=192.
    """
    prompts: list[str] = cfg["prompts"]
    system: str | None = cfg.get("system")
    model: str = cfg["model"]
    endpoint: str = cfg["endpoint"]
    temperature: float = cfg["temperature"]
    max_tokens_target: int = cfg["max_tokens"]
    max_tokens_jitter: float = cfg.get("max_tokens_jitter", 0.1)
    top_p: float = cfg["top_p"]
    request_timeout: float = cfg["request_timeout"]

    rng = random.Random((user_id * 0x9E3779B1) & 0xFFFFFFFF)
    mt_low = max(8, int(max_tokens_target * (1.0 - max_tokens_jitter)))
    mt_high = max(mt_low + 1, int(max_tokens_target * (1.0 + max_tokens_jitter)))
    req_seq = 0

    def _should_exit() -> bool:
        if state.stop.is_set():
            return True
        if exit_event is not None and exit_event.is_set():
            return True
        return False

    while not _should_exit():
        req_seq += 1
        prompt = rng.choice(prompts)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        mt = rng.randint(mt_low, mt_high)

        corr_id = f"u{user_id:03d}-r{req_seq:04d}-{int(time.time() * 1000) % 1000000:06d}"

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": mt,
            "top_p": top_p,
            "stream": True,
        }
        headers = {"X-Request-ID": corr_id}

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        chatcmpl_id: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        error_msg: str | None = None
        t_client_start = time.time()
        t_mono_start = time.monotonic()

        state.inflight += 1

        try:
            async with session.post(
                endpoint,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    error_msg = f"HTTP {resp.status}: {body_text[:200]}"
                    state.errors += 1
                else:
                    async for raw_line in resp.content:
                        if state.stop.is_set():
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        if chatcmpl_id is None:
                            _id = obj.get("id")
                            if isinstance(_id, str):
                                chatcmpl_id = _id

                        choices = obj.get("choices") or []
                        if choices:
                            choice0 = choices[0]
                            delta = choice0.get("delta") or {}
                            c = delta.get("content")
                            if c:
                                content_parts.append(c)
                                # Each delta chunk with text ≈ 1 streamed token.
                                state.recent_tokens.append((time.monotonic(), 1))
                            rc = delta.get("reasoning_content")
                            if rc:
                                reasoning_parts.append(rc)
                                state.recent_tokens.append((time.monotonic(), 1))
                            fr = choice0.get("finish_reason")
                            if fr:
                                finish_reason = fr

                        u = obj.get("usage")
                        if u:
                            usage = u
        except asyncio.TimeoutError:
            error_msg = f"client timeout after {request_timeout}s"
            state.errors += 1
        except aiohttp.ClientError as e:
            error_msg = f"aiohttp error: {e}"
            state.errors += 1
        except asyncio.CancelledError:
            # finally will still run and decrement inflight exactly once.
            raise
        finally:
            state.inflight -= 1
            state.completed += 1

        t_mono_end = time.monotonic()
        wall = t_mono_end - t_mono_start
        # Two throughput metrics per request:
        #   tg_per_sec_wall = completion_tokens / wall_seconds
        #     User-observed. Wall includes queue wait + prefill + decode,
        #     so this is a LOWER BOUND on the model's pure decode rate.
        #   tg_per_sec_pure = completion_tokens / completion_time
        #     Server-reported pure decode rate (from the `usage` field's
        #     completion_tokens_per_second, or derived from completion_time).
        #     This excludes queue and prefill time — it's what the GPU
        #     actually decodes at during this request's decode phase.
        tg_per_sec_wall: float | None = None
        tg_per_sec_pure: float | None = None
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens")
            if isinstance(ct, int) and ct > 0 and wall > 0:
                tg_per_sec_wall = round(ct / wall, 2)
            # Prefer the server's own reported rate if it sent one; fall
            # back to deriving from completion_time so this still works on
            # older servers that don't include the _per_second field.
            pure = usage.get("completion_tokens_per_second")
            if isinstance(pure, (int, float)) and pure > 0:
                tg_per_sec_pure = round(float(pure), 2)
            elif isinstance(ct, int) and ct > 0:
                comp_time = usage.get("completion_time")
                if isinstance(comp_time, (int, float)) and comp_time > 0:
                    tg_per_sec_pure = round(ct / float(comp_time), 2)

        record: dict[str, Any] = {
            "corr_id": corr_id,
            "user_id": user_id,
            "req_seq": req_seq,
            "chatcmpl_id": chatcmpl_id,
            "model": model,
            "prompt": prompt,
            "system": system,
            "max_tokens_requested": mt,
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts),
            "finish_reason": finish_reason,
            "usage": usage,
            "wall_seconds": round(wall, 3),
            "tg_per_sec_wall": tg_per_sec_wall,
            "tg_per_sec_pure": tg_per_sec_pure,
            "t_start_rel": round(t_mono_start - state.start_time, 3),
            "t_end_rel": round(t_mono_end - state.start_time, 3),
            "started_at_epoch": round(t_client_start, 3),
            "error": error_msg,
        }
        # Non-blocking put — queue is unbounded.
        state.request_queue.put_nowait(record)


# ---------------------------------------------------------------------------
# Ramp controller
# ---------------------------------------------------------------------------

async def ramp_controller(
    state: LoadState,
    session: aiohttp.ClientSession,
    cfg: dict[str, Any],
    users: list[asyncio.Task],
) -> None:
    """
    Grows and shrinks the virtual-user population according to the
    configured ramp_mode:

      linear
        Grows from 0 to target over ramp_s. No ramp-down; state.stop
        is set after hold_s and virtual users break out of their
        current SSE loop (forcing client disconnect on in-flight
        requests — causes broken-pipe errors in the server log).

      step
        Grows by ramp_step_users every ramp_step_s until target.
        Holds for hold_s. Then shrinks by ramp_down_step_users every
        ramp_down_step_s using a GRACEFUL drain: each departing user
        is signaled via its per-user exit_event, so it finishes its
        current streaming request before exiting instead of being
        cancelled mid-stream. Never sets state.stop until after every
        user has drained cleanly.
    """
    ramp_mode: str = cfg.get("ramp_mode", "linear")
    target: int = cfg["target_users"]
    hold_s: float = cfg["hold_s"]

    # Per-user exit events (parallel to `users` list).
    exit_events: list[asyncio.Event] = cfg.setdefault("_exit_events", [])

    def _spawn(uid: int) -> None:
        ev = asyncio.Event()
        exit_events.append(ev)
        users.append(
            asyncio.create_task(virtual_user(uid, session, state, cfg, ev))
        )

    if ramp_mode == "linear":
        ramp_s: float = cfg["ramp_s"]
        tick = 0.1
        ramp_start = time.monotonic()
        while True:
            elapsed = time.monotonic() - ramp_start
            if elapsed >= ramp_s:
                break
            desired = int(round(target * (elapsed / ramp_s)))
            while len(users) < desired:
                _spawn(len(users))
            state.active_target = desired
            await asyncio.sleep(tick)
        while len(users) < target:
            _spawn(len(users))
        state.active_target = target
        await asyncio.sleep(hold_s)
        state.stop.set()
        return

    # step mode
    step_users: int = cfg["ramp_step_users"]
    step_s: float = cfg["ramp_step_s"]
    down_step_users: int = cfg.get("ramp_down_step_users", step_users)
    down_step_s: float = cfg.get("ramp_down_step_s", step_s)
    jitter_pct: float = float(cfg.get("ramp_jitter_pct", 0.0))
    jitter_seed: int = int(cfg.get("ramp_jitter_seed", -1))

    # Deterministic RNG when seed provided, so runs are reproducible.
    import random as _rand
    _rng = _rand.Random(jitter_seed if jitter_seed >= 0 else None)

    def _jittered(base_s: float) -> float:
        """Return base_s ± (jitter_pct/100 * base_s), uniformly sampled.
        At jitter_pct=0 this is a no-op and returns base_s exactly."""
        if jitter_pct <= 0:
            return base_s
        frac = jitter_pct / 100.0
        low = base_s * (1.0 - frac)
        high = base_s * (1.0 + frac)
        # Clamp low at 0.01s to avoid absurdly tight bursts.
        return max(0.01, _rng.uniform(low, high))

    if jitter_pct > 0:
        print(f"[driver] ramp jitter : ±{jitter_pct:.0f}% of step_s "
              f"(seed={jitter_seed if jitter_seed >= 0 else 'random'})",
              flush=True)

    # Ramp up in discrete steps. Each step spawns `step_users` new users
    # and waits step_s seconds before the next step. This gives the
    # BatchScheduler breathing room to stabilize at each new batch size.
    current_target = 0
    while current_target < target:
        current_target = min(target, current_target + step_users)
        while len(users) < current_target:
            _spawn(len(users))
        state.active_target = current_target
        await asyncio.sleep(_jittered(step_s))

    # Hold at target.
    await asyncio.sleep(hold_s)

    # Graceful ramp-down. We signal departing users via their per-user
    # exit_event; they finish their current streaming request (the
    # user's _should_exit check happens at the top of each while-loop
    # iteration, AFTER a request completes) and then exit cleanly. We
    # do NOT cancel their tasks and we do NOT close their TCP streams
    # mid-response, which avoids flooding the server log with
    # broken-pipe errors.
    current_target = target
    while current_target > 0:
        current_target = max(0, current_target - down_step_users)
        # Signal exit on the NEWEST users first (LIFO). Oldest users
        # stay running longest — they were the first to warm up the
        # batch and it's visually coherent to have them be the last
        # to drop.
        signaled = 0
        needed = len(users) - current_target
        for i in range(len(exit_events) - 1, -1, -1):
            if signaled >= needed:
                break
            if not exit_events[i].is_set():
                exit_events[i].set()
                signaled += 1
        state.active_target = current_target
        await asyncio.sleep(_jittered(down_step_s))

    # All users have been signaled to exit. Wait for them to finish
    # their current requests (no hard timeout — this is the "let
    # conversations end" contract).
    if users:
        await asyncio.gather(*users, return_exceptions=True)

    # Now it's safe to set stop (for the sampler + writer tasks).
    state.stop.set()


# ---------------------------------------------------------------------------
# Metrics sampler
# ---------------------------------------------------------------------------

async def metrics_sampler(
    state: LoadState,
    cfg: dict[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    """
    Samples state every SAMPLE_MS and writes a JSONL trace. Returns the
    trace in memory for in-process consumers. Also prints a live progress
    breadcrumb to stderr every 2 seconds so the user can see the run.
    """
    sample_ms: int = cfg["sample_ms"]
    interval = sample_ms / 1000.0
    smoothing_window_s: float = cfg.get("smoothing_window_s", 2.0)
    trace: list[dict[str, Any]] = []

    total_s = float(cfg["ramp_s"]) + float(cfg["hold_s"])
    last_breadcrumb = 0.0
    breadcrumb_interval = 2.0
    is_tty = sys.stderr.isatty()
    bar_width = 28

    def emit_breadcrumb(sample: dict[str, Any]) -> None:
        t = sample["t"]
        pct = min(100.0, (t / total_s) * 100.0) if total_s > 0 else 0.0
        filled = int(round(bar_width * pct / 100.0))
        bar = "█" * filled + "░" * (bar_width - filled)
        phase = "ramp " if t <= float(cfg["ramp_s"]) else "hold "
        line = (
            f"\r[load] {phase}[{bar}] {pct:5.1f}%  "
            f"t={t:6.1f}s  users={sample['active']:3d}  "
            f"inflight={sample['inflight']:3d}  "
            f"tok/s={sample['agg_tps']:5d}  "
            f"done={sample['completed']:5d}  err={sample['errors']:2d}"
        )
        if is_tty:
            sys.stderr.write(line)
        else:
            sys.stderr.write(line.lstrip("\r") + "\n")
        sys.stderr.flush()

    try:
        with output_path.open("w") as f:
            while not state.stop.is_set():
                try:
                    now = time.monotonic()
                    t = now - state.start_time

                    # Proxy path: prune stale tokens from the SSE-event
                    # sliding window. We keep this running even when server
                    # metrics are available so the fallback stays warm.
                    cutoff = now - smoothing_window_s
                    while state.recent_tokens and state.recent_tokens[0][0] < cutoff:
                        state.recent_tokens.popleft()
                    proxy_tokens = len(state.recent_tokens)
                    proxy_tps = round(proxy_tokens / smoothing_window_s, 1)

                    # Prefer ground-truth from the server's /metrics
                    # endpoint when available. Falls back to the SSE
                    # proxy on older AFM builds that don't expose it.
                    if state.server_metrics_available:
                        agg_tps = round(state.server_gen_tps, 1)
                        prefill_tps = round(state.server_prefill_tps, 1)
                        chart_active = state.server_inflight
                        source = "server"
                    else:
                        agg_tps = proxy_tps
                        prefill_tps = 0.0
                        chart_active = state.inflight
                        source = "proxy"

                    sample = {
                        "t": round(t, 3),
                        "active": chart_active,
                        "target": state.active_target,
                        "inflight": state.inflight,
                        "agg_tps": agg_tps,
                        "prefill_tps": prefill_tps,
                        "proxy_tps": proxy_tps,
                        "source": source,
                        "completed": state.completed,
                        "errors": state.errors,
                        "gen_total": state.server_gen_total,
                        "prompt_total": state.server_prefill_total,
                        "cache_hits": state.server_cache_hits,
                        "cache_misses": state.server_cache_misses,
                    }
                    trace.append(sample)
                    f.write(json.dumps(sample) + "\n")
                    f.flush()

                    if t - last_breadcrumb >= breadcrumb_interval:
                        try:
                            emit_breadcrumb(sample)
                        except Exception:
                            pass  # never let a failed breadcrumb kill the sampler
                        last_breadcrumb = t
                except Exception as e:
                    # Log to stderr and keep going — one bad tick must not kill
                    # the whole trace.
                    try:
                        import traceback
                        sys.stderr.write(f"\n[sampler] tick error: {e}\n")
                        traceback.print_exc(file=sys.stderr)
                    except Exception:
                        pass

                await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        import traceback
        sys.stderr.write(f"\n[sampler] FATAL: {e}\n")
        traceback.print_exc(file=sys.stderr)
        raise

    # Clean up the \r line so the final summary starts on its own row
    if is_tty:
        sys.stderr.write("\n")
        sys.stderr.flush()

    # One final tail sample with t ticking a hair past stop for clean chart closure
    now = time.monotonic()
    t = now - state.start_time
    tail = {
        "t": round(t, 3),
        "active": state.active_target,
        "inflight": state.inflight,
        "agg_tps": 0,
        "completed": state.completed,
        "errors": state.errors,
    }
    trace.append(tail)
    with output_path.open("a") as f:
        f.write(json.dumps(tail) + "\n")
    return trace


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def request_log_writer(
    state: LoadState,
    requests_path: Path,
) -> None:
    """
    Drains per-request records from state.request_queue and appends each
    as a JSONL line to requests_path. Exits when stop is set AND the
    queue is empty.
    """
    with requests_path.open("w") as f:
        while True:
            try:
                rec = await asyncio.wait_for(state.request_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if state.stop.is_set() and state.request_queue.empty():
                    return
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            state.request_queue.task_done()


async def run_warmup(session: aiohttp.ClientSession, cfg: dict[str, Any]) -> None:
    """
    Pre-warm MLX Metal kernels by firing N fire-and-forget requests with
    a VERY small max_tokens so they complete naturally within the warmup
    window. Every request is fire-and-wait — we do NOT loop virtual users
    and we do NOT cancel in-flight requests, because cancelling a
    streaming request produces a server-side zombie: the server keeps
    decoding the request to /dev/null (because there's no client to send
    to) until it reaches its max_tokens, which pins GPU compute to dead
    requests for minutes after warmup ends.

    With max_tokens=warmup_max_tokens (default 32) and a B=N prefill
    followed by ~32 decode steps at the max batch rate, each warmup
    request completes in roughly:

        prefill(~2s) + 32 * (1 / per-seq decode rate at B=N)
        ≈ 2s + 32 * (1/5)  ≈ 8-10 seconds for N=200 on M3 Ultra

    We wait for ALL warmup requests to complete before returning. No
    zombies.
    """
    warmup_users: int = cfg["warmup_users"]
    warmup_max_tokens: int = cfg.get("warmup_max_tokens", 32)
    if warmup_users <= 0:
        return

    prompts: list[str] = cfg["prompts"]
    system: str | None = cfg.get("system")
    model: str = cfg["model"]
    endpoint: str = cfg["endpoint"]
    temperature: float = cfg["temperature"]
    top_p: float = cfg["top_p"]
    is_tty = sys.stderr.isatty()

    sys.stderr.write(
        f"\n[warmup] firing {warmup_users} requests with max_tokens={warmup_max_tokens} "
        f"to pre-compile MLX Metal kernels at peak batch size\n"
    )
    sys.stderr.flush()

    state = {"tokens_total": 0, "completed": 0, "errors": 0}
    t_start = time.monotonic()

    async def warmup_one(uid: int) -> None:
        rng = random.Random((uid * 0x9E3779B1 + 1) & 0xFFFFFFFF)
        prompt = rng.choice(prompts)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": warmup_max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        corr_id = f"warmup-u{uid:03d}-{int(time.time() * 1000) % 100000:05d}"
        try:
            async with session.post(
                endpoint,
                json=body,
                headers={"X-Request-ID": corr_id},
                timeout=aiohttp.ClientTimeout(total=300),  # generous — never cancel
            ) as resp:
                if resp.status != 200:
                    await resp.read()
                    state["errors"] += 1
                    return
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            state["tokens_total"] += 1
            state["completed"] += 1
        except Exception:
            state["errors"] += 1

    tasks = [asyncio.create_task(warmup_one(i)) for i in range(warmup_users)]

    # Print progress while warmup runs. No deadline — warmup is bounded by
    # natural completion of warmup_max_tokens tokens per request.
    async def progress():
        last_print = 0.0
        while not all(t.done() for t in tasks):
            t = time.monotonic() - t_start
            if t - last_print >= 1.0:
                done = sum(1 for x in tasks if x.done())
                line = (
                    f"\r[warmup] t={t:5.1f}s  done={done:3d}/{warmup_users}  "
                    f"tokens_streamed={state['tokens_total']:5d}"
                )
                if is_tty:
                    sys.stderr.write(line)
                else:
                    sys.stderr.write(line.lstrip("\r") + "\n")
                sys.stderr.flush()
                last_print = t
            await asyncio.sleep(0.2)

    progress_task = asyncio.create_task(progress())
    await asyncio.gather(*tasks, return_exceptions=True)
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass

    elapsed = time.monotonic() - t_start
    if is_tty:
        sys.stderr.write("\n")
    sys.stderr.write(
        f"[warmup] complete in {elapsed:.1f}s — {state['completed']}/{warmup_users} ok, "
        f"{state['errors']} errors, {state['tokens_total']} tokens streamed. "
        f"Starting real ramp.\n\n"
    )
    sys.stderr.flush()


async def run(cfg: dict[str, Any], output_path: Path) -> dict[str, Any]:
    requests_path: Path = cfg["requests_output"]
    requests_path.parent.mkdir(parents=True, exist_ok=True)

    conn = aiohttp.TCPConnector(
        limit=0,  # unlimited; we control concurrency via virtual users
        ttl_dns_cache=300,
    )
    async with aiohttp.ClientSession(connector=conn) as session:
        # Phase 1: warmup (no metrics recorded). Kernels are compiled here.
        await run_warmup(session, cfg)

        # Phase 2: real ramp with metrics recording. t=0 starts NOW.
        state = LoadState()
        state.start_time = time.monotonic()
        state.wall_clock_start = time.time()

        async def watchdog(session: aiohttp.ClientSession, state: "LoadState", cfg: dict[str, Any]) -> None:
            """Periodically probe the server. If unresponsive for 15s, stop the test."""
            base = cfg["endpoint"].rsplit("/v1/", 1)[0]
            url = f"{base}/v1/models"
            fail_since: float | None = None
            watchdog_timeout = 15.0
            while not state.stop.is_set():
                await asyncio.sleep(2.0)
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3.0)) as r:
                        if r.status == 200:
                            fail_since = None
                            continue
                except Exception:
                    pass
                now = time.monotonic()
                if fail_since is None:
                    fail_since = now
                    sys.stderr.write(f"[watchdog] server unresponsive at t={now - state.start_time:.1f}s\n")
                elif now - fail_since >= watchdog_timeout:
                    sys.stderr.write(
                        f"[watchdog] server dead for {now - fail_since:.0f}s — stopping test\n"
                    )
                    state.stop.set()
                    return

        users: list[asyncio.Task] = []
        ramp_task = asyncio.create_task(ramp_controller(state, session, cfg, users))
        sampler_task = asyncio.create_task(metrics_sampler(state, cfg, output_path))
        writer_task = asyncio.create_task(request_log_writer(state, requests_path))
        metrics_task = asyncio.create_task(poll_metrics(session, state, cfg))
        watchdog_task = asyncio.create_task(watchdog(session, state, cfg))

        await ramp_task
        # In step mode, ramp_task itself has already awaited all users
        # (graceful drain). In linear mode, state.stop was just set and
        # users break out of their SSE loops; give them a short grace
        # period to finalize before cancelling.
        ramp_mode = cfg.get("ramp_mode", "linear")
        if ramp_mode == "linear":
            try:
                await asyncio.wait_for(
                    asyncio.gather(*users, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                for u in users:
                    if not u.done():
                        u.cancel()
                await asyncio.gather(*users, return_exceptions=True)

        # -------- Cooldown tail --------
        # At this point every virtual user has been signaled to exit and
        # either finished gracefully (step mode) or been cancelled (linear
        # mode). BUT the server may still be finishing the very last
        # streaming responses — the scheduler's batch isn't empty yet. We
        # keep the sampler running for up to `cooldown_s` more seconds so
        # the chart tail shows the activity draining to zero. Exit early
        # as soon as we've seen sustained idle (inflight == 0 AND server
        # num_requests_running == 0) for ~2 consecutive seconds.
        cooldown_s = float(cfg.get("cooldown_s", 0.0))
        if cooldown_s > 0 and not state.stop.is_set():
            idle_streak_s = 0.0
            idle_threshold_s = 2.0
            tick = 0.5
            cooldown_deadline = time.monotonic() + cooldown_s
            while time.monotonic() < cooldown_deadline:
                driver_idle = state.inflight == 0
                server_idle = (
                    state.server_inflight == 0
                    if state.server_metrics_available else True
                )
                if driver_idle and server_idle:
                    idle_streak_s += tick
                    if idle_streak_s >= idle_threshold_s:
                        break
                else:
                    idle_streak_s = 0.0
                await asyncio.sleep(tick)

        # Allow sampler to emit its tail line
        await asyncio.sleep(0.3)
        sampler_task.cancel()
        try:
            await sampler_task
        except asyncio.CancelledError:
            pass
        # Stop the metrics poller and watchdog.
        metrics_task.cancel()
        try:
            await metrics_task
        except asyncio.CancelledError:
            pass
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        # Wait for request writer to drain its queue before closing.
        try:
            await asyncio.wait_for(writer_task, timeout=5.0)
        except asyncio.TimeoutError:
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass

    # Summary
    wall = time.monotonic() - state.start_time
    summary = {
        "wall_seconds": round(wall, 2),
        "completed_requests": state.completed,
        "errors": state.errors,
        "target_users": cfg["target_users"],
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_prompts(general_path: Path, prefix_cache_path: Path, mode: str) -> tuple[list[str], str | None]:
    """
    Returns (prompts_list, system_prompt_or_none) based on mode:
        general      — 100 varied prompts, no system
        prefix-cache — long shared system + short user variations
        mixed        — 50/50 blend, each user request picks a side
    """
    with general_path.open() as f:
        general = json.load(f)["prompts"]
    with prefix_cache_path.open() as f:
        pc_data = json.load(f)
    pc_prompts = pc_data["user_prompts"]
    pc_system = pc_data["system"]

    if mode == "general":
        return general, None
    if mode == "prefix-cache":
        return pc_prompts, pc_system
    if mode == "mixed":
        return general + pc_prompts, None  # mixed mode loses the shared prefix
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Concurrent load driver for AFM concurrent-throughput demo.",
    )
    ap.add_argument("--endpoint", default="http://localhost:9999/v1/chat/completions")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-35B-A3B-4bit")
    ap.add_argument("--target-users", type=int, default=200, help="Peak concurrent virtual users.")
    ap.add_argument("--ramp-mode", choices=("linear", "step"), default="step",
                    help="'linear' = classic continuous ramp over --ramp-s. "
                         "'step' = add --ramp-step-users every --ramp-step-s (default).")
    ap.add_argument("--ramp-s", type=float, default=45.0,
                    help="[linear mode only] Seconds to ramp from 0 to target.")
    ap.add_argument("--ramp-step-users", type=int, default=2,
                    help="[step mode] Users added per step.")
    ap.add_argument("--ramp-step-s", type=float, default=5.0,
                    help="[step mode] Seconds between steps during ramp-up.")
    ap.add_argument("--ramp-down-step-users", type=int, default=-1,
                    help="[step mode] Users removed per step during ramp-down. "
                         "Default (-1) = match --ramp-step-users.")
    ap.add_argument("--ramp-down-step-s", type=float, default=-1.0,
                    help="[step mode] Seconds between ramp-down steps. "
                         "Default (-1) = match --ramp-step-s.")
    ap.add_argument("--ramp-jitter-pct", type=float, default=0.0,
                    help="[step mode] Randomize each step's sleep by +/- "
                         "pct percent of ramp-step-s (e.g. 30 -> each gap "
                         "uniformly in [0.7*step_s, 1.3*step_s]). Use this "
                         "to break the ramp's fixed period when diagnosing "
                         "whether throughput dips are caused by the ramp "
                         "cadence or by something intrinsic to the server.")
    ap.add_argument("--ramp-jitter-seed", type=int, default=-1,
                    help="[step mode] RNG seed for ramp jitter. -1 = random.")
    ap.add_argument("--hold-s", type=float, default=25.0,
                    help="Seconds to hold at peak before ramp-down begins.")
    ap.add_argument("--cooldown-s", type=float, default=30.0,
                    help="After the ramp-down finishes (all users signaled to exit, all "
                         "in-flight requests drained), keep recording for up to this many "
                         "seconds so the chart tail shows the run settling to zero. "
                         "The driver exits EARLY as soon as both the driver-side "
                         "inflight counter AND the server /metrics num_requests_running "
                         "have been 0 for ~2 seconds. Set to 0 to skip cooldown.")
    ap.add_argument("--sample-ms", type=int, default=250, help="Metrics sample interval in ms.")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="Target max_tokens ceiling. Default 3000 is high enough that most responses "
                         "finish on natural EOS (including reasoning) rather than being hard-truncated. "
                         "The natural variance in response length replaces the synthetic stagger that "
                         "a smaller ceiling would need.")
    ap.add_argument("--max-tokens-jitter", type=float, default=0.1,
                    help="Fractional jitter (0..1). mt is picked uniformly in [mt*(1-j), mt*(1+j)]. "
                         "With a 3000-token ceiling, natural EOS variance dominates — 0.1 is plenty.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0, help="1.0 avoids the known TopPSampler path; set <1 to exercise it.")
    ap.add_argument("--request-timeout", type=float, default=900.0,
                    help="Per-request aiohttp client timeout (seconds). Must exceed the "
                         "expected wall time of a single request at peak batch size. "
                         "With --concurrent 200 + --max-tokens 3000, per-sequence decode "
                         "is ~5 tok/s → worst-case request ≈600s. Default 900s (15 min) "
                         "gives 50 percent headroom. Increase if you see 'client timeout' errors.")
    ap.add_argument("--smoothing-window-s", type=float, default=2.0,
                    help="Sliding window width in seconds for agg_tps computation.")
    ap.add_argument("--initial-tps-max", type=float, default=1300.0,
                    help="Initial Y2 axis maximum for the throughput chart. Axis expands if observed tps exceeds this.")
    ap.add_argument("--warmup-users", type=int, default=0,
                    help="Fire this many one-shot warmup requests BEFORE the recorded ramp "
                         "to pre-compile MLX Metal kernels at peak batch size. Set to the "
                         "target user count (e.g. 200) if you see a throughput cliff at "
                         "the top of the ramp. Default 0 — step mode normally doesn't "
                         "need warmup because batch size grows gradually.")
    ap.add_argument("--warmup-max-tokens", type=int, default=32,
                    help="max_tokens for each warmup request. Keep this small (default 32) "
                         "so warmup requests complete NATURALLY within the warmup phase. "
                         "Larger values create zombie requests (see commit message for #X).")
    ap.add_argument("--warmup-s", type=float, default=-1.0,
                    help="[deprecated] no longer used; warmup is bounded by natural "
                         "completion of warmup_max_tokens tokens per request.")
    ap.add_argument(
        "--mode",
        choices=("general", "prefix-cache", "mixed"),
        default="general",
        help="Prompt pool to use.",
    )
    ap.add_argument("--prompts-dir", default="Scripts/demo/prompts")
    ap.add_argument("--output", default="Scripts/demo/out/trace.jsonl",
                    help="Per-250ms time-series metrics output (JSONL).")
    ap.add_argument("--requests-output", default="Scripts/demo/out/requests.jsonl",
                    help="Per-request capture output (prompt, content, reasoning_content, corr_id, "
                         "chatcmpl_id, usage, timing). One JSON object per line.")
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    prompts, system = load_prompts(
        prompts_dir / "general.json",
        prompts_dir / "prefix_cache.json",
        args.mode,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warmup_users = args.warmup_users  # 0 = disabled
    down_step_users = args.ramp_down_step_users if args.ramp_down_step_users >= 0 else args.ramp_step_users
    down_step_s = args.ramp_down_step_s if args.ramp_down_step_s >= 0 else args.ramp_step_s

    # Compute expected total duration so downstream tools (renderer, watcher)
    # can pre-set their X axes.
    if args.ramp_mode == "step":
        n_up_steps = max(1, (args.target_users + args.ramp_step_users - 1) // args.ramp_step_users)
        n_down_steps = max(1, (args.target_users + down_step_users - 1) // down_step_users)
        ramp_up_s = n_up_steps * args.ramp_step_s
        ramp_down_s = n_down_steps * down_step_s
    else:
        ramp_up_s = args.ramp_s
        ramp_down_s = 0.0
    cooldown_s = max(0.0, float(args.cooldown_s))
    total_seconds = ramp_up_s + args.hold_s + ramp_down_s + cooldown_s

    cfg = {
        "endpoint": args.endpoint,
        "model": args.model,
        "target_users": args.target_users,
        "ramp_mode": args.ramp_mode,
        "ramp_s": args.ramp_s,
        "ramp_step_users": args.ramp_step_users,
        "ramp_step_s": args.ramp_step_s,
        "ramp_down_step_users": down_step_users,
        "ramp_down_step_s": down_step_s,
        "ramp_jitter_pct": args.ramp_jitter_pct,
        "ramp_jitter_seed": args.ramp_jitter_seed,
        "ramp_up_s": ramp_up_s,
        "ramp_down_s": ramp_down_s,
        "cooldown_s": cooldown_s,
        "total_seconds": total_seconds,
        "hold_s": args.hold_s,
        "sample_ms": args.sample_ms,
        "max_tokens": args.max_tokens,
        "max_tokens_jitter": args.max_tokens_jitter,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "request_timeout": args.request_timeout,
        "smoothing_window_s": args.smoothing_window_s,
        "initial_tps_max": args.initial_tps_max,
        "mode": args.mode,
        "prompts": prompts,
        "system": system,
        "requests_output": Path(args.requests_output),
        "warmup_users": warmup_users,
        "warmup_max_tokens": args.warmup_max_tokens,
    }

    print(f"[driver] endpoint    : {args.endpoint}")
    print(f"[driver] model       : {args.model}")
    print(f"[driver] mode        : {args.mode}  ({len(prompts)} prompts" + (" + shared system" if system else "") + ")")
    if warmup_users > 0:
        print(f"[driver] warmup      : {warmup_users} one-shot requests, max_tokens={args.warmup_max_tokens} (kernel pre-compile, no zombies)")
    else:
        print(f"[driver] warmup      : disabled (step mode grows batch size gradually)")
    if args.ramp_mode == "step":
        print(f"[driver] ramp mode   : step  +{args.ramp_step_users}/step, {args.ramp_step_s}s between steps")
        print(f"[driver] ramp up     : 0 -> {args.target_users} over ~{ramp_up_s:.0f}s")
        print(f"[driver] hold        : {args.hold_s}s at {args.target_users}")
        print(f"[driver] ramp down   : {args.target_users} -> 0 over ~{ramp_down_s:.0f}s, -{down_step_users}/step, {down_step_s}s between steps (graceful drain)")
        print(f"[driver] cooldown    : up to {cooldown_s:.0f}s zero-activity tail, early-exit on sustained idle")
        print(f"[driver] total       : ~{total_seconds:.0f}s recorded run ({total_seconds/60:.1f} min)")
    else:
        print(f"[driver] ramp        : 0 -> {args.target_users} users over {args.ramp_s}s, hold {args.hold_s}s (linear)")
    print(f"[driver] max_tokens  : {args.max_tokens} (±{args.max_tokens_jitter*100:.0f}% jitter), temp={args.temperature}, top_p={args.top_p}")
    print(f"[driver] trace out   : {output_path}")
    print(f"[driver] requests out: {args.requests_output}")
    print()

    summary = asyncio.run(run(cfg, output_path))
    print()
    print("[driver] done:")
    print(f"  wall seconds        : {summary['wall_seconds']}")
    print(f"  completed requests  : {summary['completed_requests']}")
    print(f"  errors              : {summary['errors']}")

    # Write summary alongside the trace
    summary_path = output_path.with_suffix(".summary.json")
    # Exclude prompt pools (too big), internal async state, and coerce
    # Path objects to strings. Anything else non-JSON is dropped.
    _SUMMARY_EXCLUDE = {"prompts", "system", "_exit_events"}
    def _jsonable(v):
        if isinstance(v, Path):
            return str(v)
        try:
            json.dumps(v)
            return v
        except TypeError:
            return repr(v)
    summary["cfg"] = {
        k: _jsonable(v)
        for k, v in cfg.items()
        if not k.startswith("_") and k not in _SUMMARY_EXCLUDE
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary             : {summary_path}")


if __name__ == "__main__":
    main()
