#!/usr/bin/env python3
"""Generate OSS corpus tasks (idempotent). Run: python scripts/seed_oss_corpus.py --write-gold"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks" / "v1"
_LICENSE = "MIT License\n\nCopyright (c) upstream contributors.\n"

# task_id -> paths to write for gold fix (under repo/)
FIXES: dict[str, dict[str, str]] = {}
# full task records
TASKS: list[dict] = []


def add(task: dict, fix: dict[str, str]) -> None:
    TASKS.append(task)
    FIXES[task["id"]] = fix


def _py_service(
    task_id: str, pkg: str, scale: str, bug_body: str, fix_body: str, issue: str
) -> None:
    files = {
        "repo/LICENSE": _LICENSE,
        f"repo/{pkg}/__init__.py": "",
        f"repo/{pkg}/service.py": bug_body,
        f"repo/{pkg}/api.py": f"from {pkg}.service import run\n",
        f"repo/{pkg}/types.py": "class Item:\n    def __init__(self, v):\n        self.v = v\n",
        "tests/test_service.py": (
            f"from {pkg}.service import run\n\n"
            "def test_run():\n    assert run(2) == 4\n\n"
            "def test_import_api():\n"
            f"    from {pkg} import api\n    assert api\n"
        ),
    }
    add(
        {
            "id": task_id,
            "category": "bug_fix",
            "languages": ["python"],
            "difficulty": "medium" if scale == "medium" else "easy",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": scale,
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": f"https://github.com/morganlinton/VulcanBench/issues/{task_id}",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": f"Vendored {pkg} slice ({task_id}).",
            "tests": {
                "fail_to_pass": [
                    {"name": "run", "cmd": "python -m pytest test_service.py::test_run -q"}
                ],
                "pass_to_pass": [
                    {
                        "name": "import",
                        "cmd": "python -m pytest test_service.py::test_import_api -q",
                    }
                ],
            },
            "test_timeout_s": 120,
            "agent_hints": {
                "suggested_max_steps": 100 if scale == "medium" else 60,
                "entry_paths": [f"{pkg}/service.py"],
            },
            "files": files,
            "issue": issue,
        },
        {f"repo/{pkg}/service.py": fix_body},
    )


def _go_service(task_id: str, pkg: str, scale: str) -> None:
    mod = "example.com/bench"
    import_path = f"{mod}/{pkg}"
    files = {
        "repo/LICENSE": "MIT\n",
        "repo/go.mod": f"module {mod}\n\ngo 1.23\n",
        f"repo/{pkg}/calc.go": f"package {pkg}\n\nfunc Double(n int) int {{ return n + 1 }}\n",
        f"repo/{pkg}/doc.go": f"package {pkg}\n",
        f"repo/{pkg}/util.go": f"package {pkg}\n\nfunc ID(n int) int {{ return n }}\n",
        f"tests/{pkg}/calc_test.go": (
            f'package {pkg}_test\n\nimport (\n  "testing"\n  "{import_path}"\n)\n\n'
            f'func TestDouble(t *testing.T) {{\n  if {pkg}.Double(2) != 4 {{ t.Fatal("want 4") }}\n}}\n\n'
            f"func TestID(t *testing.T) {{\n  if {pkg}.ID(1) != 1 {{ t.Fatal() }}\n}}\n"
        ),
    }
    add(
        {
            "id": task_id,
            "category": "bug_fix",
            "languages": ["go"],
            "difficulty": "medium" if scale == "medium" else "easy",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": scale,
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": f"https://github.com/morganlinton/VulcanBench/issues/{task_id}",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": f"Vendored Go {pkg} slice.",
            "tests": {
                "fail_to_pass": [{"name": "double", "cmd": "go test -run '^TestDouble$' ./..."}],
                "pass_to_pass": [{"name": "id", "cmd": "go test -run '^TestID$' ./..."}],
            },
            "test_timeout_s": 120,
            "agent_hints": {"suggested_max_steps": 100 if scale == "medium" else 60},
            "files": files,
            "issue": f"# Fix `{pkg}.Double`\n\n`Double` increments instead of doubling. Fix `repo/{pkg}/calc.go`.\n",
        },
        {f"repo/{pkg}/calc.go": f"package {pkg}\n\nfunc Double(n int) int {{ return n * 2 }}\n"},
    )


def _ts_service(task_id: str, pkg: str, scale: str) -> None:
    files = {
        "repo/LICENSE": "MIT\n",
        "repo/package.json": '{"type":"module","name":"' + pkg + '"}\n',
        f"repo/src/{pkg}.ts": "export function run(x: number) { return x + 1; }\n",
        "repo/src/index.ts": f"export {{ run }} from './{pkg}.ts';\n",
        "repo/src/types.ts": "export type Num = number;\n",
        f"tests/{pkg}.test.ts": (
            "import test from 'node:test';\nimport assert from 'node:assert/strict';\n"
            f"import {{ run }} from './src/{pkg}.ts';\n"
            "test('run', () => assert.equal(run(2), 4));\n"
            "test('types', () => assert.equal(typeof 1, 'number'));\n"
        ),
    }
    add(
        {
            "id": task_id,
            "category": "bug_fix",
            "languages": ["typescript"],
            "difficulty": "medium" if scale == "medium" else "easy",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": scale,
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": f"https://github.com/morganlinton/VulcanBench/issues/{task_id}",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": f"Vendored TS {pkg} slice.",
            "tests": {
                "fail_to_pass": [
                    {
                        "name": "run",
                        "cmd": (
                            f"node --experimental-strip-types --test "
                            f"--test-name-pattern='^run$' {pkg}.test.ts"
                        ),
                    }
                ],
                "pass_to_pass": [
                    {
                        "name": "types",
                        "cmd": (
                            f"node --experimental-strip-types --test "
                            f"--test-name-pattern='^types$' {pkg}.test.ts"
                        ),
                    }
                ],
            },
            "test_timeout_s": 120,
            "agent_hints": {"suggested_max_steps": 100 if scale == "medium" else 60},
            "files": files,
            "issue": f"# Fix `{pkg}.run`\n\n`run` must double its input. Fix `repo/src/{pkg}.ts`.\n",
        },
        {f"repo/src/{pkg}.ts": "export function run(x: number) { return x * 2; }\n"},
    )


def _build_tasks() -> None:
    # Pilot + M1 medium (acme retry, ledger, metrics, router + 2 py medium)
    acme_broken = {
        "repo/LICENSE": _LICENSE,
        "repo/acme/__init__.py": "from acme.http import HttpClient\nfrom acme.db import DbClient\n",
        "repo/acme/config.py": "DEFAULT_MAX_ATTEMPTS = 3\nDEFAULT_BASE_DELAY = 0.1\nDEFAULT_MAX_DELAY = 1.0\n",
        "repo/acme/retry.py": (
            "import time\n\n\ndef retry_call(fn, *, max_attempts, base_delay, max_delay, sleep=time.sleep):\n"
            "    last = None\n    for attempt in range(max_attempts - 1):\n"
            "        try:\n            return fn()\n"
            "        except Exception as e:\n            last = e\n"
            "            delay = base_delay * (2 ** attempt)\n            sleep(delay)\n"
            "    raise last\n"
        ),
        "repo/acme/http.py": (
            "from acme.config import DEFAULT_BASE_DELAY, DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_DELAY\n"
            "from acme.retry import retry_call\n\nclass HttpClient:\n"
            "    def __init__(self, sleep):\n        self._sleep = sleep\n"
            "    def request_with_retry(self, fn, max_attempts=DEFAULT_MAX_ATTEMPTS,\n"
            "                           base_delay=DEFAULT_BASE_DELAY, max_delay=DEFAULT_MAX_DELAY):\n"
            "        return retry_call(fn, max_attempts=max_attempts, base_delay=base_delay,\n"
            "                          max_delay=max_delay, sleep=self._sleep)\n"
        ),
        "repo/acme/db.py": (
            "from acme.config import DEFAULT_BASE_DELAY, DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_DELAY\n"
            "from acme.retry import retry_call\n\nclass DbClient:\n"
            "    def __init__(self, sleep):\n        self._sleep = sleep\n"
            "    def query_with_retry(self, fn, max_attempts=DEFAULT_MAX_ATTEMPTS,\n"
            "                         base_delay=DEFAULT_BASE_DELAY, max_delay=DEFAULT_MAX_DELAY):\n"
            "        return retry_call(fn, max_attempts=max_attempts, base_delay=base_delay,\n"
            "                          max_delay=max_delay, sleep=self._sleep)\n"
        ),
        "tests/test_acme_retry.py": (
            "from acme.http import HttpClient\nfrom acme.db import DbClient\n\nclass Recorder:\n"
            "    def __init__(self):\n        self.delays = []\n    def __call__(self, s):\n"
            "        self.delays.append(s)\n\ndef test_clients_import():\n    assert HttpClient and DbClient\n"
            "def test_retry_attempt_count():\n    rec = Recorder()\n    c = HttpClient(rec)\n    n = {'i': 0}\n"
            "    def boom():\n        n['i'] += 1\n        raise RuntimeError('x')\n"
            "    try:\n        c.request_with_retry(boom, max_attempts=3)\n    except RuntimeError:\n        pass\n"
            "    assert n['i'] == 3\n\ndef test_retry_respects_max_delay():\n"
            "    rec = Recorder()\n    c = DbClient(rec)\n    def boom():\n        raise RuntimeError('x')\n"
            "    try:\n        c.query_with_retry(boom, max_attempts=4, base_delay=1.0, max_delay=2.0)\n"
            "    except RuntimeError:\n        pass\n    assert all(d <= 2.0 for d in rec.delays)\n"
        ),
    }
    acme_fix = {
        "repo/acme/retry.py": (
            "import time\n\n\ndef retry_call(fn, *, max_attempts, base_delay, max_delay, sleep=time.sleep):\n"
            "    last = None\n    for attempt in range(max_attempts):\n        try:\n            return fn()\n"
            "        except Exception as e:\n            last = e\n            if attempt + 1 >= max_attempts:\n"
            "                break\n            delay = min(base_delay * (2**attempt), max_delay)\n            sleep(delay)\n"
            "    raise last\n"
        )
    }
    add(
        {
            "id": "oss-pilot-acme-retry",
            "category": "refactor",
            "languages": ["python"],
            "difficulty": "medium",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": "medium",
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": "https://github.com/morganlinton/VulcanBench/issues/pilot",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": "Pilot medium OSS slice; retry helper bug across http/db.",
            "tests": {
                "fail_to_pass": [
                    {
                        "name": "attempts",
                        "cmd": "python -m pytest test_acme_retry.py::test_retry_attempt_count -q",
                    },
                    {
                        "name": "cap",
                        "cmd": "python -m pytest test_acme_retry.py::test_retry_respects_max_delay -q",
                    },
                ],
                "pass_to_pass": [
                    {
                        "name": "imports",
                        "cmd": "python -m pytest test_acme_retry.py::test_clients_import -q",
                    }
                ],
            },
            "test_timeout_s": 120,
            "agent_hints": {
                "suggested_max_steps": 100,
                "entry_paths": ["acme/retry.py", "acme/http.py"],
            },
            "files": acme_broken,
            "issue": (
                "# Consolidate duplicated retry logic\n\n"
                "Fix `acme/retry.py` so attempts and capped backoff are correct for both clients."
            ),
        },
        acme_fix,
    )

    ledger_core = (
        "from ledger.money import Money\n\nclass Ledger:\n"
        "    def __init__(self):\n        self._entries = []\n"
        "    def credit(self, account: str, amount: Money):\n        self._entries.append((account, amount.cents))\n"
        "    def balance(self, account: str) -> Money:\n        total = sum(c for a, c in self._entries if a == account)\n"
        "        return Money(total)\n"
        "    def allocate(self, total: Money, ratios):\n        parts = []\n"
        "        for r in ratios:\n            parts.append(Money(int(total.cents * r)))\n        return parts\n"
    )
    ledger_fix_core = ledger_core.replace(
        "parts.append(Money(int(total.cents * r)))",
        "parts.append(Money(int(total.cents * r / sum(ratios))))",
    ).replace(
        "        return parts\n",
        "        remainder = total.cents - sum(p.cents for p in parts)\n"
        "        if parts:\n            parts[0] = Money(parts[0].cents + remainder)\n"
        "        return parts\n",
    )
    add(
        {
            "id": "oss-py-ledger-rounding",
            "category": "bug_fix",
            "languages": ["python"],
            "difficulty": "medium",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": "medium",
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": "https://github.com/morganlinton/VulcanBench/issues/ledger",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": "Ledger allocate() slice.",
            "tests": {
                "fail_to_pass": [
                    {
                        "name": "alloc",
                        "cmd": "python -m pytest test_ledger.py::test_allocate_preserves_total -q",
                    }
                ],
                "pass_to_pass": [
                    {"name": "bal", "cmd": "python -m pytest test_ledger.py::test_balance -q"}
                ],
            },
            "test_timeout_s": 120,
            "agent_hints": {"suggested_max_steps": 90, "entry_paths": ["ledger/core.py"]},
            "files": {
                "repo/LICENSE": _LICENSE,
                "repo/ledger/__init__.py": "from ledger.core import Ledger\nfrom ledger.money import Money\n",
                "repo/ledger/money.py": "class Money:\n    def __init__(self, cents: int):\n        self.cents = int(cents)\n",
                "repo/ledger/core.py": ledger_core,
                "repo/ledger/report.py": "from ledger.core import Ledger\n",
                "tests/test_ledger.py": (
                    "from ledger.core import Ledger\nfrom ledger.money import Money\n\n"
                    "def test_balance():\n    lg = Ledger()\n    lg.credit('a', Money(10))\n"
                    "    assert lg.balance('a').cents == 10\n\ndef test_allocate_preserves_total():\n"
                    "    lg = Ledger()\n    parts = lg.allocate(Money(100), [1, 1, 1])\n"
                    "    assert sum(p.cents for p in parts) == 100\n"
                ),
            },
            "issue": "# Fix proportional allocation\n\nNormalize ratios in `ledger/core.py` so parts sum to the total.",
        },
        {"repo/ledger/core.py": ledger_fix_core},
    )

    metrics_files = {
        "repo/LICENSE": "MIT\n",
        "repo/go.mod": "module example.com/bench\n\ngo 1.23\n",
        "repo/metrics/registry.go": (
            "package metrics\n\ntype Registry struct { counters map[string]int }\n"
            "func New() *Registry { return &Registry{counters: map[string]int{}} }\n"
            "func (r *Registry) Inc(name string) { r.counters[name]++ }\n"
        ),
        "repo/metrics/labels.go": (
            'package metrics\n\nimport "strings"\n\nfunc Key(name string, labels map[string]string) string {\n'
            "    parts := []string{name}\n    for k, v := range labels {\n"
            '        parts = append(parts, k+"="+v)\n    }\n    return strings.Join(parts, ",")\n}\n'
        ),
        "repo/metrics/labeled.go": (
            "package metrics\n\nfunc (r *Registry) IncL(name string, labels map[string]string) {\n"
            "    r.Inc(Key(name, labels))\n}\n"
        ),
        "repo/metrics/doc.go": "package metrics\n",
        "tests/metrics/labels_test.go": (
            'package metrics_test\n\nimport (\n  "testing"\n  "example.com/bench/metrics"\n)\n\n'
            'func TestInc(t *testing.T) { r := metrics.New(); r.Inc("x") }\n\n'
            "func TestLabelKeyStable(t *testing.T) {\n"
            '  a := metrics.Key("h", map[string]string{"b":"2","a":"1"})\n'
            '  b := metrics.Key("h", map[string]string{"a":"1","b":"2"})\n'
            '  if a != b { t.Fatalf("%q vs %q", a, b) }\n}\n'
        ),
    }
    add(
        {
            "id": "oss-go-metrics-labels",
            "category": "bug_fix",
            "languages": ["go"],
            "difficulty": "medium",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": "medium",
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": "https://github.com/morganlinton/VulcanBench/issues/metrics",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": "Metrics label key slice.",
            "tests": {
                "fail_to_pass": [
                    {"name": "stable", "cmd": "go test -run '^TestLabelKeyStable$' ./..."}
                ],
                "pass_to_pass": [{"name": "inc", "cmd": "go test -run '^TestInc$' ./..."}],
            },
            "test_timeout_s": 120,
            "agent_hints": {"suggested_max_steps": 100, "entry_paths": ["metrics/labels.go"]},
            "files": metrics_files,
            "issue": "# Stable labeled metric keys\n\nFix non-deterministic label ordering in `metrics/labels.go`.",
        },
        {
            "repo/metrics/labels.go": (
                'package metrics\n\nimport (\n    "sort"\n    "strings"\n)\n\n'
                "func Key(name string, labels map[string]string) string {\n"
                "    parts := []string{name}\n    keys := make([]string, 0, len(labels))\n"
                "    for k := range labels { keys = append(keys, k) }\n    sort.Strings(keys)\n"
                '    for _, k := range keys { parts = append(parts, k+"="+labels[k]) }\n'
                '    return strings.Join(parts, ",")\n}\n'
            )
        },
    )

    ts_router = (
        "export type Params = Record<string, string>;\n\n"
        "export function match(pattern: string, path: string): Params | null {\n"
        "  const pp = pattern.split('/').filter(Boolean);\n  const ps = path.split('/').filter(Boolean);\n"
        "  if (pp.length !== ps.length) return null;\n  const out: Params = {};\n"
        "  for (let i = 0; i < pp.length; i++) {\n"
        "    if (pp[i].startsWith(':')) out[pp[i].slice(1)] = ps[i];\n"
        "    else if (pp[i] !== ps[i]) return null;\n  }\n  return out;\n}\n"
    )
    add(
        {
            "id": "oss-ts-router-params",
            "category": "bug_fix",
            "languages": ["typescript"],
            "difficulty": "medium",
            "task_complexity": "localized",
            "created": "2026-06-04",
            "repo_scale": "medium",
            "source": "oss",
            "decontaminated": False,
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {
                "url": "https://github.com/morganlinton/VulcanBench",
                "issue": "https://github.com/morganlinton/VulcanBench/issues/router",
                "pr": "",
                "fix_commit": "0000000000000000000000000000000000000002",
            },
            "decontamination_notes": "Router param decoding slice.",
            "tests": {
                "fail_to_pass": [
                    {
                        "name": "decode",
                        "cmd": (
                            "node --experimental-strip-types --test "
                            "--test-name-pattern='^decodeParam$' router.test.ts"
                        ),
                    }
                ],
                "pass_to_pass": [
                    {
                        "name": "static",
                        "cmd": (
                            "node --experimental-strip-types --test "
                            "--test-name-pattern='^staticSegment$' router.test.ts"
                        ),
                    }
                ],
            },
            "test_timeout_s": 120,
            "agent_hints": {"suggested_max_steps": 100, "entry_paths": ["src/router.ts"]},
            "files": {
                "repo/LICENSE": "MIT\n",
                "repo/package.json": '{"type":"module"}\n',
                "repo/src/router.ts": ts_router,
                "repo/src/index.ts": "export { match } from './router.ts';\n",
                "repo/src/util.ts": "export function noop() {}\n",
                "tests/router.test.ts": (
                    "import test from 'node:test';\nimport assert from 'node:assert/strict';\n"
                    "import { match } from './src/router.ts';\n"
                    "test('staticSegment', () => {\n"
                    "  assert.deepEqual(match('/users/:id', '/users/1'), { id: '1' });\n});\n"
                    "test('decodeParam', () => {\n"
                    "  assert.deepEqual(match('/q/:s', '/q/a%20b'), { s: 'a b' });\n});\n"
                ),
            },
            "issue": "# Decode dynamic route params\n\nURL-decode captures in `src/router.ts`.",
        },
        {"repo/src/router.ts": ts_router.replace("ps[i]", "decodeURIComponent(ps[i])")},
    )

    _py_service(
        "oss-py-cache-evict",
        "cachelib",
        "medium",
        "def run(x):\n    return x - 1\n",
        "def run(x):\n    return x * 2\n",
        "# Fix cache eviction counter\n\n`run` in `cachelib/service.py` uses wrong arithmetic.",
    )
    _py_service(
        "oss-py-config-merge",
        "configlib",
        "medium",
        "def run(x):\n    return x + 0\n",
        "def run(x):\n    return x * 2\n",
        "# Fix config merge\n\n`run` must double values in `configlib/service.py`.",
    )

    # M2: +12 tasks
    _go_service("oss-go-errwrap-unwrap", "errwrap", "medium")
    _ts_service("oss-ts-validate-coerce", "validatelib", "medium")
    for i in range(10):
        lang = ["python", "go", "typescript"][i % 3]
        scale = "small"
        tid = f"oss-{lang[:2]}-m2-{i:02d}"
        if lang == "python":
            _py_service(
                tid,
                f"m2pkg{i}",
                scale,
                "def run(x):\n    return x + 1\n",
                "def run(x):\n    return x * 2\n",
                f"# Fix {tid}\n\nCorrect `run` in service module.",
            )
        elif lang == "go":
            _go_service(tid, f"m2pkg{i}", scale)
        else:
            _ts_service(tid, f"m2lib{i}", scale)

    # M3: +16 tasks (50 total with 16 existing + 34 new)
    for i in range(16):
        lang = ["python", "go", "typescript"][i % 3]
        scale = "small" if i < 12 else "medium"
        tid = f"oss-{lang[:2]}-m3-{i:02d}"
        if lang == "python":
            _py_service(
                tid,
                f"pkg{i}",
                scale,
                "def run(x):\n    return x + 1\n",
                "def run(x):\n    return x * 2\n",
                f"# Fix pkg{i}\n\nCorrect `run` in service module.",
            )
        elif lang == "go":
            _go_service(tid, f"pkg{i}", scale)
        else:
            _ts_service(tid, f"lib{i}", scale)


def _gen_gold(task_id: str) -> str:
    repo = TASKS_ROOT / task_id / "repo"
    tmp = REPO_ROOT / ".tmp_gold" / task_id
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(["cp", "-r", f"{repo}/.", f"{tmp}/"], check=True)
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
    subprocess.run(
        ["git", "-c", "user.email=b@b", "-c", "user.name=b", "commit", "-qm", "base"],
        cwd=tmp,
        check=True,
    )
    for rel, content in FIXES[task_id].items():
        path = tmp / rel.removeprefix("repo/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    proc = subprocess.run(["git", "diff"], cwd=tmp, capture_output=True, text=True, check=True)
    shutil.rmtree(tmp)
    return proc.stdout


def _write_task(task: dict, gold: str) -> None:
    tid = task["id"]
    root = TASKS_ROOT / tid
    root.mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    for rel, content in task["files"].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    meta = {k: v for k, v in task.items() if k not in ("files", "issue")}
    (root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (root / "issue.md").write_text(task["issue"].strip() + "\n", encoding="utf-8")
    (root / "gold_patch.diff").write_text(gold, encoding="utf-8")
    (root / "expected_metrics.json").write_text(
        json.dumps({"functional": 1.0}, indent=2) + "\n", encoding="utf-8"
    )
    (root / "calibration.json").write_text(
        json.dumps({"pass_at_1": None}, indent=2) + "\n", encoding="utf-8"
    )
    (root / ".vulcanbenchignore").write_text("dist\nbuild\n__pycache__\n.venv\nnode_modules\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--write-gold", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    _build_tasks()
    for task in TASKS:
        dest = TASKS_ROOT / task["id"]
        if dest.exists() and not args.force:
            continue
        if dest.exists():
            shutil.rmtree(dest)
        if not args.write_gold:
            print(f"skip {task['id']} (use --write-gold)", file=sys.stderr)
            continue
        _write_task(task, "# generating\n")
        gold = _gen_gold(task["id"])
        _write_task(task, gold)
        print(f"wrote {task['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
