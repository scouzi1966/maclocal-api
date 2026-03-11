#!/usr/bin/env python3
"""Generate summary table from toolcall-matrix results JSONL."""
import json, sys
from collections import defaultdict

jsonl_file = sys.argv[1]

results = []
with open(jsonl_file) as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

if not results:
    print("  No results collected.")
    sys.exit(0)

# Group by model + config + cache
groups = defaultdict(list)
for r in results:
    key = (r['model'].replace('mlx-community/', ''), r['config'], r['cache'])
    groups[key].append(r)

print(f"\n  {'Model':<28} {'Config':<22} {'Cache':<7} {'Reqs':>4} {'TC✓':>4} {'TC%':>5} {'AvgPN':>7} {'AvgPms':>8}")
print(f"  {'─'*28} {'─'*22} {'─'*7} {'─'*4} {'─'*4} {'─'*5} {'─'*7} {'─'*8}")

for (model, config, cache), items in sorted(groups.items()):
    n = len(items)
    tc_valid = sum(1 for i in items if i.get('valid_tool_call'))
    tc_pct = f"{tc_valid/n*100:.0f}%" if n > 0 else "0%"
    avg_pn = sum(i['prompt_n'] for i in items) / n if n > 0 else 0
    avg_pms = sum(i['prompt_ms'] for i in items) / n if n > 0 else 0
    print(f"  {model:<28} {config:<22} {cache:<7} {n:>4} {tc_valid:>4} {tc_pct:>5} {avg_pn:>7.0f} {avg_pms:>7.1f}ms")

# Cache comparison
print(f"\n  Cache Impact (token savings):")
print(f"  {'Model':<28} {'Config':<22} {'Cache ON':>10} {'Cache OFF':>10} {'Saved':>6}")
print(f"  {'─'*28} {'─'*22} {'─'*10} {'─'*10} {'─'*6}")

cache_groups = defaultdict(dict)
for (model, config, cache), items in groups.items():
    avg_pn = sum(i['prompt_n'] for i in items) / len(items) if items else 0
    cache_groups[(model, config)][cache] = avg_pn

for (model, config), caches in sorted(cache_groups.items()):
    if 'cache' in caches and 'nocache' in caches:
        on = caches['cache']
        off = caches['nocache']
        saved = f"{(off-on)/off*100:.0f}%" if off > 0 else "N/A"
        print(f"  {model:<28} {config:<22} {on:>9.0f} {off:>9.0f}  {saved:>5}")
