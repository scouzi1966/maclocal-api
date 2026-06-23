#!/usr/bin/env bash
#
# Generate the larger AgentBlaster suite set used by run-benchmark.sh --mode generated.
#
# AgentBlaster's built-in suites are thin capability probes (1-4 cases each, ~26
# total). The real volume comes from its generators:
#   - `agentblaster agents suite`    → representative agent-workflow suites
#                                       (opencode/openclaw/hermes/pi/codex/cline/aider/continue)
#   - `agentblaster harness generate`→ deterministic harness-engineering suites, multiplied
#                                       by --repeats (prefill/cache-replay, concurrency,
#                                       contract-fuzz, metamorphic, orchestration, ...)
#
# This script writes them into ./suites/*.yaml (deterministic given the seeds/repeats),
# producing ~94 cases. Re-run to regenerate. Requires `agentblaster` on PATH.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/suites"
mkdir -p "$OUT"
command -v agentblaster >/dev/null 2>&1 || { echo "FATAL: agentblaster not installed"; exit 1; }

echo "=== agent-workflow profiles → $OUT ==="
for p in opencode openclaw hermes pi codex cline aider continue; do
  agentblaster agents suite --profile "$p" --output "$OUT/agent-$p.yaml" >/dev/null 2>&1 \
    && echo "  agent-$p.yaml"
done

echo "=== harness generators (with --repeats for volume) → $OUT ==="
# profile|source-suite|repeats
gen() { agentblaster harness generate --profile "$1" --suite "$2" --repeats "$3" \
          --output "$OUT/harness-$1.yaml" >/dev/null 2>&1 && echo "  harness-$1.yaml"; }
gen orchestration agentic-tool-loop 3
gen concurrency   agent-fanout      3
gen metamorphic   toolcall          4
gen contract-fuzz structured        3
gen cache-replay  prefill           3

echo ""
echo "=== case counts ==="
TOTAL=0
for f in "$OUT"/*.yaml; do
  n=$(python3 -c "import yaml;print(len(yaml.safe_load(open('$f')).get('cases',[])))" 2>/dev/null || echo 0)
  TOTAL=$((TOTAL+n)); printf "  %-30s %s\n" "$(basename "$f")" "$n"
done
echo "  ----------------------------------------"
echo "  TOTAL: $TOTAL cases"
