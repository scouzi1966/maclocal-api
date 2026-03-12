#!/bin/bash
# Usage: gen-stats.sh [port] "prompt" [max_tokens]
PORT="${1:-9999}"
PROMPT="${2:-Hello}"
MAX_TOKENS="${3:-6144}"

PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'any',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': int(sys.argv[2]),
    'temperature': 0.0
}))
" "$PROMPT" "$MAX_TOKENS")

START_MS=$(python3 -c "import time; print(int(time.time()*1000))")

RESPONSE=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

END_MS=$(python3 -c "import time; print(int(time.time()*1000))")

python3 -c '
import json, sys, datetime

r = json.loads(sys.argv[1])
wall_ms = int(sys.argv[2]) - int(sys.argv[3])
wall_s = wall_ms / 1000.0

c = r["choices"][0]
u = r["usage"]
msg = c["message"]
content = msg.get("content", "")
reasoning = msg.get("reasoning_content", "")
comp_tok = u["completion_tokens"]
prompt_tok = u.get("prompt_tokens", 0)
speed = comp_tok / wall_s if wall_s > 0 else 0
fp = r.get("system_fingerprint", "")
model = fp.replace("afm_mlx__","").replace("__","/") if fp else r.get("model","")
now = datetime.datetime.now().strftime("%b %d, %Y at %I:%M %p")
G = "\033[92m"
Z = "\033[0m"
DIM = "\033[2m"
SEP = "  " + "\u2500"*40

def row(k, v):
    return "  " + k.ljust(20) + str(v).rjust(21)

# Print full response
print()
if reasoning:
    print(DIM + "  ── Thinking ──" + Z)
    print()
    print(reasoning)
    print()
if content:
    print(DIM + "  ── Output ──" + Z)
    print()
    print(content)
    print()

# Then stats
print("  \U0001F4CA  Generation Stats")
print(SEP)
print(row("Backend", "MLX"))
print(row("Model", model))
spd = G + f"{speed:>17.1f} tok/s" + Z
print("  " + "Speed".ljust(20) + spd)
print(row("Prompt Tokens", prompt_tok))
print(row("Output Tokens", comp_tok))
print(row("Wall Time", f"{wall_s:.2f}s"))
print(SEP)
if reasoning:
    think_tok = len(reasoning) // 4
    think_frac = think_tok / comp_tok if comp_tok else 0
    think_time = wall_s * think_frac
    print(row("Reasoning", ""))
    print(row("Thinking Tokens", f"~{think_tok}"))
    print(row("Thinking Time", f"~{think_time:.2f}s"))
    print(SEP)
print(row("Finish", c["finish_reason"]))
print(row("Generated", now))
print()
' "$RESPONSE" "$END_MS" "$START_MS"
