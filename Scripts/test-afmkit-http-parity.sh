#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  Scripts/test-afmkit-http-parity.sh --model <org/model> [options]

Compares the direct AFMKit MLX single-prompt path with the OpenAI-compatible
HTTP /v1/chat/completions path on the same local model.

Options:
  --model <id>        MLX model id or path. Also read from MACAFM_PARITY_MODEL.
  --prompt <text>     Prompt to send to both paths.
  --instructions <s>  System instructions for both paths.
  --port <port>       HTTP server port. Default: 19741.
  --max-tokens <n>    Maximum completion tokens. Default: 24.
  --max-logprobs <n>  Request top logprobs from both paths. Default: 0.
  --tools-json <json> OpenAI-compatible tools array for both paths.
  --enable-thinking   Request model reasoning instead of the default no-think mode.
  --skip-build        Use existing .build/release/afm.
  -h, --help          Show this help.

Environment:
  MACAFM_MLX_MODEL_CACHE  Optional model cache root consumed by AFMKit.
  MACAFM_PARITY_MODEL     Default model id when --model is omitted.
USAGE
}

MODEL="${MACAFM_PARITY_MODEL:-}"
PROMPT="Reply exactly with: AFM parity ok"
INSTRUCTIONS="You are a helpful assistant"
PORT="19741"
MAX_TOKENS="24"
MAX_LOGPROBS="0"
TOOLS_JSON=""
ENABLE_THINKING="0"
SKIP_BUILD="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?missing --model value}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:?missing --prompt value}"
      shift 2
      ;;
    --instructions)
      INSTRUCTIONS="${2:?missing --instructions value}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing --port value}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:?missing --max-tokens value}"
      shift 2
      ;;
    --max-logprobs)
      MAX_LOGPROBS="${2:?missing --max-logprobs value}"
      shift 2
      ;;
    --tools-json)
      TOOLS_JSON="${2:?missing --tools-json value}"
      shift 2
      ;;
    --enable-thinking)
      ENABLE_THINKING="1"
      shift
      ;;
    --skip-build)
      SKIP_BUILD="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "error: --model is required unless MACAFM_PARITY_MODEL is set" >&2
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AFM_BIN="$ROOT_DIR/.build/release/afm"
if [[ "$SKIP_BUILD" != "1" || ! -x "$AFM_BIN" ]]; then
  swift build -c release --product afm
fi

if [[ -z "${MACAFM_MLX_METALLIB:-}" ]]; then
  for candidate in \
    "$ROOT_DIR/.build/out/Products/Release/MacLocalAPI_AFMKitMLX.bundle/Contents/Resources/default.metallib" \
    "$ROOT_DIR/.build/out/Products/Release/MacLocalAPI_AFMKit.bundle/Contents/Resources/default.metallib" \
    "$ROOT_DIR/.build/release/MacLocalAPI_AFMKitMLX.bundle/default.metallib" \
    "$ROOT_DIR/.build/release/MacLocalAPI_AFMKit.bundle/default.metallib" \
    "$ROOT_DIR/Sources/AFMKitMLX/Resources/default.metallib"
  do
    if [[ -f "$candidate" ]]; then
      export MACAFM_MLX_METALLIB="$candidate"
      break
    fi
  done
fi

if [[ -z "${MACAFM_MLX_METALLIB:-}" ]]; then
  echo "error: default.metallib not found; set MACAFM_MLX_METALLIB=/path/to/default.metallib" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/afmkit-http-parity.XXXXXX")"
SERVER_LOG="$WORK_DIR/server.log"
DIRECT_JSON="$WORK_DIR/direct.json"
HTTP_JSON="$WORK_DIR/http.json"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> Direct AFMKit MLX path"
DIRECT_ARGS=(
  mlx
  --model "$MODEL" \
  --single-prompt "$PROMPT" \
  --instructions "$INSTRUCTIONS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --top-p 1 \
  --seed 1 \
  --json
)
if [[ "$ENABLE_THINKING" == "1" ]]; then
  DIRECT_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":true}')
else
  DIRECT_ARGS+=(--no-think)
fi
if [[ "$MAX_LOGPROBS" != "0" ]]; then
  DIRECT_ARGS+=(--max-logprobs "$MAX_LOGPROBS")
fi
if [[ -n "$TOOLS_JSON" ]]; then
  DIRECT_ARGS+=(--tools-json "$TOOLS_JSON")
fi
"$AFM_BIN" "${DIRECT_ARGS[@]}" >"$DIRECT_JSON"

echo "==> Starting HTTP MLX server on 127.0.0.1:$PORT"
SERVER_ARGS=(
  mlx
  --model "$MODEL" \
  --instructions "$INSTRUCTIONS" \
  --port "$PORT" \
  --hostname 127.0.0.1 \
  --prewarm n \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --top-p 1 \
  --seed 1
)
if [[ "$ENABLE_THINKING" == "1" ]]; then
  SERVER_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":true}')
else
  SERVER_ARGS+=(--no-think)
fi
if [[ "$MAX_LOGPROBS" != "0" ]]; then
  SERVER_ARGS+=(--max-logprobs "$MAX_LOGPROBS")
fi
"$AFM_BIN" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "error: server exited before health check passed" >&2
    cat "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "error: timed out waiting for HTTP server health check" >&2
  cat "$SERVER_LOG" >&2
  exit 1
fi

echo "==> HTTP /v1/chat/completions path"
python3 - "$MODEL" "$PROMPT" "$INSTRUCTIONS" "$MAX_TOKENS" "$MAX_LOGPROBS" "$TOOLS_JSON" "$ENABLE_THINKING" <<'PY' >"$WORK_DIR/request.json"
import json
import sys

model = sys.argv[1]
prompt = sys.argv[2]
instructions = sys.argv[3]
max_tokens = int(sys.argv[4])
max_logprobs = int(sys.argv[5])
tools_json = sys.argv[6]
enable_thinking = sys.argv[7] == "1"
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": max_tokens,
    "seed": 1,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": enable_thinking},
}
if max_logprobs > 0:
    payload["logprobs"] = True
    payload["top_logprobs"] = max_logprobs
if tools_json:
    payload["tools"] = json.loads(tools_json)
print(json.dumps(payload))
PY

curl -fsS \
  -H "Content-Type: application/json" \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  --data-binary "@$WORK_DIR/request.json" \
  >"$HTTP_JSON"

python3 - "$DIRECT_JSON" "$HTTP_JSON" "$SERVER_LOG" <<'PY'
import json
import pathlib
import sys

direct_path, http_path, server_log = map(pathlib.Path, sys.argv[1:])

VOLATILE_USAGE_KEYS = {
    "completion_time",
    "prompt_time",
    "total_time",
    "completion_tokens_per_second",
    "prompt_tokens_per_second",
    "peak_memory_gib",
}

def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_tool_calls(calls):
    result = []
    for call in calls or []:
        function = call.get("function") or {}
        result.append({
            "type": call.get("type"),
            "function": {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            },
        })
    return result

def normalize_usage(usage):
    usage = dict(usage or {})
    for key in VOLATILE_USAGE_KEYS:
        usage.pop(key, None)
    return usage

def canonical(payload):
    choices = payload.get("choices") or []
    if not choices:
        raise SystemExit(f"missing choices in response:\n{json.dumps(payload, indent=2)}")
    choice = choices[0]
    message = choice.get("message") or {}
    return {
        "model": payload.get("model"),
        "system_fingerprint": payload.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "message": {
            "role": message.get("role"),
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "tool_calls": normalize_tool_calls(message.get("tool_calls")),
        },
        "logprobs": choice.get("logprobs"),
        "usage": normalize_usage(payload.get("usage")),
    }

direct = canonical(load(direct_path))
http = canonical(load(http_path))

if direct != http:
    print("AFMKit direct/HTTP contract parity mismatch", file=sys.stderr)
    print(f"\n--- direct canonical ({direct_path}) ---", file=sys.stderr)
    print(json.dumps(direct, indent=2, sort_keys=True), file=sys.stderr)
    print(f"\n--- http canonical ({http_path}) ---", file=sys.stderr)
    print(json.dumps(http, indent=2, sort_keys=True), file=sys.stderr)
    print(f"\nServer log: {server_log}", file=sys.stderr)
    raise SystemExit(1)

content = direct["message"]["content"] or direct["message"]["reasoning_content"]
print("AFMKit direct/HTTP contract parity passed.")
print(f"Output: {content}")
PY

echo "Artifacts:"
echo "  $WORK_DIR"
