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
  --port <port>       HTTP server port. Default: 19741.
  --max-tokens <n>    Maximum completion tokens. Default: 24.
  --skip-build        Use existing .build/release/afm.
  -h, --help          Show this help.

Environment:
  MACAFM_MLX_MODEL_CACHE  Optional model cache root consumed by AFMKit.
  MACAFM_PARITY_MODEL     Default model id when --model is omitted.
USAGE
}

MODEL="${MACAFM_PARITY_MODEL:-}"
PROMPT="Reply exactly with: AFM parity ok"
PORT="19741"
MAX_TOKENS="24"
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
    --port)
      PORT="${2:?missing --port value}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:?missing --max-tokens value}"
      shift 2
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
DIRECT_OUT="$WORK_DIR/direct.out"
HTTP_JSON="$WORK_DIR/http.json"
HTTP_TEXT="$WORK_DIR/http.out"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "==> Direct AFMKit MLX path"
"$AFM_BIN" mlx \
  --model "$MODEL" \
  --single-prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --top-p 1 \
  --seed 1 \
  --no-think \
  >"$DIRECT_OUT"

echo "==> Starting HTTP MLX server on 127.0.0.1:$PORT"
"$AFM_BIN" mlx \
  --model "$MODEL" \
  --port "$PORT" \
  --hostname 127.0.0.1 \
  --prewarm n \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  --top-p 1 \
  --seed 1 \
  --no-think \
  >"$SERVER_LOG" 2>&1 &
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
python3 - "$MODEL" "$PROMPT" "$MAX_TOKENS" <<'PY' >"$WORK_DIR/request.json"
import json
import sys

model = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": max_tokens,
    "seed": 1,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
print(json.dumps(payload))
PY

curl -fsS \
  -H "Content-Type: application/json" \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  --data-binary "@$WORK_DIR/request.json" \
  >"$HTTP_JSON"

python3 - "$HTTP_JSON" >"$HTTP_TEXT" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
try:
    print(payload["choices"][0]["message"].get("content") or "")
except Exception as exc:
    raise SystemExit(f"failed to extract HTTP response content: {exc}\n{json.dumps(payload, indent=2)}")
PY

python3 - "$DIRECT_OUT" "$HTTP_TEXT" "$SERVER_LOG" "$HTTP_JSON" <<'PY'
import pathlib
import sys

direct_path, http_path, server_log, http_json = map(pathlib.Path, sys.argv[1:])

def normalize(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()

direct = normalize(direct_path.read_text(encoding="utf-8"))
http = normalize(http_path.read_text(encoding="utf-8"))

if direct != http:
    print("AFMKit direct/HTTP parity mismatch", file=sys.stderr)
    print(f"\n--- direct ({direct_path}) ---\n{direct}", file=sys.stderr)
    print(f"\n--- http ({http_path}) ---\n{http}", file=sys.stderr)
    print(f"\nServer log: {server_log}", file=sys.stderr)
    print(f"HTTP JSON: {http_json}", file=sys.stderr)
    raise SystemExit(1)

print("AFMKit direct/HTTP parity passed.")
print(f"Output: {direct}")
PY

echo "Artifacts:"
echo "  $WORK_DIR"
