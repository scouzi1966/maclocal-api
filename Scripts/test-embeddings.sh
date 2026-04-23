#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-10098}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:-apple-nl-contextual-en}"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_server() {
  local port="$1"
  shift

  swift run afm embed --port "${port}" --hostname "${HOST}" "$@" >/tmp/afm-embed.log 2>&1 &
  SERVER_PID=$!

  for _ in $(seq 1 30); do
    if curl -fsS "http://${HOST}:${port}/v1/models" >/tmp/afm-embed-models.json 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "Embeddings server failed to start on ${HOST}:${port}" >&2
  cat /tmp/afm-embed.log >&2 || true
  return 1
}

echo "Starting embeddings server on ${HOST}:${PORT} using model ${MODEL}"
start_server "${PORT}" --model "${MODEL}"

curl -fsS "http://${HOST}:${PORT}/v1/models" >/tmp/afm-embed-models.json
echo "Models endpoint is live"

single_response="$(curl -fsS -H 'Content-Type: application/json' \
  -d "{\"input\":\"hello world\",\"model\":\"${MODEL}\"}" \
  "http://${HOST}:${PORT}/v1/embeddings")"

python3 - <<'PY' "${single_response}"
import json, sys
payload = json.loads(sys.argv[1])
assert payload["object"] == "list"
assert len(payload["data"]) == 1
assert len(payload["data"][0]["embedding"]) > 0
assert payload["usage"]["prompt_tokens"] > 0
PY
echo "Single-string embeddings request passed"

array_response="$(curl -fsS -H 'Content-Type: application/json' \
  -d "{\"input\":[\"hello\",\"world\"],\"model\":\"${MODEL}\"}" \
  "http://${HOST}:${PORT}/v1/embeddings")"

python3 - <<'PY' "${array_response}"
import json, sys
payload = json.loads(sys.argv[1])
assert len(payload["data"]) == 2
assert payload["data"][0]["index"] == 0
assert payload["data"][1]["index"] == 1
PY
echo "Array embeddings request passed"

truncated_headers="$(mktemp)"
truncated_body="$(mktemp)"
curl -fsS -D "${truncated_headers}" -o "${truncated_body}" -H 'Content-Type: application/json' \
  -d "{\"input\":\"hello world\",\"model\":\"${MODEL}\",\"dimensions\":64,\"encoding_format\":\"base64\"}" \
  "http://${HOST}:${PORT}/v1/embeddings"

python3 - <<'PY' "${truncated_body}"
import json, sys, base64, struct
payload = json.load(open(sys.argv[1]))
encoded = payload["data"][0]["embedding"]
raw = base64.b64decode(encoded)
assert len(raw) == 64 * 4
vals = struct.unpack("<" + "f" * 64, raw)
assert len(vals) == 64
PY
echo "Base64 + dimensions request passed"

echo "Embeddings smoke test passed"
