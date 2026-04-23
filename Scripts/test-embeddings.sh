#!/usr/bin/env bash
set -euo pipefail

WITH_MLX=0
PORT="${PORT:-10098}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:-apple-nl-contextual-en}"
MLX_MODEL="${MLX_MODEL:-}"
SERVER_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-mlx)
      WITH_MLX=1
      shift
      ;;
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

discover_mlx_model() {
  python3 - <<'PY'
from pathlib import Path
root = Path.home()/'.cache'/'huggingface'/'hub'
if not root.exists():
    raise SystemExit(0)
for path in sorted(root.glob('models--mlx-community--*')):
    name = path.name.lower()
    if any(token in name for token in ['embed', 'embedding', 'bge', 'gte', 'minilm', 'e5', 'mxbai', 'nomic']):
        print(path.name.replace('models--', '').replace('--', '/', 1))
        raise SystemExit(0)
PY
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

if [[ "${WITH_MLX}" -eq 1 ]]; then
  if [[ -z "${MLX_MODEL}" ]]; then
    MLX_MODEL="$(discover_mlx_model || true)"
  fi

  if [[ -z "${MLX_MODEL}" ]]; then
    echo "Skipping MLX smoke test: no cached mlx-community embedding model found. Set MLX_MODEL to override."
    echo "Embeddings smoke test passed"
    exit 0
  fi

  echo "Restarting server with MLX backend using model ${MLX_MODEL}"
  cleanup
  SERVER_PID=""

  start_server "${PORT}" --backend mlx --model "${MLX_MODEL}"

  mlx_response="$(curl -fsS -H 'Content-Type: application/json' \
    -d "{\"input\":\"hello world\",\"model\":\"${MLX_MODEL}\"}" \
    "http://${HOST}:${PORT}/v1/embeddings")"

  python3 - <<'PY' "${mlx_response}" "${MLX_MODEL}"
import json, sys
payload = json.loads(sys.argv[1])
model = sys.argv[2]
assert payload["model"] == model
assert payload["object"] == "list"
assert len(payload["data"]) == 1
assert len(payload["data"][0]["embedding"]) > 0
assert payload["usage"]["prompt_tokens"] > 0
PY
  echo "MLX embeddings request passed"
fi

echo "Embeddings smoke test passed"
