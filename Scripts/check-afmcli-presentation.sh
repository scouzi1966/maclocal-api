#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CACHE="$ROOT_DIR/.build/afmcli-presentation-cache"
MODEL_DIR="$MODEL_CACHE/acme/vision-reasoning-model"
AFM_BIN="$ROOT_DIR/.build/debug/afm"

rm -rf "$MODEL_CACHE"
mkdir -p "$MODEL_DIR"

cat > "$MODEL_DIR/config.json" <<'JSON'
{
  "max_position_embeddings": 65536,
  "vision_config": {
    "model_type": "vision"
  }
}
JSON

cat > "$MODEL_DIR/tokenizer_config.json" <<'JSON'
{
  "chat_template": "{% if tools %}<tool_call>{% endif %}<think>{{ messages }}"
}
JSON

printf "weights" > "$MODEL_DIR/weights.safetensors"

if [[ "${AFMCLI_PRESENTATION_SKIP_BUILD:-0}" != "1" ]]; then
  swift build --product afm >/dev/null
elif [[ ! -x "$AFM_BIN" ]]; then
  echo "AFMCLI_PRESENTATION_SKIP_BUILD=1 set, but $AFM_BIN does not exist" >&2
  exit 1
fi

OPENCLAW_JSON="$(
  MACAFM_MLX_MODEL_CACHE="$MODEL_CACHE" "$AFM_BIN" mlx \
    -m acme/vision-reasoning-model \
    --openclaw-config
)"

OPENCLAW_JSON="$OPENCLAW_JSON" /usr/bin/python3 - <<'PY'
import json
import os

payload = json.loads(os.environ["OPENCLAW_JSON"])
model = payload["models"]["providers"]["afm"]["models"][0]
defaults = payload["agents"]["defaults"]["model"]

assert model["id"] == "vision-reasoning-model", model
assert model["name"] == "vision-reasoning-model (afm)", model
assert model["reasoning"] is True, model
assert model["input"] == ["text", "image"], model
assert model["contextWindow"] == 65536, model
assert model["maxTokens"] == 8192, model
assert defaults["primary"] == "afm/vision-reasoning-model", defaults
PY

echo "AFMCLI presentation gate passed"
