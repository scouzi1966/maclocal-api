#!/bin/bash
#
# ── LLM INSTRUCTIONS ─────────────────────────────────────────────────────────
# If you are an LLM (Claude, GPT, etc.) generating test cases for this script:
#
# PURPOSE: This is a testing framework for AFM (Apple Foundation Models), an
# OpenAI-compatible local inference server for Apple Silicon using MLX models.
# It starts a server per model, sends prompts, and records everything into a
# JSONL results file and HTML report for human review.
#
# TO GENERATE TESTS: Create a plain text prompts file and run:
#   mlx-model-test.sh --models --prompts <your-file.txt>
#   mlx-model-test.sh --model org/model --prompts <your-file.txt>
#   mlx-model-test.sh --models --prompts <your-file.txt> --smart
#
# PROMPTS FILE FORMAT:
#   - Lines starting with # are comments. Blank lines are ignored.
#   - Before any [section]: global defaults (apply to all models)
#   - [all] section: prompts sent to every model
#   - [org/model] section: per-model overrides and extra prompts
#   - [org/model @ label] section: named variant for A/B testing
#     (restarts the server with different settings for the same model)
#
# KNOWN PARAMETERS (parsed as config, not prompts):
#   max_tokens:          Max tokens to generate (int)
#   temperature:         Sampling temperature (float, 0.0-2.0)
#   top_p:               Nucleus sampling threshold (float, 0.0-1.0)
#   top_k:               Top-k sampling (int, 0 = disabled) [non-standard, via extra_body]
#   min_p:               Min-p sampling filter (float, 0.0-1.0) [non-standard, via extra_body]
#   seed:                Random seed for reproducibility (int)
#   logprobs:            Return log probabilities (true/false)
#   top_logprobs:        Number of top logprobs per token (int, 0-20)
#   presence_penalty:    Presence penalty (float)
#   repetition_penalty:  Repetition penalty (float) [non-standard, via extra_body]
#   frequency_penalty:   Frequency penalty (float)
#   stop:                Comma-separated stop sequences (e.g. stop: </s>,<|end|>)
#   response_format:     text | json_object
#   system:              System prompt text
#   afm:                 CLI flags passed to the afm server process, e.g.:
#                          --verbose / --very-verbose
#                          --no-streaming
#                          --raw            skip <think> tag extraction
#                          --enable-prefix-caching
#                          --tool-call-parser <hermes|llama3_json|gemma|mistral|qwen3_xml>
#                          --fix-tool-args
#   skip                 Skip this model entirely
#
# ANY OTHER LINE in a section is treated as a prompt.
#
# WHAT GETS RECORDED: Every test result in the JSONL output includes:
#   model, label, prompt, status, error, load_time_s, gen_time_s,
#   prompt_tokens, completion_tokens, total_tokens, tokens_per_sec,
#   content_preview, content, reasoning_content, temperature, max_tokens,
#   system_prompt, afm_args, plus any optional params that were set:
#   top_p, top_k, min_p, seed, logprobs, top_logprobs, presence_penalty,
#   repetition_penalty, frequency_penalty, stop, response_format
# All parameters are preserved end-to-end from prompts file → JSONL → HTML
# report, so humans can see exactly what settings produced each result.
#
# EXAMPLE — basic test file:
#   [all]
#   What is the capital of France?
#
# EXAMPLE — regression test with specific models:
#   max_tokens: 500
#   temperature: 0.0
#   [all]
#   Respond with exactly: Hello World
#   [mlx-community/SmolLM3-3B-4bit]
#   [mlx-community/gemma-3-4b-it-8bit]
#
# EXAMPLE — A/B sampling comparison:
#   [all]
#   Write a poem about the ocean.
#   [mlx-community/SmolLM3-3B-4bit @ greedy]
#   temperature: 0.0
#   [mlx-community/SmolLM3-3B-4bit @ creative]
#   temperature: 1.5
#   top_p: 0.95
#   min_p: 0.02
#
# EXAMPLE — tool calling test:
#   max_tokens: 1000
#   [mlx-community/Qwen3-Coder-Next-4bit]
#   afm: --tool-call-parser qwen3_xml
#   system: You have access to tools. Use the get_weather function.
#   What is the weather in Paris?
#
# TIPS FOR LLMs GENERATING TESTS:
#   - Use temperature: 0.0 for deterministic/regression tests
#   - Use the @ variant syntax to compare sampling strategies
#   - Use afm: --verbose to capture debug logs in /tmp/mlx-server-*.log
#   - Keep prompts short for fast iteration; use max_tokens: 200-500
#   - For tool calling tests, set the system prompt with tool definitions
#     and use afm: --tool-call-parser to select the right parser
#   - Add --smart to the CLI to get AI-powered analysis of results
#   - Add --no-report to skip HTML generation (JSONL only) — use this for
#     automated loops, auto-tuning, or when parsing results programmatically
#   - The HTML report auto-opens in the browser when tests complete
# ─────────────────────────────────────────────────────────────────────────────
#
# MLX Model Test Suite
# Tests each model by starting the server, sending a prompt, and collecting results.
#
# Usage:
#   mlx-model-test.sh                                    # use hardcoded model list
#   mlx-model-test.sh --models                           # auto-discover from cache
#   mlx-model-test.sh --models --model org/name          # test single model from cache
#   mlx-model-test.sh --prompts prompts.txt              # custom prompts file
#   mlx-model-test.sh --models --prompts prompts.txt     # auto-discover + custom prompts
#   mlx-model-test.sh --models --smart                   # auto-discover + AI analysis

set -uo pipefail

AFM="${AFM_BIN:-afm}"
export MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-/Volumes/edata/models/vesta-test-cache}"
PORT=9877
DEFAULT_PROMPT="Explain calculus concepts from limits through multivariable calculus with rigorous mathematical notation"
RESULTS_FILE="/tmp/mlx-test-results.jsonl"
DEFAULT_MAX_TOKENS=5000
DEFAULT_TEMPERATURE=0.7
TIMEOUT_LOAD=360     # seconds to wait for server to start (6 min)
TIMEOUT_GENERATE=900 # seconds for generation
USE_AUTO_MODELS=false
PROMPTS_FILE=""
SINGLE_MODEL=""
SMART_ANALYSIS=false
SMART_TOOLS="claude"
NO_REPORT=false

# Parse arguments
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
  case "${args[$i]}" in
    --models) USE_AUTO_MODELS=true ;;
    --model)
      i=$((i + 1))
      if [ $i -ge ${#args[@]} ]; then
        echo "Error: --model requires a value (e.g. mlx-community/SmolLM3-3B-4bit)" >&2
        exit 1
      fi
      SINGLE_MODEL="${args[$i]}"
      ;;
    --prompts)
      i=$((i + 1))
      if [ $i -ge ${#args[@]} ]; then
        echo "Error: --prompts requires a file path" >&2
        exit 1
      fi
      PROMPTS_FILE="${args[$i]}"
      if [ ! -f "$PROMPTS_FILE" ]; then
        echo "Error: Prompts file not found: $PROMPTS_FILE" >&2
        exit 1
      fi
      ;;
    --smart)
      SMART_ANALYSIS=true
      # Check if next arg is a tool list (not another flag)
      if [ $((i + 1)) -lt ${#args[@]} ] && [[ "${args[$((i + 1))]}" != --* ]]; then
        i=$((i + 1))
        SMART_TOOLS="${args[$i]}"
      fi
      ;;
    --no-report) NO_REPORT=true ;;
    -h|--help)
      cat <<'HELPEOF'
MLX Model Test Suite
====================
Tests local MLX models by starting the afm server, sending prompts,
and collecting performance metrics + responses into an HTML report.

OPTIONS
  --models              Auto-discover all models in MACAFM_MLX_MODEL_CACHE
  --model <org/name>    Test a single specific model only
  --prompts <file>      Plain text file with custom prompts and parameters
  --smart [tools]       Run AI analysis on results (default: claude)
                        Comma-separated list of CLI tools, e.g. claude,codex,qwen
  --no-report           Skip HTML report generation (JSONL only, for automation)
  -h, --help            Show this help

ENVIRONMENT
  MACAFM_MLX_MODEL_CACHE   Path to model cache (default: /Volumes/edata/models/vesta-test-cache)
  AFM_BIN                  Path to afm binary (default: afm, assumes it's in PATH)

EXAMPLES

  1. Basic — run the built-in test suite (hardcoded model list):

       mlx-model-test.sh

  2. Auto-discover — test every model in your cache:

       mlx-model-test.sh --models

  3. Single model — quick test of one model:

       mlx-model-test.sh --model mlx-community/SmolLM3-3B-4bit

  4. Custom prompts — provide your own test prompts:

       mlx-model-test.sh --models --prompts my-tests.txt

  5. Single model + custom prompts:

       mlx-model-test.sh --model mlx-community/Qwen3-Coder-Next-4bit --prompts coding-tests.txt

  6. Full pipeline — auto-discover + custom prompts + AI analysis:

       mlx-model-test.sh --models --prompts my-tests.txt --smart

  7. Multi-tool AI analysis — compare Claude vs Codex vs other CLIs:

       mlx-model-test.sh --models --smart claude,codex
       mlx-model-test.sh --model mlx-community/SmolLM3-3B-4bit --smart claude,codex,qwen

PROMPTS FILE FORMAT

  A plain text file you can create with vi, nano, or any editor.
  Lines starting with # are comments. Blank lines are ignored.

  Structure:
    - Lines before any [section] are global defaults
    - [all] section: prompts that run for every model
    - [org/model] section: per-model config and extra prompts
    - [org/model @ label] section: named variant — reruns the same model
      with different settings (each variant = separate server start)

  Known parameters (parsed as config, not prompts):
    max_tokens:          Max tokens to generate (e.g. max_tokens: 2000)
    temperature:         Sampling temperature (e.g. temperature: 0.3)
    top_p:               Nucleus sampling (e.g. top_p: 0.9)
    top_k:               Top-k sampling (e.g. top_k: 50)
    min_p:               Min-p filter (e.g. min_p: 0.05)
    seed:                Random seed (e.g. seed: 42)
    logprobs:            Return logprobs (e.g. logprobs: true)
    top_logprobs:        Top logprobs per token (e.g. top_logprobs: 5)
    presence_penalty:    Presence penalty (e.g. presence_penalty: 0.5)
    repetition_penalty:  Repetition penalty (e.g. repetition_penalty: 1.1)
    frequency_penalty:   Frequency penalty (e.g. frequency_penalty: 0.5)
    stop:                Comma-separated stop sequences (e.g. stop: </s>,<|end|>)
    response_format:     Response format (e.g. response_format: json_object)
    system:              System prompt (e.g. system: You are a helpful assistant)
    afm:                 Extra CLI flags passed to afm server
                         (e.g. afm: --verbose --enable-prefix-caching)
    skip                 Skip this model/variant entirely (no value needed)

  Everything else in a section is treated as a prompt.

  Note: top_p, presence_penalty, frequency_penalty, seed, logprobs,
  top_logprobs, stop, and response_format are sent as standard OpenAI API
  params. top_k, min_p, and repetition_penalty are sent via extra_body
  (non-standard params supported by AFM).

  afm CLI flags you can pass via afm: include:
    --verbose / --very-verbose  Debug logging
    --no-streaming              Disable streaming
    --raw                       Skip <think> tag extraction
    --enable-prefix-caching     KV cache reuse across requests
    --tool-call-parser <name>   Force tool call parser
    --chat-template <tpl>       Override chat template

  EXAMPLE — minimal (same prompt for all models):

    [all]
    What is 2+2?

  EXAMPLE — with defaults and global afm flags:

    max_tokens: 1000
    temperature: 0.7
    afm: --enable-prefix-caching

    [all]
    What is 2+2? Explain step by step.
    Write a haiku about machine learning.

  EXAMPLE — per-model overrides with CLI flags:

    max_tokens: 2000
    temperature: 0.7

    [all]
    Explain quantum entanglement in simple terms.

    [mlx-community/Qwen3-Coder-Next-4bit]
    max_tokens: 3000
    temperature: 0.3
    top_k: 40
    min_p: 0.05
    system: You are a senior Python developer.
    Write a function to find the longest palindrome in a string.

    [mlx-community/Kimi-K2.5-3bit]
    max_tokens: 500
    afm: --verbose
    Why is the sky blue? Be brief.

    [mlx-community/Qwen3.5-397B-A17B-4bit]
    skip

  EXAMPLE — same model, multiple variants (A/B testing):

    [all]
    Write a short story about a robot discovering music.

    # Default sampling
    [mlx-community/SmolLM3-3B-4bit @ default]
    temperature: 0.7

    # Conservative sampling
    [mlx-community/SmolLM3-3B-4bit @ conservative]
    temperature: 0.3
    top_k: 20
    min_p: 0.1

    # Creative sampling
    [mlx-community/SmolLM3-3B-4bit @ creative]
    temperature: 1.2
    top_p: 0.95

    # With prefix caching
    [mlx-community/SmolLM3-3B-4bit @ cached]
    temperature: 0.7
    afm: --enable-prefix-caching
    Write a short story about a robot discovering music.
    Now continue the story with a second chapter.

  How variants work:
    - Each [model @ label] section starts a fresh server instance
    - The label appears in the report so you can compare results
    - Variants get the [all] prompts PLUS any section-specific prompts
    - Use this to A/B test sampling strategies, CLI flags, system prompts, etc.
    - A plain [model] section (no @ label) works as before

OUTPUT
  Results are saved to ./test-reports/ in the current directory:
    - HTML report (auto-opens in browser)
    - JSONL raw data
    - Smart analysis markdown (with --smart, also auto-opens)

  The 5 largest models (by disk size) always run last,
  with the biggest model running very last.

HELPEOF
      exit 0
      ;;
    *) echo "Unknown option: ${args[$i]}"; exit 1 ;;
  esac
  i=$((i + 1))
done

# Validate openai SDK prerequisite
if ! python3 -c "import openai" 2>/dev/null; then
  echo "Error: OpenAI Python SDK not found. Install with: pip3 install openai" >&2
  exit 1
fi

# Validate --smart prerequisites
if $SMART_ANALYSIS; then
  IFS=',' read -ra SMART_TOOL_LIST <<< "$SMART_TOOLS"
  for tool in "${SMART_TOOL_LIST[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "Error: --smart tool '$tool' not found in PATH" >&2
      exit 1
    fi
  done
  echo "Smart analysis tools: ${SMART_TOOL_LIST[*]}"
fi

# Check for stale server on our port
STALE_PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
  echo "Warning: found existing process on port $PORT (PID $STALE_PID) — killing it"
  kill -KILL $STALE_PID 2>/dev/null || true
  sleep 1
fi

> "$RESULTS_FILE"

# ── Parse prompts file into JSON config ───────────────────────────────────────

PARSED_CONFIG=""
if [ -n "$PROMPTS_FILE" ]; then
  PARSED_CONFIG=$(python3 - "$PROMPTS_FILE" <<'PYEOF'
import sys, json, re

filepath = sys.argv[1]
config = {
    "defaults": {},
    "all": [],
    "runs": []   # ordered list of {model, label, prompts, params, skip, afm}
}

# Track model sections for merging into runs at the end
model_sections = {}  # key = "model" or "model @ label"

current_section = None  # None = defaults, "all", or section key

with open(filepath) as f:
    for raw_line in f:
        line = raw_line.strip()

        # Skip comments and blank lines
        if not line or line.startswith('#'):
            continue

        # Section header: [all] or [org/model] or [org/model @ label]
        m = re.match(r'^\[(.+)\]$', line)
        if m:
            section_name = m.group(1).strip()
            if section_name == 'all':
                current_section = 'all'
            else:
                current_section = section_name
                if section_name not in model_sections:
                    # Parse "org/model @ label" vs "org/model"
                    if ' @ ' in section_name:
                        model_id, label = section_name.split(' @ ', 1)
                        model_id = model_id.strip()
                        label = label.strip()
                    else:
                        model_id = section_name
                        label = ""
                    model_sections[section_name] = {
                        'model': model_id,
                        'label': label,
                        'prompts': [],
                        'params': {},
                        'afm': '',
                        'skip': False
                    }
            continue

        # Before any section = defaults
        if current_section is None:
            if line.startswith('max_tokens:'):
                config['defaults']['max_tokens'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('temperature:'):
                config['defaults']['temperature'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('top_p:'):
                config['defaults']['top_p'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('top_k:'):
                config['defaults']['top_k'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('min_p:'):
                config['defaults']['min_p'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('seed:'):
                config['defaults']['seed'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('logprobs:'):
                config['defaults']['logprobs'] = line.split(':', 1)[1].strip().lower() == 'true'
            elif line.startswith('top_logprobs:'):
                config['defaults']['top_logprobs'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('presence_penalty:'):
                config['defaults']['presence_penalty'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('repetition_penalty:'):
                config['defaults']['repetition_penalty'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('frequency_penalty:'):
                config['defaults']['frequency_penalty'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('stop:'):
                config['defaults']['stop'] = [s.strip() for s in line.split(':', 1)[1].split(',')]
            elif line.startswith('response_format:'):
                config['defaults']['response_format'] = line.split(':', 1)[1].strip()
            elif line.startswith('system:'):
                config['defaults']['system'] = line.split(':', 1)[1].strip()
            elif line.startswith('afm:'):
                config['defaults']['afm'] = line.split(':', 1)[1].strip()
            continue

        # [all] section: every line is a prompt
        if current_section == 'all':
            config['all'].append(line)
            continue

        # Model/variant section: parse params or treat as prompt
        sec = model_sections[current_section]
        if line == 'skip':
            sec['skip'] = True
        elif line.startswith('max_tokens:'):
            sec['params']['max_tokens'] = int(line.split(':', 1)[1].strip())
        elif line.startswith('temperature:'):
            sec['params']['temperature'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('top_p:'):
            sec['params']['top_p'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('top_k:'):
            sec['params']['top_k'] = int(line.split(':', 1)[1].strip())
        elif line.startswith('min_p:'):
            sec['params']['min_p'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('seed:'):
            sec['params']['seed'] = int(line.split(':', 1)[1].strip())
        elif line.startswith('logprobs:'):
            sec['params']['logprobs'] = line.split(':', 1)[1].strip().lower() == 'true'
        elif line.startswith('top_logprobs:'):
            sec['params']['top_logprobs'] = int(line.split(':', 1)[1].strip())
        elif line.startswith('presence_penalty:'):
            sec['params']['presence_penalty'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('repetition_penalty:'):
            sec['params']['repetition_penalty'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('frequency_penalty:'):
            sec['params']['frequency_penalty'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('stop:'):
            sec['params']['stop'] = [s.strip() for s in line.split(':', 1)[1].split(',')]
        elif line.startswith('response_format:'):
            sec['params']['response_format'] = line.split(':', 1)[1].strip()
        elif line.startswith('system:'):
            sec['params']['system'] = line.split(':', 1)[1].strip()
        elif line.startswith('afm:'):
            sec['afm'] = line.split(':', 1)[1].strip()
        else:
            sec['prompts'].append(line)

# Build ordered runs list (preserves file order)
for key in model_sections:
    config['runs'].append(model_sections[key])

print(json.dumps(config))
PYEOF
  )

  if [ $? -ne 0 ]; then
    echo "Error: Failed to parse prompts file: $PROMPTS_FILE" >&2
    exit 1
  fi

  echo "=== Loaded prompts file: $PROMPTS_FILE ==="
  all_count=$(echo "$PARSED_CONFIG" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['all']))")
  run_count=$(echo "$PARSED_CONFIG" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['runs']))")
  echo "  Global prompts: $all_count | Test runs: $run_count"
  echo ""
fi

# ── Build test run list ───────────────────────────────────────────────────────
# Each run is: MODEL_ID|LABEL|MAX_TOKENS|TEMPERATURE|SYSTEM|AFM_ARGS
# The pipe-delimited format is consumed by the main loop.

RUNS_JSON=""

if [ -n "$PARSED_CONFIG" ]; then
  # Prompts file mode: runs come from the file (explicit sections)
  # For --models or --model without explicit sections, we also generate
  # runs for models not mentioned in the file (they get [all] prompts + defaults)
  RUNS_JSON="$PARSED_CONFIG"

elif [ -n "$SINGLE_MODEL" ]; then
  MODELS=("$SINGLE_MODEL")
  echo "=== Testing single model: $SINGLE_MODEL ==="
  echo ""

elif $USE_AUTO_MODELS; then
  echo "=== Auto-discovering models from $MACAFM_MLX_MODEL_CACHE ==="
  MODELS=()
  if command -v models >/dev/null 2>&1; then
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      [[ "$line" == MACAFM_MLX_MODEL_CACHE=* ]] && continue
      [[ "$line" == *"models found"* ]] && continue
      MODELS+=("$line")
    done < <(models)
  elif [ -d "$MACAFM_MLX_MODEL_CACHE" ]; then
    SKIP="hub|gguf|xet"
    for org_dir in "$MACAFM_MLX_MODEL_CACHE"/*/; do
      org=$(basename "$org_dir")
      echo "$org" | grep -qE "^($SKIP)$" && continue
      for model_dir in "$org_dir"*/; do
        [ -d "$model_dir" ] || continue
        model=$(basename "$model_dir")
        MODELS+=("$org/$model")
      done
    done
  else
    echo "Error: MACAFM_MLX_MODEL_CACHE not set and 'models' command not found" >&2
    exit 1
  fi

  if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models found" >&2
    exit 1
  fi

  # Sort models by size on disk: run the 5 biggest last (biggest very last)
  SORTED_MODELS=()
  while IFS= read -r line; do
    model="${line#* }"
    SORTED_MODELS+=("$model")
  done < <(
    for m in "${MODELS[@]}"; do
      dir="$MACAFM_MLX_MODEL_CACHE/$m"
      if [ -d "$dir" ]; then
        size=$(du -sm "$dir" 2>/dev/null | cut -f1)
      else
        size=0
      fi
      echo "$size $m"
    done | sort -n
  )

  total_sorted=${#SORTED_MODELS[@]}
  if [ "$total_sorted" -gt 5 ]; then
    MODELS=("${SORTED_MODELS[@]:0:$((total_sorted - 5))}" "${SORTED_MODELS[@]:$((total_sorted - 5))}")
  else
    MODELS=("${SORTED_MODELS[@]}")
  fi

  echo "Found ${#MODELS[@]} models (5 largest will run last)"
  echo ""

else
  MODELS=(
    "hub/models--mlx-community--Qwen3-VL-4B-Instruct-4bit"
    "mlx-community/Apertus-8B-Instruct-2509-4bit"
    "mlx-community/exaone-4.0-1.2b-4bit"
    "mlx-community/gemma-3-4b-it-8bit"
    "mlx-community/gemma-3n-E2B-it-lm-4bit"
    "mlx-community/GLM-4.7-Flash-4bit"
    "mlx-community/GLM-5-4bit"
    "mlx-community/gpt-oss-20b-MXFP4-Q4"
    "mlx-community/gpt-oss-20b-MXFP4-Q8"
    "mlx-community/granite-4.0-h-tiny-4bit"
    "mlx-community/JoyAI-LLM-Flash-4bit-DWQ"
    "mlx-community/LFM2-2.6B-4bit"
    "mlx-community/LFM2-VL-3B-4bit"
    "mlx-community/lille-130m-instruct-8bit"
    "mlx-community/Ling-mini-2.0-4bit"
    "mlx-community/Llama-3.2-1B-Instruct-4bit"
    "mlx-community/MiniMax-M2.5-5bit"
    "mlx-community/MiniMax-M2.5-6bit"
    "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    "mlx-community/Qwen3-0.6B-4bit"
    "mlx-community/Qwen3-30B-A3B-4bit"
    "mlx-community/Qwen3-Coder-Next-4bit"
    "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    "mlx-community/Qwen3-VL-4B-Instruct-8bit"
    "mlx-community/Qwen3.5-397B-A17B-4bit"
    "mlx-community/SmolLM3-3B-4bit"
    "mlx-community/Kimi-K2.5-3bit"
  )
fi

# ── Helper: get run config for a model/variant ────────────────────────────────

# For prompts-file mode: returns JSON with merged config for a specific run
get_run_config() {
  local run_index="$1"
  python3 -c "
import json, sys

idx = int(sys.argv[1])
cfg = json.loads(sys.argv[2])

defaults = cfg.get('defaults', {})
all_prompts = cfg.get('all', [])
run = cfg['runs'][idx]

if run.get('skip', False):
    print(json.dumps({'skip': True}))
    sys.exit(0)

# Merge prompts: all + run-specific
prompts = list(all_prompts) + run.get('prompts', [])

# Merge params: defaults < run-specific
params = run.get('params', {})
max_tokens = params.get('max_tokens', defaults.get('max_tokens', $DEFAULT_MAX_TOKENS))
temperature = params.get('temperature', defaults.get('temperature', $DEFAULT_TEMPERATURE))
system = params.get('system', defaults.get('system', ''))

# Merge afm args: defaults + run-specific (concatenate)
default_afm = defaults.get('afm', '')
run_afm = run.get('afm', '')
afm_args = (default_afm + ' ' + run_afm).strip()

# Merge optional sampling params (None = not set, omitted from SDK call)
def merge_param(key):
    return params.get(key, defaults.get(key))

result = {
    'skip': False,
    'model': run['model'],
    'label': run.get('label', ''),
    'prompts': prompts,
    'max_tokens': max_tokens,
    'temperature': temperature,
    'system': system,
    'afm_args': afm_args,
    'top_p': merge_param('top_p'),
    'top_k': merge_param('top_k'),
    'min_p': merge_param('min_p'),
    'seed': merge_param('seed'),
    'logprobs': merge_param('logprobs'),
    'top_logprobs': merge_param('top_logprobs'),
    'presence_penalty': merge_param('presence_penalty'),
    'repetition_penalty': merge_param('repetition_penalty'),
    'frequency_penalty': merge_param('frequency_penalty'),
    'stop': merge_param('stop'),
    'response_format': merge_param('response_format'),
}
print(json.dumps(result))
" "$run_index" "$PARSED_CONFIG"
}

# For non-prompts-file mode: returns config for a plain model
get_legacy_config() {
  local model="$1"
  local max_tokens=$DEFAULT_MAX_TOKENS
  local temperature=$DEFAULT_TEMPERATURE
  local sys_prompt=""
  local prompt="$DEFAULT_PROMPT"

  if echo "$model" | grep -qi "gpt-oss"; then
    sys_prompt="Reasoning:low"
  fi
  if echo "$model" | grep -qi "kimi"; then
    prompt="Why is the sky blue? Be brief"
  fi

  python3 -c "
import json
print(json.dumps({
    'skip': False,
    'model': '$model',
    'label': '',
    'prompts': ['''$prompt'''],
    'max_tokens': $max_tokens,
    'temperature': $temperature,
    'system': '$sys_prompt',
    'afm_args': ''
}))
"
}

# ── Server management ─────────────────────────────────────────────────────────

SERVER_PID=0

kill_server() {
  local pid=$1
  if [ "$pid" != "0" ] && kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 0.5
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  echo ""
  echo "  Interrupted — cleaning up..."
  kill_server $SERVER_PID
  echo ""
  echo "=== Test interrupted. Partial results in $RESULTS_FILE ==="
  exit 130
}
trap cleanup INT TERM

wait_for_server() {
  local deadline=$((SECONDS + TIMEOUT_LOAD))
  while [ $SECONDS -lt $deadline ]; do
    if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      return 1
    fi
    sleep 1
  done
  return 1
}

escape_json() {
  python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$1"
}

# ── Send a single prompt and record result ────────────────────────────────────

send_prompt() {
  local run_config="$1"
  local display_name="$2"
  local prompt_text="$3"
  local load_time="$4"
  local prompt_label="$5"  # e.g. "1/3" for display

  if [ -n "$prompt_label" ]; then
    echo "  Prompt $prompt_label: ${prompt_text:0:80}..."
  fi

  # Build config JSON with display_name and prompt_text injected
  local send_config
  send_config=$(echo "$run_config" | python3 -c "
import json, sys
c = json.load(sys.stdin)
c['display_name'] = sys.argv[1]
c['prompt_text'] = sys.argv[2]
print(json.dumps(c))
" "$display_name" "$prompt_text")

  # Single python3 block: builds OpenAI SDK call, sends request, outputs JSONL result
  METRICS=$(_SEND_CONFIG="$send_config" python3 - "$PORT" "$load_time" "$TIMEOUT_GENERATE" <<'SEND_PYEOF'
import json, sys, time, os

# Read config from env var (stdin used by heredoc)
config = json.loads(os.environ['_SEND_CONFIG'])
port = int(sys.argv[1])
load_time = float(sys.argv[2])
timeout = int(sys.argv[3])

from openai import OpenAI

client = OpenAI(
    base_url=f"http://127.0.0.1:{port}/v1",
    api_key="not-needed",
    timeout=timeout,
)

model = config['model']
display_name = config['display_name']
label = config.get('label', '')
prompt_text = config['prompt_text']
system_prompt = config.get('system', '')
max_tokens = config['max_tokens']
temperature = config['temperature']
afm_args = config.get('afm_args', '')

# Build messages
messages = []
if system_prompt:
    messages.append({'role': 'system', 'content': system_prompt})
messages.append({'role': 'user', 'content': prompt_text})

# Build SDK kwargs
kwargs = {
    'model': model,
    'messages': messages,
    'max_tokens': max_tokens,
    'temperature': temperature,
    'stream': False,
}

# Standard OpenAI params (only set if non-None)
if config.get('top_p') is not None:
    kwargs['top_p'] = config['top_p']
if config.get('seed') is not None:
    kwargs['seed'] = config['seed']
if config.get('logprobs') is not None:
    kwargs['logprobs'] = config['logprobs']
if config.get('top_logprobs') is not None:
    kwargs['top_logprobs'] = config['top_logprobs']
if config.get('presence_penalty') is not None:
    kwargs['presence_penalty'] = config['presence_penalty']
if config.get('frequency_penalty') is not None:
    kwargs['frequency_penalty'] = config['frequency_penalty']
if config.get('stop') is not None:
    kwargs['stop'] = config['stop']
if config.get('response_format') is not None:
    rf = config['response_format']
    if rf == 'json_object':
        kwargs['response_format'] = {'type': 'json_object'}
    elif rf != 'text':
        kwargs['response_format'] = {'type': rf}

# Non-standard params via extra_body
extra_body = {}
if config.get('top_k') is not None:
    extra_body['top_k'] = config['top_k']
if config.get('min_p') is not None:
    extra_body['min_p'] = config['min_p']
if config.get('repetition_penalty') is not None:
    extra_body['repetition_penalty'] = config['repetition_penalty']
if extra_body:
    kwargs['extra_body'] = extra_body

try:
    gen_start = time.time()
    response = client.chat.completions.create(**kwargs)
    gen_end = time.time()
    gen_time = gen_end - gen_start

    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    msg = (message.content or '') if message else ''

    # reasoning_content is a non-standard field — access via model_extra or attribute
    reasoning = ''
    if message:
        reasoning = getattr(message, 'reasoning_content', '') or ''
        if not reasoning and hasattr(message, 'model_extra') and message.model_extra:
            reasoning = message.model_extra.get('reasoning_content', '') or ''

    full_content = msg if msg else reasoning
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0
    tps = completion_tokens / gen_time if gen_time > 0 else 0
    content_preview = full_content[:300].replace('\n', ' ')

    # Feature verification fields
    finish_reason = choice.finish_reason if choice else None
    system_fingerprint = getattr(response, 'system_fingerprint', None)

    # Logprobs: count tokens with logprobs returned
    logprobs_data = getattr(choice, 'logprobs', None) if choice else None
    logprobs_count = 0
    if logprobs_data and hasattr(logprobs_data, 'content') and logprobs_data.content:
        logprobs_count = len(logprobs_data.content)

    # JSON validation: check if content is valid JSON
    is_valid_json = None  # None = not tested
    if config.get('response_format') in ('json_object', 'json_schema') or \
       any(kw in prompt_text.lower() for kw in ('json', 'valid json')):
        try:
            json.loads(msg)
            is_valid_json = True
        except (json.JSONDecodeError, ValueError):
            is_valid_json = False

    result = {
        'model': display_name,
        'label': label,
        'prompt': prompt_text,
        'status': 'OK',
        'load_time_s': load_time,
        'gen_time_s': round(gen_time, 2),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'tokens_per_sec': round(tps, 2),
        'temperature': temperature,
        'max_tokens': max_tokens,
        'system_prompt': system_prompt,
        'afm_args': afm_args,
        'content_preview': content_preview,
        'content': msg,
        'reasoning_content': reasoning,
        'finish_reason': finish_reason,
        'logprobs_count': logprobs_count,
    }
    if system_fingerprint:
        result['system_fingerprint'] = system_fingerprint
    if is_valid_json is not None:
        result['is_valid_json'] = is_valid_json

    # Record all optional params that were set
    for key in ('top_p', 'top_k', 'min_p', 'seed', 'logprobs', 'top_logprobs',
                'presence_penalty', 'repetition_penalty', 'frequency_penalty',
                'stop', 'response_format'):
        if config.get(key) is not None:
            result[key] = config[key]

    print(json.dumps(result))

except Exception as e:
    gen_end = time.time()
    error_msg = str(e)[:500]
    result = {
        'model': display_name,
        'label': label,
        'prompt': prompt_text,
        'status': 'FAIL',
        'error': error_msg,
        'load_time_s': load_time,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'system_prompt': system_prompt,
        'afm_args': afm_args,
    }
    for key in ('top_p', 'top_k', 'min_p', 'seed', 'logprobs', 'top_logprobs',
                'presence_penalty', 'repetition_penalty', 'frequency_penalty',
                'stop', 'response_format'):
        if config.get(key) is not None:
            result[key] = config[key]
    print(json.dumps(result))
SEND_PYEOF
  )

  echo "$METRICS" >> "$RESULTS_FILE"

  # Extract status for display
  local status=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','?'))")
  if [ "$status" = "OK" ]; then
    local tps=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tokens_per_sec',0))")
    local ctok=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('completion_tokens',0))")
    local gtime=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('gen_time_s',0))")
    echo "    OK: ${ctok} tokens in ${gtime}s (${tps} tok/s)"
  else
    local error_msg=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('error','unknown')[:200])")
    echo "    FAIL: $error_msg"
  fi
}

# ── Run a single test (start server, send prompts, kill server) ───────────────

run_test() {
  local run_config="$1"
  local idx="$2"
  local total="$3"

  # Extract fields from config JSON (single python call)
  eval "$(echo "$run_config" | python3 -c "
import json, sys, shlex
c = json.load(sys.stdin)
print(f'local model={shlex.quote(c[\"model\"])}')
print(f'local label={shlex.quote(c.get(\"label\", \"\"))}')
print(f'local max_tokens={c[\"max_tokens\"]}')
print(f'local temperature={c[\"temperature\"]}')
print(f'local sys_prompt={shlex.quote(c.get(\"system\", \"\"))}')
print(f'local afm_args={shlex.quote(c.get(\"afm_args\", \"\"))}')
")"

  # Display name: "model" or "model @ label"
  local display_name="$model"
  if [ -n "$label" ]; then
    display_name="$model @ $label"
  fi

  echo "=== [$idx/$total] Testing: $display_name ==="
  if [ -n "$afm_args" ]; then
    echo "  afm flags: $afm_args"
  fi

  # Build server args
  SERVER_EXTRA_ARGS=()
  if [ -n "$sys_prompt" ]; then
    SERVER_EXTRA_ARGS+=(-i "$sys_prompt")
  fi
  # Append custom afm CLI args (word-split intentionally)
  if [ -n "$afm_args" ]; then
    read -ra AFM_EXTRA <<< "$afm_args"
    SERVER_EXTRA_ARGS+=("${AFM_EXTRA[@]}")
  fi

  # Kill anything already listening on our port (stale server from previous run)
  local stale_pid
  stale_pid=$(lsof -ti :"$PORT" 2>/dev/null || true)
  if [ -n "$stale_pid" ]; then
    echo "  Warning: killing stale process on port $PORT (PID $stale_pid)"
    kill -KILL $stale_pid 2>/dev/null || true
    sleep 1
  fi

  # Start server
  SERVER_LOG="/tmp/mlx-server-${idx}.log"
  load_start=$SECONDS
  "$AFM" mlx -m "$model" -p "$PORT" ${SERVER_EXTRA_ARGS[@]+"${SERVER_EXTRA_ARGS[@]}"} > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  # Wait for server
  if ! wait_for_server; then
    load_time=$((SECONDS - load_start))
    error_msg=$(grep -i "error\|fatal\|unsupported\|not supported\|failed" "$SERVER_LOG" | head -3 | tr '\n' ' ' | sed 's/"/\\"/g' | head -c 500)
    if [ -z "$error_msg" ]; then
      if ! kill -0 $SERVER_PID 2>/dev/null; then
        error_msg="Server process died (check $SERVER_LOG)"
      else
        error_msg="Server failed to start within ${TIMEOUT_LOAD}s"
      fi
    fi
    echo "  FAIL: $error_msg"
    echo "{\"model\":$(escape_json "$display_name"),\"label\":$(escape_json "$label"),\"status\":\"FAIL\",\"error\":$(escape_json "$error_msg"),\"load_time_s\":$load_time,\"temperature\":$temperature,\"max_tokens\":$max_tokens,\"system_prompt\":$(escape_json "$sys_prompt"),\"afm_args\":$(escape_json "$afm_args")}" >> "$RESULTS_FILE"
    kill_server $SERVER_PID
    SERVER_PID=0
    sleep 2
    echo ""
    return
  fi

  load_time=$((SECONDS - load_start))
  echo "  Server ready in ${load_time}s"

  # Get prompts list
  local prompt_list
  prompt_list=$(echo "$run_config" | python3 -c "import json,sys; [print(p) for p in json.load(sys.stdin).get('prompts',[])]")
  if [ -z "$prompt_list" ]; then
    prompt_list="$DEFAULT_PROMPT"
  fi

  readarray -t PROMPTS_ARRAY <<< "$prompt_list"
  num_prompts=${#PROMPTS_ARRAY[@]}

  for pidx in "${!PROMPTS_ARRAY[@]}"; do
    prompt_text="${PROMPTS_ARRAY[$pidx]}"
    [ -z "$prompt_text" ] && continue
    if [ "$num_prompts" -gt 1 ]; then
      prompt_label="$((pidx + 1))/$num_prompts"
    else
      prompt_label=""
    fi
    send_prompt "$run_config" "$display_name" "$prompt_text" "$load_time" "$prompt_label"
  done

  # Kill server
  kill_server $SERVER_PID
  SERVER_PID=0
  sleep 2
  echo ""
}

# ── Main test loop ────────────────────────────────────────────────────────────

if [ -n "$PARSED_CONFIG" ]; then
  # Prompts file mode: iterate runs from the file
  num_runs=$(echo "$PARSED_CONFIG" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['runs']))")

  if [ "$num_runs" -gt 0 ]; then
    # We have explicit model sections — run those
    # But also handle --models: models without a section get [all] prompts + defaults
    if $USE_AUTO_MODELS || [ -n "$SINGLE_MODEL" ]; then
      # Get models that already have explicit sections
      EXPLICIT_MODELS=$(echo "$PARSED_CONFIG" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
seen = set()
for run in cfg['runs']:
    seen.add(run['model'])
for m in sorted(seen):
    print(m)
")

      # Build list of models to test
      if [ -n "$SINGLE_MODEL" ]; then
        ALL_MODELS=("$SINGLE_MODEL")
      else
        # MODELS array was already built above
        ALL_MODELS=("${MODELS[@]}")
      fi

      # Add implicit runs for models not in the file
      EXTRA_RUNS=$(python3 -c "
import json, sys

cfg = json.loads(sys.argv[1])
explicit = set()
for run in cfg['runs']:
    explicit.add(run['model'])

for model in sys.argv[2:]:
    if model not in explicit:
        cfg['runs'].append({
            'model': model,
            'label': '',
            'prompts': [],
            'params': {},
            'afm': '',
            'skip': False
        })

print(json.dumps(cfg))
" "$PARSED_CONFIG" "${ALL_MODELS[@]}")
      PARSED_CONFIG="$EXTRA_RUNS"
      num_runs=$(echo "$PARSED_CONFIG" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['runs']))")
    fi

    echo "=== $num_runs test run(s) ==="
    echo ""

    for ((ri=0; ri<num_runs; ri++)); do
      run_config=$(get_run_config "$ri")
      skip=$(echo "$run_config" | python3 -c "import json,sys; print(json.load(sys.stdin).get('skip', False))")
      if [ "$skip" = "True" ]; then
        skip_name=$(echo "$PARSED_CONFIG" | python3 -c "
import json,sys
r = json.load(sys.stdin)['runs'][$ri]
name = r['model']
if r.get('label'): name += ' @ ' + r['label']
print(name)
")
        echo "=== [$((ri+1))/$num_runs] SKIPPED: $skip_name ==="
        echo ""
        continue
      fi
      run_test "$run_config" "$((ri+1))" "$num_runs"
    done

  else
    # No model sections in file, just [all] prompts — apply to all models
    if [ -z "$SINGLE_MODEL" ] && ! $USE_AUTO_MODELS && [ ${#MODELS[@]-0} -eq 0 ]; then
      echo "Error: Prompts file has no [model] sections. Use --models or --model to specify which models to test." >&2
      exit 1
    fi

    total=${#MODELS[@]}
    idx=0
    for model in "${MODELS[@]}"; do
      idx=$((idx + 1))
      run_config=$(python3 -c "
import json, sys
cfg = json.loads(sys.argv[1])
defaults = cfg.get('defaults', {})
all_prompts = cfg.get('all', [])
result = {
    'skip': False,
    'model': sys.argv[2],
    'label': '',
    'prompts': all_prompts,
    'max_tokens': defaults.get('max_tokens', $DEFAULT_MAX_TOKENS),
    'temperature': defaults.get('temperature', $DEFAULT_TEMPERATURE),
    'system': defaults.get('system', ''),
    'afm_args': defaults.get('afm', ''),
}
for key in ('top_p', 'top_k', 'min_p', 'seed', 'logprobs', 'top_logprobs',
            'presence_penalty', 'repetition_penalty', 'frequency_penalty',
            'stop', 'response_format'):
    if key in defaults:
        result[key] = defaults[key]
print(json.dumps(result))
" "$PARSED_CONFIG" "$model")
      run_test "$run_config" "$idx" "$total"
    done
  fi

else
  # No prompts file — legacy mode
  total=${#MODELS[@]}
  idx=0
  for model in "${MODELS[@]}"; do
    idx=$((idx + 1))
    run_config=$(get_legacy_config "$model")
    run_test "$run_config" "$idx" "$total"
  done
fi

echo "=== All tests complete. Results in $RESULTS_FILE ==="
echo ""

# ── Smart analysis (runs BEFORE report so HTML can embed it) ─────────────────

if $SMART_ANALYSIS; then
  SMART_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
  ANALYSIS_PROMPT="$(cat <<'ANALYSIS_PROMPT_END'
You are a QA engineer for AFM (Apple Foundation Models), an OpenAI-compatible local
inference server for Apple Silicon. Your job is to review automated test results from
a model compatibility test suite.

CONTEXT: AFM loads MLX-format models from Hugging Face and serves them via
/v1/chat/completions. This test suite starts the server for each model, sends one or
more prompts, and records the response, timing, and token counts. The results you are
reading are the raw JSONL output from that test run.

Models may appear multiple times with different labels (e.g. "model @ default",
"model @ conservative") — these are A/B test variants with different sampling
parameters or CLI flags. Compare their outputs when analyzing.

YOUR TASK: Analyze these results to help the developer quickly identify which models
are broken, degraded, or behaving unexpectedly so they can prioritize debugging.
Focus on actionable findings — not general commentary.

Produce a concise markdown report covering:

1. **Broken Models**: Models that failed to load or crashed. Group by error type
   (e.g. unsupported architecture, missing files, timeout). Note which errors are
   AFM bugs vs. model incompatibilities.

2. **Anomalies & Red Flags**: Models that loaded and generated but produced suspicious output:
   - Excessive repetition (repeated phrases, sentences, or paragraphs looping)
   - Degenerate output (gibberish, token soup, endless punctuation, or HTML/XML noise)
   - Truncated or incomplete responses that suggest generation stalled
   - Responses that completely ignore the prompt (wrong language, off-topic)
   - Unusual token/time ratios (e.g. very high tok/s with nonsense output)

3. **Variant Comparison**: For models tested with multiple variants (@ labels),
   compare the outputs. Note which sampling settings produced better/worse results.
   Flag cases where one variant works but another degrades.

4. **Quality Assessment**: For each model that produced output, rate:
   - Coherence (1-5): Is the text well-formed and readable?
   - Relevance (1-5): Does it actually address the prompt?
   Flag any model scoring below 3 on either metric.

5. **Performance Summary**: Table of all models sorted by tokens/sec.
   Flag outliers — unusually slow for their size class, or suspiciously fast
   (which may indicate degenerate short-circuit generation).

6. **Recommendations**: Prioritized list of models that need investigation,
   with specific reasons. Separate into:
   - "Likely AFM bug" (model works elsewhere, fails here)
   - "Model quality issue" (model itself is poor)
   - "Working well" (no action needed)

Format as clean markdown with tables where appropriate. Be specific — quote
problematic output snippets (first 100 chars) when flagging issues. Keep the
report concise and scannable.

CRITICAL — MACHINE-READABLE SCORES:
At the very end of your output, you MUST include a scores block for the HTML report.
Score EVERY JSONL line (by 0-based line index) on a 1-5 scale:
  5 = excellent (correct, coherent, addresses prompt well)
  4 = good (minor issues but solid response)
  3 = acceptable (noticeable issues but usable)
  2 = poor (significant problems — repetition, off-topic, garbled)
  1 = broken (failed to load, server error, or status=FAIL)

IMPORTANT scoring notes:
- Base your score on "content", "content_preview", AND "reasoning_content" fields.
- THINKING-BUDGET EXHAUSTION: If "content" is empty but "reasoning_content" is non-empty
  and completion_tokens is close to max_tokens, the model spent its entire token budget
  reasoning and never emitted a response. This is NOT a harness failure or empty response —
  it means the max_tokens was too low for a thinking model. Score based on the quality of
  the reasoning content (typically 2-3, since no actual response was produced, but the model
  was functioning correctly). Flag this pattern explicitly in your anomalies section.
- If BOTH "content" AND "reasoning_content" are empty but status=OK and completion_tokens > 0,
  the test harness failed to capture output — score 3 (unknown quality) NOT 1.
- Only score 1 for actual failures (status=FAIL).
- A response that works but has minor formatting issues is still a 4 or 5.
- Repetitive/looping text is a 2. Completely off-topic or garbled is a 2.
- Reserve score 1 strictly for: status=FAIL, server crashes, load failures.

Output the block EXACTLY like this (one line, valid JSON array):
<!-- AI_SCORES [{"i":0,"s":5},{"i":1,"s":3},{"i":2,"s":1}] -->

Rules:
- Include an entry for EVERY line in the JSONL input, including failed ones (score 1)
- The "i" field is the 0-based line index in the JSONL
- The "s" field is the integer score 1-5
- This line must be the LAST line of your output
- Do NOT wrap it in a code block
ANALYSIS_PROMPT_END
)"

  # Build combined prompt+data for tools that need it in one stream
  SMART_INPUT="$(mktemp /tmp/smart-input-XXXXXX.txt)"
  { echo "$ANALYSIS_PROMPT"; echo ""; echo "--- JSONL DATA ---"; cat "$RESULTS_FILE"; } > "$SMART_INPUT"

  for tool in "${SMART_TOOL_LIST[@]}"; do
    echo "=== Running AI analysis with: $tool ==="
    SMART_REPORT="$(pwd)/test-reports/smart-analysis-${tool}-${SMART_TIMESTAMP}.md"

    case "$tool" in
      claude)
        # claude -p "prompt" < data (stderr separate to avoid polluting report)
        "$tool" -p "$ANALYSIS_PROMPT" < "$RESULTS_FILE" > "$SMART_REPORT" 2>/tmp/smart-${tool}-stderr.log
        ;;
      codex)
        # codex exec: prompt as argument, data file path embedded in prompt
        "$tool" exec --skip-git-repo-check "$ANALYSIS_PROMPT

--- JSONL DATA (read from file) ---
File: $RESULTS_FILE
$(cat "$RESULTS_FILE")" > "$SMART_REPORT" 2>/tmp/smart-${tool}-stderr.log
        ;;
      afm)
        # afm uses Apple Foundation Models — context window is limited (~4K tokens).
        # Two-pass approach: score each result individually, then summarize.
        AFM_SCORES_DIR="$(mktemp -d /tmp/smart-afm-XXXXXX)"
        total_lines=$(wc -l < "$RESULTS_FILE" | tr -d ' ')
        echo "  Pass 1: Scoring $total_lines results individually..."

        line_idx=0
        while IFS= read -r jsonl_line; do
          [ -z "$jsonl_line" ] && continue
          echo -n "    [$((line_idx + 1))/$total_lines] "
          model_name=$(echo "$jsonl_line" | python3 -c "import json,sys; print(json.load(sys.stdin).get('model','?')[:40])")
          echo -n "$model_name... "

          AFM_SCORE=$("$AFM" -s "$(cat <<SCORE_PROMPT_END
You are a QA engineer scoring a single test result from an LLM inference server.

Score this result on a 1-5 scale:
  5 = excellent (correct, coherent, addresses prompt well)
  4 = good (minor issues but solid response)
  3 = acceptable (noticeable issues but usable)
  2 = poor (significant problems — repetition, off-topic, garbled)
  1 = broken (failed to load, server error, or status=FAIL)

Scoring rules:
- If status=FAIL, score 1.
- If "content" is empty but "reasoning_content" is non-empty and completion_tokens
  is close to max_tokens, the model spent its token budget reasoning. Score 2-3.
- If BOTH content and reasoning_content are empty but status=OK, score 3.
- Repetitive/looping text is a 2. Off-topic or garbled is a 2.

Respond with EXACTLY one line of JSON, nothing else:
{"score": N, "reason": "brief explanation"}

TEST RESULT:
$jsonl_line
SCORE_PROMPT_END
)" 2>/tmp/smart-afm-stderr.log)

          echo "$AFM_SCORE" > "$AFM_SCORES_DIR/score_${line_idx}.txt"
          # Extract score for display
          score_val=$(echo "$AFM_SCORE" | python3 -c "
import json, sys, re
text = sys.stdin.read().strip()
# Try to extract JSON from response
m = re.search(r'\{[^}]*\"score\"\s*:\s*(\d+)[^}]*\}', text)
if m:
    print(m.group(1))
else:
    print('?')
" 2>/dev/null)
          echo "score=$score_val"
          line_idx=$((line_idx + 1))
        done < "$RESULTS_FILE"

        echo "  Pass 2: Generating summary report..."

        # Build compact scores + one-liner metadata for summary pass (must fit in ~4K tokens)
        AFM_PASS2=$(python3 -c "
import json, sys, os, re

scores_dir = sys.argv[1]
results_file = sys.argv[2]

scores = []
compact_lines = []
with open(results_file) as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line: continue
        r = json.loads(line)

        # Read score file
        score_file = os.path.join(scores_dir, f'score_{idx}.txt')
        score_val = 3
        if os.path.exists(score_file):
            with open(score_file) as sf:
                text = sf.read().strip()
            m = re.search(r'\"score\"\s*:\s*(\d+)', text)
            if m:
                score_val = int(m.group(1))

        scores.append({'i': idx, 's': score_val})

        # Compact one-liner: idx, model@label, status, score, tps
        model = r.get('model', '')
        label = r.get('label', '')
        name = f'{model} @ {label}' if label else model
        tps = r.get('tokens_per_sec', 0)
        status = r.get('status', '')
        compact_lines.append(f'{idx}: {name} | {status} score={score_val} tps={tps}')

print('SCORES_JSON=' + json.dumps(scores))
print('---')
for cl in compact_lines:
    print(cl)
" "$AFM_SCORES_DIR" "$RESULTS_FILE")

        # Extract scores line for later
        AFM_SCORES_JSON=$(echo "$AFM_PASS2" | head -1 | sed 's/^SCORES_JSON=//')
        AFM_META_DATA=$(echo "$AFM_PASS2" | tail -n +3)

        # Summary pass — compact one-liners, fits in ~4K context
        AFM_SUMMARY=$("$AFM" -s "$(cat <<SUMMARY_PROMPT_END
You are a QA engineer for AFM, an OpenAI-compatible local LLM server.
Below are scored test results. Each line: index, model/variant, status, AI score (1-5), tokens/sec.

Produce a brief markdown report:
1. **Issues**: List score 1-2 results with likely cause
2. **Working Well**: Score 4-5 highlights
3. **Summary**: Overall pass rate and recommendations

$AFM_META_DATA
SUMMARY_PROMPT_END
)" 2>/tmp/smart-afm-stderr.log)

        # Assemble final report
        {
          echo "$AFM_SUMMARY"
          echo ""
          echo "<!-- AI_SCORES $AFM_SCORES_JSON -->"
        } > "$SMART_REPORT"

        rm -rf "$AFM_SCORES_DIR"
        ;;
      *)
        # Generic: try -p first (claude-like), fall back to piping everything
        "$tool" -p "$ANALYSIS_PROMPT" < "$RESULTS_FILE" > "$SMART_REPORT" 2>/tmp/smart-${tool}-stderr.log
        ;;
    esac

    if [ $? -eq 0 ] && [ -s "$SMART_REPORT" ]; then
      echo "  Saved: $SMART_REPORT"
    else
      echo "  Warning: $tool analysis failed or produced empty output"
      echo "  Check: $SMART_REPORT"
    fi
  done
  rm -f "$SMART_INPUT"
  echo ""
fi

# ── HTML report ──────────────────────────────────────────────────────────────

if ! $NO_REPORT; then
  echo "=== Generating HTML report ==="
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPORT_OUTPUT_DIR="$(pwd)" SMART_TIMESTAMP="${SMART_TIMESTAMP:-}" python3 "$SCRIPT_DIR/generate-report.py"
else
  echo "=== Skipping HTML report (--no-report) ==="
  echo "  JSONL: $RESULTS_FILE"
fi
