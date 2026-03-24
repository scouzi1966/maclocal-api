#!/bin/zsh

set -uo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"
cd "$repo_root"

model="${AFM_MODEL:-mlx-community/Qwen3.5-35B-A3B-4bit}"
afm_binary="${AFM_BINARY:-.build/arm64-apple-macosx/release/afm}"
out_dir="${AFM_PROMPTFOO_OUT_DIR:-/Volumes/edata/promptfoo/data/maclocal-api/current}"
mode="${1:-all}"
port="${AFM_PROMPTFOO_PORT:-9999}"
server_pid=""
overall_status=0

mkdir -p "$out_dir"

if ! command -v promptfoo >/dev/null 2>&1; then
  echo "promptfoo is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -x "$afm_binary" ]]; then
  echo "AFM binary not found or not executable: $afm_binary" >&2
  exit 1
fi

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}

trap cleanup EXIT INT TERM

wait_for_health() {
  local attempts=60
  local health_url="http://127.0.0.1:${port}/health"
  local i

  for (( i = 1; i <= attempts; i++ )); do
    if curl -sf "$health_url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "AFM server did not become healthy on :${port}" >&2
  return 1
}

start_server() {
  local profile="$1"
  local -a extra_args=()
  local log_file="${out_dir}/server-${profile}.log"

  case "$profile" in
    default)
      ;;
    adaptive-xml)
      extra_args+=(--tool-call-parser afm_adaptive_xml)
      ;;
    adaptive-xml-grammar)
      extra_args+=(--tool-call-parser afm_adaptive_xml --enable-grammar-constraints)
      ;;
    grammar-enabled)
      extra_args+=(--enable-grammar-constraints)
      ;;
    grammar-enabled-adaptive-xml)
      extra_args+=(--enable-grammar-constraints --tool-call-parser afm_adaptive_xml)
      ;;
    grammar-enabled-concurrent)
      extra_args+=(--enable-grammar-constraints --concurrent 2)
      ;;
    grammar-enabled-prefix-cache)
      extra_args+=(--enable-grammar-constraints --enable-prefix-caching)
      ;;
    grammar-enabled-concurrent-cache)
      extra_args+=(--enable-grammar-constraints --concurrent 2 --enable-prefix-caching)
      ;;
    *)
      echo "Unknown AFM profile: $profile" >&2
      exit 1
      ;;
  esac

  cleanup
  : > "$log_file"
  MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-}" \
    "$afm_binary" mlx -m "$model" --port "$port" "${extra_args[@]}" >"$log_file" 2>&1 &
  server_pid="$!"
  wait_for_health
}

run_structured() {
  local output="${out_dir}/structured-$(print -r -- "$model" | tr '/:' '__').json"
  start_server default
  AFM_MODEL="$model" \
  AFM_BASE_URL_DEFAULT="http://127.0.0.1:${port}/v1" \
  AFM_BINARY="$afm_binary" \
  MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-}" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.structured.yaml \
      -j 1 \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_structured_stress() {
  local output="${out_dir}/structured-stress-$(print -r -- "$model" | tr '/:' '__').json"
  start_server default
  AFM_MODEL="$model" \
  AFM_BASE_URL_DEFAULT="http://127.0.0.1:${port}/v1" \
  AFM_BINARY="$afm_binary" \
  MACAFM_MLX_MODEL_CACHE="${MACAFM_MLX_MODEL_CACHE:-}" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.structured-stress.yaml \
      -j 1 \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_toolcall_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/toolcall-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_toolcall_all() {
  run_toolcall_profile default '^afm-default$' AFM_BASE_URL_DEFAULT
  run_toolcall_profile adaptive-xml '^afm-adaptive-xml$' AFM_BASE_URL_ADAPTIVE_XML
  run_toolcall_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_toolcall_quality_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/toolcall-quality-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.toolcall-quality.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_toolcall_quality_all() {
  run_toolcall_quality_profile default '^afm-default-quality$' AFM_BASE_URL_DEFAULT
  run_toolcall_quality_profile adaptive-xml '^afm-adaptive-xml-quality$' AFM_BASE_URL_ADAPTIVE_XML
  run_toolcall_quality_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-quality$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_agentic_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/agentic-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.agentic.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_agentic_all() {
  run_agentic_profile default '^afm-default-agentic$' AFM_BASE_URL_DEFAULT
  run_agentic_profile adaptive-xml '^afm-adaptive-xml-agentic$' AFM_BASE_URL_ADAPTIVE_XML
  run_agentic_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-agentic$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_frameworks_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/frameworks-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.agentic-frameworks.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_frameworks_all() {
  run_frameworks_profile default '^afm-default-frameworks$' AFM_BASE_URL_DEFAULT
  run_frameworks_profile adaptive-xml '^afm-adaptive-xml-frameworks$' AFM_BASE_URL_ADAPTIVE_XML
  run_frameworks_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-frameworks$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_opencode_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/opencode-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.opencode.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_opencode_all() {
  run_opencode_profile default '^afm-default-opencode$' AFM_BASE_URL_DEFAULT
  run_opencode_profile adaptive-xml '^afm-adaptive-xml-opencode$' AFM_BASE_URL_ADAPTIVE_XML
  run_opencode_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-opencode$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_pi_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/pi-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.pi.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_pi_all() {
  run_pi_profile default '^afm-default-pi$' AFM_BASE_URL_DEFAULT
  run_pi_profile adaptive-xml '^afm-adaptive-xml-pi$' AFM_BASE_URL_ADAPTIVE_XML
  run_pi_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-pi$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_openclaw_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/openclaw-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.openclaw.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_openclaw_all() {
  run_openclaw_profile default '^afm-default-openclaw$' AFM_BASE_URL_DEFAULT
  run_openclaw_profile adaptive-xml '^afm-adaptive-xml-openclaw$' AFM_BASE_URL_ADAPTIVE_XML
  run_openclaw_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-openclaw$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_hermes_profile() {
  local profile="$1"
  local filter_regex="$2"
  local env_name="$3"
  local output="${out_dir}/hermes-${profile}-$(print -r -- "$model" | tr '/:' '__').json"
  local base_url="http://127.0.0.1:${port}/v1"

  start_server "$profile"
  env \
    AFM_MODEL="$model" \
    "$env_name=$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.hermes.yaml \
      -j 1 \
      --filter-targets "$filter_regex" \
      -o "$output"
  local exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

run_hermes_all() {
  run_hermes_profile default '^afm-default-hermes$' AFM_BASE_URL_DEFAULT
  run_hermes_profile adaptive-xml '^afm-adaptive-xml-hermes$' AFM_BASE_URL_ADAPTIVE_XML
  run_hermes_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar-hermes$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
}

run_grammar_constraints() {
  local base_url="http://127.0.0.1:${port}/v1"

  # Phase 1: default server (no --enable-grammar-constraints) — tests downgrade path
  start_server default

  local output_schema_no="${out_dir}/grammar-schema-no-grammar-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_DEFAULT="$base_url" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints.yaml \
      -j 1 \
      --filter-targets '^afm-no-grammar$' \
      -o "$output_schema_no"
  local exit_code=$?
  (( overall_status |= exit_code ))

  local output_tools_no="${out_dir}/grammar-tools-no-grammar-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_DEFAULT="$base_url" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints-tools.yaml \
      -j 1 \
      --filter-targets '^afm-no-grammar-tools$' \
      -o "$output_tools_no"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Phase 2: grammar-enabled server — tests enforcement path
  start_server grammar-enabled

  local output_schema_on="${out_dir}/grammar-schema-grammar-enabled-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_DEFAULT="$base_url" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled$' \
      -o "$output_schema_on"
  exit_code=$?
  (( overall_status |= exit_code ))

  local output_tools_on="${out_dir}/grammar-tools-grammar-enabled-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_DEFAULT="$base_url" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints-tools.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled-tools$' \
      -o "$output_tools_on"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Phase 3: grammar-enabled-adaptive-xml — regression guard: grammar works with afm_adaptive_xml parser
  start_server grammar-enabled-adaptive-xml

  local output_tools_adaptive="${out_dir}/grammar-tools-adaptive-xml-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints-tools.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled-tools$' \
      -o "$output_tools_adaptive"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Phase 4: grammar-enabled-concurrent — concurrent path grammar
  start_server grammar-enabled-concurrent

  local output_schema_conc="${out_dir}/grammar-schema-concurrent-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled$' \
      -o "$output_schema_conc"
  exit_code=$?
  (( overall_status |= exit_code ))

  local output_tools_conc="${out_dir}/grammar-tools-concurrent-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints-tools.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled-tools$' \
      -o "$output_tools_conc"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Phase 5: grammar-enabled-prefix-cache — prefix caching + grammar interaction
  start_server grammar-enabled-prefix-cache

  local output_schema_cache="${out_dir}/grammar-schema-prefix-cache-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled$' \
      -o "$output_schema_cache"
  exit_code=$?
  (( overall_status |= exit_code ))

  local output_tools_cache="${out_dir}/grammar-tools-prefix-cache-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints-tools.yaml \
      -j 1 \
      --filter-targets '^afm-grammar-enabled-tools$' \
      -o "$output_tools_cache"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Phase 6: mixed strict — json_schema + tool strict in same request
  start_server grammar-enabled

  local output_mixed="${out_dir}/grammar-mixed-strict-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-constraints-mixed.yaml \
      -j 1 \
      -o "$output_mixed"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Phase 7: header assertions — validate X-Grammar-Constraints response header

  # Downgraded header (default server, no --enable-grammar-constraints)
  start_server default

  local output_header_down="${out_dir}/grammar-header-downgraded-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_DEFAULT="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-header-downgraded.yaml \
      -j 1 \
      -o "$output_header_down"
  exit_code=$?
  (( overall_status |= exit_code ))

  # Enforced header (grammar-enabled server — no downgrade header expected)
  start_server grammar-enabled

  local output_header_enf="${out_dir}/grammar-header-enforced-$(print -r -- "$model" | tr '/:' '__').json"
  env \
    AFM_MODEL="$model" \
    AFM_BASE_URL_GRAMMAR="$base_url" \
    promptfoo eval \
      -c Scripts/feature-promptfoo-agentic/promptfooconfig.grammar-header-enforced.yaml \
      -j 1 \
      -o "$output_header_enf"
  exit_code=$?
  (( overall_status |= exit_code ))
  return $exit_code
}

case "$mode" in
  all)
    run_structured
    run_structured_stress
    run_toolcall_all
    run_toolcall_quality_all
    run_grammar_constraints
    run_agentic_all
    run_frameworks_all
    run_opencode_all
    run_pi_all
    run_openclaw_all
    run_hermes_all
    ;;
  structured)
    run_structured
    ;;
  structured-stress)
    run_structured_stress
    ;;
  toolcall)
    run_toolcall_all
    ;;
  toolcall-quality)
    run_toolcall_quality_all
    ;;
  agentic)
    run_agentic_all
    ;;
  frameworks)
    run_frameworks_all
    ;;
  opencode)
    run_opencode_all
    ;;
  pi)
    run_pi_all
    ;;
  openclaw)
    run_openclaw_all
    ;;
  hermes)
    run_hermes_all
    ;;
  grammar-constraints)
    run_grammar_constraints
    ;;
  default)
    run_toolcall_profile default '^afm-default$' AFM_BASE_URL_DEFAULT
    ;;
  adaptive-xml)
    run_toolcall_profile adaptive-xml '^afm-adaptive-xml$' AFM_BASE_URL_ADAPTIVE_XML
    ;;
  adaptive-xml-grammar)
    run_toolcall_profile adaptive-xml-grammar '^afm-adaptive-xml-grammar$' AFM_BASE_URL_ADAPTIVE_XML_GRAMMAR
    ;;
  *)
    echo "Usage: Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh [all|structured|structured-stress|toolcall|toolcall-quality|grammar-constraints|agentic|frameworks|opencode|pi|openclaw|hermes|default|adaptive-xml|adaptive-xml-grammar]" >&2
    exit 1
    ;;
esac

exit "$overall_status"
