#!/bin/bash
set -uo pipefail
# Note: no -e — expect timeouts return exit 1 but shouldn't kill the whole run

# === Configuration ===
AFM_BIN="/Volumes/edata/dev/git/NIGHTLY/maclocal-api/.build/release/afm"
PROMPT_FILE="/Volumes/edata/dev/git/NIGHTLY/maclocal-api/test-reports/opencode-tooling-20260308_112707/prompt.md"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="/Volumes/edata/dev/git/NIGHTLY/maclocal-api/test-reports/opencode-tooling-${TIMESTAMP}"
TEST_PORT=9999
OC_PORT=4096
ITERATIONS=3
MODELS=(
    "mlx-community/Qwen3-Coder-Next-4bit"
    "mlx-community/Qwen3.5-27B-8bit"
    "mlx-community/Qwen3.5-35B-A3B-4bit"
)

mkdir -p "$REPORT_DIR"
cp "$PROMPT_FILE" "$REPORT_DIR/prompt.md"
PROMPT=$(cat "$PROMPT_FILE")

echo "=============================================="
echo "OpenCode Tooling Test — $TIMESTAMP"
echo "Models: ${MODELS[*]}"
echo "Iterations per model: $ITERATIONS"
echo "Report dir: $REPORT_DIR"
echo "=============================================="

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG="${MODEL//\//_}"
    echo ""
    echo "====== MODEL: $MODEL ======"

    # Kill anything on the ports
    kill $(lsof -ti :$TEST_PORT) 2>/dev/null || true
    kill $(lsof -ti :$OC_PORT) 2>/dev/null || true
    sleep 2

    # Start afm with --vv
    echo "[$(date +%H:%M:%S)] Starting afm: $MODEL on port $TEST_PORT with --vv"
    AFM_DEBUG=1 MACAFM_MLX_MODEL_CACHE=/Volumes/edata/models/vesta-test-cache \
        "$AFM_BIN" mlx -m "$MODEL" --port $TEST_PORT \
        --enable-prefix-caching --tool-call-parser afm_adaptive_xml \
        --vv --seed 0 \
        > "$REPORT_DIR/${MODEL_SLUG}-afm.log" 2>&1 &
    AFM_PID=$!
    echo "[$(date +%H:%M:%S)] afm PID: $AFM_PID"

    # Wait for afm to be ready
    echo -n "[$(date +%H:%M:%S)] Waiting for afm..."
    for i in $(seq 1 120); do
        if curl -sf http://127.0.0.1:${TEST_PORT}/v1/models >/dev/null 2>&1; then
            echo " ready (${i}s)"
            break
        fi
        if ! kill -0 $AFM_PID 2>/dev/null; then
            echo " CRASHED"
            echo "afm crashed during model load. Check $REPORT_DIR/${MODEL_SLUG}-afm.log"
            break 2
        fi
        sleep 1
    done

    if ! curl -sf http://127.0.0.1:${TEST_PORT}/v1/models >/dev/null 2>&1; then
        echo " TIMEOUT — skipping model"
        kill $AFM_PID 2>/dev/null || true
        wait $AFM_PID 2>/dev/null || true
        continue
    fi

    for RUN in $(seq 1 $ITERATIONS); do
        echo ""
        echo "  --- Run $RUN/$ITERATIONS for $MODEL ---"

        # Clean workdir
        OC_WORKDIR="/tmp/opencode-serve-${TIMESTAMP}-run${RUN}"
        rm -rf "$OC_WORKDIR"
        mkdir -p "$OC_WORKDIR"
        cd "$OC_WORKDIR" && git init -q && cd - >/dev/null

        # Write OpenCode config
        cat > "$OC_WORKDIR/opencode.json" << OCEOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "afm-test",
      "options": {
        "baseURL": "http://localhost:${TEST_PORT}/v1"
      },
      "models": {
        "${MODEL}": {
          "name": "${MODEL}"
        }
      }
    }
  }
}
OCEOF

        # Start opencode serve
        echo "  [$(date +%H:%M:%S)] Starting opencode serve on port $OC_PORT"
        cd "$OC_WORKDIR"
        opencode serve --port $OC_PORT --print-logs --log-level "DEBUG" \
            >> "$REPORT_DIR/${MODEL_SLUG}-opencode-serve.log" 2>&1 &
        OC_SERVE_PID=$!
        cd - >/dev/null

        # Wait for serve
        echo -n "  [$(date +%H:%M:%S)] Waiting for opencode serve..."
        for i in $(seq 1 30); do
            if curl -sf http://127.0.0.1:${OC_PORT}/ >/dev/null 2>&1; then
                echo " ready (${i}s)"
                break
            fi
            sleep 1
        done

        if ! curl -sf http://127.0.0.1:${OC_PORT}/ >/dev/null 2>&1; then
            echo " TIMEOUT — skipping run"
            kill $OC_SERVE_PID 2>/dev/null || true
            wait $OC_SERVE_PID 2>/dev/null || true
            continue
        fi

        # Run opencode via expect (PTY required)
        echo "  [$(date +%H:%M:%S)] Running opencode run --attach (format json, thinking enabled)"
        /usr/bin/expect << EXPECT_EOF > "$REPORT_DIR/${MODEL_SLUG}-run${RUN}-opencode.json" 2>&1
set timeout 900
log_user 1

spawn opencode run --attach http://localhost:${OC_PORT} --log-level "DEBUG" --print-logs --format json --thinking "${PROMPT}"
expect {
    timeout { puts "TIMEOUT"; exit 1 }
    eof { puts "EOF"; exit 0 }
}
EXPECT_EOF
        OC_EXIT=$?
        echo "  [$(date +%H:%M:%S)] opencode run finished (exit=$OC_EXIT)"

        # Copy the latest opencode log for this run
        LATEST_OC_LOG=$(ls -t ~/.local/share/opencode/log/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_OC_LOG" ]; then
            cp "$LATEST_OC_LOG" "$REPORT_DIR/${MODEL_SLUG}-run${RUN}-opencode-debug.log"
        fi

        # Stop opencode serve
        kill $OC_SERVE_PID 2>/dev/null || true
        wait $OC_SERVE_PID 2>/dev/null || true

        # Clean up workdir
        rm -rf "$OC_WORKDIR"

        echo "  [$(date +%H:%M:%S)] Run $RUN complete"
        sleep 2
    done

    # Stop afm
    echo "[$(date +%H:%M:%S)] Stopping afm (PID $AFM_PID)"
    kill $AFM_PID 2>/dev/null || true
    wait $AFM_PID 2>/dev/null || true
    sleep 2
done

echo ""
echo "=============================================="
echo "All runs complete. Report dir: $REPORT_DIR"
echo "=============================================="
ls -la "$REPORT_DIR/"
