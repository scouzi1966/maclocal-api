#!/bin/bash

# Default port
PORT=${1:-8080}

echo "Testing afm streaming metrics functionality..."
echo "Port: $PORT"
echo "================================================"
echo ""

# Test streaming functionality with focus on final metrics chunk
echo "Looking for metrics in final chunk..."
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "foundation",
    "messages": [{"role": "user", "content": "Tell me a very short story"}],
    "stream": true
  }' \
  --no-buffer -s | grep -E "(prompt_tokens_per_second|completion_tokens_per_second|completion_time|total_time|usage)" | tail -5

echo ""
echo "Full streaming response with verbose output:"
echo "-------------------------------------------"

curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "foundation", 
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }' \
  --no-buffer

echo ""
echo "Metrics test completed."