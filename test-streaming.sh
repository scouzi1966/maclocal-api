#!/bin/bash

# Default port
PORT=${1:-8080}

echo "Testing streaming functionality..."
echo "Port: $PORT"
echo "=================================="

# Test streaming functionality with verbose output
curl -X POST http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "foundation",
    "messages": [{"role": "user", "content": "Say hello world"}],
    "stream": true
  }' \
  --no-buffer -v

echo ""
echo "Test completed."
