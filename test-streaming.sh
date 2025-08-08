#!/bin/bash

echo "Testing streaming functionality..."
echo "=================================="

# Test streaming functionality with verbose output
curl -X POST http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Say hello world"}],
    "stream": true
  }' \
  --no-buffer -v

echo ""
echo "Test completed."
