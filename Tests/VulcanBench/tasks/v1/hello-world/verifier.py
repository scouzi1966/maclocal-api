#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> None:
    workspace = Path("/workspace/repo") if Path("/workspace/repo").exists() else Path(".")
    hello = workspace / "hello.py"
    if not hello.exists():
        print(json.dumps({"functional": 0.0, "details": ["hello.py missing"]}))
        sys.exit(1)
    content = hello.read_text()
    if "hello from vulcanbench" in content:
        print(json.dumps({"functional": 1.0, "details": ["exact string present"]}))
    else:
        print(json.dumps({"functional": 0.0, "details": ["string mismatch"]}))
    sys.exit(0)

if __name__ == "__main__":
    main()
