#!/usr/bin/env python3
"""
XGrammar bridge for structured output in AFM.

Communicates via JSON-lines over stdin/stdout. The Swift server spawns
this process and sends commands to compile JSON schemas, generate token
masks, and advance grammar state.

Protocol:
  -> {"cmd":"compile","schema":{...},"vocab_size":151936,"tokenizer_path":"/path/to/tokenizer.json"}
  <- {"ok":true,"grammar_id":"abc123"}

  -> {"cmd":"mask","grammar_id":"abc123"}
  <- {"ok":true,"allowed":[0,1,2,...]}

  -> {"cmd":"accept","grammar_id":"abc123","token_id":42}
  <- {"ok":true}

  -> {"cmd":"reset","grammar_id":"abc123"}
  <- {"ok":true}

  -> {"cmd":"is_terminated","grammar_id":"abc123"}
  <- {"ok":true,"terminated":false}

  -> {"cmd":"release","grammar_id":"abc123"}
  <- {"ok":true}
"""

import sys
import json
import uuid

try:
    import xgrammar as xgr
    XGRAMMAR_AVAILABLE = True
except ImportError:
    XGRAMMAR_AVAILABLE = False


def send(obj):
    """Write a JSON response to stdout and flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def send_error(msg):
    """Write an error response to stdout and flush."""
    send({"ok": False, "error": msg})


def send_not_installed():
    """Write the standard 'not installed' error response."""
    send_error("xgrammar not installed")


# Cache tokenizer info objects keyed by tokenizer path to avoid reloading
tokenizer_cache = {}

# Active grammar matchers keyed by grammar_id
grammar_matchers = {}


def handle_compile(req):
    schema = req.get("schema")
    vocab_size = req.get("vocab_size")
    tokenizer_path = req.get("tokenizer_path")

    if schema is None:
        send_error("missing field: schema")
        return
    if vocab_size is None:
        send_error("missing field: vocab_size")
        return
    if tokenizer_path is None:
        send_error("missing field: tokenizer_path")
        return

    # Load or retrieve cached TokenizerInfo
    if tokenizer_path not in tokenizer_cache:
        try:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer_path, vocab_size=vocab_size
            )
            tokenizer_cache[tokenizer_path] = tokenizer_info
        except Exception as e:
            send_error(f"failed to load tokenizer: {e}")
            return
    else:
        tokenizer_info = tokenizer_cache[tokenizer_path]

    # Compile the JSON schema grammar
    try:
        schema_str = json.dumps(schema)
        compiled_grammar = xgr.Grammar.from_json_schema(schema_str)
        compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_grammar(compiled_grammar)
    except Exception as e:
        send_error(f"failed to compile grammar: {e}")
        return

    # Create a GrammarMatcher and store it
    try:
        matcher = xgr.GrammarMatcher(compiled)
    except Exception as e:
        send_error(f"failed to create grammar matcher: {e}")
        return

    grammar_id = uuid.uuid4().hex[:12]
    grammar_matchers[grammar_id] = matcher
    send({"ok": True, "grammar_id": grammar_id})


def handle_mask(req):
    grammar_id = req.get("grammar_id")
    if grammar_id is None:
        send_error("missing field: grammar_id")
        return

    matcher = grammar_matchers.get(grammar_id)
    if matcher is None:
        send_error(f"unknown grammar_id: {grammar_id}")
        return

    try:
        bitmask = matcher.find_next_token_bitmask()
        allowed = xgr.bitmask_to_list(bitmask)
        send({"ok": True, "allowed": allowed})
    except Exception as e:
        send_error(f"mask error: {e}")


def handle_accept(req):
    grammar_id = req.get("grammar_id")
    token_id = req.get("token_id")

    if grammar_id is None:
        send_error("missing field: grammar_id")
        return
    if token_id is None:
        send_error("missing field: token_id")
        return

    matcher = grammar_matchers.get(grammar_id)
    if matcher is None:
        send_error(f"unknown grammar_id: {grammar_id}")
        return

    try:
        matcher.accept_token(token_id)
        send({"ok": True})
    except Exception as e:
        send_error(f"accept error: {e}")


def handle_reset(req):
    grammar_id = req.get("grammar_id")
    if grammar_id is None:
        send_error("missing field: grammar_id")
        return

    matcher = grammar_matchers.get(grammar_id)
    if matcher is None:
        send_error(f"unknown grammar_id: {grammar_id}")
        return

    try:
        matcher.reset()
        send({"ok": True})
    except Exception as e:
        send_error(f"reset error: {e}")


def handle_is_terminated(req):
    grammar_id = req.get("grammar_id")
    if grammar_id is None:
        send_error("missing field: grammar_id")
        return

    matcher = grammar_matchers.get(grammar_id)
    if matcher is None:
        send_error(f"unknown grammar_id: {grammar_id}")
        return

    try:
        terminated = matcher.is_terminated()
        send({"ok": True, "terminated": terminated})
    except Exception as e:
        send_error(f"is_terminated error: {e}")


def handle_release(req):
    grammar_id = req.get("grammar_id")
    if grammar_id is None:
        send_error("missing field: grammar_id")
        return

    if grammar_id not in grammar_matchers:
        send_error(f"unknown grammar_id: {grammar_id}")
        return

    del grammar_matchers[grammar_id]
    send({"ok": True})


HANDLERS = {
    "compile": handle_compile,
    "mask": handle_mask,
    "accept": handle_accept,
    "reset": handle_reset,
    "is_terminated": handle_is_terminated,
    "release": handle_release,
}


def main():
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            send_error(f"invalid JSON: {e}")
            continue

        cmd = req.get("cmd")
        if not cmd:
            send_error("missing field: cmd")
            continue

        if not XGRAMMAR_AVAILABLE:
            send_not_installed()
            continue

        handler = HANDLERS.get(cmd)
        if handler is None:
            send_error(f"unknown command: {cmd}")
            continue

        try:
            handler(req)
        except Exception as e:
            send_error(f"unhandled error in {cmd}: {e}")


if __name__ == "__main__":
    main()
