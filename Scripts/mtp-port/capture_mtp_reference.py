#!/usr/bin/env python3
"""P0 reference-capture for the afm MTP Swift port.

Drives the mtplx Python runtime to produce DETERMINISTIC ground truth for one MTP draft
step, so the Swift `Qwen3_5MTPHead` can be validated bit-for-bit against it.

What it captures (to Scripts/mtp-port/fixtures/mtp_ref_step.npz + .json):
  - prompt tokens + the trunk's last hidden state h_t after prefill (input to the MTP head)
  - the "primary" token that seeds the draft (argmax of the trunk's next-token logits)
  - the MTP head's draft logits for t+2, and its argmax (the drafted token)
  - a small slice of the logits + key intermediate stats for quick eyeballing

Run with the mtplx venv:
  MTPLX_MODEL_DIR=/Volumes/Crucial4TB/models/mtplx \
  /Volumes/Crucial4TB/bench/mtplxenv/bin/python Scripts/mtp-port/capture_mtp_reference.py
"""
import os, sys, json, pathlib
import numpy as np
import mlx.core as mx

OUT = pathlib.Path(__file__).resolve().parent / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = os.environ.get("MTPLX_MODEL_DIR", "/Volumes/Crucial4TB/models/mtplx")
MODEL = os.path.join(MODEL_DIR, "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed")

# Deterministic prompt — fixed, no sampling.
PROMPT = "The capital of France is"

def to_np(a):
    # MLX bf16 can't be buffer-converted by NumPy; cast to float32 in MLX first.
    if isinstance(a, mx.array):
        a = a.astype(mx.float32)
        mx.eval(a)              # eval() returns None — materialize in place, then convert
        return np.array(a)
    return np.asarray(a, dtype=np.float32)

def main():
    from mtplx import runtime as rt_mod
    print(f"[ref] loading {MODEL}", file=sys.stderr)
    rt = rt_mod.load(MODEL, mtp=True)
    assert rt.mtp_enabled, "MTP not enabled"
    tok = rt.tokenizer

    ids = tok.encode(PROMPT)
    print(f"[ref] prompt={PROMPT!r} -> {len(ids)} tokens: {ids}", file=sys.stderr)
    input_ids = mx.array([ids])

    # --- prefill the trunk, get last hidden + next-token logits (greedy primary) ---
    cache = rt.make_cache()
    logits, hidden = rt.forward_ar(input_ids, cache=cache, return_hidden=True)
    mx.eval(logits, hidden)
    # last position
    last_logits = logits[:, -1, :]            # (1, vocab)
    last_hidden = hidden[:, -1:, :]           # (1, 1, hidden)  -> input to MTP head
    primary = int(mx.argmax(last_logits, axis=-1).item())
    print(f"[ref] primary (argmax next token) = {primary} -> {tok.decode([primary])!r}", file=sys.stderr)

    # --- capture the trunk token embedding of `primary` (the head's other input) ---
    inner = getattr(rt.model, "language_model", rt.model)
    inner = getattr(inner, "model", inner)
    primary_embed = inner.embed_tokens(mx.array([[primary]]))   # (1,1,H)
    mx.eval(primary_embed)

    # --- one MTP draft step: head(last_hidden, primary) -> draft logits for t+2 ---
    mtp_cache = rt.make_mtp_cache()
    draft_logits, draft_hidden = rt.draft_mtp(
        last_hidden, [[primary]], mtp_cache=mtp_cache, return_hidden=True,
        mtp_hidden_variant="post_norm", mtp_depth=0, position_offset=None)
    mx.eval(draft_logits, draft_hidden)
    draft_last = draft_logits[:, -1, :]       # (1, vocab)
    drafted = int(mx.argmax(draft_last, axis=-1).item())
    print(f"[ref] drafted token (argmax MTP logits) = {drafted} -> {tok.decode([drafted])!r}", file=sys.stderr)

    # --- save fixture ---
    h_np = np.atleast_3d(to_np(last_hidden).astype(np.float32))   # (1,1,H)
    dl_np = np.atleast_2d(to_np(draft_last).astype(np.float32))    # (1,vocab)
    dh_np = np.atleast_3d(to_np(draft_hidden[:, -1:, :]).astype(np.float32))
    np.savez(OUT / "mtp_ref_step.npz",
             prompt_ids=np.array(ids, dtype=np.int32),
             primary=np.int32(primary),
             last_hidden=h_np,
             draft_logits=dl_np,
             draft_hidden=dh_np,
             drafted=np.int32(drafted))
    meta = {
        "model": MODEL,
        "prompt": PROMPT,
        "prompt_ids": ids,
        "primary_token": primary,
        "primary_text": tok.decode([primary]),
        "drafted_token": drafted,
        "drafted_text": tok.decode([drafted]),
        "hidden_size": int(h_np.shape[-1]),
        "vocab_size": int(dl_np.shape[-1]),
        "last_hidden_stats": {"mean": float(h_np.mean()), "std": float(h_np.std()),
                               "min": float(h_np.min()), "max": float(h_np.max())},
        "draft_logits_top5": [[int(i), float(dl_np[0, i])]
                               for i in np.argsort(-dl_np[0])[:5].tolist()],
    }
    (OUT / "mtp_ref_step.json").write_text(json.dumps(meta, indent=2))

    # --- Swift-friendly safetensors fixture (mlx-swift loads these natively) ---
    # Keep everything float32; the Swift test loads the head, runs on these inputs,
    # and must reproduce draft_hidden (post_norm) and draft_logits / drafted token.
    pe = to_np(primary_embed).astype(np.float32)
    mx.save_safetensors(str(OUT / "mtp_ref_step.safetensors"), {
        "last_hidden":  mx.array(h_np),       # (1,1,H)  head input: trunk hidden
        "primary_embed": mx.array(np.atleast_3d(pe)),  # (1,1,H) head input: embed(primary)
        "draft_logits": mx.array(dl_np),      # (1,vocab) expected
        "draft_hidden": mx.array(dh_np),      # (1,1,H)  expected post_norm hidden
    })
    print(f"[ref] wrote {OUT/'mtp_ref_step.npz'}, .json, .safetensors", file=sys.stderr)
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
