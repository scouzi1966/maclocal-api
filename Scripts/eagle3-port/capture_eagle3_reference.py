#!/usr/bin/env python3
"""P0 reference-capture for the afm EAGLE3 Swift port (dense Gemma4-31B).

Drives the mlx-vlm 0.6.0 EAGLE3 path to dump deterministic ground truth for one prefill +
one draft step, so the Swift Gemma4Eagle3Drafter and the verifier hidden-capture hook can be
validated bit/near-exactly against it.

Captures (to Scripts/eagle3-port/fixtures/eagle3_ref.safetensors + .json):
  - prompt_ids
  - cap2, cap30, cap57 : verifier hidden states at layers [2,30,57], LAST position (the
    `hidden_sink` append point — residual-stream output of each layer, pre final-norm)
  - fused_hidden       : drafter.fc(concat([cap2,cap30,cap57]))  (= _prepare_target_hidden)
  - primary            : argmax of the verifier's prefill logits (the seed token)
  - draft_logits_hot   : drafter _logits(...) over the 32000 hot vocab for the first draft step
  - draft_token        : the drafted token (hot->full via d2t)

Run:
  MACAFM_MLX_MODEL_CACHE=/Volumes/Crucial4TB/models/vesta-test-cache \
  HF_HOME=/Volumes/Crucial4TB/models/huggingface \
  /Volumes/Crucial4TB/bench/mlxvlm6env/bin/python Scripts/eagle3-port/capture_eagle3_reference.py
"""
import os, sys, json, pathlib
import numpy as np
import mlx.core as mx

OUT = pathlib.Path(__file__).resolve().parent / "fixtures"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = os.environ.get("MACAFM_MLX_MODEL_CACHE", "/Volumes/Crucial4TB/models/vesta-test-cache")
VERIFIER = os.path.join(CACHE, "mlx-community", "gemma-4-31b-it-4bit")
DRAFTER = "RedHatAI/gemma-4-31B-it-speculator.eagle3"
PROMPT = "The capital of France is"

def f32(a): a = a.astype(mx.float32); mx.eval(a); return np.array(a)

def main():
    from mlx_vlm.utils import load
    from mlx_vlm.speculative import load_drafter
    print(f"[ref] loading verifier {VERIFIER}", file=sys.stderr)
    model, processor = load(VERIFIER, processor_kwargs={"trust_remote_code": True})
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print(f"[ref] loading drafter {DRAFTER}", file=sys.stderr)
    drafter, _kind = load_drafter(DRAFTER, kind="eagle3")
    drafter.bind(model.language_model if hasattr(model, "language_model") else model)

    cap_ids = sorted(int(i) for i in drafter.config.eagle_aux_hidden_state_layer_ids)
    print(f"[ref] capture layer ids = {cap_ids}", file=sys.stderr)

    ids = tok.encode(PROMPT)
    print(f"[ref] prompt={PROMPT!r} -> {ids}", file=sys.stderr)
    input_ids = mx.array([ids])

    # --- verifier prefill with hidden capture ---
    lm = model.language_model if hasattr(model, "language_model") else model
    import mlx_vlm.models.gemma4.language as g4
    from mlx_vlm.models import cache as kvcache
    prompt_cache = kvcache.make_prompt_cache(lm)
    sink = []
    out = lm(input_ids, cache=prompt_cache, capture_layer_ids=cap_ids, hidden_sink=sink)
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits, *sink)
    primary = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    print(f"[ref] primary = {primary} -> {tok.decode([primary])!r}; captured {len(sink)} layers", file=sys.stderr)

    # last-position captured hidden at each layer
    caps = [s[:, -1:, :] for s in sink]            # each (1,1,5376)
    concat = mx.concatenate(caps, axis=-1)          # (1,1,3*5376)
    fused = drafter._prepare_target_hidden(concat)  # (1,1,5376) via fc
    mx.eval(fused)

    # --- one drafter forward step: feed primary + fused hidden ---
    drafter._cache = drafter.make_cache()
    drafter._next_position = 1
    tokn = mx.array([[primary]], dtype=mx.int32)
    h = drafter._forward_tokens(tokn, fused, mx.int32)   # (1,1,5376)
    dlogits = drafter._logits(h)                          # (1,1,32000) hot
    mx.eval(h, dlogits)
    hot = int(mx.argmax(dlogits[:, -1, :], axis=-1).item())
    full = int(drafter._draft_to_target(mx.array([hot]), mx.int32).item())
    print(f"[ref] draft hot={hot} -> full={full} -> {tok.decode([full])!r}", file=sys.stderr)

    mx.save_safetensors(str(OUT / "eagle3_ref.safetensors"), {
        "cap0": mx.array(f32(caps[0])), "cap1": mx.array(f32(caps[1])), "cap2": mx.array(f32(caps[2])),
        "fused_hidden": mx.array(f32(fused)),
        "draft_hidden": mx.array(f32(h)),
        "draft_logits_hot": mx.array(f32(dlogits[:, -1, :])),
    })
    meta = {
        "verifier": VERIFIER, "drafter": DRAFTER, "prompt": PROMPT, "prompt_ids": ids,
        "capture_layer_ids": cap_ids, "primary": primary, "primary_text": tok.decode([primary]),
        "draft_hot": hot, "draft_full": full, "draft_text": tok.decode([full]),
        "hidden_size": int(fused.shape[-1]), "draft_vocab": int(dlogits.shape[-1]),
        "draft_logits_top5": [[int(i), float(np.array(f32(dlogits[:, -1, :]))[0, i])]
                               for i in np.argsort(-np.array(f32(dlogits[:, -1, :]))[0])[:5]],
    }
    (OUT / "eagle3_ref.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"[ref] wrote {OUT/'eagle3_ref.safetensors'} + .json", file=sys.stderr)

if __name__ == "__main__":
    main()
