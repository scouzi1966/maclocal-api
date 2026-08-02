#!/usr/bin/env python3
"""Emit compact layer signatures from Vontra's DeepSeek-V4 0731 MLX runtime."""

import argparse
import importlib.util
from pathlib import Path

import mlx.core as mx


def load_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "vontra_deepseek_v4", model_dir / "deepseek_v4.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load deepseek_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(label: str, value: mx.array) -> None:
    flat = value.astype(mx.float32).reshape(-1)
    mx.eval(flat)
    count = min(4, flat.size)
    sample = [float(v) for v in flat[:count].tolist()]
    mean = float(mx.mean(flat))
    rms = float(mx.sqrt(mx.mean(mx.square(flat))))
    maximum = float(mx.max(mx.abs(flat)))
    print(
        f"[DSV4Python] {label} shape={list(value.shape)} "
        f"mean={mean:.9g} rms={rms:.9g} max={maximum:.9g} sample={sample}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--tokens",
        default="0,3476,477,260,11502,22896,128803,19905,9045,28,20370,128804,128822",
    )
    args = parser.parse_args()

    dsv4 = load_module(args.model_dir)
    config = dsv4.ModelArgs.from_json(args.model_dir / "config.json")
    model = dsv4.Transformer(config)
    loaded, names = dsv4.load_weights(model, args.model_dir, verbose=True)
    missing = names - loaded
    if missing:
        raise RuntimeError(f"Missing {len(missing)} tensors: {sorted(missing)[:8]}")

    token_values = [int(value) for value in args.tokens.split(",") if value]
    inputs = mx.array([token_values], mx.int32)
    print(f"[DSV4Python] tokens={token_values}", flush=True)

    h = model.embed(inputs)
    trace("embedding", h)
    h = mx.broadcast_to(
        h[:, :, None, :], (*h.shape[:-1], config.hc_mult, h.shape[-1])
    )
    for index, layer in enumerate(model.layers):
        if index == 0:
            residual = h
            x, post, comb = layer.hc_pre(
                h, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base
            )
            trace("layer.0.attn_hc_pre.x", x)
            trace("layer.0.attn_hc_pre.post", post)
            trace("layer.0.attn_hc_pre.comb", comb)
            x = layer.attn_norm(x)
            trace("layer.0.attn_norm", x)
            q_a = layer.attn.wq_a(x)
            q_a_activation = dsv4.act_quant_sim(x, 128)
            q_a_weight = mx.dequantize(
                layer.attn.wq_a.weight,
                layer.attn.wq_a.scales,
                group_size=32,
                bits=8,
                mode="mxfp8",
                dtype=x.dtype,
            )
            trace("layer.0.attention.wq_a.activation", q_a_activation)
            trace("layer.0.attention.wq_a.weight", q_a_weight)
            trace(
                "layer.0.attention.wq_a.direct",
                q_a_activation @ q_a_weight.T,
            )
            q_residual = layer.attn.q_norm(q_a)
            q_projected = layer.attn.wq_b(q_residual)
            trace("layer.0.attention.wq_a", q_a)
            trace("layer.0.attention.q_norm", q_residual)
            trace("layer.0.attention.wq_b", q_projected)
            q = q_projected.reshape(
                x.shape[0], x.shape[1], layer.attn.n_heads, layer.attn.head_dim
            )
            qf = q.astype(mx.float32)
            q = (
                qf
                * mx.rsqrt(
                    mx.mean(mx.square(qf), axis=-1, keepdims=True)
                    + layer.attn.eps
                )
            ).astype(q.dtype)
            trace("layer.0.attention.q_head_norm", q)
            rd = layer.attn.rope_head_dim
            q = mx.concatenate(
                [q[..., :-rd], layer.attn.rope(q[..., -rd:], 0)], axis=-1
            )
            kv_projected = layer.attn.wkv(x)
            kv = layer.attn.kv_norm(kv_projected)
            trace("layer.0.attention.wkv", kv_projected)
            trace("layer.0.attention.kv_norm", kv)
            kv = mx.concatenate(
                [kv[..., :-rd], layer.attn.rope(kv[..., -rd:], 0)], axis=-1
            )
            kv = mx.concatenate(
                [dsv4.act_quant_sim(kv[..., :-rd], 64), kv[..., -rd:]], axis=-1
            )
            trace("layer.0.attention.q", q)
            trace("layer.0.attention.kv", kv)
            topk = mx.array(
                dsv4.get_window_topk_idxs(
                    layer.attn.window_size, x.shape[0], x.shape[1], 0
                )
            )
            raw = dsv4.sparse_attn(
                q, kv, layer.attn.attn_sink, topk, layer.attn.softmax_scale
            )
            trace("layer.0.attention.raw", raw)
            derotated = mx.concatenate(
                [raw[..., :-rd], layer.attn.rope(raw[..., -rd:], 0, inverse=True)],
                axis=-1,
            )
            trace("layer.0.attention.derotated", derotated)
            grouped = derotated.reshape(
                x.shape[0], x.shape[1], layer.attn.n_groups, -1
            )
            if getattr(layer.attn, "_wo_a_deq", None) is None:
                layer.attn._wo_a_deq = mx.dequantize(
                    layer.attn.wo_a.weight,
                    layer.attn.wo_a.scales,
                    group_size=32,
                    bits=8,
                    mode="mxfp8",
                    dtype=mx.bfloat16,
                ).reshape(layer.attn.n_groups, layer.attn.o_lora_rank, -1)
                mx.eval(layer.attn._wo_a_deq)
            oproj_a = mx.einsum("bsgd,grd->bsgr", grouped, layer.attn._wo_a_deq)
            oproj_a = oproj_a.reshape(x.shape[0], x.shape[1], -1)
            trace("layer.0.attention.oproj_a", oproj_a)
            x = layer.attn.wo_b(oproj_a)
            trace("layer.0.attention.oproj_b", x)
            trace("layer.0.attention", x)
            h = layer.hc_post(x, residual, post, comb)
            trace("layer.0.attn_hc_post", h)

            residual = h
            x, post, comb = layer.hc_pre(
                h, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
            )
            trace("layer.0.ffn_hc_pre.x", x)
            trace("layer.0.ffn_hc_pre.post", post)
            trace("layer.0.ffn_hc_pre.comb", comb)
            x = layer.ffn_norm(x)
            trace("layer.0.ffn_norm", x)
            x = layer.ffn(x, inputs)
            trace("layer.0.moe", x)
            h = layer.hc_post(x, residual, post, comb)
            trace("layer.0.ffn_hc_post", h)
        else:
            h = layer(h, 0, inputs)
        trace(f"layer.{index}", h)
    h = dsv4.hc_head_reduce(
        h,
        model.hc_head_fn,
        model.hc_head_scale,
        model.hc_head_base,
        config.norm_eps,
        config.hc_eps,
    )
    trace("hc_head", h)
    h = model.norm(h)
    trace("norm", h)
    logits = model.head(h)
    trace("logits", logits)
    print(f"[DSV4Python] argmax={int(mx.argmax(logits[0, -1]))}", flush=True)


if __name__ == "__main__":
    main()
