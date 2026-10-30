#!/usr/bin/env python3
"""
Convert a Hugging Face-format Llama 3.x checkpoint (model.safetensors + config.json +
tokenizer.json) into Meta's original format used by ExecuTorch's Llama export:

    consolidated.00.pth   weights with Meta parameter names
    params.json           model hyper-parameters
    tokenizer.model       tiktoken-format BPE ranks (base vocabulary, no special tokens)

Usage:
    python advisor/convert_hf_to_meta.py --hf ~/traceiq/llm/llama32-1b-instruct \
                                         --out ~/traceiq/llm/llama32-1b-instruct-meta
"""
import argparse, base64, json
from pathlib import Path

import torch
from safetensors.torch import load_file


def unpermute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Undo the rotary-embedding row permutation applied by Meta->HF conversion."""
    dim1, dim2 = w.shape
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def ffn_hidden(dim: int, multiplier: float, multiple_of: int) -> int:
    """Meta's FeedForward hidden-size rule."""
    h = int(2 * (4 * dim) / 3)
    h = int(multiplier * h)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def bytes_to_unicode():
    """GPT-2 byte-level BPE mapping (byte -> printable unicode char)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="folder with model.safetensors, config.json, tokenizer.json")
    ap.add_argument("--out", required=True, help="output folder for Meta-format files")
    a = ap.parse_args()
    hf, out = Path(a.hf).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # ---------- params.json ----------
    cfg = json.loads((hf / "config.json").read_text())
    dim, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    rope = cfg.get("rope_scaling") or {}
    params = {
        "dim": dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "vocab_size": cfg["vocab_size"],
        "ffn_dim_multiplier": 1.5,
        "multiple_of": 256,
        "norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg.get("rope_theta", 500000.0),
        "use_scaled_rope": rope.get("rope_type") == "llama3",
    }
    if ffn_hidden(dim, params["ffn_dim_multiplier"], params["multiple_of"]) != cfg["intermediate_size"]:
        raise SystemExit(f"FFN size rule does not reproduce intermediate_size={cfg['intermediate_size']}; "
                         "this converter is written for Llama 3.2 1B/3B.")
    (out / "params.json").write_text(json.dumps(params, indent=2))
    print("params.json:", params)

    # ---------- weights ----------
    shards = sorted(hf.glob("*.safetensors"))
    sd = {}
    for s in shards:
        sd.update(load_file(str(s)))
    head_dim = dim // n_heads
    meta = {
        "tok_embeddings.weight": sd["model.embed_tokens.weight"],
        "norm.weight": sd["model.norm.weight"],
        # Llama 3.2 1B/3B tie the output layer to the embeddings
        "output.weight": sd.get("lm_head.weight", sd["model.embed_tokens.weight"]),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        q = f"layers.{i}."
        meta[q + "attention.wq.weight"] = unpermute(sd[p + "self_attn.q_proj.weight"], n_heads)
        meta[q + "attention.wk.weight"] = unpermute(sd[p + "self_attn.k_proj.weight"], n_kv)
        meta[q + "attention.wv.weight"] = sd[p + "self_attn.v_proj.weight"]
        meta[q + "attention.wo.weight"] = sd[p + "self_attn.o_proj.weight"]
        meta[q + "feed_forward.w1.weight"] = sd[p + "mlp.gate_proj.weight"]
        meta[q + "feed_forward.w2.weight"] = sd[p + "mlp.down_proj.weight"]
        meta[q + "feed_forward.w3.weight"] = sd[p + "mlp.up_proj.weight"]
        meta[q + "attention_norm.weight"] = sd[p + "input_layernorm.weight"]
        meta[q + "ffn_norm.weight"] = sd[p + "post_attention_layernorm.weight"]
        assert meta[q + "attention.wk.weight"].shape == (n_kv * head_dim, dim)
    meta = {k: v.to(torch.bfloat16).contiguous() for k, v in meta.items()}
    used = 3 + 9 * n_layers
    print(f"converted {len(meta)} tensors (expected {used}); HF tensors read: {len(sd)}")
    torch.save(meta, out / "consolidated.00.pth")
    print("wrote", out / "consolidated.00.pth")

    # ---------- tokenizer.model (tiktoken format) ----------
    tok = json.loads((hf / "tokenizer.json").read_text())
    vocab = tok["model"]["vocab"]
    byte_dec = {c: b for b, c in bytes_to_unicode().items()}
    special = {t["content"] for t in tok.get("added_tokens", [])}
    lines, n_base = [], 0
    for token, rank in sorted(vocab.items(), key=lambda kv: kv[1]):
        if token in special:
            continue
        raw = bytes(byte_dec[ch] for ch in token)
        lines.append(f"{base64.b64encode(raw).decode()} {rank}")
        n_base += 1
    (out / "tokenizer.model").write_text("\n".join(lines) + "\n")
    print(f"wrote tokenizer.model with {n_base} base tokens (Llama 3 expects 128000)")
    if n_base != 128000:
        print("WARNING: unexpected base-vocabulary size; check tokenizer.json")

    # keep the original HF tokenizer too (some ExecuTorch runners read tokenizer.json directly)
    (out / "tokenizer.json").write_bytes((hf / "tokenizer.json").read_bytes())
    print("done:", out)


if __name__ == "__main__":
    main()
