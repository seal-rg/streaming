#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_init_to_tokens.py

Add ONLY these paired XML-style routing tokens and initialize their embeddings:

  <to:all> </to:all>
  <to:C0>  </to:C0>
  <to:C1>  </to:C1>
  ...

Initialization = "anchor copy/mean":
- copy: new_token_emb = emb[anchor_id]
- mean: new_token_emb = mean(emb[anchor_ids])

By default, anchor is eos (or pad). You can also provide --anchor-text to derive anchor_ids.

Example:
  python add_init_to_tokens.py \
    --model /path/to/model \
    --tokenizer /path/to/tokenizer \
    --out /path/to/out \
    --max-channels 12 \
    --init-strategy copy \
    --anchor-text "" \
    --verify
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# ----------------------------- helpers -----------------------------

def is_tied(model: PreTrainedModel) -> bool:
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if emb is None or head is None:
        return False
    try:
        return emb.weight.data_ptr() == head.weight.data_ptr()
    except Exception:
        return False


def default_anchor_id(tok: PreTrainedTokenizer) -> int:
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    return 0


def tokenize_anchor_ids(tok: PreTrainedTokenizer, anchor_text: str) -> List[int]:
    if not anchor_text:
        return []
    ids = tok(anchor_text, add_special_tokens=False).input_ids
    unk = tok.unk_token_id
    if unk is None:
        return [int(i) for i in ids]
    return [int(i) for i in ids if int(i) != int(unk)]


def build_to_tokens(max_channels: int, include_all: bool = True, channel_prefix: str = "C") -> List[str]:
    toks: List[str] = []
    if include_all:
        toks += ["<to:all>", "</to:all>"]
    for i in range(int(max_channels)):
        ch = f"{channel_prefix}{i}"
        toks += [f"<to:{ch}>", f"</to:{ch}>"]
    # de-dupe keep order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@torch.no_grad()
def init_token_embedding(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    token: str,
    anchor_ids: List[int],
    strategy: str = "copy",   # copy | mean
    noise_std: float = 0.0,
) -> Dict:
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = is_tied(model)

    new_id = int(tok.convert_tokens_to_ids(token))
    if new_id < 0:
        raise ValueError(f"Token not found after adding: {token}")

    if not anchor_ids:
        anchor_ids = [default_anchor_id(tok)]

    W_in = emb.weight
    if strategy == "mean":
        ids_t = torch.tensor(anchor_ids, device=W_in.device, dtype=torch.long)
        v_in = W_in[ids_t].mean(dim=0)
    else:
        v_in = W_in[int(anchor_ids[0])].clone()

    if noise_std > 0:
        v_in = v_in + torch.randn_like(v_in) * float(noise_std)

    W_in[new_id].copy_(v_in)

    # lm_head if untied and same vocab
    if head is not None and (not tied) and head.weight.shape[0] == W_in.shape[0]:
        W_out = head.weight
        if strategy == "mean":
            ids_t = torch.tensor(anchor_ids, device=W_out.device, dtype=torch.long)
            v_out = W_out[ids_t].mean(dim=0)
        else:
            v_out = W_out[int(anchor_ids[0])].clone()
        if noise_std > 0:
            v_out = v_out + torch.randn_like(v_out) * float(noise_std)
        W_out[new_id].copy_(v_out)

    return {
        "token": token,
        "id": new_id,
        "anchor_ids": [int(x) for x in anchor_ids],
        "strategy": strategy,
        "noise_std": float(noise_std),
        "tied": bool(tied),
    }


def add_and_init_tokens(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    max_channels: int,
    include_all: bool,
    channel_prefix: str,
    init_strategy: str,
    anchor_text: str,
    noise_std: float,
    pad_to_multiple_of: int = 64,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict]:
    # ensure pad exists (safe)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    desired = build_to_tokens(max_channels=max_channels, include_all=include_all, channel_prefix=channel_prefix)
    vocab = tok.get_vocab() or {}
    new_tokens = [t for t in desired if t not in vocab]

    info: Dict = {
        "requested_tokens": desired,
        "new_tokens": new_tokens,
        "token_to_id": {},
        "init_details": [],
    }

    if not new_tokens:
        info["note"] = "All tokens already exist; no resize/init performed."
        info["token_to_id"] = {t: int(tok.convert_tokens_to_ids(t)) for t in desired if t in tok.get_vocab()}
        return model, tok, info

    # add tokens
    added = tok.add_special_tokens({"additional_special_tokens": new_tokens})

    # resize
    old_n = int(model.get_input_embeddings().weight.shape[0])
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=pad_to_multiple_of)
    new_n = int(model.get_input_embeddings().weight.shape[0])
    info["resize"] = {"old_vocab": old_n, "new_vocab": new_n, "added": int(added)}

    # anchors
    anchor_ids = tokenize_anchor_ids(tok, anchor_text)
    if not anchor_ids:
        anchor_ids = [default_anchor_id(tok)]
    info["anchor_text"] = anchor_text
    info["anchor_ids"] = [int(x) for x in anchor_ids]
    info["init_strategy"] = init_strategy
    info["noise_std"] = float(noise_std)

    # init each new token
    for t in new_tokens:
        info["init_details"].append(
            init_token_embedding(model, tok, t, anchor_ids=anchor_ids, strategy=init_strategy, noise_std=noise_std)
        )

    info["token_to_id"] = {t: int(tok.convert_tokens_to_ids(t)) for t in desired if t in tok.get_vocab()}
    return model, tok, info


def verify_norms(model: PreTrainedModel, tok: PreTrainedTokenizer, tokens: List[str]) -> None:
    W = model.get_input_embeddings().weight.detach()
    print("\nVerify embedding norms:")
    for t in tokens:
        if t not in tok.get_vocab():
            print(f"  ✗ {t:12s} : not in vocab")
            continue
        tid = int(tok.convert_tokens_to_ids(t))
        norm = float(W[tid].norm().cpu().item())
        print(f"  ✓ {t:12s} : id={tid:7d} norm={norm:.4f}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser("Add+init <to:...></to:...> tokens (C-style)")
    ap.add_argument("--model", required=True, help="Base model path/name")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer path/name")
    ap.add_argument("--out", required=True, help="Output dir for updated model+tokenizer")
    ap.add_argument("--max-channels", type=int, default=10, help="Add C0..C{N-1}")
    ap.add_argument("--channel-prefix", type=str, default="C", help="Prefix used in tags: C0,C1,...")
    ap.add_argument("--no-all", action="store_true", help="Do not add <to:all></to:all>")

    ap.add_argument("--init-strategy", choices=["copy", "mean"], default="copy")
    ap.add_argument("--anchor-text", type=str, default="", help='Anchor text; "" -> fallback eos/pad anchor')
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--pad-to-multiple-of", type=int, default=64)

    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--save-info-json", action="store_true", help="Save init info as init_info.json in --out")

    args = ap.parse_args()

    dtype_map = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map="auto" )
    model.to(args.device)

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    model, tok, info = add_and_init_tokens(
        model=model,
        tok=tok,
        max_channels=args.max_channels,
        include_all=(not args.no_all),
        channel_prefix=args.channel_prefix,
        init_strategy=args.init_strategy,
        anchor_text=args.anchor_text,
        noise_std=args.noise_std,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    print("\n=== Summary ===")
    print(f"Requested tokens: {len(info['requested_tokens'])}")
    print(f"New tokens added: {len(info['new_tokens'])}")
    if "resize" in info:
        print(f"Resize: {info['resize']}")
    print(f"Anchor ids: {info.get('anchor_ids')}")
    if info["new_tokens"]:
        print("New tokens (preview):", info["new_tokens"][:12])

    if args.verify:
        samples = []
        if "<to:all>" in tok.get_vocab():
            samples += ["<to:all>", "</to:all>"]
        samples += [f"<to:{args.channel_prefix}0>", f"</to:{args.channel_prefix}0>"]
        if args.max_channels > 1:
            samples += [f"<to:{args.channel_prefix}1>", f"</to:{args.channel_prefix}1>"]
        verify_norms(model, tok, samples)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\n✓ Saved model+tokenizer to: {args.out}")

    if args.save_info_json:
        with open(os.path.join(args.out, "init_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print("✓ Saved init_info.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
