#!/usr/bin/env python3
"""Convert lean ZeRO-3 checkpoints to HF safetensors. CPU-only, no GPU needed.

Usage:
    # Single checkpoint:
    uv run python scripts/convert_lean_checkpoint.py results/s18-qwen27b_.../lean_best \
        --output ${CHECKPOINTS_ROOT}/s18-qwen27b_...

    # All lean checkpoints under a directory:
    uv run python scripts/convert_lean_checkpoint.py results/ \
        --output ${CHECKPOINTS_ROOT}/ --all

    # Dry run (just list what would be converted):
    uv run python scripts/convert_lean_checkpoint.py results/ --all --dry-run

    # Recover scattered partitions (ranks wrote to different output_dirs):
    uv run python scripts/convert_lean_checkpoint.py --merge \
        results/s18-qwen27b_..._192145/lean_best \
        results/s18-qwen27b_..._192147/lean_best \
        results/s18-qwen27b_..._192144/lean_best \
        --output ${CHECKPOINTS_ROOT}/s18-qwen27b_...
"""

import argparse
import glob
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def collect_shards(tag_dir):
    """Collect optim_states and model_states files from a tag directory."""
    optim = sorted(glob.glob(os.path.join(tag_dir, "*_optim_states.pt")))
    model = sorted(glob.glob(os.path.join(tag_dir, "*_model_states.pt")))
    return optim, model


def merge_scattered_checkpoints(dirs, merged_dir, tag="best"):
    """Merge partitions scattered across multiple directories into one."""
    os.makedirs(os.path.join(merged_dir, tag), exist_ok=True)

    all_optim = []
    all_model = []
    config_source = None

    for d in dirs:
        tag_dir = os.path.join(d, tag) if os.path.isdir(os.path.join(d, tag)) else d
        optim, model = collect_shards(tag_dir)
        all_optim.extend(optim)
        all_model.extend(model)
        # Pick first dir with config/tokenizer files as source
        if config_source is None:
            for fname in os.listdir(d if os.path.isdir(d) else os.path.dirname(d)):
                if fname in ("config.json", "tokenizer.json"):
                    config_source = d if os.path.isdir(d) else os.path.dirname(d)
                    break

    if not all_optim:
        log.error(f"No optim_states files found across {len(dirs)} directories")
        return None

    # Check for duplicates (same rank file from different dirs)
    basenames = [os.path.basename(f) for f in all_optim]
    if len(basenames) != len(set(basenames)):
        # Deduplicate — keep the largest file for each rank (most complete)
        by_name = {}
        for f in all_optim:
            bn = os.path.basename(f)
            if bn not in by_name or os.path.getsize(f) > os.path.getsize(by_name[bn]):
                by_name[bn] = f
        all_optim = sorted(by_name.values())
        log.info(f"  Deduplicated to {len(all_optim)} unique shards")

    # Copy all shards into merged directory
    merged_tag = os.path.join(merged_dir, tag)
    for f in all_optim + all_model:
        dst = os.path.join(merged_tag, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)

    # Also copy latest file for zero_to_fp32
    latest_path = os.path.join(merged_dir, "latest")
    with open(latest_path, "w") as f:
        f.write(tag)

    log.info(f"  Merged {len(all_optim)} optim + {len(all_model)} model shards into {merged_tag}")
    return merged_dir, config_source


def convert(ckpt_dir, out_dir, tag="best", keep=False):
    """Convert one lean checkpoint to HF safetensors."""
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint  # type: ignore
    from safetensors.torch import save_file

    tag_dir = os.path.join(ckpt_dir, tag)
    if not os.path.isdir(tag_dir):
        log.warning(f"No checkpoint at {tag_dir}, skipping")
        return False

    optim_files = sorted(glob.glob(os.path.join(tag_dir, "*_optim_states.pt")))
    if not optim_files:
        log.warning(f"No optim_states files in {tag_dir}, skipping")
        return False

    log.info(f"Converting {ckpt_dir} ({len(optim_files)} shards) → {out_dir}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)

    # Deduplicate tied tensors and cast to bf16
    import torch
    seen_ptrs = {}
    deduped = {}
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs[ptr] = k
            deduped[k] = v.contiguous().to(torch.bfloat16)
    n_deduped = len(state_dict) - len(deduped)
    del state_dict

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.safetensors")
    save_file(deduped, out_path)
    size_gb = os.path.getsize(out_path) / 1e9
    n_params = sum(v.numel() for v in deduped.values())
    del deduped

    log.info(f"  {n_params / 1e9:.1f}B params, {size_gb:.1f}GB, {n_deduped} tied tensors deduped")

    # Copy all non-checkpoint files (config, tokenizer, metadata, etc.)
    for search_dir in [ckpt_dir, os.path.dirname(ckpt_dir)]:
        if not os.path.isdir(search_dir):
            continue
        for fname in os.listdir(search_dir):
            src = os.path.join(search_dir, fname)
            if os.path.isfile(src) and not fname.endswith(".pt"):
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    if not keep:
        shutil.rmtree(tag_dir)
        log.info(f"  Cleaned up {tag_dir}")

    return True


def find_lean_checkpoints(base_dir):
    """Find directories under base_dir that contain a best/ lean checkpoint.

    Handles both layouts:
      - base_dir/<run>/best/*_optim_states.pt          (new)
      - base_dir/<run>/lean_best/best/*_optim_states.pt (old)

    Returns list of (ckpt_dir, run_name) where ckpt_dir contains best/.
    """
    results = []
    for entry in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        # Try direct: <run>/best/
        best_dir = os.path.join(run_dir, "best")
        if os.path.isdir(best_dir) and glob.glob(os.path.join(best_dir, "*_optim_states.pt")):
            results.append((run_dir, entry))
            continue
        # Try old layout: <run>/lean_best/best/
        lean_dir = os.path.join(run_dir, "lean_best")
        best_dir = os.path.join(lean_dir, "best")
        if os.path.isdir(best_dir) and glob.glob(os.path.join(best_dir, "*_optim_states.pt")):
            results.append((lean_dir, entry))
    return results


def find_scattered_groups(base_dir):
    """Find groups of scattered lean checkpoints (same run name, different timestamps).

    Returns dict: run_name_prefix → [list of lean_best dirs]
    """
    import re

    lean_dirs = find_lean_checkpoints(base_dir)
    groups = {}
    for ckpt_dir, run_name in lean_dirs:
        # Strip timestamp suffix: s18-qwen27b_..._ep2_20260316_192145 → s18-qwen27b_..._ep2
        prefix = re.sub(r"_\d{8}_\d{6}$", "", run_name)
        groups.setdefault(prefix, []).append(ckpt_dir)
    return {k: v for k, v in groups.items() if len(v) > 1}


def main():
    parser = argparse.ArgumentParser(description="Convert lean ZeRO-3 checkpoints to HF safetensors")
    parser.add_argument("input", nargs="*", help="Lean checkpoint dir(s)")
    parser.add_argument("--output", "-o", required=True, help="Output dir for HF model(s)")
    parser.add_argument("--all", action="store_true", help="Find and convert all lean_best/ dirs under input")
    parser.add_argument("--merge", action="store_true", help="Merge scattered partitions from multiple input dirs into one checkpoint")
    parser.add_argument("--recover", action="store_true", help="Auto-detect and recover scattered checkpoints under input dir")
    parser.add_argument("--keep", action="store_true", help="Keep checkpoint shards after conversion")
    parser.add_argument("--dry-run", action="store_true", help="List checkpoints without converting")
    parser.add_argument("--tag", default="best", help="Checkpoint tag (default: best)")
    args = parser.parse_args()

    if args.recover:
        # Auto-detect scattered checkpoints and merge+convert them
        if not args.input:
            parser.error("--recover requires an input directory")
        base = args.input[0]
        groups = find_scattered_groups(base)
        if not groups:
            log.info("No scattered checkpoints found")
            # Fall through to normal --all processing
            args.all = True
        else:
            log.info(f"Found {len(groups)} scattered checkpoint groups")
            for prefix, dirs in groups.items():
                shard_counts = []
                for d in dirs:
                    optim, _ = collect_shards(os.path.join(d, args.tag))
                    shard_counts.append(len(optim))
                total = sum(shard_counts)
                log.info(f"  {prefix}: {len(dirs)} dirs, {total} total shards {shard_counts}")
                out_dir = os.path.join(args.output, prefix)
                if args.dry_run:
                    log.info(f"    Would merge → {out_dir}")
                    continue
                # Merge directly into output dir, then convert in-place
                result = merge_scattered_checkpoints(dirs, out_dir, tag=args.tag)
                if result:
                    merged_dir, _ = result
                    convert(merged_dir, out_dir, tag=args.tag, keep=args.keep)
            return

    if args.merge:
        if len(args.input) < 2:
            parser.error("--merge requires at least 2 input directories")
        if args.dry_run:
            for d in args.input:
                optim, model = collect_shards(os.path.join(d, args.tag))
                log.info(f"  {d}: {len(optim)} optim, {len(model)} model shards")
            log.info(f"Would merge → {args.output}")
            return
        result = merge_scattered_checkpoints(args.input, args.output, tag=args.tag)
        if result:
            merged_dir, config_source = result
            convert(merged_dir, args.output, tag=args.tag, keep=args.keep)

    elif args.all:
        if not args.input:
            parser.error("--all requires an input directory")
        base = args.input[0]
        ckpts = find_lean_checkpoints(base)
        if not ckpts:
            log.info(f"No lean checkpoints found under {base}")
            return
        log.info(f"Found {len(ckpts)} lean checkpoints")
        for ckpt_dir, run_name in ckpts:
            out_dir = os.path.join(args.output, run_name)
            if args.dry_run:
                optim, _ = collect_shards(os.path.join(ckpt_dir, args.tag))
                log.info(f"  Would convert: {ckpt_dir} ({len(optim)} shards) → {out_dir}")
            else:
                try:
                    convert(ckpt_dir, out_dir, tag=args.tag, keep=args.keep)
                except Exception as e:
                    log.error(f"  Failed: {e}")
                    continue
    else:
        if not args.input:
            parser.error("Provide input directory, or use --all/--merge/--recover")
        if args.dry_run:
            log.info(f"Would convert: {args.input[0]} → {args.output}")
        else:
            convert(args.input[0], args.output, tag=args.tag, keep=args.keep)


if __name__ == "__main__":
    main()
