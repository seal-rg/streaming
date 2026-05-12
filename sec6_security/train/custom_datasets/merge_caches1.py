#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from glob import glob


def load_index(cache_dir):
    idx = os.path.join(cache_dir, "index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)
    return None


def list_npz(cache_dir, index_data):
    """
    Prefer index.json order; fallback to filename sort
    """
    if index_data and "sample_files" in index_data and index_data["sample_files"]:
        files = [os.path.join(cache_dir, x["file"]) for x in index_data["sample_files"]]
        return [f for f in files if os.path.exists(f)]
    return sorted(glob(os.path.join(cache_dir, "sample_*.npz")))


def merge_caches(cache_dirs, out_dir, seed=None, shuffle=True):
    os.makedirs(out_dir, exist_ok=True)

    if seed is not None:
        print(f"🔀 Using random seed = {seed}")
        random.seed(seed)

    # ----------- global collectors -----------
    all_samples = []  # (src_path, source_cache, source_file)
    merged_sources = []
    merged_filter_stats = {
        "total_processed": 0,
        "filtered_by_head_length": 0,
        "filtered_by_length_ratio": 0,
        "filtered_by_cv": 0,
        "filtered_by_padding": 0,
        "passed_filter": 0,
    }
    merged_config = None

    # ----------- collect phase -----------
    for order, cdir in enumerate(cache_dirs):
        if not os.path.isdir(cdir):
            print(f"⚠️  Skip missing dir: {cdir}")
            continue

        idx = load_index(cdir)
        npz_files = list_npz(cdir, idx)

        print(f"📁 Cache {order}: {cdir} | {len(npz_files)} samples")

        merged_sources.append(
            {
                "cache_dir": cdir,
                "order": order,
                "num_samples": len(npz_files),
            }
        )

        if idx and merged_config is None:
            merged_config = idx.get("config", None)

        if idx:
            fs = idx.get("filter_statistics", {})
            for k in merged_filter_stats:
                if k in fs:
                    merged_filter_stats[k] += fs[k]

        for src in npz_files:
            all_samples.append(
                {
                    "src": src,
                    "source_cache": os.path.basename(os.path.abspath(cdir)),
                    "source_file": os.path.basename(src),
                }
            )

    print(f"\n📦 Collected total samples: {len(all_samples)}")

    # ----------- GLOBAL SHUFFLE -----------
    if shuffle:
        random.shuffle(all_samples)
        print("🔀 Applied GLOBAL shuffle across all caches")
    else:
        print("➡️  Global shuffle disabled")

    # ----------- copy & reindex -----------
    merged_samples = []

    for global_idx, item in enumerate(all_samples):
        dst = os.path.join(out_dir, f"sample_{global_idx:06d}.npz")
        shutil.copy2(item["src"], dst)

        merged_samples.append(
            {
                "file": os.path.basename(dst),
                "sample_idx": global_idx,
                "source_cache": item["source_cache"],
                "source_file": item["source_file"],
            }
        )

    merged_index = {
        "version": "merge-1.2-global-shuffle",
        "total_samples": len(merged_samples),
        "sources": merged_sources,
        "config": merged_config,
        "filter_statistics": merged_filter_stats,
        "sample_files": merged_samples,
    }

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(merged_index, f, indent=2)

    print(f"\n✅ Merged & shuffled {len(merged_samples)} samples into: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output merged cache directory")
    ap.add_argument("--caches", nargs="+", required=True, help="Cache directories")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--no-shuffle", action="store_true", help="Disable global shuffle")
    args = ap.parse_args()

    merge_caches(
        cache_dirs=args.caches,
        out_dir=args.out,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()
