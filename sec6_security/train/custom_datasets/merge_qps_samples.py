#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def stable_hash(obj: dict[str, Any], keys: list[str] | None = None) -> str:
    """
    用于去重：优先用 keys 子集做hash；否则对整个json做hash（不保证不同字段顺序一致，所以先sort keys）。
    """
    if keys:
        base = {k: obj.get(k, None) for k in keys}
        s = json.dumps(base, ensure_ascii=False, sort_keys=True)
    else:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def parse_take_list(take_list: list[str]) -> dict[str, int]:
    """
    解析 --take "file=1000" "file2=300"
    """
    out = {}
    for item in take_list:
        if "=" not in item:
            raise ValueError(f"Bad --take item: {item} (expected path=N)")
        p, n = item.split("=", 1)
        p = p.strip()
        n = int(n.strip())
        out[p] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", nargs="+", required=True, help='Per-file take counts, e.g. "/path/qps2.jsonl=1000" "/path/qps3.jsonl=800"')
    ap.add_argument("--output", required=True, help="Output merged jsonl path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle final merged data")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate merged data")
    ap.add_argument("--dedup_keys", nargs="*", default=None, help="Keys used for dedup hash, e.g. --dedup_keys id problem question")
    ap.add_argument("--strict", action="store_true", help="If set, error when requested take > available, otherwise take all available.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    take_map = parse_take_list(args.take)

    merged: list[dict[str, Any]] = []
    for path, need in take_map.items():
        data = read_jsonl(path)
        avail = len(data)

        if need < 0:
            raise ValueError(f"take must be >=0, got {need} for {path}")

        if need > avail:
            if args.strict:
                raise ValueError(f"Requested {need} but only {avail} available in {path}")
            # 不严格：尽可能拿完
            chosen = data
            print(f"[WARN] {path}: requested {need}, available {avail}, taking {avail}")
        else:
            # 无放回抽样
            idx = list(range(avail))
            rng.shuffle(idx)
            chosen = [data[i] for i in idx[:need]]

        # 给每条加来源信息（可选但很推荐，方便追踪）
        base = os.path.basename(path)
        for obj in chosen:
            if isinstance(obj, dict):
                obj["_src_file"] = base
                obj["_src_path"] = path
        merged.extend(chosen)

        print(f"[OK] took {len(chosen)} / {avail} from {path}")

    if args.dedup:
        seen = set()
        uniq = []
        for obj in merged:
            h = stable_hash(obj, keys=args.dedup_keys)
            if h in seen:
                continue
            seen.add(h)
            uniq.append(obj)
        print(f"[DEDUP] {len(merged)} -> {len(uniq)}")
        merged = uniq

    if args.shuffle:
        rng.shuffle(merged)

    write_jsonl(args.output, merged)
    print(f"[DONE] merged -> {args.output}, total={len(merged)}")


if __name__ == "__main__":
    main()
