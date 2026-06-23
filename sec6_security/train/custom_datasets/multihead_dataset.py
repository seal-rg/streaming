"""Dataset and collator for multi-head multi-stream training.

Loads per-sample .npz files from a cache directory. Each .npz contains:
  input_ids, labels, position_ids, attention_mask, num_heads, seq_length,
  boundaries_json (JSON-encoded span metadata).

Samples with missing or empty boundaries_json are filtered out at init time
using zipfile inspection (no full array decompression) to avoid OOM.
"""

import json
import os
from io import BytesIO
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Lightweight .npz validation (zipfile, no array decompression) ─────────────


def _decode_bytes(bj: Any) -> str:
    if isinstance(bj, (bytes, np.bytes_)):
        return bj.decode("utf-8", errors="replace")
    try:
        bj2 = bj.item()
    except Exception:
        bj2 = bj
    if isinstance(bj2, (bytes, np.bytes_)):
        return bj2.decode("utf-8", errors="replace")
    return str(bj2)


def _npz_has_valid_boundaries(npz_path: str) -> tuple[bool, str]:
    """Check boundaries_json exists and parses to a non-empty dict."""
    import zipfile
    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            if "boundaries_json.npy" not in set(zf.namelist()):
                return False, "missing_boundaries_json"
            with zf.open("boundaries_json.npy") as f:
                bj = np.load(BytesIO(f.read()), allow_pickle=True)
            try:
                boundaries = json.loads(_decode_bytes(bj))
            except Exception as e:
                return False, f"json_parse_error:{type(e).__name__}"
            if not isinstance(boundaries, dict) or not boundaries:
                return False, "empty_or_non_dict_boundaries"
            return True, "ok"
    except Exception as e:
        return False, f"zip_read_error:{type(e).__name__}"


# ── Dataset ───────────────────────────────────────────────────────────────────


class CustomMultiHeadDataset(Dataset):
    """Loads multi-head training samples from a cached .npz directory."""

    def __init__(
        self,
        cache_dir: str,
        max_seq_length: int | None = None,
        preload_to_memory: bool = False,
        filter_missing_boundaries: bool = True,
        write_filtered_index: bool = False,
        filtered_index_name: str = "filtered_index.json",
    ):
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.preload_to_memory = preload_to_memory

        with open(os.path.join(cache_dir, "index.json")) as f:
            self.index_data = json.load(f)

        self.sample_files = self.index_data["sample_files"]
        self.config = self.index_data.get("config", {})

        if filter_missing_boundaries:
            kept, bad = [], []
            for s in self.sample_files:
                fn = s.get("file")
                if not fn:
                    bad.append({**s, "reason": "missing_file_field"}); continue
                npz_path = os.path.join(self.cache_dir, fn)
                if not os.path.exists(npz_path):
                    bad.append({**s, "reason": "missing_npz_file"}); continue
                ok, reason = _npz_has_valid_boundaries(npz_path)
                (kept if ok else bad).append({**s, "reason": reason} if not ok else s)

            if bad:
                from collections import Counter
                rc = Counter(b.get("reason", "unknown") for b in bad)
                print(f"[dataset filter] kept={len(kept)} dropped={len(bad)}")
                for k, v in sorted(rc.items(), key=lambda x: -x[1])[:10]:
                    print(f"  {k}: {v}")

            self.sample_files = kept

            if write_filtered_index:
                out = {**self.index_data, "sample_files": kept, "total_samples": len(kept)}
                out_path = os.path.join(self.cache_dir, filtered_index_name)
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)

        self.total_samples = len(self.sample_files)
        print(f"Dataset: {cache_dir}  samples={self.total_samples}  heads={self.config.get('fixed_heads', '?')}")

        self.memory_cache: dict = {}
        if preload_to_memory:
            for idx in range(self.total_samples):
                self.memory_cache[idx] = self._load_sample(idx)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        info = self.sample_files[idx]
        path = os.path.join(self.cache_dir, info["file"])

        with np.load(path) as data:
            boundaries = json.loads(_decode_bytes(data["boundaries_json"]))
            sample = {
                "input_ids":      torch.from_numpy(data["input_ids"].astype(np.int64)),
                "labels":         torch.from_numpy(data["labels"].astype(np.int64)),
                "position_ids":   torch.from_numpy(data["position_ids"].astype(np.int64)),
                "attention_mask": torch.from_numpy(data["attention_mask"].astype(np.float32)),
                "num_heads":  int(data["num_heads"]),
                "seq_length": int(data["seq_length"]),
                "boundaries": boundaries,
            }

        if self.max_seq_length and sample["seq_length"] > self.max_seq_length:
            L = int(self.max_seq_length)
            sample["input_ids"]    = sample["input_ids"][:L]
            sample["labels"]       = sample["labels"][:L]
            sample["position_ids"] = (sample["position_ids"][:L, :] if sample["position_ids"].dim() == 2
                                      else sample["position_ids"][:L])
            attn = sample["attention_mask"]
            sample["attention_mask"] = (attn[:, :, :L, :L] if attn.dim() == 4 else attn[:L, :L])
            sample["seq_length"] = L

        return sample

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx in self.memory_cache:
            return self.memory_cache[idx]
        return self._load_sample(idx)


# ── Collator ──────────────────────────────────────────────────────────────────


class CustomDataCollator:
    NEG_INF = -1e4

    def __init__(
        self,
        pad_token_id: int,
        assume_visibility_mask: bool = True,
        additive_threshold: float = -1e3,
        supervise_im_end: bool = True,
    ):
        self.pad_token_id = int(pad_token_id)
        self.assume_visibility_mask = bool(assume_visibility_mask)
        self.additive_threshold = float(additive_threshold)
        self.supervise_im_end = bool(supervise_im_end)

    def _to_additive_mask(self, attn: torch.Tensor) -> torch.Tensor:
        if attn.dim() == 4:
            if attn.size(0) == 1 and attn.size(1) == 1:
                attn = attn[0, 0]
            else:
                raise ValueError(f"Expected [1,1,S,S] mask, got {tuple(attn.shape)}")
        elif attn.dim() != 2:
            raise ValueError(f"Unexpected attention_mask dim: {attn.dim()}")
        attn = attn.to(torch.float32)
        if not self.assume_visibility_mask:
            return attn
        if attn.numel() > 0 and float(attn.min()) <= self.additive_threshold:
            return attn
        # convert boolean visibility mask → additive
        return torch.where(attn > 0.5, torch.zeros_like(attn), torch.full_like(attn, self.NEG_INF))

    def _extract_head_spans(self, boundaries: Any, num_heads: int, S: int):
        H = int(num_heads)
        hs  = torch.zeros((H,), dtype=torch.long)
        he  = torch.zeros((H,), dtype=torch.long)
        hok = torch.zeros((H,), dtype=torch.bool)

        if not isinstance(boundaries, dict):
            return hs, he, hok

        all_heads = boundaries.get("all_heads", [])
        ahidx     = boundaries.get("assistant_head_indices", [])
        if not isinstance(all_heads, list) or not isinstance(ahidx, list):
            return hs, he, hok

        for hi in range(H):
            if hi >= len(ahidx):
                continue
            try:
                hid = int(ahidx[hi])
            except Exception:
                continue
            if not (0 <= hid < len(all_heads)):
                continue
            h = all_heads[hid]
            if not isinstance(h, dict):
                continue

            start = int(h.get("assistant_solution_start", h.get("content_start", -1)))
            end   = int(h.get("real_content_end", h.get("content_end", -1)))
            start = max(1, start)
            end   = min(max(end, 0), S)

            if self.supervise_im_end:
                ie = int(h.get("im_end_pos", -1))
                if 0 <= ie < S:
                    end = min(max(end, ie + 1), S)

            if end > start:
                hs[hi] = start; he[hi] = end; hok[hi] = True

        return hs, he, hok

    def _extract_user_span(self, boundaries: Any, S: int) -> tuple[int, int]:
        if not isinstance(boundaries, dict):
            return 0, 0
        for h in boundaries.get("all_heads", []):
            if isinstance(h, dict) and h.get("role") == "user":
                st = max(0, min(int(h.get("content_start", 0)), S))
                ed = max(0, min(int(h.get("content_end",   0)), S))
                return (st, ed) if ed > st else (0, 0)
        return 0, 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len   = max(int(f["seq_length"]) for f in features)
        max_heads = max(int(f["num_heads"])   for f in features)

        batch_ids, batch_lbl, batch_pos, batch_attn = [], [], [], []
        batch_hs, batch_he, batch_hok = [], [], []
        batch_ust, batch_ued = [], []
        batch_nh, batch_bd = [], []

        for f in features:
            L   = int(f["seq_length"])
            pad = max_len - L

            ids = f["input_ids"]
            lbl = f["labels"]
            if pad:
                ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=torch.long)])
                lbl = torch.cat([lbl, torch.full((pad,), -100,              dtype=torch.long)])

            pos = f["position_ids"]
            if pad:
                if pos.dim() == 2:
                    last_c = int(pos[L-1, 0]) if L > 0 else 0
                    max_y  = int(pos[:L, 1].max()) if L > 0 else 0
                    pad_p  = torch.zeros((pad, pos.size(1)), dtype=torch.long)
                    pad_p[:, 0] = last_c; pad_p[:, 1] = max_y + 1
                else:
                    pad_p = torch.zeros((pad,), dtype=torch.long)
                pos = torch.cat([pos, pad_p])

            attn = self._to_additive_mask(f["attention_mask"])
            if pad:
                padded = torch.full((max_len, max_len), self.NEG_INF, dtype=torch.float32)
                padded[:L, :L] = attn
                diag = torch.arange(L, max_len)
                padded[diag, diag] = 0.0
                attn = padded

            nh = int(f["num_heads"])
            bd = f["boundaries"]
            hs, he, hok = self._extract_head_spans(bd, nh, L)
            u_st, u_ed  = self._extract_user_span(bd, L)

            if max_heads > nh:
                ph = max_heads - nh
                hs  = torch.cat([hs,  torch.zeros((ph,), dtype=torch.long)])
                he  = torch.cat([he,  torch.zeros((ph,), dtype=torch.long)])
                hok = torch.cat([hok, torch.zeros((ph,), dtype=torch.bool)])

            batch_ids.append(ids);  batch_lbl.append(lbl);  batch_pos.append(pos)
            batch_attn.append(attn.unsqueeze(0).unsqueeze(0))
            batch_hs.append(hs);    batch_he.append(he);    batch_hok.append(hok)
            batch_ust.append(torch.tensor(u_st, dtype=torch.long))
            batch_ued.append(torch.tensor(u_ed, dtype=torch.long))
            batch_nh.append(nh);    batch_bd.append(bd)

        return {
            "input_ids":      torch.stack(batch_ids),
            "labels":         torch.stack(batch_lbl),
            "position_ids":   torch.stack(batch_pos),
            "attention_mask": torch.cat(batch_attn),
            "num_heads":      torch.tensor(batch_nh, dtype=torch.long),
            "boundaries":     batch_bd,
            "head_start":     torch.stack(batch_hs),
            "head_end":       torch.stack(batch_he),
            "head_ok":        torch.stack(batch_hok),
            "user_start":     torch.stack(batch_ust),
            "user_end":       torch.stack(batch_ued),
        }
