# # coding=utf-8
# """
# custom_datasets.py

# ✅ FINAL collator implementation (drop-in replacement) + user span export

# - Load cached per-sample tensors from .npz:
#    - input_ids: [S]
#    - labels: [S]
#    - position_ids: [S,2] or [S]
#    - attention_mask: [S,S] or [1,1,S,S]
#    - boundaries_json -> boundaries (dict)

# - Collate into batch with dynamic padding:
#    - input_ids: [B,Smax]
#    - labels: [B,Smax]
#    - position_ids: [B,Smax,2] or [B,Smax]
#    - attention_mask: additive mask [B,1,Smax,Smax] where visible->0, blocked->NEG_INF
#    - head_start/head_end/head_ok: [B,Hmax]  (Hmax = max num_heads in batch)
#    - ✅ user_start/user_end: [B] (user content span [start,end) in input_ids space)

# Notes:
# - NEG_INF = -1e4 (bf16/fp16/flash safe)
# - Robust detection additive vs visibility masks
# - Safe padding query rows: diagonal=0 for padded region
# """

# import json
# import os
# from typing import List, Dict, Any, Optional

# import numpy as np
# import torch
# from torch.utils.data import Dataset


# # ----------------------------
# # Dataset
# # ----------------------------
# class CustomMultiHeadDataset(Dataset):
#     """Dataset class for loading data from cached .npz files"""

#     def __init__(
#         self,
#         cache_dir: str,
#         max_seq_length: Optional[int] = None,
#         preload_to_memory: bool = False,
#     ):
#         self.cache_dir = cache_dir
#         self.max_seq_length = max_seq_length
#         self.preload_to_memory = preload_to_memory

#         with open(os.path.join(cache_dir, "index.json"), "r") as f:
#             self.index_data = json.load(f)

#         self.sample_files = self.index_data["sample_files"]
#         self.total_samples = len(self.sample_files)
#         self.config = self.index_data.get("config", {})

#         print(f"Loaded dataset from cache: {cache_dir}")
#         print(f"Total samples: {self.total_samples}")
#         print(f"Fixed heads: {self.config.get('fixed_heads', 'unknown')}")

#         self.memory_cache = {}
#         if self.preload_to_memory:
#             print("Preloading all samples to memory...")
#             for idx in range(self.total_samples):
#                 self.memory_cache[idx] = self._load_sample(idx)
#             print(f"Preloaded {len(self.memory_cache)} samples")

#     @staticmethod
#     def _decode_boundaries_json(bj: Any) -> str:
#         """Robust decode for boundaries_json stored in npz."""
#         if isinstance(bj, (bytes, np.bytes_)):
#             return bj.decode("utf-8")

#         try:
#             bj2 = bj.item()
#         except Exception:
#             bj2 = bj

#         if isinstance(bj2, (bytes, np.bytes_)):
#             return bj2.decode("utf-8")

#         return str(bj2)

#     def _load_sample(self, idx: int) -> Dict[str, Any]:
#         sample_info = self.sample_files[idx]
#         sample_path = os.path.join(self.cache_dir, sample_info["file"])

#         with np.load(sample_path) as data:
#             boundaries_json = self._decode_boundaries_json(data["boundaries_json"])
#             boundaries = json.loads(boundaries_json)

#             input_ids = torch.from_numpy(data["input_ids"].astype(np.int64))
#             labels = torch.from_numpy(data["labels"].astype(np.int64))
#             position_ids = torch.from_numpy(data["position_ids"].astype(np.int64))
#             attention_mask = torch.from_numpy(data["attention_mask"].astype(np.float32))

#             sample = {
#                 "input_ids": input_ids,                 # [S]
#                 "labels": labels,                       # [S]
#                 "position_ids": position_ids,           # [S,2] or [S]
#                 "attention_mask": attention_mask,       # [S,S] or [1,1,S,S]
#                 "num_heads": int(data["num_heads"]),
#                 "seq_length": int(data["seq_length"]),
#                 "boundaries": boundaries,               # dict
#             }

#         # Runtime truncation
#         if self.max_seq_length and sample["seq_length"] > self.max_seq_length:
#             L = int(self.max_seq_length)
#             sample["input_ids"] = sample["input_ids"][:L]
#             sample["labels"] = sample["labels"][:L]

#             if sample["position_ids"].dim() == 2:
#                 sample["position_ids"] = sample["position_ids"][:L, :]
#             else:
#                 sample["position_ids"] = sample["position_ids"][:L]

#             attn = sample["attention_mask"]
#             if attn.dim() == 4:
#                 sample["attention_mask"] = attn[:, :, :L, :L]
#             elif attn.dim() == 2:
#                 sample["attention_mask"] = attn[:L, :L]
#             else:
#                 raise ValueError(f"Unexpected attention_mask dim: {attn.dim()}")

#             sample["seq_length"] = L

#         return sample

#     def __len__(self):
#         return self.total_samples

#     def __getitem__(self, idx):
#         if idx in self.memory_cache:
#             return self.memory_cache[idx]
#         return self._load_sample(idx)


# # ----------------------------
# # Collator
# # ----------------------------
# class CustomDataCollator:
#     NEG_INF = -1e4

#     def __init__(
#         self,
#         pad_token_id: int,
#         assume_visibility_mask: bool = True,
#         additive_threshold: float = -1e3,
#         supervise_im_end: bool = True,
#     ):
#         self.pad_token_id = int(pad_token_id)
#         self.assume_visibility_mask = bool(assume_visibility_mask)
#         self.additive_threshold = float(additive_threshold)
#         self.supervise_im_end = bool(supervise_im_end)

#     def _normalize_mask_to_2d(self, attn: torch.Tensor) -> torch.Tensor:
#         if attn.dim() == 4:
#             if attn.size(0) == 1 and attn.size(1) == 1:
#                 attn = attn[0, 0]
#             else:
#                 raise ValueError(f"Expected single-sample 4D mask [1,1,S,S], got {tuple(attn.shape)}")
#         elif attn.dim() != 2:
#             raise ValueError(f"Unexpected attention_mask dim: {attn.dim()}")
#         return attn.to(torch.float32)

#     def _to_additive_mask(self, m2d: torch.Tensor) -> torch.Tensor:
#         m2d = m2d.to(torch.float32)

#         if not self.assume_visibility_mask:
#             return m2d

#         if m2d.numel() > 0 and float(m2d.min().item()) <= self.additive_threshold:
#             return m2d

#         visible = m2d > 0.5
#         return torch.where(visible, torch.zeros_like(m2d), torch.full_like(m2d, self.NEG_INF))

#     # -------------------------
#     # Extract per-head (start,end) spans from boundaries
#     # -------------------------
#     def _extract_head_spans(
#         self,
#         boundaries: Any,
#         num_heads: int,
#         S_cur: int,
#     ):
#         """
#         Returns:
#           head_start: [H] long
#           head_end:   [H] long
#           head_ok:    [H] bool

#         IMPORTANT:
#         - start/end are in input_ids space, consistent with your pipeline
#         """
#         H = int(num_heads)
#         head_start = torch.zeros((H,), dtype=torch.long)
#         head_end   = torch.zeros((H,), dtype=torch.long)
#         head_ok    = torch.zeros((H,), dtype=torch.bool)

#         if not isinstance(boundaries, dict):
#             return head_start, head_end, head_ok

#         all_heads = boundaries.get("all_heads", [])
#         ahidx = boundaries.get("assistant_head_indices", [])
#         if not isinstance(all_heads, list) or not isinstance(ahidx, list):
#             return head_start, head_end, head_ok

#         for head_idx in range(H):
#             if head_idx >= len(ahidx):
#                 continue
#             try:
#                 hid = int(ahidx[head_idx])
#             except Exception:
#                 continue
#             if hid < 0 or hid >= len(all_heads):
#                 continue
#             h = all_heads[hid]
#             if not isinstance(h, dict):
#                 continue

#             start = int(h.get("assistant_solution_start", h.get("content_start", -1)))
#             end   = int(h.get("real_content_end", h.get("content_end", -1)))

#             start = max(1, start)
#             end = min(max(end, 0), S_cur)

#             if self.supervise_im_end:
#                 ie = int(h.get("im_end_pos", -1))
#                 if 0 <= ie < S_cur:
#                     end = min(max(end, ie + 1), S_cur)

#             if end <= start:
#                 continue

#             head_start[head_idx] = start
#             head_end[head_idx] = end
#             head_ok[head_idx] = True

#         return head_start, head_end, head_ok

#     # -------------------------
#     # ✅ Extract user content span from boundaries
#     # -------------------------
#     def _extract_user_span(self, boundaries: Any, S_cur: int):
#         """
#         Returns: (user_start, user_end) in input_ids space, [start,end)
#         user_start/end refer to USER CONTENT only (not prefix tokens).
#         """
#         if not isinstance(boundaries, dict):
#             return 0, 0
#         all_heads = boundaries.get("all_heads", [])
#         if not isinstance(all_heads, list):
#             return 0, 0

#         for h in all_heads:
#             if isinstance(h, dict) and h.get("role") == "user":
#                 st = int(h.get("content_start", 0))
#                 ed = int(h.get("content_end", 0))
#                 st = max(0, min(st, S_cur))
#                 ed = max(0, min(ed, S_cur))
#                 if ed > st:
#                     return st, ed
#                 return 0, 0
#         return 0, 0

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         B = len(features)
#         batch_max_length = max(int(f["seq_length"]) for f in features)
#         batch_max_heads = max(int(f["num_heads"]) for f in features)

#         batch_input_ids = []
#         batch_labels = []
#         batch_position_ids = []
#         batch_attention_mask = []
#         batch_num_heads = []
#         batch_boundaries = []

#         batch_head_start = []
#         batch_head_end = []
#         batch_head_ok = []

#         # ✅ user spans
#         batch_user_start = []
#         batch_user_end = []

#         for f in features:
#             cur_len = int(f["seq_length"])
#             pad_len = batch_max_length - cur_len

#             # ---- input_ids / labels ----
#             input_ids = f["input_ids"]
#             labels = f["labels"]

#             if pad_len > 0:
#                 input_ids = torch.cat(
#                     [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)],
#                     dim=0,
#                 )
#                 labels = torch.cat(
#                     [labels, torch.full((pad_len,), -100, dtype=torch.long)],
#                     dim=0,
#                 )

#             # ---- position_ids ----
#             pos = f["position_ids"]
#             if pad_len > 0:
#                 if pos.dim() == 2:
#                     pad_pos = torch.zeros((pad_len, pos.size(1)), dtype=torch.long)
#                     if cur_len > 0:
#                         last_c = int(pos[cur_len - 1, 0].item())
#                         max_y = int(pos[:cur_len, 1].max().item())
#                     else:
#                         last_c, max_y = 0, 0
#                     pad_pos[:, 0] = last_c
#                     pad_pos[:, 1] = max_y + 1
#                     pos = torch.cat([pos, pad_pos], dim=0)
#                 elif pos.dim() == 1:
#                     pad_pos = torch.zeros((pad_len,), dtype=torch.long)
#                     pos = torch.cat([pos, pad_pos], dim=0)
#                 else:
#                     raise ValueError(f"Unexpected position_ids dim: {pos.dim()}")

#             # ---- attention_mask ----
#             attn = self._normalize_mask_to_2d(f["attention_mask"])  # [cur_len,cur_len]
#             attn_add = self._to_additive_mask(attn)                 # additive [cur_len,cur_len]

#             if pad_len > 0:
#                 padded = torch.full(
#                     (batch_max_length, batch_max_length),
#                     self.NEG_INF,
#                     dtype=torch.float32,
#                 )
#                 padded[:cur_len, :cur_len] = attn_add

#                 # safe padded query rows: set diagonal to 0.0 so no all-NEG_INF row
#                 idx = torch.arange(cur_len, batch_max_length)
#                 padded[idx, idx] = 0.0
#             else:
#                 padded = attn_add

#             batch_attention_mask.append(padded.unsqueeze(0).unsqueeze(0))  # [1,1,S,S]

#             # ---- boundaries + head spans ----
#             boundaries = f["boundaries"]
#             nh = int(f["num_heads"])
#             hs, he, hok = self._extract_head_spans(boundaries, nh, cur_len)

#             # ✅ user span (content only)
#             u_st, u_ed = self._extract_user_span(boundaries, cur_len)

#             # pad spans to batch_max_heads
#             if batch_max_heads > nh:
#                 pad_h = batch_max_heads - nh
#                 hs = torch.cat([hs, torch.zeros((pad_h,), dtype=torch.long)], dim=0)
#                 he = torch.cat([he, torch.zeros((pad_h,), dtype=torch.long)], dim=0)
#                 hok = torch.cat([hok, torch.zeros((pad_h,), dtype=torch.bool)], dim=0)

#             batch_head_start.append(hs)  # [Hmax]
#             batch_head_end.append(he)    # [Hmax]
#             batch_head_ok.append(hok)    # [Hmax]

#             batch_user_start.append(torch.tensor(u_st, dtype=torch.long))
#             batch_user_end.append(torch.tensor(u_ed, dtype=torch.long))

#             batch_input_ids.append(input_ids)
#             batch_labels.append(labels)
#             batch_position_ids.append(pos)
#             batch_num_heads.append(nh)
#             batch_boundaries.append(boundaries)

#         out = {
#             "input_ids": torch.stack(batch_input_ids, dim=0),          # [B,S]
#             "labels": torch.stack(batch_labels, dim=0),                # [B,S]
#             "position_ids": torch.stack(batch_position_ids, dim=0),    # [B,S,2] or [B,S]
#             "attention_mask": torch.cat(batch_attention_mask, dim=0),  # [B,1,S,S] additive
#             "num_heads": torch.tensor(batch_num_heads, dtype=torch.long),  # [B]
#             "boundaries": batch_boundaries,  # list[dict]
#             "head_start": torch.stack(batch_head_start, dim=0),        # [B,Hmax]
#             "head_end": torch.stack(batch_head_end, dim=0),            # [B,Hmax]
#             "head_ok": torch.stack(batch_head_ok, dim=0),              # [B,Hmax]
#             "user_start": torch.stack(batch_user_start, dim=0),        # [B]
#             "user_end": torch.stack(batch_user_end, dim=0),            # [B]
#         }
#         return out


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
custom_datasets.py

Dataset + collator (drop-in) with SAFE index-time filtering:
- Keep _load_sample() logic unchanged (still reads data["boundaries_json"]).
- Filter samples at __init__ so training never indexes bad npz:
    * missing boundaries_json
    * boundaries_json parses to empty dict {}

Filtering uses zipfile to avoid decompressing huge arrays (no OOM).
Optionally writes filtered_index.json.

Expected .npz members:
  - input_ids.npy, labels.npy, position_ids.npy, attention_mask.npy
  - num_heads.npy, seq_length.npy
  - boundaries_json.npy  (required; must parse to non-empty dict)
"""

import json
import os
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------
# Index-time lightweight checks (NO big array decompression)
# ----------------------------
def _decode_boundaries_json_any(bj: Any) -> str:
    if isinstance(bj, (bytes, np.bytes_)):
        return bj.decode("utf-8", errors="replace")
    try:
        bj2 = bj.item()
    except Exception:
        bj2 = bj
    if isinstance(bj2, (bytes, np.bytes_)):
        return bj2.decode("utf-8", errors="replace")
    return str(bj2)


def _check_npz_has_nonempty_boundaries_json(npz_path: str) -> Tuple[bool, str]:
    """
    Returns (ok, reason). Uses zipfile to avoid loading huge arrays.
    Conditions:
      - boundaries_json.npy exists
      - json.loads(boundaries_json) returns dict
      - dict is non-empty
    """
    import zipfile

    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            names = set(zf.namelist())
            if "boundaries_json.npy" not in names:
                return False, "missing_boundaries_json"

            # Only decompress this small member
            with zf.open("boundaries_json.npy") as f:
                buf = BytesIO(f.read())
            bj = np.load(buf, allow_pickle=True)

            s = _decode_boundaries_json_any(bj)
            try:
                boundaries = json.loads(s)
            except Exception as e:
                return False, f"boundary_json_invalid:{type(e).__name__}"

            if not isinstance(boundaries, dict):
                return False, "boundary_not_dict"
            if len(boundaries) == 0:
                return False, "empty_boundaries_dict"

            return True, "ok"

    except Exception as e:
        return False, f"npz_zip_read_error:{type(e).__name__}"


# ----------------------------
# Dataset
# ----------------------------
class CustomMultiHeadDataset(Dataset):
    """Dataset class for loading data from cached .npz files (with safe pre-filtering)."""

    def __init__(
        self,
        cache_dir: str,
        max_seq_length: Optional[int] = None,
        preload_to_memory: bool = False,
        # ✅ new knobs (do not change _load_sample logic)
        filter_missing_boundaries: bool = True,
        write_filtered_index: bool = False,
        filtered_index_name: str = "filtered_index.json",
        verbose_filter: bool = True,
    ):
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.preload_to_memory = preload_to_memory

        index_path = os.path.join(cache_dir, "index.json")
        with open(index_path, "r") as f:
            self.index_data = json.load(f)

        self.sample_files = self.index_data["sample_files"]
        self.config = self.index_data.get("config", {})

        # ✅ Filter at init so __getitem__ never hits bad npz
        self._filter_report = None
        if filter_missing_boundaries:
            kept = []
            bad = []
            for s in self.sample_files:
                fn = s.get("file")
                if not fn:
                    bad.append({**s, "reason": "missing_file_field"})
                    continue
                npz_path = os.path.join(self.cache_dir, fn)
                if not os.path.exists(npz_path):
                    bad.append({**s, "reason": "missing_npz_file"})
                    continue

                ok, reason = _check_npz_has_nonempty_boundaries_json(npz_path)
                if ok:
                    kept.append(s)
                else:
                    bad.append({**s, "reason": reason})

            self.sample_files = kept
            self._filter_report = {
                "cache_dir": cache_dir,
                "original_total_samples": int(self.index_data.get("total_samples", len(self.index_data.get("sample_files", [])))),
                "kept": len(kept),
                "dropped": len(bad),
                "bad_samples": bad[:200],  # cap to avoid huge prints
            }

            if verbose_filter:
                # Count reasons
                from collections import Counter
                rc = Counter([b.get("reason", "unknown") for b in bad])
                print(f"[cache filter] kept={len(kept)} dropped={len(bad)} (missing/empty boundaries_json removed)")
                if len(bad) > 0:
                    print("[cache filter] top reasons:")
                    for k, v in sorted(rc.items(), key=lambda x: (-x[1], x[0]))[:20]:
                        print(f"  {k}: {v}")

            if write_filtered_index:
                out = dict(self.index_data)
                out["sample_files"] = self.sample_files
                out["total_samples"] = len(self.sample_files)
                out_path = os.path.join(self.cache_dir, filtered_index_name)
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                if verbose_filter:
                    print(f"[cache filter] wrote {out_path}")

        self.total_samples = len(self.sample_files)

        print(f"Loaded dataset from cache: {cache_dir}")
        print(f"Total samples: {self.total_samples}")
        print(f"Fixed heads: {self.config.get('fixed_heads', 'unknown')}")

        self.memory_cache = {}
        if self.preload_to_memory:
            # NOTE: preloading will load big tensors and can OOM; only use if you are sure.
            print("Preloading all samples to memory...")
            for idx in range(self.total_samples):
                self.memory_cache[idx] = self._load_sample(idx)
            print(f"Preloaded {len(self.memory_cache)} samples")

    @staticmethod
    def _decode_boundaries_json(bj: Any) -> str:
        """Robust decode for boundaries_json stored in npz."""
        if isinstance(bj, (bytes, np.bytes_)):
            return bj.decode("utf-8")

        try:
            bj2 = bj.item()
        except Exception:
            bj2 = bj

        if isinstance(bj2, (bytes, np.bytes_)):
            return bj2.decode("utf-8")

        return str(bj2)

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """
        ✅ DO NOT CHANGE logic per your requirement.
        """
        sample_info = self.sample_files[idx]
        sample_path = os.path.join(self.cache_dir, sample_info["file"])

        with np.load(sample_path) as data:
            boundaries_json = self._decode_boundaries_json(data["boundaries_json"])
            boundaries = json.loads(boundaries_json)

            input_ids = torch.from_numpy(data["input_ids"].astype(np.int64))
            labels = torch.from_numpy(data["labels"].astype(np.int64))
            position_ids = torch.from_numpy(data["position_ids"].astype(np.int64))
            attention_mask = torch.from_numpy(data["attention_mask"].astype(np.float32))

            sample = {
                "input_ids": input_ids,                 # [S]
                "labels": labels,                       # [S]
                "position_ids": position_ids,           # [S,2] or [S]
                "attention_mask": attention_mask,       # [S,S] or [1,1,S,S]
                "num_heads": int(data["num_heads"]),
                "seq_length": int(data["seq_length"]),
                "boundaries": boundaries,               # dict
            }

        # Runtime truncation
        if self.max_seq_length and sample["seq_length"] > self.max_seq_length:
            L = int(self.max_seq_length)
            sample["input_ids"] = sample["input_ids"][:L]
            sample["labels"] = sample["labels"][:L]

            if sample["position_ids"].dim() == 2:
                sample["position_ids"] = sample["position_ids"][:L, :]
            else:
                sample["position_ids"] = sample["position_ids"][:L]

            attn = sample["attention_mask"]
            if attn.dim() == 4:
                sample["attention_mask"] = attn[:, :, :L, :L]
            elif attn.dim() == 2:
                sample["attention_mask"] = attn[:L, :L]
            else:
                raise ValueError(f"Unexpected attention_mask dim: {attn.dim()}")

            sample["seq_length"] = L

        return sample

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx in self.memory_cache:
            return self.memory_cache[idx]
        return self._load_sample(idx)


# ----------------------------
# Collator
# ----------------------------
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

    def _normalize_mask_to_2d(self, attn: torch.Tensor) -> torch.Tensor:
        if attn.dim() == 4:
            if attn.size(0) == 1 and attn.size(1) == 1:
                attn = attn[0, 0]
            else:
                raise ValueError(f"Expected single-sample 4D mask [1,1,S,S], got {tuple(attn.shape)}")
        elif attn.dim() != 2:
            raise ValueError(f"Unexpected attention_mask dim: {attn.dim()}")
        return attn.to(torch.float32)

    def _to_additive_mask(self, m2d: torch.Tensor) -> torch.Tensor:
        m2d = m2d.to(torch.float32)

        if not self.assume_visibility_mask:
            return m2d

        if m2d.numel() > 0 and float(m2d.min().item()) <= self.additive_threshold:
            return m2d

        visible = m2d > 0.5
        return torch.where(visible, torch.zeros_like(m2d), torch.full_like(m2d, self.NEG_INF))

    def _extract_head_spans(self, boundaries: Any, num_heads: int, S_cur: int):
        H = int(num_heads)
        head_start = torch.zeros((H,), dtype=torch.long)
        head_end   = torch.zeros((H,), dtype=torch.long)
        head_ok    = torch.zeros((H,), dtype=torch.bool)

        if not isinstance(boundaries, dict):
            return head_start, head_end, head_ok

        all_heads = boundaries.get("all_heads", [])
        ahidx = boundaries.get("assistant_head_indices", [])
        if not isinstance(all_heads, list) or not isinstance(ahidx, list):
            return head_start, head_end, head_ok

        for head_idx in range(H):
            if head_idx >= len(ahidx):
                continue
            try:
                hid = int(ahidx[head_idx])
            except Exception:
                continue
            if hid < 0 or hid >= len(all_heads):
                continue
            h = all_heads[hid]
            if not isinstance(h, dict):
                continue

            start = int(h.get("assistant_solution_start", h.get("content_start", -1)))
            end   = int(h.get("real_content_end", h.get("content_end", -1)))

            start = max(1, start)
            end = min(max(end, 0), S_cur)

            if self.supervise_im_end:
                ie = int(h.get("im_end_pos", -1))
                if 0 <= ie < S_cur:
                    end = min(max(end, ie + 1), S_cur)

            if end <= start:
                continue

            head_start[head_idx] = start
            head_end[head_idx] = end
            head_ok[head_idx] = True

        return head_start, head_end, head_ok

    def _extract_user_span(self, boundaries: Any, S_cur: int):
        if not isinstance(boundaries, dict):
            return 0, 0
        all_heads = boundaries.get("all_heads", [])
        if not isinstance(all_heads, list):
            return 0, 0

        for h in all_heads:
            if isinstance(h, dict) and h.get("role") == "user":
                st = int(h.get("content_start", 0))
                ed = int(h.get("content_end", 0))
                st = max(0, min(st, S_cur))
                ed = max(0, min(ed, S_cur))
                if ed > st:
                    return st, ed
                return 0, 0
        return 0, 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(features)
        batch_max_length = max(int(f["seq_length"]) for f in features)
        batch_max_heads = max(int(f["num_heads"]) for f in features)

        batch_input_ids = []
        batch_labels = []
        batch_position_ids = []
        batch_attention_mask = []
        batch_num_heads = []
        batch_boundaries = []

        batch_head_start = []
        batch_head_end = []
        batch_head_ok = []

        batch_user_start = []
        batch_user_end = []

        for f in features:
            cur_len = int(f["seq_length"])
            pad_len = batch_max_length - cur_len

            input_ids = f["input_ids"]
            labels = f["labels"]

            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)],
                    dim=0,
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=torch.long)],
                    dim=0,
                )

            pos = f["position_ids"]
            if pad_len > 0:
                if pos.dim() == 2:
                    pad_pos = torch.zeros((pad_len, pos.size(1)), dtype=torch.long)
                    if cur_len > 0:
                        last_c = int(pos[cur_len - 1, 0].item())
                        max_y = int(pos[:cur_len, 1].max().item())
                    else:
                        last_c, max_y = 0, 0
                    pad_pos[:, 0] = last_c
                    pad_pos[:, 1] = max_y + 1
                    pos = torch.cat([pos, pad_pos], dim=0)
                elif pos.dim() == 1:
                    pad_pos = torch.zeros((pad_len,), dtype=torch.long)
                    pos = torch.cat([pos, pad_pos], dim=0)
                else:
                    raise ValueError(f"Unexpected position_ids dim: {pos.dim()}")

            attn = self._normalize_mask_to_2d(f["attention_mask"])
            attn_add = self._to_additive_mask(attn)

            if pad_len > 0:
                padded = torch.full(
                    (batch_max_length, batch_max_length),
                    self.NEG_INF,
                    dtype=torch.float32,
                )
                padded[:cur_len, :cur_len] = attn_add
                idx = torch.arange(cur_len, batch_max_length)
                padded[idx, idx] = 0.0
            else:
                padded = attn_add

            batch_attention_mask.append(padded.unsqueeze(0).unsqueeze(0))

            boundaries = f["boundaries"]
            nh = int(f["num_heads"])
            hs, he, hok = self._extract_head_spans(boundaries, nh, cur_len)

            u_st, u_ed = self._extract_user_span(boundaries, cur_len)

            if batch_max_heads > nh:
                pad_h = batch_max_heads - nh
                hs = torch.cat([hs, torch.zeros((pad_h,), dtype=torch.long)], dim=0)
                he = torch.cat([he, torch.zeros((pad_h,), dtype=torch.long)], dim=0)
                hok = torch.cat([hok, torch.zeros((pad_h,), dtype=torch.bool)], dim=0)

            batch_head_start.append(hs)
            batch_head_end.append(he)
            batch_head_ok.append(hok)

            batch_user_start.append(torch.tensor(u_st, dtype=torch.long))
            batch_user_end.append(torch.tensor(u_ed, dtype=torch.long))

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_position_ids.append(pos)
            batch_num_heads.append(nh)
            batch_boundaries.append(boundaries)

        out = {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
            "position_ids": torch.stack(batch_position_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "num_heads": torch.tensor(batch_num_heads, dtype=torch.long),
            "boundaries": batch_boundaries,
            "head_start": torch.stack(batch_head_start, dim=0),
            "head_end": torch.stack(batch_head_end, dim=0),
            "head_ok": torch.stack(batch_head_ok, dim=0),
            "user_start": torch.stack(batch_user_start, dim=0),
            "user_end": torch.stack(batch_user_end, dim=0),
        }
        return out