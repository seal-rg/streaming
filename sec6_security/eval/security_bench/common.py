"""Shared utilities: seeding, string cleanup, mean/std aggregation."""
import os
import random
import re
from typing import List, Protocol, Tuple, Any, Dict

import numpy as np
import torch
import transformers


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        transformers.set_seed(seed)
    except Exception:
        pass


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


_SPECIAL_TOKEN_PAT = re.compile(r"<\|[^>]+?\|>")  # <|im_end|>, <|eot_id|>, ...


def strip_special_tokens_text(s: str) -> str:
    if not s:
        return ""
    return _SPECIAL_TOKEN_PAT.sub("", s).strip()


def mean_std_pct(xs: List[float]) -> Tuple[float, float]:
    """Population mean/std (ddof=0), returned as percentages."""
    if not xs:
        return 0.0, 0.0
    arr = np.array(xs, dtype=float)
    return float(arr.mean() * 100.0), float(arr.std() * 100.0)


class InferenceBackend(Protocol):
    """Interface every model backend (Vanilla, Stream, ...) must expose."""

    tokenizer: Any

    def generate_1head_system_user(
        self,
        system_text: str,
        user_text: str,
        final_prefix: str = "",
        max_length_per_head: int = 256,
        max_steps: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        do_sample: bool = False,
        stop_on_im_end: bool = True,
    ) -> Dict[str, Any]:
        """Run one system+user prompt through the model, return
        {"text_raw": <decoded text incl. special tokens>, "n_new_tokens": int}."""
        ...
