#!/usr/bin/env python3

"""
Medusa Model Inference Script (TRAIN-ALIGNED to SingleFileDatasetProcessor)

Key change you asked:
- inference prompt: assistant blocks DO NOT contain <|im_end|>
- prompt contains: assistant_prefix + intro+wait (optional)
- generation stops when generated token == <|im_end|>

system:
  <|im_start|>system\n{system_message}<|im_end|>         (NO newline after im_end)

user:
  <|im_start|>user\n{question}<|im_end|>                (NO newline after im_end)

assistant heads (num_heads times):
  <|im_start|>assistant\n + (intro + <|wait|>*K)         (NO im_end here)

where:
  base_align = max(len(sys_content_tokens), len(user_content_tokens))
  intro_i    = "I am channel C{i}. "  (with trailing space)
  K_i        = max(0, base_align - len(intro_i_tokens))
"""

import argparse
import os
import sys
import warnings
from typing import Any

import torch
import transformers

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

# make sure your local package path is included
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_medusa import Qwen2ForMedusa


def _as_list(x: torch.Tensor) -> list[int]:
    return [int(t) for t in x.detach().cpu().tolist()]


class MedusaInference:
    def __init__(self, model_path: str, device: str = "auto", torch_dtype: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.dtype = self._get_dtype(torch_dtype)

        self.system_message = "You are a helpful assistant."
        self.wait_token_str = "<|wait|>"
        self.channel_intro_template = "I am channel C{idx}."
        self.add_channel_intro_and_wait = True
        self.include_system_user_im_end = True

        print("[Init] Model Path:", model_path)
        print("[Init] Device:", self.device)
        print("[Init] DType:", self.dtype)

        self.model: Qwen2ForMedusa | None = None
        self.tokenizer: transformers.PreTrainedTokenizerBase | None = None

        self._load_model_and_tokenizer()
        self._setup_special_token_ids()

    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _get_dtype(self, torch_dtype: str) -> torch.dtype:
        if torch_dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        if torch_dtype == "float16":
            return torch.float16
        if torch_dtype == "bfloat16":
            return torch.bfloat16
        if torch_dtype == "float32":
            return torch.float32
        return torch.float32

    def _load_model_and_tokenizer(self):
        print("[Load] Loading model...")
        self.model, loading_info = Qwen2ForMedusa.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device != "cpu" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        print("[Load] loading_info:", loading_info)

        if self.device != "cpu":
            self.model = self.model.to(self.device)

        # Ensure use_cache is enabled for incremental gen
        try:
            self.model.config.use_cache = True
        except Exception:
            pass

        print("[Load] Loading tokenizer...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "<pad>"

        print("[Load] Model type:", type(self.model).__name__)
        print("[Load] vocab size:", len(self.tokenizer))
        print("[Load] medusa_num_heads:", getattr(self.model, "medusa_num_heads", "unknown"))

    def _setup_special_token_ids(self):
        assert self.tokenizer is not None

        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_start_id is None or self.im_start_id < 0:
            raise ValueError("Tokenizer missing <|im_start|>")
        if self.im_end_id is None or self.im_end_id < 0:
            raise ValueError("Tokenizer missing <|im_end|>")

        nl = self.tokenizer.encode("\n", add_special_tokens=False)
        if len(nl) == 0:
            raise ValueError("Tokenizer cannot encode newline")
        self.newline_id = nl[0]

        # role tokens
        self.system_tokens = self.tokenizer.encode("system", add_special_tokens=False)
        self.user_tokens = self.tokenizer.encode("user", add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)

        self.system_prefix = [self.im_start_id] + self.system_tokens + [self.newline_id]
        self.user_prefix = [self.im_start_id] + self.user_tokens + [self.newline_id]
        self.assistant_prefix = [self.im_start_id] + self.assistant_tokens + [self.newline_id]

        # wait token must exist in the SAME tokenizer you used for training
        self.wait_token_id = self.tokenizer.convert_tokens_to_ids(self.wait_token_str)
        if self.wait_token_id is None or self.wait_token_id < 0:
            raise ValueError(
                f"Tokenizer missing {self.wait_token_str}. Please load the tokenizer saved by your processor (save_tokenizer_to)."
            )

        print(f"[Tokens] im_start={self.im_start_id} im_end={self.im_end_id} newline={self.newline_id} wait={self.wait_token_id}")

    def _encode_channel_intro(self, idx: int) -> list[int]:
        assert self.tokenizer is not None
        txt = self.channel_intro_template.format(idx=idx)
        if not txt.endswith(" "):
            txt += " "
        return self.tokenizer.encode(txt, add_special_tokens=False)

    # ============================================================
    # NEW: inference prompt builder without <|im_end|> in assistant blocks
    # ============================================================
    def build_infer_prompt_no_im_end(self, question: str, num_heads: int) -> torch.Tensor:
        """
        TRAIN-ALIGNED IDs prompt builder:

        system:  system_prefix + sys_content + im_end
        user:    user_prefix   + usr_content + im_end
        assistant i:
                 assistant_prefix + (intro_i + wait*K_i)    # NO im_end

        base_align = max(len(sys_content), len(usr_content))
        K_i = max(0, base_align - len(intro_i_tokens))

        Returns: [1, S] LongTensor on CPU; caller moves to device.
        """
        assert self.tokenizer is not None

        sys_content = self.tokenizer.encode(self.system_message, add_special_tokens=False)
        usr_content = self.tokenizer.encode(question, add_special_tokens=False)
        base_align = max(len(sys_content), len(usr_content))

        ids: list[int] = []

        # system (NO newline after im_end)
        ids.extend(self.system_prefix)
        ids.extend(sys_content)
        if self.include_system_user_im_end:
            ids.append(self.im_end_id)

        # user (NO newline after im_end)
        ids.extend(self.user_prefix)
        ids.extend(usr_content)
        if self.include_system_user_im_end:
            ids.append(self.im_end_id)

        # assistants: prefix + (intro + wait*K), NO im_end
        for i in range(num_heads):
            ids.extend(self.assistant_prefix)

            if self.add_channel_intro_and_wait:
                intro = self._encode_channel_intro(i)
                n_wait = max(0, base_align - len(intro))
                ids.extend(intro)
                if n_wait > 0:
                    ids.extend([self.wait_token_id] * n_wait)

        return torch.tensor([ids], dtype=torch.long)

    def _debug_visibility_sanity(self, input_ids: torch.Tensor):
        """
        Sanity check: assistant prompt-context y should be >= max_y(system/user im_end)
        so cross-channel y-rule can "see" system/user content.
        """
        assert self.tokenizer is not None
        assert input_ids.dim() == 2 and input_ids.size(0) == 1

        ids = _as_list(input_ids[0])
        im_start = self.im_start_id
        im_end = self.im_end_id
        nl = self.newline_id

        sys_pat = [im_start] + self.system_tokens + [nl]
        usr_pat = [im_start] + self.user_tokens + [nl]
        as_pat = [im_start] + self.assistant_tokens + [nl]

        def match_at(x, i, pat):
            return i + len(pat) <= len(x) and x[i : i + len(pat)] == pat

        starts = [i for i, t in enumerate(ids) if t == im_start]
        if len(starts) < 2:
            print("[DEBUG y] cannot find enough blocks")
            return

        if not match_at(ids, starts[0], sys_pat) or not match_at(ids, starts[1], usr_pat):
            print("[DEBUG y] block header mismatch (not train-aligned?)")
            return

        sys_bs = starts[0]
        usr_bs = starts[1]
        try:
            sys_ie = ids.index(im_end, sys_bs + len(sys_pat))
            usr_ie = ids.index(im_end, usr_bs + len(usr_pat))
        except ValueError:
            print("[DEBUG y] cannot find system/user im_end")
            return

        sys_prefix_len = len(sys_pat)
        usr_prefix_len = len(usr_pat)
        sys_content_len = sys_ie - (sys_bs + sys_prefix_len)
        usr_content_len = usr_ie - (usr_bs + usr_prefix_len)

        max_y_sys = sys_prefix_len + sys_content_len  # im_end y
        max_y_usr = usr_prefix_len + usr_content_len
        max_y_sysusr = max(max_y_sys, max_y_usr)

        base_align = max(sys_content_len, usr_content_len)
        as_prefix_len = len(as_pat)

        if self.add_channel_intro_and_wait:
            as_ctx_y = as_prefix_len + (base_align - 1) if base_align > 0 else (as_prefix_len - 1)
        else:
            as_ctx_y = as_prefix_len - 1

        print(f"[DEBUG y] sys_content_len={sys_content_len}, usr_content_len={usr_content_len}, base_align={base_align}")
        print(f"[DEBUG y] max_y_sysusr(im_end_y)={max_y_sysusr}, assistant_ctx_y(estimated)={as_ctx_y}")
        print(f"[DEBUG y] assistant_should_see_sysusr = {as_ctx_y >= max_y_sysusr}")

    def generate(
        self,
        prompt: str,
        num_heads: int,
        max_length_per_head: int = 64,
        max_steps: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 0,
        do_sample: bool = False,
        allow_same_step_visible: bool = False,
        stop_on_im_end: bool = True,
        debug_y: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        assert self.model is not None
        assert self.tokenizer is not None

        model_heads = int(getattr(self.model, "medusa_num_heads", num_heads))
        if num_heads != model_heads:
            print(f"[Warn] num_heads({num_heads}) != model.medusa_num_heads({model_heads}), using model value.")
            num_heads = model_heads

        print("[Medusa] Building inference prompt...")
        print(f"[Medusa] prompt: {prompt}")
        # IMPORTANT: use the new builder that ensures NO <|im_end|> in assistant blocks
        input_ids = self.build_infer_prompt_no_im_end(question=prompt, num_heads=num_heads).to(self.device)

        print(f"[Medusa] input length: {input_ids.shape[1]} | heads={num_heads}")
        print(f"[Medusa] max_steps={max_steps}, max_len/head={max_length_per_head}, temp={temperature}, do_sample={do_sample}")

        if debug_y:
            self._debug_visibility_sanity(input_ids)

        ids = input_ids.detach().cpu().tolist()
        print("\n========== FULL INPUT (RAW DECODE) ==========")
        print(self.tokenizer.decode(ids[0], skip_special_tokens=False))

        with torch.no_grad():
            outputs = self.model.medusa_generate_interleaved(
                input_ids=input_ids,
                max_length_per_head=max_length_per_head,
                max_steps=max_steps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                allow_same_step_visible=allow_same_step_visible,
                stop_on_im_end=stop_on_im_end,
                **kwargs,
            )

        decoded: dict[int, str] = {}
        for h, toks in outputs.items():
            if toks.numel() == 0:
                decoded[h] = ""
            else:
                decoded[h] = self.tokenizer.decode(
                    toks.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )

        return {"input_ids": input_ids, "outputs": outputs, "decoded": decoded}


def main():
    ap = argparse.ArgumentParser("Medusa train-aligned inference")

    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)

    ap.add_argument("--num_heads", type=int, default=3)
    ap.add_argument("--max_length_per_head", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=256)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--allow_same_step_visible", action="store_true")
    ap.add_argument("--stop_on_im_end", action="store_true", help="Stop each head when token == <|im_end|>")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype",
    )

    # prompt alignment knobs (must match training)
    ap.add_argument("--system_message", type=str, default="You are a helpful assistant.")
    ap.add_argument("--wait_token_str", type=str, default="<|wait|>")
    ap.add_argument("--channel_intro_template", type=str, default="I am channel C{idx}.")
    ap.add_argument("--disable_intro_wait", action="store_true")
    ap.add_argument("--no_debug_y", action="store_true")

    args = ap.parse_args()

    infer = MedusaInference(args.model_path, device=args.device, torch_dtype=args.torch_dtype)
    infer.system_message = args.system_message
    infer.wait_token_str = args.wait_token_str
    infer.channel_intro_template = args.channel_intro_template
    infer.add_channel_intro_and_wait = not args.disable_intro_wait

    # after overriding wait_token_str/template, re-setup ids (wait token id depends on string)
    infer._setup_special_token_ids()

    res = infer.generate(
        prompt=args.prompt,
        num_heads=args.num_heads,
        max_length_per_head=args.max_length_per_head,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        allow_same_step_visible=args.allow_same_step_visible,
        stop_on_im_end=args.stop_on_im_end,
        debug_y=(not args.no_debug_y),
    )

    print("\n========== Decoded outputs ==========")
    for h in sorted(res["decoded"].keys()):
        print(f"\n[head {h}]")
        print(res["decoded"][h])


if __name__ == "__main__":
    main()
