"""Model backends. Each backend loads one checkpoint and exposes
`generate_1head_system_user(system_text, user_text, ...)`, matching the
`InferenceBackend` protocol in common.py.

VanillaBackend  -- plain AutoModelForCausalLM.generate(), for single-stream
                   (non multi-channel) checkpoints.
StreamBackend   -- Qwen3ForMultiStream.generate_stream(), for the multi-stream
                   (system/user/output channel) checkpoints.
"""
import importlib.util
import inspect
import os
import sys
from typing import Any, Dict

import torch
import transformers

_MULTISTREAM_QWEN3_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vendor", "qwen3_multistream"
)


def _resolve_dtype(device: str, torch_dtype: str) -> torch.dtype:
    if torch_dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class VanillaBackend:
    def __init__(self, model_path: str, device: str = "auto", torch_dtype: str = "auto"):
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(self.device, torch_dtype)

        print("[Init] (vanilla) model_path:", model_path)
        print("[Init] device:", self.device, "dtype:", self.dtype)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device != "cpu" else None,
            attn_implementation="sdpa",
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        try:
            self.model.config.use_cache = True
        except Exception:
            pass

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

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
        messages = [
            {"role": "system", "content": system_text or "You are a helpful assistant."},
            {"role": "user", "content": user_text or ""},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if final_prefix:
            prompt = prompt + final_prefix
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_length_per_head,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        if stop_on_im_end:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is not None and im_end_id >= 0:
                gen_kwargs["eos_token_id"] = im_end_id

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        text_raw = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
        return {"text_raw": text_raw, "n_new_tokens": int(gen_ids.numel())}


def _load_multistream_qwen3():
    """streaming_qwen3_4b/qwen3 is a standalone package (not installed),
    loaded by explicit file path so it can't collide with any other
    same-named package on sys.path."""
    if "qwen3_multistream" in sys.modules:
        return sys.modules["qwen3_multistream"]
    spec = importlib.util.spec_from_file_location(
        "qwen3_multistream",
        os.path.join(_MULTISTREAM_QWEN3_DIR, "__init__.py"),
        submodule_search_locations=[_MULTISTREAM_QWEN3_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_multistream"] = module
    spec.loader.exec_module(module)
    return module


class StreamBackend:
    def __init__(self, model_path: str, device: str = "auto", torch_dtype: str = "auto"):
        Qwen3ForMultiStream = _load_multistream_qwen3().Qwen3ForMultiStream

        self.model_path = model_path
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(self.device, torch_dtype)

        print("[Init] (stream) model_path:", model_path)
        print("[Init] device:", self.device, "dtype:", self.dtype)

        self.model, _ = Qwen3ForMultiStream.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device != "cpu" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        try:
            self.model.config.use_cache = True
        except Exception:
            pass

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        self._gen_fn = self.model.generate_stream
        self._gen_params = set(inspect.signature(self._gen_fn).parameters.keys())

    def _call_generate_stream(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in self._gen_params}
        return self._gen_fn(**kwargs)

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
        setattr(self.model, "system_message", system_text or "You are a helpful assistant.")

        out = self._call_generate_stream(
            question_text=user_text or "",
            assistant_prefix_text=final_prefix,
            max_new_tokens=max_length_per_head,
            max_steps=max_steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stop_on_im_end=stop_on_im_end,
            include_im_end_in_cross_channel=True,
            allow_same_step_visible=False,
            disable_assistant_cross_channel=False,
            presence_penalty=presence_penalty,
        )
        toks = out.get(0, None)
        if toks is None or toks.numel() == 0:
            return {"text_raw": "", "n_new_tokens": 0}

        text_raw = self.tokenizer.decode(
            toks.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return {"text_raw": text_raw, "n_new_tokens": int(toks.numel())}


BACKENDS = {
    "vanilla": VanillaBackend,
    "stream": StreamBackend,
}
