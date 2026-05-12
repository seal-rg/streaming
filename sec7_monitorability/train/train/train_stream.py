#!/usr/bin/env python3
"""
Training entry point for 10-channel parallel stream data.

Uses Qwen3ForCausalLM (standard causal LM, no medusa heads) with:
  - config.num_channels = 10 (10-channel embedding)
  - config.role_gating_enabled = False (initially)
  - Block-causal attention mask (same-row tokens cannot see each other)
  - Shift-by-10 loss (next-row same-channel prediction)

Usage:
  accelerate launch train/train/train_stream.py --config config/baseline.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import socket
import sys
import warnings
from datetime import datetime

import torch
import yaml

# Check device health immediately after loading torch and standard libraries without loading cuda/hip:
nvml_count = torch.cuda._device_count_amdsmi() if torch.version.hip else torch.cuda._device_count_nvml()
if nvml_count < 1:
    raise ValueError(f"Node failure! Device manager init failed on {socket.gethostname()}. Exiting immediately.")


import transformers
import trl

# Silence noisy startup output
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS.*")
logging.getLogger("accelerate.accelerator").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"
os.environ.setdefault("OMP_NUM_THREADS", "4")  # avoid torch.distributed warning
os.environ.setdefault("DS_LOG_LEVEL", "WARNING")  # silence DeepSpeed memory stats

# Add parent directory to path so we can import sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_dataset import StreamDataCollator, StreamDataset, StreamEvalDataset
from custom_dataset.chat_dataset import ChatDataCollator, ChatDataset
from custom_trainer_stream import StreamTrainer


def detect_architecture(model_path):
    """Read config.json from model path and return architecture name."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return "qwen3"  # default fallback
    with open(config_path) as f:
        model_json = json.load(f)
    model_type = model_json.get("model_type", "qwen3")
    return model_type


def load_stream_model(model_path, model_config_overrides):
    """Load the appropriate model class based on config.json model_type."""
    arch = detect_architecture(model_path)
    logging.info(f"Detected architecture: {arch}")

    if arch in ("qwen3_5", "qwen3_5_text"):
        from qwen3_5 import StreamQwen3_5ForCausalLM, StreamQwen3_5TextConfig

        config_cls = StreamQwen3_5TextConfig
        model_cls = StreamQwen3_5ForCausalLM
    elif arch in ("qwen3_5_moe", "qwen3_5_moe_text"):
        from qwen3_5_moe import StreamQwen3_5MoeForCausalLM, StreamQwen3_5MoeTextConfig

        config_cls = StreamQwen3_5MoeTextConfig
        model_cls = StreamQwen3_5MoeForCausalLM
    else:
        from qwen3 import Qwen3ForCausalLM, Qwen3MedusaConfig

        config_cls = Qwen3MedusaConfig
        model_cls = Qwen3ForCausalLM

    # Qwen3.5 models have nested config: top-level wraps text_config + vision_config.
    # We load just the text_config portion for text-only training.
    if arch in ("qwen3_5", "qwen3_5_moe"):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            full_config = json.load(f)
        text_config_dict = full_config.get("text_config", full_config)
        model_config = config_cls(**text_config_dict)
    else:
        model_config = config_cls.from_pretrained(model_path)

    for k, v in model_config_overrides.items():
        setattr(model_config, k, v)

    # Ensure pad_token_id exists before model construction (some configs omit it)
    if getattr(model_config, "pad_token_id", None) is None:
        model_config.pad_token_id = getattr(model_config, "eos_token_id", 0)

    model, loading_info = model_cls.from_pretrained(  # type: ignore
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    return model, model_config, loading_info


class WandbConfigCallback(transformers.TrainerCallback):
    """Push the full YAML config to wandb.config at train start."""

    def __init__(self, config):
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(self.config, allow_val_change=True)
        except ImportError:
            pass


class PackingEpochCallback(transformers.TrainerCallback):
    """Update dataset epoch so packing randomization varies across epochs."""

    def __init__(self, dataset):
        self.dataset = dataset
        self._last_epoch = -1

    def on_step_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)  # type: ignore
        if epoch != self._last_epoch:
            self.dataset.set_epoch(epoch)
            self._last_epoch = epoch


def _save_lean_checkpoint(ds_engine, save_dir, tag="best"):
    """Save a lean ZeRO-3 checkpoint: fp32 model partitions only, no optimizer state.

    Each rank writes its own partition (~27GB for 27B model on 4 GPUs).
    No allgather, no NCCL timeout risk.  Reconstruct offline with:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state_dict = get_fp32_state_dict_from_zero_checkpoint(save_dir, tag="best")
    """
    from deepspeed.runtime.zero.config import ZeroStageEnum

    optimizer = ds_engine.optimizer
    original = optimizer._rigid_state_dict

    def _lean_state_dict():
        return {
            "zero_stage": ZeroStageEnum.weights,
            "partition_count": optimizer.partition_count,
            "fp32_flat_groups": optimizer.fp32_partitioned_groups_flat,
        }

    optimizer._rigid_state_dict = _lean_state_dict
    try:
        ds_engine.save_checkpoint(save_dir, tag=tag)
    finally:
        optimizer._rigid_state_dict = original


class BestModelCallback(transformers.TrainerCallback):
    """Save model checkpoint whenever eval loss improves.

    ZeRO-2: uses trainer.save_model() (direct HF format).
    ZeRO-3: uses lean checkpoint (fp32 partitions, no optimizer state,
    no allgather). Convert to HF format post-training via convert_best().
    """

    def __init__(self, save_dir, tokenizer, metric=None):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.metric = metric
        self.best_loss = float("inf")
        self.best_step = -1
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _is_zero3(self):
        ds_plugin = getattr(self.trainer.accelerator.state, "deepspeed_plugin", None)  # type: ignore
        if ds_plugin is not None:
            return getattr(ds_plugin, "zero_stage", 0) == 3
        return False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.trainer is None:
            return
        # Two codepaths emit eval metrics under different keys:
        #   stream trainer (compute_metrics in custom_trainer_stream.py):
        #     eval/loss (with slash), eval/loss_<channel>, etc.
        #   chat_baseline (vanilla HF Trainer on ChatDataset):
        #     eval_loss (underscore), no channel split.
        # If self.metric is set explicitly, honor it; else auto-detect.
        if self.metric is not None:
            loss = metrics.get(self.metric)
        else:
            # Prefer the stream key if present, fall back to HF default.
            loss = metrics.get("eval/loss")
            if loss is None:
                loss = metrics.get("eval_loss")
        if loss is None or loss >= self.best_loss:
            return
        self.best_loss = loss
        self.best_step = state.global_step
        metric_label = self.metric or ("eval/loss" if "eval/loss" in metrics else "eval_loss")
        logging.info(f"[BestModel] New best {metric_label}={loss:.4f} at step {state.global_step}, saving...")
        if self._is_zero3():
            ds_engine = self.trainer.model_wrapped
            grad_state = {n: p.requires_grad for n, p in ds_engine.named_parameters()}
            for p in ds_engine.parameters():
                p.requires_grad = True
            _save_lean_checkpoint(ds_engine, self.save_dir)
            for n, p in ds_engine.named_parameters():
                p.requires_grad = grad_state[n]
        else:
            self.trainer.save_model(self.save_dir)
        # Write metadata (rank 0 only)
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            import json

            # Save metadata
            meta = {
                "step": state.global_step,
                "metric": self.metric,
                "loss": loss,
                "metrics": {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float))},
            }
            with open(os.path.join(self.save_dir, "best_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            # Save config + tokenizer (needed for inference loading)
            unwrapped = self.trainer.accelerator.unwrap_model(self.trainer.model)
            unwrapped.config.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
        logging.info(f"[BestModel] Saved to {self.save_dir}")


_is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

logging.basicConfig(
    level=logging.INFO if _is_main else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    ALL_CHANNEL_NAMES = [
        "user",
        "output",
        "analytical",
        "skeptical",
        "intuitive",
        "between",
        "curious",
        "void",
        "instinct",
        "synthesis",
    ]

    # Allow env-var overrides for paths that differ across machines
    if os.environ.get("MODEL_PATH"):
        cfg["model"] = os.path.join(os.environ["MODEL_PATH"], os.path.basename(cfg["model"]))
    if os.environ.get("DATA_PATH"):
        cfg["data"] = os.path.join(os.environ["DATA_PATH"], os.path.basename(cfg["data"]))

    optim = cfg.get("optim", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("eval", {})
    longce_cfg = cfg.get("longce", {})
    arch_cfg = cfg.get("architecture", {})
    trainer_type = cfg.get("trainer", "baseline")

    # Parse channels: int (first N) or list of names
    channels_cfg = cfg.get("channels", 10)
    if isinstance(channels_cfg, int):
        active_channels = list(range(channels_cfg))
    else:
        active_channels = [ALL_CHANNEL_NAMES.index(name) for name in channels_cfg]
    num_channels = len(active_channels)
    active_channel_names = [ALL_CHANNEL_NAMES[i] for i in active_channels]
    user_channel_idx = active_channels.index(0) if 0 in active_channels else None
    output_channel_idx = active_channels.index(1) if 1 in active_channels else None

    # Wandb
    os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "StreamLLM")
    bb_lr = float(optim.get("backbone_lr", 1e-5))
    ce_lr = float(optim.get("channel_embedding_lr", 1e-4))
    run_name = cfg.get("name") or f"{trainer_type}-bb{bb_lr:.0e}-ce{ce_lr:.0e}-ep{train_cfg.get('epochs', 3)}"

    # Unique output dir per run — must be consistent across all ranks
    output_dir = os.environ.get("STREAM_OUTPUT_DIR") or cfg.get("output_dir")
    if not output_dir:
        # MASTER_PORT is set by accelerate, unique per launch, same across all ranks
        uid = os.environ.get("MASTER_PORT", "local")
        output_dir = f"results/{run_name}_{uid}"

    logging.info(f"[TRAINER] Run starting at {datetime.now().isoformat()}")
    logging.info(f"Config: {cfg}")
    logging.info(f"Run name: {run_name}")

    # Architecture ablation settings
    channel_embedding_method = arch_cfg.get("channel_embedding", "additive")
    attention_mask_type = arch_cfg.get("attention_mask", "block_causal")

    # Chat baseline: force no channel embedding, standard causal attention
    if trainer_type == "chat_baseline":
        channel_embedding_method = "none"
        attention_mask_type = "causal"
        num_channels = 1  # no multi-channel structure
    deltanet_block_causal = str(arch_cfg.get("deltanet_block_causal", "block_causal"))
    deltanet_conv = arch_cfg.get("deltanet_conv", "column")

    # Role gating settings
    rg_cfg = cfg.get("role_gating", {})

    # Build model config overrides
    model_config_overrides = {
        "num_channels": num_channels,
        "channel_embedding_method": channel_embedding_method,
        "deltanet_block_causal": deltanet_block_causal,
        "deltanet_conv": deltanet_conv,
        "role_gating_enabled": bool(rg_cfg.get("enabled", False)),
        "role_gating_granularity": str(rg_cfg.get("granularity", "layer")),
        "role_gating_mode": str(rg_cfg.get("mode", "query")),
        "role_gating_mlp_hidden": int(rg_cfg.get("mlp_hidden", 0)),
        "role_gating_tau": float(rg_cfg.get("tau", 2.0)),
        "role_gating_beta_max": float(rg_cfg.get("beta_max", 0.8)),
        "role_gating_log_eps": float(rg_cfg.get("log_eps", 1e-4)),
        "role_gating_log_clip_min": float(rg_cfg.get("log_clip_min", -6.0)),
        "role_gating_uniform_mix": float(rg_cfg.get("uniform_mix", 0.05)),
        "use_cache": False,
    }
    if train_cfg.get("attention_dropout") is not None:
        model_config_overrides["attention_dropout"] = float(train_cfg["attention_dropout"])

    # Load model (auto-detects architecture from config.json)
    model, model_config, loading_info = load_stream_model(cfg["model"], model_config_overrides)

    n_miss = len(loading_info["missing_keys"])
    n_unex = len(loading_info["unexpected_keys"])
    logging.info(f"Loaded model: {n_miss} missing keys (expected: channel_embedding + gate_in_norm), {n_unex} unexpected")
    if n_unex > 0:
        logging.warning(f"Unexpected keys: {loading_info['unexpected_keys']}")
    if loading_info["mismatched_keys"]:
        logging.warning(f"Mismatched keys: {loading_info['mismatched_keys']}")

    for param in model.parameters():
        param.data = param.data.contiguous()

    # Selective component freezing
    trainable = train_cfg.get("trainable", None)
    if trainable is not None:
        COMPONENT_PATTERNS = {
            "channel_embedding": lambda n: "channel_embedding" in n,
            "attention": lambda n: "self_attn" in n,
            "deltanet": lambda n: "linear_attn" in n,
            "mlp": lambda n: ".mlp." in n,
            "embed": lambda n: "embed_tokens" in n or "lm_head" in n,
            "norm": lambda n: "layernorm" in n.lower() or n == "model.norm.weight",
        }
        trainable_set = set(trainable)
        frozen_count = 0
        for name, param in model.named_parameters():
            if not any(COMPONENT_PATTERNS[c](name) for c in trainable_set if c in COMPONENT_PATTERNS):
                param.requires_grad = False
                frozen_count += 1
        total = sum(1 for _ in model.parameters())
        logging.info(f"Trainable components: {sorted(trainable_set)}, froze {frozen_count}/{total} parameters")

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # SentencePiece (▁ prefix) vs BPE (Ġ prefix) need different encoding for dash
    _sp = tokenizer.convert_ids_to_tokens(tokenizer.encode("test", add_special_tokens=False))[0].startswith("▁")
    silence_token_id = tokenizer.encode("-" if _sp else " -", add_special_tokens=False)[0]
    logging.info(f"Silence token: '-' = token ID {silence_token_id} (sentencepiece={_sp})")

    # Dataset and collator
    block_size = cfg.get("block_size", 5120)
    is_chat_baseline = trainer_type == "chat_baseline"

    if is_chat_baseline:
        full_set = ChatDataset(
            data_path=cfg["data"],
            tokenizer=tokenizer,
            max_seq_length=block_size,
            silence_token_id=silence_token_id,
            thinking_mode=cfg.get("chat_thinking_mode", "none"),
        )
        collator = ChatDataCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        full_set = StreamDataset(
            data_path=cfg["data"],
            active_channels=active_channels,
            max_seq_length=block_size,
            prefix_rows=cfg.get("prefix_rows", 0),
            pack_samples=cfg.get("pack_samples", 0),
            pack_trim=cfg.get("pack_trim", "top"),
        )
        collator = StreamDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            num_channels=num_channels,
            attention_mask_type=attention_mask_type,
            silence_token_id=silence_token_id,
        )

    # Deterministic train/val split
    n_val = min(eval_cfg.get("val_samples", 30), len(full_set))
    val_set_ctx1 = val_set_ctx2 = None
    if n_val > 0:
        rng = random.Random(233)
        all_idx = list(range(len(full_set)))
        rng.shuffle(all_idx)
        val_indices = sorted(all_idx[:n_val])
        train_indices = sorted(all_idx[n_val:])
        # Restrict packing pool to training indices only (prevents val data leak)
        if hasattr(full_set, "restrict_pack_pool"):
            full_set.restrict_pack_pool(train_indices)  # type: ignore
        train_set = torch.utils.data.Subset(full_set, train_indices)
        if is_chat_baseline:
            # Chat baseline: simple Subset, no context eval
            val_set = torch.utils.data.Subset(full_set, val_indices)
        else:
            # Stream: always unpacked eval (fair comparison across all runs)
            val_set = StreamEvalDataset(full_set, val_indices, n_context=0)
            val_set_ctx1 = StreamEvalDataset(full_set, val_indices, n_context=1)
            val_set_ctx2 = StreamEvalDataset(full_set, val_indices, n_context=2)
    else:
        train_set = full_set
        val_set = None
    logging.info(f"Dataset: {len(full_set)} total, {len(train_set)} train, {n_val} val, channels={active_channel_names}")

    # Compute warmup_steps from ratio
    ga = train_cfg.get("gradient_accumulation_steps", 4)
    mbs = train_cfg.get("micro_batch_size", 2)
    gpu_count = int(os.environ.get("WORLD_SIZE", 1))
    max_steps = train_cfg.get("max_steps", -1)
    epochs = train_cfg.get("epochs", 3)
    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = math.ceil(len(train_set) / (mbs * gpu_count * ga)) * epochs
    warmup_ratio = float(optim.get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)
    logging.info(f"Training: {total_steps} steps, {warmup_steps} warmup steps ({warmup_ratio:.0%})")

    # SFTConfig from YAML
    args = trl.SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        max_length=block_size,
        per_device_train_batch_size=mbs,
        per_device_eval_batch_size=mbs,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=bb_lr,
        weight_decay=float(optim.get("weight_decay", 1e-4)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=optim.get("scheduler", "constant_with_warmup"),
        adam_beta1=optim.get("adam_beta1", 0.9),
        adam_beta2=optim.get("adam_beta2", 0.95),
        max_grad_norm=optim.get("max_grad_norm", 0.5),
        bf16=True,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        # MoE expert routing creates variable-shaped tensors; use_reentrant=True
        # avoids shape-validation errors during gradient checkpoint recomputation.
        gradient_checkpointing_kwargs=(
            {"use_reentrant": True} if getattr(model_config, "model_type", "") in ("qwen3_5_moe", "qwen3_5_moe_text") else {}
        ),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_strategy="steps" if train_cfg.get("save_steps", 0) > 0 else "no",
        save_steps=train_cfg.get("save_steps", 0),
        save_only_model=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        neftune_noise_alpha=train_cfg.get("neftune_alpha", None),
        eval_strategy="steps",
        eval_steps=eval_cfg.get("eval_steps", 50),
        report_to=["wandb"],
    )

    # Create trainer
    if is_chat_baseline:
        # Standard SFT: no channel embeddings, no block-causal, shift-by-1 loss
        # Labels are already built in ChatDataset with assistant-only masking
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_set,
            eval_dataset=val_set,
            processing_class=tokenizer,
            args=args,
            data_collator=collator,
        )
    else:
        trainer = StreamTrainer(
            model,
            train_dataset=train_set,
            eval_dataset=val_set,
            processing_class=tokenizer,
            args=args,
            data_collator=collator,
            backbone_lr=bb_lr,
            channel_embedding_lr=ce_lr,
            cooldown_ratio=float(optim.get("cooldown_ratio", 0.0)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
            z_loss_weight=float(train_cfg.get("z_loss_weight", 0.0)),
            num_channels=num_channels,
            enable_longce=(trainer_type == "longce"),
            longce_gamma=longce_cfg.get("gamma", 50.0),
            longce_warmup_steps=longce_cfg.get("warmup_steps", 0),
            longce_ramp_steps=longce_cfg.get("ramp_steps", 0),
            longce_prob=longce_cfg.get("prob", 1.0),
            longce_num_channels_per_step=longce_cfg.get("num_channels_per_step", 3),
            longce_uniform_mix=float(longce_cfg.get("uniform_mix", 0.0)),
            # Importance-weighting knobs (previously hidden defaults of False).
            # isoft_per_channel_mean1: normalize weights per-channel so each
            # channel contributes equal expected loss mass, avoiding one
            # high-LSD channel dominating the step.
            # self_only_drop_input_ids: replace other-channel input_ids with
            # a drop token in the self-only forward. Cosmetic if attention
            # is the sole cross-token pathway (it is), so default False.
            isoft_per_channel_mean1=bool(longce_cfg.get("per_channel_mean1", False)),
            self_only_drop_input_ids=bool(longce_cfg.get("self_only_drop_input_ids", False)),
            # "random" (legacy) samples channels uniformly each forward;
            # "cycle" round-robins deterministically so every channel gets
            # equal LongCE coverage per cycle — see trainer._select_longce_channels.
            # Ignored when self_only_mode=="global" (coverage is complete).
            longce_selection_mode=str(longce_cfg.get("selection_mode", "random")),
            # "per_channel" runs N self-only forwards for N selected channels;
            # "global" runs ONE self-only forward with a mask that blocks all
            # cross-column attention — covers every channel in 2 forwards
            # (full + global-self-only) at strictly lower compute than
            # 1+num_channels_per_step forwards. Requires DeltaNet in
            # column-mode for hybrid models (the default for qwen3_5 configs).
            longce_self_only_mode=str(longce_cfg.get("self_only_mode", "per_channel")),
            eval_num_samples=eval_cfg.get("num_samples", 8),
            eval_prefill_frac=eval_cfg.get("prefill_frac", 0.25),
            eval_gen_rows=eval_cfg.get("gen_rows", 100),
            eval_temperature=eval_cfg.get("temperature", 0.8),
            eval_top_p=eval_cfg.get("top_p", 0.9),
            eval_top_k=eval_cfg.get("top_k", 0),
            mask_user_loss=cfg.get("mask_user_loss", False),
            user_channel_idx=user_channel_idx,
            output_channel_idx=output_channel_idx,
            channel_names=active_channel_names,
            silence_token_id=silence_token_id,
            attention_mask_type=attention_mask_type,
            eval_context_datasets=[val_set_ctx1, val_set_ctx2],
            optim_8bit=optim.get("adam_8bit", False),
        )

    trainer.add_callback(WandbConfigCallback(cfg))
    if hasattr(train_set, "set_epoch"):
        trainer.add_callback(PackingEpochCallback(train_set))

    # Best-model saving: track eval/loss and save whenever it improves
    save_dir = cfg.get("save_dir")
    if train_cfg.get("save_best_model", False):
        # metric=None → callback auto-detects: prefers "eval/loss" (emitted
        # by the stream compute_metrics) and falls back to "eval_loss" (HF
        # default, what the chat_baseline path emits). Chat_baseline runs
        # were silently noop'ing on the previous hardcoded "eval/loss".
        best_cb = BestModelCallback(
            save_dir=args.output_dir,
            tokenizer=tokenizer,
        )
        best_cb.set_trainer(trainer)
        trainer.add_callback(best_cb)

    logging.info(f"[TRAINER] Run starting at {datetime.now().isoformat()}")
    trainer.train()

    if train_cfg.get("save_final_model", False):
        # Re-enable requires_grad on all params so ZeRO-3 gathers them for saving
        for param in model.parameters():
            param.requires_grad = True
        save_dir = cfg.get("save_dir", args.output_dir)
        if save_dir != args.output_dir:
            save_dir = os.path.join(save_dir, run_name)
            os.makedirs(save_dir, exist_ok=True)
        logging.info(f"[SaveFinal] START saving to {save_dir} at {datetime.now().isoformat()}")
        trainer.save_model(output_dir=save_dir)
        tokenizer.save_pretrained(save_dir)
        logging.info(f"[SaveFinal] END saving at {datetime.now().isoformat()}")
    else:
        # Remove empty output dir created by Trainer when no saving is configured
        if trainer.accelerator.is_main_process:
            try:
                os.rmdir(args.output_dir)  # type: ignore
            except OSError:
                pass  # not empty — something was saved, keep it
    trainer.accelerator.wait_for_everyone()
    logging.info(f"[TRAINER] Run completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    train()
