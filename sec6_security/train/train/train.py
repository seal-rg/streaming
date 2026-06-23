import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import transformers
import trl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_datasets import CustomDataCollator, CustomMultiHeadDataset
from custom_trainer import CustomizedTrainer
from qwen2 import Qwen2ForMultiStream

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-7B")
    train_file_path: str = field(default="")
    block_size: int = field(default=8096)
    wandb_project: Optional[str] = field(default="multistream-security")

    def __post_init__(self):
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project


def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    logging.info(f"Training config: {asdict(config)}")

    model, loading_info = Qwen2ForMultiStream.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        use_cache=False,
    )
    logging.info(f"Missing keys:    {loading_info['missing_keys']}")
    logging.info(f"Unexpected keys: {loading_info['unexpected_keys']}")
    logging.info(f"Mismatched keys: {loading_info['mismatched_keys']}")

    for param in model.parameters():
        param.data = param.data.contiguous()

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_set = CustomMultiHeadDataset(
        cache_dir=config.train_file_path,
        max_seq_length=config.block_size,
        preload_to_memory=False,
    )
    collator = CustomDataCollator(pad_token_id=tokenizer.pad_token_id)

    args.max_seq_length = config.block_size
    args.remove_unused_columns = False
    args.skip_prepare_dataset = True
    args.eval_strategy = "no"

    trainer = CustomizedTrainer(
        model,
        train_dataset=train_set,
        args=args,
        data_collator=collator,
        backbone_lr=1e-5,
        head_lr=1e-4,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
