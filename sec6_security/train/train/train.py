import os
import sys
import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional
import torch
import torch.distributed as dist
import torch
import transformers
import trl
from datasets import load_dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from qwen_medusa import Qwen2ForMedusa, Qwen2MedusaConfig, ResBlock
#from custom_trainer import CustomizedTrainer
from custom_trainer_contrast_new5 import CustomizedTrainer
from custom_datasets import CustomMultiHeadDataset, CustomDataCollator
import torch.nn as nn
# Configure environment
# WANDB API key: read from WANDB_API_KEY env var (do not commit a key).
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=8096)
    wandb_project: Optional[str] = field(default="multistream-security")
    train_file_path: Optional[str] = field(
        default="Multiverse4FM/Autoregressive-1K-mixed"
    )
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def train():
    """Main training function."""
    # Parse arguments
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
 
    # Load model with Medusa heads
    medusa_lm_head, loading_info = Qwen2ForMedusa.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        use_cache=False
    )
    #init_medusa_from_lm_head(medusa_lm_head)
    logging.info(f"Missing keys: {loading_info['missing_keys']}")
    logging.info(f"Unexpected keys: {loading_info['unexpected_keys']}")
    logging.info(f"Mismatched keys: {loading_info['mismatched_keys']}")

    # Ensure parameters are contiguous in memory
    # for param in medusa_lm_head.parameters():
    #     param.data = param.data.contiguous()
    
    # Setup tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, 
        use_fast=True
    )
    

    train_set = CustomMultiHeadDataset(
        cache_dir=config.train_file_path,
        max_seq_length=config.block_size,
        preload_to_memory=False
    )
    
    # Create data collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = CustomDataCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Configure training arguments
    args.max_seq_length = config.block_size
    args.remove_unused_columns = False
    args.skip_prepare_dataset = True
    args.eval_strategy = 'no'
    
  
    trainer = CustomizedTrainer(
        medusa_lm_head,
        train_dataset=train_set,
        args=args,
        data_collator=collator,
        backbone_lr=1e-5,   # 小
        head_lr=1e-4,     # 大
    )

    # Train and save
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()