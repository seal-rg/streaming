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
from qwen3 import Qwen3ForMedusa, Qwen3MedusaConfig
#from custom_trainer import CustomizedTrainer
from custom_trainer_optimized import CustomizedTrainer

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
    wandb_project: Optional[str] = field(default="multistream-efficiency")
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
    medusa_lm_head, loading_info = Qwen3ForMedusa.from_pretrained(
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
    for param in medusa_lm_head.parameters():
        param.data = param.data.contiguous()
    
    # Setup tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, 
        use_fast=True
    )
    
    # Configure template and padding tokens based on model type
    # if "Llama" in config.model_name:
    #     instruction_template = "<|start_header_id|>user<|end_header_id|>"
    #     response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #     tokenizer.pad_token = "<|reserved_special_token_5|>"
    # else:
    #     instruction_template = "<|im_start|>user"
    #     response_template = "<|im_start|>assistant\n"
    #     tokenizer.pad_token = "<|fim_pad|>"

    # ${RESULTS_ROOT}/cache_dataset_new6
    # ${MODELS_ROOT}/prepare_output/extracted_results1
    # Load training dataset
    # ${RESULTS_ROOT}/cache_dataset_1221
    # ${RESULTS_ROOT}/cache_dataset_1223_merged
    # ${RESULTS_ROOT}/cache_dataset_1224_merged
    # ${RESULTS_ROOT}/cache_dataset_1224
    # ${RESULTS_ROOT}/cache_dataset_1226_v1
    #${RESULTS_ROOT}/cache_dataset_1225
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
    
    # Initialize trainer
    # trainer = CustomizedTrainer(
    #     medusa_lm_head,
    #     train_dataset=train_set,
    #     args=args,
    #     data_collator=collator,
    # )
    #freeze_for_attention_only(medusa_lm_head, train_layernorm=True, train_embeddings=True, last_n_layers=None)
    # 只训练 medusa_head + channel_embedding
    #freeze_for_medusa_and_channel_embedding(medusa_lm_head, require_embed_type_ori=True)

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