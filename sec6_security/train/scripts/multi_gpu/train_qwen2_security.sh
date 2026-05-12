#!/bin/bash
#
# Section 6 (Security) training launcher — Qwen2.5 base + Medusa-style heads.
# Edit SEC6_ROOT and the BASE_MODEL / TRAIN_FILE paths for your environment.

set -e

# Project root (this repo)
SEC6_ROOT="${SEC6_ROOT:-/path/to/sec6_security}"

# Base model with extended (medusa) vocab. Produce via
# train/custom_datasets/example_full_pipeline.sh, then point here.
BASE_MODEL="${BASE_MODEL:-/path/to/Qwen2.5-7B/snapshots/medusa}"

# Pre-processed cache (run train/custom_datasets/process_data.sh first)
TRAIN_FILE="${TRAIN_FILE:-/path/to/processed_cache}"

# Hyperparameters
lr=1e-5
epochs=3
weight_decay=1e-4
micro_batch_size=4
gradient_accumulation_steps=4
max_steps=-1

uid="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-./results/security_${uid}}"

export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info

accelerate launch \
    --num_processes=4 \
    --main_process_port=29501 \
    --config_file ${SEC6_ROOT}/train/train/deepspeed_zero2.yaml \
    ${SEC6_ROOT}/train/train/train.py \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --max_steps=${max_steps} \
    --train_file_path="${TRAIN_FILE}" \
    --model_name=${BASE_MODEL} \
    --warmup_ratio=0.1 \
    --bf16=True \
    --eval_strategy="steps" \
    --eval_steps=50 \
    --logging_steps=1 \
    --max_grad_norm=0.5 \
    --lr_scheduler_type="constant_with_warmup" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${OUTPUT_DIR}" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --save_strategy="steps" \
    --save_steps=50
