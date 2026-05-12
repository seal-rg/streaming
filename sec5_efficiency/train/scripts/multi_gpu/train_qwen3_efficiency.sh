#!/bin/bash
#
# Section 5 (Efficiency) training launcher — Qwen3 base + Medusa-style heads.
# Edit SEC5_ROOT and the BASE_MODEL / TRAIN_FILE paths for your environment.
#
# Usage:
#   bash train_qwen3_efficiency.sh                  # uses TRAIN_FILE default
#   bash train_qwen3_efficiency.sh _ 0              # picks TASKS[0] as TRAIN_FILE
#
# The TASKS array below mirrors the original sweep over several
# pre-processed datasets (LogicNLI, PubMedQA, SQuAD, ProofWriter,
# OpenMath, etc.). Set $2 to an index, or just override TRAIN_FILE.

set -e

SEC5_ROOT="${SEC5_ROOT:-/path/to/sec5_efficiency}"
BASE_MODEL="${BASE_MODEL:-/path/to/Qwen3-1.7B/snapshots/medusa}"

# Pre-processed dataset caches — replace with your paths.
TASKS=(
  "/path/to/logicnli_cache"
  "/path/to/pubmedqa_cache"
  "/path/to/squad_cache"
  "/path/to/proofwriter_cache"
  "/path/to/openmath_cache"
)
PID="${2:-}"
if [ -n "${PID}" ]; then
  TRAIN_FILE="${TASKS[$PID]}"
else
  TRAIN_FILE="${TRAIN_FILE:-${TASKS[0]}}"
fi

lr=1e-5
epochs=3
weight_decay=1e-4
micro_batch_size=4
gradient_accumulation_steps=4
max_steps=-1

uid="$(date +%Y%m%d_%H%M%S_%N)"
OUTPUT_DIR="${OUTPUT_DIR:-./results/efficiency_${uid}}"

export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_LOG_LEVEL=info

accelerate launch \
    --num_processes=4 \
    --main_process_port=29501 \
    --config_file ${SEC5_ROOT}/train/train/deepspeed_zero2.yaml \
    ${SEC5_ROOT}/train/train/train_qwen3.py \
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
    --save_steps=100
