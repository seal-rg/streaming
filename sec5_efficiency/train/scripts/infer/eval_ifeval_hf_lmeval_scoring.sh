#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Conda
# ---------------------------



# MODELS=(
#   #"${RESULTS_ROOT}/p_20260217_230739_605921948"
#   "${RESULTS_ROOT}/p_20260216_011040_177578477"
# )

# ---------------------------
# Env
# ---------------------------
export TEMP=/tmp

python3 ${SEC5_ROOT}/train/eval/eval_ifeval_hf_lmeval_scoring.py \
  --model ${RESULTS_ROOT}/p_20260216_011040_177578477 \
  --out_jsonl ${SEC5_ROOT}/train/scripts/infer/ifeval_hf_qwen2_2.jsonl \
  --max_new_tokens 8192 \
  --temperature 0.7 \
  --top_p 1.0