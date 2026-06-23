#!/usr/bin/env bash
# Evaluate a Sec 5 multi-stream checkpoint on one benchmark task.
#
# Required env vars:
#   SEC5_ROOT   — path to sec5_efficiency/
#   MODEL       — path to the trained Qwen3ForMultiStream checkpoint
#
# Optional env vars (with defaults shown):
#   TASK        — gsm8k | proofwriter | logicnli | mathqa | logiqa | arc_c |
#                 math500 | strategyqa | squad | pubmedqa   (default: gsm8k)
#   N           — number of samples, 0 = all                (default: 0)
#   MAX_TOKENS  — max new tokens per sample                  (default: 1024)
#   OUT_DIR     — output directory                           (default: ${SEC5_ROOT}/infer/output)

TASK=${TASK:-gsm8k}
N=${N:-0}
MAX_TOKENS=${MAX_TOKENS:-1024}
OUT_DIR=${OUT_DIR:-${SEC5_ROOT}/infer/output/${TASK}}

python3 ${SEC5_ROOT}/infer/infer_qwen3.py \
  --model   ${MODEL} \
  --task    ${TASK} \
  --n       ${N} \
  --max_new_tokens ${MAX_TOKENS} \
  --stream \
  --out_dir ${OUT_DIR}
