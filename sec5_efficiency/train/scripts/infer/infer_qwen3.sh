#!/usr/bin/env bash
set -e

# ---------------------------
# Conda
# ---------------------------



# ---------------------------
# Env
# ---------------------------
export TEMP=/tmp

# ---------------------------
# Task list (顺序 = PROCESS id)
# ---------------------------
TASKS=(
  math500
  pubmedqa
  squad
  proofwriter
  #folio
  gsm8k
  # mathqa
  logicnli
  logiqa
  #squad
  #strategyqa
  # mmlu_redux
  # arc_c
)

# ---------------------------
# Select task by PROCESS
# ---------------------------
PID=$2

if [ "$PID" -ge "${#TASKS[@]}" ]; then
  echo "[ERROR] PROCESS=$PID exceeds number of tasks (${#TASKS[@]})"
  exit 1
fi

TASK=${TASKS[$PID]}

echo "======================================"
echo " Condor PROCESS = $PID"
echo " Evaluating task = $TASK"
echo "======================================"

# ---------------------------
# Output dir (per task)
# ---------------------------
OUT_DIR=${DATA_ROOT}/para_out/eval_qwen3_17b_all_new${TASK}

mkdir -p "$OUT_DIR"

# ${MODELS_ROOT}/Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
# ${MODELS_ROOT}/Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
# ${MODELS_ROOT}/Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
# ${MODELS_ROOT}/Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796/
# ${MODELS_ROOT}/Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
# ${MODELS_ROOT}/Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
# ---------------------------
# Run evaluation
# ---------------------------
# ${SEC5_ROOT}/infer
python ${SEC5_ROOT}/infer/infer_qwen3.py \
  --model ${MODELS_ROOT}/Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --task "$TASK" \
  --split test \
  --n 0 \
  --out_dir "$OUT_DIR" \
  --use_qwen_best_practices \
  --think_mode think


echo "[DONE] Task $TASK finished."
