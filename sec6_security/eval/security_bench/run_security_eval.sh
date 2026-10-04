#!/usr/bin/env bash
# Security benchmark suite runner: TensorTrust/Purple/Gandalf/RuLES/StruQ/
# IFEval (/NESSiE), mean+/-std over --seeds_to_run seeds.
# Routed through security_bench.cli with an explicit --backend.
#
# Usage: run_security_eval.sh <config_yaml_unused> <PID>
#   PID 0 = Vanilla, PID 1 = Stream

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate parallm

export TEMP="${TEMP:-/path/to/tempdir}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_BASE="${OUT_BASE:-/path/to/security_eval_out}"

# Point these at local checkpoint dirs (e.g. a `huggingface-cli download` of
# guinansu/vanilla-qwen3-4b-security / guinansu/stream-qwen3-4b-security, see
# sec6_security/README.md, or your own trained checkpoints).
VANILLA_MODEL="${VANILLA_MODEL:-/path/to/vanilla_qwen3_4b_security}"
STREAM_MODEL="${STREAM_MODEL:-/path/to/stream_qwen3_4b_security}"

SEEDS_TO_RUN="${SEEDS_TO_RUN:-3}"

PID="${2:-0}"
DATASETS="${3:-tensortrust purple gandalf rules struq ifeval}"
SEED_ARG="${4:-}"
if [[ -n "${SEED_ARG}" ]]; then
  SEED_FLAGS=(--seeds "${SEED_ARG}")
else
  SEED_FLAGS=(--seeds_to_run "${SEEDS_TO_RUN}")
fi

case "${PID}" in
  0) MODEL="${VANILLA_MODEL}"; TAG="vanilla4b"; RUN_PY="${SCRIPT_DIR}/run_vanilla.py" ;;
  1) MODEL="${STREAM_MODEL}"; TAG="stream4b"; RUN_PY="${SCRIPT_DIR}/run_stream.py" ;;
  *) echo "[ERROR] Unknown PID=${PID}"; exit 1 ;;
esac

if [[ ! -d "${MODEL}" ]]; then
  echo "[ERROR] Checkpoint dir for PID=${PID} (${TAG}) not found: ${MODEL}"
  exit 1
fi

OUT_DIR="${OUT_DIR_OVERRIDE:-${OUT_BASE}/${TAG}}"
mkdir -p "${OUT_DIR}"

echo "=================================================="
echo "[SELECT] PID=${PID} -> tag=${TAG} model=${MODEL}"
echo "=================================================="

python3 "${RUN_PY}" \
  --model_path "${MODEL}" \
  --out_dir "${OUT_DIR}" \
  --datasets ${DATASETS} \
  "${SEED_FLAGS[@]}" \
  --do_sample 1 --temperature 0.7 --top_p 1.0 \
  --max_length_per_head 8192 --max_steps 8192 \
  --stop_on_im_end 1

echo "[DONE] Security eval finished: ${TAG}"
