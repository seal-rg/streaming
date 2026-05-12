



export SFT_MODEL_PATH=${RESULTS_ROOT}/p_20260226_231512_135450742
export MULTI_MODEL_PATH=${RESULTS_ROOT}/p_20260227_112501_210125262
# export BENCH_USE_CHAT_TEMPLATE="${BENCH_USE_CHAT_TEMPLATE:-0}"
export TEMP=/tmp
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-${DATA_ROOT}/para_out/bench_medusa_e2e}"

python3 ${SEC6_ROOT}/train/scripts/infer/benchmark_medusa_e2e.py
