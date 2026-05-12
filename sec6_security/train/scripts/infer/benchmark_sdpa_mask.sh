


export TEMP=/tmp
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-${DATA_ROOT}/para_out/bench_sdpa_mask}"

python3 ${SEC6_ROOT}/train/scripts/infer/benchmark_sdpa_mask.py
