


export TEMP=/tmp
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-${DATA_ROOT}/para_out/bench_origin_generate}"

python3 ${SEC5_ROOT}/train/scripts/infer/benchmark_origin_generate.py
