


export TEMP=/tmp
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-${DATA_ROOT}/para_out/bench_compare_origin_multi}"

python3 ${SEC6_ROOT}/train/scripts/infer/benchmark_compare_origin_multi.py
