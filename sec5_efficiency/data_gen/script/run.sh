#!/bin/bash

# Activate conda environment

conda activate moetest

# Parse job ID from arguments
JOB_ID=$1

# Configuration 817
TOTAL_SAMPLES=1000
NUM_JOBS=10  # Adjust based on your needs
SAMPLES_PER_JOB=$((TOTAL_SAMPLES / NUM_JOBS))
REMAINDER=$((TOTAL_SAMPLES % NUM_JOBS))

# Calculate start and end indices for this job
START_IDX=$((JOB_ID * SAMPLES_PER_JOB))

# Handle remainder for last job         §   §§
if [ $JOB_ID -eq $((NUM_JOBS - 1)) ]; then
    END_IDX=$((START_IDX + SAMPLES_PER_JOB + REMAINDER))
else
    END_IDX=$((START_IDX + SAMPLES_PER_JOB))
fi

# Output directory for this job
OUTPUT_DIR="${RESULTS_ROOT}/s11_1017_3agent${JOB_ID}"

echo "Job ${JOB_ID}: Processing samples ${START_IDX} to ${END_IDX}"
echo "Output directory: ${OUTPUT_DIR}"

# ${WORKSPACE_ROOT}/hogwild_llm/gen_data_hymath.py
# Run the Python script ${WORKSPACE_ROOT}/hogwild_llm/gen_data_multi.py
# ${WORKSPACE_ROOT}/hogwild_llm/gen_data5.py
python3 ${SEC5_ROOT}/data_gen/gen_data_multi.py \
    --output_dir "${OUTPUT_DIR}" \
    --start "${START_IDX}" \§
    --end "${END_IDX}"