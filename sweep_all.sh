#!/bin/bash

# Start MPS daemon so CUDA_MPS_ACTIVE_THREAD_PERCENTAGE takes effect
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40

# Ensure MPS daemon is stopped on exit (even on error/interrupt)
trap 'echo quit | nvidia-cuda-mps-control' EXIT

run_sweep() {
    local OUTPUT_CSV="$1"
    shift
    local REGISTRIES=("$@")

    echo "Registry,Case,M,K,N,Block,NNZ_Blocks,Avg_ms,Avg_TFLOPs,Max_TFLOPs,Avg_GBs,Max_GBs" > "$OUTPUT_CSV"

    for REG in "${REGISTRIES[@]}"; do
        echo "Running Registry $REG ..."
        OUTPUT=$(./spmm_bell $REG)
        echo "$OUTPUT"

        while IFS= read -r line; do
            if ! echo "$line" | grep -qP '^\[\d+\]'; then
                continue
            fi
            if echo "$line" | grep -q 'SKIPPED'; then
                continue
            fi

            header="$line"
            IFS= read -r avg_line
            IFS= read -r max_line

            CASE=$(echo "$header" | sed 's/^\[[0-9]*\] //' | sed 's/:.*//')
            M=$(echo "$header" | grep -oP 'M=\K[0-9]+')
            K=$(echo "$header" | grep -oP 'K=\K[0-9]+')
            N=$(echo "$header" | grep -oP 'N=\K[0-9]+')
            BLK=$(echo "$header" | grep -oP 'block=\K[0-9]+')
            NNZ=$(echo "$header" | grep -oP 'nnz_blocks=\K[0-9]+')

            AVG_MS=$(echo "$avg_line" | grep -oP 'Avg time: \K[0-9.]+')
            AVG_TFLOPS=$(echo "$avg_line" | grep -oP 'Avg TFLOPS: \K[0-9.]+')
            AVG_GBS=$(echo "$avg_line" | grep -oP 'Avg GB/s: \K[0-9.]+')

            MAX_TFLOPS=$(echo "$max_line" | grep -oP 'Max TFLOPS: \K[0-9.]+')
            MAX_GBS=$(echo "$max_line" | grep -oP 'Max GB/s: \K[0-9.]+')

            echo "$REG,$CASE,$M,$K,$N,$BLK,$NNZ,$AVG_MS,$AVG_TFLOPS,$MAX_TFLOPS,$AVG_GBS,$MAX_GBS" >> "$OUTPUT_CSV"

        done <<< "$OUTPUT"
    done

    echo "Results saved to $OUTPUT_CSV"
}

echo "=== Running ultra-sparse sweep ==="
run_sweep "ultra_sparse_results.csv" 17 19 21 23 25 27 28

echo ""
echo "=== Running semi-sparse sweep ==="
run_sweep "semi_sparse_results.csv" 2 3 4 5 13 14 15 16
