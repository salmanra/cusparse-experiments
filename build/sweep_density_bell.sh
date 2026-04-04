#!/bin/bash

# Start MPS daemon so CUDA_MPS_ACTIVE_THREAD_PERCENTAGE takes effect
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40

# Ensure MPS daemon is stopped on exit (even on error/interrupt)
trap 'echo quit | nvidia-cuda-mps-control' EXIT

DENSITY_CSV="${1:-/home/rsalman/cusparse-experiments/sweep_density_bell.csv}"
ULTRA_CSV="${2:-/home/rsalman/cusparse-experiments/sweep_ultra_bell.csv}"

parse_output() {
    local REG="$1"
    local CSV="$2"
    local OUTPUT="$3"

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

        echo "$REG,$CASE,$M,$K,$N,$BLK,$NNZ,$AVG_MS,$AVG_TFLOPS,$MAX_TFLOPS,$AVG_GBS,$MAX_GBS" >> "$CSV"

    done <<< "$OUTPUT"
}

CSV_HEADER="Registry,Case,M,K,N,Block,NNZ_Blocks,Avg_ms,Avg_TFLOPs,Max_TFLOPs,Avg_GBs,Max_GBs"

# --- Density sweep: registries 9 (SweepDensity 256) and 12 (SweepDensity128) ---
echo "$CSV_HEADER" > "$DENSITY_CSV"
for REG in 9 12; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"
    parse_output "$REG" "$DENSITY_CSV" "$OUTPUT"
done
echo ""
echo "Density results saved to $DENSITY_CSV"

# --- Ultra-low density: registries 10 (UltraLowDensity32) and 11 (UltraLowDensity64) ---
echo "$CSV_HEADER" > "$ULTRA_CSV"
for REG in 10 11; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"
    parse_output "$REG" "$ULTRA_CSV" "$OUTPUT"
done
echo ""
echo "Ultra-low density results saved to $ULTRA_CSV"
