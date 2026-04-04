#!/bin/bash

# Start MPS daemon so CUDA_MPS_ACTIVE_THREAD_PERCENTAGE takes effect
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40

# Ensure MPS daemon is stopped on exit (even on error/interrupt)
trap 'echo quit | nvidia-cuda-mps-control' EXIT

OUTDIR="${1:-/home/rsalman/cusparse-experiments}"

CSV_32="$OUTDIR/sweep_pattern_32.csv"
CSV_64="$OUTDIR/sweep_pattern_64.csv"
CSV_128="$OUTDIR/sweep_pattern_128.csv"
CSV_256="$OUTDIR/sweep_pattern_256.csv"

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

# --- 32x32 patterns (registries 17-22) ---
echo "$CSV_HEADER" > "$CSV_32"
for REG in 17 18 19 20 21 22; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"
    parse_output "$REG" "$CSV_32" "$OUTPUT"
done
echo "Results saved to $CSV_32"
echo ""

# --- 64x64 patterns (registries 23-28) ---
echo "$CSV_HEADER" > "$CSV_64"
for REG in 23 24 25 26 27 28; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"
    parse_output "$REG" "$CSV_64" "$OUTPUT"
done
echo "Results saved to $CSV_64"
echo ""

# --- 128x128 patterns (registries 13-16) ---
echo "$CSV_HEADER" > "$CSV_128"
for REG in 13 14 15 16; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"
    parse_output "$REG" "$CSV_128" "$OUTPUT"
done
echo "Results saved to $CSV_128"
echo ""

# --- 256x256 patterns (registries 2-5) ---
echo "$CSV_HEADER" > "$CSV_256"
for REG in 2 3 4 5; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"
    parse_output "$REG" "$CSV_256" "$OUTPUT"
done
echo "Results saved to $CSV_256"
