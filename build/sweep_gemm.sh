#!/bin/bash

OUTPUT_CSV="${1:-gemm_sweep_results.csv}"
touch "$OUTPUT_CSV"
echo "Registry,Case,M,K,N,Avg_ms,Avg_TFLOPs,Max_TFLOPs,Avg_GBs,Max_GBs" > "$OUTPUT_CSV"

for REG in 0 1 2 3 4; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./gemm_demo $REG)
    echo "$OUTPUT"

    while IFS= read -r line; do
        if ! echo "$line" | grep -qP '^\[\d+\]'; then
            continue
        fi

        header="$line"
        IFS= read -r avg_line
        IFS= read -r max_line

        CASE=$(echo "$header" | sed 's/^\[[0-9]*\] //' | sed 's/:.*//')
        M=$(echo "$header" | grep -oP 'M=\K[0-9]+')
        K=$(echo "$header" | grep -oP 'K=\K[0-9]+')
        N=$(echo "$header" | grep -oP 'N=\K[0-9]+')

        AVG_MS=$(echo "$avg_line" | grep -oP 'Avg time: \K[0-9.]+')
        AVG_TFLOPS=$(echo "$avg_line" | grep -oP 'Avg TFLOPS: \K[0-9.]+')
        AVG_GBS=$(echo "$avg_line" | grep -oP 'Avg GB/s: \K[0-9.]+')

        MAX_TFLOPS=$(echo "$max_line" | grep -oP 'Max TFLOPS: \K[0-9.]+')
        MAX_GBS=$(echo "$max_line" | grep -oP 'Max GB/s: \K[0-9.]+')

        echo "$REG,$CASE,$M,$K,$N,$AVG_MS,$AVG_TFLOPS,$MAX_TFLOPS,$AVG_GBS,$MAX_GBS" >> "$OUTPUT_CSV"

    done <<< "$OUTPUT"
done

echo ""
echo "Results saved to $OUTPUT_CSV"
