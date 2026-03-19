#!/bin/bash

OUTPUT_CSV="${1:-/home/rsalman/cusparse-experiments/sweep_results.csv}"
touch "$OUTPUT_CSV"
echo "Registry,Case,M,K,N,Block,NNZ_Blocks,Avg_ms,Avg_TFLOPs,Max_TFLOPs,Avg_GBs,Max_GBs" > "$OUTPUT_CSV"

for REG in 8 9 10 11; do
    echo "Running Registry $REG ..."
    OUTPUT=$(./spmm_bell $REG)
    echo "$OUTPUT"

    # Process output line by line: look for case headers, then grab next two lines
    while IFS= read -r line; do
        # Skip non-header lines
        if ! echo "$line" | grep -qP '^\[\d+\]'; then
            continue
        fi
        # Skip cases that were SKIPPED
        if echo "$line" | grep -q 'SKIPPED'; then
            continue
        fi

        header="$line"
        IFS= read -r avg_line
        IFS= read -r max_line

        # Parse header: [idx] name: M=... K=... N=... block=... nnz_blocks=... (10 iters)
        CASE=$(echo "$header" | sed 's/^\[[0-9]*\] //' | sed 's/:.*//')
        M=$(echo "$header" | grep -oP 'M=\K[0-9]+')
        K=$(echo "$header" | grep -oP 'K=\K[0-9]+')
        N=$(echo "$header" | grep -oP 'N=\K[0-9]+')
        BLK=$(echo "$header" | grep -oP 'block=\K[0-9]+')
        NNZ=$(echo "$header" | grep -oP 'nnz_blocks=\K[0-9]+')

        # Parse avg line
        AVG_MS=$(echo "$avg_line" | grep -oP 'Avg time: \K[0-9.]+')
        AVG_TFLOPS=$(echo "$avg_line" | grep -oP 'Avg TFLOPS: \K[0-9.]+')
        AVG_GBS=$(echo "$avg_line" | grep -oP 'Avg GB/s: \K[0-9.]+')

        # Parse max line
        MAX_TFLOPS=$(echo "$max_line" | grep -oP 'Max TFLOPS: \K[0-9.]+')
        MAX_GBS=$(echo "$max_line" | grep -oP 'Max GB/s: \K[0-9.]+')

        echo "$REG,$CASE,$M,$K,$N,$BLK,$NNZ,$AVG_MS,$AVG_TFLOPS,$MAX_TFLOPS,$AVG_GBS,$MAX_GBS" >> "$OUTPUT_CSV"

    done <<< "$OUTPUT"
done

echo ""
echo "Results saved to $OUTPUT_CSV"
