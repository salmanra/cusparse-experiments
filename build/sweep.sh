#!/bin/bash

OUTPUT_CSV="${1:-/home/rsalman/cusparse-experiments/sweep_results.csv}"
touch "$OUTPUT_CSV"
echo "N,Avg_TFLOPs,Max_TFLOPs,Avg_GBs,Max_GBs" > "$OUTPUT_CSV"

for N in 8 9 10 11; do
    echo "Running Registry $N ..."
    OUTPUT=$(./spmm_bell $N)
    echo "$OUTPUT" | grep -E "TFLOPS|GB/s"

    AVG_TFLOPS=$(echo "$OUTPUT" | grep "Avg TFLOPS" | awk '{print $NF}')
    MAX_TFLOPS=$(echo "$OUTPUT" | grep "Max TFLOPS" | awk '{print $NF}')
    AVG_GBS=$(echo "$OUTPUT" | grep "Avg GB/s" | awk '{print $NF}')
    MAX_GBS=$(echo "$OUTPUT" | grep "Max GB/s" | awk '{print $NF}')

    echo "$N,$AVG_TFLOPS,$MAX_TFLOPS,$AVG_GBS,$MAX_GBS" >> "$OUTPUT_CSV"
done

echo ""
echo "Results saved to $OUTPUT_CSV"Results saved to cusparse-experiments/sweep_results.csv
