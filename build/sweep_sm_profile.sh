#!/bin/bash
# Sweep SM utilization profiling for Blocked-ELL SpMM.
# Outputs CSV with kernel launch configs and active SM counts.
# Uses CUPTI (no sudo required, no hardware counters).

OUTPUT_CSV="${1:-/home/rsalman/cusparse-experiments/sweep_sm_profile.csv}"

echo "Registry,Case,M,K,N,Block,NNZ_Blocks,Kernel_Name,Grid_X,Grid_Y,Grid_Z,Total_Blocks,Block_X,Block_Y,Block_Z,Threads_Per_Block,Active_SMs,SM_Util_Pct,Static_Shmem,Dynamic_Shmem,Regs_Per_Thread" > "$OUTPUT_CSV"

for REG in 8 9 10 11; do
    echo "Profiling Registry $REG ..." >&2
    # stdout = CSV data (skip header line and GPU info line), stderr = progress
    ./spmm_bell_sm_profile $REG 2>/dev/null | tail -n +3 >> "$OUTPUT_CSV"
done

echo "" >&2
echo "Results saved to $OUTPUT_CSV" >&2
