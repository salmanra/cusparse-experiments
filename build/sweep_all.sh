#!/bin/bash

# Start MPS daemon so CUDA_MPS_ACTIVE_THREAD_PERCENTAGE takes effect
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40

# Ensure MPS daemon is stopped on exit (even on error/interrupt)
trap 'echo quit | nvidia-cuda-mps-control' EXIT

echo "=== Running GEMM sweep ==="
bash sweep_gemm.sh "$@"

echo ""
echo "=== Running SpMM sweep ==="
bash sweep.sh "$@"
