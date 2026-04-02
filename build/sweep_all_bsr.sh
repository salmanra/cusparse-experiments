#!/bin/bash

# Start MPS daemon so CUDA_MPS_ACTIVE_THREAD_PERCENTAGE takes effect
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40

# Ensure MPS daemon is stopped on exit (even on error/interrupt)
trap 'echo quit | nvidia-cuda-mps-control' EXIT

echo "=== Running BSR SpMM sweep ==="
bash sweep_bsr.sh "$@"

echo ""
echo "=== Running BSR Ultra-sparse sweep ==="
bash sweep_ultra_bsr.sh "$@"
