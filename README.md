# cuSPARSE Blocked-ELL SpMM Benchmark

Benchmarks `cusparseSpMM` with the Blocked-ELL sparse format (FP16 values,
FP32 compute, Tensor Cores) across a suite of synthetic block-sparse matrices
of varying block size, density, and sparsity pattern.

This repository is the artifact for the paper *[paper title]* (Supercomputing
[year]). All results reported in the paper were produced by the scripts and
source in this repository.

## Repository layout

```
CMakeLists.txt                  Build definition
src/main_bell.cu                Benchmark driver (one executable: spmm_bell)
include/bsr_matrix.hpp          Host-side block-sparse-row matrix type
include/cuda_bsr_matrix.hpp     Device-side BSR helper
include/cuda_profiling_suite.hpp Case registries (M/K/N/block/density sweeps)
include/cuda_utils.h            CUDA / cuSPARSE error-check macros
sweep_all.sh                    Driver script that runs two registry groups
                                and writes CSV results
results/ultra_sparse_results.csv  Sweep output: ultra-sparse cases
results/semi_sparse_results.csv   Sweep output: semi-sparse cases
hardware.txt                    Full environment snapshot for the committed
                                results
clean.sh                        Removes CMake build artifacts
LICENSE                         MIT
```

> **Note on repository state.** Earlier commits tracked the CMake `build/`
> directory; it has since been removed from version control. If you are
> looking at an older tag, delete `build/` before rebuilding.

## Requirements

| Component          | Tested version                    |
|--------------------|-----------------------------------|
| GPU                | NVIDIA RTX 4090 (compute 8.9, Ada) |
| CUDA Toolkit       | 12.5 (nvcc 12.5.82)               |
| NVIDIA driver      | 555.42.02                         |
| CMake              | ≥ 3.18 (tested 3.22.1)            |
| C++ standard       | C++17                             |
| OS                 | Ubuntu 22.04.5 LTS, kernel 6.8    |

`CMakeLists.txt` compiles for `sm_70 sm_75 sm_80 sm_86 sm_89 sm_90`. Blocked-ELL
SpMM requires compute capability ≥ 7.0. The published results target `sm_89`.

Full environment details are in [hardware.txt](hardware.txt).

## Build

```
mkdir -p build && cd build
cmake ..
make -j
```

Produces `build/spmm_bell`.

## Running the benchmark

### Single registry

```
./build/spmm_bell <registry_id>
```

Invoking `./spmm_bell` without an argument prints a sanity check and lists all
registries. Each case runs 1 warmup iteration plus 10 timed iterations with
CUDA events and reports `Avg_ms`, `Avg_TFLOPs`, `Max_TFLOPs`, `Avg_GBs`,
`Max_GBs`.

### Full sweep

```
bash sweep_all.sh
```

This is what produced the committed CSVs. It:

1. Starts the CUDA MPS daemon (`nvidia-cuda-mps-control -d`).
2. Exports `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40`.
3. Runs the two registry groups (below) and writes CSVs to `results/`.
4. Shuts the MPS daemon down on exit.

Run it from the repo root. The script looks for the binary at
`./build/spmm_bell` by default; override with `SPMM_BIN=/path/to/spmm_bell
bash sweep_all.sh` if your build tree differs.

## About `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40`

**This is a deliberate experimental parameter, not a mistake.** The RTX 4090
has 128 SMs; setting the MPS active thread percentage to 40% restricts each
CUDA client to ~51 SMs, approximating a smaller GPU and matching the
throughput profile studied in the paper.

To reproduce our numbers exactly, keep this setting. To sweep against the
full GPU, remove the two lines at the top of `sweep_all.sh` (the `mps-control
-d` line and the `export` line) and the `trap` line.

MPS requires that the GPU be in a compute mode compatible with the daemon;
on most single-user workstations the default `DEFAULT` mode works. See
[NVIDIA MPS docs](https://docs.nvidia.com/deploy/mps/index.html) for system
prerequisites.

## Registry → test-case mapping

`sweep_all.sh` runs two groups of registries. Full registry names come from
`include/cuda_profiling_suite.hpp`.

### Ultra-sparse group → `results/ultra_sparse_results.csv`

| Registry ID | Name                  | Block size | Density class |
|-------------|-----------------------|------------|---------------|
| 17          | PatternUltra32_30     | 32         | Very sparse   |
| 19          | PatternUltra32_300    | 32         | Very sparse   |
| 21          | PatternUltra32_3000   | 32         | Ultra sparse  |
| 23          | PatternUltra64_60     | 64         | Very sparse   |
| 25          | PatternUltra64_600    | 64         | Very sparse   |
| 27          | PatternUltra64_6000   | 64         | Ultra sparse  |
| 28          | PatternUltra64_10000  | 64         | Ultra sparse  |

### Semi-sparse group → `results/semi_sparse_results.csv`

| Registry ID | Name            | Block size | Density class    |
|-------------|-----------------|------------|------------------|
| 2           | PatternD5       | 32         | ~5% block-dense  |
| 3           | PatternD10      | 32         | ~10% block-dense |
| 4           | PatternD25      | 32         | ~25% block-dense |
| 5           | PatternD50      | 32         | ~50% block-dense |
| 13          | PatternD5_128   | 128        | ~5% block-dense  |
| 14          | PatternD10_128  | 128        | ~10% block-dense |
| 15          | PatternD25_128  | 128        | ~25% block-dense |
| 16          | PatternD50_128  | 128        | ~50% block-dense |

The full registry list (29 registries in total, including microbenchmarks
and single-axis sweeps that are not part of the paper) is printed by
`./spmm_bell` with no arguments.

## Output CSV schema

Columns in both `*_results.csv` files:

| Column       | Meaning                                                    |
|--------------|------------------------------------------------------------|
| `Registry`   | Integer registry ID                                        |
| `Case`       | Case name from the registry                                |
| `M`, `K`, `N`| SpMM dimensions: C[M×N] = A[M×K] · B[K×N]                  |
| `Block`      | Blocked-ELL block size                                     |
| `NNZ_Blocks` | Non-zero blocks in A                                       |
| `Avg_ms`     | Mean per-iteration wall time (10 timed iterations)         |
| `Avg_TFLOPs` | `2·nnz_blocks·block²·N / (Avg_ms·10⁻³) / 10¹²`             |
| `Max_TFLOPs` | Same formula using the minimum iteration time              |
| `Avg_GBs`    | Effective bandwidth using mean time                        |
| `Max_GBs`    | Effective bandwidth using minimum time                     |

FLOP / byte formulas are defined in `src/main_bell.cu::run_registry`.

## Reproducing the paper numbers

On a single RTX 4090 with the versions listed above, the full
`sweep_all.sh` run takes approximately **3 minutes 10 seconds** (≈ 190 s
of GPU wall time); the CMake configure plus `make -j` build takes
about 6 seconds. Results should match the committed CSVs to within
natural CUDA-event timing jitter — in our clean-clone reproduction the
per-case deltas were in the third significant figure (sub-percent on
average TFLOPS; `Max_TFLOPs` is more stable).

Per-case granularity:

- Most ultra-sparse cases complete in < 1 s.
- Semi-sparse cases with 256×256 blocks at 50% density are the longest
  at roughly 17 ms per iteration × 10 iterations.

## Troubleshooting

- **`cusparseCreateBlockedEll: CUSPARSE_STATUS_NOT_SUPPORTED`** — compute
  capability too low. Blocked-ELL requires `sm_70`+.
- **MPS daemon fails to start** — an MPS daemon may already be running, or
  the GPU may be in a compute mode MPS does not accept. Run `echo quit |
  nvidia-cuda-mps-control` to stop any stale daemon.
- **Out of memory on the 256×256 / 50%-dense cases** — these allocate
  several hundred MB of FP16 values; ensure no other CUDA processes are
  holding device memory.

## Citation

If you use this artifact, please cite:

```
[BibTeX entry for the paper]
```

Archived release (Zenodo DOI): `[insert DOI after minting]`

## License

MIT — see [LICENSE](LICENSE).
