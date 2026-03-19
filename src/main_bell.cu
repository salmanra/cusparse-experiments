#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#include "cuda_utils.h"
#include "cuda_profiling_suite.hpp"

// Convert a cuda_bsr_matrix<float> to Blocked-ELL format with FP16 values.
//
// Blocked-ELL layout (from cuSPARSE docs):
//   - ellBlockSize: square block dimension (we require R == C)
//   - Each block row is padded to have the same number of column entries
//     (= max blocks in any block row). Missing blocks use col index -1.
//   - ellCols = max_blocks_per_block_row * ellBlockSize
//   - ellColInd: size = num_block_rows * max_blocks_per_block_row
//     One column index per block, -1 for padding.
//   - ellValue: size = rows * ellCols (in __half)
//     Values are stored block-row by block-row. Within each block row,
//     blocks are laid out contiguously (blockSize x blockSize each),
//     padded blocks are zero.
struct BlockedELLData {
    std::vector<int> col_ind;
    std::vector<__half> values;
    int64_t ell_cols;          // = max_blocks_per_block_row * blockSize
    int64_t ell_blocksize;
    int64_t num_blocks;        // total number of column index entries (including padding)
};

BlockedELLData bsr_to_blocked_ell(const cuda_bsr_matrix<float>& bsr) {
    // Blocked ELL requires square blocks
    if (bsr.R != bsr.C) {
        fprintf(stderr, "Blocked ELL requires square blocks (R=%zu != C=%zu)\n", bsr.R, bsr.C);
        exit(EXIT_FAILURE);
    }

    const int64_t blockSize = bsr.R;
    const int64_t brows = bsr.H / blockSize;
    const int64_t bcols = bsr.W / blockSize;

    // Find max blocks per block row
    int64_t max_bpr = 0;
    for (int64_t i = 0; i < brows; i++) {
        int64_t count = bsr.indptr[i + 1] - bsr.indptr[i];
        max_bpr = std::max(max_bpr, count);
    }

    BlockedELLData ell;
    ell.ell_blocksize = blockSize;
    ell.ell_cols = max_bpr * blockSize;
    ell.num_blocks = brows * max_bpr;

    // Allocate col_ind (one per block slot) and values
    ell.col_ind.resize(brows * max_bpr, -1);  // -1 = padding
    ell.values.resize(bsr.H * ell.ell_cols, __float2half(0.0f));

    for (int64_t bi = 0; bi < brows; bi++) {
        int64_t row_start = bsr.indptr[bi];
        int64_t row_end   = bsr.indptr[bi + 1];
        int64_t blocks_in_row = row_end - row_start;

        for (int64_t slot = 0; slot < blocks_in_row; slot++) {
            int64_t bsr_idx = row_start + slot;
            int64_t bj = bsr.indices[bsr_idx];

            // Column index for this block slot
            ell.col_ind[bi * max_bpr + slot] = bj;

            // Copy block values (BSR block is R*C floats in row-major)
            // ELL value layout: for block row bi, slot s, the block occupies
            // rows [bi*blockSize .. (bi+1)*blockSize) and
            // value columns [slot*blockSize .. (slot+1)*blockSize) within the ELL storage.
            // ELL values are stored as: values[row * ellCols + col_within_ell]
            const float* src_block = &bsr.data[bsr_idx * blockSize * blockSize];
            for (int64_t r = 0; r < blockSize; r++) {
                for (int64_t c = 0; c < blockSize; c++) {
                    int64_t ell_row = bi * blockSize + r;
                    int64_t ell_col = slot * blockSize + c;
                    ell.values[ell_row * ell.ell_cols + ell_col] =
                        __float2half(src_block[r * blockSize + c]);
                }
            }
        }
    }

    return ell;
}

static const int NUM_ITERS = 10;

// Run cuSPARSE Blocked-ELL SpMM with FP16 + FP32 compute (Tensor Cores).
// Returns per-iteration elapsed times in milliseconds.
std::vector<float> run_spmm_bell(const cuda_bsr_matrix<float>& bsr, const cuda_dense_matrix<float>& B) {
    auto ell = bsr_to_blocked_ell(bsr);

    const int64_t m = bsr.H;
    const int64_t k = bsr.W;
    const int64_t n = B.W;
    const int64_t ldb = k;   // column-major: leading dim = num rows
    const int64_t ldc = m;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Convert dense B to FP16, column-major
    std::vector<__half> h_B(k * n);
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < k; i++) {
            // B.data is row-major: B[i][j] = B.data[i * W + j]
            h_B[j * ldb + i] = __float2half(B.data[i * n + j]);
        }
    }

    // C output in FP16, column-major
    std::vector<__half> h_C(m * n, __float2half(0.0f));

    // Device memory
    int*    d_ell_col_ind = nullptr;
    __half* d_ell_values  = nullptr;
    __half* d_B = nullptr;
    __half* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_ell_col_ind, ell.num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ell_values, bsr.H * ell.ell_cols * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_ell_col_ind, ell.col_ind.data(),
                          ell.num_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ell_values, ell.values.data(),
                          bsr.H * ell.ell_cols * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), m * n * sizeof(__half), cudaMemcpyHostToDevice));

    // cuSPARSE setup
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create Blocked-ELL sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateBlockedEll(
        &matA,
        m, k,                        // element-level rows, cols
        ell.ell_blocksize,           // square block size
        ell.ell_cols,                // actual ELL columns (max_bpr * blockSize)
        d_ell_col_ind,
        d_ell_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_16F                   // FP16 values
    ));

    // Dense B: column-major, FP16
    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB, k, n, ldb, d_B, CUDA_R_16F, CUSPARSE_ORDER_COL
    ));

    // Dense C: column-major, FP16
    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC, m, n, ldc, d_C, CUDA_R_16F, CUSPARSE_ORDER_COL
    ));

    // Buffer size
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F,                          // FP32 compute -> Tensor Cores
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1,      // Blocked-ELL specific algorithm
        &bufferSize
    ));

    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // Warmup
    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
        d_buffer
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> iter_times(NUM_ITERS);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUSPARSE(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F,
            CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
            d_buffer
        ));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        iter_times[iter] = elapsed_ms;
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_ell_col_ind));
    CHECK_CUDA(cudaFree(d_ell_values));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_buffer));

    return iter_times;
}

template <size_t N>
void run_registry(const char* registry_name,
                  cuda_profiling_suite::ProfileCaseFunctionPtr (&registry)[N]) {
    printf("\n========== %s (%zu cases) ==========\n", registry_name, N);
    for (size_t i = 0; i < N; i++) {
        auto [bsr, dense, test_name] = registry[i]();

        if (bsr.R != bsr.C) {
            printf("[%zu] %s: SKIPPED (non-square blocks R=%zu C=%zu)\n",
                   i, test_name.c_str(), bsr.R, bsr.C);
            continue;
        }

        printf("[%zu] %s: M=%zu K=%zu N=%zu block=%zu nnz_blocks=%zu (%d iters)\n",
               i, test_name.c_str(), bsr.H, bsr.W, dense.W, bsr.R, bsr.nblocks, NUM_ITERS);
        fflush(stdout);

        std::vector<float> iter_times = run_spmm_bell(bsr, dense);

        float flops = 2.0f * bsr.nblocks * bsr.R * bsr.C * dense.W;
        float bytes = (bsr.nblocks * bsr.R * bsr.C) * sizeof(__half) +
                      (bsr.nblocks) * sizeof(int) +
                      (bsr.W * dense.W + bsr.H * dense.W) * sizeof(__half);

        float sum_ms = 0.0f, min_ms = iter_times[0];
        for (int j = 0; j < NUM_ITERS; j++) {
            sum_ms += iter_times[j];
            if (iter_times[j] < min_ms) min_ms = iter_times[j];
        }
        float avg_ms = sum_ms / NUM_ITERS;

        float avg_tflops = flops / 1e12 / (avg_ms / 1e3);
        float max_tflops = flops / 1e12 / (min_ms / 1e3);
        float avg_gbps   = bytes / (avg_ms / 1e3) / 1e9;
        float max_gbps   = bytes / (min_ms / 1e3) / 1e9;

        printf("  Avg time: %.3f ms  |  Avg TFLOPS: %.3f  |  Avg GB/s: %.3f\n",
               avg_ms, avg_tflops, avg_gbps);
        printf("  Min time: %.3f ms  |  Max TFLOPS: %.3f  |  Max GB/s: %.3f\n",
               min_ms, max_tflops, max_gbps);
    }
}

int main(int argc, char* argv[]) {
    // Check compute capability for tensor core support
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s (compute %d.%d)\n", props.name, props.major, props.minor);
    if (props.major < 7) {
        fprintf(stderr, "Blocked-ELL SpMM requires compute capability >= 7.0\n");
        return EXIT_FAILURE;
    }
    if (props.major >= 7)
        printf("Tensor Core FP16 SpMM: ENABLED\n");

    int registry_id = -1;
    if (argc > 1) {
        registry_id = atoi(argv[1]);
    }

    if (registry_id < 0) {
        auto [bsr, dense, name] = cuda_profiling_suite::profile_case_sanity_check();
        printf("\nSanity check: %s\n", name.c_str());
        std::vector<float> times = run_spmm_bell(bsr, dense);
        float sum = 0.0f; for (auto t : times) sum += t;
        printf("  Avg time: %.3f ms (%d iters)\n", sum / NUM_ITERS, NUM_ITERS);
        printf("\nUsage: %s <registry_id>\n", argv[0]);
        printf("  0 = Small sparse cases\n");
        printf("  1 = Dense ablation\n");
        printf("  2 = Large sparse cases\n");
        printf("  4 = Sweep N\n");
        printf("  5 = Sweep density\n");
        printf("  6 = Sweep K\n");
        printf("  7 = Sweep block size\n");
        printf("  8 = Sweep sparsity pattern (d=25%%)\n");
        printf("  9 = Sweep sparsity pattern (d=10%%)\n");
        printf(" 10 = Sweep sparsity pattern (d=5%%)\n");
        printf(" 11 = Sweep sparsity pattern (d=50%%)\n");
        printf(" 12 = Large sparse, large blocks\n");
        return 0;
    }

    switch (registry_id) {
        case 0:
            run_registry("ProfileCaseRegistry",
                         cuda_profiling_suite::ProfileCaseRegistry);
            break;
        case 1:
            run_registry("ProfileDenseAblationRegistry",
                         cuda_profiling_suite::ProfileDenseAblationRegistry);
            break;
        case 2:
            run_registry("ProfileLargeSparseRegistry",
                         cuda_profiling_suite::ProfileLargeSparseRegistry);
            break;
        case 4:
            run_registry("ProfileSweepNRegistry",
                         cuda_profiling_suite::ProfileSweepNRegistry);
            break;
        case 5:
            run_registry("ProfileSweepDensityRegistry",
                         cuda_profiling_suite::ProfileSweepDensityRegistry);
            break;
        case 6:
            run_registry("ProfileSweepKRegistry",
                         cuda_profiling_suite::ProfileSweepKRegistry);
            break;
        case 7:
            run_registry("ProfileSweepBlockSizeRegistry",
                         cuda_profiling_suite::ProfileSweepBlockSizeRegistry);
            break;
        case 8:
            run_registry("ProfileSweepSparsityPatternRegistry",
                         cuda_profiling_suite::ProfileSweepSparsityPatternRegistry);
            break;
        case 9:
            run_registry("ProfileSweepSparsityPatternRegistryD10",
                         cuda_profiling_suite::ProfileSweepSparsityPatternRegistryD10);
            break;
        case 10:
            run_registry("ProfileSweepSparsityPatternRegistryD5",
                         cuda_profiling_suite::ProfileSweepSparsityPatternRegistryD5);
            break;
        case 11:
            run_registry("ProfileSweepSparsityPatternRegistryD50",
                         cuda_profiling_suite::ProfileSweepSparsityPatternRegistryD50);
            break;
        case 12:
            run_registry("ProfileLargeSparseLargeBlocksRegistry",
                         cuda_profiling_suite::ProfileLargeSparseLargeBlocksRegistry);
            break;
        default:
            fprintf(stderr, "Unknown registry ID: %d\n", registry_id);
            return 1;
    }

    printf("\nDone.\n");
    return 0;
}
