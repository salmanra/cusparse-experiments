#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

#include "cuda_utils.h"
#include "cuda_profiling_suite.hpp"

// Run cuSPARSE BSR SpMM for a single profile case.
// Uses cusparseCreateBsr directly — no CSR conversion.
// Returns elapsed time in milliseconds.
float run_spmm(const cuda_bsr_matrix<float>& bsr, const cuda_dense_matrix<float>& B) {
    const int64_t brows = bsr.H / bsr.R;  // number of block rows
    const int64_t bcols = bsr.W / bsr.C;  // number of block cols
    const int64_t bnnz  = bsr.nblocks;    // number of non-zero blocks
    const int64_t m = bsr.H;
    const int64_t k = bsr.W;
    const int64_t n = B.W;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Device memory for BSR arrays
    int* d_bsr_row_offsets = nullptr;
    int* d_bsr_col_ind = nullptr;
    float* d_bsr_values = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_bsr_row_offsets, (brows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bsr_col_ind, bnnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bsr_values, bnnz * bsr.R * bsr.C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_bsr_row_offsets, bsr.indptr.data(),
                          (brows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsr_col_ind, bsr.indices.data(),
                          bnnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsr_values, bsr.data.data(),
                          bnnz * bsr.R * bsr.C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data.data(),
                          k * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(float)));

    // cuSPARSE setup
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create BSR sparse matrix descriptor directly
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateBsr(
        &matA,
        brows, bcols, bnnz,
        bsr.R, bsr.C,               // rowBlockSize, colBlockSize
        d_bsr_row_offsets,
        d_bsr_col_ind,
        d_bsr_values,
        CUSPARSE_INDEX_32I,          // row offsets type
        CUSPARSE_INDEX_32I,          // col indices type
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW           // blocks stored in row-major order
    ));

    // B is row-major: (k x n), leading dimension = n
    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB, k, n, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW
    ));

    // C is row-major: (m x n), leading dimension = n
    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC, m, n, n, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW
    ));

    // Buffer
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize
    ));

    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // Warmup
    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer
    ));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_bsr_row_offsets));
    CHECK_CUDA(cudaFree(d_bsr_col_ind));
    CHECK_CUDA(cudaFree(d_bsr_values));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_buffer));

    return elapsed_ms;
}

template <size_t N>
void run_registry(const char* registry_name,
                  cuda_profiling_suite::ProfileCaseFunctionPtr (&registry)[N]) {
    printf("\n========== %s (%zu cases) ==========\n", registry_name, N);
    for (size_t i = 0; i < N; i++) {
        auto [bsr, dense, test_name] = registry[i]();
        printf("[%zu] %s: M=%zu K=%zu N=%zu R=%zu C=%zu nnz_blocks=%zu ... ",
               i, test_name.c_str(), bsr.H, bsr.W, dense.W, bsr.R, bsr.C, bsr.nblocks);
        fflush(stdout);

        float elapsed_ms = run_spmm(bsr, dense);
        // TODO: print TFLOPS and GB/s
        float flops = 2.0f * bsr.nblocks * bsr.R * bsr.C * dense.W;
        float tflops = flops / 1e12 / (elapsed_ms / 1e3);
        float bytes = (bsr.nblocks * bsr.R * bsr.C + bsr.nblocks * 2) * sizeof(float) +  // values + row_offsets + col_indices
                      (bsr.W * dense.W + bsr.H * dense.W) * sizeof(float);  // B and C
        float gbps = bytes / (elapsed_ms / 1e3) / 1e9;  
        printf("%.3f ms\n", elapsed_ms);
        printf("  TFLOPS: %.3f\n", tflops);
        printf("  GB/s: %.3f\n", gbps);
    }
}

int main(int argc, char* argv[]) {
    int registry_id = -1;  // default: run sanity check only
    if (argc > 1) {
        registry_id = atoi(argv[1]);
    }

    if (registry_id < 0) {
        // Sanity check
        auto [bsr, dense, name] = cuda_profiling_suite::profile_case_sanity_check();
        printf("Sanity check: %s\n", name.c_str());
        float elapsed_ms = run_spmm(bsr, dense);
        printf("  Time: %.3f ms\n", elapsed_ms);
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
