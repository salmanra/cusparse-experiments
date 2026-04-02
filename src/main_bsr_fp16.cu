#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "cuda_utils.h"
#include "cuda_profiling_suite.hpp"

static const int NUM_ITERS = 10;

// Run cuSPARSE BSR SpMM with FP16 values + FP32 compute (Tensor Cores).
// This mirrors run_spmm_bell but uses BSR format instead of Blocked-ELL.
// Returns per-iteration elapsed times in milliseconds.
std::vector<float> run_spmm_bsr(const cuda_bsr_matrix<float>& bsr, const cuda_dense_matrix<float>& B) {
    const int64_t brows = bsr.H / bsr.R;
    const int64_t bcols = bsr.W / bsr.C;
    const int64_t bnnz  = bsr.nblocks;
    const int64_t m = bsr.H;
    const int64_t k = bsr.W;
    const int64_t n = B.W;
    const int64_t ldb = k;   // column-major: leading dim = num rows
    const int64_t ldc = m;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Convert BSR values to FP16
    std::vector<__half> h_bsr_values(bnnz * bsr.R * bsr.C);
    for (int64_t i = 0; i < (int64_t)(bnnz * bsr.R * bsr.C); i++) {
        h_bsr_values[i] = __float2half(bsr.data[i]);
    }

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
    int*    d_bsr_row_offsets = nullptr;
    int*    d_bsr_col_ind     = nullptr;
    __half* d_bsr_values      = nullptr;
    __half* d_B = nullptr;
    __half* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_bsr_row_offsets, (brows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bsr_col_ind, bnnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_bsr_values, bnnz * bsr.R * bsr.C * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_bsr_row_offsets, bsr.indptr.data(),
                          (brows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsr_col_ind, bsr.indices.data(),
                          bnnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsr_values, h_bsr_values.data(),
                          bnnz * bsr.R * bsr.C * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), m * n * sizeof(__half), cudaMemcpyHostToDevice));

    // cuSPARSE setup
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create BSR sparse matrix descriptor with FP16 values
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
        CUDA_R_16F,                  // FP16 values
        CUSPARSE_ORDER_COLUMN           // blocks stored in column-major order
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
        CUSPARSE_SPMM_ALG_DEFAULT,
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
        CUSPARSE_SPMM_ALG_DEFAULT,
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
            CUSPARSE_SPMM_ALG_DEFAULT,
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

    CHECK_CUDA(cudaFree(d_bsr_row_offsets));
    CHECK_CUDA(cudaFree(d_bsr_col_ind));
    CHECK_CUDA(cudaFree(d_bsr_values));
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

        printf("[%zu] %s: M=%zu K=%zu N=%zu R=%zu C=%zu nnz_blocks=%zu (%d iters)\n",
               i, test_name.c_str(), bsr.H, bsr.W, dense.W, bsr.R, bsr.C, bsr.nblocks, NUM_ITERS);
        fflush(stdout);

        std::vector<float> iter_times = run_spmm_bsr(bsr, dense);

        float flops = 2.0f * bsr.nblocks * bsr.R * bsr.C * dense.W;
        float bytes = (bsr.nblocks * bsr.R * bsr.C) * sizeof(__half) +
                      (bsr.nblocks + bsr.H / bsr.R + 1) * sizeof(int) +
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
        fprintf(stderr, "BSR FP16 SpMM requires compute capability >= 7.0\n");
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
        std::vector<float> times = run_spmm_bsr(bsr, dense);
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
        printf(" 13 = Ultra-low density (block=32)\n");
        printf(" 14 = Ultra-low density (block=64)\n");
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
        case 13:
            run_registry("ProfileSweepUltraLowDensity32Registry",
                         cuda_profiling_suite::ProfileSweepUltraLowDensity32Registry);
            break;
        case 14:
            run_registry("ProfileSweepUltraLowDensity64Registry",
                         cuda_profiling_suite::ProfileSweepUltraLowDensity64Registry);
            break;
        default:
            fprintf(stderr, "Unknown registry ID: %d\n", registry_id);
            return 1;
    }

    printf("\nDone.\n");
    return 0;
}
