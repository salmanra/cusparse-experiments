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
#include "kernel_launch_tracker.hpp"

// ---- Blocked-ELL conversion (same as main_bell.cu) ----

struct BlockedELLData {
    std::vector<int> col_ind;
    std::vector<__half> values;
    int64_t ell_cols;
    int64_t ell_blocksize;
    int64_t num_blocks;
};

BlockedELLData bsr_to_blocked_ell(const cuda_bsr_matrix<float>& bsr) {
    if (bsr.R != bsr.C) {
        fprintf(stderr, "Blocked ELL requires square blocks (R=%zu != C=%zu)\n", bsr.R, bsr.C);
        exit(EXIT_FAILURE);
    }

    const int64_t blockSize = bsr.R;
    const int64_t brows = bsr.H / blockSize;

    int64_t max_bpr = 0;
    for (int64_t i = 0; i < brows; i++) {
        int64_t count = bsr.indptr[i + 1] - bsr.indptr[i];
        max_bpr = std::max(max_bpr, count);
    }

    BlockedELLData ell;
    ell.ell_blocksize = blockSize;
    ell.ell_cols = max_bpr * blockSize;
    ell.num_blocks = brows * max_bpr;

    ell.col_ind.resize(brows * max_bpr, -1);
    ell.values.resize(bsr.H * ell.ell_cols, __float2half(0.0f));

    for (int64_t bi = 0; bi < brows; bi++) {
        int64_t row_start = bsr.indptr[bi];
        int64_t row_end   = bsr.indptr[bi + 1];
        int64_t blocks_in_row = row_end - row_start;

        for (int64_t slot = 0; slot < blocks_in_row; slot++) {
            int64_t bsr_idx = row_start + slot;
            int64_t bj = bsr.indices[bsr_idx];
            ell.col_ind[bi * max_bpr + slot] = bj;

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

// ---- Run single SpMM under CUPTI tracing ----

std::vector<KernelLaunch> run_spmm_bell_profiled(
    const cuda_bsr_matrix<float>& bsr, const cuda_dense_matrix<float>& B) {

    auto ell = bsr_to_blocked_ell(bsr);

    const int64_t m = bsr.H;
    const int64_t k = bsr.W;
    const int64_t n = B.W;
    const int64_t ldb = k;
    const int64_t ldc = m;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Convert dense B to FP16, column-major
    std::vector<__half> h_B(k * n);
    for (int64_t j = 0; j < n; j++)
        for (int64_t i = 0; i < k; i++)
            h_B[j * ldb + i] = __float2half(B.data[i * n + j]);

    std::vector<__half> h_C(m * n, __float2half(0.0f));

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

    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateBlockedEll(
        &matA, m, k, ell.ell_blocksize, ell.ell_cols,
        d_ell_col_ind, d_ell_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F
    ));

    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, k, n, ldb, d_B, CUDA_R_16F, CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, m, n, ldc, d_C, CUDA_R_16F, CUSPARSE_ORDER_COL));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bufferSize
    ));

    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // Warmup (untracked)
    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, d_buffer
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Tracked run: single iteration under CUPTI
    KernelLaunchTracker tracker;
    tracker.start();

    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_BLOCKED_ELL_ALG1, d_buffer
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto launches = tracker.stop();

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA(cudaFree(d_ell_col_ind));
    CHECK_CUDA(cudaFree(d_ell_values));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_buffer));

    return launches;
}

// ---- Main ----

int main(int argc, char* argv[]) {
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    int total_sms = props.multiProcessorCount;

    printf("GPU: %s (compute %d.%d, %d SMs)\n",
           props.name, props.major, props.minor, total_sms);
    if (props.major < 7) {
        fprintf(stderr, "Blocked-ELL SpMM requires compute capability >= 7.0\n");
        return EXIT_FAILURE;
    }

    int registry_id = -1;
    if (argc > 1) registry_id = atoi(argv[1]);

    if (registry_id < 0 || registry_id >= cuda_profiling_suite::NUM_REGISTRIES) {
        fprintf(stderr, "Usage: %s <registry_id>\n", argv[0]);
        for (int i = 0; i < cuda_profiling_suite::NUM_REGISTRIES; i++)
            fprintf(stderr, "  %2d = %s (%d cases)\n", i,
                    cuda_profiling_suite::RegistryNames[i],
                    cuda_profiling_suite::RegistrySizes[i]);
        return 1;
    }

    auto* registry = cuda_profiling_suite::Registries[registry_id];
    int N = cuda_profiling_suite::RegistrySizes[registry_id];

    // CSV header
    printf("Registry,Case,M,K,N,Block,NNZ_Blocks,Kernel_Name,Grid_X,Grid_Y,Grid_Z,Total_Blocks,"
           "Block_X,Block_Y,Block_Z,Threads_Per_Block,Active_SMs,SM_Util_Pct,"
           "Static_Shmem,Dynamic_Shmem,Regs_Per_Thread\n");

    for (int i = 0; i < N; i++) {
        auto [bsr, dense, test_name] = registry[i]();

        if (bsr.R != bsr.C) {
            fprintf(stderr, "[%d] %s: SKIPPED (non-square blocks)\n", i, test_name.c_str());
            continue;
        }

        // Compute ELL padding stats
        int64_t blockSize = bsr.R;
        int64_t brows = bsr.H / blockSize;
        int64_t max_bpr = 0, total_ell_blocks = 0;
        for (int64_t r = 0; r < brows; r++) {
            int64_t count = bsr.indptr[r + 1] - bsr.indptr[r];
            max_bpr = std::max(max_bpr, count);
        }
        total_ell_blocks = brows * max_bpr;
        int64_t padding_blocks = total_ell_blocks - (int64_t)bsr.nblocks;
        double padding_pct = 100.0 * padding_blocks / std::max(total_ell_blocks, (int64_t)1);

        fprintf(stderr, "[%d/%d] %s  M=%zu K=%zu N=%zu blk=%zu nnz=%zu  "
                "max_bpr=%ld  ell_blocks=%ld  padding=%.1f%%\n",
                i, N, test_name.c_str(), bsr.H, bsr.W, dense.W, bsr.R, bsr.nblocks,
                max_bpr, total_ell_blocks, padding_pct);

        auto launches = run_spmm_bell_profiled(bsr, dense);

        for (const auto& kl : launches) {
            int64_t total_blocks = kl.totalBlocks();
            int active_sms = (int)std::min((int64_t)total_sms, total_blocks);
            double sm_util_pct = 100.0 * active_sms / total_sms;

            printf("%d,%s,%zu,%zu,%zu,%zu,%zu,%s,%d,%d,%d,%ld,%d,%d,%d,%ld,%d,%.1f,%d,%d,%d\n",
                   registry_id, test_name.c_str(),
                   bsr.H, bsr.W, dense.W, bsr.R, bsr.nblocks,
                   kl.name.c_str(),
                   kl.gridX, kl.gridY, kl.gridZ, total_blocks,
                   kl.blockX, kl.blockY, kl.blockZ, kl.threadsPerBlock(),
                   active_sms, sm_util_pct,
                   kl.staticSharedMem, kl.dynamicSharedMem, kl.registersPerThread);
        }
    }

    fprintf(stderr, "\nDone.\n");
    return 0;
}
