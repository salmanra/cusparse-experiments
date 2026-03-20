#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

#include "cuda_utils.h"

static const int NUM_ITERS = 10;

struct GemmCase {
    int M, N, K;
    std::string name;
};

// Run cuBLAS HGEMM: C = alpha * A * B + beta * C
// A is (M x K), B is (K x N), C is (M x N), all FP16
// Uses Tensor Cores by default on sm_70+ via cublasGemmEx with CUBLAS_COMPUTE_32F.
// Returns per-iteration elapsed times in milliseconds.
std::vector<float> run_gemm(cublasHandle_t handle, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    // Host allocation and random init (generate as float, convert to half)
    std::vector<__half> h_A(size_A), h_B(size_B);
    for (size_t i = 0; i < size_A; i++) h_A[i] = __float2half((float)rand() / RAND_MAX);
    for (size_t i = 0; i < size_B; i++) h_B[i] = __float2half((float)rand() / RAND_MAX);

    // Device memory
    __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, size_C * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(__half)));

    // cuBLAS uses column-major. For row-major A(M,K) * B(K,N) = C(M,N),
    // we compute: C^T = B^T * A^T using column-major layout.
    // cublasGemmEx with CUDA_R_16F inputs and CUBLAS_COMPUTE_32F accumulates in FP32 using Tensor Cores.

    // Warmup
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_16F, N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> iter_times(NUM_ITERS);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_16F, N,
                                  d_A, CUDA_R_16F, K,
                                  &beta,
                                  d_C, CUDA_R_16F, N,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        iter_times[iter] = elapsed_ms;
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return iter_times;
}

void run_cases(const char* registry_name, const std::vector<GemmCase>& cases) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Enable Tensor Core math
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    printf("\n========== %s (%zu cases) ==========\n", registry_name, cases.size());
    for (size_t i = 0; i < cases.size(); i++) {
        const auto& c = cases[i];
        printf("[%zu] %s: M=%d K=%d N=%d block=1 nnz_blocks=0 (%d iters)\n",
               i, c.name.c_str(), c.M, c.K, c.N, NUM_ITERS);
        fflush(stdout);

        std::vector<float> iter_times = run_gemm(handle, c.M, c.N, c.K);

        // 2*M*N*K FLOPs for dense GEMM
        double flops = 2.0 * c.M * c.N * (double)c.K;
        // Bytes: read A (M*K) + read B (K*N) + write C (M*N), all FP16
        double bytes = ((double)c.M * c.K + (double)c.K * c.N + (double)c.M * c.N) * sizeof(__half);

        float sum_ms = 0.0f, min_ms = iter_times[0];
        for (int j = 0; j < NUM_ITERS; j++) {
            sum_ms += iter_times[j];
            if (iter_times[j] < min_ms) min_ms = iter_times[j];
        }
        float avg_ms = sum_ms / NUM_ITERS;

        double avg_tflops = flops / 1e12 / (avg_ms / 1e3);
        double max_tflops = flops / 1e12 / (min_ms / 1e3);
        double avg_gbps   = bytes / (avg_ms / 1e3) / 1e9;
        double max_gbps   = bytes / (min_ms / 1e3) / 1e9;

        printf("  Avg time: %.3f ms  |  Avg TFLOPS: %.3f  |  Avg GB/s: %.3f\n",
               avg_ms, avg_tflops, avg_gbps);
        printf("  Min time: %.3f ms  |  Max TFLOPS: %.3f  |  Max GB/s: %.3f\n",
               min_ms, max_tflops, max_gbps);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
}

// Helper to build a case name
static std::string make_name(const char* prefix, int M, int N, int K) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_M%d_N%d_K%d", prefix, M, N, K);
    return std::string(buf);
}

int main(int argc, char* argv[]) {
    int registry_id = -1;
    if (argc > 1) {
        registry_id = atoi(argv[1]);
    }

    // Registry 0: Sweep M (fixed K=8192, N=8192)
    std::vector<GemmCase> sweep_M = {
        {512,  8192, 8192, make_name("gemm", 512,  8192, 8192)},
        {1024, 8192, 8192, make_name("gemm", 1024, 8192, 8192)},
        {2048, 8192, 8192, make_name("gemm", 2048, 8192, 8192)},
        {4096, 8192, 8192, make_name("gemm", 4096, 8192, 8192)},
        {8192, 8192, 8192, make_name("gemm", 8192, 8192, 8192)},
    };

    // Registry 1: Sweep N (fixed M=8192, K=8192) — matches SpMM registry 4
    std::vector<GemmCase> sweep_N = {
        {8192, 512,  8192, make_name("gemm", 8192, 512,  8192)},
        {8192, 1024, 8192, make_name("gemm", 8192, 1024, 8192)},
        {8192, 2048, 8192, make_name("gemm", 8192, 2048, 8192)},
        {8192, 4096, 8192, make_name("gemm", 8192, 4096, 8192)},
        {8192, 8192, 8192, make_name("gemm", 8192, 8192, 8192)},
    };

    // Registry 2: Sweep K (fixed M=8192, N=8192) — matches SpMM registry 6
    std::vector<GemmCase> sweep_K = {
        {8192, 8192, 512,  make_name("gemm", 8192, 8192, 512)},
        {8192, 8192, 1024, make_name("gemm", 8192, 8192, 1024)},
        {8192, 8192, 2048, make_name("gemm", 8192, 8192, 2048)},
        {8192, 8192, 4096, make_name("gemm", 8192, 8192, 4096)},
        {8192, 8192, 8192, make_name("gemm", 8192, 8192, 8192)},
    };

    // Registry 3: Square matrices sweep
    std::vector<GemmCase> sweep_square = {
        {1024,  1024,  1024,  make_name("gemm", 1024,  1024,  1024)},
        {2048,  2048,  2048,  make_name("gemm", 2048,  2048,  2048)},
        {4096,  4096,  4096,  make_name("gemm", 4096,  4096,  4096)},
        {8192,  8192,  8192,  make_name("gemm", 8192,  8192,  8192)},
        {16384, 16384, 16384, make_name("gemm", 16384, 16384, 16384)},
    };

    // Registry 4: Dense ablation (matches SpMM registry 1 dimensions)
    std::vector<GemmCase> dense_ablation = {
        {32768, 32768, 512,  make_name("gemm", 32768, 32768, 512)},
        {32768, 32768, 1024, make_name("gemm", 32768, 32768, 1024)},
        {32768, 32768, 2048, make_name("gemm", 32768, 32768, 2048)},
        {32768, 32768, 4096, make_name("gemm", 32768, 32768, 4096)},
    };

    if (registry_id < 0) {
        // Sanity check
        printf("Sanity check: cuBLAS FP16 GEMM 256x256x256 (Tensor Cores)\n");
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        std::vector<float> times = run_gemm(handle, 256, 256, 256);
        CHECK_CUBLAS(cublasDestroy(handle));
        float sum = 0.0f; for (auto t : times) sum += t;
        printf("  Avg time: %.3f ms (%d iters)\n", sum / NUM_ITERS, NUM_ITERS);
        printf("\nUsage: %s <registry_id>\n", argv[0]);
        printf("  0 = Sweep M (K=8192, N=8192)\n");
        printf("  1 = Sweep N (M=8192, K=8192)\n");
        printf("  2 = Sweep K (M=8192, N=8192)\n");
        printf("  3 = Square matrices sweep\n");
        printf("  4 = Dense ablation (M=32768, N=32768)\n");
        return 0;
    }

    switch (registry_id) {
        case 0: run_cases("GemmSweepM", sweep_M); break;
        case 1: run_cases("GemmSweepN", sweep_N); break;
        case 2: run_cases("GemmSweepK", sweep_K); break;
        case 3: run_cases("GemmSquare", sweep_square); break;
        case 4: run_cases("GemmDenseAblation", dense_ablation); break;
        default:
            fprintf(stderr, "Unknown registry ID: %d\n", registry_id);
            return 1;
    }

    printf("\nDone.\n");
    return 0;
}
