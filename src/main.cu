#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

// SpMM: C = alpha * A * B + beta * C
// A: sparse matrix (m x k) in CSR format
// B: dense matrix (k x n)
// C: dense matrix (m x n)

int main(int argc, char* argv[]) {
    // Matrix dimensions
    const int m = 4;  // rows of A and C
    const int k = 4;  // cols of A, rows of B
    const int n = 3;  // cols of B and C

    // Scalars for SpMM: C = alpha * A * B + beta * C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Example sparse matrix A in CSR format (4x4)
    // A = | 1  0  2  0 |
    //     | 0  3  0  0 |
    //     | 4  0  5  6 |
    //     | 0  0  0  7 |
    const int A_nnz = 7;  // number of non-zeros

    // CSR format: row_offsets, column_indices, values
    std::vector<int> h_A_row_offsets = {0, 2, 3, 6, 7};
    std::vector<int> h_A_col_indices = {0, 2, 1, 0, 2, 3, 3};
    std::vector<float> h_A_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    // Dense matrix B (k x n) in column-major order
    std::vector<float> h_B = {
        1.0f, 2.0f, 3.0f, 4.0f,  // column 0
        5.0f, 6.0f, 7.0f, 8.0f,  // column 1
        9.0f, 10.0f, 11.0f, 12.0f // column 2
    };
    const int ldb = k;  // leading dimension of B

    // Dense matrix C (m x n) in column-major order - output
    std::vector<float> h_C(m * n, 0.0f);
    const int ldc = m;  // leading dimension of C

    // Device memory pointers
    int* d_A_row_offsets = nullptr;
    int* d_A_col_indices = nullptr;
    float* d_A_values = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A_row_offsets, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_col_indices, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_values, A_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A_row_offsets, h_A_row_offsets.data(),
                          (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_col_indices, h_A_col_indices.data(),
                          A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_values, h_A_values.data(),
                          A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(),
                          k * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(),
                          m * n * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuSPARSE handle
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix descriptor for A (CSR format)
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        m, k, A_nnz,
        d_A_row_offsets,
        d_A_col_indices,
        d_A_values,
        CUSPARSE_INDEX_32I,      // row offsets index type
        CUSPARSE_INDEX_32I,      // column indices index type
        CUSPARSE_INDEX_BASE_ZERO, // base index
        CUDA_R_32F               // data type
    ));

    // Create dense matrix descriptor for B
    cusparseDnMatDescr_t matB;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB,
        k, n, ldb,
        d_B,
        CUDA_R_32F,
        CUSPARSE_ORDER_COL  // column-major order
    ));

    // Create dense matrix descriptor for C
    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC,
        m, n, ldc,
        d_C,
        CUDA_R_32F,
        CUSPARSE_ORDER_COL  // column-major order
    ));

    // Determine buffer size for SpMM
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
        CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,                        // compute type
        CUSPARSE_SPMM_ALG_DEFAULT,         // algorithm
        &bufferSize
    ));

    // Allocate external buffer
    void* d_buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    printf("cuSPARSE SpMM buffer size: %zu bytes\n", bufferSize);

    // Execute SpMM: C = alpha * A * B + beta * C
    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        d_buffer
    ));

    // Synchronize to ensure computation is complete
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C,
                          m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    printf("\nResult matrix C (%d x %d):\n", m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f ", h_C[j * ldc + i]);  // column-major access
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_A_row_offsets));
    CHECK_CUDA(cudaFree(d_A_col_indices));
    CHECK_CUDA(cudaFree(d_A_values));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_buffer));

    printf("\nSpMM completed successfully!\n");
    return 0;
}
