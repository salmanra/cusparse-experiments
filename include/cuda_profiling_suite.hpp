#pragma once

#include <cstdint>
#include <cmath>
#include <tuple>
#include <string>
#include "cuda_bsr_matrix.hpp"

namespace cuda_profiling_suite {

    using ProfileCaseReturnType = std::tuple<cuda_bsr_matrix<float>, cuda_dense_matrix<float>, std::string>;
    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();

    ////////////////////////////////////////////////////////////////////////////
    // Dense Cases
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_square() {
        uint32_t M = 32768;
        uint32_t N = 32768;
        // For dense, make a fully-populated BSR (every block present)
        uint32_t R = 32; uint32_t C = 32;
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        uint32_t nblocks = bh * bw;
        cuda_bsr_matrix<float> src0(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> src1(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_dense_square_K%u", K);
        return std::make_tuple(std::move(src0), std::move(src1), std::string(buf, n));
    }

    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_tall() {
        uint32_t M = 32768;
        uint32_t N = 1024;
        uint32_t R = 32; uint32_t C = 32;
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        uint32_t nblocks = bh * bw;
        cuda_bsr_matrix<float> src0(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> src1(K, N, CUDA_RAND);
        return std::make_tuple(std::move(src0), std::move(src1), std::string("profile_case_dense_tall"));
    }

    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_wide() {
        uint32_t M = 1024;
        uint32_t N = 32768;
        uint32_t R = 32; uint32_t C = 32;
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        uint32_t nblocks = bh * bw;
        cuda_bsr_matrix<float> src0(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> src1(K, N, CUDA_RAND);
        return std::make_tuple(std::move(src0), std::move(src1), std::string("profile_case_dense_wide"));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Sparse Cases (small: 1024x1024)
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_single_input_block() {
        uint32_t M = 1024, N = 1024, K = 1024;
        uint32_t nblocks = 1;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_single_block_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_diagonal() {
        uint32_t M = 1024, N = 1024, K = 1024;
        uint32_t bh = M / R;
        uint32_t nblocks = bh;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_DIAG, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_diagonal_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_column() {
        uint32_t M = 1024, N = 1024, K = 1024;
        uint32_t bh = M / R;
        uint32_t nblocks = bh;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_COL, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_fill_column_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_row() {
        uint32_t M = 1024, N = 1024, K = 1024;
        uint32_t bh = M / R;
        uint32_t nblocks = bh;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_fill_row_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_random() {
        uint32_t M = 1024, N = 1024, K = 1024;
        uint32_t bh = M / R;
        uint32_t nblocks = bh;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Parametric Cases
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (bh * bw) / divisor;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[100];
        size_t n = sprintf(buf, "parametric_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_row() {
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (bh * bw) / divisor;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[100];
        size_t n = sprintf(buf, "parametric_row_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_col() {
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (bh * bw) / divisor;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_COL, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[100];
        size_t n = sprintf(buf, "parametric_col_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_diag() {
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (bh * bw) / divisor;
        uint32_t diag_len = std::min(bh, bw);
        nblocks = std::min(nblocks, diag_len);
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_DIAG, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[100];
        size_t n = sprintf(buf, "parametric_diag_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_multi_diag() {
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (bh * bw) / divisor;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_MULTI_DIAG, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[100];
        size_t n = sprintf(buf, "parametric_multi_diag_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Large Sparse Cases (8192x8192)
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_diagonal_large() {
        uint32_t M = 8192, N = 8192, K = 8192;
        uint32_t bh = M / R;
        uint32_t nblocks = bh;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_DIAG, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_diagonal_large_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_column_large() {
        uint32_t M = 8192, N = 8192, K = 8192;
        uint32_t bh = M / R;
        uint32_t nblocks = bh;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_COL, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_fill_column_large_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_row_large() {
        uint32_t M = 8192, N = 8192, K = 8192;
        uint32_t bw = K / C;
        uint32_t nblocks = bw;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_ROW, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_fill_row_large_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_random_large() {
        uint32_t M = 8192, N = 8192, K = 8192;
        uint32_t bh = M / R;
        uint32_t bw = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (bh * bw) / divisor;
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[64];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_large_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_lower_triangular_large() {
        uint32_t M = 8192, N = 8192, K = 8192;
        uint32_t nblocks = 0; // unused in TRIL constructor
        cuda_bsr_matrix<float> bsr(M, K, R, C, nblocks, CUDA_FILL_TRIL, CUDA_RAND);
        cuda_dense_matrix<float> dense(K, N, CUDA_RAND);
        char buf[80];
        size_t n = sprintf(buf, "profile_case_sparse_fill_lower_triangular_large_R%u_C%u", R, C);
        return std::make_tuple(std::move(bsr), std::move(dense), std::string(buf, n));
    }

    inline ProfileCaseReturnType profile_case_sanity_check() {
        uint32_t M = 32, N = 32, K = 32;
        uint32_t R = 32, C = 32;
        uint32_t nblocks = 1;
        cuda_bsr_matrix<float> src0(M, K, R, C, nblocks, CUDA_RAND);
        cuda_dense_matrix<float> src1(K, N, CUDA_RAND);
        return std::make_tuple(std::move(src0), std::move(src1), std::string("profile_case_sanity_check"));
    }

    ////////////////////////////////////////////////////////////////////////////
    // Registries (mirroring the TT profiling_suite)
    ////////////////////////////////////////////////////////////////////////////

    // Registry 0: Small sparse cases
    static ProfileCaseFunctionPtr ProfileCaseRegistry[] = {
        profile_case_sparse_single_input_block<32, 32>,
        profile_case_sparse_single_input_block<64, 64>,
        profile_case_sparse_single_input_block<128, 128>,
        profile_case_sparse_diagonal<32, 32>,
        profile_case_sparse_diagonal<64, 64>,
        profile_case_sparse_diagonal<128, 128>,
        profile_case_sparse_fill_column<32, 32>,
        profile_case_sparse_fill_column<64, 64>,
        profile_case_sparse_fill_column<128, 128>,
        profile_case_sparse_fill_row<32, 32>,
        profile_case_sparse_fill_row<64, 64>,
        profile_case_sparse_fill_row<128, 128>,
        profile_case_sparse_fill_random<32, 32>,
        profile_case_sparse_fill_random<64, 64>,
        profile_case_sparse_fill_random<128, 128>,
    };

    // Registry 1: Dense ablation
    static ProfileCaseFunctionPtr ProfileDenseAblationRegistry[] = {
        profile_case_dense_square<512>,
        profile_case_dense_square<1024>,
        profile_case_dense_square<2048>,
        profile_case_dense_square<4096>,
    };

    // Registry 2: Large sparse cases
    static ProfileCaseFunctionPtr ProfileLargeSparseRegistry[] = {
        profile_case_sparse_diagonal_large<32, 32, 25>,
        profile_case_sparse_diagonal_large<64, 64, 25>,
        profile_case_sparse_diagonal_large<128, 128, 25>,
        profile_case_sparse_fill_column_large<32, 32, 25>,
        profile_case_sparse_fill_column_large<64, 64, 25>,
        profile_case_sparse_fill_column_large<128, 128, 25>,
        profile_case_sparse_fill_row_large<32, 32, 25>,
        profile_case_sparse_fill_row_large<64, 64, 25>,
        profile_case_sparse_fill_row_large<128, 128, 25>,
        profile_case_sparse_fill_random_large<32, 32, 25>,
        profile_case_sparse_fill_random_large<64, 64, 25>,
        profile_case_sparse_fill_random_large<128, 128, 25>,
        profile_case_sparse_fill_lower_triangular_large<32, 32>,
        profile_case_sparse_fill_lower_triangular_large<64, 64>,
        profile_case_sparse_fill_lower_triangular_large<128, 128>,
    };

    // Registry 4: Sweep N
    static ProfileCaseFunctionPtr ProfileSweepNRegistry[] = {
        profile_case_parametric_random<8192, 512,  8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 1024, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 2048, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 4096, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 5: Sweep density
    static ProfileCaseFunctionPtr ProfileSweepDensityRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  5>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 75>,
    };

    // Registry 6: Sweep K
    static ProfileCaseFunctionPtr ProfileSweepKRegistry[] = {
        profile_case_parametric_random<8192, 8192,  512, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 1024, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 2048, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 4096, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 7: Sweep block size
    static ProfileCaseFunctionPtr ProfileSweepBlockSizeRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 25>,
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 8: Sweep sparsity pattern (density=25%)
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistry[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 9: Sweep sparsity pattern (density=10%)
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD10[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
    };

    // Registry 10: Sweep sparsity pattern (density=5%)
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD5[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 5>,
    };

    // Registry 11: Sweep sparsity pattern (density=50%)
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD50[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50>,
    };

    // Large sparse with large blocks
    static ProfileCaseFunctionPtr ProfileLargeSparseLargeBlocksRegistry[] = {
        profile_case_sparse_diagonal_large<256, 256, 25>,
        profile_case_sparse_diagonal_large<512, 512, 25>,
        profile_case_sparse_fill_column_large<256, 256, 25>,
        profile_case_sparse_fill_column_large<512, 512, 25>,
        profile_case_sparse_fill_row_large<256, 256, 25>,
        profile_case_sparse_fill_row_large<512, 512, 25>,
        profile_case_sparse_fill_random_large<256, 256, 25>,
        profile_case_sparse_fill_random_large<512, 512, 25>,
        profile_case_sparse_fill_lower_triangular_large<256, 256>,
        profile_case_sparse_fill_lower_triangular_large<512, 512>,
    };

} // namespace cuda_profiling_suite
