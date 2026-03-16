#pragma once

#include <cstdint>
#include <cmath>
#include <random>
#include "include_me.hpp"
#include "block_spmm/inc/bsr_matrix.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/bfloat4.hpp"

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

namespace profiling_suite {
    // ehh... what are the interesting test cases to profile?

    // invariably, big test cases are interesting.
    // Dense
        // Square
        // Tall (we're screwed)
        // Wide
    // Sparse (p < 0.1)
        // single input block
        // diagonal
            // perm
        // column fill
        // row fill
    // Semi-sparse (0.1 < p < 0.5)
        // Random placement
        // Row fill
        // Column fill
        // Checkerboard
        // Diagonal

    // What were the largest dimensions the dense cases supported?
        //         // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
        //                                                          < 2^15 = 32,768
        // --> M=N=2^7=4096 is the largest s quare output which is power of 2
        // I tried M=8192 N=4096 K=512 and, after a bunch of time, got an error (floating point exception) which is apparently a divide by 0 error
        //      where does the matmul kernel divide by 0? curious but not important.
        //
        //


    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Profile Case Declarations ////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    using ProfileCaseReturnType = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string>;
    template <uint32_t>
    ProfileCaseReturnType profile_case_dense_square();

    template <uint32_t>
    ProfileCaseReturnType profile_case_dense_tall();
    template <uint32_t>
    ProfileCaseReturnType profile_case_dense_wide();

    // template on block sizes R and C.
    // we could template on fill_type, but too late :P
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_single_input_block();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_diagonal();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_row();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_column();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_random();

    // Fully parametric sparse cases for sweeps
    // Template params: M, N, K, R, C, DensityPercent (density as %, e.g. 25 = 25%)
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_random();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_row();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_col();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_diag();
    template <uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_parametric_multi_diag();

    // Large sparse cases (32768x32768 matrices)
    // DensityPercent: density as percentage (e.g., 25 = 0.25, 10 = 0.10)
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_diagonal_large();
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_column_large();
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_row_large();
    template <uint32_t, uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_random_large();
    template <uint32_t, uint32_t>
    ProfileCaseReturnType profile_case_sparse_fill_lower_triangular_large();

    ProfileCaseReturnType profile_case_sanity_check();

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Profile Case Registry ////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    using ProfileCaseFunctionPtr = ProfileCaseReturnType (*)();
    static ProfileCaseFunctionPtr ProfileCaseRegistry[] = {
        profile_case_sparse_single_input_block<32, 32>, // 3
        profile_case_sparse_single_input_block<64, 64>, // 4
        profile_case_sparse_single_input_block<128, 128>, // 5
        profile_case_sparse_diagonal<32, 32>, // 6
        profile_case_sparse_diagonal<64, 64>, // 7
        profile_case_sparse_diagonal<128, 128>, // 8
        profile_case_sparse_fill_column<32, 32>, // 9
        profile_case_sparse_fill_column<64, 64>, // 10
        profile_case_sparse_fill_column<128, 128>, // 11
        profile_case_sparse_fill_row<32, 32>, // 12
        profile_case_sparse_fill_row<64, 64>, // 13
        profile_case_sparse_fill_row<128, 128>, // 14
        profile_case_sparse_fill_random<32, 32>, // 15
        profile_case_sparse_fill_random<64, 64>, // 16
        profile_case_sparse_fill_random<128, 128>, // 17
    };

    static ProfileCaseFunctionPtr ProfileDenseAblationRegistry[] = {
        profile_case_dense_square<512>, // 0
        profile_case_dense_square<1024>, // 1
        profile_case_dense_square<2048>, // 2
        profile_case_dense_square<4096>, // 3
    };

    static ProfileCaseFunctionPtr ProfileLargeSparseRegistry[] = {
        profile_case_sparse_diagonal_large<32, 32, 25>, // 0 - 25% density
        profile_case_sparse_diagonal_large<64, 64, 25>, // 1
        profile_case_sparse_diagonal_large<128, 128, 25>, // 2
        profile_case_sparse_fill_column_large<32, 32, 25>, // 3
        profile_case_sparse_fill_column_large<64, 64, 25>, // 4
        profile_case_sparse_fill_column_large<128, 128, 25>, // 5
        profile_case_sparse_fill_row_large<32, 32, 25>, // 6
        profile_case_sparse_fill_row_large<64, 64, 25>, // 7
        profile_case_sparse_fill_row_large<128, 128, 25>, // 8
        profile_case_sparse_fill_random_large<32, 32, 25>, // 9
        profile_case_sparse_fill_random_large<64, 64, 25>, // 10
        profile_case_sparse_fill_random_large<128, 128, 25>, // 11
        profile_case_sparse_fill_lower_triangular_large<32, 32>, // 12
        profile_case_sparse_fill_lower_triangular_large<64, 64>, // 13
        profile_case_sparse_fill_lower_triangular_large<128, 128>, // 14
    };

    // -----------------------------------------------------------------------
    // Sweep registries for parametric cost-class profiling (registries 4-7)
    // Each registry varies one axis while keeping the others roughly constant.
    // Naming: profile_case_parametric_random<M, N, K, R, C, DensityPercent>
    //
    // Cost metric estimates (in block-tile units):
    //   sparse_reads  ∝ nnz_blocks × Rt × Ct
    //   dense_reads   ∝ nnz_blocks × Ct × Nt
    //   tile_mults    ∝ nnz_blocks × Rt × Ct × Nt
    //   dram_writes   ∝ nnz_rows   × Rt × Nt
    // -----------------------------------------------------------------------

    // Registry 4: Sweep N (dense output width) — holds M=8192,K=8192,R=C=64,density=25%
    static ProfileCaseFunctionPtr ProfileSweepNRegistry[] = {
        profile_case_parametric_random<8192, 512,  8192, 256, 256, 25>,  // N= 512
        profile_case_parametric_random<8192, 1024, 8192, 256, 256, 25>,  // N=1024
        profile_case_parametric_random<8192, 2048, 8192, 256, 256, 25>,  // N=2048
        profile_case_parametric_random<8192, 4096, 8192, 256, 256, 25>,  // N=4096
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // N=8192
    };

    // Registry 5: Sweep density — holds M=N=K=8192,R=C=64, vary density
    static ProfileCaseFunctionPtr ProfileSweepDensityRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  5>,  //  5%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,  // 10%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // 25%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50>,  // 50%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 75>,  // 75%
    };

    // Registry 6: Sweep K (reduction dimension) — holds M=N=8192,R=C=64,density=25%
    static ProfileCaseFunctionPtr ProfileSweepKRegistry[] = {
        profile_case_parametric_random<8192, 8192,  512, 256, 256, 25>,  // K= 512
        profile_case_parametric_random<8192, 8192, 1024, 256, 256, 25>,  // K=1024
        profile_case_parametric_random<8192, 8192, 2048, 256, 256, 25>,  // K=2048
        profile_case_parametric_random<8192, 8192, 4096, 256, 256, 25>,  // K=4096
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // K=8192
    };

    // Registry 7: Sweep block size — holds M=N=K=8192,density=25%
    static ProfileCaseFunctionPtr ProfileSweepBlockSizeRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 25>,  // R=C= 32
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 25>,  // R=C= 64
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 25>,  // R=C=128
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,  // R=C=256
        // profile_case_parametric_random<8192, 8192, 8192, 512, 512, 25>,  // R=C=512
    };

    // Registry 8: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=25%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistry[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 25>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 25>,
    };

    // Registry 9: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=10%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD10[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 10>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 10>,
    };


    // Registry 10: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=5%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD5[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 5>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 5>,
    };


    // Registry 11: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=50%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD50[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_diag<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 50>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50>,
    };

    static ProfileCaseFunctionPtr ProfileLargeSparseLargeBlocksRegistry[] = {
        profile_case_sparse_diagonal_large<256, 256, 25>, // 
        profile_case_sparse_diagonal_large<512, 512, 25>, // 
        // profile_case_sparse_diagonal_large<1024, 512, 25>, // 
        // profile_case_sparse_diagonal_large<2048, 2048, 25>, // 
        profile_case_sparse_fill_column_large<256, 256, 25>, // 5
        profile_case_sparse_fill_column_large<512, 512, 25>, // 5
        // profile_case_sparse_fill_column_large<1024, 1024, 25>, // 5
        // profile_case_sparse_fill_column_large<2048, 2048, 25>, // 5
        profile_case_sparse_fill_row_large<256, 256, 25>, // 8
        profile_case_sparse_fill_row_large<512, 512, 25>, // 8
        // profile_case_sparse_fill_row_large<1024, 1024, 25>, // 8
        // profile_case_sparse_fill_row_large<2048, 2048, 25>, // 8
        profile_case_sparse_fill_random_large<256, 256, 25>, // 11
        profile_case_sparse_fill_random_large<512, 512, 25>, // 11
        // profile_case_sparse_fill_random_large<1024, 1024, 25>, // 11
        // profile_case_sparse_fill_random_large<2048, 2048, 25>, // 11
        profile_case_sparse_fill_lower_triangular_large<256, 256>, // 14
        profile_case_sparse_fill_lower_triangular_large<512, 512>, // 14
        // profile_case_sparse_fill_lower_triangular_large<1024, 1024>, // 14
        // profile_case_sparse_fill_lower_triangular_large<2048, 2048>, // 14
    };

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Profile Case Definitions /////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    ///////// Sanity Check /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    inline ProfileCaseReturnType profile_case_sanity_check() {
        uint32_t M = 32;
        uint32_t N = 32;
        uint32_t K = 32;

        uint32_t R = 32;
        uint32_t C = 32;
        uint32_t nblocks = 1;

        bsr_matrix<float> src0(M, K, R, C, nblocks, RAND);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "profile_case_sanity_check");
    }
    ////////////////////////////////////////////////////////////////////////////
    ///////// Parametric Cases /////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_row() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_row_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_col() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_col_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_diag() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;
        // Cap at main diagonal length
        uint32_t diag_len = std::min(block_matrix_height, block_matrix_width);
        nblocks = std::min(nblocks, diag_len);

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_diag_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_parametric_multi_diag() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor;

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_MULTI_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "parametric_multi_diag_M%u_N%u_K%u_R%u_C%u_d%u", M, N, K, R, C, DensityPercent);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    ////////////////////////////////////////////////////////////////////////////
    ///////// Large Cases //////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_lower_triangular_large(){
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;

        uint32_t nblocks = 0; // unused in triangular constructor 

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_TRIL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[100];
        size_t n = sprintf(buf, "profile_case_sparse_fill_lower_triangular_large%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_diagonal_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;

        uint32_t nblocks = block_matrix_height; 

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_diagonal_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    // TODO: add versions of this which use multiple of block_matrix_height for nblocks
    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_column_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t nblocks = block_matrix_height; 

        // nz blocks fill the first column
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_column_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_row_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        uint32_t nblocks =  block_matrix_width; 

        // nz blocks fill the first row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_row_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPercent = 25>
    inline ProfileCaseReturnType profile_case_sparse_fill_random_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPercent / 100.0f;
        uint32_t divisor = uint32_t(std::round(1.0 / density));
        uint32_t nblocks = (block_matrix_height * block_matrix_width) / divisor; 

        // nz blocks placed randomly
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_large_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }


    ////////////////////////////////////////////////////////////////////////////
    ///////// Dense Cases //////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_square() {
        uint32_t M = 32768;
        uint32_t N = 32768;

        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> src0(tmp, N);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_dense_square_K%i", K);
        std::string test_name(buf, n);
        return std::make_tuple(src0_bfoat16, src1_bfloat16, test_name);
    }

    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_tall() {
        uint32_t M = 32768;
        uint32_t N = 1024;
        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> src0(tmp, N);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "profile_case_dense_tall");
    }
    template <uint32_t K = 512>
    inline ProfileCaseReturnType profile_case_dense_wide() {
        uint32_t M = 1024;
        uint32_t N = 32768;

        dense_matrix<float> tmp(M, K, RAND);
        bsr_matrix<float> src0(tmp, N);
        dense_matrix<float> src1(K, N, RAND);

        bsr_matrix<bfloat16> src0_bfoat16 = src0.bfloat16_cast();
        dense_matrix<bfloat16> src1_bfloat16 = src1.bfloat16_cast();
        return std::make_tuple(src0_bfoat16, src1_bfloat16, "profile_case_dense_wide");
    }

    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ///////// Sparse Cases /////////////////////////////////////////////////////
    ///////// p < 0.1 //////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////


    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_single_input_block() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t nblocks = 1;
        uint32_t block_matrix_height = M / R;

        // nz block is in the first position
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_single_block_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_diagonal() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the diagonal
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_diagonal_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_column() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the first column
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_column_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_row() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks fill the first row
        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_row_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t R = 32, uint32_t C = 32>
    inline ProfileCaseReturnType profile_case_sparse_fill_random() {
        // matmul params setup
        uint32_t M = 1024;
        uint32_t N = 1024;
        uint32_t K = 1024;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t nblocks = block_matrix_height;

        // nz blocks placed randomly
        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);


        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[50];
        size_t n = sprintf(buf, "profile_case_sparse_fill_random_R%i_C%d", R, C);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

} // namespace profiling_suite
