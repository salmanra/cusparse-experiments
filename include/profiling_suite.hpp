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
    // Template params: M, N, K, R, C, DensityPPM (density in parts-per-million, e.g. 250000 = 25%)
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
    // DensityPPM: density in parts-per-million (e.g., 250000 = 25%, 10 = 0.001%)
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
        profile_case_sparse_diagonal_large<32, 32, 250000>, // 0 - 25% density
        profile_case_sparse_diagonal_large<64, 64, 250000>, // 1
        profile_case_sparse_diagonal_large<128, 128, 250000>, // 2
        profile_case_sparse_fill_column_large<32, 32, 250000>, // 3
        profile_case_sparse_fill_column_large<64, 64, 250000>, // 4
        profile_case_sparse_fill_column_large<128, 128, 250000>, // 5
        profile_case_sparse_fill_row_large<32, 32, 250000>, // 6
        profile_case_sparse_fill_row_large<64, 64, 250000>, // 7
        profile_case_sparse_fill_row_large<128, 128, 250000>, // 8
        profile_case_sparse_fill_random_large<32, 32, 250000>, // 9
        profile_case_sparse_fill_random_large<64, 64, 250000>, // 10
        profile_case_sparse_fill_random_large<128, 128, 250000>, // 11
        profile_case_sparse_fill_lower_triangular_large<32, 32>, // 12
        profile_case_sparse_fill_lower_triangular_large<64, 64>, // 13
        profile_case_sparse_fill_lower_triangular_large<128, 128>, // 14
    };

    // -----------------------------------------------------------------------
    // Sweep registries for parametric cost-class profiling (registries 4-7)
    // Each registry varies one axis while keeping the others roughly constant.
    // Naming: profile_case_parametric_random<M, N, K, R, C, DensityPPM>
    //
    // Cost metric estimates (in block-tile units):
    //   sparse_reads  ∝ nnz_blocks × Rt × Ct
    //   dense_reads   ∝ nnz_blocks × Ct × Nt
    //   tile_mults    ∝ nnz_blocks × Rt × Ct × Nt
    //   dram_writes   ∝ nnz_rows   × Rt × Nt
    // -----------------------------------------------------------------------

    // Registry 4: Sweep N (dense output width) — holds M=8192,K=8192,R=C=64,density=25%
    static ProfileCaseFunctionPtr ProfileSweepNRegistry[] = {
        profile_case_parametric_random<8192, 512,  8192, 256, 256, 250000>,  // N= 512
        profile_case_parametric_random<8192, 1024, 8192, 256, 256, 250000>,  // N=1024
        profile_case_parametric_random<8192, 2048, 8192, 256, 256, 250000>,  // N=2048
        profile_case_parametric_random<8192, 4096, 8192, 256, 256, 250000>,  // N=4096
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,  // N=8192
    };

    // Registry 5: Sweep density — holds M=N=K=8192,R=C=64, vary density
    static ProfileCaseFunctionPtr ProfileSweepDensityRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 256, 256,  50000>,  //  5%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 100000>,  // 10%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,  // 25%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 500000>,  // 50%
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 750000>,  // 75%
    };

    // Registry 6: Sweep K (reduction dimension) — holds M=N=8192,R=C=64,density=25%
    static ProfileCaseFunctionPtr ProfileSweepKRegistry[] = {
        profile_case_parametric_random<8192, 8192,  512, 256, 256, 250000>,  // K= 512
        profile_case_parametric_random<8192, 8192, 1024, 256, 256, 250000>,  // K=1024
        profile_case_parametric_random<8192, 8192, 2048, 256, 256, 250000>,  // K=2048
        profile_case_parametric_random<8192, 8192, 4096, 256, 256, 250000>,  // K=4096
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,  // K=8192
    };

    // Registry 7: Sweep block size — holds M=N=K=8192,density=25%
    static ProfileCaseFunctionPtr ProfileSweepBlockSizeRegistry[] = {
        profile_case_parametric_random<8192, 8192, 8192,  32,  32, 250000>,  // R=C= 32
        profile_case_parametric_random<8192, 8192, 8192,  64,  64, 250000>,  // R=C= 64
        profile_case_parametric_random<8192, 8192, 8192, 128, 128, 250000>,  // R=C=128
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,  // R=C=256
    };

    // Registry 8: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=25%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistry[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 250000>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 250000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 250000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 250000>,
    };

    // Registry 9: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=10%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD10[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 100000>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 100000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 100000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 100000>,
    };


    // Registry 10: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=5%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD5[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 50000>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 50000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 50000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 50000>,
    };


    // Registry 11: Sweep sparsity pattern — holds M=N=K=8192, R=C=256, density=50%
    static ProfileCaseFunctionPtr ProfileSweepSparsityPatternRegistryD50[] = {
        profile_case_parametric_row<8192, 8192, 8192, 256, 256, 500000>,
        profile_case_parametric_col<8192, 8192, 8192, 256, 256, 500000>,
        profile_case_parametric_multi_diag<8192, 8192, 8192, 256, 256, 500000>,
        profile_case_parametric_random<8192, 8192, 8192, 256, 256, 500000>,
    };

    // Registry 12: Sweep ultra-low density — M=N=K=8192, R=C=32, 30–10000 PPM
    static ProfileCaseFunctionPtr ProfileSweepUltraLowDensity32Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,    30>,  //  0.003%  (~2 blocks / 65536)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,   100>,  //  0.01%   (~7 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,   300>,  //  0.03%   (~20 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,  1000>,  //  0.1%    (~66 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32,  3000>,  //  0.3%    (~197 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 32, 32, 10000>,  //  1.0%    (~655 blocks)
    };

    // Registry 13: Sweep ultra-low density — M=N=K=8192, R=C=64, 60–10000 PPM
    static ProfileCaseFunctionPtr ProfileSweepUltraLowDensity64Registry[] = {
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,    60>,  //  0.006%  (~1 block / 16384)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,   200>,  //  0.02%   (~3 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,   600>,  //  0.06%   (~10 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,  2000>,  //  0.2%    (~33 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64,  6000>,  //  0.6%    (~98 blocks)
        profile_case_parametric_random<8192, 8192, 8192, 64, 64, 10000>,  //  1.0%    (~164 blocks)
    };

    static ProfileCaseFunctionPtr ProfileLargeSparseLargeBlocksRegistry[] = {
        profile_case_sparse_diagonal_large<256, 256, 250000>, //
        // profile_case_sparse_diagonal_large<512, 512, 250000>, //
        // profile_case_sparse_diagonal_large<1024, 512, 250000>, //
        // profile_case_sparse_diagonal_large<2048, 2048, 250000>, //
        profile_case_sparse_fill_column_large<256, 256, 250000>, // 5
        // profile_case_sparse_fill_column_large<512, 512, 250000>, // 5
        // profile_case_sparse_fill_column_large<1024, 1024, 250000>, // 5
        // profile_case_sparse_fill_column_large<2048, 2048, 250000>, // 5
        profile_case_sparse_fill_row_large<256, 256, 250000>, // 8
        // profile_case_sparse_fill_row_large<512, 512, 250000>, // 8
        // profile_case_sparse_fill_row_large<1024, 1024, 250000>, // 8
        // profile_case_sparse_fill_row_large<2048, 2048, 250000>, // 8
        profile_case_sparse_fill_random_large<256, 256, 250000>, // 11
        // profile_case_sparse_fill_random_large<512, 512, 250000>, // 11
        // profile_case_sparse_fill_random_large<1024, 1024, 250000>, // 11
        // profile_case_sparse_fill_random_large<2048, 2048, 250000>, // 11
        profile_case_sparse_fill_lower_triangular_large<256, 256>, // 14
        // profile_case_sparse_fill_lower_triangular_large<512, 512>, // 14
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
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_random() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_row() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_ROW, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_row_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_col() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_COL, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_col_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_diag() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));
        // Cap at main diagonal length
        uint32_t diag_len = std::min(block_matrix_height, block_matrix_width);
        nblocks = std::min(nblocks, diag_len);

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_diag_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
        std::string test_name(buf, n);
        return std::make_tuple(bsr_bfloat16, dense_bfloat16, test_name);
    }

    template <uint32_t M = 8192, uint32_t N = 8192, uint32_t K = 8192,
              uint32_t R = 64, uint32_t C = 64, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_parametric_multi_diag() {
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width  = K / C;
        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density)));

        bsr_matrix<float> bsr(M, K, R, C, nblocks, FILL_MULTI_DIAG, RAND);
        dense_matrix<float> dense(K, N, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16   = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();

        char buf[128];
        size_t n = sprintf(buf, "parametric_multi_diag_M%u_N%u_K%u_R%u_C%u_dppm%u", M, N, K, R, C, DensityPPM);
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

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPPM = 250000>
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
    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_sparse_fill_column_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPPM / 1000000.0f;
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

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPPM = 250000>
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

    template <uint32_t R = 32, uint32_t C = 32, uint32_t DensityPPM = 250000>
    inline ProfileCaseReturnType profile_case_sparse_fill_random_large() {
        // matmul params setup
        uint32_t M = 8192;
        uint32_t N = 8192;
        uint32_t K = 8192;
        // block params setup
        uint32_t block_matrix_height = M / R;
        uint32_t block_matrix_width = K / C;

        constexpr float density = DensityPPM / 1000000.0f;
        uint32_t nblocks = std::max(1u, uint32_t(std::round(block_matrix_height * block_matrix_width * density))); 

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
