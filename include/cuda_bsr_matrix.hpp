#ifndef CUDA_BSR_MATRIX_HPP
#define CUDA_BSR_MATRIX_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <random>
#include <cmath>
#include <chrono>

#define CUDA_FILL_ROW 1
#define CUDA_FILL_COL 2
#define CUDA_FILL_DIAG 3
#define CUDA_FILL_TRIL 4
#define CUDA_FILL_MULTI_DIAG 5

#define CUDA_RAND_DENOM (2 << 10)
#define CUDA_SIGNED_RAND_MAX (RAND_MAX / 2)

enum cuda_content_type {
    CUDA_RAND,
    CUDA_UNIFORM,
    CUDA_ARANGE,
    CUDA_ID
};

template <typename T>
class cuda_dense_matrix {
public:
    std::vector<T> data;
    size_t H;
    size_t W;

    cuda_dense_matrix() : H(0), W(0) {}

    cuda_dense_matrix(int rows, int cols) : H(rows), W(cols) {
        data.resize(rows * cols, T(0));
    }

    cuda_dense_matrix(int rows, int cols, cuda_content_type content) : H(rows), W(cols) {
        data.resize(rows * cols);
        uint32_t k = 0;
        switch (content) {
            case CUDA_RAND:
                std::generate(data.begin(), data.end(), []() {
                    return static_cast<T>(CUDA_SIGNED_RAND_MAX - rand()) / static_cast<T>(CUDA_RAND_DENOM);
                });
                break;
            case CUDA_ID:
                for (auto it = data.begin(); it != data.end(); it++) {
                    *it = static_cast<T>(((k / W) == (k % W)) ? 1.0 : 0.0);
                    k++;
                }
                break;
            case CUDA_UNIFORM:
                std::fill(data.begin(), data.end(), static_cast<T>(1.0));
                break;
            case CUDA_ARANGE:
                for (auto it = data.begin(); it != data.end(); it++) {
                    *it = static_cast<T>(k++);
                }
                break;
        }
    }

    cuda_dense_matrix(const std::vector<T>& d, int rows, int cols) : data(d), H(rows), W(cols) {
        assert(d.size() == (size_t)(rows * cols));
    }
};

template <typename T>
class cuda_bsr_matrix {
public:
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<T> data;

    size_t H;
    size_t W;
    size_t nblocks;
    size_t R;
    size_t C;

    static std::vector<std::pair<size_t, size_t>> multi_diag_positions(
            size_t blocked_h, size_t blocked_w, size_t nb) {
        std::vector<std::pair<size_t, size_t>> positions;
        positions.reserve(nb);
        for (int d = 0; positions.size() < nb; ) {
            if (d >= 0) {
                for (size_t i = 0; i < blocked_h && i + d < blocked_w && positions.size() < nb; i++)
                    positions.emplace_back(i, i + d);
            } else {
                size_t ad = static_cast<size_t>(-d);
                for (size_t j = 0; j < blocked_w && j + ad < blocked_h && positions.size() < nb; j++)
                    positions.emplace_back(j + ad, j);
            }
            if (d <= 0) d = -d + 1; else d = -d;
            if (static_cast<size_t>(std::abs(d)) >= std::max(blocked_h, blocked_w)) break;
        }
        std::sort(positions.begin(), positions.end());
        return positions;
    }

    cuda_bsr_matrix() : H(0), W(0), nblocks(0), R(0), C(0) {}

    // Fill-pattern constructor (matches bsr_matrix from TT code)
    cuda_bsr_matrix(size_t rows, size_t cols, size_t block_rows, size_t block_cols,
                    size_t num_blocks, int fill_type, cuda_content_type content)
        : H(rows), W(cols), R(block_rows), C(block_cols), nblocks(num_blocks) {
        init_fill(fill_type, content);
    }

    // Random-placement constructor (content_type only, like bsr_matrix(..., RAND))
    cuda_bsr_matrix(size_t rows, size_t cols, size_t block_rows, size_t block_cols,
                    size_t num_blocks, cuda_content_type content)
        : H(rows), W(cols), R(block_rows), C(block_cols), nblocks(num_blocks) {
        init_random(content);
    }

private:
    void push_block_data(cuda_content_type content, size_t block_elems) {
        for (size_t k = 0; k < block_elems; k++) {
            switch (content) {
                case CUDA_RAND:
                    data.push_back(static_cast<T>(CUDA_SIGNED_RAND_MAX - rand()) / static_cast<T>(CUDA_RAND_DENOM));
                    break;
                case CUDA_UNIFORM:
                    data.push_back(static_cast<T>(1.0));
                    break;
                case CUDA_ID: {
                    float temp = ((k / C) == (k % C)) ? 1.0f : 0.0f;
                    data.push_back(static_cast<T>(temp));
                    break;
                }
                case CUDA_ARANGE:
                    data.push_back(static_cast<T>(k));
                    break;
            }
        }
    }

    void init_fill(int fill_type, cuda_content_type content) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0 && C > 0);
        assert(H % R == 0 && W % C == 0);

        size_t bh = H / R;
        size_t bw = W / C;

        indptr.resize(bh + 1, 0);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);

        if (fill_type == CUDA_FILL_ROW) {
            for (size_t i = 0; i < bh; i++) {
                for (size_t j = 0; j < bw; j++) {
                    if (i * bw + j < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        push_block_data(content, R * C);
                    }
                }
            }
        } else if (fill_type == CUDA_FILL_COL) {
            for (size_t j = 0; j < bw; j++) {
                for (size_t i = 0; i < bh; i++) {
                    if (i + (j * bh) < nblocks) {
                        indptr[i + 1]++;
                        indices.push_back(j);
                        push_block_data(content, R * C);
                    }
                }
            }
        } else if (fill_type == CUDA_FILL_DIAG) {
            assert(nblocks <= std::min(bh, bw));
            for (size_t i = 0; i < nblocks; i++) {
                indptr[i + 1]++;
                indices.push_back(i);
                push_block_data(content, R * C);
            }
        } else if (fill_type == CUDA_FILL_TRIL) {
            assert(bh == bw);
            nblocks = bh * (bh + 1) / 2;
            indices.reserve(nblocks);
            data.reserve(nblocks * R * C);
            for (size_t i = 0; i < bh; i++) {
                for (size_t j = 0; j <= i; j++) {
                    indptr[i + 1]++;
                    indices.push_back(j);
                    push_block_data(content, R * C);
                }
            }
        } else if (fill_type == CUDA_FILL_MULTI_DIAG) {
            auto positions = multi_diag_positions(bh, bw, nblocks);
            nblocks = positions.size();
            for (auto& [row, col] : positions) {
                indptr[row + 1]++;
                indices.push_back(col);
                push_block_data(content, R * C);
            }
        } else {
            throw std::invalid_argument("Invalid fill type");
        }

        // Prefix sum
        for (size_t i = 1; i < indptr.size(); i++)
            indptr[i] += indptr[i - 1];
    }

    void init_random(cuda_content_type content) {
        assert(H * W >= nblocks * R * C);
        assert(R > 0 && C > 0);
        assert(H % R == 0 && W % C == 0);

        size_t bh = H / R;
        size_t bw = W / C;

        indptr.resize(bh + 1, 0);
        indices.reserve(nblocks);
        data.reserve(nblocks * R * C);

        std::vector<int> block_indices(bh * bw, 0);
        std::fill(block_indices.begin(), block_indices.begin() + nblocks, 1);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(block_indices.begin(), block_indices.end(), std::default_random_engine(seed));

        for (size_t i = 0; i < block_indices.size(); i++) {
            if (block_indices[i] == 1) {
                size_t row = i / bw;
                size_t col = i % bw;
                indptr[row + 1]++;
                indices.push_back(col);
                push_block_data(content, R * C);
            }
        }

        for (size_t i = 1; i < indptr.size(); i++)
            indptr[i] += indptr[i - 1];
    }

public:
    // Convert BSR to CSR format for cuSPARSE
    // Returns: {csr_row_offsets, csr_col_indices, csr_values, nnz}
    struct CSRData {
        std::vector<int> row_offsets;
        std::vector<int> col_indices;
        std::vector<T> values;
        int nnz;
    };

    CSRData to_csr() const {
        CSRData csr;
        size_t bh = H / R;

        // Expand BSR blocks into element-level CSR
        csr.row_offsets.resize(H + 1, 0);
        csr.col_indices.reserve(nblocks * R * C);
        csr.values.reserve(nblocks * R * C);

        // First pass: count nnz per row
        for (size_t bi = 0; bi < bh; bi++) {
            int blocks_in_row = indptr[bi + 1] - indptr[bi];
            for (size_t r = 0; r < R; r++) {
                csr.row_offsets[bi * R + r + 1] = blocks_in_row * C;
            }
        }
        // Prefix sum
        for (size_t i = 1; i <= H; i++)
            csr.row_offsets[i] += csr.row_offsets[i - 1];

        csr.nnz = csr.row_offsets[H];
        csr.col_indices.resize(csr.nnz);
        csr.values.resize(csr.nnz);

        // Second pass: fill values
        // Track write position per row
        std::vector<int> row_pos(H, 0);

        for (size_t bi = 0; bi < bh; bi++) {
            for (int idx = indptr[bi]; idx < indptr[bi + 1]; idx++) {
                int bj = indices[idx];
                const T* block_data = &data[idx * R * C];
                for (size_t r = 0; r < R; r++) {
                    size_t global_row = bi * R + r;
                    int write_pos = csr.row_offsets[global_row] + row_pos[global_row];
                    for (size_t c = 0; c < C; c++) {
                        csr.col_indices[write_pos + c] = bj * C + c;
                        csr.values[write_pos + c] = block_data[r * C + c];
                    }
                    row_pos[global_row] += C;
                }
            }
        }

        return csr;
    }

    // SpMM on host for verification: C = A * B
    cuda_dense_matrix<T> spmm(const cuda_dense_matrix<T>& B) const {
        assert(W == B.H);
        cuda_dense_matrix<T> output(H, B.W);
        size_t bh = H / R;

        for (size_t i = 0; i < bh; i++) {
            for (int idx = indptr[i]; idx < indptr[i + 1]; idx++) {
                size_t j = indices[idx];
                const T* block = &data[idx * R * C];
                for (size_t r = 0; r < R; r++) {
                    for (size_t p = 0; p < B.W; p++) {
                        T sum = 0;
                        for (size_t c = 0; c < C; c++) {
                            sum += block[r * C + c] * B.data[(j * C + c) * B.W + p];
                        }
                        output.data[(i * R + r) * B.W + p] += sum;
                    }
                }
            }
        }
        return output;
    }
};

#endif // CUDA_BSR_MATRIX_HPP
