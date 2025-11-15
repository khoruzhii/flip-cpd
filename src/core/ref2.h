#pragma once
#include <cstdint>
#include <vector>
#include <bit>
#include <algorithm>

using U8 = std::uint8_t;
using U64 = std::uint64_t;

// Dense bit-packed matrix over F2, row-major
struct BitMatrix2 {
    int m = 0, n = 0;
    int W = 0;  // words per row = ceil(n/64)
    std::vector<U64> data;

    BitMatrix2() = default;
    BitMatrix2(int m_, int n_) : m(m_), n(n_) {
        W = (n + 63) >> 6;
        data.assign((long long)m * (long long)W, 0);
    }

    U64* row_ptr(int i) { return data.data() + (long long)i * W; }
    const U64* row_ptr(int i) const { return data.data() + (long long)i * W; }

    void flip(int i, int j) {
        int w = j >> 6;
        int b = j & 63;
        row_ptr(i)[w] ^= (U64(1) << b);
    }

    std::vector<U64> release() {
        std::vector<U64> out;
        out.swap(data);
        return out;
    }
};

// REF factorization with repeated-solve support
struct RefFactor2 {
    int m = 0, n = 0, W = 0;
    int rank = 0;

    std::vector<U64> mat;
    std::vector<U64*> row;
    std::vector<int> row_perm;      // current->original mapping
    std::vector<int> pivot_col;

    // For each pivot r: records ORIGINAL indices of rows that were XOR'ed
    struct FwdMask {
        int pivot_orig_idx = 0;     // original index of pivot row
        std::vector<int> target_orig_idx;  // original indices of XOR'ed rows
    };
    std::vector<FwdMask> fwd_masks;

    std::vector<U8> is_zero_row;

    static inline int word_of(int j) { return j >> 6; }
    static inline int bit_of(int j)  { return j & 63; }

    inline void xor_full(U64* dst, const U64* src) const {
        for (int t = 0; t < W; ++t) dst[t] ^= src[t];
    }

    inline U8 get_bit(const U64* rp, int j) const {
        return (rp[word_of(j)] >> bit_of(j)) & 1U;
    }

    int find_pivot_in_col(int r, int j) const {
        int w = word_of(j);
        U64 mask = (U64(1) << bit_of(j));
        for (int i = r; i < m; ++i) {
            if (row[i][w] & mask) return i;
        }
        return -1;
    }

    void factorize(BitMatrix2& A) {
        m = A.m; n = A.n; W = A.W;
        mat = A.release();
        row.resize(m);
        for (int i = 0; i < m; ++i) row[i] = mat.data() + (long long)i * W;

        row_perm.resize(m);
        for (int i = 0; i < m; ++i) row_perm[i] = i;

        pivot_col.clear();
        fwd_masks.clear();

        rank = 0;
        int j = 0;

        while (rank < m && j < n) {
            int p = find_pivot_in_col(rank, j);
            if (p < 0) { ++j; continue; }

            if (p != rank) {
                std::swap(row[p], row[rank]);
                std::swap(row_perm[p], row_perm[rank]);
            }

            // Build forward mask with ORIGINAL indices
            FwdMask fm;
            fm.pivot_orig_idx = row_perm[rank];

            int wj = word_of(j);
            U64 mask_bit = (U64(1) << bit_of(j));
            
            for (int i = rank + 1; i < m; ++i) {
                if (row[i][wj] & mask_bit) {
                    fm.target_orig_idx.push_back(row_perm[i]);  // store ORIGINAL index
                    xor_full(row[i], row[rank]);
                }
            }

            pivot_col.push_back(j);
            fwd_masks.push_back(std::move(fm));

            ++rank;
            ++j;
        }

        is_zero_row.assign(m, 0);
        for (int i = 0; i < m; ++i) {
            U8 any = 0;
            for (int t = 0; t < W; ++t) {
                if (row[i][t]) { any = 1; break; }
            }
            is_zero_row[i] = (any ? 0 : 1);
        }
    }

    bool solve(const std::vector<U8>& b, std::vector<U8>& x) const {
        if (m == 0 || n == 0) return false;
        if ((int)b.size() != m || (int)x.size() != n) return false;

        // Work with a copy of b using ORIGINAL indices
        std::vector<U8> y = b;

        // Apply forward operations using ORIGINAL indices
        for (int r = 0; r < rank; ++r) {
            const FwdMask& fm = fwd_masks[r];
            U8 pivot_val = y[fm.pivot_orig_idx] & 1u;
            
            if (pivot_val) {
                for (int orig_idx : fm.target_orig_idx) {
                    y[orig_idx] ^= pivot_val;
                }
            }
        }

        // Check inconsistency: zero rows in CURRENT positions
        for (int i = 0; i < m; ++i) {
            if (is_zero_row[i]) {
                int orig_idx = row_perm[i];
                if (y[orig_idx] & 1u) return false;
            }
        }

        // Back substitution
        std::vector<U64> X((n + 63) >> 6, 0);
        std::fill(x.begin(), x.end(), 0);

        for (int r = rank - 1; r >= 0; --r) {
            int j = pivot_col[r];
            int wj = word_of(j);
            int bj = bit_of(j);

            // Calculate parity from already-determined variables
            int parity = 0;
            U64 m0 = (bj == 63) ? 0ull : (~U64(0) << (bj + 1));
            parity ^= (std::popcount((row[r][wj] & m0) & X[wj]) & 1);
            for (int t = wj + 1; t < W; ++t)
                parity ^= (std::popcount(row[r][t] & X[t]) & 1);

            int orig_idx = row_perm[r];
            U8 xj = (y[orig_idx] ^ U8(parity)) & 1u;
            x[j] = xj;

            if (xj) {
                X[wj] ^= (U64(1) << bj);
            }
        }

        return true;
    }
};