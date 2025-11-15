#pragma once
#include "field.h"
#include <vector>
#include <algorithm>

// Dense matrix over F3, row-major, packed as B3
struct BitMatrix3 {
    int m = 0, n = 0;
    int W = 0;  // words per row = ceil(n/64)
    std::vector<B3> data;

    BitMatrix3() = default;
    BitMatrix3(int m_, int n_) : m(m_), n(n_) {
        W = (n + 63) / 64;
        data.assign((long long)m * (long long)W, B3{});
    }

    B3* row_ptr(int i) { return data.data() + (long long)i * W; }
    const B3* row_ptr(int i) const { return data.data() + (long long)i * W; }

    U64 get(int i, int j) const {
        int w = j / 64;
        int b = j % 64;
        return row_ptr(i)[w].get(b);
    }

    void set(int i, int j, U64 val) {
        int w = j / 64;
        int b = j % 64;
        row_ptr(i)[w].set(b, val);
    }

    std::vector<B3> release() {
        std::vector<B3> out;
        out.swap(data);
        return out;
    }
};

// REF factorization over F3 with repeated-solve support
struct RefFactor3 {
    int m = 0, n = 0, W = 0;
    int rank = 0;

    std::vector<B3> mat;
    std::vector<B3*> row;
    std::vector<int> row_perm;      // current->original mapping
    std::vector<int> pivot_col;

    // For each pivot r: records ORIGINAL indices and coefficients
    struct FwdMask {
        int pivot_orig_idx = 0;
        U64 pivot_norm_coeff = 1;  // normalization multiplier (1 or 2)
        std::vector<std::pair<int, U64>> target_coeff;  // (original_idx, coeff)
    };
    std::vector<FwdMask> fwd_masks;
    std::vector<U8> is_zero_row;

    static inline int word_of(int j) { return j / 64; }
    static inline int bit_of(int j)  { return j % 64; }

    inline void add_scaled(B3* dst, const B3* src, U64 c) const {
        for (int t = 0; t < W; ++t) dst[t] += (src[t] * c);
    }

    inline U64 get_bit(const B3* rp, int j) const {
        return rp[word_of(j)].get(bit_of(j));
    }

    int find_pivot_in_col(int r, int j) const {
        for (int i = r; i < m; ++i) {
            if (get_bit(row[i], j) != 0) return i;
        }
        return -1;
    }

    void factorize(BitMatrix3& A) {
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

            // Normalize pivot: if pivot = 2, multiply by 2 (inverse in F3)
            U64 pivot_val = get_bit(row[rank], j);
            U64 norm_coeff = 1;
            if (pivot_val == 2) {
                norm_coeff = 2;
                for (int t = 0; t < W; ++t) row[rank][t] = row[rank][t] * 2;
            }

            // Build forward mask
            FwdMask fm;
            fm.pivot_orig_idx = row_perm[rank];
            fm.pivot_norm_coeff = norm_coeff;

            for (int i = rank + 1; i < m; ++i) {
                U64 coeff = get_bit(row[i], j);
                if (coeff != 0) {
                    fm.target_coeff.push_back({row_perm[i], coeff});
                    add_scaled(row[i], row[rank], (3 - coeff) % 3);  // subtract
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
                if (!row[i][t].is_zero()) { any = 1; break; }
            }
            is_zero_row[i] = (any ? 0 : 1);
        }
    }

    bool solve(const std::vector<U8>& b, std::vector<U8>& x) const {
        if (m == 0 || n == 0) return false;
        if ((int)b.size() != m || (int)x.size() != n) return false;

        // Work with a copy using ORIGINAL indices
        std::vector<U8> y(m);
        for (int i = 0; i < m; ++i) y[i] = b[i] % 3;

        // Apply forward operations
        for (int r = 0; r < rank; ++r) {
            const FwdMask& fm = fwd_masks[r];
            
            // Apply normalization to b vector
            if (fm.pivot_norm_coeff == 2) {
                y[fm.pivot_orig_idx] = (y[fm.pivot_orig_idx] * 2) % 3;
            }
            
            U8 pivot_val = y[fm.pivot_orig_idx] % 3;
            
            for (auto [orig_idx, coeff] : fm.target_coeff) {
                y[orig_idx] = (y[orig_idx] + (3 - coeff) * pivot_val) % 3;
            }
        }

        // Check inconsistency
        for (int i = 0; i < m; ++i) {
            if (is_zero_row[i]) {
                int orig_idx = row_perm[i];
                if (y[orig_idx] % 3 != 0) return false;
            }
        }

        // Back substitution
        std::fill(x.begin(), x.end(), 0);

        for (int r = rank - 1; r >= 0; --r) {
            int j = pivot_col[r];
            
            // Calculate value from already-determined variables
            int val = 0;
            for (int jj = j + 1; jj < n; ++jj) {
                U64 a_ij = get_bit(row[r], jj);
                if (a_ij != 0) val += a_ij * x[jj];
            }
            val %= 3;

            int orig_idx = row_perm[r];
            U8 xj = (y[orig_idx] + 3 - val) % 3;
            x[j] = xj;
        }

        return true;
    }
};