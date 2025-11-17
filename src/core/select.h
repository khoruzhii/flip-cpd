// select.h v3
// Scheme selection and recursion-analysis utilities
//
// Note on WDIAG:
//   By default (no WDIAG), t-operations treat a term as "t-recursive"
//   if U_q == V_q (Gram-style criterion).
//   If compiled with -DWDIAG, t-recursions for "xt" operations are detected
//   using the diagonal support of W_q: a term is "t-recursive" if it
//   contributes only to diagonal output entries (upper-triangular-only work).

#pragma once

#include "rational.h"
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>
#include <string>
#include <cstdint>
#include <stdexcept>

// ------------------------------
// Basic metrics
// ------------------------------

// Count nonzeros in rational array
inline size_t count_nnz(const std::vector<Rational>& arr) {
    return std::count_if(arr.begin(), arr.end(),
                         [](const Rational& r) { return r != 0; });
}

// Check if all rationals are integers
inline bool is_integer_scheme(const std::vector<Rational>& U,
                              const std::vector<Rational>& V,
                              const std::vector<Rational>& W) {
    auto check = [](const std::vector<Rational>& arr) {
        return std::all_of(arr.begin(), arr.end(),
                           [](const Rational& r) { return r.denominator() == 1; });
    };
    return check(U) && check(V) && check(W);
}

// Find max denominator across all factors
inline I64 max_denominator(const std::vector<Rational>& U,
                           const std::vector<Rational>& V,
                           const std::vector<Rational>& W) {
    I64 max_den = 1;
    for (const auto& r : U) max_den = std::max(max_den, r.denominator());
    for (const auto& r : V) max_den = std::max(max_den, r.denominator());
    for (const auto& r : W) max_den = std::max(max_den, r.denominator());
    return max_den;
}

// Scheme metrics for selection
struct SchemeMetrics {
    size_t index;
    size_t nnz;
    bool is_integer;
    I64 max_den;

    // Legacy order: nnz (asc), is_integer (desc), max_den (asc)
    bool operator<(const SchemeMetrics& other) const {
        if (nnz != other.nnz) return nnz < other.nnz;
        if (is_integer != other.is_integer) return is_integer > other.is_integer;
        return max_den < other.max_den;
    }
};

// Select best scheme from a collection (legacy API)
inline size_t select_best_scheme(
    const std::vector<std::vector<Rational>>& schemes_U,
    const std::vector<std::vector<Rational>>& schemes_V,
    const std::vector<std::vector<Rational>>& schemes_W)
{
    size_t num_schemes = schemes_U.size();
    if (num_schemes == 0) {
        throw std::runtime_error("No schemes to select from");
    }

    std::vector<SchemeMetrics> metrics;
    metrics.reserve(num_schemes);

    for (size_t idx = 0; idx < num_schemes; ++idx) {
        size_t nnz = count_nnz(schemes_U[idx]) +
                     count_nnz(schemes_V[idx]) +
                     count_nnz(schemes_W[idx]);
        bool is_int = is_integer_scheme(schemes_U[idx], schemes_V[idx], schemes_W[idx]);
        I64 max_den = max_denominator(schemes_U[idx], schemes_V[idx], schemes_W[idx]);

        metrics.push_back({idx, nnz, is_int, max_den});
    }

    std::sort(metrics.begin(), metrics.end());
    return metrics[0].index;
}

// ------------------------------
// Index packing helpers
// ------------------------------

inline std::size_t sym_pack_cpp(std::size_t i, std::size_t j, std::size_t n) {
    std::size_t i2 = (i < j) ? i : j;
    std::size_t j2 = (i < j) ? j : i;
    return (i2 * (2 * n - i2 - 1)) / 2 + j2;
}

inline std::size_t skew_pack_cpp(std::size_t i, std::size_t j, std::size_t n) {
    std::size_t i2 = (i < j) ? i : j;
    std::size_t j2 = (i < j) ? j : i;
    return (i2 * (2 * n - i2 - 1)) / 2 + (j2 - i2 - 1);
}

// ------------------------------
// Mask builders for recursion analysis
// ------------------------------

enum class StructChar : char { G='g', U='u', L='l', S='s', K='k', W='w' };

// Build diagonal-only mask for parameterization of size nPar
// For u,l,s,w: indices sym_pack(i,i,n) are marked 1; for g,k returns all zeros
inline std::vector<unsigned char> build_diag_mask_left(StructChar a, int n1, int nPar) {
    std::vector<unsigned char> mask(nPar, 0);
    if (a == StructChar::U || a == StructChar::L || a == StructChar::S || a == StructChar::W) {
        for (int t = 0; t < n1; ++t) {
            std::size_t idx = sym_pack_cpp((std::size_t)t, (std::size_t)t, (std::size_t)n1);
            if ((int)idx < nPar) mask[(int)idx] = 1;
        }
    }
    return mask;
}

inline std::vector<unsigned char> build_diag_mask_right(StructChar b, int n3, int nPar) {
    std::vector<unsigned char> mask(nPar, 0);
    if (b == StructChar::U || b == StructChar::L || b == StructChar::S || b == StructChar::W) {
        for (int t = 0; t < n3; ++t) {
            std::size_t idx = sym_pack_cpp((std::size_t)t, (std::size_t)t, (std::size_t)n3);
            if ((int)idx < nPar) mask[(int)idx] = 1;
        }
    }
    return mask;
}

// Get list of diagonal parameter indices for a structure
inline std::vector<int> get_diag_indices(StructChar s, int n, int nPar) {
    std::vector<int> indices;
    if (s == StructChar::U || s == StructChar::L || s == StructChar::S || s == StructChar::W) {
        for (int t = 0; t < n; ++t) {
            int idx = (int)sym_pack_cpp((std::size_t)t, (std::size_t)t, (std::size_t)n);
            if (idx < nPar) indices.push_back(idx);
        }
    }
    // For g, k: return empty
    return indices;
}

// Build per-parameter column bitmasks for left_policy of structure a
inline std::vector<std::uint64_t> build_cols_mask_left(StructChar a, int n1, int n2, int nPar) {
    if (n2 > 64) {
        throw std::runtime_error("n2 must be <= 64 to use 64-bit column mask");
    }
    std::vector<std::uint64_t> mask(nPar, 0);

    for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
            bool ok = false;
            std::size_t alpha = 0;

            switch (a) {
                case StructChar::G:
                    alpha = (std::size_t)i * (std::size_t)n2 + (std::size_t)j;
                    ok = true;
                    break;
                case StructChar::U:
                    if (j >= i) { alpha = sym_pack_cpp((std::size_t)i, (std::size_t)j, (std::size_t)n1); ok = true; }
                    break;
                case StructChar::L:
                    if (j <= i) { alpha = sym_pack_cpp((std::size_t)i, (std::size_t)j, (std::size_t)n1); ok = true; }
                    break;
                case StructChar::S:
                    alpha = sym_pack_cpp((std::size_t)i, (std::size_t)j, (std::size_t)n1); ok = true;
                    break;
                case StructChar::K:
                    if (i != j) {
                        std::size_t p = (i < j) ? (std::size_t)i : (std::size_t)j;
                        std::size_t q = (i < j) ? (std::size_t)j : (std::size_t)i;
                        alpha = skew_pack_cpp(p, q, (std::size_t)n1);
                        ok = true;
                    }
                    break;
                case StructChar::W:
                    alpha = sym_pack_cpp((std::size_t)std::min(i, j),
                                         (std::size_t)std::max(i, j),
                                         (std::size_t)n1);
                    ok = true;
                    break;
            }

            if (ok && (int)alpha < nPar) {
                mask[(int)alpha] |= (1ULL << j);
            }
        }
    }
    return mask;
}

// ------------------------------
// Row-level predicates
// ------------------------------

// Check if all nonzeros in row are inside a boolean mask (0/1)
inline bool row_subset_mask(const std::vector<Rational>& row,
                            const std::vector<unsigned char>& mask) {
    const int n = (int)row.size();
    for (int t = 0; t < n; ++t) {
        if (row[t] != 0 && !mask[t]) return false;
    }
    return true;
}

// Compute intersection of column bitmasks across all nonzero entries
inline std::uint64_t row_columns_intersection(const std::vector<Rational>& row,
                                              const std::vector<std::uint64_t>& alpha_cols_mask,
                                              int n2) {
    if (row.empty()) return 0;
    std::uint64_t all_cols = (n2 == 64) ? ~0ULL : ((1ULL << n2) - 1ULL);
    std::uint64_t inter = all_cols;
    bool seen_nz = false;

    const int n = (int)row.size();
    for (int t = 0; t < n; ++t) {
        if (row[t] == 0) continue;
        seen_nz = true;
        inter &= alpha_cols_mask[t];
        if (inter == 0) return 0;
    }
    return seen_nz ? inter : 0;
}

// ------------------------------
// Triples and comparison
// ------------------------------

struct TripleKey {
    int a;
    int b;
    int c;

    bool operator<(const TripleKey& o) const {
        if (a != o.a) return a < o.a;
        if (b != o.b) return b < o.b;
        return c < o.c;
    }
    
    bool operator==(const TripleKey& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
};

// Check if t1 strictly dominates t2: all components >= and at least one >
inline bool triple_dominates(const TripleKey& t1, const TripleKey& t2) {
    if (t1 == t2) return false;
    return t1.a >= t2.a && t1.b >= t2.b && t1.c >= t2.c;
}

// Comparator for choosing best per triple: first by max_den (asc), then by nnz (asc)
struct BestByTripleCmp {
    I64 max_den;
    size_t nnz;

    bool operator<(const BestByTripleCmp& o) const {
        if (max_den != o.max_den) return max_den < o.max_den;
        return nnz < o.nnz;
    }
};

// Check if operation is symmetric (aa type: gg, uu, ll, ss, kk, ww)
inline bool is_symmetric_op(const std::string& op) {
    return op.size() == 2 && op[0] == op[1];
}

// Normalize triple for symmetric operations: (ab, ag+ga, 0)
inline TripleKey normalize_triple_symmetric(const TripleKey& key) {
    return {key.a, key.b + key.c, 0};
}

// ------------------------------
// Per-scheme triple calculators
// ------------------------------

// Compute (#ab, #ag, #bg) for an "ab"-operation scheme
// Special handling for 'g' structures:
// - For "xg": ag contributes to first number, bg does not count (gg recursion)
// - For "gx": bg contributes to first number, ag does not count (gg recursion)
// - For "gg": nothing counts
inline TripleKey scheme_triple_ab(const std::vector<Rational>& U,
                                  const std::vector<Rational>& V,
                                  int r, int nU, int nV,
                                  int n1, int n2, int n3,
                                  StructChar a, StructChar b,
                                  bool is_symmetric = false) {
    auto diagU = build_diag_mask_left(a, n1, nU);
    auto diagV = build_diag_mask_right(b, n3, nV);

    int ab = 0, ag = 0, bg = 0;

    for (int mu = 0; mu < r; ++mu) {
        std::vector<Rational> U_row(U.begin() + mu * nU, U.begin() + (mu + 1) * nU);
        std::vector<Rational> V_row(V.begin() + mu * nV, V.begin() + (mu + 1) * nV);

        bool left_diag_only  = row_subset_mask(U_row, diagU);
        bool right_diag_only = row_subset_mask(V_row, diagV);

        if (left_diag_only && right_diag_only) ++ab;
        else if (left_diag_only && !right_diag_only) ++ag;
        else if (!left_diag_only && right_diag_only) ++bg;
    }

    // Handle 'g' structures
    if (a == StructChar::G && b == StructChar::G) {
        // gg: all recursions are gg - not counted
        return {0, 0, 0};
    } else if (a != StructChar::G && b == StructChar::G) {
        // xg: ag gives xg recursion (same as operation), bg gives gg (not counted)
        return {ab + ag, 0, 0};
    } else if (a == StructChar::G && b != StructChar::G) {
        // gx: bg gives gx recursion (same as operation), ag gives gg (not counted)
        return {ab + bg, 0, 0};
    } else {
        // xx: both structures are not g, standard logic
        TripleKey key = {ab, ag, bg};
        if (is_symmetric) {
            key = normalize_triple_symmetric(key);
        }
        return key;
    }
}

// Compute (#at, #ag, #gt) for a "t"-operation scheme (C = Up(X X^T))
//
// Legacy model (no WDIAG):
//   flag_t: U_q == V_q (element-wise)
//   flag_a: at least one of U_q, V_q has all nonzeros on diagonal
//   Contributions:
//     flag_t && flag_a -> at
//     !flag_t && flag_a -> ag
//     flag_t && !flag_a -> gt
//
// WDIAG model (-DWDIAG):
//   flag_t: W_q has nonzeros only on diagonal output entries
//   flag_a: same as legacy (structure of X via U/V)
//   Contributions remain mapped via (flag_t, flag_a) as above.
inline TripleKey scheme_triple_t(const std::vector<Rational>& U,
                                 const std::vector<Rational>& V,
                                 const std::vector<Rational>& W,
                                 int r, int nU, int nV, int nW,
                                 int n1, int n2, int n3,
                                 StructChar s) {
    // For 'k' (skew), always return (0,0,0)
    if (s == StructChar::K) {
        return {0, 0, 0};
    }

    auto diag_U = get_diag_indices(s, n1, nU);
    auto diag_V = get_diag_indices(s, n3, nV);

#if defined(WDIAG)
    // Diagonal mask for W: result is Up(X X^T) over an n1 x n1 matrix,
    // so diagonal entries correspond to sym_pack_cpp(t, t, n1).
    std::vector<unsigned char> diagW_mask(nW, 0);
    for (int t = 0; t < n1; ++t) {
        std::size_t idx = sym_pack_cpp((std::size_t)t, (std::size_t)t, (std::size_t)n1);
        if ((int)idx < nW) {
            diagW_mask[(int)idx] = 1;
        }
    }
#endif

    int at = 0, ag = 0, gt = 0;

    for (int mu = 0; mu < r; ++mu) {
        std::vector<Rational> U_row(U.begin() + mu * nU, U.begin() + (mu + 1) * nU);
        std::vector<Rational> V_row(V.begin() + mu * nV, V.begin() + (mu + 1) * nV);
        std::vector<Rational> W_row(W.begin() + mu * nW, W.begin() + (mu + 1) * nW);

        bool flag_t = false;
#if defined(WDIAG)
        // WDIAG: term is t-recursive if it uses only diagonal outputs
        flag_t = row_subset_mask(W_row, diagW_mask);
#else
        // Legacy: term is t-recursive if U_q == V_q (Gram type)
        flag_t = (nU == nV) && std::equal(U_row.begin(), U_row.end(), V_row.begin());
#endif

        // Check flag_a: at least one row has all nonzeros on diagonal
        size_t U_nnz = count_nnz(U_row);
        size_t U_diag_nnz = 0;
        for (int idx : diag_U) {
            if (U_row[idx] != 0) ++U_diag_nnz;
        }
        bool flag_a_U = (U_nnz > 0 && U_nnz == U_diag_nnz);

        size_t V_nnz = count_nnz(V_row);
        size_t V_diag_nnz = 0;
        for (int idx : diag_V) {
            if (V_row[idx] != 0) ++V_diag_nnz;
        }
        bool flag_a_V = (V_nnz > 0 && V_nnz == V_diag_nnz);

        bool flag_a = flag_a_U || flag_a_V;

        // Count contributions
        if (flag_t && flag_a) {
            ++at;
        } else if (!flag_t && flag_a) {
            ++ag;
        } else if (flag_t && !flag_a) {
            ++gt;
        }
    }

    // Handle 'g' structure: at and gt both give gt recursion
    if (s == StructChar::G) {
        return {at + gt, 0, 0};
    } else {
        // ut, lt, st, wt: all three numbers have distinct meanings
        return {at, ag, gt};
    }
}

// ------------------------------
// Pareto front construction
// ------------------------------

struct SchemeData {
    size_t index;
    TripleKey triple;
    I64 max_den;
    size_t nnz;
    
    bool better_than(const SchemeData& other) const {
        BestByTripleCmp me{max_den, nnz};
        BestByTripleCmp them{other.max_den, other.nnz};
        return me < them;
    }
};

// Build Pareto front from schemes with their triples
inline std::vector<size_t> build_pareto_front(
    const std::vector<TripleKey>& triples,
    const std::vector<size_t>& scheme_indices,
    const std::vector<std::vector<Rational>>& schemes_U,
    const std::vector<std::vector<Rational>>& schemes_V,
    const std::vector<std::vector<Rational>>& schemes_W)
{
    if (triples.size() != scheme_indices.size()) {
        throw std::runtime_error("Mismatched triples and indices size");
    }
    
    std::vector<SchemeData> schemes;
    schemes.reserve(triples.size());
    
    for (size_t i = 0; i < triples.size(); ++i) {
        size_t idx = scheme_indices[i];
        I64 max_den = max_denominator(schemes_U[idx], schemes_V[idx], schemes_W[idx]);
        size_t nnz = count_nnz(schemes_U[idx]) + 
                     count_nnz(schemes_V[idx]) + 
                     count_nnz(schemes_W[idx]);
        schemes.push_back({idx, triples[i], max_den, nnz});
    }
    
    std::vector<SchemeData> pareto;
    
    for (const auto& candidate : schemes) {
        bool is_dominated = false;
        
        for (auto it = pareto.begin(); it != pareto.end(); ) {
            if (it->triple == candidate.triple) {
                if (candidate.better_than(*it)) {
                    it = pareto.erase(it);
                } else {
                    is_dominated = true;
                    break;
                }
            } else if (triple_dominates(it->triple, candidate.triple)) {
                is_dominated = true;
                break;
            } else if (triple_dominates(candidate.triple, it->triple)) {
                it = pareto.erase(it);
            } else {
                ++it;
            }
        }
        
        if (!is_dominated) {
            pareto.push_back(candidate);
        }
    }
    
    std::vector<size_t> result;
    result.reserve(pareto.size());
    for (const auto& s : pareto) {
        result.push_back(s.index);
    }
    
    return result;
}

// ------------------------------
// Best per triple (without histogram)
// ------------------------------

// Find best scheme for each unique triple (by min max_den, then min nnz)
inline std::map<TripleKey, size_t> find_best_per_triple_ab(
    const std::vector<std::vector<Rational>>& schemes_U,
    const std::vector<std::vector<Rational>>& schemes_V,
    const std::vector<std::vector<Rational>>& schemes_W,
    int r, int nU, int nV, int /*nW*/,
    int n1, int n2, int n3,
    StructChar a, StructChar b,
    bool is_symmetric = false)
{
    std::map<TripleKey, size_t> best_idx_by_triple;
    
    const size_t N = schemes_U.size();
    for (size_t idx = 0; idx < N; ++idx) {
        const auto& U = schemes_U[idx];
        const auto& V = schemes_V[idx];
        const auto& W = schemes_W[idx];

        TripleKey key = scheme_triple_ab(U, V, r, nU, nV, n1, n2, n3, a, b, is_symmetric);

        I64 md = max_denominator(U, V, W);
        size_t nnz = count_nnz(U) + count_nnz(V) + count_nnz(W);

        auto it = best_idx_by_triple.find(key);
        if (it == best_idx_by_triple.end()) {
            best_idx_by_triple.emplace(key, idx);
        } else {
            const auto& U_best = schemes_U[it->second];
            const auto& V_best = schemes_V[it->second];
            const auto& W_best = schemes_W[it->second];
            I64 md_best = max_denominator(U_best, V_best, W_best);
            size_t nnz_best = count_nnz(U_best) + count_nnz(V_best) + count_nnz(W_best);

            BestByTripleCmp cur{md, nnz};
            BestByTripleCmp old{md_best, nnz_best};
            if (cur < old) it->second = idx;
        }
    }
    
    return best_idx_by_triple;
}

// ------------------------------
// Operation string parsing
// ------------------------------

// Returns true if op is a "t"-operation like "gt","ut","st","kt","wt"
inline bool is_t_op(const std::string& op) {
    return (op.size() == 2 && op[1] == 't');
}

// Parse ab structures: op like "gg", "ul", "sw", ...
inline std::pair<StructChar, StructChar> parse_ab(const std::string& op) {
    if (op.size() != 2) throw std::runtime_error("Invalid ab op");
    return { (StructChar)op[0], (StructChar)op[1] };
}

// Parse t structure: op like "gt","ut","st","kt","wt"
inline StructChar parse_t(const std::string& op) {
    if (!is_t_op(op)) throw std::runtime_error("Invalid t op");
    return (StructChar)op[0];
}
