// utils.h
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cassert>
#include "field.h"
#include "cnpy.h"

using U8  = std::uint8_t;
using U64 = std::uint64_t;

// -----------------------------------------------------------------------------
// Packed (i,j,k) key helpers: 8 bits per index
// Layout: bits [0..7]=i, [8..15]=j, [16..23]=k
// -----------------------------------------------------------------------------

inline U64 pack_ijk(U8 i, U8 j, U8 k) {
    return (U64)i | ((U64)j << 8) | ((U64)k << 16);
}

inline void unpack_ijk(U64 key, U8& i, U8& j, U8& k) {
    i = (U8)( key        & 0xFFu);
    j = (U8)((key >> 8 ) & 0xFFu);
    k = (U8)((key >> 16) & 0xFFu);
}

// -----------------------------------------------------------------------------
// COO tensor dictionary and metadata
// -----------------------------------------------------------------------------

// COO dictionary: maps packed (i,j,k) -> value in [0..p-1] where p ∈ {2,3}.
using COODict = std::unordered_map<U64, U8>;

struct Dimensions {
    int du = 0; // dim(U)
    int dv = 0; // dim(V)
    int dw = 0; // dim(W)
};

struct TensorCOO {
    COODict    nz;     // non-zeros: key=(i,j,k) -> val mod p
    Dimensions dims;   // inferred as 1 + max along each axis
    int        mod = 2; // 2 for B2, 3 for B3
};

// -----------------------------------------------------------------------------
// Internal helpers for modulo normalization and accumulation
// -----------------------------------------------------------------------------

inline U8 mod_reduce_int8(int v, int mod) {
    if (mod == 2) {
        return (U8)(v & 1);
    } else if (mod == 3) {
        int t = v % 3;
        if (t < 0) t += 3;
        return (U8)t;
    } else {
        throw std::invalid_argument("Unsupported modulus (expected 2 or 3).");
    }
}

inline void mod_accumulate(U8& acc, U8 add, int mod) {
    if (mod == 2) {
        acc = (U8)((acc ^ (add & 1)) & 1);
    } else if (mod == 3) {
        int s = (int)acc + (int)add;
        if (s >= 3) s -= 3;
        acc = (U8)s;
    } else {
        throw std::invalid_argument("Unsupported modulus (expected 2 or 3).");
    }
}

// -----------------------------------------------------------------------------
// Loading and normalization
// -----------------------------------------------------------------------------

// Load a COO tensor from one .npy file of shape (nnz, 4) with int8 entries.
// Each row is [i, j, k, val]. Applies modulo 'mod' to values, drops zeros,
// verifies 0 <= i,j,k < 64, infers dims as 1 + max index on each axis.
inline TensorCOO load_coo_from_npy(const std::string& npy_path, int mod) {
    if (!(mod == 2 || mod == 3)) {
        throw std::invalid_argument("load_coo_from_npy: 'mod' must be 2 or 3.");
    }

    cnpy::NpyArray arr = cnpy::npy_load(npy_path);
    if (arr.word_size != sizeof(int8_t)) {
        throw std::runtime_error("load_coo_from_npy: dtype must be int8.");
    }
    if (arr.shape.size() != 2 || arr.shape[1] != 4) {
        throw std::runtime_error("load_coo_from_npy: expected 2D array with shape (nnz, 4).");
    }

    const size_t nnz = arr.shape[0];
    const int8_t* ptr = arr.data<int8_t>();

    TensorCOO T;
    T.mod = mod;

    int max_i = -1, max_j = -1, max_k = -1;

    for (size_t r = 0; r < nnz; ++r) {
        int i = (int)ptr[4*r + 0];
        int j = (int)ptr[4*r + 1];
        int k = (int)ptr[4*r + 2];
        int v = (int)ptr[4*r + 3];

        if (i < 0 || j < 0 || k < 0 || i >= 64 || j >= 64 || k >= 64) {
            throw std::runtime_error("load_coo_from_npy: index out of range [0,63].");
        }

        U8 vmod = mod_reduce_int8(v, mod);
        if (vmod == 0) continue;

        U64 key = pack_ijk((U8)i, (U8)j, (U8)k);

        auto it = T.nz.find(key);
        if (it == T.nz.end()) {
            T.nz.emplace(key, vmod);
        } else {
            mod_accumulate(it->second, vmod, mod);
            if (it->second == 0) {
                // Keep zeros or erase; both are fine. Choose erase to keep dict sparse.
                T.nz.erase(it);
            }
        }

        if (i > max_i) max_i = i;
        if (j > max_j) max_j = j;
        if (k > max_k) max_k = k;
    }

    // Infer dimensions (1 + max) or 0 if empty
    T.dims.du = (max_i >= 0) ? (max_i + 1) : 0;
    T.dims.dv = (max_j >= 0) ? (max_j + 1) : 0;
    T.dims.dw = (max_k >= 0) ? (max_k + 1) : 0;

    return T;
}

// Recompute dims from a COODict (1 + max index per axis).
inline Dimensions infer_dims_from_coo(const COODict& dict) {
    int max_i = -1, max_j = -1, max_k = -1;
    for (const auto& kv : dict) {
        U8 i, j, k;
        unpack_ijk(kv.first, i, j, k);
        if ((int)i > max_i) max_i = (int)i;
        if ((int)j > max_j) max_j = (int)j;
        if ((int)k > max_k) max_k = (int)k;
    }
    return Dimensions{
        (max_i >= 0) ? (max_i + 1) : 0,
        (max_j >= 0) ? (max_j + 1) : 0,
        (max_k >= 0) ? (max_k + 1) : 0
    };
}

// -----------------------------------------------------------------------------
// Field accessors (coefficients) and setters
// -----------------------------------------------------------------------------

// Coefficient getters at position 'idx' (0..63).
inline int get_coefficient(const B2& f, int idx) {
    return (int)((f.val >> idx) & 1ULL);
}

inline int get_coefficient(const B3& f, int idx) {
    int lo_bit = (int)((f.lo >> idx) & 1ULL);
    int hi_bit = (int)((f.hi >> idx) & 1ULL);
    return lo_bit + 2 * hi_bit; // 0,1,2
}

// Coefficient setters at position 'idx' (write value modulo p).
inline void set_coefficient(B2& f, int idx, int value_mod2) {
    U64 mask = (1ULL << idx);
    if (value_mod2 & 1) {
        f.val |= mask;
    } else {
        f.val &= ~mask;
    }
}

inline void set_coefficient(B3& f, int idx, int value_mod3) {
    U64 mask = (1ULL << idx);
    // Clear both planes at idx
    f.lo &= ~mask;
    f.hi &= ~mask;
    // Set plane based on value
    if (value_mod3 == 1) {
        f.lo |= mask;
    } else if (value_mod3 == 2) {
        f.hi |= mask;
    }
}

// Optional helpers: single-bit index utilities.
// For B2: returns -1 if zero; undefined behavior if multiple bits are set.
inline int single_bit_index(const B2& x) {
#if defined(__GNUG__) || defined(__clang__)
    if (x.val == 0ULL) return -1;
    return __builtin_ctzll(x.val);
#else
    if (x.val == 0ULL) return -1;
    // Portable fallback: scan least-significant set bit
    U64 v = x.val;
    int idx = 0;
    while ((v & 1ULL) == 0ULL) { v >>= 1; ++idx; }
    return idx;
#endif
}

// For B3 used as a binary vector (lo carries bits, hi == 0).
inline int single_bit_index(const B3& x) {
    assert(x.hi == 0ULL && "single_bit_index(B3): expected hi==0 for binary vector.");
    B2 t{x.lo};
    return single_bit_index(t);
}

// -----------------------------------------------------------------------------
// Trivial rank-1 decomposition from COO
// -----------------------------------------------------------------------------

template<typename Field>
inline std::vector<Field> generate_trivial_decomposition(const TensorCOO& T);

// B2 specialization
template<>
inline std::vector<B2> generate_trivial_decomposition<B2>(const TensorCOO& T) {
    if (T.mod != 2) {
        throw std::invalid_argument("generate_trivial_decomposition<B2>: TensorCOO.mod must be 2.");
    }

    std::vector<B2> data;
    data.reserve(T.nz.size() * 3);

    for (const auto& kv : T.nz) {
        const U64 key = kv.first;
        const U8  val = kv.second; // 1 only (mod 2)
        if (val == 0) continue;

        U8 i, j, k;
        unpack_ijk(key, i, j, k);

        B2 u(0), v(0), w(0);
        set_coefficient(u, (int)i, 1);
        set_coefficient(v, (int)j, 1);
        set_coefficient(w, (int)k, 1); // val==1 guaranteed

        data.push_back(u);
        data.push_back(v);
        data.push_back(w);
    }

    return data;
}

// B3 specialization
template<>
inline std::vector<B3> generate_trivial_decomposition<B3>(const TensorCOO& T) {
    if (T.mod != 3) {
        throw std::invalid_argument("generate_trivial_decomposition<B3>: TensorCOO.mod must be 3.");
    }

    std::vector<B3> data;
    data.reserve(T.nz.size() * 3);

    for (const auto& kv : T.nz) {
        const U64 key = kv.first;
        const U8  val = kv.second; // 1 or 2 (mod 3)
        if (val == 0) continue;

        U8 i, j, k;
        unpack_ijk(key, i, j, k);

        B3 u{0,0}, v{0,0}, w{0,0};
        set_coefficient(u, (int)i, 1);          // e_i (binary in lo plane)
        set_coefficient(v, (int)j, 1);          // e_j (binary in lo plane)
        set_coefficient(w, (int)k, (int)val);   // val * e_k (1 -> lo, 2 -> hi)

        data.push_back(u);
        data.push_back(v);
        data.push_back(w);
    }

    return data;
}

// Rank (number of non-zero terms), counting terms whose 'u' is not zero.
template<typename Field>
inline int get_rank(const std::vector<Field>& data) {
    const int n_terms = (int)data.size() / 3;
    int rank = 0;
    for (int t = 0; t < n_terms; ++t) {
        if (!data[3*t].is_zero()) ++rank;
    }
    return rank;
}

// -----------------------------------------------------------------------------
// Verification: reconstruct S from the scheme and compare with COO tensor T
// -----------------------------------------------------------------------------

template<typename Field>
inline bool verify_scheme(const std::vector<Field>& data, const TensorCOO& T) {
    const int mod = field_traits<Field>::is_mod2 ? 2 : 3;

    // Reconstruct S(i,j,k) = sum_l u_l[i]*v_l[j]*w_l[k] mod p
    COODict S;
    const int r = (int)data.size() / 3;

    for (int l = 0; l < r; ++l) {
        const Field& u = data[3*l + 0];
        const Field& v = data[3*l + 1];
        const Field& w = data[3*l + 2];

        if (u.is_zero() || v.is_zero() || w.is_zero()) continue;

        // Iterate over supports; dimensions are bounded by 64.
        for (int i = 0; i < T.dims.du; ++i) {
            int cu = get_coefficient(u, i);
            if (!cu) continue;

            for (int j = 0; j < T.dims.dv; ++j) {
                int cv = get_coefficient(v, j);
                if (!cv) continue;

                for (int k = 0; k < T.dims.dw; ++k) {
                    int cw = get_coefficient(w, k);
                    if (!cw) continue;

                    int prod = (cu * cv) % mod;
                    prod = (prod * cw) % mod;
                    if (prod == 0) continue;

                    U64 key = pack_ijk((U8)i, (U8)j, (U8)k);
                    U8& acc = S[key]; // default-initialized to 0 on first access
                    mod_accumulate(acc, (U8)prod, mod);
                    if (acc == 0) {
                        // optional: keep zeros removed for sparsity; not required
                        // S.erase(key);
                    }
                }
            }
        }
    }

    // Compare S with T: exact equality modulo 'mod'
    // 1) All keys in T must match in S.
    for (const auto& kv : T.nz) {
        const U64 key = kv.first;
        const U8  tv  = kv.second; // already mod-reduced and non-zero
        auto it = S.find(key);
        U8 sv = (it == S.end()) ? 0 : it->second;
        if (sv != tv) return false;
    }

    // 2) No extra non-zeros in S that are not in T.
    for (const auto& kv : S) {
        if (kv.second == 0) continue;
        if (T.nz.find(kv.first) == T.nz.end()) return false;
    }

    return true;
}
