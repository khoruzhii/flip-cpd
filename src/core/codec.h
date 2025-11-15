// codec.h
// Utilities for tensor operations and scheme encoding/decoding (real and complex-aware)
#pragma once

#include "cnpy.h"
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cassert>

using U64 = std::uint64_t;
using I64 = std::int64_t;
using I8  = std::int8_t;

// Compile-time modulus
#ifdef MOD3
    constexpr int MOD = 3;
#else
    constexpr int MOD = 2;
#endif

// Path constants
namespace paths {
    constexpr const char* SCHEMES_DIR = "data/schemes_modp";
    constexpr const char* TENSORS_DIR = "data/tensors";
}

// Expand bits from uint64 vector (F2). Interprets each r-length vector as r packed bitfields of width 'width'.
inline void expand_bits(const U64* vec, int r, int width, std::vector<U64>& out) {
    out.resize(static_cast<size_t>(r) * width);
    for (int mu = 0; mu < r; ++mu) {
        U64 val = vec[mu];
        for (int b = 0; b < width; ++b) {
            out[static_cast<size_t>(mu) * width + b] = (val >> b) & 1;
        }
    }
}

// Unpack scheme over F2: row = [u, v, w, u, v, w, ...], each entry is a packed 64-bit vector
inline void unpack_scheme_f2(const U64* row, int row_len, int nU, int nV, int nW,
                             std::vector<U64>& U, std::vector<U64>& V, std::vector<U64>& W) {
    int r = row_len / 3;

    std::vector<U64> u_vec(r), v_vec(r), w_vec(r);
    for (int i = 0; i < r; ++i) {
        u_vec[i] = row[3 * i];
        v_vec[i] = row[3 * i + 1];
        w_vec[i] = row[3 * i + 2];
    }

    expand_bits(u_vec.data(), r, nU, U);
    expand_bits(v_vec.data(), r, nV, V);
    expand_bits(w_vec.data(), r, nW, W);
}

// Pack lifted scheme: [U | V | W] flat layout (expanded, not packed)
inline void pack_lifted_scheme(const std::vector<U64>& U,
                               const std::vector<U64>& V,
                               const std::vector<U64>& W,
                               std::vector<U64>& row) {
    row.clear();
    row.reserve(U.size() + V.size() + W.size());
    row.insert(row.end(), U.begin(), U.end());
    row.insert(row.end(), V.begin(), V.end());
    row.insert(row.end(), W.begin(), W.end());
}

// Load sparse tensor from (nnz, 4) format into flat 3D array T[alpha][beta][gamma]
inline void load_sparse_tensor(const std::string& path, int nU, int nV, int nW,
                               std::vector<I64>& T) {
    cnpy::NpyArray arr = cnpy::npy_load(path);

    if (arr.shape.size() != 2 || arr.shape[1] != 4) {
        throw std::runtime_error(
            "Tensor must have shape (nnz, 4), got (" +
            std::to_string(arr.shape[0]) + ", " +
            std::to_string(arr.shape.size() > 1 ? arr.shape[1] : 0) + ")");
    }

    size_t nnz = arr.shape[0];
    I8* data = arr.data<I8>();

    T.assign(static_cast<size_t>(nU) * nV * nW, 0);

    for (size_t e = 0; e < nnz; ++e) {
        int alpha = static_cast<int>(data[e * 4 + 0]);
        int beta  = static_cast<int>(data[e * 4 + 1]);
        int gamma = static_cast<int>(data[e * 4 + 2]);
        I8  val   = data[e * 4 + 3];

        int idx = alpha * (nV * nW) + beta * nW + gamma;
        T[static_cast<size_t>(idx)] += static_cast<I64>(val);
    }
}

// Expand two bit planes (lo, hi) into F3 values {0,1,2}
// Encoding: 0=00, 1=01, 2=10
inline void expand_bits_f3(const U64* lo_vec, const U64* hi_vec,
                           int r, int width, std::vector<U64>& out) {
    assert(width <= 64 && "width must fit in 64 bits");

    out.resize(static_cast<size_t>(r) * width);

    for (int mu = 0; mu < r; ++mu) {
        U64 lo = lo_vec[mu];
        U64 hi = hi_vec[mu];

        for (int b = 0; b < width; ++b) {
            U64 lo_bit = (lo >> b) & 1;
            U64 hi_bit = (hi >> b) & 1;
            out[static_cast<size_t>(mu) * width + b] = lo_bit | (hi_bit << 1);  // 0,1,2
        }
    }
}

// Unpack F3 scheme from .npy format: [u.lo, u.hi, v.lo, v.hi, w.lo, w.hi, ...]
// row_len must be r*6 (6 U64s per term)
inline void unpack_scheme_f3(const U64* row, int row_len,
                             int nU, int nV, int nW,
                             std::vector<U64>& U,
                             std::vector<U64>& V,
                             std::vector<U64>& W) {
    assert(row_len % 6 == 0 && "row_len must be divisible by 6 for F3");

    int r = row_len / 6;

    std::vector<U64> u_lo(r), u_hi(r);
    std::vector<U64> v_lo(r), v_hi(r);
    std::vector<U64> w_lo(r), w_hi(r);

    for (int mu = 0; mu < r; ++mu) {
        u_lo[mu] = row[6 * mu + 0];
        u_hi[mu] = row[6 * mu + 1];
        v_lo[mu] = row[6 * mu + 2];
        v_hi[mu] = row[6 * mu + 3];
        w_lo[mu] = row[6 * mu + 4];
        w_hi[mu] = row[6 * mu + 5];
    }

    expand_bits_f3(u_lo.data(), u_hi.data(), r, nU, U);
    expand_bits_f3(v_lo.data(), v_hi.data(), r, nV, V);
    expand_bits_f3(w_lo.data(), w_hi.data(), r, nW, W);
}

// Pack F3 scheme back to compressed format (for saving)
// Input: U, V, W as expanded vectors (r*nU, r*nV, r*nW elements in {0,1,2})
// Output: row as [u.lo, u.hi, v.lo, v.hi, w.lo, w.hi, ...]
inline void pack_scheme_f3(const std::vector<U64>& U, int r, int nU,
                           const std::vector<U64>& V, int nV,
                           const std::vector<U64>& W, int nW,
                           std::vector<U64>& row) {
    assert(U.size() == static_cast<size_t>(r * nU));
    assert(V.size() == static_cast<size_t>(r * nV));
    assert(W.size() == static_cast<size_t>(r * nW));

    row.clear();
    row.resize(static_cast<size_t>(r) * 6, 0);

    // Pack U
    for (int mu = 0; mu < r; ++mu) {
        U64 lo = 0, hi = 0;
        for (int i = 0; i < nU; ++i) {
            U64 val = U[static_cast<size_t>(mu) * nU + i] % 3;
            if (val & 1) lo |= (U64(1) << i);
            if (val & 2) hi |= (U64(1) << i);
        }
        row[6 * mu + 0] = lo;
        row[6 * mu + 1] = hi;
    }

    // Pack V
    for (int mu = 0; mu < r; ++mu) {
        U64 lo = 0, hi = 0;
        for (int j = 0; j < nV; ++j) {
            U64 val = V[static_cast<size_t>(mu) * nV + j] % 3;
            if (val & 1) lo |= (U64(1) << j);
            if (val & 2) hi |= (U64(1) << j);
        }
        row[6 * mu + 2] = lo;
        row[6 * mu + 3] = hi;
    }

    // Pack W
    for (int mu = 0; mu < r; ++mu) {
        U64 lo = 0, hi = 0;
        for (int k = 0; k < nW; ++k) {
            U64 val = W[static_cast<size_t>(mu) * nW + k] % 3;
            if (val & 1) lo |= (U64(1) << k);
            if (val & 2) hi |= (U64(1) << k);
        }
        row[6 * mu + 4] = lo;
        row[6 * mu + 5] = hi;
    }
}
