// hensel.h
// Hensel lifting for tensor decomposition over F2 and F3

#pragma once

#include <vector>
#include <cstdint>
#include <cassert>

#ifdef MOD3
    #include "ref3.h"
    using BitMatrix = BitMatrix3;
    using RefFactor = RefFactor3;
    constexpr int FIELD_MOD = 3;
#else
    #include "ref2.h"
    using BitMatrix = BitMatrix2;
    using RefFactor = RefFactor2;
    constexpr int FIELD_MOD = 2;
#endif

using U8 = std::uint8_t;
using U64 = std::uint64_t;
using I64 = std::int64_t;

// Compute tensor T = U ⊗ V ⊗ W (same for both F2 and F3)
// U: (r x nU), V: (r x nV), W: (r x nW) row-major
// T: (nU x nV x nW) with k fastest (C-order)
inline void compute_tensor(
    const std::vector<U64>& U, const std::vector<U64>& V, const std::vector<U64>& W,
    int r, int nU, int nV, int nW,
    std::vector<I64>& T)
{
    assert(r > 0 && nU > 0 && nV > 0 && nW > 0);
    assert(U.size() == (size_t)(r * nU));
    assert(V.size() == (size_t)(r * nV));
    assert(W.size() == (size_t)(r * nW));
    
    T.assign(nU * nV * nW, 0);
    
    // Reordered loops for better cache locality
    for (int mu = 0; mu < r; ++mu) {
        for (int i = 0; i < nU; ++i) {
            I64 u = (I64)U[mu * nU + i];
            for (int j = 0; j < nV; ++j) {
                I64 uv = u * (I64)V[mu * nV + j];
                for (int k = 0; k < nW; ++k) {
                    T[i * (nV * nW) + j * nW + k] += uv * (I64)W[mu * nW + k];
                }
            }
        }
    }
}

#ifdef MOD3

// Build Jacobian matrix over F3 using factors mod 3
inline BitMatrix3 build_jacobian_f3(
    const std::vector<U64>& U, const std::vector<U64>& V, const std::vector<U64>& W,
    int r, int nU, int nV, int nW)
{
    assert(r > 0 && nU > 0 && nV > 0 && nW > 0);
    
    int rows = nU * nV * nW;
    int cols = r * (nU + nV + nW);
    BitMatrix3 J(rows, cols);
    
    int v_offset = r * nU;
    int w_offset = r * (nU + nV);
    
    for (int i = 0; i < nU; ++i) {
        for (int j = 0; j < nV; ++j) {
            for (int k = 0; k < nW; ++k) {
                int row = i * (nV * nW) + j * nW + k;
                
                for (int mu = 0; mu < r; ++mu) {
                    // U-block: J[row, i*r + mu] = V[mu,j] * W[mu,k] mod 3
                    U64 vw = (V[mu * nV + j] * W[mu * nW + k]) % 3;
                    if (vw != 0)
                        J.set(row, i * r + mu, vw);
                    
                    // V-block: J[row, r*nU + j*r + mu] = U[mu,i] * W[mu,k] mod 3
                    U64 uw = (U[mu * nU + i] * W[mu * nW + k]) % 3;
                    if (uw != 0)
                        J.set(row, v_offset + j * r + mu, uw);
                    
                    // W-block: J[row, r*(nU+nV) + k*r + mu] = U[mu,i] * V[mu,j] mod 3
                    U64 uv = (U[mu * nU + i] * V[mu * nV + j]) % 3;
                    if (uv != 0)
                        J.set(row, w_offset + k * r + mu, uv);
                }
            }
        }
    }
    
    return J;
}

// Update factor matrix with lifting correction (F3 version)
// exponent: current power of 3 (F[i] is known mod 3^(exponent+1) after update)
inline void update_factor_f3(
    std::vector<U64>& F, int r, int nF, 
    const std::vector<U8>& x, int x_offset,
    int exponent)
{
    assert(r > 0 && nF > 0 && exponent > 0);
    assert(F.size() == (size_t)(r * nF));
    
    // Compute 3^(exponent+1) for masking
    I64 mod_power = 1;
    for (int p = 0; p <= exponent; ++p) mod_power *= 3;
    
    // Compute 3^exponent (multiplier for correction)
    I64 correction_mult = mod_power / 3;
    
    for (int idx = 0; idx < nF; ++idx) {
        for (int mu = 0; mu < r; ++mu) {
            I64 current = (I64)F[mu * nF + idx];
            I64 correction = (I64)x[x_offset + idx * r + mu];
            I64 updated = current + correction * correction_mult;
            
            // Keep result in range [0, mod_power)
            updated = ((updated % mod_power) + mod_power) % mod_power;
            F[mu * nF + idx] = (U64)updated;
        }
    }
}

#else

// Build Jacobian matrix over F2 using factors mod 2
inline BitMatrix2 build_jacobian_f2(
    const std::vector<U64>& U, const std::vector<U64>& V, const std::vector<U64>& W,
    int r, int nU, int nV, int nW)
{
    assert(r > 0 && nU > 0 && nV > 0 && nW > 0);
    
    int rows = nU * nV * nW;
    int cols = r * (nU + nV + nW);
    BitMatrix2 J(rows, cols);
    
    int v_offset = r * nU;
    int w_offset = r * (nU + nV);
    
    for (int i = 0; i < nU; ++i) {
        for (int j = 0; j < nV; ++j) {
            for (int k = 0; k < nW; ++k) {
                int row = i * (nV * nW) + j * nW + k;
                
                for (int mu = 0; mu < r; ++mu) {
                    // U-block: J[row, i*r + mu] = V[mu,j] * W[mu,k] mod 2
                    if ((V[mu * nV + j] & W[mu * nW + k]) & 1)
                        J.flip(row, i * r + mu);
                    
                    // V-block: J[row, r*nU + j*r + mu] = U[mu,i] * W[mu,k] mod 2
                    if ((U[mu * nU + i] & W[mu * nW + k]) & 1)
                        J.flip(row, v_offset + j * r + mu);
                    
                    // W-block: J[row, r*(nU+nV) + k*r + mu] = U[mu,i] * V[mu,j] mod 2
                    if ((U[mu * nU + i] & V[mu * nV + j]) & 1)
                        J.flip(row, w_offset + k * r + mu);
                }
            }
        }
    }
    
    return J;
}

// Update factor matrix with lifting correction (F2 version)
inline void update_factor_f2(
    std::vector<U64>& F, int r, int nF, 
    const std::vector<U8>& x, int x_offset,
    int exponent)
{
    assert(r > 0 && nF > 0 && exponent > 0);
    assert(F.size() == (size_t)(r * nF));
    
    I64 mask = (I64(1) << (exponent + 1)) - 1;
    
    for (int idx = 0; idx < nF; ++idx) {
        for (int mu = 0; mu < r; ++mu) {
            I64 upd = (I64)F[mu * nF + idx] + ((I64)x[x_offset + idx * r + mu] << exponent);
            F[mu * nF + idx] = (U64)(upd & mask);
        }
    }
}

#endif

// Perform k Hensel lifting steps for tensor decomposition
inline bool lift_tensor(
    const std::vector<I64>& T0,
    std::vector<U64>& U, std::vector<U64>& V, std::vector<U64>& W,
    int r, int nU, int nV, int nW,
    int k_steps)
{
    assert(r > 0 && nU > 0 && nV > 0 && nW > 0 && k_steps > 0);
    assert(T0.size() == (size_t)(nU * nV * nW));
    assert(U.size() == (size_t)(r * nU));
    assert(V.size() == (size_t)(r * nV));
    assert(W.size() == (size_t)(r * nW));
    
    int exponent = 1; // Start at 2^1 or 3^1
    int tensor_size = nU * nV * nW;
    int n_vars = r * (nU + nV + nW);
    
    // Pre-allocate buffers (reused across iterations)
    std::vector<I64> E;
    E.reserve(tensor_size);
    std::vector<U8> b(tensor_size);
    std::vector<U8> x(n_vars);
    
    // Verify initial scheme mod FIELD_MOD
    compute_tensor(U, V, W, r, nU, nV, nW, E);
    
#ifdef MOD3
    for (int idx = 0; idx < tensor_size; ++idx) {
        I64 t0_mod = ((T0[idx] % 3) + 3) % 3;
        I64 e_mod = ((E[idx] % 3) + 3) % 3;
        if (t0_mod != e_mod) return false;
    }
#else
    for (int idx = 0; idx < tensor_size; ++idx) {
        if ((T0[idx] & 1) != (E[idx] & 1)) return false;
    }
#endif
    
    // Build and factorize Jacobian once
#ifdef MOD3
    BitMatrix J = build_jacobian_f3(U, V, W, r, nU, nV, nW);
#else
    BitMatrix J = build_jacobian_f2(U, V, W, r, nU, nV, nW);
#endif
    
    RefFactor ref;
    ref.factorize(J);
    
    // Hensel lifting loop
    for (int step = 0; step < k_steps; ++step) {
        // Compute E = U ⊗ V ⊗ W
        compute_tensor(U, V, W, r, nU, nV, nW, E);
        
#ifdef MOD3
        // Compute modulus for current step: 3^exponent
        I64 divisor = 1;
        for (int p = 0; p < exponent; ++p) divisor *= 3;
        
        // Compute RHS: b = ((T0 - E) / 3^exponent) mod 3
        for (int idx = 0; idx < tensor_size; ++idx) {
            I64 residual = T0[idx] - E[idx];
            I64 quotient = residual / divisor;
            b[idx] = (U8)(((quotient % 3) + 3) % 3);
        }
#else
        // Compute RHS: b = ((T0 - E) >> exponent) & 1
        for (int idx = 0; idx < tensor_size; ++idx) {
            I64 residual = T0[idx] - E[idx];
            b[idx] = U8((residual >> exponent) & 1);
        }
#endif
        
        // Solve J @ x = b
        if (!ref.solve(b, x)) return false;
        
        // Update factors
#ifdef MOD3
        update_factor_f3(U, r, nU, x, 0, exponent);
        update_factor_f3(V, r, nV, x, r * nU, exponent);
        update_factor_f3(W, r, nW, x, r * (nU + nV), exponent);
#else
        update_factor_f2(U, r, nU, x, 0, exponent);
        update_factor_f2(V, r, nV, x, r * nU, exponent);
        update_factor_f2(W, r, nW, x, r * (nU + nV), exponent);
#endif
        
        ++exponent;
    }
    
    return true;
}