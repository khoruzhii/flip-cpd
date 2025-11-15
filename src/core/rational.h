// rational.h
// Rational arithmetic and reconstruction for tensor decomposition

#pragma once

#include <cstdint>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

using I64 = std::int64_t;
using U64 = std::uint64_t;

// Rational number p/q with q > 0 (sign in numerator)
struct Rational {
    I64 p;
    U64 q;
    
    Rational(I64 num = 0, I64 den = 1) : p(num), q(std::abs(den)) {
        // if (den == 0) throw std::invalid_argument("Denominator cannot be zero");
        if (den < 0) p = -p;
        normalize();
    }
    
    void normalize() {
        if (p == 0) {
            q = 1;
            return;
        }
        U64 g = std::gcd((U64)std::abs(p), q);
        if (g > 1) {
            p /= (I64)g;
            q /= g;
        }
    }
    
    I64 numerator() const { return p; }
    I64 denominator() const { return (I64)q; }
    
    Rational operator-() const {
        return Rational(-p, (I64)q);
    }
    
    Rational operator+(const Rational& r) const {
        I64 num = p * (I64)r.q + r.p * (I64)q;
        I64 den = (I64)q * (I64)r.q;
        return Rational(num, den);
    }
    
    Rational operator-(const Rational& r) const {
        return *this + (-r);
    }
    
    Rational operator*(const Rational& r) const {
        // Cross-cancel to avoid overflow: (p1/q1) * (p2/q2) = (p1/q2) * (p2/q1)
        U64 g1 = std::gcd((U64)std::abs(p), r.q);
        U64 g2 = std::gcd((U64)std::abs(r.p), q);
        
        I64 num = (p / (I64)g1) * (r.p / (I64)g2);
        I64 den = (I64)(q / g2) * (I64)(r.q / g1);
        return Rational(num, den);
    }
    
    Rational operator/(const Rational& r) const {
        if (r.p == 0) throw std::invalid_argument("Division by zero");
        return *this * Rational((I64)r.q, r.p);
    }
    
    Rational& operator*=(const Rational& r) {
        *this = *this * r;
        return *this;
    }
    
    Rational& operator/=(const Rational& r) {
        *this = *this / r;
        return *this;
    }
    
    bool operator==(const Rational& r) const {
        return p == r.p && q == r.q;
    }
    
    bool operator==(I64 n) const {
        return p == n && q == 1;
    }
    
    bool operator!=(const Rational& r) const {
        return !(*this == r);
    }
    
    bool operator!=(I64 n) const {
        return !(*this == n);
    }
};

// Extended GCD: returns (gcd, x, y) where ax + by = gcd(a,b)
inline std::tuple<I64, I64, I64> extended_gcd(I64 a, I64 b) {
    if (b == 0) return {a, 1, 0};
    
    I64 x0 = 1, x1 = 0;
    I64 y0 = 0, y1 = 1;
    I64 r0 = a, r1 = b;
    
    while (r1 != 0) {
        I64 q = r0 / r1;
        I64 r2 = r0 - q * r1;
        I64 x2 = x0 - q * x1;
        I64 y2 = y0 - q * y1;
        
        r0 = r1; r1 = r2;
        x0 = x1; x1 = x2;
        y0 = y1; y1 = y2;
    }
    
    return {r0, x0, y0};
}

// Modular inverse: returns x such that ax ≡ 1 (mod m)
inline I64 mod_inverse(I64 a, I64 m) {
    auto [g, x, y] = extended_gcd(a, m);
    if (g != 1) {
        throw std::runtime_error("Modular inverse does not exist");
    }
    return ((x % m) + m) % m;
}

// Default bound for rational reconstruction: floor(sqrt(M/2))
inline I64 default_bound(I64 M) {
    return (I64)std::floor(std::sqrt((double)M / 2.0));
}

// Rational reconstruction result
struct RatResult {
    I64 p, q;
    bool success;
};

// Rational reconstruction using Extended GCD
// Given: a (mod M), find p/q such that a*q ≡ p (mod M) with |p|, |q| <= bound
inline RatResult rational_reconstruct(I64 a, I64 M, I64 bound) {
    a = ((a % M) + M) % M;
    
    I64 r0 = M, r1 = a;
    I64 t0 = 0, t1 = 1;
    
    while (r1 != 0 && r1 > bound) {
        I64 q = r0 / r1;
        I64 r2 = r0 - q * r1;
        I64 t2 = t0 - q * t1;
        
        r0 = r1; r1 = r2;
        t0 = t1; t1 = t2;
    }
    
    I64 p = r1;
    I64 q = t1;
    
    // Check bounds and validity
    if (std::abs(p) > bound || std::abs(q) > bound || q == 0) {
        return {0, 1, false};
    }
    
    // Ensure q > 0 and gcd(p,q) = 1
    if (q < 0) {
        p = -p;
        q = -q;
    }
    
    if (std::gcd(std::abs(p), std::abs(q)) != 1) {
        return {0, 1, false};
    }
    
    return {p, q, true};
}

// GCD of absolute values of numerators (ignoring zeros)
inline I64 gcd_numerators(const std::vector<Rational>& arr) {
    I64 result = 0;
    for (const auto& r : arr) {
        I64 num = r.numerator();
        if (num != 0) {
            result = std::gcd(result, std::abs(num));
        }
    }
    return (result == 0) ? 1 : result;
}

// LCM of denominators
inline I64 lcm_denominators(const std::vector<Rational>& arr) {
    I64 result = 1;
    for (const auto& r : arr) {
        I64 den = r.denominator();
        result = std::lcm(result, den);
    }
    return result;
}

// Canonicalize rank-1 component: extract common factors from V,W into U and normalize signs
inline void canonicalize_component(
    std::vector<Rational>& U_row,
    std::vector<Rational>& V_row,
    std::vector<Rational>& W_row
) {
    // Extract common factor from V: c_v = gcd(nums) / lcm(dens)
    I64 g_vn = gcd_numerators(V_row);
    I64 l_vd = lcm_denominators(V_row);
    Rational c_v(g_vn, l_vd);
    
    for (auto& v : V_row) v /= c_v;
    for (auto& u : U_row) u *= c_v;
    
    // Extract common factor from W
    I64 g_wn = gcd_numerators(W_row);
    I64 l_wd = lcm_denominators(W_row);
    Rational c_w(g_wn, l_wd);
    
    for (auto& w : W_row) w /= c_w;
    for (auto& u : U_row) u *= c_w;
    
    // Normalize sign: first nonzero V numerator positive
    auto it_v = std::find_if(V_row.begin(), V_row.end(), 
                             [](const Rational& r) { return r != 0; });
    if (it_v != V_row.end() && it_v->numerator() < 0) {
        for (auto& v : V_row) v = -v;
        for (auto& u : U_row) u = -u;
    }
    
    // Normalize sign: first nonzero U numerator positive (flip W if needed)
    auto it_u = std::find_if(U_row.begin(), U_row.end(),
                             [](const Rational& r) { return r != 0; });
    if (it_u != U_row.end() && it_u->numerator() < 0) {
        for (auto& u : U_row) u = -u;
        for (auto& w : W_row) w = -w;
    }
}