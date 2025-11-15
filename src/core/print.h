// print.h
// Scheme formatting and printing utilities

#pragma once

#include "rational.h"
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdexcept>

// Format coefficient magnitude without sign
inline std::string fmt_coeff_magnitude(I64 num, I64 den) {
    I64 abs_num = std::abs(num);
    
    if (den == 1) {
        if (abs_num == 1) return "";  // implicit 1
        return std::to_string(abs_num);
    }
    
    return std::to_string(abs_num) + "/" + std::to_string(den);
}

// Format linear form like "a1 + 2a3 - 5/2*a7"
inline std::string format_linform(const std::vector<Rational>& row, char symbol, bool with_parens) {
    std::vector<std::pair<bool, std::string>> parts;
    
    for (size_t j = 0; j < row.size(); ++j) {
        if (row[j] == 0) continue;
        
        I64 num = row[j].numerator();
        I64 den = row[j].denominator();
        
        bool negative = (num < 0);
        std::string mag = fmt_coeff_magnitude(num, den);
        std::string var = symbol + std::to_string(j + 1);
        
        std::string term = mag + var;
        parts.push_back({negative, term});
    }
    
    if (parts.empty()) return "0";
    
    std::string out;
    bool first_neg = parts[0].first;
    out = (first_neg ? "-" : "") + parts[0].second;
    
    for (size_t i = 1; i < parts.size(); ++i) {
        out += (parts[i].first ? " - " : " + ") + parts[i].second;
    }
    
    if (with_parens) return "(" + out + ")";
    return out;
}

// Print scheme to text file
inline void print_scheme(const std::string& path,
                         const std::vector<Rational>& U,
                         const std::vector<Rational>& V,
                         const std::vector<Rational>& W,
                         int r, int nU, int nV, int nW,
                         bool semicolon) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
    
    std::string line_end = semicolon ? ";\n" : "\n";
    
    // m-lines: m[i] = (linear form of a's)(linear form of b's)
    std::vector<std::string> m_names;
    std::vector<std::string> m_rhs;
    
    for (int i = 0; i < r; ++i) {
        m_names.push_back("m" + std::to_string(i + 1));
        
        std::vector<Rational> U_row(U.begin() + i * nU, U.begin() + (i + 1) * nU);
        std::vector<Rational> V_row(V.begin() + i * nV, V.begin() + (i + 1) * nV);
        
        std::string la = format_linform(U_row, 'a', true);
        std::string lb = format_linform(V_row, 'b', true);
        
        std::string prod = (la == "0" || lb == "0") ? "0" : la + lb;
        m_rhs.push_back(prod);
    }
    
    // Find max m_name width for alignment
    size_t m_col = 0;
    for (const auto& name : m_names) m_col = std::max(m_col, name.size());
    
    // Write m-lines
    for (int i = 0; i < r; ++i) {
        f << m_names[i];
        for (size_t j = m_names[i].size(); j < m_col; ++j) f << ' ';
        f << " = " << m_rhs[i] << line_end;
    }
    
    f << "\n";
    
    // c-lines: c[g] = linear form of m's
    std::vector<std::string> c_names;
    for (int g = 0; g < nW; ++g) {
        c_names.push_back("c" + std::to_string(g + 1));
    }
    
    size_t c_col = 0;
    for (const auto& name : c_names) c_col = std::max(c_col, name.size());
    
    for (int g = 0; g < nW; ++g) {
        std::vector<Rational> W_col;
        for (int mu = 0; mu < r; ++mu) {
            W_col.push_back(W[mu * nW + g]);
        }
        
        std::string rhs = format_linform(W_col, 'm', false);
        
        f << c_names[g];
        for (size_t j = c_names[g].size(); j < c_col; ++j) f << ' ';
        f << " = " << rhs << line_end;
    }
}