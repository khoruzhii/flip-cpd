// select_modular.cpp
// Select modular schemes (mod 2 or mod 3) by minimal NNZ and export as rational .npy + .txt
// Uses data/schemes_modp/<tensor>/modX-rankR*.npy as input and writes to data/schemes_selected.

#include "core/codec.h"
#include "core/rational.h"
#include "core/print.h"
#include "core/meta.h"
#include "CLI11.hpp"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <algorithm>

namespace fs = std::filesystem;

// Convert expanded modular coefficients {0,1} or {0,1,2} to Rational array.
// For MOD=3, coefficient 2 is kept as +2 (not -1), so the .txt uses "2".
inline void modular_to_rational(const std::vector<U64>& src,
                                int mod,
                                std::vector<Rational>& dst) {
    dst.clear();
    dst.reserve(src.size());
    for (U64 v_raw : src) {
        I64 v = static_cast<I64>(v_raw % static_cast<U64>(mod));
        dst.emplace_back(v, 1);
    }
}

// Pack rational scheme: [U_num | U_den | V_num | V_den | W_num | W_den]
inline void pack_rational_scheme(const std::vector<Rational>& U,
                                 const std::vector<Rational>& V,
                                 const std::vector<Rational>& W,
                                 std::vector<I64>& row) {
    row.clear();
    row.reserve(2 * (U.size() + V.size() + W.size()));

    for (const auto& r : U) {
        row.push_back(r.numerator());
        row.push_back(r.denominator());
    }
    for (const auto& r : V) {
        row.push_back(r.numerator());
        row.push_back(r.denominator());
    }
    for (const auto& r : W) {
        row.push_back(r.numerator());
        row.push_back(r.denominator());
    }
}

int main(int argc, char** argv) {
    CLI::App app{"Modular scheme selection (F2/F3) by minimal NNZ"};

    std::string name;               // tensor name (for metadata lookup)
    int         rank;               // rank of schemes
    std::string output_dir = "data/schemes_selected";
    bool        semicolon  = false;

    app.add_option("name", name,
                   "Tensor name (e.g. gt-333, ggc-333)")
        ->required();
    app.add_option("rank", rank,
                   "Rank of schemes")
        ->required();
    app.add_option("--output-dir", output_dir,
                   "Output directory")
        ->default_val("data/schemes_selected");
    app.add_flag("--semicolon", semicolon,
                 "Add semicolons in .txt output");

    CLI11_PARSE(app, argc, argv);

    try {
        // Load tensor metadata
        fs::path meta_path = fs::path(paths::TENSORS_DIR) / (name + ".meta.json");
        TensorMeta meta = load_tensor_meta(meta_path);

        int nU = meta.nU;
        int nV = meta.nV;
        int nW = meta.nW;

        if (nU <= 0 || nV <= 0 || nW <= 0) {
            throw std::runtime_error(
                "Invalid or missing nU/nV/nW in metadata for '" + meta.name + "'.");
        }

        // Base stem used for schemes (same as search.cpp / lift.cpp)
        std::string scheme_stem = meta.name;

        // Logging
        std::cout << "=== Modular Scheme Selection (mod " << MOD << ") ===\n";
        std::cout << "Tensor (meta.name): " << scheme_stem << "\n";
        std::cout << "Metadata file:      " << meta_path.string() << "\n";
        std::cout << "Rank:               " << rank << "\n";
        std::cout << "nU=" << nU << ", nV=" << nV << ", nW=" << nW << "\n";

        // Locate modular schemes: data/schemes_modp/<scheme_stem>/mod<MOD>-rank<rank>*.npy
        fs::path scheme_dir = fs::path(paths::SCHEMES_DIR) / scheme_stem;
        if (!fs::exists(scheme_dir) || !fs::is_directory(scheme_dir)) {
            throw std::runtime_error(
                "Schemes directory not found: " + scheme_dir.string());
        }

        std::ostringstream prefix_ss;
        prefix_ss << "mod" << MOD << "-rank" << rank;
        std::string file_prefix = prefix_ss.str();

        std::vector<fs::path> scheme_files;
        for (const auto& entry : fs::directory_iterator(scheme_dir)) {
            if (!entry.is_regular_file()) continue;
            fs::path p = entry.path();
            if (p.extension() != ".npy") continue;

            std::string filename = p.filename().string();
            if (filename.rfind(file_prefix, 0) == 0) {
                scheme_files.push_back(p);
            }
        }

        if (scheme_files.empty()) {
            std::ostringstream oss;
            oss << "No modular schemes found in " << scheme_dir.string()
                << " matching prefix '" << file_prefix << "'.";
            throw std::runtime_error(oss.str());
        }

        std::sort(scheme_files.begin(), scheme_files.end());

        std::cout << "Schemes dir:        " << scheme_dir.string() << "\n";
        std::cout << "Matching files:     " << scheme_files.size() << "\n\n";

        // Scan all schemes and pick the one with minimal NNZ
        size_t best_nnz = std::numeric_limits<size_t>::max();
        bool   have_best = false;
        std::vector<U64> best_U, best_V, best_W;
        int              best_r = 0;
        fs::path         best_path;
        size_t           best_idx = 0;

        size_t total_schemes = 0;

        for (const auto& p : scheme_files) {
            cnpy::NpyArray arr = cnpy::npy_load(p.string());
            if (arr.word_size != sizeof(U64)) {
                throw std::runtime_error(
                    "Schemes must be uint64 in file: " + p.string());
            }
            if (arr.shape.size() != 2) {
                throw std::runtime_error(
                    "Schemes array must be 2D in file: " + p.string());
            }

            size_t num_schemes = arr.shape[0];
            size_t row_len     = arr.shape[1];

#if defined(MOD3)
            if (row_len % 6 != 0) {
                throw std::runtime_error(
                    "Invalid row_len=" + std::to_string(row_len) +
                    " for MOD3 (must be divisible by 6) in file: " + p.string());
            }
            int r_in_file = static_cast<int>(row_len / 6);
#else
            if (row_len % 3 != 0) {
                throw std::runtime_error(
                    "Invalid row_len=" + std::to_string(row_len) +
                    " (must be divisible by 3) in file: " + p.string());
            }
            int r_in_file = static_cast<int>(row_len / 3);
#endif

            if (r_in_file != rank) {
                std::ostringstream oss;
                oss << "Warning: file '" << p.string()
                    << "' has r=" << r_in_file
                    << " but CLI rank=" << rank << ". Skipping this file.";
                std::cout << oss.str() << "\n";
                continue;
            }

            U64* data = arr.data<U64>();
            std::cout << "Scanning file: " << p.filename().string()
                      << " (" << num_schemes << " schemes, r=" << r_in_file << ")\n";

            std::vector<U64> U;
            std::vector<U64> V;
            std::vector<U64> W;
            U.reserve(static_cast<size_t>(r_in_file) * nU);
            V.reserve(static_cast<size_t>(r_in_file) * nV);
            W.reserve(static_cast<size_t>(r_in_file) * nW);

            for (size_t s = 0; s < num_schemes; ++s) {
                const U64* row = data + s * row_len;

#if defined(MOD3)
                unpack_scheme_f3(row, static_cast<int>(row_len), nU, nV, nW, U, V, W);
#else
                unpack_scheme_f2(row, static_cast<int>(row_len), nU, nV, nW, U, V, W);
#endif
                size_t nnz = 0;
                for (U64 v : U) if (v != 0) ++nnz;
                for (U64 v : V) if (v != 0) ++nnz;
                for (U64 v : W) if (v != 0) ++nnz;

                ++total_schemes;

                if (!have_best || nnz < best_nnz) {
                    have_best = true;
                    best_nnz = nnz;
                    best_U = U;
                    best_V = V;
                    best_W = W;
                    best_r = r_in_file;
                    best_path = p;
                    best_idx = s;
                }
            }
        }

        if (!have_best) {
            std::cout << "\nNo schemes with matching rank were found.\n";
            return 0;
        }

        std::cout << "\nTotal schemes scanned: " << total_schemes << "\n";
        std::cout << "Best scheme: " << best_path.filename().string()
                  << " (row index " << best_idx << "), NNZ = " << best_nnz << "\n";

        // Convert best modular scheme to Rational with coefficients in {0,1} or {0,1,2}
        std::vector<Rational> U_rat, V_rat, W_rat;
        modular_to_rational(best_U, MOD, U_rat);
        modular_to_rational(best_V, MOD, V_rat);
        modular_to_rational(best_W, MOD, W_rat);

        // Prepare output
        fs::create_directories(output_dir);

        std::string field_tag = (MOD == 2) ? "f2" : "f3";

        std::ostringstream base_ss;
        base_ss << output_dir << "/" << scheme_stem
                << "-rank" << rank
                << "-" << field_tag;
        std::string base = base_ss.str();

        // Save .npy in rational format (same layout as lift/select for Rational schemes)
        std::vector<I64> row_out;
        pack_rational_scheme(U_rat, V_rat, W_rat, row_out);
        cnpy::npy_save(base + ".npy",
                       row_out.data(),
                       std::vector<size_t>{1, row_out.size()},
                       "w");

        // Save .txt using existing pretty-printer (will show coefficient 2 as "2")
        print_scheme(base + ".txt",
                     U_rat, V_rat, W_rat,
                     best_r, nU, nV, nW,
                     semicolon);

        std::cout << "Saved selected scheme to:\n";
        std::cout << "  " << base << ".npy\n";
        std::cout << "  " << base << ".txt\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
