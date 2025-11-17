// select.cpp
// Pareto-optimal scheme selection with recursion analysis

#include "core/rational.h"
#include "core/select.h"
#include "core/print.h"
#include "core/codec.h"
#include "core/meta.h"
#include "CLI11.hpp"

#include <iostream>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <algorithm>

namespace fs = std::filesystem;

// Unpack rational scheme: [U_num | U_den | V_num | V_den | W_num | W_den]
inline void unpack_rational_scheme(const I64* row, int r, int nU, int nV, int nW,
                                   std::vector<Rational>& U,
                                   std::vector<Rational>& V,
                                   std::vector<Rational>& W) {
    int u_size = r * nU;
    int v_size = r * nV;
    int w_size = r * nW;

    U.clear();
    V.clear();
    W.clear();
    U.reserve(u_size);
    V.reserve(v_size);
    W.reserve(w_size);

    for (int i = 0; i < u_size; ++i) {
        U.emplace_back(row[2 * i], row[2 * i + 1]);
    }

    int v_offset = 2 * u_size;
    for (int i = 0; i < v_size; ++i) {
        V.emplace_back(row[v_offset + 2 * i], row[v_offset + 2 * i + 1]);
    }

    int w_offset = 2 * (u_size + v_size);
    for (int i = 0; i < w_size; ++i) {
        W.emplace_back(row[w_offset + 2 * i], row[w_offset + 2 * i + 1]);
    }
}

// Separate schemes into integer and rational indices
inline void separate_schemes(
    const std::vector<std::vector<Rational>>& all_U,
    const std::vector<std::vector<Rational>>& all_V,
    const std::vector<std::vector<Rational>>& all_W,
    std::vector<size_t>& integer_indices,
    std::vector<size_t>& rational_indices)
{
    integer_indices.clear();
    rational_indices.clear();

    for (size_t i = 0; i < all_U.size(); ++i) {
        if (is_integer_scheme(all_U[i], all_V[i], all_W[i])) {
            integer_indices.push_back(i);
        } else {
            rational_indices.push_back(i);
        }
    }
}

// Count occurrences of each triple in a set of schemes
inline std::map<TripleKey, size_t> count_triples(
    const std::vector<TripleKey>& triples)
{
    std::map<TripleKey, size_t> counts;
    for (const auto& t : triples) {
        ++counts[t];
    }
    return counts;
}

// Merge result structure
struct MergeResult {
    std::vector<size_t>     final_indices;
    std::vector<TripleKey>  final_triples;
    std::vector<size_t>     added_rational_indices;
    bool                    has_new_rational;
};

// Merge integer and rational Pareto fronts
inline MergeResult merge_pareto_fronts(
    const std::vector<size_t>& int_pareto_indices,
    const std::vector<TripleKey>& int_pareto_triples,
    const std::vector<size_t>& rat_pareto_indices,
    const std::vector<TripleKey>& rat_pareto_triples)
{
    MergeResult result;
    result.final_indices = int_pareto_indices;
    result.final_triples = int_pareto_triples;
    result.has_new_rational = false;

    for (size_t i = 0; i < rat_pareto_indices.size(); ++i) {
        const auto& rat_triple = rat_pareto_triples[i];
        size_t rat_idx = rat_pareto_indices[i];

        bool dominated_by_integer = false;

        for (size_t j = 0; j < int_pareto_triples.size(); ++j) {
            const auto& int_triple = int_pareto_triples[j];

            if (rat_triple == int_triple) {
                dominated_by_integer = true;
                break;
            }

            if (triple_dominates(int_triple, rat_triple)) {
                dominated_by_integer = true;
                break;
            }
        }

        if (!dominated_by_integer) {
            result.final_indices.push_back(rat_idx);
            result.final_triples.push_back(rat_triple);
            result.added_rational_indices.push_back(rat_idx);
            result.has_new_rational = true;
        }
    }

    return result;
}

int main(int argc, char** argv) {
    CLI::App app{"Pareto-optimal scheme selection with recursion analysis"};

    std::string name;               // tensor/scheme name (stem)
    int         rank;
    std::string path       = "";    // optional single rational .npy file
    std::string output_dir = "data/schemes_selected";
    bool        semicolon  = false;

    app.add_option("name", name,
                   "Tensor name (e.g., gt-333, ggc-333)")
        ->required();
    app.add_option("rank", rank, "Rank of schemes")->required();
    app.add_option("--path", path,
                   "Path to rational schemes .npy. "
                   "If omitted, all matching files are loaded from "
                   "data/schemes_rational/<name>/");
    app.add_option("--output-dir", output_dir,
                   "Output directory")->default_val("data/schemes_selected");
    app.add_flag("--semicolon", semicolon,
                 "Add semicolons in .txt output");

    CLI11_PARSE(app, argc, argv);

    try {
        // Load tensor metadata
        fs::path meta_path = fs::path(paths::TENSORS_DIR) / (name + ".meta.json");
        TensorMeta meta = load_tensor_meta(meta_path);

        std::string scheme_stem = meta.name.empty() ? name : meta.name;

        const int nU = meta.nU;
        const int nV = meta.nV;
        const int nW = meta.nW;

        if (nU <= 0 || nV <= 0 || nW <= 0) {
            throw std::runtime_error(
                "Invalid or missing nU/nV/nW in metadata for '" + scheme_stem + "'.");
        }

        const bool have_op       = meta.op.has_value();
        const bool have_sizes    = meta.n1.has_value() &&
                                   meta.n2.has_value() &&
                                   meta.n3.has_value();
        const bool have_rec_info = have_op && have_sizes;

        std::string op;
        int n1 = 0, n2 = 0, n3 = 0;
        if (have_op) {
            op = *meta.op;
        }
        if (have_sizes) {
            n1 = *meta.n1;
            n2 = *meta.n2;
            n3 = *meta.n3;
        }

        const bool is_t = have_op && is_t_op(op);

        // Determine where to read rational schemes from
        std::vector<fs::path> rational_files;

        if (!path.empty()) {
            // Use exactly the file specified by --path
            fs::path p(path);
            if (!fs::exists(p)) {
                throw std::runtime_error("Rational schemes file not found: " + p.string());
            }
            if (fs::is_directory(p)) {
                throw std::runtime_error("Expected file for --path, got directory: " + p.string());
            }
            rational_files.push_back(fs::canonical(p));
        } else {
            // Auto-discover all files for this tensor and rank:
            // data/schemes_rational/<scheme_stem>/rank<rank>-rational-*.npy
            fs::path rat_dir = fs::path("data/schemes_rational") / scheme_stem;
            if (!fs::exists(rat_dir) || !fs::is_directory(rat_dir)) {
                throw std::runtime_error(
                    "Rational schemes directory not found: " + rat_dir.string());
            }

            std::string prefix =
                "rank" + std::to_string(rank) + "-rational-";

            for (const auto& entry : fs::directory_iterator(rat_dir)) {
                if (!entry.is_regular_file()) continue;
                fs::path p = entry.path();
                if (p.extension() != ".npy") continue;

                std::string filename = p.filename().string();
                if (filename.rfind(prefix, 0) == 0) {
                    rational_files.push_back(p);
                }
            }

            if (rational_files.empty()) {
                std::ostringstream oss;
                oss << "No rational scheme files found in " << rat_dir.string()
                    << " matching prefix '" << prefix << "'.";
                throw std::runtime_error(oss.str());
            }

            std::sort(rational_files.begin(), rational_files.end());
        }

        // Header
        std::cout << "=== Recursion Analysis ===\n";
        std::cout << "Tensor name: " << scheme_stem << "\n";

        if (have_op) {
            std::cout << "Operation: " << op;
            if (is_symmetric_op(op)) {
                std::cout << " (symmetric)";
            }
            if (is_t) {
                std::cout << " (t-operation)";
            }
            std::cout << "\n";
        } else {
            std::cout << "Operation: (unknown, not in metadata)\n";
        }

        if (have_sizes) {
            std::cout << "Shape:     (" << n1 << ", " << n2 << ", " << n3 << ")\n";
        } else {
            std::cout << "Shape:     (unknown; n1/n2/n3 not in metadata)\n";
        }

        std::cout << "Rank:      " << rank << "\n";

        if (!path.empty()) {
            std::cout << "Input:     " << rational_files.front().string() << "\n";
        } else {
            fs::path rat_dir = rational_files.front().parent_path();
            std::cout << "Input dir: " << rat_dir.string() << "\n";
            std::cout << "Files:     " << rational_files.size() << " matching file(s)\n";
        }

        std::cout << "Params:    nU=" << nU << ", nV=" << nV << ", nW=" << nW << "\n";

        if (!have_rec_info) {
            std::cout << "\nNote: op and/or (n1,n2,n3) are missing in metadata.\n";
            std::cout << "      Recursion triples will be treated as (0,0,0) for all schemes,\n";
            std::cout << "      selection will be based only on denominators and NNZ.\n";
        }
        std::cout << "\n";

        // Load all rational schemes from all selected files and concatenate
        std::vector<I64> rat_data_vec;
        size_t num_schemes = 0;
        size_t row_len     = 0;

        for (const auto& p : rational_files) {
            cnpy::NpyArray rat_arr = cnpy::npy_load(p.string());
            if (rat_arr.word_size != sizeof(I64)) {
                throw std::runtime_error("Rational schemes must be int64: " + p.string());
            }
            if (rat_arr.shape.size() != 2) {
                throw std::runtime_error("Rational schemes array must be 2D: " + p.string());
            }

            size_t rows = rat_arr.shape[0];
            size_t cols = rat_arr.shape[1];

            if (row_len == 0) {
                row_len = cols;
            } else if (cols != row_len) {
                std::ostringstream oss;
                oss << "Inconsistent row length in file " << p.string()
                    << ": got " << cols << ", expected " << row_len;
                throw std::runtime_error(oss.str());
            }

            I64* data_ptr = rat_arr.data<I64>();
            rat_data_vec.insert(rat_data_vec.end(),
                                data_ptr, data_ptr + rows * cols);
            num_schemes += rows;
        }

        if (num_schemes == 0) {
            std::cout << "No schemes found in the provided file(s).\n";
            return 0;
        }

        int expected_len = 2 * rank * (nU + nV + nW);
        if ((int)row_len != expected_len) {
            throw std::runtime_error(
                "Unexpected row length: " + std::to_string(row_len) +
                ", expected " + std::to_string(expected_len));
        }

        I64* rat_data = rat_data_vec.data();

        std::cout << "Loaded " << num_schemes << " rational schemes from "
                  << rational_files.size() << " file(s)\n";

        // Unpack all schemes
        std::vector<std::vector<Rational>> all_U, all_V, all_W;
        all_U.reserve(num_schemes);
        all_V.reserve(num_schemes);
        all_W.reserve(num_schemes);

        for (size_t i = 0; i < num_schemes; ++i) {
            std::vector<Rational> U, V, W;
            unpack_rational_scheme(rat_data + i * row_len,
                                   rank, nU, nV, nW,
                                   U, V, W);
            all_U.push_back(std::move(U));
            all_V.push_back(std::move(V));
            all_W.push_back(std::move(W));
        }

        // Separate into integer and rational
        std::vector<size_t> integer_indices, rational_indices;
        separate_schemes(all_U, all_V, all_W, integer_indices, rational_indices);

        std::cout << "\nInteger schemes:  " << integer_indices.size() << "\n";
        std::cout << "Rational schemes: " << rational_indices.size() << "\n\n";

        if (integer_indices.empty() && rational_indices.empty()) {
            std::cout << "No schemes to analyze.\n";
            return 0;
        }

        // Compute triples for all schemes
        std::vector<TripleKey> all_triples;
        all_triples.reserve(num_schemes);

        if (have_rec_info) {
            if (is_t) {
                // t-operation: use scheme_triple_t
                StructChar s = parse_t(op);

                for (size_t i = 0; i < num_schemes; ++i) {
                    TripleKey triple = scheme_triple_t(
                        all_U[i], all_V[i], all_W[i],
                        rank, nU, nV, nW,
                        n1, n2, n3,
                        s
                    );
                    all_triples.push_back(triple);
                }
            } else {
                // ab-operation: use scheme_triple_ab
                auto [a, b] = parse_ab(op);
                bool symmetric = is_symmetric_op(op);

                for (size_t i = 0; i < num_schemes; ++i) {
                    TripleKey triple = scheme_triple_ab(
                        all_U[i], all_V[i],
                        rank, nU, nV,
                        n1, n2, n3,
                        a, b, symmetric
                    );
                    all_triples.push_back(triple);
                }
            }
        } else {
            // No recursion info: assign neutral triple to all schemes
            for (size_t i = 0; i < num_schemes; ++i) {
                all_triples.push_back(TripleKey{0, 0, 0});
            }
        }

        // Find best per triple for integer schemes
        std::map<TripleKey, size_t> int_best_by_triple;
        if (!integer_indices.empty()) {
            for (size_t idx : integer_indices) {
                TripleKey key = all_triples[idx];

                auto it = int_best_by_triple.find(key);
                if (it == int_best_by_triple.end()) {
                    int_best_by_triple[key] = idx;
                } else {
                    I64 cur_den = max_denominator(all_U[idx], all_V[idx], all_W[idx]);
                    size_t cur_nnz = count_nnz(all_U[idx]) +
                                     count_nnz(all_V[idx]) +
                                     count_nnz(all_W[idx]);

                    I64 best_den = max_denominator(all_U[it->second],
                                                   all_V[it->second],
                                                   all_W[it->second]);
                    size_t best_nnz = count_nnz(all_U[it->second]) +
                                      count_nnz(all_V[it->second]) +
                                      count_nnz(all_W[it->second]);

                    BestByTripleCmp cur{cur_den, cur_nnz};
                    BestByTripleCmp best{best_den, best_nnz};
                    if (cur < best) {
                        it->second = idx;
                    }
                }
            }
        }

        // Find best per triple for rational schemes
        std::map<TripleKey, size_t> rat_best_by_triple;
        if (!rational_indices.empty()) {
            for (size_t idx : rational_indices) {
                TripleKey key = all_triples[idx];

                auto it = rat_best_by_triple.find(key);
                if (it == rat_best_by_triple.end()) {
                    rat_best_by_triple[key] = idx;
                } else {
                    I64 cur_den = max_denominator(all_U[idx], all_V[idx], all_W[idx]);
                    size_t cur_nnz = count_nnz(all_U[idx]) +
                                     count_nnz(all_V[idx]) +
                                     count_nnz(all_W[idx]);

                    I64 best_den = max_denominator(all_U[it->second],
                                                   all_V[it->second],
                                                   all_W[it->second]);
                    size_t best_nnz = count_nnz(all_U[it->second]) +
                                      count_nnz(all_V[it->second]) +
                                      count_nnz(all_W[it->second]);

                    BestByTripleCmp cur{cur_den, cur_nnz};
                    BestByTripleCmp best{best_den, best_nnz};
                    if (cur < best) {
                        it->second = idx;
                    }
                }
            }
        }

        std::cout << "  Unique triples (integer):  " << int_best_by_triple.size() << "\n";
        std::cout << "  Unique triples (rational): " << rat_best_by_triple.size() << "\n\n";

        // Build integer Pareto front
        std::vector<size_t>    int_pareto_indices;
        std::vector<TripleKey> int_pareto_triples;

        if (!int_best_by_triple.empty()) {
            std::vector<size_t>    int_best_indices;
            std::vector<TripleKey> int_best_triples;

            for (const auto& [triple, idx] : int_best_by_triple) {
                int_best_indices.push_back(idx);
                int_best_triples.push_back(triple);
            }

            int_pareto_indices = build_pareto_front(
                int_best_triples, int_best_indices,
                all_U, all_V, all_W
            );

            for (size_t idx : int_pareto_indices) {
                int_pareto_triples.push_back(all_triples[idx]);
            }
        }

        // Build rational Pareto front
        std::vector<size_t>    rat_pareto_indices;
        std::vector<TripleKey> rat_pareto_triples;

        if (!rat_best_by_triple.empty()) {
            std::vector<size_t>    rat_best_indices;
            std::vector<TripleKey> rat_best_triples;

            for (const auto& [triple, idx] : rat_best_by_triple) {
                rat_best_indices.push_back(idx);
                rat_best_triples.push_back(triple);
            }

            rat_pareto_indices = build_pareto_front(
                rat_best_triples, rat_best_indices,
                all_U, all_V, all_W
            );

            for (size_t idx : rat_pareto_indices) {
                rat_pareto_triples.push_back(all_triples[idx]);
            }
        }

        std::cout << "  Pareto front (integer):  " << int_pareto_indices.size()
                  << " non-dominated triple(s)\n";
        std::cout << "  Pareto front (rational): " << rat_pareto_indices.size()
                  << " non-dominated triple(s)\n\n";

        // Merge fronts
        std::cout << "Merging fronts...\n";

        if (int_pareto_indices.empty() && !rat_pareto_indices.empty()) {
            std::cout << "  Integer front is empty, using rational front only\n\n";
        }

        MergeResult merge = merge_pareto_fronts(
            int_pareto_indices, int_pareto_triples,
            rat_pareto_indices, rat_pareto_triples
        );

        if (!int_pareto_indices.empty() && merge.has_new_rational) {
            std::cout << "  Extended front with "
                      << merge.added_rational_indices.size()
                      << " rational scheme(s)\n\n";
        } else if (!int_pareto_indices.empty() && !merge.has_new_rational) {
            std::cout << "  No rational schemes added "
                         "(all dominated by or equal to integer front)\n\n";
        }

        // Count occurrences for final schemes
        std::map<TripleKey, size_t> triple_counts = count_triples(all_triples);

        // Print final table
        size_t final_count = merge.final_indices.size();
        std::cout << "=== Final Pareto Front: " << final_count << " scheme"
                  << (final_count == 1 ? "" : "s") << " ===\n";
        std::cout << std::left
                  << std::setw(16) << "Triple" << " | "
                  << std::setw(7)  << "Count"  << " | "
                  << std::setw(7)  << "Field"  << " | "
                  << std::setw(6)  << "NNZ"    << " | "
                  << std::setw(10) << "MaxDenom" << "\n";
        std::cout << std::string(60, '-') << "\n";

        // Sort by triple for consistent output
        std::vector<std::pair<TripleKey, size_t>> sorted_final;
        sorted_final.reserve(merge.final_indices.size());
        for (size_t i = 0; i < merge.final_indices.size(); ++i) {
            sorted_final.push_back({merge.final_triples[i], merge.final_indices[i]});
        }
        std::sort(sorted_final.begin(), sorted_final.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        for (const auto& [triple, idx] : sorted_final) {
            bool   is_int  = is_integer_scheme(all_U[idx], all_V[idx], all_W[idx]);
            size_t nnz     = count_nnz(all_U[idx]) +
                             count_nnz(all_V[idx]) +
                             count_nnz(all_W[idx]);
            I64    max_den = max_denominator(all_U[idx], all_V[idx], all_W[idx]);
            size_t count   = triple_counts[triple];

            std::ostringstream triple_str;
            triple_str << "(" << std::setw(2) << triple.a << ","
                       << std::setw(2) << triple.b << ","
                       << std::setw(2) << triple.c << ")";

            std::cout << std::left
                      << std::setw(16) << triple_str.str() << " | "
                      << std::setw(7)  << count            << " | "
                      << std::setw(7)  << (is_int ? "Z" : "Q") << " | "
                      << std::setw(6)  << nnz              << " | "
                      << std::setw(10) << max_den          << "\n";
        }
        std::cout << "\n";

        // Save selected schemes
        fs::create_directories(output_dir);

        std::cout << "Saving selected schemes...\n";

        for (const auto& [triple, idx] : sorted_final) {
            const auto& U_best = all_U[idx];
            const auto& V_best = all_V[idx];
            const auto& W_best = all_W[idx];

            bool is_int = is_integer_scheme(U_best, V_best, W_best);
            std::string field = is_int ? "z" : "q";

            // Base filename
            std::ostringstream base_name;
            base_name << output_dir << "/" << scheme_stem
                      << "-rank" << rank
                      << "-rec-" << triple.a << "-" << triple.b << "-" << triple.c
                      << "-" << field;

            // Save .npy file
            std::string npy_path = base_name.str() + ".npy";
            std::vector<I64> selected_row;
            selected_row.reserve(2 * (U_best.size() + V_best.size() + W_best.size()));

            for (const auto& r : U_best) {
                selected_row.push_back(r.numerator());
                selected_row.push_back(r.denominator());
            }
            for (const auto& r : V_best) {
                selected_row.push_back(r.numerator());
                selected_row.push_back(r.denominator());
            }
            for (const auto& r : W_best) {
                selected_row.push_back(r.numerator());
                selected_row.push_back(r.denominator());
            }

            cnpy::npy_save(npy_path, selected_row.data(),
                           {1, selected_row.size()}, "w");

            // Save .txt file
            std::string txt_path = base_name.str() + ".txt";
            print_scheme(txt_path, U_best, V_best, W_best,
                         rank, nU, nV, nW, semicolon);
        }

        std::cout << "Saved " << merge.final_indices.size()
                  << " selected scheme"
                  << (merge.final_indices.size() == 1 ? "" : "s")
                  << " (.npy + .txt) to " << output_dir << "/\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
