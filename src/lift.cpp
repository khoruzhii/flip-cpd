// lift.cpp
// Hensel lifting + rational reconstruction for tensor decomposition

#include "core/hensel.h"
#include "core/codec.h"
#include "core/rational.h"
#include "core/select.h"
#include "core/meta.h"
#include "CLI11.hpp"

#include <iostream>
#include <sstream>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

// Thread-safe work queue
class WorkQueue {
    std::queue<size_t> tasks;
    std::mutex mtx;

public:
    void push(size_t idx) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push(idx);
    }

    bool pop(size_t& idx) {
        std::lock_guard<std::mutex> lock(mtx);
        if (tasks.empty()) return false;
        idx = tasks.front();
        tasks.pop();
        return true;
    }
};

// Progress tracking
struct ProgressTracker {
    std::atomic<size_t> processed{0};
    std::atomic<size_t> successful{0};
    size_t total{};
    std::mutex print_mtx;

    inline void update(bool success) {
        size_t p = ++processed;
        if (success) ++successful;

        if (p % 50 == 0 || p == total) {
            std::lock_guard<std::mutex> lock(print_mtx);
            std::cout << "Progress: " << p << "/" << total
                      << " (" << successful.load() << " successful)\r"
                      << std::flush;
        }
    }
};

// Unpack lifted scheme: [U | V | W] flat layout
inline void unpack_lifted_scheme(const U64* row, int r, int nU, int nV, int nW,
                                 std::vector<U64>& U, std::vector<U64>& V, std::vector<U64>& W) {
    int u_size = r * nU;
    int v_size = r * nV;
    int w_size = r * nW;

    U.assign(row, row + u_size);
    V.assign(row + u_size, row + u_size + v_size);
    W.assign(row + u_size + v_size, row + u_size + v_size + w_size);
}

// Reconstruct array of U64 to Rational
inline bool reconstruct_array(const std::vector<U64>& arr, I64 M, I64 bound,
                              std::vector<Rational>& out) {
    out.clear();
    out.reserve(arr.size());

    for (U64 val : arr) {
        RatResult res = rational_reconstruct((I64)val, M, bound);
        if (!res.success) return false;
        out.emplace_back(res.p, res.q);
    }
    return true;
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

// Verify reconstructed scheme against tensor
inline bool verify_tensor(const std::vector<Rational>& U,
                          const std::vector<Rational>& V,
                          const std::vector<Rational>& W,
                          int r, int nU, int nV, int nW,
                          const std::vector<I64>& T0) {
    for (int i = 0; i < nU; ++i) {
        for (int j = 0; j < nV; ++j) {
            for (int k = 0; k < nW; ++k) {
                Rational sum(0, 1);
                for (int mu = 0; mu < r; ++mu) {
                    sum = sum + U[mu * nU + i] * V[mu * nV + j] * W[mu * nW + k];
                }

                I64 expected = T0[i * (nV * nW) + j * nW + k];
                if (sum != expected) return false;
            }
        }
    }
    return true;
}

// Canonicalize all rank-1 components
inline void canonicalize_factors(std::vector<Rational>& U,
                                 std::vector<Rational>& V,
                                 std::vector<Rational>& W,
                                 int r, int nU, int nV, int nW) {
    for (int q = 0; q < r; ++q) {
        std::vector<Rational> U_row(U.begin() + q * nU, U.begin() + (q + 1) * nU);
        std::vector<Rational> V_row(V.begin() + q * nV, V.begin() + (q + 1) * nV);
        std::vector<Rational> W_row(W.begin() + q * nW, W.begin() + (q + 1) * nW);

        canonicalize_component(U_row, V_row, W_row);

        std::copy(U_row.begin(), U_row.end(), U.begin() + q * nU);
        std::copy(V_row.begin(), V_row.end(), V.begin() + q * nV);
        std::copy(W_row.begin(), W_row.end(), W.begin() + q * nW);
    }
}

// Worker thread for Hensel lifting
void worker_thread(
    WorkQueue& queue,
    ProgressTracker& progress,
    const U64* schemes_data,
    const std::vector<I64>& T0,
    int row_len, int r,
    int nU, int nV, int nW, int k_steps,
    std::vector<std::vector<U64>>& local_results)
{
    std::vector<U64> U;
    std::vector<U64> V;
    std::vector<U64> W;
    U.reserve(static_cast<size_t>(r) * nU);
    V.reserve(static_cast<size_t>(r) * nV);
    W.reserve(static_cast<size_t>(r) * nW);

    size_t scheme_idx;
    while (queue.pop(scheme_idx)) {
        U.clear();
        V.clear();
        W.clear();

        const U64* row = schemes_data + scheme_idx * row_len;
#if defined(MOD3)
        unpack_scheme_f3(row, row_len, nU, nV, nW, U, V, W);
#else
        unpack_scheme_f2(row, row_len, nU, nV, nW, U, V, W);
#endif
        bool success = lift_tensor(T0, U, V, W, r, nU, nV, nW, k_steps);

        if (success) {
            std::vector<U64> lifted_row;
            pack_lifted_scheme(U, V, W, lifted_row);
            local_results.push_back(std::move(lifted_row));
        }

        progress.update(success);
    }
}

int main(int argc, char** argv) {
    CLI::App app{"Hensel lifting + rational reconstruction"};

    // New CLI: tensor name and rank
    std::string name;
    int rank;
    int k_steps = 10;
    std::string id;
    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    I64 bound = -1;
    bool verify = false;

    app.add_option("name", name, "Tensor name (e.g., gt-333)")
        ->required();
    app.add_option("rank", rank, "Rank of schemes")
        ->required();
    app.add_option("-k,--steps", k_steps, "Number of lifting steps")
        ->default_val(10);
    app.add_option("--id", id, "Output identifier (affects output paths only)")
        ->default_val("");
    app.add_option("-t,--threads", num_threads, "Number of threads");
    app.add_option("-b,--bound", bound, "Reconstruction bound (default: sqrt(M/2))");
    app.add_flag("-v,--verify", verify, "Verify reconstructed schemes against tensor");

    CLI11_PARSE(app, argc, argv);

    try {
        // Load tensor metadata
        fs::path meta_path = fs::path(paths::TENSORS_DIR) / (name + ".meta.json");
        TensorMeta meta = load_tensor_meta(meta_path);

        int nU = meta.nU;
        int nV = meta.nV;
        int nW = meta.nW;

        // Placeholders for logging if metadata does not contain op/n1/n2/n3
        std::string op_log = meta.op.value_or("-");
        int n1_log = meta.n1.value_or(0);
        int n2_log = meta.n2.value_or(0);
        int n3_log = meta.n3.value_or(0);

        // Base names from metadata
        std::string scheme_stem = meta.name;
        std::string tensor_base = meta.name;

        // Build paths
        std::ostringstream oss;

        oss << paths::SCHEMES_DIR << "/" << scheme_stem
            << "/mod" << MOD << "-rank" << rank << id << ".npy";
        std::string schemes_path = oss.str();

        oss.str("");
        oss.clear();
        oss << paths::TENSORS_DIR << "/" << tensor_base << ".npy";
        std::string tensor_path = oss.str();

        oss.str("");
        oss.clear();
        oss << "data/schemes_lifted/" << scheme_stem
            << "/rank" << rank << "-" << MOD << "pow" << (k_steps + 1) << id << ".npy";
        std::string lifted_path = oss.str();

        oss.str("");
        oss.clear();
        oss << "data/schemes_rational/" << scheme_stem
            << "/rank" << rank << "-rational-"
            << MOD << "pow" << (k_steps + 1) << id << ".npy";
        std::string rational_path = oss.str();

        if (!fs::exists(schemes_path)) {
            throw std::runtime_error("Schemes file not found: " + schemes_path);
        }
        if (!fs::exists(tensor_path)) {
            throw std::runtime_error("Tensor file not found: " + tensor_path);
        }

        // Compute modulus M = MOD^(k_steps+1)
        I64 M = 1;
        for (int i = 0; i <= k_steps; ++i) {
            M *= MOD;
        }

        if (bound <= 0) {
            bound = default_bound(M);
        }

        std::cout << "=== Hensel Lifting + Rational Reconstruction (mod " << MOD << ") ===\n";
        std::cout << "Tensor name: " << meta.name << "\n";
        std::cout << "Operation:   " << op_log << "\n";
        std::cout << "Shape:       (" << n1_log << ", " << n2_log << ", " << n3_log << ")\n";
        std::cout << "Tensor:      " << tensor_path << "\n";
        std::cout << "Output ID:   " << (id.empty() ? "(none)" : id) << "\n";
        std::cout << "Params:      nU=" << nU << ", nV=" << nV << ", nW=" << nW << "\n";
        std::cout << "Rank:        " << rank << "\n";
        std::cout << "Modulus:     " << M << " = " << MOD << "^" << (k_steps + 1) << "\n";
        std::cout << "Threads:     " << num_threads << "\n";
        std::cout << "Bound:       " << bound << "\n";
        std::cout << "Verify:      " << (verify ? "yes" : "no") << "\n\n";

        // Load tensor
        std::vector<I64> T0;
        load_sparse_tensor(tensor_path, nU, nV, nW, T0);

        // Load schemes
        cnpy::NpyArray schemes_arr = cnpy::npy_load(schemes_path);
        if (schemes_arr.word_size != sizeof(U64)) {
            throw std::runtime_error("Schemes must be uint64");
        }

        if (schemes_arr.shape.size() != 2) {
            throw std::runtime_error("Schemes array must be 2D");
        }

        size_t num_schemes = schemes_arr.shape[0];
        size_t row_len = schemes_arr.shape[1];

#if defined(MOD3)
        if (row_len % 6 != 0) {
            throw std::runtime_error(
                "Invalid row_len=" + std::to_string(row_len) +
                " for MOD3 (must be divisible by 6)"
            );
        }
        int r = static_cast<int>(row_len / 6);
#else
        if (row_len % 3 != 0) {
            throw std::runtime_error(
                "Invalid row_len=" + std::to_string(row_len) +
                " (must be divisible by 3)"
            );
        }
        int r = static_cast<int>(row_len / 3);
#endif

        U64* schemes_data = schemes_arr.data<U64>();
        std::cout << "Loaded " << num_schemes << " schemes (r=" << r << ")\n\n";

        // === PHASE 1: HENSEL LIFTING ===
        std::cout << "Phase 1: Hensel lifting...\n";

        WorkQueue queue;
        for (size_t i = 0; i < num_schemes; ++i) {
            queue.push(i);
        }

        ProgressTracker progress;
        progress.total = num_schemes;

        std::vector<std::vector<std::vector<U64>>> per_thread_results(num_threads);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        auto t_lift_start = std::chrono::high_resolution_clock::now();

        for (unsigned t = 0; t < num_threads; ++t) {
            threads.emplace_back(
                worker_thread,
                std::ref(queue),
                std::ref(progress),
                schemes_data,
                std::cref(T0),
                static_cast<int>(row_len),
                r,
                nU,
                nV,
                nW,
                k_steps,
                std::ref(per_thread_results[t])
            );
        }

        for (auto& th : threads) {
            th.join();
        }

        auto t_lift_end = std::chrono::high_resolution_clock::now();
        double lift_elapsed =
            std::chrono::duration<double>(t_lift_end - t_lift_start).count();

        // Merge lifting results
        std::vector<std::vector<U64>> lifted_schemes;
        for (const auto& thread_vec : per_thread_results) {
            lifted_schemes.insert(
                lifted_schemes.end(),
                thread_vec.begin(),
                thread_vec.end()
            );
        }

        std::cout << "\n\nLifting complete:\n";
        std::cout << "  Time:       " << lift_elapsed << " s\n";
        std::cout << "  Successful: " << lifted_schemes.size()
                  << "/" << num_schemes << "\n\n";

        if (lifted_schemes.empty()) {
            std::cout << "No schemes were successfully lifted.\n";
            return 0;
        }

        // Save lifted schemes
        fs::path lifted_out(lifted_path);
        fs::create_directories(lifted_out.parent_path());

        size_t lift_rows = lifted_schemes.size();
        size_t lift_cols = lifted_schemes[0].size();
        std::vector<U64> lift_data;
        lift_data.reserve(lift_rows * lift_cols);

        for (const auto& row : lifted_schemes) {
            lift_data.insert(lift_data.end(), row.begin(), row.end());
        }

        cnpy::npy_save(lifted_path, lift_data.data(), {lift_rows, lift_cols}, "w");

        std::cout << "Saved lifted schemes to: " << lifted_path << "\n";
        std::cout << "Format: (" << lift_rows << ", " << lift_cols << ") uint64\n\n";

        // === PHASE 2: RATIONAL RECONSTRUCTION ===
        std::cout << "Phase 2: Rational reconstruction...\n";

        std::vector<std::vector<I64>> rational_schemes;
        size_t successful = 0;
        size_t integer_schemes = 0;
        size_t min_nnz = std::numeric_limits<size_t>::max();
        size_t max_nnz = 0;
        size_t sum_nnz = 0;

        auto t_rat_start = std::chrono::high_resolution_clock::now();

        for (size_t idx = 0; idx < lift_rows; ++idx) {
            const U64* row = lift_data.data() + idx * lift_cols;

            std::vector<U64> U_lifted;
            std::vector<U64> V_lifted;
            std::vector<U64> W_lifted;
            unpack_lifted_scheme(row, r, nU, nV, nW, U_lifted, V_lifted, W_lifted);

            std::vector<Rational> U_rat;
            std::vector<Rational> V_rat;
            std::vector<Rational> W_rat;
            bool success =
                reconstruct_array(U_lifted, M, bound, U_rat) &&
                reconstruct_array(V_lifted, M, bound, V_rat) &&
                reconstruct_array(W_lifted, M, bound, W_rat);

            if (!success) {
                continue;
            }

            canonicalize_factors(U_rat, V_rat, W_rat, r, nU, nV, nW);

            if (verify) {
                if (!verify_tensor(U_rat, V_rat, W_rat,
                                   r, nU, nV, nW, T0)) {
                    continue;
                }
            }

            size_t nnz = count_nnz(U_rat) +
                         count_nnz(V_rat) +
                         count_nnz(W_rat);
            min_nnz = std::min(min_nnz, nnz);
            max_nnz = std::max(max_nnz, nnz);
            sum_nnz += nnz;

            if (is_integer_scheme(U_rat, V_rat, W_rat)) {
                ++integer_schemes;
            }

            std::vector<I64> rational_row;
            pack_rational_scheme(U_rat, V_rat, W_rat, rational_row);
            rational_schemes.push_back(std::move(rational_row));

            ++successful;

            if ((idx + 1) % 50 == 0 || (idx + 1) == lift_rows) {
                std::cout << "Progress: " << (idx + 1) << "/" << lift_rows
                          << " (" << successful << " successful)\r"
                          << std::flush;
            }
        }

        auto t_rat_end = std::chrono::high_resolution_clock::now();
        double rat_elapsed =
            std::chrono::duration<double>(t_rat_end - t_rat_start).count();

        std::cout << "\n\nReconstruction complete:\n";
        std::cout << "  Time:       " << rat_elapsed << " s\n";
        std::cout << "  Successful: " << successful << "/" << lift_rows << "\n";
        std::cout << "  Integer:    " << integer_schemes << "/" << successful << "\n";

        if (successful > 0) {
            size_t avg_nnz = sum_nnz / successful;
            std::cout << "  NNZ (min/avg/max): "
                      << min_nnz << " / " << avg_nnz << " / " << max_nnz << "\n";
        }
        std::cout << "\n";

        if (rational_schemes.empty()) {
            std::cout << "No schemes were successfully reconstructed.\n";
            return 0;
        }

        // Save rational schemes
        fs::path rat_out(rational_path);
        fs::create_directories(rat_out.parent_path());

        size_t rat_rows = rational_schemes.size();
        size_t rat_cols = rational_schemes[0].size();
        std::vector<I64> rat_data;
        rat_data.reserve(rat_rows * rat_cols);

        for (const auto& row : rational_schemes) {
            rat_data.insert(rat_data.end(), row.begin(), row.end());
        }

        cnpy::npy_save(rational_path, rat_data.data(), {rat_rows, rat_cols}, "w");

        std::cout << "Saved rational schemes to: " << rational_path << "\n";
        std::cout << "Format:   (" << rat_rows << ", " << rat_cols << ") int64\n";
        std::cout << "Layout:   [U_num | U_den | V_num | V_den | W_num | W_den]\n";
        std::cout << "Sizes:    r*nU=" << (r * nU)
                  << ", r*nV=" << (r * nV)
                  << ", r*nW=" << (r * nW) << "\n\n";

        std::cout << "Total time: " << (lift_elapsed + rat_elapsed) << " s\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
