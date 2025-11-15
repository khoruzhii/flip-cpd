// search.cpp

#include "core/field.h"
#include "core/codec.h"
#include "core/log.h"
#include "core/utils.h"
#include "core/scheme.h"
#include "core/meta.h"
#include "cnpy.h"
#include "CLI11.hpp"

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <chrono>
#include <condition_variable>
#include <algorithm>
#include <iomanip>
#include <optional>
#include <filesystem>
#include <cassert>
#include <sstream>

#if defined(MOD3)
    using FieldType = B3;
#else
    using FieldType = B2;
#endif

// -----------------------------------------------------------------------------
// SchemeData: scheme triple-list and its metadata
// -----------------------------------------------------------------------------
template<typename Field>
struct SchemeData {
    std::vector<Field> data; // flat: [u0,v0,w0, u1,v1,w1, ...]
    int rank;
    U32 seed;                // RNG seed that generated this scheme

    SchemeData() : rank(0), seed(0) {}
    SchemeData(const std::vector<Field>& d, int r, U32 s)
        : data(d), rank(r), seed(s) {}
};

// -----------------------------------------------------------------------------
// Save pool to disk (only verified schemes)
// -----------------------------------------------------------------------------
template<typename Field>
void save_pool(const std::vector<SchemeData<Field>>& pool,
               int rank,
               const std::string& tensor_name,
               const std::string& id) {
    if (pool.empty()) return;

    const std::string field_name = field_traits<Field>::is_mod2 ? "mod2" : "mod3";
    const std::filesystem::path dir =
        std::filesystem::path(paths::SCHEMES_DIR) / tensor_name;
    std::filesystem::create_directories(dir);
    const std::string filename =
        (dir / (field_name + "-rank" + std::to_string(rank) + id + ".npy")).string();

    const int lanes = field_traits<Field>::is_mod2 ? 1 : 2;
    const size_t elements_per_term = 3 * lanes;
    const size_t elements_per_scheme =
        static_cast<size_t>(rank) * elements_per_term;

    std::vector<U64> save_data(pool.size() * elements_per_scheme, 0ULL);

    for (size_t r = 0; r < pool.size(); ++r) {
        const auto& scheme = pool[r];
        size_t written = 0;
        size_t terms_written = 0;

        for (size_t i = 0; i + 2 < scheme.data.size(); i += 3) {
            if (scheme.data[i].is_zero()) continue;

            if constexpr (!field_traits<Field>::is_mod2) {
                const B3 u = static_cast<B3>(scheme.data[i + 0]);
                const B3 v = static_cast<B3>(scheme.data[i + 1]);
                const B3 w = static_cast<B3>(scheme.data[i + 2]);

                save_data[r * elements_per_scheme + written + 0] = u.lo;
                save_data[r * elements_per_scheme + written + 1] = u.hi;
                save_data[r * elements_per_scheme + written + 2] = v.lo;
                save_data[r * elements_per_scheme + written + 3] = v.hi;
                save_data[r * elements_per_scheme + written + 4] = w.lo;
                save_data[r * elements_per_scheme + written + 5] = w.hi;
                written += 6;
            } else {
                save_data[r * elements_per_scheme + written + 0] =
                    pack_field(scheme.data[i + 0]);
                save_data[r * elements_per_scheme + written + 1] =
                    pack_field(scheme.data[i + 1]);
                save_data[r * elements_per_scheme + written + 2] =
                    pack_field(scheme.data[i + 2]);
                written += 3;
            }

            ++terms_written;
            if (terms_written == static_cast<size_t>(rank)) break;
        }

        assert(terms_written == static_cast<size_t>(rank) &&
               "save_pool: fewer non-zero terms than rank");
        assert(written == elements_per_scheme &&
               "save_pool: row size mismatch");
    }

    const std::vector<size_t> shape = { pool.size(), elements_per_scheme };
    cnpy::npy_save(filename, save_data.data(), shape);
}

// -----------------------------------------------------------------------------
// Pool-based search
// -----------------------------------------------------------------------------
template<typename Field>
class PoolSearch {
private:
    // Problem parameters
    std::string name;         // tensor name (stem, e.g. "gt-333")
    std::string tensor_path;  // path to data/tensors/<name>.npy
    std::string op;           // for logging only (may be "-")
    int n1, n2, n3;           // for logging only (may be 0)
    std::string id;           // affects output scheme names only

    // Search parameters
    int path_limit;
    int pool_size;
    int target_rank;
    int plus_lim;
    int threads;
    int max_attempts;
    int stop_attempts;
    bool use_plus;
    bool save_pools;
    PoolRunLogger logger;

    // Tensor ground truth
    TensorCOO T;

    // Concurrency
    std::mutex pool_mutex;
    std::mutex result_mutex;
    std::condition_variable pool_cv;
    std::atomic<bool> stop_workers{false};
    std::atomic<int> active_workers{0};
    std::atomic<int> attempts_made{0};

    // Pools
    std::vector<SchemeData<Field>> current_pool;
    std::vector<SchemeData<Field>> next_pool;

    // RNG
    std::mt19937 seed_gen;

    std::string get_stem() const {
        // In the new setup, the stem is just the tensor name.
        return name;
    }

public:
    PoolSearch(std::string tensor_name,
               std::string tensor_path_,
               std::string operation_for_log,
               int dim1_for_log,
               int dim2_for_log,
               int dim3_for_log,
               std::string identifier,
               int path_lim,
               int pool_sz,
               int target,
               int plus_l,
               int thr,
               int max_att,
               int stop_att,
               bool use_p,
               bool save_p)
        : name(std::move(tensor_name)),
          tensor_path(std::move(tensor_path_)),
          op(std::move(operation_for_log)),
          n1(dim1_for_log),
          n2(dim2_for_log),
          n3(dim3_for_log),
          id(std::move(identifier)),
          path_limit(path_lim),
          pool_size(pool_sz),
          target_rank(target),
          plus_lim(plus_l),
          threads(thr),
          max_attempts(max_att),
          stop_attempts(stop_att),
          use_plus(use_p),
          save_pools(save_p),
          seed_gen(std::random_device{}()) {}

    // Try to find a scheme of strictly smaller rank starting from 'start'
    std::optional<SchemeData<Field>> search_from_scheme(
        const SchemeData<Field>& start) {
        const U32 search_seed = seed_gen();
        Scheme<Field> scheme(start.data, search_seed);

        int current_rank = start.rank;
        int best_rank = current_rank;
        int flips_since_improvement = 0;

        for (int step = 0; step < path_limit; ++step) {
            if (!scheme.flip()) {
                if (use_plus && !scheme.plus()) {
                    break;
                }
                flips_since_improvement = 0;
            }

            const int new_rank = get_rank(scheme.get_data_field());

            if (new_rank < current_rank) {
                return SchemeData<Field>(scheme.get_data_field(),
                                         new_rank,
                                         search_seed);
            }

            if (new_rank < best_rank) {
                best_rank = new_rank;
                flips_since_improvement = 0;
            } else {
                ++flips_since_improvement;
            }

            if (use_plus && flips_since_improvement >= plus_lim) {
                if (scheme.plus()) {
                    flips_since_improvement = 0;
                }
            }
        }

        return std::nullopt;
    }

    // Worker thread routine
    void worker() {
        while (!stop_workers) {
            SchemeData<Field>* picked = nullptr;

            {
                std::unique_lock<std::mutex> lock(pool_mutex);
                pool_cv.wait(lock, [this] {
                    return !current_pool.empty() || stop_workers;
                });

                if (stop_workers) break;

                if (!current_pool.empty()) {
                    std::uniform_int_distribution<size_t> dist(
                        0, current_pool.size() - 1);
                    size_t idx = dist(seed_gen);
                    picked = new SchemeData<Field>(current_pool[idx]);
                    active_workers++;
                    attempts_made++;
                }
            }

            if (picked) {
                auto res = search_from_scheme(*picked);
                if (res.has_value()) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    next_pool.push_back(*res);
                }
                delete picked;
                active_workers--;
            }
        }
    }

    // Main driver
    void run() {
        // Load tensor directly from tensor_path
        const int mod = field_traits<Field>::is_mod2 ? 2 : 3;
        T = load_coo_from_npy(tensor_path, mod);
        assert((T.mod == mod) &&
               "Loaded tensor modulus differs from Field modulus");

        auto initial_data = generate_trivial_decomposition<Field>(T);
        const int initial_rank = get_rank(initial_data);

        const std::string field_name =
            field_traits<Field>::is_mod2 ? "mod2" : "mod3";
        const std::string stem = get_stem();

        // Initialize JSON logger if saving
        if (save_pools) {
            logger.begin(
                tensor_path,
                stem,
                field_name,
                id,
                T.dims.du,
                T.dims.dv,
                T.dims.dw,
                T.nz.size(),
                path_limit,
                pool_size,
                target_rank,
                plus_lim,
                threads,
                max_attempts,
                use_plus,
                save_pools
            );
        }

        // Log header
        std::cout << "=== Pool-based Tensor Search ===\n";
        std::cout << "Tensor name: " << name << "\n";
        std::cout << "Operation: " << op << "\n";
        std::cout << "Dimensions: " << n1 << "x" << n2 << "x" << n3 << "\n";
        std::cout << "Output ID: "
                  << (id.empty() ? "(none)" : id) << "\n";
        std::cout << "Tensor: " << tensor_path << "\n";
        std::cout << "Field: " << field_name << "\n";
        std::cout << "Dims: U=" << T.dims.du
                  << ", V=" << T.dims.dv
                  << ", W=" << T.dims.dw
                  << "  (nnz=" << T.nz.size() << ")\n";
        std::cout << "Path limit: " << path_limit
                  << ", Pool size: " << pool_size
                  << ", Target rank: " << target_rank
                  << ", Stop attempts: " << stop_attempts
                  << ", Threads: " << threads;
        if (use_plus) {
            std::cout << ", Plus transitions: enabled (limit: "
                      << plus_lim << ")";
        } else {
            std::cout << ", Plus transitions: disabled";
        }
        std::cout << "\n\n";

        // Seed pool
        current_pool.clear();
        current_pool.push_back(
            SchemeData<Field>(initial_data, initial_rank, 0));
        std::cout << "Starting from rank " << initial_rank << "\n\n";

        // Rank descent loop
        for (int current_target = initial_rank - 1;
             current_target >= target_rank;
             --current_target) {

            std::cout << "=== Searching for rank "
                      << current_target << " ===\n";

            next_pool.clear();
            stop_workers = false;
            attempts_made = 0;

            auto start_time = std::chrono::steady_clock::now();

            // Start workers
            std::vector<std::thread> workers;
            workers.reserve(threads);
            for (int i = 0; i < threads; ++i) {
                workers.emplace_back(&PoolSearch::worker, this);
            }

            // Wait for enough candidates or attempts exhausted
            while (next_pool.size() <
                       static_cast<size_t>(pool_size) &&
                   attempts_made < max_attempts) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100));
                pool_cv.notify_all();

                if (attempts_made > 0 &&
                    attempts_made % 100 == 0) {
                    std::cout << "  Attempts: " << attempts_made
                              << ", Found: " << next_pool.size()
                              << "/" << pool_size << "\r"
                              << std::flush;
                }

                // Stop early if nothing found after stop_attempts
                if (next_pool.empty() &&
                    attempts_made >= stop_attempts) {
                    break;
                }
            }
            std::cout << "\n";

            // Stop workers
            stop_workers = true;
            pool_cv.notify_all();
            for (auto& w : workers) {
                w.join();
            }

            auto end_time = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(
                              end_time - start_time)
                              .count();

            std::cout << "Completed " << attempts_made
                      << " attempts in "
                      << std::fixed << std::setprecision(1)
                      << secs << "s\n";
            std::cout << "Found " << next_pool.size()
                      << " candidate schemes of rank "
                      << current_target << "\n";

            if (next_pool.empty()) {
                std::cout << "Failed to find schemes of rank "
                          << current_target << " after "
                          << attempts_made << " attempts\n";
                if (save_pools) {
                    logger.add_progress_entry(
                        current_target,
                        attempts_made.load(),
                        next_pool.size(),
                        0,
                        secs,
                        false,
                        std::string{}
                    );
                    logger.flush_partial();
                }
                break;
            }

            // Verify candidates
            std::vector<SchemeData<Field>> verified_pool;
            verified_pool.reserve(next_pool.size());
            for (const auto& scheme : next_pool) {
                if (verify_scheme<Field>(scheme.data, T)) {
                    verified_pool.push_back(scheme);
                }
            }
            std::cout << "Verified: " << verified_pool.size()
                      << "/" << next_pool.size() << "\n\n";

            // Save verified schemes
            bool saved_npy = false;
            std::string npy_path;

            if (save_pools && !verified_pool.empty()) {
                const std::filesystem::path dir =
                    std::filesystem::path(paths::SCHEMES_DIR) / stem;
                npy_path =
                    (dir / (field_name + "-rank" +
                            std::to_string(current_target) + id +
                            ".npy"))
                        .string();
                save_pool<Field>(
                    verified_pool, current_target, name, id);
                saved_npy = true;
            }

            // Log progress
            if (save_pools) {
                logger.add_progress_entry(
                    current_target,
                    attempts_made.load(),
                    next_pool.size(),
                    verified_pool.size(),
                    secs,
                    saved_npy,
                    npy_path
                );
                logger.flush_partial();
            }

            if (verified_pool.empty()) {
                std::cout << "No verified schemes at rank "
                          << current_target << ". Stopping.\n";
                break;
            }

            // Move to next level
            current_pool = std::move(verified_pool);

            if (current_pool.size() >
                static_cast<size_t>(pool_size)) {
                std::shuffle(current_pool.begin(),
                             current_pool.end(),
                             seed_gen);
                current_pool.erase(
                    current_pool.begin() + pool_size,
                    current_pool.end());
            }
        }

        // Final result
        if (!current_pool.empty()) {
            std::cout << "\n=== Final Results ===\n";
            std::cout << "Best rank achieved: "
                      << current_pool[0].rank << "\n";
            std::cout << "Pool size at best rank: "
                      << current_pool.size() << "\n";
        }

        // Write final JSON log
        if (save_pools) {
            const int best_rank = current_pool.empty()
                                      ? initial_rank
                                      : current_pool[0].rank;
            const std::size_t final_pool_size =
                current_pool.size();
            logger.finish_and_write(best_rank, final_pool_size);
        }
    }
};

// -----------------------------------------------------------------------------
// CLI entry
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    CLI::App app{"Pool-based Tensor Flip Graph Search"};

    // Positional arguments
    std::string tensor_name;

    app.add_option("name",
                   tensor_name,
                   "Tensor name (e.g., gt-333)")
        ->required();

    // Optional parameters
    std::string id = "";
    int path_limit = 1'000'000;
    int pool_size = 200;
    int target_rank = 0;
    int plus_lim = 50'000;
    int threads = 4;
    int max_attempts = 1000;
    int stop_attempts = 20000;
    bool use_plus = false;
    bool save_pools = false;

    app.add_option("--id",
                   id,
                   "Output identifier (affects scheme names only)")
        ->default_val("");
    app.add_option("-f,--path-limit",
                   path_limit,
                   "Path length limit")
        ->default_val(1'000'000);
    app.add_option("-s,--pool-size",
                   pool_size,
                   "Pool size limit")
        ->default_val(200);
    app.add_option("-r,--target-rank",
                   target_rank,
                   "Target rank")
        ->default_val(0);
    app.add_option("-p,--plus-lim",
                   plus_lim,
                   "Flips before plus transition")
        ->default_val(50'000);
    app.add_option("-t,--threads",
                   threads,
                   "Number of worker threads")
        ->default_val(4)->check(CLI::PositiveNumber);
    app.add_option("-m,--max-attempts",
                   max_attempts,
                   "Max attempts per rank level")
        ->default_val(1000)->check(CLI::PositiveNumber);
    app.add_option("--stop",
                   stop_attempts,
                   "Stop if nothing found after this many attempts")
        ->default_val(20000)->check(CLI::PositiveNumber);
    app.add_flag("--plus",
                 use_plus,
                 "Enable plus transitions");
    app.add_flag("--save",
                 save_pools,
                 "Save verified pools to files and JSON logs");

    CLI11_PARSE(app, argc, argv);

    // Load metadata from data/tensors/<name>.meta.json
    std::filesystem::path meta_path =
        std::filesystem::path(paths::TENSORS_DIR) /
        (tensor_name + ".meta.json");
    TensorMeta meta = load_tensor_meta(meta_path);

    // Placeholder values for logging if op, n1, n2, n3 are absent
    std::string op_log = meta.op.value_or("-");
    int n1_log = meta.n1.value_or(0);
    int n2_log = meta.n2.value_or(0);
    int n3_log = meta.n3.value_or(0);

    // Tensor path: data/tensors/<meta.name>.npy
    std::filesystem::path tensor_path =
        std::filesystem::path(paths::TENSORS_DIR) /
        (meta.name + ".npy");

    PoolSearch<FieldType> search(
        meta.name,
        tensor_path.string(),
        op_log,
        n1_log,
        n2_log,
        n3_log,
        id,
        path_limit,
        pool_size,
        target_rank,
        plus_lim,
        threads,
        max_attempts,
        stop_attempts,
        use_plus,
        save_pools
    );

    search.run();

    return 0;
}
