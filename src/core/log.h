// log.h
#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <system_error>

#include "picojson.h"

struct PoolRunLogger {
    // Persistent run metadata
    std::string field_name;
    std::string stem;
    std::string id;
    std::string started_at_iso;

    // Output locations
    std::filesystem::path out_dir;
    std::filesystem::path out_path;

    // JSON document under construction
    picojson::object root;
    picojson::array progress;

    // Returns current UTC time in ISO-8601 Zulu format, e.g. "2025-10-12T09:15:22Z".
    static std::string iso8601_utc_now() {
        using clock = std::chrono::system_clock;
        const auto now = clock::now();
        const std::time_t t = clock::to_time_t(now);
        std::tm tm{};
        #if defined(_WIN32)
            gmtime_s(&tm, &t);
        #else
            tm = *std::gmtime(&t);
        #endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
        return oss.str();
    }

    static std::filesystem::path log_dir_for(const std::string& stem) {
        return std::filesystem::path("data/logs") / stem;
    }

    // File name: <field><id>.json (e.g., "mod2.json" or "mod2-v2.json")
    static std::filesystem::path log_path_for(const std::string& stem,
                                              const std::string& field,
                                              const std::string& id) {
        return log_dir_for(stem) / (field + id + ".json");
    }

    // Initialize the run-level JSON log and write an initial shell to disk.
    void begin(const std::string& tensor_path_with_ext,
               const std::string& stem_in,
               const std::string& field_in,
               const std::string& id_in,
               int du, int dv, int dw, std::size_t nnz,
               int path_limit, int pool_size, int target_rank,
               int plus_lim, int threads, int max_attempts,
               bool plus_enabled, bool save_flag) {
        field_name = field_in;
        stem = stem_in;
        id = id_in;
        started_at_iso = iso8601_utc_now();

        out_dir  = log_dir_for(stem);
        out_path = log_path_for(stem, field_name, id);
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);

        picojson::object dims;
        dims["U"]   = picojson::value(static_cast<double>(du));
        dims["V"]   = picojson::value(static_cast<double>(dv));
        dims["W"]   = picojson::value(static_cast<double>(dw));
        dims["nnz"] = picojson::value(static_cast<double>(nnz));

        picojson::object cfg;
        cfg["path_limit"]   = picojson::value(static_cast<double>(path_limit));
        cfg["pool_size"]    = picojson::value(static_cast<double>(pool_size));
        cfg["target_rank"]  = picojson::value(static_cast<double>(target_rank));
        cfg["threads"]      = picojson::value(static_cast<double>(threads));
        cfg["max_attempts"] = picojson::value(static_cast<double>(max_attempts));
        cfg["plus_enabled"] = picojson::value(plus_enabled);
        cfg["plus_lim"]     = picojson::value(static_cast<double>(plus_lim));
        cfg["save"]         = picojson::value(save_flag);

        picojson::object run;
        run["started_at"]  = picojson::value(started_at_iso);
        run["status"]      = picojson::value("running");
        run["last_update"] = picojson::value(started_at_iso);

        root["tensor"] = picojson::value(tensor_path_with_ext);
        root["stem"]   = picojson::value(stem);
        root["field"]  = picojson::value(field_name);
        root["dims"]   = picojson::value(dims);
        root["config"] = picojson::value(cfg);
        root["run"]    = picojson::value(run);

        // Create initial shell on disk.
        write_inplace_(/*attach_progress=*/false);
    }

    // Add one progress record for a completed pool (rank level).
    void add_progress_entry(int rank, int attempts, std::size_t found, std::size_t verified,
                            double time_sec, bool saved_npy, const std::string& npy_path) {
        picojson::object e;
        e["rank"]      = picojson::value(static_cast<double>(rank));
        e["attempts"]  = picojson::value(static_cast<double>(attempts));
        e["found"]     = picojson::value(static_cast<double>(found));
        e["verified"]  = picojson::value(static_cast<double>(verified));
        e["time_sec"]  = picojson::value(time_sec);
        e["saved_npy"] = picojson::value(saved_npy);
        if (saved_npy && !npy_path.empty()) {
            e["npy_path"] = picojson::value(npy_path);
        }
        progress.emplace_back(e);
    }

    // Rewrite the JSON file after a pool finishes.
    bool flush_partial() {
        picojson::object run = root["run"].get<picojson::object>();
        run["last_update"] = picojson::value(iso8601_utc_now());
        root["run"] = picojson::value(run);
        return write_inplace_(/*attach_progress=*/true);
    }

    // Finalize the JSON and rewrite the file once more.
    bool finish_and_write(int best_rank, std::size_t final_pool_size) {
        picojson::object run = root["run"].get<picojson::object>();
        const std::string finished = iso8601_utc_now();
        run["finished_at"] = picojson::value(finished);
        run["status"]      = picojson::value("finished");
        run["last_update"] = picojson::value(finished);
        root["run"] = picojson::value(run);

        picojson::object result;
        result["best_rank_achieved"] = picojson::value(static_cast<double>(best_rank));
        result["final_pool_size"]    = picojson::value(static_cast<double>(final_pool_size));
        root["result"] = picojson::value(result);

        return write_inplace_(/*attach_progress=*/true);
    }

private:
    // In-place rewrite using ofstream + truncation.
    bool write_inplace_(bool attach_progress) {
        if (attach_progress) {
            root["progress"] = picojson::value(progress);
        }
        std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
        if (!ofs) return false;

        const std::string json = picojson::value(root).serialize();
        ofs.write(json.data(), static_cast<std::streamsize>(json.size()));
        ofs.flush();
        return static_cast<bool>(ofs);
    }
};